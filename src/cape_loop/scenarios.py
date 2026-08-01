"""Versioned scenario catalogs, validation, and deterministic materialization."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import read_control_bytes
from .domains import DATA_SPLITS, get_domain
from .rng import semantic_seed
from .schemas import InteractionContext, Option

CATALOG_SCHEMA_VERSION = 1
SELECTION_POLICY = "deterministic-stratified-v1"
CATALOG_STATUSES = frozenset({"frozen-development", "frozen-paper"})
CATALOG_ELIGIBILITIES = frozenset(
    {"simulation-and-pilot-only", "paper-eligible"}
)
SUPPORTED_MECHANISMS = frozenset(
    {
        "balanced",
        "restricted",
        "default",
        "suggested",
        "ranking",
        "suggestion",
    }
)
_CATALOG_KEYS = {
    "schema_version",
    "catalog_id",
    "catalog_version",
    "catalog_status",
    "eligibility",
    "language",
    "locale",
    "source",
    "license",
    "created_on",
    "frozen_on",
    "split_policy",
    "selection_policy",
    "attribute_order",
    "authoring_provenance",
    "scenarios",
}
_SCENARIO_KEYS = {
    "scenario_id",
    "family_id",
    "revision",
    "status",
    "split",
    "domain",
    "task_family",
    "target_attribute",
    "target_key",
    "target_half_span",
    "nuisance_attribute",
    "nuisance_key",
    "nuisance_direction",
    "prompt",
    "wording_template_id",
    "negative_option",
    "positive_option",
    "negative_same_direction_option",
    "positive_same_direction_option",
    "supported_mechanisms",
    "quality_assertions",
    "review",
}
_OPTION_KEYS = {"option_id", "label", "features"}
_QUALITY_KEYS = {
    "neutral_wording",
    "symmetric_surface",
    "no_treatment_cues",
    "no_split_cues",
    "no_real_entities",
    "no_time_sensitive_facts",
    "no_objective_dominance",
    "all_surface_facts_modeled_or_matched",
    "feature_role_contract",
}
_REVIEW_KEYS = {
    "automated_validation",
    "surface_human_review",
    "scientific_human_review",
    "paper_eligible",
    "note",
}
_SAFE_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._:-]*$")


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        unknown = sorted(observed - expected)
        details = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if unknown:
            details.append("unknown=" + ",".join(unknown))
        raise ValueError(f"{label} fields must be exact ({'; '.join(details)})")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return value


def _text(value: Any, label: str, *, maximum: int = 500) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    if len(value) > maximum:
        raise ValueError(f"{label} must contain at most {maximum} characters")
    if any(character in value for character in ("\x00", "\r", "\n")):
        raise ValueError(f"{label} must be one line without control characters")
    return value


def _identifier(value: Any, label: str) -> str:
    result = _text(value, label, maximum=160)
    if _SAFE_IDENTIFIER.fullmatch(result) is None:
        raise ValueError(f"{label} must be a lowercase stable identifier")
    return result


def _strict_json(raw: bytes, *, label: str) -> Mapping[str, Any]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} must be UTF-8: {exc}") from exc

    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"{label} contains non-finite number {value}")

    try:
        decoded = json.loads(
            text,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {exc}") from exc
    return _mapping(decoded, label)


def _normal_surface(value: str) -> str:
    return " ".join(value.casefold().split())


@dataclass(frozen=True, slots=True)
class ScenarioOption:
    option_id: str
    label: str
    features: tuple[float, float, float]

    @classmethod
    def parse(
        cls,
        raw: Any,
        *,
        label: str,
        domain_id: str,
    ) -> "ScenarioOption":
        payload = _mapping(raw, label)
        _exact_keys(payload, _OPTION_KEYS, label)
        option_id = _identifier(payload["option_id"], f"{label}.option_id")
        surface = _text(payload["label"], f"{label}.label", maximum=300)
        features = payload["features"]
        if (
            not isinstance(features, list)
            or len(features) != 3
            or any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                for value in features
            )
        ):
            raise ValueError(f"{label}.features must contain exactly three numbers")
        parsed = tuple(float(value) for value in features)
        option = Option(
            option_id=option_id,
            label=surface,
            features=parsed,  # type: ignore[arg-type]
            domain=domain_id,
        )
        return cls(option.option_id, option.label, option.features)

    def materialize(self, domain_id: str) -> Option:
        return Option(
            option_id=self.option_id,
            label=self.label,
            features=self.features,
            domain=domain_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "option_id": self.option_id,
            "label": self.label,
            "features": list(self.features),
        }


@dataclass(frozen=True, slots=True)
class ScenarioSpec:
    scenario_id: str
    family_id: str
    revision: int
    status: str
    split: str
    domain: str
    task_family: str
    target_attribute: int
    target_key: str
    target_half_span: float
    nuisance_attribute: int
    nuisance_key: str
    nuisance_direction: int
    prompt: str
    wording_template_id: str
    negative_option: ScenarioOption
    positive_option: ScenarioOption
    negative_same_direction_option: ScenarioOption
    positive_same_direction_option: ScenarioOption
    supported_mechanisms: tuple[str, ...]
    quality_assertions: Mapping[str, bool]
    review: Mapping[str, Any]

    @classmethod
    def parse(cls, raw: Any, *, index: int) -> "ScenarioSpec":
        label = f"scenarios[{index}]"
        payload = _mapping(raw, label)
        _exact_keys(payload, _SCENARIO_KEYS, label)
        scenario_id = _identifier(payload["scenario_id"], f"{label}.scenario_id")
        family_id = _identifier(payload["family_id"], f"{label}.family_id")
        revision = payload["revision"]
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
            raise ValueError(f"{label}.revision must be a positive integer")
        status = payload["status"]
        if status not in {"provisional", "approved"}:
            raise ValueError(f"{label}.status must be provisional or approved")
        split = payload["split"]
        if split not in DATA_SPLITS:
            raise ValueError(f"{label}.split must be one of {DATA_SPLITS}")
        domain = payload["domain"]
        if domain not in {"travel", "writing"}:
            raise ValueError(f"{label}.domain must be travel or writing")
        task_family = _identifier(
            payload["task_family"],
            f"{label}.task_family",
        )
        target = payload["target_attribute"]
        if (
            isinstance(target, bool)
            or not isinstance(target, int)
            or not 0 <= target < 3
        ):
            raise ValueError(f"{label}.target_attribute must be 0, 1, or 2")
        target_key = _identifier(payload["target_key"], f"{label}.target_key")
        expected_key = get_domain(domain).attributes[target].key
        if target_key != expected_key:
            raise ValueError(
                f"{label}.target_key is {target_key!r}; expected {expected_key!r}"
            )
        raw_target_half_span = payload["target_half_span"]
        if (
            isinstance(raw_target_half_span, bool)
            or not isinstance(raw_target_half_span, (int, float))
            or not math.isfinite(float(raw_target_half_span))
            or not 0.0 < float(raw_target_half_span) <= 0.56
        ):
            raise ValueError(
                f"{label}.target_half_span must be finite and lie in (0, 0.56]"
            )
        target_half_span = float(raw_target_half_span)
        nuisance = payload["nuisance_attribute"]
        if (
            isinstance(nuisance, bool)
            or not isinstance(nuisance, int)
            or not 0 <= nuisance < 3
            or nuisance == target
        ):
            raise ValueError(
                f"{label}.nuisance_attribute must be a non-target attribute"
            )
        nuisance_key = _identifier(
            payload["nuisance_key"],
            f"{label}.nuisance_key",
        )
        expected_nuisance_key = get_domain(domain).attributes[nuisance].key
        if nuisance_key != expected_nuisance_key:
            raise ValueError(
                f"{label}.nuisance_key is {nuisance_key!r}; "
                f"expected {expected_nuisance_key!r}"
            )
        nuisance_direction = payload["nuisance_direction"]
        if (
            isinstance(nuisance_direction, bool)
            or not isinstance(nuisance_direction, int)
            or nuisance_direction not in {-1, 1}
        ):
            raise ValueError(f"{label}.nuisance_direction must be -1 or +1")
        prompt = _text(payload["prompt"], f"{label}.prompt", maximum=500)
        wording_template_id = _identifier(
            payload["wording_template_id"],
            f"{label}.wording_template_id",
        )
        options = {
            name: ScenarioOption.parse(
                payload[name],
                label=f"{label}.{name}",
                domain_id=domain,
            )
            for name in (
                "negative_option",
                "positive_option",
                "negative_same_direction_option",
                "positive_same_direction_option",
            )
        }
        option_ids = {option.option_id for option in options.values()}
        if len(option_ids) != len(options):
            raise ValueError(f"{label} option IDs must be unique")
        option_labels = {_normal_surface(option.label) for option in options.values()}
        if len(option_labels) != len(options):
            raise ValueError(f"{label} option labels must be distinct")
        negative = [0.0, 0.0, 0.0]
        negative[target] = -target_half_span
        positive = [0.0, 0.0, 0.0]
        positive[target] = target_half_span
        negative_peer = list(negative)
        negative_peer[nuisance] = 0.25 * nuisance_direction
        positive_peer = list(positive)
        positive_peer[nuisance] = 0.25 * nuisance_direction
        expected_features = {
            "negative_option": tuple(negative),
            "positive_option": tuple(positive),
            "negative_same_direction_option": tuple(negative_peer),
            "positive_same_direction_option": tuple(positive_peer),
        }
        for name, expected in expected_features.items():
            if options[name].features != expected:
                raise ValueError(f"{label}.{name}.features must equal {list(expected)}")
        mechanisms = payload["supported_mechanisms"]
        if (
            not isinstance(mechanisms, list)
            or not mechanisms
            or not all(isinstance(item, str) for item in mechanisms)
            or len(mechanisms) != len(set(mechanisms))
            or set(mechanisms) != SUPPORTED_MECHANISMS
        ):
            raise ValueError(
                f"{label}.supported_mechanisms must contain the complete "
                "declared mechanism set"
            )
        quality = _mapping(payload["quality_assertions"], f"{label}.quality_assertions")
        _exact_keys(quality, _QUALITY_KEYS, f"{label}.quality_assertions")
        if any(value is not True for value in quality.values()):
            raise ValueError(f"{label}.quality_assertions must all be true")
        review = _mapping(payload["review"], f"{label}.review")
        _exact_keys(review, _REVIEW_KEYS, f"{label}.review")
        if review["automated_validation"] not in {"pending", "passed"}:
            raise ValueError(
                f"{label}.review.automated_validation must be pending or passed"
            )
        for key in ("surface_human_review", "scientific_human_review"):
            if review[key] not in {"not_completed", "passed"}:
                raise ValueError(
                    f"{label}.review.{key} must be not_completed or passed"
                )
        if not isinstance(review["paper_eligible"], bool):
            raise ValueError(f"{label}.review.paper_eligible must be Boolean")
        reviews_complete = (
            review["automated_validation"] == "passed"
            and review["surface_human_review"] == "passed"
            and review["scientific_human_review"] == "passed"
        )
        if status == "approved" and not reviews_complete:
            raise ValueError(f"{label} approved scenarios require every review to pass")
        if review["paper_eligible"] and (status != "approved" or not reviews_complete):
            raise ValueError(
                f"{label} paper eligibility requires approved status and "
                "completed automated, surface, and scientific reviews"
            )
        _text(review["note"], f"{label}.review.note", maximum=500)
        return cls(
            scenario_id=scenario_id,
            family_id=family_id,
            revision=revision,
            status=status,
            split=split,
            domain=domain,
            task_family=task_family,
            target_attribute=target,
            target_key=target_key,
            target_half_span=target_half_span,
            nuisance_attribute=nuisance,
            nuisance_key=nuisance_key,
            nuisance_direction=nuisance_direction,
            prompt=prompt,
            wording_template_id=wording_template_id,
            negative_option=options["negative_option"],
            positive_option=options["positive_option"],
            negative_same_direction_option=options["negative_same_direction_option"],
            positive_same_direction_option=options["positive_same_direction_option"],
            supported_mechanisms=tuple(mechanisms),
            quality_assertions=dict(quality),
            review=dict(review),
        )

    @property
    def options(self) -> tuple[ScenarioOption, ...]:
        return (
            self.negative_option,
            self.positive_option,
            self.negative_same_direction_option,
            self.positive_same_direction_option,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "family_id": self.family_id,
            "revision": self.revision,
            "status": self.status,
            "split": self.split,
            "domain": self.domain,
            "task_family": self.task_family,
            "target_attribute": self.target_attribute,
            "target_key": self.target_key,
            "target_half_span": self.target_half_span,
            "nuisance_attribute": self.nuisance_attribute,
            "nuisance_key": self.nuisance_key,
            "nuisance_direction": self.nuisance_direction,
            "prompt": self.prompt,
            "wording_template_id": self.wording_template_id,
            "negative_option": self.negative_option.to_dict(),
            "positive_option": self.positive_option.to_dict(),
            "negative_same_direction_option": (
                self.negative_same_direction_option.to_dict()
            ),
            "positive_same_direction_option": (
                self.positive_same_direction_option.to_dict()
            ),
            "supported_mechanisms": list(self.supported_mechanisms),
            "quality_assertions": dict(self.quality_assertions),
            "review": dict(self.review),
        }

    def option_for_features(self, features: tuple[float, float, float]) -> Option:
        # Generic policy contexts use a canonical positive nuisance peer.
        # Catalog scenarios may use another target half-span and may reverse
        # that peer contrast to counterbalance nuisance direction. Map the
        # generic semantic role by target sign and peer presence instead of
        # requiring either numeric magnitude to match.
        target_value = features[self.target_attribute]
        non_target = tuple(
            value
            for index, value in enumerate(features)
            if index != self.target_attribute
        )
        if (
            target_value != 0.0
            and sum(value != 0.0 for value in non_target) <= 1
        ):
            positive = target_value > 0.0
            peer = any(value != 0.0 for value in non_target)
            if positive and peer:
                return self.positive_same_direction_option.materialize(self.domain)
            if positive:
                return self.positive_option.materialize(self.domain)
            if peer:
                return self.negative_same_direction_option.materialize(self.domain)
            return self.negative_option.materialize(self.domain)
        raise ValueError(
            f"scenario {self.scenario_id} has no option for features {features}"
        )


@dataclass(frozen=True, slots=True)
class ScenarioCatalog:
    catalog_id: str
    catalog_version: str
    catalog_status: str
    eligibility: str
    language: str
    locale: str
    source: str
    license: str
    created_on: str
    frozen_on: str
    split_policy: str
    selection_policy: str
    attribute_order: Mapping[str, tuple[str, str, str]]
    authoring_provenance: Mapping[str, Any]
    scenarios: tuple[ScenarioSpec, ...]
    _by_id: Mapping[str, ScenarioSpec] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        by_id = {scenario.scenario_id: scenario for scenario in self.scenarios}
        if len(by_id) != len(self.scenarios):
            raise ValueError("scenario IDs must be globally unique")
        family_splits: dict[str, set[str]] = {}
        wording_splits: dict[str, set[str]] = {}
        option_ids: set[str] = set()
        for scenario in self.scenarios:
            family_splits.setdefault(scenario.family_id, set()).add(scenario.split)
            wording_splits.setdefault(
                scenario.wording_template_id,
                set(),
            ).add(scenario.split)
            for option in scenario.options:
                if option.option_id in option_ids:
                    raise ValueError(
                        f"catalog option ID is duplicated: {option.option_id}"
                    )
                option_ids.add(option.option_id)
        leaking = sorted(
            family for family, splits in family_splits.items() if len(splits) != 1
        )
        if leaking:
            raise ValueError(
                "scenario families cross data splits: " + ", ".join(leaking)
            )
        leaking_wording = sorted(
            wording for wording, splits in wording_splits.items() if len(splits) != 1
        )
        if leaking_wording:
            raise ValueError(
                "wording templates cross data splits: " + ", ".join(leaking_wording)
            )
        if self.eligibility == "simulation-and-pilot-only" and any(
            bool(scenario.review["paper_eligible"]) for scenario in self.scenarios
        ):
            raise ValueError(
                "simulation-and-pilot-only catalogs cannot contain "
                "paper-eligible scenarios"
            )
        if self.catalog_status == "frozen-paper":
            if self.eligibility != "paper-eligible":
                raise ValueError(
                    "frozen-paper catalogs must declare paper-eligible"
                )
            if not all(
                scenario.status == "approved"
                and bool(scenario.review["paper_eligible"])
                for scenario in self.scenarios
            ):
                raise ValueError(
                    "frozen-paper catalogs require every scenario to be "
                    "approved and paper-eligible"
                )
        elif self.eligibility != "simulation-and-pilot-only":
            raise ValueError(
                "frozen-development catalogs must declare "
                "simulation-and-pilot-only"
            )
        for domain in ("travel", "writing"):
            for split in DATA_SPLITS:
                for target in range(3):
                    if not self.eligible(domain, split, target):
                        raise ValueError(
                            "catalog lacks an eligible scenario for "
                            f"{domain}/{split}/attribute-{target}"
                        )
        visible_by_split: dict[str, dict[str, str]] = {
            split: {} for split in DATA_SPLITS
        }
        for scenario in self.scenarios:
            surfaces = (
                scenario.prompt,
                *(option.label for option in scenario.options),
            )
            for surface in surfaces:
                normalized = _normal_surface(surface)
                owner = next(
                    (
                        other
                        for other in DATA_SPLITS
                        if other != scenario.split
                        and normalized in visible_by_split[other]
                    ),
                    None,
                )
                if owner is not None:
                    raise ValueError(
                        "visible scenario text crosses data splits: "
                        f"{scenario.scenario_id} and "
                        f"{visible_by_split[owner][normalized]}"
                    )
                visible_by_split[scenario.split][normalized] = scenario.scenario_id
        object.__setattr__(self, "_by_id", by_id)

    @classmethod
    def parse(cls, raw: Mapping[str, Any]) -> "ScenarioCatalog":
        _exact_keys(raw, _CATALOG_KEYS, "scenario catalog")
        if raw["schema_version"] != CATALOG_SCHEMA_VERSION:
            raise ValueError(
                f"scenario catalog schema_version must be {CATALOG_SCHEMA_VERSION}"
            )
        catalog_id = _identifier(raw["catalog_id"], "catalog_id")
        catalog_version = _text(
            raw["catalog_version"],
            "catalog_version",
            maximum=40,
        )
        if raw["catalog_status"] not in CATALOG_STATUSES:
            raise ValueError(
                f"catalog_status must be one of {sorted(CATALOG_STATUSES)}"
            )
        if raw["eligibility"] not in CATALOG_ELIGIBILITIES:
            raise ValueError(
                f"eligibility must be one of {sorted(CATALOG_ELIGIBILITIES)}"
            )
        for field_name in (
            "language",
            "locale",
            "license",
            "created_on",
            "frozen_on",
        ):
            _text(raw[field_name], field_name, maximum=100)
        if raw["source"] != "project-authored-synthetic":
            raise ValueError("source must be 'project-authored-synthetic'")
        if raw["split_policy"] != "scenario-family-disjoint-v1":
            raise ValueError("split_policy must be 'scenario-family-disjoint-v1'")
        if raw["selection_policy"] != SELECTION_POLICY:
            raise ValueError(f"selection_policy must be {SELECTION_POLICY!r}")
        attribute_order = _mapping(raw["attribute_order"], "attribute_order")
        if set(attribute_order) != {"travel", "writing"}:
            raise ValueError("attribute_order must contain travel and writing")
        expected_order = {
            domain.domain_id: tuple(attribute.key for attribute in domain.attributes)
            for domain in (get_domain("travel"), get_domain("writing"))
        }
        parsed_order: dict[str, tuple[str, str, str]] = {}
        for domain_id, expected in expected_order.items():
            values = attribute_order[domain_id]
            if not isinstance(values, list) or tuple(values) != expected:
                raise ValueError(
                    f"attribute_order[{domain_id!r}] must equal {list(expected)}"
                )
            parsed_order[domain_id] = expected
        authoring = _mapping(
            raw["authoring_provenance"],
            "authoring_provenance",
        )
        if not authoring:
            raise ValueError("authoring_provenance must not be empty")
        scenarios = raw["scenarios"]
        if not isinstance(scenarios, list) or not scenarios:
            raise ValueError("scenarios must be a non-empty array")
        parsed_scenarios = tuple(
            ScenarioSpec.parse(item, index=index)
            for index, item in enumerate(scenarios)
        )
        return cls(
            catalog_id=catalog_id,
            catalog_version=catalog_version,
            catalog_status=raw["catalog_status"],
            eligibility=raw["eligibility"],
            language=raw["language"],
            locale=raw["locale"],
            source=raw["source"],
            license=raw["license"],
            created_on=raw["created_on"],
            frozen_on=raw["frozen_on"],
            split_policy=raw["split_policy"],
            selection_policy=raw["selection_policy"],
            attribute_order=parsed_order,
            authoring_provenance=dict(authoring),
            scenarios=parsed_scenarios,
        )

    def eligible(
        self,
        domain: str,
        split: str,
        target_attribute: int,
    ) -> tuple[ScenarioSpec, ...]:
        return tuple(
            scenario
            for scenario in self.scenarios
            if scenario.domain == domain
            and scenario.split == split
            and scenario.target_attribute == target_attribute
            and scenario.status in {"provisional", "approved"}
        )

    def select(
        self,
        *,
        domain: str,
        split: str,
        target_attribute: int,
        seed: int,
        selection_key: Any,
    ) -> ScenarioSpec:
        candidates = tuple(
            sorted(
                self.eligible(domain, split, target_attribute),
                key=lambda scenario: scenario.scenario_id,
            )
        )
        if not candidates:
            raise ValueError(
                "no eligible catalog scenario for "
                f"{domain}/{split}/attribute-{target_attribute}"
            )
        index = semantic_seed(
            seed,
            "scenario-selection",
            self.catalog_id,
            self.catalog_version,
            domain,
            split,
            target_attribute,
            selection_key,
        ) % len(candidates)
        return candidates[index]

    def select_cycle(
        self,
        *,
        domain: str,
        split: str,
        target_attribute: int,
        seed: int,
        cycle_key: Any,
        occurrence_index: int,
    ) -> ScenarioSpec:
        """Select without replacement until one target cell is exhausted.

        ``cycle_key`` identifies the matched trajectory schedule.
        Counterfactual branches with the same key therefore receive the same
        scenario order, while successive occurrences of an attribute traverse
        every available scenario before any scenario repeats.
        """

        if (
            isinstance(occurrence_index, bool)
            or not isinstance(occurrence_index, int)
            or occurrence_index < 0
        ):
            raise ValueError("occurrence_index must be a non-negative integer")
        candidates = self.eligible(domain, split, target_attribute)
        if not candidates:
            raise ValueError(
                "no eligible catalog scenario for "
                f"{domain}/{split}/attribute-{target_attribute}"
            )
        ordered = tuple(
            sorted(
                candidates,
                key=lambda scenario: (
                    semantic_seed(
                        seed,
                        "scenario-cycle-order",
                        self.catalog_id,
                        self.catalog_version,
                        domain,
                        split,
                        target_attribute,
                        cycle_key,
                        scenario.scenario_id,
                    ),
                    scenario.scenario_id,
                ),
            )
        )
        return ordered[occurrence_index % len(ordered)]

    def scenario(self, scenario_id: str) -> ScenarioSpec:
        try:
            return self._by_id[scenario_id]
        except KeyError as exc:
            raise KeyError(f"unknown scenario ID: {scenario_id}") from exc

    def family_ids(self, *, split: str, domains: Sequence[str]) -> tuple[str, ...]:
        requested = set(domains)
        return tuple(
            sorted(
                {
                    scenario.family_id
                    for scenario in self.scenarios
                    if scenario.split == split and scenario.domain in requested
                }
            )
        )

    def option_ids(self, *, split: str, domains: Sequence[str]) -> tuple[str, ...]:
        requested = set(domains)
        return tuple(
            sorted(
                {
                    option.option_id
                    for scenario in self.scenarios
                    if scenario.split == split and scenario.domain in requested
                    for option in scenario.options
                }
            )
        )

    def wording_ids(self, *, split: str, domains: Sequence[str]) -> tuple[str, ...]:
        requested = set(domains)
        return tuple(
            sorted(
                {
                    scenario.wording_template_id
                    for scenario in self.scenarios
                    if scenario.split == split and scenario.domain in requested
                }
            )
        )

    def coverage_report(self) -> dict[str, Any]:
        cells = []
        for domain in ("travel", "writing"):
            for split in DATA_SPLITS:
                for target in range(3):
                    rows = self.eligible(domain, split, target)
                    cells.append(
                        {
                            "domain": domain,
                            "split": split,
                            "target_attribute": target,
                            "target_key": get_domain(domain).attributes[target].key,
                            "scenario_count": len(rows),
                            "family_count": len(
                                {scenario.family_id for scenario in rows}
                            ),
                            "task_families": sorted(
                                {scenario.task_family for scenario in rows}
                            ),
                            "nuisance_designs": sorted(
                                {
                                    (
                                        scenario.nuisance_key,
                                        scenario.nuisance_direction,
                                    )
                                    for scenario in rows
                                }
                            ),
                        }
                    )
        return {
            "schema_version": 1,
            "coverage_kind": "catalog_availability",
            "realized_consumption_artifact": ("metrics/split-leakage-audit.json"),
            "catalog_id": self.catalog_id,
            "catalog_version": self.catalog_version,
            "catalog_status": self.catalog_status,
            "eligibility": self.eligibility,
            "scenario_count": len(self.scenarios),
            "family_count": len({scenario.family_id for scenario in self.scenarios}),
            "approved_scenario_count": sum(
                scenario.status == "approved" for scenario in self.scenarios
            ),
            "provisional_scenario_count": sum(
                scenario.status == "provisional" for scenario in self.scenarios
            ),
            "paper_eligible": all(
                bool(scenario.review["paper_eligible"]) for scenario in self.scenarios
            ),
            "cells": cells,
        }


@dataclass(frozen=True, slots=True)
class LoadedScenarioCatalog:
    catalog: ScenarioCatalog
    source_path: Path
    source_bytes: bytes = field(repr=False)
    source_sha256: str

    def input_manifest(self) -> dict[str, Any]:
        coverage = self.catalog.coverage_report()
        return {
            "schema_version": 1,
            "input_kind": "scenario_catalog",
            "catalog_id": self.catalog.catalog_id,
            "catalog_version": self.catalog.catalog_version,
            "catalog_status": self.catalog.catalog_status,
            "eligibility": self.catalog.eligibility,
            "selection_policy": self.catalog.selection_policy,
            "source_filename": self.source_path.name,
            "source_sha256": self.source_sha256,
            "retained_file": "inputs/scenario-catalog.json",
            "scenario_count": coverage["scenario_count"],
            "family_count": coverage["family_count"],
            "paper_eligible": coverage["paper_eligible"],
        }


def load_scenario_catalog(
    path: str | Path,
    *,
    expected_sha256: str,
) -> LoadedScenarioCatalog:
    source = Path(path)
    raw = read_control_bytes(source, label="scenario catalog")
    actual = sha256(raw).hexdigest()
    if actual != expected_sha256:
        raise ValueError(
            "scenario catalog SHA-256 mismatch: "
            f"expected {expected_sha256}, received {actual}"
        )
    catalog = ScenarioCatalog.parse(
        _strict_json(raw, label=f"scenario catalog {source}")
    )
    return LoadedScenarioCatalog(
        catalog=catalog,
        source_path=source.resolve(),
        source_bytes=raw,
        source_sha256=actual,
    )


def materialize_context(
    context: InteractionContext,
    scenario: ScenarioSpec,
) -> InteractionContext:
    """Replace generic option surfaces with one validated catalog scenario."""

    if context.domain != scenario.domain:
        raise ValueError("scenario domain does not match interaction context")
    if context.target_attribute != scenario.target_attribute:
        raise ValueError("scenario target attribute does not match interaction context")
    replacements = {
        option.option_id: scenario.option_for_features(option.features)
        for option in context.options
    }
    return InteractionContext(
        context_id=f"{context.context_id}:scenario:{scenario.scenario_id}",
        options=tuple(replacements[option.option_id] for option in context.options),
        ranking=tuple(
            replacements[option_id].option_id for option_id in context.ranking
        ),
        domain=context.domain,
        scenario_id=scenario.scenario_id,
        turn_id=context.turn_id,
        default_option_id=(
            None
            if context.default_option_id is None
            else replacements[context.default_option_id].option_id
        ),
        suggested_option_id=(
            None
            if context.suggested_option_id is None
            else replacements[context.suggested_option_id].option_id
        ),
        wording_template=scenario.wording_template_id,
        question_type=context.question_type,
        target_attribute=context.target_attribute,
        prompt=scenario.prompt,
    )


def materialize_matched_anchor_set(
    matched: Any,
    scenario: ScenarioSpec,
) -> Any:
    """Materialize a MatchedAnchorSet without introducing an import cycle."""

    from .elicitation import MatchedAnchorSet

    contexts = {
        mechanism: materialize_context(context, scenario)
        for mechanism, context in matched.contexts.items()
    }
    anchor_context = contexts["balanced"]
    anchor = next(
        option
        for option in anchor_context.options
        if option.features[scenario.target_attribute] * matched.anchor_direction > 0
    )
    return MatchedAnchorSet(
        domain_id=matched.domain_id,
        scenario_id=scenario.scenario_id,
        target_attribute=matched.target_attribute,
        anchor_direction=matched.anchor_direction,
        anchor_option_id=anchor.option_id,
        contexts=contexts,
    )
