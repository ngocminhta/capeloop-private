"""Leakage-guarded surface and terminal held-out evaluation.

This module turns the proposal's split bookkeeping into executable evaluation
contracts.  It deliberately does not invoke an LLM: callers generate bound
requests from :class:`HeldOutParaphraseCase` objects, retain provider outputs,
and then construct :class:`ParaphraseEvaluationRecord` rows.

The terminal battery here is separate from the legacy projection battery.  Its
options, scenario families, and wording templates are genuinely absent from the
controlled training pools, and its scorer validates the question type and
wording binding before interpreting an action.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from string import Formatter
from typing import Any, Iterable, Mapping, Sequence
import json
import math

from .domains import DomainSpec
from .schemas import NUM_ATTRIBUTES, Theta, validate_theta


SPLITS = ("train", "development", "test")
MECHANISMS = ("balanced", "restricted", "ranking", "default", "suggested")
_SURFACE_FIELDS = frozenset(
    {"selected_label", "selected_ordinal", "selected_option_id", "domain"}
)
_TERMINAL_QUESTION_TYPES = frozenset(
    {
        "forced_choice",
        "counterfactual_choice",
        "direct_preference_probe",
        "cross_context_choice",
    }
)


def _require_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _digest(value: Any) -> str:
    return sha256(_canonical(value).encode("utf-8")).hexdigest()


def _validate_digest(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class SurfaceParaphraseTemplate:
    """One semantically constrained response template in exactly one split."""

    template_id: str
    family_id: str
    split: str
    pattern: str
    semantic_intent: str = "local_option_acceptance"

    def __post_init__(self) -> None:
        _require_text(self.template_id, "template_id")
        _require_text(self.family_id, "family_id")
        if self.split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        _require_text(self.pattern, "pattern")
        _require_text(self.semantic_intent, "semantic_intent")
        fields: set[str] = set()
        for _, field_name, format_spec, conversion in Formatter().parse(
            self.pattern
        ):
            if field_name is None:
                continue
            if field_name not in _SURFACE_FIELDS:
                raise ValueError(
                    f"unsupported paraphrase placeholder: {field_name!r}"
                )
            if format_spec or conversion:
                raise ValueError("paraphrase placeholders cannot use formatting")
            fields.add(field_name)
        if not fields & {"selected_label", "selected_option_id"}:
            raise ValueError(
                "a paraphrase must identify the selected option or its label"
            )

    @property
    def template_sha256(self) -> str:
        return _digest(
            {
                "template_id": self.template_id,
                "family_id": self.family_id,
                "split": self.split,
                "pattern": self.pattern,
                "semantic_intent": self.semantic_intent,
            }
        )

    def render(
        self,
        *,
        selected_label: str,
        selected_ordinal: str,
        selected_option_id: str,
        domain: str,
    ) -> str:
        values = {
            "selected_label": _require_text(selected_label, "selected_label"),
            "selected_ordinal": _require_text(
                selected_ordinal, "selected_ordinal"
            ),
            "selected_option_id": _require_text(
                selected_option_id, "selected_option_id"
            ),
            "domain": _require_text(domain, "domain"),
        }
        return self.pattern.format_map(values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "template_id": self.template_id,
            "family_id": self.family_id,
            "split": self.split,
            "pattern": self.pattern,
            "semantic_intent": self.semantic_intent,
            "template_sha256": self.template_sha256,
        }


@dataclass(frozen=True, slots=True)
class ParaphraseSuite:
    """Content-addressed template collection with family-level split guards."""

    suite_id: str
    templates: tuple[SurfaceParaphraseTemplate, ...]
    suite_sha256: str = ""

    def __post_init__(self) -> None:
        _require_text(self.suite_id, "suite_id")
        material = tuple(self.templates)
        if not material:
            raise ValueError("a paraphrase suite cannot be empty")
        if len({item.template_id for item in material}) != len(material):
            raise ValueError("paraphrase template IDs must be unique")
        if len({item.pattern for item in material}) != len(material):
            raise ValueError("paraphrase surface patterns must be unique")
        family_splits: dict[str, set[str]] = {}
        for item in material:
            family_splits.setdefault(item.family_id, set()).add(item.split)
        crossed = sorted(
            family for family, splits in family_splits.items() if len(splits) != 1
        )
        if crossed:
            raise ValueError(
                "paraphrase families cannot cross splits: " + ", ".join(crossed)
            )
        if set(item.split for item in material) != set(SPLITS):
            raise ValueError("a suite must contain train, development, and test")
        object.__setattr__(
            self,
            "templates",
            tuple(sorted(material, key=lambda item: item.template_id)),
        )
        payload = {
            "suite_id": self.suite_id,
            "templates": [item.to_dict() for item in self.templates],
        }
        expected = _digest(payload)
        if self.suite_sha256 and self.suite_sha256 != expected:
            raise ValueError("paraphrase suite digest does not match its content")
        object.__setattr__(self, "suite_sha256", expected)

    def for_split(self, split: str) -> tuple[SurfaceParaphraseTemplate, ...]:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        return tuple(item for item in self.templates if item.split == split)

    def assert_no_test_leakage(
        self,
        *,
        fitted_template_ids: Iterable[str] = (),
        fitted_template_sha256: Iterable[str] = (),
        fitted_surface_patterns: Iterable[str] = (),
    ) -> None:
        """Reject any test template reused during fitting or prompt development."""

        test = self.for_split("test")
        leaked_ids = sorted(
            {item.template_id for item in test} & set(fitted_template_ids)
        )
        leaked_digests = sorted(
            {item.template_sha256 for item in test}
            & set(fitted_template_sha256)
        )
        leaked_patterns = sorted(
            {item.pattern for item in test} & set(fitted_surface_patterns)
        )
        if leaked_ids or leaked_digests or leaked_patterns:
            raise ValueError(
                "held-out paraphrase leakage detected: "
                + _canonical(
                    {
                        "template_ids": leaked_ids,
                        "template_sha256": leaked_digests,
                        "surface_patterns": leaked_patterns,
                    }
                )
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "suite_id": self.suite_id,
            "suite_sha256": self.suite_sha256,
            "templates": [item.to_dict() for item in self.templates],
        }


def build_default_paraphrase_suite() -> ParaphraseSuite:
    """Return fixed response families with two independently held-out families."""

    return ParaphraseSuite(
        suite_id="cape-loop-surface-paraphrases-v1",
        templates=(
            SurfaceParaphraseTemplate(
                "surface-train-label-v1",
                "surface-train-explicit",
                "train",
                "I would choose {selected_label}.",
            ),
            SurfaceParaphraseTemplate(
                "surface-train-ordinal-v1",
                "surface-train-explicit",
                "train",
                "The {selected_ordinal} option works for me "
                "({selected_option_id}).",
            ),
            SurfaceParaphraseTemplate(
                "surface-development-contextual-v1",
                "surface-development-contextual",
                "development",
                "Among these {domain} choices, {selected_label} seems suitable.",
            ),
            SurfaceParaphraseTemplate(
                "surface-test-localized-v1",
                "surface-test-localized",
                "test",
                "Given only what is shown here, I can use {selected_label}.",
            ),
            SurfaceParaphraseTemplate(
                "surface-test-imperative-v1",
                "surface-test-imperative",
                "test",
                "For this choice, select {selected_label} "
                "({selected_option_id}).",
            ),
        ),
    )


@dataclass(frozen=True, slots=True)
class ParaphraseSource:
    """A structured response source, bound to its complete visible context."""

    source_trial_id: str
    domain_id: str
    mechanism: str
    selected_option_id: str
    selected_label: str
    selected_ordinal: str
    context_sha256: str

    def __post_init__(self) -> None:
        _require_text(self.source_trial_id, "source_trial_id")
        _require_text(self.domain_id, "domain_id")
        if self.mechanism not in MECHANISMS:
            raise ValueError(f"mechanism must be one of {MECHANISMS}")
        _require_text(self.selected_option_id, "selected_option_id")
        _require_text(self.selected_label, "selected_label")
        _require_text(self.selected_ordinal, "selected_ordinal")
        _validate_digest(self.context_sha256, "context_sha256")

    @classmethod
    def build(
        cls,
        *,
        source_trial_id: str,
        domain_id: str,
        mechanism: str,
        selected_option_id: str,
        selected_label: str,
        selected_ordinal: str,
        visible_context: Mapping[str, Any],
    ) -> "ParaphraseSource":
        if not isinstance(visible_context, Mapping) or not visible_context:
            raise ValueError("visible_context must be a nonempty object")
        return cls(
            source_trial_id=source_trial_id,
            domain_id=domain_id,
            mechanism=mechanism,
            selected_option_id=selected_option_id,
            selected_label=selected_label,
            selected_ordinal=selected_ordinal,
            context_sha256=_digest(visible_context),
        )


@dataclass(frozen=True, slots=True)
class HeldOutParaphraseCase:
    case_id: str
    source_trial_id: str
    domain_id: str
    mechanism: str
    selected_option_id: str
    template_id: str
    family_id: str
    split: str
    template_sha256: str
    context_sha256: str
    surface_response: str
    binding_sha256: str = ""

    def __post_init__(self) -> None:
        for name in (
            "case_id",
            "source_trial_id",
            "domain_id",
            "selected_option_id",
            "template_id",
            "family_id",
            "surface_response",
        ):
            _require_text(getattr(self, name), name)
        if self.mechanism not in MECHANISMS:
            raise ValueError(f"mechanism must be one of {MECHANISMS}")
        if self.split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        _validate_digest(self.template_sha256, "template_sha256")
        _validate_digest(self.context_sha256, "context_sha256")
        payload = self._binding_payload()
        expected = _digest(payload)
        if self.binding_sha256 and self.binding_sha256 != expected:
            raise ValueError("paraphrase case digest does not match its binding")
        object.__setattr__(self, "binding_sha256", expected)

    def _binding_payload(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "source_trial_id": self.source_trial_id,
            "domain_id": self.domain_id,
            "mechanism": self.mechanism,
            "selected_option_id": self.selected_option_id,
            "template_id": self.template_id,
            "family_id": self.family_id,
            "split": self.split,
            "template_sha256": self.template_sha256,
            "context_sha256": self.context_sha256,
            "surface_response": self.surface_response,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            **self._binding_payload(),
            "binding_sha256": self.binding_sha256,
        }


def generate_paraphrase_cases(
    sources: Iterable[ParaphraseSource],
    suite: ParaphraseSuite,
    *,
    split: str = "test",
) -> tuple[HeldOutParaphraseCase, ...]:
    """Render a deterministic source × template evaluation design."""

    templates = suite.for_split(split)
    material = tuple(sources)
    if not material:
        raise ValueError("at least one paraphrase source is required")
    if len({item.source_trial_id for item in material}) != len(material):
        raise ValueError("paraphrase source trial IDs must be unique")
    result: list[HeldOutParaphraseCase] = []
    for source in sorted(material, key=lambda item: item.source_trial_id):
        for template in templates:
            case_id = (
                f"{source.source_trial_id}:surface:{template.template_id}"
            )
            result.append(
                HeldOutParaphraseCase(
                    case_id=case_id,
                    source_trial_id=source.source_trial_id,
                    domain_id=source.domain_id,
                    mechanism=source.mechanism,
                    selected_option_id=source.selected_option_id,
                    template_id=template.template_id,
                    family_id=template.family_id,
                    split=template.split,
                    template_sha256=template.template_sha256,
                    context_sha256=source.context_sha256,
                    surface_response=template.render(
                        selected_label=source.selected_label,
                        selected_ordinal=source.selected_ordinal,
                        selected_option_id=source.selected_option_id,
                        domain=source.domain_id,
                    ),
                )
            )
    return tuple(result)


@dataclass(frozen=True, slots=True)
class ParaphraseEvaluationRecord:
    """One bound updater score for one surface case."""

    case_id: str
    binding_sha256: str
    source_trial_id: str
    template_id: str
    family_id: str
    split: str
    domain_id: str
    mechanism: str
    updater_id: str
    brier: float
    belief_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "case_id",
            "source_trial_id",
            "template_id",
            "family_id",
            "domain_id",
            "updater_id",
        ):
            _require_text(getattr(self, name), name)
        _validate_digest(self.binding_sha256, "binding_sha256")
        _validate_digest(self.belief_sha256, "belief_sha256")
        if self.split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        if self.mechanism not in MECHANISMS:
            raise ValueError(f"mechanism must be one of {MECHANISMS}")
        brier = _finite(self.brier, "brier")
        if not 0.0 <= brier <= 2.0:
            raise ValueError("brier must lie in [0, 2]")
        object.__setattr__(self, "brier", brier)

    @classmethod
    def from_case(
        cls,
        case: HeldOutParaphraseCase,
        *,
        updater_id: str,
        brier: float,
        belief_payload: Mapping[str, Any],
    ) -> "ParaphraseEvaluationRecord":
        if not isinstance(belief_payload, Mapping) or not belief_payload:
            raise ValueError("belief_payload must be a nonempty object")
        return cls(
            case_id=case.case_id,
            binding_sha256=case.binding_sha256,
            source_trial_id=case.source_trial_id,
            template_id=case.template_id,
            family_id=case.family_id,
            split=case.split,
            domain_id=case.domain_id,
            mechanism=case.mechanism,
            updater_id=updater_id,
            brier=brier,
            belief_sha256=_digest(belief_payload),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "case_id": self.case_id,
            "binding_sha256": self.binding_sha256,
            "source_trial_id": self.source_trial_id,
            "template_id": self.template_id,
            "family_id": self.family_id,
            "split": self.split,
            "domain_id": self.domain_id,
            "mechanism": self.mechanism,
            "updater_id": self.updater_id,
            "brier": self.brier,
            "belief_sha256": self.belief_sha256,
        }


@dataclass(frozen=True, slots=True)
class ParaphraseTransferCriterion:
    """Held-out surface readiness plus noncontrolling score diagnostics.

    ``verified`` is outcome neutral: it reflects complete updater/case
    coverage and structural preservation of the selected option and visible
    context.  The historical fitted-aware Brier gaps remain available in
    ``mean_gaps`` but do not determine Gate 1.
    """

    verified: bool | None
    complete: bool
    response_invariant: bool
    invariance_failures: tuple[str, ...]
    material_gap: float
    required_mechanisms: int
    covered_domains: tuple[str, ...]
    covered_template_ids: tuple[str, ...]
    expected_template_ids: tuple[str, ...]
    qualifying_mechanisms: tuple[str, ...]
    mean_gaps: tuple[tuple[str, str, str, float], ...]
    missing_pairs: tuple[str, ...]
    secondary_missing_pairs: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "criterion_id": "held-out-paraphrase-transfer",
            "verified": self.verified,
            "complete": self.complete,
            "response_invariant": self.response_invariant,
            "invariance_failures": list(self.invariance_failures),
            "material_gap": self.material_gap,
            "required_mechanisms": self.required_mechanisms,
            "covered_domains": list(self.covered_domains),
            "covered_template_ids": list(self.covered_template_ids),
            "expected_template_ids": list(self.expected_template_ids),
            "qualifying_mechanisms": list(self.qualifying_mechanisms),
            "mean_gaps": [
                {
                    "family_id": family,
                    "domain_id": domain,
                    "mechanism": mechanism,
                    "full_context_minus_aware_brier": gap,
                }
                for family, domain, mechanism, gap in self.mean_gaps
            ],
            "missing_pairs": list(self.missing_pairs),
            "secondary_missing_pairs": list(self.secondary_missing_pairs),
            "gate_1_argument": self.verified,
        }


def evaluate_gate1_paraphrase_transfer(
    cases: Iterable[HeldOutParaphraseCase],
    records: Iterable[ParaphraseEvaluationRecord],
    *,
    suite: ParaphraseSuite,
    full_context_updater_id: str = "llm_full_context",
    aware_updater_id: str = "fitted_action_aware",
    material_gap: float = 0.01,
    required_mechanisms: int = 2,
    required_domains: Sequence[str] = ("travel", "writing"),
    required_mechanism_ids: Sequence[str] | None = None,
) -> ParaphraseTransferCriterion:
    """Check held-out coverage/invariance and retain score gaps descriptively."""

    gap = _finite(material_gap, "material_gap")
    if gap < 0.0:
        raise ValueError("material_gap must be non-negative")
    if (
        isinstance(required_mechanisms, bool)
        or not isinstance(required_mechanisms, int)
        or required_mechanisms <= 0
    ):
        raise ValueError("required_mechanisms must be a positive integer")
    expected_domains = tuple(sorted(set(required_domains)))
    if len(expected_domains) != len(tuple(required_domains)):
        raise ValueError("required_domains must be unique")
    if not expected_domains:
        raise ValueError("required_domains cannot be empty")
    if required_mechanism_ids is None:
        expected_mechanisms = None
    else:
        expected_mechanisms = tuple(sorted(set(required_mechanism_ids)))
        if len(expected_mechanisms) != len(tuple(required_mechanism_ids)):
            raise ValueError("required_mechanism_ids must be unique")
        if not expected_mechanisms:
            raise ValueError("required_mechanism_ids cannot be empty")
        if not set(expected_mechanisms) <= set(MECHANISMS):
            raise ValueError(
                f"required_mechanism_ids must be selected from {MECHANISMS}"
            )

    test_cases = tuple(case for case in cases if case.split == "test")
    if len({case.case_id for case in test_cases}) != len(test_cases):
        raise ValueError("held-out case IDs must be unique")
    suite_templates = {
        item.template_id: item for item in suite.for_split("test")
    }
    for case in test_cases:
        try:
            template = suite_templates[case.template_id]
        except KeyError as exc:
            raise ValueError(
                f"held-out case uses unknown test template {case.template_id!r}"
            ) from exc
        if (
            case.template_sha256 != template.template_sha256
            or case.family_id != template.family_id
            or case.split != template.split
        ):
            raise ValueError(
                f"held-out case is not bound to suite template "
                f"{case.template_id!r}"
            )
    case_by_id = {case.case_id: case for case in test_cases}
    expected_templates = tuple(
        sorted(item.template_id for item in suite.for_split("test"))
    )
    rows = tuple(record for record in records if record.split == "test")
    row_keys = [(row.case_id, row.updater_id) for row in rows]
    if len(set(row_keys)) != len(row_keys):
        raise ValueError("duplicate updater score for held-out case")
    for row in rows:
        try:
            case = case_by_id[row.case_id]
        except KeyError as exc:
            raise ValueError(
                f"evaluation row references unknown test case {row.case_id!r}"
            ) from exc
        bound = (
            row.binding_sha256 == case.binding_sha256
            and row.source_trial_id == case.source_trial_id
            and row.template_id == case.template_id
            and row.family_id == case.family_id
            and row.domain_id == case.domain_id
            and row.mechanism == case.mechanism
        )
        if not bound:
            raise ValueError(f"evaluation row is not bound to case {case.case_id}")

    lookup = {(row.case_id, row.updater_id): row for row in rows}
    missing: list[str] = []
    secondary_missing: list[str] = []
    paired: list[
        tuple[HeldOutParaphraseCase, ParaphraseEvaluationRecord, ParaphraseEvaluationRecord]
    ] = []
    for case in test_cases:
        aware = lookup.get((case.case_id, aware_updater_id))
        full = lookup.get((case.case_id, full_context_updater_id))
        if aware is None:
            secondary_missing.append(
                f"{case.case_id}:{aware_updater_id}"
            )
        if full is None:
            missing.append(f"{case.case_id}:{full_context_updater_id}")
        if aware is not None and full is not None:
            paired.append((case, aware, full))

    grouped: dict[tuple[str, str, str], list[float]] = {}
    for case, aware, full in paired:
        grouped.setdefault(
            (case.family_id, case.domain_id, case.mechanism), []
        ).append(full.brier - aware.brier)
    mean_gaps = tuple(
        (
            family,
            domain,
            mechanism,
            math.fsum(values) / len(values),
        )
        for (family, domain, mechanism), values in sorted(grouped.items())
    )

    covered_domains = tuple(sorted({case.domain_id for case in test_cases}))
    covered_templates = tuple(sorted({case.template_id for case in test_cases}))
    families = tuple(
        sorted({item.family_id for item in suite.for_split("test")})
    )
    all_observed_mechanisms = tuple(
        sorted({case.mechanism for case in test_cases})
    )
    observed_mechanisms = tuple(
        sorted(
            {
                case.mechanism
                for case in test_cases
                if case.mechanism != "balanced"
            }
        )
    )
    coverage_mechanisms = (
        all_observed_mechanisms
        if expected_mechanisms is None
        else expected_mechanisms
    )
    mechanisms = (
        observed_mechanisms
        if expected_mechanisms is None
        else tuple(
            mechanism
            for mechanism in expected_mechanisms
            if mechanism != "balanced"
        )
    )
    mean_lookup = {
        (family, domain, mechanism): value
        for family, domain, mechanism, value in mean_gaps
    }
    qualifying = tuple(
        mechanism
        for mechanism in mechanisms
        if all(
            mean_lookup.get((family, domain, mechanism), -math.inf) > gap
            for family in families
            for domain in expected_domains
        )
    )

    by_source: dict[str, list[HeldOutParaphraseCase]] = {}
    for case in test_cases:
        by_source.setdefault(case.source_trial_id, []).append(case)
    invariance_failures: list[str] = []
    for source_trial_id, source_cases in sorted(by_source.items()):
        invariants = {
            "selected_option_id": {
                case.selected_option_id for case in source_cases
            },
            "context_sha256": {
                case.context_sha256 for case in source_cases
            },
            "domain_id": {case.domain_id for case in source_cases},
            "mechanism": {case.mechanism for case in source_cases},
        }
        for field, values in invariants.items():
            if len(values) != 1:
                invariance_failures.append(
                    f"{source_trial_id}:{field}"
                )
    response_invariant = not invariance_failures
    design_complete = (
        set(covered_domains) == set(expected_domains)
        and set(covered_templates) == set(expected_templates)
        and (
            expected_mechanisms is None
            or {
                case.mechanism for case in test_cases
            } == set(expected_mechanisms)
        )
        and all(
            any(
                case.template_id == template_id
                and case.domain_id == domain
                and case.mechanism == mechanism
                for case in test_cases
            )
            for template_id in expected_templates
            for domain in expected_domains
            for mechanism in coverage_mechanisms
        )
    )
    complete = design_complete and not missing
    verified = (
        None
        if not complete
        else response_invariant
    )
    return ParaphraseTransferCriterion(
        verified=verified,
        complete=complete,
        response_invariant=response_invariant,
        invariance_failures=tuple(invariance_failures),
        material_gap=gap,
        required_mechanisms=required_mechanisms,
        covered_domains=covered_domains,
        covered_template_ids=covered_templates,
        expected_template_ids=expected_templates,
        qualifying_mechanisms=qualifying,
        mean_gaps=mean_gaps,
        missing_pairs=tuple(sorted(missing)),
        secondary_missing_pairs=tuple(sorted(secondary_missing)),
    )


@dataclass(frozen=True, slots=True)
class HeldOutTerminalOption:
    option_id: str
    label: str
    features: tuple[float, float, float]

    def __post_init__(self) -> None:
        _require_text(self.option_id, "option_id")
        _require_text(self.label, "label")
        if len(self.features) != NUM_ATTRIBUTES:
            raise ValueError("terminal option must have three features")
        object.__setattr__(
            self,
            "features",
            tuple(
                _finite(value, f"features[{index}]")
                for index, value in enumerate(self.features)
            ),
        )

    @property
    def fingerprint(self) -> str:
        return _digest(
            {
                "option_id": self.option_id,
                "label": self.label,
                "features": self.features,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "option_id": self.option_id,
            "label": self.label,
            "features": list(self.features),
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True, slots=True)
class HeldOutTerminalItem:
    item_id: str
    family_id: str
    domain_id: str
    scenario_family_id: str
    wording_template_id: str
    question_type: str
    prompt: str
    options: tuple[HeldOutTerminalOption, ...] = ()
    target_attribute: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "item_id",
            "family_id",
            "domain_id",
            "scenario_family_id",
            "wording_template_id",
            "prompt",
        ):
            _require_text(getattr(self, name), name)
        if self.question_type not in _TERMINAL_QUESTION_TYPES:
            raise ValueError(
                "unknown terminal question type: " + self.question_type
            )
        options = tuple(self.options)
        object.__setattr__(self, "options", options)
        if len({item.option_id for item in options}) != len(options):
            raise ValueError("terminal option IDs must be unique within an item")
        if self.question_type == "direct_preference_probe":
            if options:
                raise ValueError("direct preference probes cannot contain options")
            if self.target_attribute is None:
                raise ValueError("direct probes require a target attribute")
        elif len(options) < 2:
            raise ValueError("choice terminal items require at least two options")
        if self.target_attribute is not None and (
            isinstance(self.target_attribute, bool)
            or not isinstance(self.target_attribute, int)
            or not 0 <= self.target_attribute < NUM_ATTRIBUTES
        ):
            raise ValueError("target_attribute must be 0, 1, 2, or None")

    @property
    def item_sha256(self) -> str:
        return _digest(self._payload())

    def _payload(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "family_id": self.family_id,
            "domain_id": self.domain_id,
            "scenario_family_id": self.scenario_family_id,
            "wording_template_id": self.wording_template_id,
            "question_type": self.question_type,
            "prompt": self.prompt,
            "options": [item.to_dict() for item in self.options],
            "target_attribute": self.target_attribute,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "item_sha256": self.item_sha256}


@dataclass(frozen=True, slots=True)
class HeldOutTerminalSuite:
    suite_id: str
    domain_id: str
    items: tuple[HeldOutTerminalItem, ...]
    suite_sha256: str = ""

    def __post_init__(self) -> None:
        _require_text(self.suite_id, "suite_id")
        _require_text(self.domain_id, "domain_id")
        items = tuple(self.items)
        if not items:
            raise ValueError("terminal suite cannot be empty")
        if len({item.item_id for item in items}) != len(items):
            raise ValueError("terminal item IDs must be unique")
        if any(item.domain_id != self.domain_id for item in items):
            raise ValueError("terminal item has the wrong domain")
        if set(item.question_type for item in items) != _TERMINAL_QUESTION_TYPES:
            raise ValueError("terminal suite must exercise every question type")
        object.__setattr__(
            self, "items", tuple(sorted(items, key=lambda item: item.item_id))
        )
        expected = _digest(
            {
                "suite_id": self.suite_id,
                "domain_id": self.domain_id,
                "items": [item.to_dict() for item in self.items],
            }
        )
        if self.suite_sha256 and self.suite_sha256 != expected:
            raise ValueError("terminal suite digest does not match its content")
        object.__setattr__(self, "suite_sha256", expected)

    def assert_genuinely_held_out(
        self,
        *,
        training_option_ids: Iterable[str],
        training_feature_vectors: Iterable[Sequence[float]],
        training_wording_template_ids: Iterable[str],
        training_scenario_family_ids: Iterable[str] = (),
    ) -> None:
        option_ids = {
            option.option_id for item in self.items for option in item.options
        }
        feature_vectors = {
            tuple(option.features) for item in self.items for option in item.options
        }
        wording_ids = {item.wording_template_id for item in self.items}
        scenario_ids = {item.scenario_family_id for item in self.items}
        overlaps = {
            "option_ids": sorted(option_ids & set(training_option_ids)),
            "feature_vectors": sorted(
                feature_vectors
                & {
                    tuple(float(value) for value in vector)
                    for vector in training_feature_vectors
                }
            ),
            "wording_template_ids": sorted(
                wording_ids & set(training_wording_template_ids)
            ),
            "scenario_family_ids": sorted(
                scenario_ids & set(training_scenario_family_ids)
            ),
        }
        if any(overlaps.values()):
            raise ValueError(
                "terminal suite overlaps training material: " + _canonical(overlaps)
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "suite_id": self.suite_id,
            "domain_id": self.domain_id,
            "suite_sha256": self.suite_sha256,
            "items": [item.to_dict() for item in self.items],
        }


def build_heldout_terminal_suite(
    domain: DomainSpec,
    *,
    version: str = "heldout-terminal-v2",
) -> HeldOutTerminalSuite:
    """Build novel options and distinct response contracts for one domain."""

    _require_text(version, "version")
    family = f"{domain.domain_id}:{version}"
    scenario = f"{domain.domain_id}:novel-transfer-scenarios-v2"
    items: list[HeldOutTerminalItem] = []
    for attribute, spec in enumerate(domain.attributes):
        negative_features = [0.0, 0.0, 0.0]
        positive_features = [0.0, 0.0, 0.0]
        negative_features[attribute] = -0.73
        positive_features[attribute] = 0.73
        negative = HeldOutTerminalOption(
            option_id=f"{family}:novel-{spec.key}:negative",
            label=f"unseen scenario emphasizing {spec.negative_label}",
            features=tuple(negative_features),  # type: ignore[arg-type]
        )
        positive = HeldOutTerminalOption(
            option_id=f"{family}:novel-{spec.key}:positive",
            label=f"unseen scenario emphasizing {spec.positive_label}",
            features=tuple(positive_features),  # type: ignore[arg-type]
        )
        items.extend(
            (
                HeldOutTerminalItem(
                    item_id=f"{family}:forced:{attribute}",
                    family_id=f"{family}:isolated-transfer",
                    domain_id=domain.domain_id,
                    scenario_family_id=scenario,
                    wording_template_id=f"{family}:choose-between-novel-options",
                    question_type="forced_choice",
                    prompt=(
                        "For this new situation only, which of the two "
                        "descriptions would you choose?"
                    ),
                    options=(negative, positive),
                    target_attribute=attribute,
                ),
                HeldOutTerminalItem(
                    item_id=f"{family}:counterfactual:{attribute}",
                    family_id=f"{family}:counterbalanced-transfer",
                    domain_id=domain.domain_id,
                    scenario_family_id=scenario,
                    wording_template_id=f"{family}:counterbalanced-order",
                    question_type="counterfactual_choice",
                    prompt=(
                        "The descriptions are shown in reverse order. Which "
                        "one better fits the user's general preference?"
                    ),
                    options=(positive, negative),
                    target_attribute=attribute,
                ),
                HeldOutTerminalItem(
                    item_id=f"{family}:probe:{attribute}",
                    family_id=f"{family}:neutral-probes",
                    domain_id=domain.domain_id,
                    scenario_family_id=scenario,
                    wording_template_id=f"{family}:direction-without-options",
                    question_type="direct_preference_probe",
                    prompt=(
                        "Without referring to an earlier choice, is the user's "
                        f"general preference closer to {spec.negative_label} "
                        f"(-1) or {spec.positive_label} (+1)?"
                    ),
                    target_attribute=attribute,
                ),
            )
        )

    # These feature magnitudes and combinations do not occur in the controlled
    # ±0.5 domain pools.  They require applying more than one latent attribute.
    cross_pairs = (
        ((-0.61, 0.29, 0.17), (0.61, -0.29, -0.17)),
        ((0.23, -0.67, 0.31), (-0.23, 0.67, -0.31)),
    )
    for index, (left_features, right_features) in enumerate(cross_pairs):
        items.append(
            HeldOutTerminalItem(
                item_id=f"{family}:cross-context:{index}",
                family_id=f"{family}:cross-context-transfer",
                domain_id=domain.domain_id,
                scenario_family_id=scenario,
                wording_template_id=f"{family}:novel-composite-{index}",
                question_type="cross_context_choice",
                prompt=(
                    "Apply the inferred preferences to this unfamiliar "
                    "combination and select the better fit."
                ),
                options=(
                    HeldOutTerminalOption(
                        f"{family}:composite-{index}:left",
                        f"unseen composite {index + 1}A",
                        left_features,
                    ),
                    HeldOutTerminalOption(
                        f"{family}:composite-{index}:right",
                        f"unseen composite {index + 1}B",
                        right_features,
                    ),
                ),
            )
        )
    suite = HeldOutTerminalSuite(
        suite_id=family,
        domain_id=domain.domain_id,
        items=tuple(items),
    )
    suite.assert_genuinely_held_out(
        training_option_ids=(
            option.option_id
            for option in domain.option_pool + domain.isolated_options
        ),
        training_feature_vectors=(
            option.features
            for option in domain.option_pool + domain.isolated_options
        ),
        training_wording_template_ids=(
            "neutral_choice",
            "heldout_neutral_choice",
            "heldout_counterbalanced_choice",
            "heldout_neutral_direct_probe",
            "heldout_cross_context_choice",
        ),
        training_scenario_family_ids=(
            "travel-hotel",
            "travel-itinerary",
            "writing-revision",
            "terminal-diagnostic",
        ),
    )
    return suite


@dataclass(frozen=True, slots=True)
class TerminalAction:
    """An action bound to the exact item wording and response contract."""

    item_id: str
    item_sha256: str
    wording_template_id: str
    question_type: str
    selected_option_id: str | None = None
    declared_direction: int | None = None

    def __post_init__(self) -> None:
        _require_text(self.item_id, "item_id")
        _validate_digest(self.item_sha256, "item_sha256")
        _require_text(self.wording_template_id, "wording_template_id")
        if self.question_type not in _TERMINAL_QUESTION_TYPES:
            raise ValueError("unknown terminal action question type")
        if self.question_type == "direct_preference_probe":
            if self.selected_option_id is not None:
                raise ValueError("direct probes cannot select an option")
            if self.declared_direction not in (-1, 1):
                raise ValueError("direct probes require direction -1 or +1")
        else:
            if (
                not isinstance(self.selected_option_id, str)
                or not self.selected_option_id.strip()
            ):
                raise ValueError("choice actions require selected_option_id")
            if self.declared_direction is not None:
                raise ValueError("choice actions cannot declare a direction")


@dataclass(frozen=True, slots=True)
class HeldOutTerminalScore:
    overall_accuracy: float
    fractional_accuracy: float
    mean_choice_regret: float
    evaluated_item_count: int
    accuracy_by_question_type: tuple[tuple[str, float], ...]
    count_by_question_type: tuple[tuple[str, int], ...]
    intrinsic_tie_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "score_contract": "wording-and-question-type-aware-v1",
            "overall_accuracy": self.overall_accuracy,
            "fractional_accuracy": self.fractional_accuracy,
            "mean_choice_regret": self.mean_choice_regret,
            "evaluated_item_count": self.evaluated_item_count,
            "accuracy_by_question_type": dict(self.accuracy_by_question_type),
            "count_by_question_type": dict(self.count_by_question_type),
            "intrinsic_tie_count": self.intrinsic_tie_count,
        }


def score_heldout_terminal_actions(
    suite: HeldOutTerminalSuite,
    actions: Iterable[TerminalAction],
    truth: Theta,
    *,
    tie_tolerance: float = 1e-12,
) -> HeldOutTerminalScore:
    """Score actions only after exact item, wording, and question-type binding."""

    canonical_truth = validate_theta(truth)
    tolerance = _finite(tie_tolerance, "tie_tolerance")
    if tolerance < 0:
        raise ValueError("tie_tolerance must be non-negative")
    material = tuple(actions)
    if len({action.item_id for action in material}) != len(material):
        raise ValueError("terminal actions contain duplicate item IDs")
    action_by_id = {action.item_id: action for action in material}
    expected_ids = {item.item_id for item in suite.items}
    if set(action_by_id) != expected_ids:
        raise ValueError(
            "terminal actions must cover the suite exactly: "
            + _canonical(
                {
                    "missing": sorted(expected_ids - set(action_by_id)),
                    "unexpected": sorted(set(action_by_id) - expected_ids),
                }
            )
        )

    scores: list[float] = []
    fractional: list[float] = []
    regrets: list[float] = []
    by_type: dict[str, list[float]] = {}
    ties = 0
    for item in suite.items:
        action = action_by_id[item.item_id]
        if (
            action.item_sha256 != item.item_sha256
            or action.wording_template_id != item.wording_template_id
            or action.question_type != item.question_type
        ):
            raise ValueError(
                f"terminal action binding mismatch for {item.item_id}"
            )
        if item.question_type == "direct_preference_probe":
            assert item.target_attribute is not None
            correct = action.declared_direction == (
                1 if canonical_truth[item.target_attribute] > 0 else -1
            )
            score = 1.0 if correct else 0.0
            fraction = score
        else:
            assert action.selected_option_id is not None
            option_by_id = {
                option.option_id: option for option in item.options
            }
            if action.selected_option_id not in option_by_id:
                raise ValueError(
                    f"action selected an option absent from {item.item_id}"
                )
            utilities = {
                option.option_id: math.fsum(
                    coefficient * feature
                    for coefficient, feature in zip(
                        canonical_truth, option.features
                    )
                )
                for option in item.options
            }
            optimum = max(utilities.values())
            maximizers = {
                option_id
                for option_id, value in utilities.items()
                if math.isclose(
                    value, optimum, rel_tol=0.0, abs_tol=tolerance
                )
            }
            ties += len(maximizers) > 1
            correct = action.selected_option_id in maximizers
            score = 1.0 if correct else 0.0
            fraction = (1.0 / len(maximizers)) if correct else 0.0
            regrets.append(optimum - utilities[action.selected_option_id])
        scores.append(score)
        fractional.append(fraction)
        by_type.setdefault(item.question_type, []).append(score)

    return HeldOutTerminalScore(
        overall_accuracy=math.fsum(scores) / len(scores),
        fractional_accuracy=math.fsum(fractional) / len(fractional),
        mean_choice_regret=(
            0.0 if not regrets else math.fsum(regrets) / len(regrets)
        ),
        evaluated_item_count=len(scores),
        accuracy_by_question_type=tuple(
            (question_type, math.fsum(values) / len(values))
            for question_type, values in sorted(by_type.items())
        ),
        count_by_question_type=tuple(
            (question_type, len(values))
            for question_type, values in sorted(by_type.items())
        ),
        intrinsic_tie_count=ties,
    )
