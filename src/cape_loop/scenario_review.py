"""Version-bound, outcome-blind review evidence for scenario promotion.

The scenario audit creates fillable response contracts.  This module verifies
completed copies against the exact catalog, conversation bank, audit, and
opaque item map; aggregates only prespecified criteria; and, when every
criterion passes, derives a new paper catalog in a fresh directory.  It never
edits either source input and never treats catalog status strings as evidence.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import date
from hashlib import sha256
from pathlib import Path
from statistics import median, stdev
from typing import Any, Mapping, Sequence
import json
import math

from .artifacts import canonical_json, file_sha256
from .conversation_surfaces import ConversationTemplateBank
from .scenario_calibration import AUDIT_POLICY, AUDIT_SCHEMA_VERSION
from .scenarios import ScenarioCatalog


REVIEW_SCHEMA_VERSION = 1
REVIEW_CONTRACT_VERSION = "scenario-review-evidence-v1"
MIN_SURFACE_REVIEWERS = 2
MIN_SCIENTIFIC_REVIEWERS = 2
SURFACE_MEDIAN_MINIMUM = 4.0
TARGET_DIRECTION_RECOVERY_MINIMUM = 0.90
UNINTENDED_DIMENSION_MAXIMUM = 0.10
NEUTRAL_CHOICE_MINIMUM_PER_ITEM = 40
NEUTRAL_CHOICE_HARD_FLOOR = 0.20
NEUTRAL_CHOICE_PRIMARY_RANGE = (0.30, 0.70)
NEUTRAL_CHOICE_WILSON_RANGE = (0.20, 0.80)
ATTRACTIVENESS_MINIMUM_PER_ITEM = 80
ATTRACTIVENESS_EQUIVALENCE_MARGIN = 0.20
NORMAL_90_Z = 1.6448536269514722

_EVIDENCE_KINDS = frozenset(
    {
        "surface_review",
        "scientific_review",
        "neutral_choice_pretest",
        "masked_attractiveness_pretest",
    }
)
_FACT_ROLES = frozenset({"target", "nuisance", "held_constant"})
_COMPONENT_IDS = (
    "prompt",
    "option_a",
    "option_b",
    "option_a_peer",
    "option_b_peer",
)


def _digest(value: Any) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _exact(value: object, keys: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    observed = set(value)
    if observed != keys:
        missing = sorted(keys - observed)
        unknown = sorted(observed - keys)
        raise ValueError(
            f"{label} fields must be exact (missing={missing}; unknown={unknown})"
        )
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _true(value: object, label: str) -> None:
    if value is not True:
        raise ValueError(f"{label} must be true")


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"review evidence must be a regular file: {path}")

    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{path} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        decoded = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"{path} contains non-finite number {value}")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read review evidence {path}: {exc}") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return decoded


def _source_binding(
    catalog: ScenarioCatalog,
    bank: ConversationTemplateBank,
    *,
    catalog_sha256: str,
    conversation_bank_sha256: str,
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "catalog_id": catalog.catalog_id,
        "catalog_version": catalog.catalog_version,
        "catalog_sha256": catalog_sha256,
        "conversation_bank_id": bank.bank_id,
        "conversation_bank_sha256": conversation_bank_sha256,
        "audit_schema_version": audit.get("schema_version"),
        "audit_policy": audit.get("audit_policy"),
        "audit_selected_split": audit.get("catalog", {}).get("selected_split"),
    }


def _blank_surface_template(
    protocol_id: str,
    binding: Mapping[str, Any],
    items: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "contract_version": REVIEW_CONTRACT_VERSION,
        "evidence_kind": "surface_review",
        "protocol_id": protocol_id,
        "source_binding": dict(binding),
        "reviewer": {
            "reviewer_id": "",
            "independent_from_authors_and_other_reviewers": None,
            "outcome_blind": None,
        },
        "items": [
            {
                "item_id": item["item_id"],
                "material": deepcopy(item["surface_material"]),
                "ratings": [
                    {
                        "surface_id": surface["surface_id"],
                        "naturalness": None,
                        "neutrality": None,
                        "notes": "",
                    }
                    for surface in item["surface_material"]["surfaces"]
                ],
                "assertions": {
                    "grammatical_and_coherent": None,
                    "comparable_clarity_and_specificity": None,
                    "no_unexplained_choice_pressure": None,
                },
            }
            for item in items
        ],
    }


def _blank_scientific_template(
    protocol_id: str,
    binding: Mapping[str, Any],
    items: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "contract_version": REVIEW_CONTRACT_VERSION,
        "evidence_kind": "scientific_review",
        "protocol_id": protocol_id,
        "source_binding": dict(binding),
        "reviewer": {
            "reviewer_id": "",
            "independent_from_authors_and_other_reviewers": None,
            "outcome_blind": None,
        },
        "items": [
            {
                "item_id": item["item_id"],
                "material": deepcopy(item["scientific_material"]),
                "fact_mapping": [
                    {
                        "component_id": component,
                        "visible_fact_count": None,
                        "mapped_fact_count": None,
                        "unmodeled_fact_count": None,
                        "ambiguous_fact_count": None,
                        "cross_loading_fact_count": None,
                        "assigned_roles": [],
                        "notes": "",
                    }
                    for component in _COMPONENT_IDS
                ],
                "judgments": {
                    "recovered_positive_option": "",
                    "unintended_dimension_assignment_count": None,
                    "tradeoff_valid": None,
                    "non_dominating": None,
                    "treatment_isolated": None,
                },
                "masking_review": {
                    "masked_option_a": "",
                    "masked_option_b": "",
                    "target_language_removed": None,
                    "non_target_facts_preserved": None,
                },
                "warning_dispositions": [
                    {
                        "warning_id": warning_id,
                        "disposition": "",
                        "rationale": "",
                    }
                    for warning_id in item["warning_ids"]
                ],
            }
            for item in items
        ],
    }


def _blank_choice_template(
    protocol_id: str,
    binding: Mapping[str, Any],
    items: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "contract_version": REVIEW_CONTRACT_VERSION,
        "evidence_kind": "neutral_choice_pretest",
        "protocol_id": protocol_id,
        "source_binding": dict(binding),
        "collection": {
            "collector_id": "",
            "preregistration_id": "",
            "collected_before_evaluated_model_outcomes": None,
            "exclusions_applied_as_preregistered": None,
            "independent_participants": None,
        },
        "items": [
            {
                "item_id": item["item_id"],
                "material": deepcopy(item["choice_material"]),
                "responses": [],
            }
            for item in items
        ],
    }


def _blank_attractiveness_template(
    protocol_id: str,
    binding: Mapping[str, Any],
    items: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "contract_version": REVIEW_CONTRACT_VERSION,
        "evidence_kind": "masked_attractiveness_pretest",
        "protocol_id": protocol_id,
        "source_binding": dict(binding),
        "collection": {
            "collector_id": "",
            "preregistration_id": "",
            "collected_before_evaluated_model_outcomes": None,
            "exclusions_applied_as_preregistered": None,
            "independent_participants": None,
            "masking_scientific_reviewer_ids": [],
        },
        "items": [
            {
                "item_id": item["item_id"],
                "unmasked_material": deepcopy(item["choice_material"]),
                "masked_option_a": "",
                "masked_option_b": "",
                "target_language_masked": None,
                "responses": [],
            }
            for item in items
        ],
    }


def build_scenario_review_kit(
    catalog: ScenarioCatalog,
    bank: ConversationTemplateBank,
    audit: Mapping[str, Any],
    *,
    catalog_sha256: str,
    conversation_bank_sha256: str,
) -> dict[str, dict[str, Any]]:
    """Create one deterministic reviewer kit from exact frozen inputs."""

    if audit.get("schema_version") != AUDIT_SCHEMA_VERSION:
        raise ValueError(f"audit schema_version must be {AUDIT_SCHEMA_VERSION}")
    if audit.get("audit_policy") != AUDIT_POLICY:
        raise ValueError(f"audit_policy must be {AUDIT_POLICY!r}")
    bank.validate_catalog(catalog)
    rows = audit.get("human_review_packet", {}).get("scenarios")
    if not isinstance(rows, list) or not rows:
        raise ValueError("audit must contain human-review scenarios")
    by_scenario = {scenario.scenario_id: scenario for scenario in catalog.scenarios}
    row_by_scenario = {str(row.get("scenario_id")): row for row in rows}
    if set(row_by_scenario) - set(by_scenario):
        raise ValueError("audit contains a scenario outside the bound catalog")

    binding = _source_binding(
        catalog,
        bank,
        catalog_sha256=catalog_sha256,
        conversation_bank_sha256=conversation_bank_sha256,
        audit=audit,
    )
    ordered_scenario_ids = sorted(
        row_by_scenario,
        key=lambda scenario_id: sha256(
            (
                catalog_sha256
                + conversation_bank_sha256
                + REVIEW_CONTRACT_VERSION
                + scenario_id
            ).encode("utf-8")
        ).hexdigest(),
    )
    mapping_entries = []
    for index, scenario_id in enumerate(ordered_scenario_ids, start=1):
        scenario = by_scenario[scenario_id]
        swap = (
            int(
                sha256(
                    (conversation_bank_sha256 + scenario_id).encode("utf-8")
                ).hexdigest(),
                16,
            )
            % 2
            == 1
        )
        mapping_entries.append(
            {
                "item_id": f"review-item-{index:03d}",
                "scenario_id": scenario_id,
                "scenario_revision": scenario.revision,
                "option_a_role": "positive" if swap else "negative",
                "option_b_role": "negative" if swap else "positive",
            }
        )
    mapping_core = {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "contract_version": REVIEW_CONTRACT_VERSION,
        "source_binding": binding,
        "items": mapping_entries,
    }
    mapping_id = _digest(mapping_core)
    mapping = {
        **mapping_core,
        "mapping_id": mapping_id,
        "access": "researcher_only_do_not_give_to_blinded_reviewers_or_pretest_participants",
    }

    scenario_to_item = {entry["scenario_id"]: entry for entry in mapping_entries}
    warning_records = []
    warning_ids_by_scenario: dict[str, list[str]] = {
        scenario_id: [] for scenario_id in ordered_scenario_ids
    }
    for index, warning in enumerate(audit.get("warnings", {}).get("all", []), start=1):
        warning_id = f"machine-warning-{index:03d}"
        affected = sorted(
            {
                scenario_to_item[scenario_id]["item_id"]
                for key in ("scenario_id", "scenario_a", "scenario_b")
                if isinstance((scenario_id := warning.get(key)), str)
                and scenario_id in scenario_to_item
            }
        )
        for item_id in affected:
            scenario_id = next(
                entry["scenario_id"]
                for entry in mapping_entries
                if entry["item_id"] == item_id
            )
            warning_ids_by_scenario[scenario_id].append(warning_id)
        warning_records.append(
            {
                "warning_id": warning_id,
                "kind": warning.get("kind"),
                "affected_item_ids": affected,
                "review_rule": (
                    "both scientific reviewers must independently record "
                    "resolved_valid with a non-empty rationale; otherwise revise"
                ),
            }
        )

    items = []
    for entry in mapping_entries:
        scenario = by_scenario[entry["scenario_id"]]
        row = row_by_scenario[scenario.scenario_id]
        negative = scenario.negative_option
        positive = scenario.positive_option
        negative_peer = scenario.negative_same_direction_option
        positive_peer = scenario.positive_same_direction_option
        if entry["option_a_role"] == "negative":
            option_a, option_b = negative, positive
            peer_a, peer_b = negative_peer, positive_peer
        else:
            option_a, option_b = positive, negative
            peer_a, peer_b = positive_peer, negative_peer
        surface_material = {
            "surfaces": [
                {
                    "surface_id": f"surface-{surface_index:02d}",
                    "assistant_message": example["assistant_message"],
                    "user_message": example["user_message"],
                }
                for surface_index, example in enumerate(
                    row["rendered_examples"], start=1
                )
            ]
        }
        choice_material = {
            "prompt": scenario.prompt,
            "option_a": option_a.label,
            "option_b": option_b.label,
            "presentation": "balanced_without_default_suggestion_or_rank_emphasis",
        }
        scientific_material = {
            "domain": scenario.domain,
            "target_dimension": scenario.target_key,
            "nuisance_dimension": scenario.nuisance_key,
            "components": [
                {"component_id": "prompt", "text": scenario.prompt},
                {"component_id": "option_a", "text": option_a.label},
                {"component_id": "option_b", "text": option_b.label},
                {"component_id": "option_a_peer", "text": peer_a.label},
                {"component_id": "option_b_peer", "text": peer_b.label},
            ],
        }
        items.append(
            {
                "item_id": entry["item_id"],
                "surface_material": surface_material,
                "choice_material": choice_material,
                "scientific_material": scientific_material,
                "warning_ids": warning_ids_by_scenario[scenario.scenario_id],
            }
        )

    thresholds = {
        "minimum_distinct_surface_reviewers": MIN_SURFACE_REVIEWERS,
        "minimum_distinct_scientific_reviewers": MIN_SCIENTIFIC_REVIEWERS,
        "surface_scenario_median_minimum": SURFACE_MEDIAN_MINIMUM,
        "target_direction_recovery_minimum": TARGET_DIRECTION_RECOVERY_MINIMUM,
        "unintended_dimension_assignment_maximum": UNINTENDED_DIMENSION_MAXIMUM,
        "neutral_choice_minimum_responses_per_item": (NEUTRAL_CHOICE_MINIMUM_PER_ITEM),
        "neutral_choice_each_option_hard_floor": NEUTRAL_CHOICE_HARD_FLOOR,
        "neutral_choice_primary_observed_range_inclusive": list(
            NEUTRAL_CHOICE_PRIMARY_RANGE
        ),
        "neutral_choice_90_percent_wilson_interval_must_lie_within": list(
            NEUTRAL_CHOICE_WILSON_RANGE
        ),
        "attractiveness_minimum_paired_responses_per_item": (
            ATTRACTIVENESS_MINIMUM_PER_ITEM
        ),
        "attractiveness_standardized_equivalence_margin": (
            ATTRACTIVENESS_EQUIVALENCE_MARGIN
        ),
        "attractiveness_interval": (
            "two-sided_90_percent_normal_approximation_for_paired_standardized_mean"
        ),
    }
    protocol_core = {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "contract_version": REVIEW_CONTRACT_VERSION,
        "source_binding": binding,
        "mapping_id": mapping_id,
        "item_count": len(items),
        "outcome_data_allowed": False,
        "evaluated_model_outputs_allowed": False,
        "thresholds": thresholds,
        "machine_warnings": warning_records,
    }
    protocol_id = _digest(protocol_core)
    protocol = {**protocol_core, "protocol_id": protocol_id}
    return {
        "protocol": protocol,
        "mapping": mapping,
        "surface_template": _blank_surface_template(protocol_id, binding, items),
        "scientific_template": _blank_scientific_template(protocol_id, binding, items),
        "neutral_choice_template": _blank_choice_template(protocol_id, binding, items),
        "attractiveness_template": _blank_attractiveness_template(
            protocol_id, binding, items
        ),
    }


def render_review_kit_markdown(kit: Mapping[str, Mapping[str, Any]]) -> str:
    """Render the surface template without revealing its private item map."""

    template = kit["surface_template"]
    lines = [
        "# Blinded scenario surface-review packet",
        "",
        "> Use a separate completed copy of `surface-review.template.json` for ",
        "> each reviewer. Do not give reviewers the researcher-only item map.",
        "",
        "Rate every surface from 1 (poor) to 5 (fully natural/neutral).",
    ]
    for item in template["items"]:
        lines.extend(["", "---", "", f"## {item['item_id']}", ""])
        for surface in item["material"]["surfaces"]:
            lines.extend(
                [
                    f"### {surface['surface_id']}",
                    "",
                    f"> **Assistant:** {surface['assistant_message']}",
                    ">",
                    f"> **User:** {surface['user_message']}",
                    "",
                    "- Naturalness (1–5): ____",
                    "- Neutrality (1–5): ____",
                    "- Notes: ________________________________________________",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def _validate_common(
    payload: Mapping[str, Any],
    *,
    kind: str,
    protocol: Mapping[str, Any],
) -> None:
    if payload.get("schema_version") != REVIEW_SCHEMA_VERSION:
        raise ValueError(f"{kind} schema_version must be {REVIEW_SCHEMA_VERSION}")
    if payload.get("contract_version") != REVIEW_CONTRACT_VERSION:
        raise ValueError(f"{kind} contract_version is not supported")
    if payload.get("evidence_kind") != kind:
        raise ValueError(f"expected evidence_kind {kind!r}")
    if payload.get("protocol_id") != protocol["protocol_id"]:
        raise ValueError(f"{kind} protocol_id does not match the frozen protocol")
    if payload.get("source_binding") != protocol["source_binding"]:
        raise ValueError(f"{kind} source binding does not match")


def _reviewer(payload: Mapping[str, Any], label: str) -> str:
    reviewer = _exact(
        payload.get("reviewer"),
        {
            "reviewer_id",
            "independent_from_authors_and_other_reviewers",
            "outcome_blind",
        },
        f"{label}.reviewer",
    )
    reviewer_id = _text(reviewer["reviewer_id"], f"{label}.reviewer_id")
    _true(
        reviewer["independent_from_authors_and_other_reviewers"],
        f"{label}.independent_from_authors_and_other_reviewers",
    )
    _true(reviewer["outcome_blind"], f"{label}.outcome_blind")
    return reviewer_id


def _validate_item_material(
    received: Sequence[Any],
    expected: Sequence[Any],
    *,
    label: str,
    material_key: str,
) -> None:
    if len(received) != len(expected):
        raise ValueError(f"{label} must cover every protocol item exactly once")
    received_ids = [
        item.get("item_id") for item in received if isinstance(item, Mapping)
    ]
    expected_ids = [item["item_id"] for item in expected]
    if received_ids != expected_ids:
        raise ValueError(f"{label} item order/coverage does not match the template")
    for index, (item, reference) in enumerate(zip(received, expected)):
        if not isinstance(item, Mapping):
            raise TypeError(f"{label}.items[{index}] must be an object")
        if item.get(material_key) != reference[material_key]:
            raise ValueError(f"{label} material changed for {reference['item_id']}")


def _surface_report(
    reviews: Sequence[Mapping[str, Any]],
    template: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> tuple[dict[str, Any], set[str]]:
    expected = template["items"]
    reviewers: set[str] = set()
    natural: dict[str, list[int]] = {item["item_id"]: [] for item in expected}
    neutral: dict[str, list[int]] = {item["item_id"]: [] for item in expected}
    assertion_failures: list[str] = []
    for review_index, review in enumerate(reviews):
        label = f"surface_review[{review_index}]"
        _exact(
            review,
            {
                "schema_version",
                "contract_version",
                "evidence_kind",
                "protocol_id",
                "source_binding",
                "reviewer",
                "items",
            },
            label,
        )
        _validate_common(review, kind="surface_review", protocol=protocol)
        reviewer_id = _reviewer(review, label)
        if reviewer_id in reviewers:
            raise ValueError("surface reviewer IDs must be distinct")
        reviewers.add(reviewer_id)
        items = review["items"]
        if not isinstance(items, list):
            raise TypeError(f"{label}.items must be an array")
        _validate_item_material(items, expected, label=label, material_key="material")
        for item, reference in zip(items, expected):
            _exact(
                item,
                {"item_id", "material", "ratings", "assertions"},
                f"{label}.{reference['item_id']}",
            )
            ratings = item["ratings"]
            if not isinstance(ratings, list) or len(ratings) != len(
                reference["ratings"]
            ):
                raise ValueError(f"{label} must rate every surface")
            for rating, expected_rating in zip(ratings, reference["ratings"]):
                row = _exact(
                    rating,
                    {"surface_id", "naturalness", "neutrality", "notes"},
                    f"{label}.rating",
                )
                if row["surface_id"] != expected_rating["surface_id"]:
                    raise ValueError(f"{label} surface IDs do not match")
                naturalness = _integer(
                    row["naturalness"], f"{label}.naturalness", minimum=1
                )
                neutrality = _integer(
                    row["neutrality"], f"{label}.neutrality", minimum=1
                )
                if naturalness > 5 or neutrality > 5:
                    raise ValueError("surface ratings must lie from 1 to 5")
                if not isinstance(row["notes"], str):
                    raise TypeError("surface notes must be a string")
                natural[item["item_id"]].append(naturalness)
                neutral[item["item_id"]].append(neutrality)
            assertions = _exact(
                item["assertions"],
                {
                    "grammatical_and_coherent",
                    "comparable_clarity_and_specificity",
                    "no_unexplained_choice_pressure",
                },
                f"{label}.assertions",
            )
            for name, value in assertions.items():
                if value is not True:
                    assertion_failures.append(f"{item['item_id']}:{name}")
    if len(reviewers) < MIN_SURFACE_REVIEWERS:
        raise ValueError(
            f"at least {MIN_SURFACE_REVIEWERS} distinct surface reviewers are required"
        )
    item_reports = []
    for item_id in natural:
        natural_median = float(median(natural[item_id]))
        neutral_median = float(median(neutral[item_id]))
        item_reports.append(
            {
                "item_id": item_id,
                "naturalness_median": natural_median,
                "neutrality_median": neutral_median,
                "passed": (
                    natural_median >= SURFACE_MEDIAN_MINIMUM
                    and neutral_median >= SURFACE_MEDIAN_MINIMUM
                    and not any(
                        value.startswith(item_id + ":") for value in assertion_failures
                    )
                ),
            }
        )
    return (
        {
            "reviewer_count": len(reviewers),
            "threshold": SURFACE_MEDIAN_MINIMUM,
            "assertion_failures": assertion_failures,
            "items": item_reports,
            "passed": not assertion_failures
            and all(item["passed"] for item in item_reports),
        },
        reviewers,
    )


def _scientific_report(
    reviews: Sequence[Mapping[str, Any]],
    template: Mapping[str, Any],
    protocol: Mapping[str, Any],
    positive_option_by_item: Mapping[str, str],
) -> tuple[dict[str, Any], set[str], dict[str, tuple[str, str]]]:
    expected = template["items"]
    reviewers: set[str] = set()
    mapping_signatures: dict[str, list[tuple[Any, ...]]] = {
        item["item_id"]: [] for item in expected
    }
    masking_signatures: dict[str, list[tuple[str, str]]] = {
        item["item_id"]: [] for item in expected
    }
    item_failures: dict[str, list[str]] = {item["item_id"]: [] for item in expected}
    recovered = 0
    direction_judgments = 0
    unintended = 0
    mapped_facts = 0
    for review_index, review in enumerate(reviews):
        label = f"scientific_review[{review_index}]"
        _exact(
            review,
            {
                "schema_version",
                "contract_version",
                "evidence_kind",
                "protocol_id",
                "source_binding",
                "reviewer",
                "items",
            },
            label,
        )
        _validate_common(review, kind="scientific_review", protocol=protocol)
        reviewer_id = _reviewer(review, label)
        if reviewer_id in reviewers:
            raise ValueError("scientific reviewer IDs must be distinct")
        reviewers.add(reviewer_id)
        items = review["items"]
        if not isinstance(items, list):
            raise TypeError(f"{label}.items must be an array")
        _validate_item_material(items, expected, label=label, material_key="material")
        for item, reference in zip(items, expected):
            item_id = item["item_id"]
            _exact(
                item,
                {
                    "item_id",
                    "material",
                    "fact_mapping",
                    "judgments",
                    "masking_review",
                    "warning_dispositions",
                },
                f"{label}.{item_id}",
            )
            mappings = item["fact_mapping"]
            if not isinstance(mappings, list) or len(mappings) != len(_COMPONENT_IDS):
                raise ValueError(f"{label}.{item_id} fact mapping is incomplete")
            signature = []
            for component, component_id in zip(mappings, _COMPONENT_IDS):
                row = _exact(
                    component,
                    {
                        "component_id",
                        "visible_fact_count",
                        "mapped_fact_count",
                        "unmodeled_fact_count",
                        "ambiguous_fact_count",
                        "cross_loading_fact_count",
                        "assigned_roles",
                        "notes",
                    },
                    f"{label}.{item_id}.fact_mapping",
                )
                if row["component_id"] != component_id:
                    raise ValueError(f"{label}.{item_id} component order changed")
                visible = _integer(
                    row["visible_fact_count"], "visible_fact_count", minimum=1
                )
                mapped = _integer(row["mapped_fact_count"], "mapped_fact_count")
                unmodeled = _integer(
                    row["unmodeled_fact_count"], "unmodeled_fact_count"
                )
                ambiguous = _integer(
                    row["ambiguous_fact_count"], "ambiguous_fact_count"
                )
                cross = _integer(
                    row["cross_loading_fact_count"], "cross_loading_fact_count"
                )
                roles = row["assigned_roles"]
                if (
                    not isinstance(roles, list)
                    or not roles
                    or len(roles) != len(set(roles))
                    or any(role not in _FACT_ROLES for role in roles)
                ):
                    raise ValueError(
                        "assigned_roles must be a distinct non-empty role list"
                    )
                if not isinstance(row["notes"], str):
                    raise TypeError("fact-mapping notes must be a string")
                mapped_facts += mapped
                if mapped != visible or unmodeled or ambiguous or cross:
                    item_failures[item_id].append(
                        f"{reviewer_id}:{component_id}:incomplete_or_invalid_mapping"
                    )
                signature.append((component_id, visible, tuple(sorted(roles))))
            mapping_signatures[item_id].append(tuple(signature))
            judgments = _exact(
                item["judgments"],
                {
                    "recovered_positive_option",
                    "unintended_dimension_assignment_count",
                    "tradeoff_valid",
                    "non_dominating",
                    "treatment_isolated",
                },
                f"{label}.{item_id}.judgments",
            )
            direction_judgments += 1
            recovered_option = judgments["recovered_positive_option"]
            if recovered_option not in {"a", "b"}:
                raise ValueError("recovered_positive_option must be a or b")
            if recovered_option == positive_option_by_item[item_id]:
                recovered += 1
            unexpected = _integer(
                judgments["unintended_dimension_assignment_count"],
                "unintended_dimension_assignment_count",
            )
            unintended += unexpected
            for name in ("tradeoff_valid", "non_dominating", "treatment_isolated"):
                if judgments[name] is not True:
                    item_failures[item_id].append(f"{reviewer_id}:{name}")
            masking = _exact(
                item["masking_review"],
                {
                    "masked_option_a",
                    "masked_option_b",
                    "target_language_removed",
                    "non_target_facts_preserved",
                },
                f"{label}.{item_id}.masking_review",
            )
            masked_a = _text(masking["masked_option_a"], "masked_option_a")
            masked_b = _text(masking["masked_option_b"], "masked_option_b")
            if masked_a == masked_b:
                raise ValueError("masked alternatives must remain distinguishable")
            for name in ("target_language_removed", "non_target_facts_preserved"):
                if masking[name] is not True:
                    item_failures[item_id].append(f"{reviewer_id}:masking:{name}")
            masking_signatures[item_id].append((masked_a, masked_b))
            dispositions = item["warning_dispositions"]
            expected_dispositions = reference["warning_dispositions"]
            if not isinstance(dispositions, list) or len(dispositions) != len(
                expected_dispositions
            ):
                raise ValueError(f"{label}.{item_id} warning coverage is incomplete")
            for disposition, expected_disposition in zip(
                dispositions, expected_dispositions
            ):
                row = _exact(
                    disposition,
                    {"warning_id", "disposition", "rationale"},
                    f"{label}.{item_id}.warning_disposition",
                )
                if row["warning_id"] != expected_disposition["warning_id"]:
                    raise ValueError("warning disposition IDs do not match")
                if (
                    row["disposition"] != "resolved_valid"
                    or not isinstance(row["rationale"], str)
                    or not row["rationale"].strip()
                ):
                    item_failures[item_id].append(
                        f"{reviewer_id}:{row['warning_id']}:unresolved"
                    )
    if len(reviewers) < MIN_SCIENTIFIC_REVIEWERS:
        raise ValueError(
            f"at least {MIN_SCIENTIFIC_REVIEWERS} distinct scientific reviewers are required"
        )
    disagreements = []
    for item_id, signatures in mapping_signatures.items():
        if len(set(signatures)) != 1:
            disagreements.append(item_id)
            item_failures[item_id].append("independent_fact_mappings_disagree")
        if len(set(masking_signatures[item_id])) != 1:
            disagreements.append(item_id + ":masking")
            item_failures[item_id].append("independent_masking_reviews_disagree")
    recovery_rate = recovered / direction_judgments
    unintended_rate = unintended / mapped_facts if mapped_facts else None
    direction_passed = recovery_rate >= TARGET_DIRECTION_RECOVERY_MINIMUM
    unintended_passed = (
        unintended_rate is not None and unintended_rate <= UNINTENDED_DIMENSION_MAXIMUM
    )
    item_reports = [
        {
            "item_id": item_id,
            "failures": failures,
            "passed": not failures,
        }
        for item_id, failures in item_failures.items()
    ]
    approved_masks = {
        item_id: signatures[0] for item_id, signatures in masking_signatures.items()
    }
    return (
        {
            "reviewer_count": len(reviewers),
            "fact_mapping_disagreement_item_ids": disagreements,
            "target_direction_recovery_rate": recovery_rate,
            "target_direction_recovery_minimum": TARGET_DIRECTION_RECOVERY_MINIMUM,
            "unintended_dimension_assignment_rate": unintended_rate,
            "unintended_dimension_assignment_maximum": UNINTENDED_DIMENSION_MAXIMUM,
            "items": item_reports,
            "passed": (
                direction_passed
                and unintended_passed
                and all(item["passed"] for item in item_reports)
            ),
        },
        reviewers,
        approved_masks,
    )


def _collection(
    payload: Mapping[str, Any], label: str, *, masked: bool
) -> tuple[str, set[str]]:
    keys = {
        "collector_id",
        "preregistration_id",
        "collected_before_evaluated_model_outcomes",
        "exclusions_applied_as_preregistered",
        "independent_participants",
    }
    if masked:
        keys.add("masking_scientific_reviewer_ids")
    collection = _exact(payload.get("collection"), keys, f"{label}.collection")
    collector = _text(collection["collector_id"], f"{label}.collector_id")
    _text(collection["preregistration_id"], f"{label}.preregistration_id")
    for field in (
        "collected_before_evaluated_model_outcomes",
        "exclusions_applied_as_preregistered",
        "independent_participants",
    ):
        _true(collection[field], f"{label}.{field}")
    reviewers: set[str] = set()
    if masked:
        raw = collection["masking_scientific_reviewer_ids"]
        if not isinstance(raw, list) or len(raw) < MIN_SCIENTIFIC_REVIEWERS:
            raise ValueError("masking must be approved by both scientific reviewers")
        reviewers = {_text(value, "masking reviewer ID") for value in raw}
        if len(reviewers) != len(raw):
            raise ValueError("masking reviewer IDs must be distinct")
    return collector, reviewers


def _wilson(successes: int, total: int) -> tuple[float, float]:
    z = NORMAL_90_Z
    estimate = successes / total
    denominator = 1.0 + z * z / total
    center = (estimate + z * z / (2.0 * total)) / denominator
    half = (
        z
        * math.sqrt(estimate * (1.0 - estimate) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return center - half, center + half


def _choice_report(
    payload: Mapping[str, Any],
    template: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    _exact(
        payload,
        {
            "schema_version",
            "contract_version",
            "evidence_kind",
            "protocol_id",
            "source_binding",
            "collection",
            "items",
        },
        "neutral_choice_pretest",
    )
    _validate_common(payload, kind="neutral_choice_pretest", protocol=protocol)
    collector, _ = _collection(payload, "neutral_choice_pretest", masked=False)
    items = payload["items"]
    expected = template["items"]
    if not isinstance(items, list):
        raise TypeError("neutral choice items must be an array")
    _validate_item_material(
        items, expected, label="neutral_choice_pretest", material_key="material"
    )
    reports = []
    for item, reference in zip(items, expected):
        _exact(item, {"item_id", "material", "responses"}, "neutral choice item")
        responses = item["responses"]
        if not isinstance(responses, list):
            raise TypeError("neutral choice responses must be an array")
        participants: set[str] = set()
        option_a = 0
        order_counts = {"a_first": 0, "b_first": 0}
        for response in responses:
            row = _exact(
                response,
                {"participant_id", "display_order", "selected_option"},
                "neutral choice response",
            )
            participant = _text(row["participant_id"], "participant_id")
            if participant in participants:
                raise ValueError("a participant may respond only once per choice item")
            participants.add(participant)
            if row["display_order"] not in order_counts:
                raise ValueError("display_order must be a_first or b_first")
            if row["selected_option"] not in {"a", "b"}:
                raise ValueError("selected_option must be a or b")
            order_counts[row["display_order"]] += 1
            option_a += row["selected_option"] == "a"
        total = len(responses)
        if total < NEUTRAL_CHOICE_MINIMUM_PER_ITEM:
            raise ValueError(
                f"{item['item_id']} requires at least "
                f"{NEUTRAL_CHOICE_MINIMUM_PER_ITEM} choice responses"
            )
        if abs(order_counts["a_first"] - order_counts["b_first"]) > 1:
            raise ValueError(f"{item['item_id']} display order is not balanced")
        proportion = option_a / total
        interval = _wilson(option_a, total)
        passed = (
            min(proportion, 1.0 - proportion) >= NEUTRAL_CHOICE_HARD_FLOOR
            and NEUTRAL_CHOICE_PRIMARY_RANGE[0]
            <= proportion
            <= NEUTRAL_CHOICE_PRIMARY_RANGE[1]
            and interval[0] >= NEUTRAL_CHOICE_WILSON_RANGE[0]
            and interval[1] <= NEUTRAL_CHOICE_WILSON_RANGE[1]
        )
        reports.append(
            {
                "item_id": reference["item_id"],
                "response_count": total,
                "option_a_share": proportion,
                "option_a_90_percent_wilson_interval": list(interval),
                "display_order_counts": order_counts,
                "passed": passed,
            }
        )
    return {
        "items": reports,
        "passed": all(item["passed"] for item in reports),
    }, collector


def _attractiveness_report(
    payload: Mapping[str, Any],
    template: Mapping[str, Any],
    protocol: Mapping[str, Any],
    approved_masks: Mapping[str, tuple[str, str]],
) -> tuple[dict[str, Any], str, set[str]]:
    _exact(
        payload,
        {
            "schema_version",
            "contract_version",
            "evidence_kind",
            "protocol_id",
            "source_binding",
            "collection",
            "items",
        },
        "masked_attractiveness_pretest",
    )
    _validate_common(payload, kind="masked_attractiveness_pretest", protocol=protocol)
    collector, mask_reviewers = _collection(
        payload, "masked_attractiveness_pretest", masked=True
    )
    items = payload["items"]
    expected = template["items"]
    if not isinstance(items, list) or len(items) != len(expected):
        raise ValueError("attractiveness pretest must cover every item")
    reports = []
    for item, reference in zip(items, expected):
        row = _exact(
            item,
            {
                "item_id",
                "unmasked_material",
                "masked_option_a",
                "masked_option_b",
                "target_language_masked",
                "responses",
            },
            "attractiveness item",
        )
        if (
            row["item_id"] != reference["item_id"]
            or row["unmasked_material"] != reference["unmasked_material"]
        ):
            raise ValueError("attractiveness item identity/material changed")
        masked_a = _text(row["masked_option_a"], "masked_option_a")
        masked_b = _text(row["masked_option_b"], "masked_option_b")
        if masked_a == masked_b:
            raise ValueError("masked alternatives must remain distinguishable")
        if (masked_a, masked_b) != approved_masks.get(str(row["item_id"])):
            raise ValueError(
                "attractiveness masks do not match both scientific reviews"
            )
        _true(row["target_language_masked"], "target_language_masked")
        responses = row["responses"]
        if not isinstance(responses, list):
            raise TypeError("attractiveness responses must be an array")
        participants: set[str] = set()
        differences = []
        order_counts = {"a_first": 0, "b_first": 0}
        for response in responses:
            response_row = _exact(
                response,
                {"participant_id", "display_order", "rating_a", "rating_b"},
                "attractiveness response",
            )
            participant = _text(response_row["participant_id"], "participant_id")
            if participant in participants:
                raise ValueError(
                    "a participant may respond only once per attractiveness item"
                )
            participants.add(participant)
            if response_row["display_order"] not in order_counts:
                raise ValueError("display_order must be a_first or b_first")
            order_counts[response_row["display_order"]] += 1
            rating_a = _integer(response_row["rating_a"], "rating_a", minimum=1)
            rating_b = _integer(response_row["rating_b"], "rating_b", minimum=1)
            if rating_a > 5 or rating_b > 5:
                raise ValueError("attractiveness ratings must lie from 1 to 5")
            differences.append(float(rating_a - rating_b))
        total = len(differences)
        if total < ATTRACTIVENESS_MINIMUM_PER_ITEM:
            raise ValueError(
                f"{item['item_id']} requires at least "
                f"{ATTRACTIVENESS_MINIMUM_PER_ITEM} attractiveness responses"
            )
        if abs(order_counts["a_first"] - order_counts["b_first"]) > 1:
            raise ValueError(f"{item['item_id']} display order is not balanced")
        mean_difference = math.fsum(differences) / total
        difference_sd = stdev(differences)
        if difference_sd == 0.0:
            standardized = (
                0.0
                if mean_difference == 0.0
                else math.copysign(math.inf, mean_difference)
            )
            half_width = 0.0 if mean_difference == 0.0 else math.inf
        else:
            standardized = mean_difference / difference_sd
            half_width = NORMAL_90_Z / math.sqrt(total)
        interval = (standardized - half_width, standardized + half_width)
        passed = (
            math.isfinite(standardized)
            and interval[0] >= -ATTRACTIVENESS_EQUIVALENCE_MARGIN
            and interval[1] <= ATTRACTIVENESS_EQUIVALENCE_MARGIN
        )
        finite_estimate = math.isfinite(standardized) and all(
            math.isfinite(value) for value in interval
        )
        reports.append(
            {
                "item_id": reference["item_id"],
                "response_count": total,
                "paired_standardized_mean_difference": (
                    standardized if finite_estimate else None
                ),
                "paired_standardized_90_percent_interval": (
                    list(interval) if finite_estimate else [None, None]
                ),
                "display_order_counts": order_counts,
                "passed": passed,
            }
        )
    return (
        {"items": reports, "passed": all(item["passed"] for item in reports)},
        collector,
        mask_reviewers,
    )


def verify_scenario_review_evidence(
    kit: Mapping[str, Mapping[str, Any]],
    evidence_payloads: Sequence[Mapping[str, Any]],
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Strictly validate and aggregate completed review response objects."""

    by_kind: dict[str, list[Mapping[str, Any]]] = {kind: [] for kind in _EVIDENCE_KINDS}
    for payload in evidence_payloads:
        kind = payload.get("evidence_kind")
        if kind not in _EVIDENCE_KINDS:
            raise ValueError(f"unknown scenario-review evidence kind: {kind!r}")
        by_kind[str(kind)].append(payload)
    if len(by_kind["surface_review"]) != MIN_SURFACE_REVIEWERS:
        raise ValueError(
            f"exactly {MIN_SURFACE_REVIEWERS} surface reviews are required"
        )
    if len(by_kind["scientific_review"]) != MIN_SCIENTIFIC_REVIEWERS:
        raise ValueError(
            f"exactly {MIN_SCIENTIFIC_REVIEWERS} scientific reviews are required"
        )
    for kind in ("neutral_choice_pretest", "masked_attractiveness_pretest"):
        if len(by_kind[kind]) != 1:
            raise ValueError(f"exactly one {kind} file is required")

    protocol = kit["protocol"]
    surface, surface_reviewers = _surface_report(
        by_kind["surface_review"], kit["surface_template"], protocol
    )
    positive_option_by_item = {
        item["item_id"]: ("a" if item["option_a_role"] == "positive" else "b")
        for item in kit["mapping"]["items"]
    }
    scientific, scientific_reviewers, approved_masks = _scientific_report(
        by_kind["scientific_review"],
        kit["scientific_template"],
        protocol,
        positive_option_by_item,
    )
    if surface_reviewers & scientific_reviewers:
        raise ValueError("surface and scientific reviewer sets must be distinct")
    choice, choice_collector = _choice_report(
        by_kind["neutral_choice_pretest"][0],
        kit["neutral_choice_template"],
        protocol,
    )
    attractiveness, attractiveness_collector, mask_reviewers = _attractiveness_report(
        by_kind["masked_attractiveness_pretest"][0],
        kit["attractiveness_template"],
        protocol,
        approved_masks,
    )
    if mask_reviewers != scientific_reviewers:
        raise ValueError(
            "masked attractiveness materials require approval by the exact "
            "two scientific reviewers"
        )
    human_ids = surface_reviewers | scientific_reviewers
    if choice_collector in human_ids or attractiveness_collector in human_ids:
        raise ValueError("pretest collectors must be independent of reviewers")
    if choice_collector == attractiveness_collector:
        raise ValueError(
            "choice and attractiveness pretests require distinct collectors"
        )

    engineering = audit.get("readiness", {}).get("engineering_pilot", {})
    automated_criteria = {
        "engineering_audit_ready": engineering.get("ready") is True,
        "probability_guardrails_pass": audit.get("probability_calibration", {}).get(
            "all_cells_passed"
        )
        is True,
        "nuisance_design_pass": audit.get("restricted_peer_nuisance_design", {}).get(
            "all_cells_passed"
        )
        is True,
        "conversation_frame_design_pass": audit.get(
            "conversation_frame_design", {}
        ).get("passed")
        is True,
        "machine_rendering_complete": audit.get("machine_surface_rendering", {}).get(
            "complete"
        )
        is True,
        "machine_rendering_hygiene_clean": audit.get(
            "machine_surface_rendering", {}
        ).get("hygiene_clean")
        is True,
    }
    automated_passed = all(automated_criteria.values())
    criteria = {
        "automated_validation_evidence_verified": automated_passed,
        "independent_surface_human_review_passed": surface["passed"],
        "independent_scientific_human_review_passed": scientific["passed"],
        "neutral_choice_pretest_passed": choice["passed"],
        "masked_attractiveness_equivalence_passed": attractiveness["passed"],
        "whole_catalog_is_covered": (
            protocol["item_count"]
            == audit.get("catalog", {}).get("selected_scenario_count")
            and audit.get("catalog", {}).get("selected_split") == "all"
        ),
    }
    passed = all(criteria.values())
    return {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "artifact_kind": "scenario-review-evidence-report",
        "contract_version": REVIEW_CONTRACT_VERSION,
        "protocol_id": protocol["protocol_id"],
        "source_binding": protocol["source_binding"],
        "outcome_data_used": False,
        "evaluated_model_outputs_used": False,
        "catalog_review_strings_used_as_evidence": False,
        "automated_validation": {
            "criteria": automated_criteria,
            "passed": automated_passed,
        },
        "surface_review": surface,
        "scientific_review": scientific,
        "neutral_choice_pretest": choice,
        "masked_attractiveness_pretest": attractiveness,
        "promotion_criteria": criteria,
        "promotion_eligible": passed,
        "readiness": {
            "scientific_pilot": {
                "ready": passed,
                "criteria": criteria,
                "blocking_reasons": sorted(
                    name for name, value in criteria.items() if not value
                ),
                "evidence_source": "verified_external_review_bundle",
            },
            "paper_inputs": {
                "ready": passed,
                "requires_new_derived_catalog": True,
            },
        },
        "claim_status": "no_experiment_outcome_claim",
    }


def derive_reviewed_catalog(
    source_payload: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    new_catalog_version: str,
    frozen_on: str,
) -> dict[str, Any]:
    """Return a validated paper catalog only after verified evidence passes."""

    if report.get("promotion_eligible") is not True:
        raise ValueError(
            "review evidence did not pass; no reviewed catalog may be derived"
        )
    version = _text(new_catalog_version, "new_catalog_version")
    if version == source_payload.get("catalog_version"):
        raise ValueError("reviewed catalog must use a new catalog version")
    try:
        date.fromisoformat(frozen_on)
    except (TypeError, ValueError) as exc:
        raise ValueError("frozen_on must use YYYY-MM-DD") from exc
    result = deepcopy(dict(source_payload))
    result["catalog_version"] = version
    result["catalog_status"] = "frozen-paper"
    result["eligibility"] = "paper-eligible"
    provenance = dict(result["authoring_provenance"])
    provenance["independent_review_evidence"] = {
        "contract_version": REVIEW_CONTRACT_VERSION,
        "protocol_id": report["protocol_id"],
        "source_catalog_sha256": report["source_binding"]["catalog_sha256"],
        "source_conversation_bank_sha256": report["source_binding"][
            "conversation_bank_sha256"
        ],
        "outcome_data_used": False,
    }
    result["authoring_provenance"] = provenance
    result["frozen_on"] = frozen_on
    for scenario in result["scenarios"]:
        scenario["status"] = "approved"
        scenario["review"] = {
            "automated_validation": "passed",
            "surface_human_review": "passed",
            "scientific_human_review": "passed",
            "paper_eligible": True,
            "note": (
                "Approved by independent version-bound review protocol "
                f"{report['protocol_id']}; see the external evidence report."
            ),
        }
    ScenarioCatalog.parse(result)
    return result


def derive_reviewed_conversation_bank(
    source_bank: ConversationTemplateBank,
    report: Mapping[str, Any],
    *,
    new_catalog_version: str,
) -> dict[str, Any]:
    """Give the reviewed companion bank a truthful new identity/provenance."""

    if report.get("promotion_eligible") is not True:
        raise ValueError(
            "review evidence did not pass; no reviewed bank may be derived"
        )
    version = _text(new_catalog_version, "new_catalog_version")
    protocol_id = _text(report.get("protocol_id"), "protocol_id")
    reviewed_source = (
        "project-standardized-neutral-frame-v1; independently-reviewed-under:"
        + protocol_id
    )
    reviewed_bank = ConversationTemplateBank(
        bank_id=f"{source_bank.bank_id}-reviewed-{version}",
        source=reviewed_source,
        templates=tuple(
            replace(template, source=reviewed_source)
            for template in source_bank.templates
        ),
    )
    return reviewed_bank.to_dict()


def load_evidence_directory(path: Path) -> tuple[tuple[Path, dict[str, Any]], ...]:
    if path.is_symlink() or not path.is_dir():
        raise ValueError("evidence_dir must be a regular directory")
    files = tuple(sorted(path.glob("*.json")))
    if not files:
        raise ValueError("evidence_dir contains no JSON response files")
    return tuple((file, _read_json(file)) for file in files)


def write_review_promotion(
    output_dir: Path,
    *,
    kit: Mapping[str, Mapping[str, Any]],
    report: Mapping[str, Any],
    evidence_files: Sequence[tuple[Path, Mapping[str, Any]]],
    reviewed_catalog: Mapping[str, Any] | None,
    reviewed_conversation_bank: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Write a fresh derived report and optional reviewed inputs."""

    if output_dir.is_symlink() or output_dir.exists():
        raise FileExistsError(f"review promotion output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    (output_dir / "review-protocol.json").write_text(
        json.dumps(kit["protocol"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "review-item-map.json").write_text(
        json.dumps(kit["mapping"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    evidence_output = output_dir / "evidence"
    evidence_output.mkdir()
    counts: dict[str, int] = {}
    for _, payload in evidence_files:
        kind = str(payload["evidence_kind"])
        counts[kind] = counts.get(kind, 0) + 1
        destination = evidence_output / f"{kind}-{counts[kind]:02d}.json"
        destination.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    report_payload = dict(report)
    report_payload["reviewed_catalog_written"] = reviewed_catalog is not None
    (output_dir / "evidence-report.json").write_text(
        json.dumps(report_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if (reviewed_catalog is None) != (reviewed_conversation_bank is None):
        raise ValueError(
            "reviewed catalog and conversation bank must be written together"
        )
    if reviewed_catalog is not None and reviewed_conversation_bank is not None:
        catalog_path = output_dir / "reviewed-scenario-catalog.json"
        catalog_path.write_text(
            json.dumps(reviewed_catalog, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        bank_path = output_dir / "reviewed-conversation-templates.json"
        bank_path.write_text(
            json.dumps(reviewed_conversation_bank, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        report_payload["reviewed_catalog_sha256"] = file_sha256(catalog_path)
        report_payload["reviewed_catalog_version"] = reviewed_catalog["catalog_version"]
        report_payload["reviewed_conversation_bank_sha256"] = file_sha256(bank_path)
        report_payload["reviewed_conversation_bank_id"] = reviewed_conversation_bank[
            "bank_id"
        ]
        (output_dir / "evidence-report.json").write_text(
            json.dumps(report_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report_payload


__all__ = [
    "REVIEW_CONTRACT_VERSION",
    "REVIEW_SCHEMA_VERSION",
    "build_scenario_review_kit",
    "derive_reviewed_catalog",
    "derive_reviewed_conversation_bank",
    "load_evidence_directory",
    "render_review_kit_markdown",
    "verify_scenario_review_evidence",
    "write_review_promotion",
]
