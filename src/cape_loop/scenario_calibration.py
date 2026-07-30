"""Prospective scenario calibration and human-review packets.

This module audits frozen scenario inputs before an experiment is run.  It
accepts only catalog, conversation-bank, declared response-model, and planning
inputs; experiment outcomes and evaluated-model outputs are deliberately
outside the API.  The audit is side-effect free and never changes catalog
review fields.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from itertools import combinations
from typing import Any

from .beliefs import THETA_STATES
from .conversation_surfaces import ConversationTemplateBank
from .domains import DATA_SPLITS, get_domain
from .elicitation import build_matched_anchor_set
from .population import susceptibility_grid
from .response import RandomUtilityModel
from .scenarios import (
    ScenarioCatalog,
    ScenarioSpec,
    materialize_matched_anchor_set,
)
from .schemas import InteractionContext, PolicyProvenance, Susceptibility

AUDIT_SCHEMA_VERSION = 2
AUDIT_POLICY = "prospective-scenario-calibration-v2"
REFERENCE_TURNS = 16
BALANCED_PROBABILITY_RANGE = (0.10, 0.90)
RESTRICTED_PROBABILITY_RANGE = (0.20, 0.80)
MEAN_EFFECT_RANGE = (0.02, 0.20)
RAW_LABEL_WORD_COUNT_DIFFERENCE_WARNING = 2
RAW_LABEL_WORD_COUNT_RATIO_RANGE = (0.85, 1.15)
CROSS_SPLIT_LEXICAL_JACCARD_WARNING = 0.60
WITHIN_SPLIT_LEXICAL_JACCARD_WARNING = 0.65
CYCLIC_TARGET_POLICIES = frozenset(
    {
        "balanced",
        "soft_profile_conditioned",
        "exploratory",
        "fixed_bias",
        "hard_filter",
    }
)
MACHINE_RENDERING_MECHANISMS = (
    ("balanced", "balanced"),
    ("restricted", "restriction"),
    ("default", "default"),
    ("suggested", "suggestion"),
    ("balanced", "ranking"),
)
MACHINE_SURFACES_PER_SCENARIO = (
    2  # anchor directions
    * 2  # display orders
    * 2  # selected options
    * len(MACHINE_RENDERING_MECHANISMS)
)

_WORD = re.compile(r"\b[^\W_]+(?:[-'][^\W_]+)*\b", re.UNICODE)
_DOUBLE_PUNCTUATION = re.compile(r"[.!?][,.]")
_ARTICLE_BEFORE_VOWEL = re.compile(r"\ba\s+[AEIOU][A-Za-z-]*\b")
_CAPITALIZED_INFINITIVE = re.compile(r"\bto\s+Choose\b")
_ASSISTANT_AGENCY = re.compile(
    r"\bi\s+(?:found|prepared|drafted|have|can|could|am)\b",
    re.IGNORECASE,
)


def _round(value: float) -> float:
    """Return a stable, human-readable finite float."""

    if not math.isfinite(value):
        raise ValueError("scenario calibration produced a non-finite value")
    return round(value, 12)


def _summary(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        raise ValueError("scenario calibration summary cannot be empty")
    return {
        "count": len(values),
        "minimum": _round(min(values)),
        "maximum": _round(max(values)),
        "mean": _round(math.fsum(values) / len(values)),
    }


def _range_guardrail(
    values: Sequence[float],
    *,
    lower: float,
    upper: float,
) -> dict[str, Any]:
    outside = sum(not lower <= value <= upper for value in values)
    return {
        "lower_inclusive": lower,
        "upper_inclusive": upper,
        "evaluated_count": len(values),
        "outside_count": outside,
        "passed": outside == 0,
    }


def _mean_guardrail(
    values: Sequence[float],
    *,
    lower: float,
    upper: float,
) -> dict[str, Any]:
    mean = math.fsum(values) / len(values)
    return {
        "lower_inclusive": lower,
        "upper_inclusive": upper,
        "observed_mean": _round(mean),
        "passed": lower <= mean <= upper,
    }


def _validated_levels(levels: Sequence[float]) -> tuple[float, ...]:
    if isinstance(levels, (str, bytes)) or not isinstance(levels, Sequence):
        raise TypeError("susceptibility_levels must be a sequence")
    numeric: list[float] = []
    for index, raw in enumerate(levels):
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise TypeError(f"susceptibility_levels[{index}] must be numeric")
        value = float(raw)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("susceptibility_levels must be finite and non-negative")
        numeric.append(value)
    if not numeric:
        raise ValueError("susceptibility_levels cannot be empty")
    if len(numeric) != len(set(numeric)):
        raise ValueError("susceptibility_levels must be distinct")
    return tuple(sorted(numeric))


def _validated_names(
    values: Sequence[str] | None,
    *,
    name: str,
    default: Sequence[str],
) -> tuple[str, ...]:
    if values is None:
        return tuple(sorted(default))
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence")
    prepared = []
    for index, raw in enumerate(values):
        if not isinstance(raw, str) or not raw.strip():
            raise TypeError(f"{name}[{index}] must be a non-empty string")
        prepared.append(raw)
    if not prepared:
        raise ValueError(f"{name} cannot be empty")
    if len(prepared) != len(set(prepared)):
        raise ValueError(f"{name} must be distinct")
    return tuple(sorted(prepared))


def _validated_minimum_probability(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("minimum_matched_probability must be numeric")
    minimum = float(value)
    if not math.isfinite(minimum) or not 0.0 < minimum < 0.5:
        raise ValueError(
            "minimum_matched_probability must lie strictly between 0 and 0.5"
        )
    return minimum


def _tier(criteria: Mapping[str, bool]) -> dict[str, Any]:
    copied = dict(criteria)
    blocking = sorted(name for name, passed in copied.items() if not passed)
    return {
        "ready": not blocking,
        "criteria": copied,
        "blocking_reasons": blocking,
    }


def _word_count(text: str) -> int:
    return len(_WORD.findall(text))


def _surface_tokens(
    scenario: ScenarioSpec,
    bank: ConversationTemplateBank,
) -> frozenset[str]:
    # Compare only scenario-specific lexical material. Shared conversation
    # scaffolding and position-assigned A–D names are intentionally reusable
    # across splits and would otherwise turn a grammar standardization into a
    # false near-duplicate warning.
    bank.template(scenario.scenario_id)
    material = " ".join(
        (
            scenario.prompt,
            *(option.label for option in scenario.options),
        )
    )
    return frozenset(token.casefold() for token in _WORD.findall(material))


def _lexical_overlap_warnings(
    catalog: ScenarioCatalog,
    bank: ConversationTemplateBank,
    *,
    split: str,
    domains: Sequence[str],
) -> list[dict[str, Any]]:
    domain_scope = frozenset(domains)
    token_sets = {
        scenario.scenario_id: _surface_tokens(scenario, bank)
        for scenario in catalog.scenarios
        if scenario.domain in domain_scope
    }
    warnings: list[dict[str, Any]] = []
    ordered = tuple(
        sorted(
            (
                scenario
                for scenario in catalog.scenarios
                if scenario.domain in domain_scope
            ),
            key=lambda item: item.scenario_id,
        )
    )
    for first, second in combinations(ordered, 2):
        within_selected_split = first.split == second.split == split
        cross_split = first.split != second.split
        if not within_selected_split and not cross_split:
            continue
        first_tokens = token_sets[first.scenario_id]
        second_tokens = token_sets[second.scenario_id]
        union = first_tokens | second_tokens
        score = 0.0 if not union else len(first_tokens & second_tokens) / len(union)
        threshold = (
            WITHIN_SPLIT_LEXICAL_JACCARD_WARNING
            if within_selected_split
            else CROSS_SPLIT_LEXICAL_JACCARD_WARNING
        )
        if score < threshold:
            continue
        warnings.append(
            {
                "kind": (
                    "within_split_lexical_redundancy_candidate"
                    if within_selected_split
                    else "cross_split_lexical_overlap_candidate"
                ),
                "comparison_method": (
                    "scenario_prompt_and_option_label_unigram_set_jaccard"
                ),
                "semantic_similarity_claimed": False,
                "scenario_a": first.scenario_id,
                "split_a": first.split,
                "scenario_b": second.scenario_id,
                "split_b": second.split,
                "token_jaccard": _round(score),
                "warning_threshold_inclusive": threshold,
                "blocks_machine_readiness": False,
                "blocks_recorded_scientific_readiness": True,
            }
        )
    return sorted(
        warnings,
        key=lambda item: (
            -float(item["token_jaccard"]),
            str(item["scenario_a"]),
            str(item["scenario_b"]),
        ),
    )


def _cross_split_task_family_reuse_warnings(
    catalog: ScenarioCatalog,
    *,
    domains: Sequence[str],
) -> list[dict[str, Any]]:
    domain_scope = frozenset(domains)
    grouped: dict[tuple[str, str], list[ScenarioSpec]] = {}
    for scenario in catalog.scenarios:
        if scenario.domain not in domain_scope:
            continue
        grouped.setdefault(
            (scenario.domain, scenario.task_family),
            [],
        ).append(scenario)
    warnings = []
    for (domain, task_family), scenarios in sorted(grouped.items()):
        splits = sorted({scenario.split for scenario in scenarios})
        if len(splits) <= 1:
            continue
        warnings.append(
            {
                "kind": "cross_split_exact_task_family_reuse_review_flag",
                "domain": domain,
                "task_family": task_family,
                "splits": splits,
                "scenario_ids": sorted(
                    scenario.scenario_id for scenario in scenarios
                ),
                "semantic_similarity_claimed": False,
                "blocks_machine_readiness": False,
                "blocks_recorded_scientific_readiness": True,
            }
        )
    return warnings


def _raw_label_word_count_warnings(
    scenario: ScenarioSpec,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    roles = {
        "negative": scenario.negative_option,
        "positive": scenario.positive_option,
        "negative_peer": scenario.negative_same_direction_option,
        "positive_peer": scenario.positive_same_direction_option,
    }
    counts = {role: _word_count(option.label) for role, option in roles.items()}
    displayed_pairs = (
        ("balanced", "negative", "positive"),
        ("restricted_negative", "negative", "negative_peer"),
        ("restricted_positive", "positive", "positive_peer"),
    )
    warnings = []
    for presentation, first, second in displayed_pairs:
        difference = abs(counts[first] - counts[second])
        ratio = (
            counts[first] / counts[second]
            if counts[second] > 0
            else None
        )
        common = {
            "scenario_id": scenario.scenario_id,
            "presentation": presentation,
            "first_role": first,
            "first_word_count": counts[first],
            "second_role": second,
            "second_word_count": counts[second],
            "blocks_machine_readiness": False,
            "blocks_recorded_scientific_readiness": True,
        }
        if difference > RAW_LABEL_WORD_COUNT_DIFFERENCE_WARNING:
            warnings.append(
                {
                    "kind": "option_label_raw_word_count_difference",
                    **common,
                    "absolute_difference": difference,
                    "warning_threshold_exclusive": (
                        RAW_LABEL_WORD_COUNT_DIFFERENCE_WARNING
                    ),
                }
            )
        if not (
            ratio is not None
            and RAW_LABEL_WORD_COUNT_RATIO_RANGE[0]
            <= ratio
            <= RAW_LABEL_WORD_COUNT_RATIO_RANGE[1]
        ):
            warnings.append(
                {
                    "kind": "option_label_raw_word_count_ratio_outside_range",
                    **common,
                    "first_to_second_ratio": (
                        None if ratio is None else _round(ratio)
                    ),
                    "warning_range_inclusive": {
                        "lower": RAW_LABEL_WORD_COUNT_RATIO_RANGE[0],
                        "upper": RAW_LABEL_WORD_COUNT_RATIO_RANGE[1],
                    },
                }
            )
    return warnings, counts


def _ordered(
    context: InteractionContext,
    anchor_option_id: str,
    *,
    anchor_first: bool,
    suffix: str,
) -> InteractionContext:
    other = next(
        option_id for option_id in context.option_ids if option_id != anchor_option_id
    )
    ranking = (anchor_option_id, other) if anchor_first else (other, anchor_option_id)
    return replace(
        context,
        context_id=f"{context.context_id}:calibration:{suffix}",
        ranking=ranking,
    )


def _anchor_probability(
    model: RandomUtilityModel,
    theta: tuple[int, int, int],
    susceptibility: Susceptibility,
    context: InteractionContext,
    anchor_option_id: str,
) -> float:
    return model.probability_map(
        theta,
        susceptibility,
        context,
    )[anchor_option_id]


def _empty_metrics() -> dict[str, list[float]]:
    return {
        "balanced_order_averaged_probability": [],
        "restricted_order_averaged_probability": [],
        "ranking_increment": [],
        "default_increment": [],
        "suggestion_increment": [],
    }


def _empty_physical_probabilities() -> dict[str, list[float]]:
    return {
        "balanced": [],
        "restricted": [],
        "default": [],
        "suggestion": [],
    }


def _metric_report(metrics: Mapping[str, Sequence[float]]) -> dict[str, Any]:
    return {name: _summary(values) for name, values in metrics.items()}


def _binary_response_probability_guardrail(
    values: Sequence[float],
    *,
    minimum: float,
) -> dict[str, Any]:
    complementary_ceiling = 1.0 - minimum
    at_or_below = sum(value <= minimum for value in values)
    at_or_above = sum(value >= complementary_ceiling for value in values)
    return {
        "minimum_matched_probability_exclusive": minimum,
        "complementary_maximum_probability_exclusive": (
            complementary_ceiling
        ),
        "evaluated_physical_probability_count": len(values),
        "anchor_at_or_below_minimum_count": at_or_below,
        "anchor_at_or_above_complementary_ceiling_count": at_or_above,
        "either_binary_response_at_or_below_minimum_count": (
            at_or_below + at_or_above
        ),
        "passed": at_or_below == 0 and at_or_above == 0,
    }


def _metric_guardrails(
    metrics: Mapping[str, Sequence[float]],
    physical_probabilities: Mapping[str, Sequence[float]],
    *,
    minimum_matched_probability: float,
) -> dict[str, Any]:
    balanced = _range_guardrail(
        metrics["balanced_order_averaged_probability"],
        lower=BALANCED_PROBABILITY_RANGE[0],
        upper=BALANCED_PROBABILITY_RANGE[1],
    )
    restricted = _range_guardrail(
        metrics["restricted_order_averaged_probability"],
        lower=RESTRICTED_PROBABILITY_RANGE[0],
        upper=RESTRICTED_PROBABILITY_RANGE[1],
    )
    physical = {
        mechanism: _binary_response_probability_guardrail(
            values,
            minimum=minimum_matched_probability,
        )
        for mechanism, values in physical_probabilities.items()
    }
    effects = {
        mechanism: _mean_guardrail(
            metrics[f"{mechanism}_increment"],
            lower=MEAN_EFFECT_RANGE[0],
            upper=MEAN_EFFECT_RANGE[1],
        )
        for mechanism in ("ranking", "default", "suggestion")
    }
    return {
        "order_averaged_balanced_probability": balanced,
        "order_averaged_restricted_probability": restricted,
        "physical_mechanism_probabilities": physical,
        "mean_incremental_effects": effects,
        "passed": (
            balanced["passed"]
            and restricted["passed"]
            and all(item["passed"] for item in physical.values())
            and all(item["passed"] for item in effects.values())
        ),
    }


def _numeric_signature(
    matched: Any,
) -> tuple[Any, ...]:
    """Identify unique numeric designs without scenario wording or IDs."""

    anchor = matched.anchor_option_id
    contexts = []
    for mechanism in ("balanced", "restricted", "default", "suggested"):
        context = matched.context(mechanism)
        contexts.append(
            (
                mechanism,
                tuple(
                    (
                        option.option_id == anchor,
                        tuple(float(value) for value in option.features),
                    )
                    for option in context.options
                ),
                tuple(option_id == anchor for option_id in context.ranking),
                context.default_option_id == anchor,
                context.suggested_option_id == anchor,
            )
        )
    return (
        matched.target_attribute,
        matched.anchor_direction,
        tuple(contexts),
    )


def _probability_calibration(
    scenarios: Sequence[ScenarioSpec],
    model: RandomUtilityModel,
    susceptibilities: Sequence[Susceptibility],
    *,
    minimum_matched_probability: float,
) -> dict[str, Any]:
    global_metrics = _empty_metrics()
    global_physical = _empty_physical_probabilities()
    cell_metrics: dict[tuple[str, int], dict[str, list[float]]] = {}
    cell_physical: dict[tuple[str, int], dict[str, list[float]]] = {}
    global_signatures: set[tuple[Any, ...]] = set()
    cell_signatures: dict[tuple[str, int], set[tuple[Any, ...]]] = {}
    cell_anchor_instances: dict[tuple[str, int], int] = {}
    for scenario in scenarios:
        cell = (scenario.domain, scenario.target_attribute)
        metrics = cell_metrics.setdefault(cell, _empty_metrics())
        physical = cell_physical.setdefault(cell, _empty_physical_probabilities())
        signatures = cell_signatures.setdefault(cell, set())
        for anchor_direction in (-1, 1):
            generic = build_matched_anchor_set(
                get_domain(scenario.domain),
                target_attribute=scenario.target_attribute,
                anchor_direction=anchor_direction,
                scenario_id=f"calibration:{scenario.scenario_id}",
            )
            matched = materialize_matched_anchor_set(generic, scenario)
            signature = _numeric_signature(matched)
            global_signatures.add(signature)
            signatures.add(signature)
            cell_anchor_instances[cell] = cell_anchor_instances.get(cell, 0) + 1
            anchor = matched.anchor_option_id
            balanced = matched.context("balanced")
            restricted = matched.context("restricted")
            default = matched.context("default")
            suggested = matched.context("suggested")
            contexts = {
                "balanced_first": _ordered(
                    balanced,
                    anchor,
                    anchor_first=True,
                    suffix="balanced-first",
                ),
                "balanced_second": _ordered(
                    balanced,
                    anchor,
                    anchor_first=False,
                    suffix="balanced-second",
                ),
                "restricted_first": _ordered(
                    restricted,
                    anchor,
                    anchor_first=True,
                    suffix="restricted-first",
                ),
                "restricted_second": _ordered(
                    restricted,
                    anchor,
                    anchor_first=False,
                    suffix="restricted-second",
                ),
                "default_first": _ordered(
                    default,
                    anchor,
                    anchor_first=True,
                    suffix="default-first",
                ),
                "default_second": _ordered(
                    default,
                    anchor,
                    anchor_first=False,
                    suffix="default-second",
                ),
                "suggested_first": _ordered(
                    suggested,
                    anchor,
                    anchor_first=True,
                    suffix="suggested-first",
                ),
                "suggested_second": _ordered(
                    suggested,
                    anchor,
                    anchor_first=False,
                    suffix="suggested-second",
                ),
            }
            for theta in THETA_STATES:
                for susceptibility in susceptibilities:
                    probabilities = {
                        name: _anchor_probability(
                            model,
                            theta,
                            susceptibility,
                            context,
                            anchor,
                        )
                        for name, context in contexts.items()
                    }
                    values = {
                        "balanced_order_averaged_probability": (
                            probabilities["balanced_first"]
                            + probabilities["balanced_second"]
                        )
                        / 2.0,
                        "restricted_order_averaged_probability": (
                            probabilities["restricted_first"]
                            + probabilities["restricted_second"]
                        )
                        / 2.0,
                        "ranking_increment": (
                            probabilities["balanced_first"]
                            - probabilities["balanced_second"]
                        ),
                        "default_increment": (
                            (
                                probabilities["default_first"]
                                - probabilities["balanced_first"]
                            )
                            + (
                                probabilities["default_second"]
                                - probabilities["balanced_second"]
                            )
                        )
                        / 2.0,
                        "suggestion_increment": (
                            (
                                probabilities["suggested_first"]
                                - probabilities["balanced_first"]
                            )
                            + (
                                probabilities["suggested_second"]
                                - probabilities["balanced_second"]
                            )
                        )
                        / 2.0,
                    }
                    for name, value in values.items():
                        metrics[name].append(value)
                        global_metrics[name].append(value)
                    physical_values = {
                        "balanced": (
                            probabilities["balanced_first"],
                            probabilities["balanced_second"],
                        ),
                        "restricted": (
                            probabilities["restricted_first"],
                            probabilities["restricted_second"],
                        ),
                        "default": (
                            probabilities["default_first"],
                            probabilities["default_second"],
                        ),
                        "suggestion": (
                            probabilities["suggested_first"],
                            probabilities["suggested_second"],
                        ),
                    }
                    for mechanism, mechanism_values in physical_values.items():
                        physical[mechanism].extend(mechanism_values)
                        global_physical[mechanism].extend(mechanism_values)

    cells = []
    for (domain, target), metrics in sorted(cell_metrics.items()):
        physical = cell_physical[(domain, target)]
        guardrails = _metric_guardrails(
            metrics,
            physical,
            minimum_matched_probability=minimum_matched_probability,
        )
        anchor_instances = cell_anchor_instances[(domain, target)]
        unique_signatures = len(cell_signatures[(domain, target)])
        cells.append(
            {
                "domain": domain,
                "target_attribute": target,
                "target_key": get_domain(domain).attributes[target].key,
                "summary": _metric_report(metrics),
                "physical_probability_summary": _metric_report(physical),
                "guardrails": guardrails,
                "scenario_anchor_instance_count": anchor_instances,
                "unique_numeric_signature_count": unique_signatures,
                "numeric_signature_repetition_factor": _round(
                    anchor_instances / unique_signatures
                ),
            }
        )
    global_guardrails = _metric_guardrails(
        global_metrics,
        global_physical,
        minimum_matched_probability=minimum_matched_probability,
    )
    anchor_instances = len(scenarios) * 2
    unique_signatures = len(global_signatures)
    return {
        "scope": (
            "all THETA_STATES x declared susceptibility grid x both anchor "
            "directions x both counterbalanced display orders"
        ),
        "support_weighting": "uniform_cartesian_design_grid",
        "theta_state_count": len(THETA_STATES),
        "susceptibility_profile_count": len(susceptibilities),
        "scenario_anchor_instance_count": anchor_instances,
        "unique_numeric_signature_count": unique_signatures,
        "numeric_signature_repetition_factor": _round(
            anchor_instances / unique_signatures
        ),
        "scenario_anchor_state_count": len(
            global_metrics["balanced_order_averaged_probability"]
        ),
        "physical_probability_evaluation_count": (
            sum(len(values) for values in global_physical.values())
        ),
        "counterbalanced_order_count": 2,
        "summary": _metric_report(global_metrics),
        "physical_probability_summary": _metric_report(global_physical),
        "guardrails": global_guardrails,
        "cells": cells,
        "all_cells_passed": (
            global_guardrails["passed"]
            and all(cell["guardrails"]["passed"] for cell in cells)
        ),
    }


def _review_counts(scenarios: Sequence[ScenarioSpec]) -> dict[str, Any]:
    rows = tuple(scenarios)

    def counts(field: str) -> dict[str, int]:
        values: dict[str, int] = {}
        for scenario in rows:
            value = str(scenario.review[field])
            values[value] = values.get(value, 0) + 1
        return dict(sorted(values.items()))

    return {
        "scenario_count": len(rows),
        "status": {
            status: sum(scenario.status == status for scenario in rows)
            for status in ("provisional", "approved")
        },
        "automated_validation": counts("automated_validation"),
        "surface_human_review": counts("surface_human_review"),
        "scientific_human_review": counts("scientific_human_review"),
        "paper_eligible_count": sum(
            bool(scenario.review["paper_eligible"]) for scenario in rows
        ),
        "all_automated_validation_passed": all(
            scenario.review["automated_validation"] == "passed" for scenario in rows
        ),
        "all_surface_human_review_passed": all(
            scenario.review["surface_human_review"] == "passed" for scenario in rows
        ),
        "all_scientific_human_review_passed": all(
            scenario.review["scientific_human_review"] == "passed" for scenario in rows
        ),
        "all_approved": all(scenario.status == "approved" for scenario in rows),
        "all_paper_eligible": all(
            bool(scenario.review["paper_eligible"]) for scenario in rows
        ),
    }


def _nuisance_design_report(
    scenarios: Sequence[ScenarioSpec],
    *,
    domains: Sequence[str],
) -> dict[str, Any]:
    """Report prospective restricted-peer orthogonality by target cell."""

    cells = []
    for domain in domains:
        for target in range(3):
            rows = tuple(
                scenario
                for scenario in scenarios
                if scenario.domain == domain
                and scenario.target_attribute == target
            )
            attribute_counts = {
                get_domain(domain).attributes[index].key: sum(
                    scenario.nuisance_attribute == index for scenario in rows
                )
                for index in range(3)
                if index != target
            }
            direction_counts = {
                "-1": sum(scenario.nuisance_direction == -1 for scenario in rows),
                "+1": sum(scenario.nuisance_direction == 1 for scenario in rows),
            }
            attribute_values = tuple(attribute_counts.values())
            direction_values = tuple(direction_counts.values())
            attributes_balanced = bool(attribute_values) and (
                max(attribute_values) - min(attribute_values) <= 1
            )
            directions_balanced = bool(direction_values) and (
                max(direction_values) - min(direction_values) <= 1
            )
            if len(rows) >= 2:
                attributes_balanced = attributes_balanced and all(
                    value > 0 for value in attribute_values
                )
                directions_balanced = directions_balanced and all(
                    value > 0 for value in direction_values
                )
            joint_counts = {
                f"{get_domain(domain).attributes[index].key}:{direction:+d}": sum(
                    scenario.nuisance_attribute == index
                    and scenario.nuisance_direction == direction
                    for scenario in rows
                )
                for index in range(3)
                if index != target
                for direction in (-1, 1)
            }
            joint_values = tuple(joint_counts.values())
            joint_balanced = bool(joint_values) and (
                max(joint_values) - min(joint_values) <= 1
            )
            if len(rows) >= 4:
                joint_balanced = joint_balanced and all(
                    value > 0 for value in joint_values
                )
            cells.append(
                {
                    "domain": domain,
                    "target_attribute": target,
                    "target_key": get_domain(domain).attributes[target].key,
                    "scenario_count": len(rows),
                    "attribute_counts": attribute_counts,
                    "direction_counts": direction_counts,
                    "joint_counts": joint_counts,
                    "nuisance_attributes_balanced_within_one": (
                        attributes_balanced
                    ),
                    "nuisance_directions_balanced_within_one": (
                        directions_balanced
                    ),
                    "nuisance_joint_combinations_balanced_within_one": (
                        joint_balanced
                    ),
                    "passed": (
                        attributes_balanced
                        and directions_balanced
                        and joint_balanced
                    ),
                }
            )
    return {
        "design": (
            "both non-target attributes and both peer-minus-anchor "
            "directions, including their joint combinations, balanced "
            "within one per domain-target cell"
        ),
        "magnitude": 0.25,
        "cells": cells,
        "all_cells_passed": all(cell["passed"] for cell in cells),
    }


def _conversation_frame_design_report(
    scenarios: Sequence[ScenarioSpec],
    bank: ConversationTemplateBank,
    *,
    domains: Sequence[str],
) -> dict[str, Any]:
    """Report outcome-blind balance of neutral conversation base families."""

    bases = {
        scenario.scenario_id: bank.template(
            scenario.scenario_id
        ).presentation_templates["balanced"]
        for scenario in scenarios
    }
    unique_bases = tuple(sorted(set(bases.values())))
    frame_ids = {
        base: f"frame_{index:02d}"
        for index, base in enumerate(unique_bases, start=1)
    }
    source_counts = Counter(
        bank.template(scenario.scenario_id).source
        for scenario in scenarios
    )
    agency_candidates = sorted(
        scenario_id
        for scenario_id, base in bases.items()
        if _ASSISTANT_AGENCY.search(base)
    )
    cells = []
    for domain in domains:
        for target in range(3):
            rows = tuple(
                scenario
                for scenario in scenarios
                if scenario.domain == domain
                and scenario.target_attribute == target
            )
            counts = {
                frame_ids[base]: sum(
                    bases[scenario.scenario_id] == base
                    for scenario in rows
                )
                for base in unique_bases
            }
            values = tuple(counts.values())
            balanced = bool(values) and max(values) - min(values) <= 1
            if len(rows) >= len(unique_bases):
                balanced = balanced and all(value > 0 for value in values)
            cells.append(
                {
                    "domain": domain,
                    "target_attribute": target,
                    "target_key": get_domain(domain).attributes[target].key,
                    "scenario_count": len(rows),
                    "frame_counts": counts,
                    "balanced_within_one": balanced,
                }
            )
    return {
        "design": (
            "source-neutral base families counterbalanced within one per "
            "domain-target cell"
        ),
        "frame_family_count": len(unique_bases),
        "frame_counts": dict(
            sorted(
                Counter(frame_ids[base] for base in bases.values()).items()
            )
        ),
        "source_counts": dict(sorted(source_counts.items())),
        "assistant_agency_heuristic": (
            "no first-person found/prepared/drafted/have/can/could/am framing"
        ),
        "assistant_agency_candidates": agency_candidates,
        "source_neutral_heuristic_passed": not agency_candidates,
        "cells": cells,
        "all_cells_balanced_within_one": all(
            cell["balanced_within_one"] for cell in cells
        ),
        "passed": (
            len(unique_bases) >= 2
            and not agency_candidates
            and all(cell["balanced_within_one"] for cell in cells)
        ),
    }


def _capacity_report(
    scenarios: Sequence[ScenarioSpec],
    *,
    planned_turns: int,
    domains: Sequence[str],
    policies: Sequence[str],
) -> dict[str, Any]:
    non_cyclic_policies = tuple(
        policy for policy in policies if policy not in CYCLIC_TARGET_POLICIES
    )
    adaptive_worst_case = bool(non_cyclic_policies)
    planned_required = (
        planned_turns if adaptive_worst_case else math.ceil(planned_turns / 3)
    )
    cyclic_reference_required = math.ceil(REFERENCE_TURNS / 3)
    cells = []
    for domain in domains:
        for target in range(3):
            rows = tuple(
                scenario
                for scenario in scenarios
                if scenario.domain == domain and scenario.target_attribute == target
            )
            available = len(rows)
            cells.append(
                {
                    "domain": domain,
                    "target_attribute": target,
                    "target_key": get_domain(domain).attributes[target].key,
                    "available_scenarios": available,
                    "scenario_ids": sorted(scenario.scenario_id for scenario in rows),
                    "planned_no_repeat_required": planned_required,
                    "planned_no_repeat_sufficient": (available >= planned_required),
                    "cyclic_reference_16_turn_no_repeat_required": (
                        cyclic_reference_required
                    ),
                    "cyclic_reference_16_turn_no_repeat_sufficient": (
                        available >= cyclic_reference_required
                    ),
                }
            )
    return {
        "target_cycle_length": 3,
        "configured_domains": list(domains),
        "configured_policies": list(policies),
        "non_cyclic_or_unknown_target_policies": list(non_cyclic_policies),
        "planned_turns": planned_turns,
        "planned_capacity_basis": (
            "adaptive_or_unknown_policy_worst_case_all_turns_in_one_cell"
            if adaptive_worst_case
            else "declared_balanced_coverage_at_most_ceil_turns_over_3"
        ),
        "planned_per_cell_no_repeat_required": planned_required,
        "planned_all_cells_sufficient": all(
            cell["planned_no_repeat_sufficient"] for cell in cells
        ),
        "cyclic_reference_turns": REFERENCE_TURNS,
        "cyclic_reference_capacity_basis": (
            "turn_modulo_3_reference_only_not_adaptive_policy_capacity"
        ),
        "cyclic_reference_per_cell_no_repeat_required": (
            cyclic_reference_required
        ),
        "cyclic_reference_all_cells_sufficient": all(
            cell["cyclic_reference_16_turn_no_repeat_sufficient"] for cell in cells
        ),
        "cells": cells,
    }


def _provenance(mechanism: str) -> PolicyProvenance:
    return PolicyProvenance(
        policy_id=f"scenario-calibration-{mechanism}",
        policy_version=AUDIT_POLICY,
        presentation_mechanism=mechanism,
        profile_conditioned=False,
    )


def _rendered_examples(
    scenario: ScenarioSpec,
    bank: ConversationTemplateBank,
) -> list[dict[str, Any]]:
    """Render a concise, labeled preview for independent human review."""

    domain = get_domain(scenario.domain)
    negative = materialize_matched_anchor_set(
        build_matched_anchor_set(
            domain,
            target_attribute=scenario.target_attribute,
            anchor_direction=-1,
            scenario_id=f"review:{scenario.scenario_id}:negative",
        ),
        scenario,
    )
    positive = materialize_matched_anchor_set(
        build_matched_anchor_set(
            domain,
            target_attribute=scenario.target_attribute,
            anchor_direction=1,
            scenario_id=f"review:{scenario.scenario_id}:positive",
        ),
        scenario,
    )
    negative_anchor = negative.anchor_option_id
    positive_anchor = positive.anchor_option_id
    reversed_balanced = _ordered(
        negative.context("balanced"),
        negative_anchor,
        anchor_first=False,
        suffix="human-review-ranking",
    )
    specs = (
        (
            "balanced",
            negative.context("balanced"),
            "balanced",
            negative_anchor,
        ),
        (
            "restricted_negative",
            negative.context("restricted"),
            "restriction",
            negative_anchor,
        ),
        (
            "restricted_positive",
            positive.context("restricted"),
            "restriction",
            positive_anchor,
        ),
        (
            "default",
            negative.context("default"),
            "default",
            positive_anchor,
        ),
        (
            "suggestion",
            negative.context("suggested"),
            "suggestion",
            positive_anchor,
        ),
        (
            "ranking_reversed",
            reversed_balanced,
            "ranking",
            negative_anchor,
        ),
    )
    examples = []
    for label, context, mechanism, selected in specs:
        rendered = bank.render(
            context,
            _provenance(mechanism),
            selected,
        )
        examples.append(
            {
                "review_surface": label,
                "presentation_mechanism": mechanism,
                "ranking": list(context.ranking),
                "default_option_id": context.default_option_id,
                "suggested_option_id": context.suggested_option_id,
                "selected_option_id": selected,
                "surface_id": rendered.surface_id,
                "assistant_message": rendered.assistant_message,
                "user_message": rendered.user_message,
            }
        )
    return examples


def _machine_rendered_surfaces(
    scenario: ScenarioSpec,
    bank: ConversationTemplateBank,
) -> list[dict[str, Any]]:
    """Render every machine-relevant case in the declared binary design."""

    domain = get_domain(scenario.domain)
    surfaces = []
    for anchor_direction in (-1, 1):
        matched = materialize_matched_anchor_set(
            build_matched_anchor_set(
                domain,
                target_attribute=scenario.target_attribute,
                anchor_direction=anchor_direction,
                scenario_id=(
                    f"machine-audit:{scenario.scenario_id}:"
                    f"{anchor_direction:+d}"
                ),
            ),
            scenario,
        )
        anchor = matched.anchor_option_id
        for context_mechanism, presentation_mechanism in (
            MACHINE_RENDERING_MECHANISMS
        ):
            base = matched.context(context_mechanism)
            for anchor_first in (True, False):
                order_label = "anchor_first" if anchor_first else "anchor_second"
                context = _ordered(
                    base,
                    anchor,
                    anchor_first=anchor_first,
                    suffix=f"{presentation_mechanism}-{order_label}",
                )
                for selected in context.option_ids:
                    rendered = bank.render(
                        context,
                        _provenance(presentation_mechanism),
                        selected,
                    )
                    surfaces.append(
                        {
                            "anchor_direction": anchor_direction,
                            "context_mechanism": context_mechanism,
                            "presentation_mechanism": presentation_mechanism,
                            "display_order": order_label,
                            "selected_role": (
                                "anchor" if selected == anchor else "comparison"
                            ),
                            "selected_option_id": selected,
                            "ranking": list(context.ranking),
                            "surface_id": rendered.surface_id,
                            "assistant_message": rendered.assistant_message,
                            "user_message": rendered.user_message,
                        }
                    )
    return surfaces


def _render_hygiene_issues(
    surface: Mapping[str, Any],
) -> list[tuple[str, str]]:
    issues = []
    patterns = (
        (
            "adjacent_sentence_and_clause_punctuation",
            _DOUBLE_PUNCTUATION,
        ),
        ("article_before_vowel_sound", _ARTICLE_BEFORE_VOWEL),
        ("capitalized_prompt_after_infinitive", _CAPITALIZED_INFINITIVE),
    )
    for message_role, field in (
        ("assistant", "assistant_message"),
        ("user", "user_message"),
    ):
        text = str(surface[field])
        for issue, pattern in patterns:
            if pattern.search(text):
                issues.append((message_role, issue))
    return issues


def _scenario_review_rows(
    scenarios: Sequence[ScenarioSpec],
    bank: ConversationTemplateBank,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    rows = []
    all_warnings: list[dict[str, Any]] = []
    total_machine_surfaces = 0
    role_options = (
        ("negative", "negative_option"),
        ("positive", "positive_option"),
        ("negative_peer", "negative_same_direction_option"),
        ("positive_peer", "positive_same_direction_option"),
    )
    for scenario in scenarios:
        warnings, word_counts = _raw_label_word_count_warnings(scenario)
        template = bank.template(scenario.scenario_id)
        rendered_examples = _rendered_examples(scenario, bank)
        machine_surfaces = _machine_rendered_surfaces(scenario, bank)
        total_machine_surfaces += len(machine_surfaces)
        for surface in machine_surfaces:
            for message_role, issue in _render_hygiene_issues(surface):
                warnings.append(
                    {
                        "kind": "rendered_surface_hygiene",
                        "scenario_id": scenario.scenario_id,
                        "anchor_direction": surface["anchor_direction"],
                        "context_mechanism": surface["context_mechanism"],
                        "presentation_mechanism": surface[
                            "presentation_mechanism"
                        ],
                        "display_order": surface["display_order"],
                        "selected_role": surface["selected_role"],
                        "message_role": message_role,
                        "issue": issue,
                        "blocks_machine_readiness": False,
                        "blocks_recorded_scientific_readiness": True,
                    }
                )
        all_warnings.extend(warnings)
        options = []
        for role, attribute in role_options:
            option = getattr(scenario, attribute)
            options.append(
                {
                    "role": role,
                    "option_id": option.option_id,
                    "stored_name_pool_entry": (
                        template.display_names[option.option_id]
                    ),
                    "label": option.label,
                    "label_word_count": word_counts[role],
                    "features": list(option.features),
                }
            )
        rows.append(
            {
                "scenario_id": scenario.scenario_id,
                "family_id": scenario.family_id,
                "domain": scenario.domain,
                "split": scenario.split,
                "task_family": scenario.task_family,
                "target_attribute": scenario.target_attribute,
                "target_key": scenario.target_key,
                "nuisance_attribute": scenario.nuisance_attribute,
                "nuisance_key": scenario.nuisance_key,
                "nuisance_direction": scenario.nuisance_direction,
                "prompt": scenario.prompt,
                "status": scenario.status,
                "quality_assertions_are_author_declarations": dict(
                    sorted(scenario.quality_assertions.items())
                ),
                "review": dict(sorted(scenario.review.items())),
                "options": options,
                "warnings": warnings,
                "rendered_preview_label": (
                    "concise_human_review_preview_not_exhaustive"
                ),
                "rendered_examples": rendered_examples,
                "machine_surface_count": len(machine_surfaces),
                "expected_machine_surface_count": (
                    MACHINE_SURFACES_PER_SCENARIO
                ),
            }
        )
    expected_total = len(scenarios) * MACHINE_SURFACES_PER_SCENARIO
    rendering = {
        "enumeration": (
            "anchor_direction_x_display_order_x_mechanism_x_selected_option"
        ),
        "anchor_direction_count": 2,
        "display_order_count": 2,
        "mechanism_count": len(MACHINE_RENDERING_MECHANISMS),
        "selected_option_count_per_context": 2,
        "expected_surface_count_per_scenario": MACHINE_SURFACES_PER_SCENARIO,
        "human_preview_surface_count_per_scenario": 6,
        "expected_total_surface_count": expected_total,
        "rendered_total_surface_count": total_machine_surfaces,
        "complete": total_machine_surfaces == expected_total
        and all(
            row["machine_surface_count"] == MACHINE_SURFACES_PER_SCENARIO
            for row in rows
        ),
        "used_for_hygiene_and_readiness": True,
        "stored_in_human_packet": False,
    }
    return rows, all_warnings, rendering


def build_scenario_calibration_audit(
    catalog: ScenarioCatalog,
    conversation_bank: ConversationTemplateBank,
    response_model: RandomUtilityModel,
    *,
    susceptibility_levels: Sequence[float],
    split: str,
    planned_turns: int,
    domains: Sequence[str] | None = None,
    policies: Sequence[str] | None = None,
    minimum_matched_probability: float = 0.05,
) -> dict[str, Any]:
    """Build one deterministic, JSON-ready prospective scenario audit.

    The function consumes no experimental outcomes or evaluated-model outputs,
    performs no I/O, and never mutates ``catalog`` or ``conversation_bank``.
    Invalid structural inputs raise. Machine warnings do not block structural
    engineering use, but they remain unresolved and block recorded scientific
    readiness because this audit has no adjudication-record contract.
    """

    if not isinstance(catalog, ScenarioCatalog):
        raise TypeError("catalog must be a parsed ScenarioCatalog")
    if not isinstance(conversation_bank, ConversationTemplateBank):
        raise TypeError("conversation_bank must be a parsed ConversationTemplateBank")
    if not isinstance(response_model, RandomUtilityModel):
        raise TypeError("response_model must be a RandomUtilityModel")
    if split not in DATA_SPLITS:
        raise ValueError(f"split must be one of {DATA_SPLITS}")
    if (
        isinstance(planned_turns, bool)
        or not isinstance(planned_turns, int)
        or planned_turns <= 0
    ):
        raise ValueError("planned_turns must be a positive integer")
    levels = _validated_levels(susceptibility_levels)
    domain_scope = _validated_names(
        domains,
        name="domains",
        default=("travel", "writing"),
    )
    for domain in domain_scope:
        try:
            get_domain(domain)
        except KeyError as exc:
            raise ValueError(f"unknown domain: {domain!r}") from exc
    policy_scope = _validated_names(
        policies,
        name="policies",
        default=("balanced",),
    )
    minimum_probability = _validated_minimum_probability(
        minimum_matched_probability
    )
    susceptibilities = susceptibility_grid(levels)

    # Exact bank coverage is a structural prerequisite, not a review judgment.
    conversation_bank.validate_catalog(catalog)
    scenarios = tuple(
        sorted(
            (
                scenario
                for scenario in catalog.scenarios
                if scenario.split == split and scenario.domain in domain_scope
            ),
            key=lambda item: item.scenario_id,
        )
    )
    if not scenarios:
        raise ValueError(f"catalog has no scenarios in split {split!r}")

    capacity = _capacity_report(
        scenarios,
        planned_turns=planned_turns,
        domains=domain_scope,
        policies=policy_scope,
    )
    nuisance_design = _nuisance_design_report(
        scenarios,
        domains=domain_scope,
    )
    conversation_frame_design = _conversation_frame_design_report(
        scenarios,
        conversation_bank,
        domains=domain_scope,
    )
    probability = _probability_calibration(
        scenarios,
        response_model,
        susceptibilities,
        minimum_matched_probability=minimum_probability,
    )
    selected_reviews = _review_counts(scenarios)
    catalog_reviews = _review_counts(catalog.scenarios)
    review_rows, scenario_warnings, machine_rendering = _scenario_review_rows(
        scenarios,
        conversation_bank,
    )
    lexical_overlaps = _lexical_overlap_warnings(
        catalog,
        conversation_bank,
        split=split,
        domains=domain_scope,
    )
    task_family_reuse = _cross_split_task_family_reuse_warnings(
        catalog,
        domains=domain_scope,
    )
    warnings = sorted(
        (*scenario_warnings, *lexical_overlaps, *task_family_reuse),
        key=lambda item: (
            str(item["kind"]),
            str(item.get("scenario_id", item.get("scenario_a", ""))),
            str(item.get("scenario_b", "")),
            str(item.get("presentation", "")),
        ),
    )
    rendered_hygiene_warnings = [
        warning
        for warning in scenario_warnings
        if warning["kind"] == "rendered_surface_hygiene"
    ]
    machine_rendering["hygiene_warning_count"] = len(
        rendered_hygiene_warnings
    )
    machine_rendering["hygiene_clean"] = not rendered_hygiene_warnings

    engineering = _tier(
        {
            "catalog_is_structurally_parsed": True,
            "conversation_bank_has_exact_catalog_coverage": True,
            "selected_split_has_every_domain_attribute_cell": all(
                cell["available_scenarios"] >= 1 for cell in capacity["cells"]
            ),
            "prospective_probability_grid_is_complete": (
                probability["scenario_anchor_state_count"]
                == (len(scenarios) * 2 * len(THETA_STATES) * len(susceptibilities))
            ),
            "concise_human_review_previews_rendered": all(
                len(row["rendered_examples"]) == 6 for row in review_rows
            ),
            "exhaustive_machine_surface_rendering_is_complete": bool(
                machine_rendering["complete"]
            ),
        }
    )
    scientific = _tier(
        {
            "engineering_pilot_ready": bool(engineering["ready"]),
            "independent_human_review_evidence_bundle_verified": False,
            "planned_horizon_has_no_repeat_capacity": bool(
                capacity["planned_all_cells_sufficient"]
            ),
            "prospective_probability_guardrails_pass": bool(
                probability["all_cells_passed"]
            ),
            "restricted_peer_nuisance_design_is_counterbalanced": bool(
                nuisance_design["all_cells_passed"]
            ),
            "neutral_conversation_frame_families_are_counterbalanced": bool(
                conversation_frame_design["passed"]
            ),
            "automated_validation_recorded_as_passed": bool(
                selected_reviews["all_automated_validation_passed"]
            ),
            "independent_surface_human_review_passed": bool(
                selected_reviews["all_surface_human_review_passed"]
            ),
            "independent_scientific_human_review_passed": bool(
                selected_reviews["all_scientific_human_review_passed"]
            ),
            "exhaustive_machine_surface_hygiene_is_clean": bool(
                machine_rendering["hygiene_clean"]
            ),
            "unresolved_machine_warning_count_is_zero": not warnings,
        }
    )
    paper = _tier(
        {
            "recorded_scientific_pilot_ready": bool(scientific["ready"]),
            "cyclic_reference_16_turn_horizon_has_no_repeat_capacity": bool(
                capacity["cyclic_reference_all_cells_sufficient"]
            ),
            "selected_scenarios_are_approved": bool(selected_reviews["all_approved"]),
            "selected_scenarios_are_paper_eligible": bool(
                selected_reviews["all_paper_eligible"]
            ),
            "catalog_is_not_simulation_and_pilot_only": (
                catalog.eligibility != "simulation-and-pilot-only"
            ),
            "catalog_is_not_frozen_development": (
                catalog.catalog_status != "frozen-development"
            ),
        }
    )

    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "audit_kind": "prospective_scenario_calibration",
        "audit_policy": AUDIT_POLICY,
        "outcome_data_used": False,
        "review_status_mutated": False,
        "catalog": {
            "catalog_id": catalog.catalog_id,
            "catalog_version": catalog.catalog_version,
            "catalog_status": catalog.catalog_status,
            "eligibility": catalog.eligibility,
            "selection_policy": catalog.selection_policy,
            "selected_split": split,
            "selected_domains": list(domain_scope),
            "selected_scenario_count": len(scenarios),
        },
        "conversation_bank": {
            "bank_id": conversation_bank.bank_id,
            "source": conversation_bank.source,
            "scenario_count": len(conversation_bank.templates),
            "exact_catalog_coverage": True,
            "runtime_display_name_policy": "presentation-position-a-b-v1",
            "model_visible_structured_id_policy": (
                "presented-option-position-alias-v1"
            ),
        },
        "response_model": {
            "family": "random_utility",
            "beta": response_model.beta,
            "ranking_scale": response_model.ranking_scale,
            "default_scale": response_model.default_scale,
            "suggestion_scale": response_model.suggestion_scale,
            "susceptibility_levels": list(levels),
            "minimum_matched_probability": minimum_probability,
        },
        "capacity": capacity,
        "restricted_peer_nuisance_design": nuisance_design,
        "conversation_frame_design": conversation_frame_design,
        "probability_calibration": probability,
        "machine_surface_rendering": machine_rendering,
        "human_review_counts": {
            "selected_split": selected_reviews,
            "whole_catalog": catalog_reviews,
        },
        "warnings": {
            "blocks_machine_readiness": False,
            "blocks_recorded_scientific_readiness": bool(warnings),
            "version_bound_adjudication_mechanism_available": False,
            "raw_option_label_word_count_difference_threshold_exclusive": (
                RAW_LABEL_WORD_COUNT_DIFFERENCE_WARNING
            ),
            "raw_option_label_word_count_ratio_range_inclusive": {
                "lower": RAW_LABEL_WORD_COUNT_RATIO_RANGE[0],
                "upper": RAW_LABEL_WORD_COUNT_RATIO_RANGE[1],
            },
            "cross_split_lexical_overlap_token_jaccard_threshold_inclusive": (
                CROSS_SPLIT_LEXICAL_JACCARD_WARNING
            ),
            "within_split_lexical_redundancy_token_jaccard_threshold_inclusive": (
                WITHIN_SPLIT_LEXICAL_JACCARD_WARNING
            ),
            "raw_option_label_word_count_warnings": [
                warning
                for warning in scenario_warnings
                if warning["kind"]
                in {
                    "option_label_raw_word_count_difference",
                    "option_label_raw_word_count_ratio_outside_range",
                }
            ],
            "rendered_surface_hygiene": rendered_hygiene_warnings,
            "lexical_overlap_candidates": lexical_overlaps,
            "cross_split_exact_task_family_reuse": task_family_reuse,
            "all": warnings,
            "warning_count": len(warnings),
        },
        "readiness_contract": {
            "scientific_pilot_key_interpretation": (
                "recorded_scientific_pilot_readiness"
            ),
            "human_review_evidence": {
                "verification_supported": False,
                "verified": False,
                "reason": (
                    "No version-bound human-review and neutral-choice pretest "
                    "evidence import contract exists yet. Catalog review status "
                    "strings are declarations and cannot satisfy this criterion."
                ),
            },
        },
        "readiness": {
            "engineering_pilot": engineering,
            "scientific_pilot": scientific,
            "paper": paper,
        },
        "human_review_packet": {
            "instructions": [
                (
                    "Review only the frozen scenario and conversation "
                    "surfaces; do not inspect experiment or evaluated-model "
                    "outcomes."
                ),
                (
                    "Machine warnings are review candidates rather than proof "
                    "of invalidity, but any unresolved warning blocks recorded "
                    "scientific readiness. This audit defines no version-bound "
                    "warning-adjudication record, so publish a new source "
                    "revision in which the warning is absent."
                ),
                (
                    "Record naturalness and neutrality separately from "
                    "scientific feature alignment, tradeoff validity, and "
                    "non-dominance."
                ),
            ],
            "scenarios": review_rows,
        },
    }


def _markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _yes_no(value: Any) -> str:
    return "yes" if bool(value) else "no"


def render_blinded_surface_review_markdown(
    audit: Mapping[str, Any],
) -> str:
    """Render an opaque surface-only packet for independent human review.

    The packet intentionally exposes only the natural-language exchanges that
    a surface reviewer must judge. Scenario identifiers, experimental
    dimensions, option roles, feature vectors, split membership, nuisance
    metadata, and mechanism labels remain in the researcher audit only.
    Producing this packet records no review evidence and changes no readiness
    state.
    """

    if not isinstance(audit, Mapping):
        raise TypeError("audit must be a mapping")
    if audit.get("schema_version") != AUDIT_SCHEMA_VERSION:
        raise ValueError(f"audit schema_version must be {AUDIT_SCHEMA_VERSION}")
    if audit.get("audit_policy") != AUDIT_POLICY:
        raise ValueError(f"audit_policy must be {AUDIT_POLICY!r}")

    scenarios = audit["human_review_packet"]["scenarios"]
    lines = [
        "# Blinded scenario surface-review packet",
        "",
        "> Review only the language shown below. Item and surface labels are "
        "opaque. Experimental metadata and evaluated-model outcomes are "
        "intentionally withheld.",
        "",
        "This packet records no review evidence, does not approve any item, "
        "and does not change scientific or paper readiness.",
        "",
        "## Reviewer instructions",
        "",
        "For every surface, record:",
        "",
        "- naturalness from 1 (very unnatural) to 5 (fully natural);",
        "- neutrality from 1 (strong unexplained pressure) to 5 (no unexplained "
        "pressure beyond any explicit presentation cue in the text); and",
        "- a short note for any grammatical problem, ambiguity, unequal "
        "specificity, or objectively superior description.",
        "",
        "Do not inspect experiment results or evaluated-model outputs while "
        "reviewing this packet.",
        "",
        f"Items to review: **{len(scenarios)}**",
    ]
    for item_index, scenario in enumerate(scenarios, start=1):
        lines.extend(
            [
                "",
                "---",
                "",
                f"## Item {item_index:03d}",
                "",
            ]
        )
        for surface_index, example in enumerate(
            scenario["rendered_examples"],
            start=1,
        ):
            lines.extend(
                [
                    f"### Surface {surface_index:02d}",
                    "",
                    f"> **Assistant:** {_markdown_cell(example['assistant_message'])}",
                    ">",
                    f"> **User:** {_markdown_cell(example['user_message'])}",
                    "",
                    "- Naturalness (1–5): ____",
                    "- Neutrality (1–5): ____",
                    "- Notes: ________________________________________________",
                    "",
                ]
            )
        lines.extend(
            [
                "### Item summary",
                "",
                "- [ ] All displayed language is grammatical and coherent.",
                "- [ ] The alternatives are described with comparable clarity "
                "and specificity.",
                "- [ ] No wording beyond an explicit presentation cue makes one "
                "alternative seem objectively required.",
                "- Item notes: _____________________________________________",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"


def render_scenario_calibration_markdown(
    audit: Mapping[str, Any],
) -> str:
    """Render a deterministic human-review packet from one audit dictionary."""

    if not isinstance(audit, Mapping):
        raise TypeError("audit must be a mapping")
    if audit.get("schema_version") != AUDIT_SCHEMA_VERSION:
        raise ValueError(f"audit schema_version must be {AUDIT_SCHEMA_VERSION}")
    if audit.get("audit_policy") != AUDIT_POLICY:
        raise ValueError(f"audit_policy must be {AUDIT_POLICY!r}")

    catalog = audit["catalog"]
    capacity = audit["capacity"]
    nuisance = audit["restricted_peer_nuisance_design"]
    conversation_frames = audit["conversation_frame_design"]
    probability = audit["probability_calibration"]
    reviews = audit["human_review_counts"]["selected_split"]
    warnings = audit["warnings"]
    packet = audit["human_review_packet"]
    lines = [
        "# Prospective scenario calibration and researcher review packet",
        "",
        "> This packet uses frozen inputs and declared simulator parameters "
        "only. It contains no experiment outcomes or evaluated-model results, "
        "and producing it does not change any review status.",
        "",
        "> This is the detailed researcher packet: it exposes experimental "
        "metadata for scientific inspection and must not be substituted for "
        "the separate blinded surface-review packet.",
        "",
        "## Scope",
        "",
        f"- Catalog: `{_markdown_cell(catalog['catalog_id'])}` "
        f"version `{_markdown_cell(catalog['catalog_version'])}`",
        f"- Split: `{_markdown_cell(catalog['selected_split'])}`",
        f"- Domains: `{_markdown_cell(', '.join(catalog['selected_domains']))}`",
        f"- Policies: `{_markdown_cell(', '.join(capacity['configured_policies']))}`",
        f"- Scenarios: **{catalog['selected_scenario_count']}**",
        f"- Planned turns: **{capacity['planned_turns']}**",
        f"- Planned capacity basis: "
        f"`{_markdown_cell(capacity['planned_capacity_basis'])}`",
        f"- Audit policy: `{AUDIT_POLICY}`",
        "",
        "## Readiness",
        "",
        "| Tier | Ready | Blocking criteria |",
        "| --- | --- | --- |",
    ]
    for key, label in (
        ("engineering_pilot", "Engineering pilot"),
        ("scientific_pilot", "Recorded scientific pilot"),
        ("paper", "Paper"),
    ):
        tier = audit["readiness"][key]
        blocking = (
            ", ".join(f"`{item}`" for item in tier["blocking_reasons"])
            if tier["blocking_reasons"]
            else "none"
        )
        lines.append(f"| {label} | **{_yes_no(tier['ready'])}** | {blocking} |")

    lines.extend(
        [
            "",
            "Engineering readiness means the inputs parse, cover every cell, "
            "render, and support the complete prospective probability grid. "
            "Recorded scientific-pilot readiness additionally requires the planned "
            "no-repeat capacity, probability guardrails, completed automated "
            "review, and verified version-bound evidence for independent human "
            "reviews and pretests, with no unresolved machine warnings. Catalog "
            "status strings alone are insufficient. Paper readiness also "
            "requires the separately "
            "labeled cyclic 16-turn reference capacity and explicit approved, "
            "paper-eligible catalog status.",
            "",
            "## No-repeat capacity",
            "",
            "| Domain | Attribute | Available | Planned required | Planned "
            "sufficient | Cyclic 16-turn required | Cyclic 16-turn sufficient |",
            "| --- | --- | ---: | ---: | --- | ---: | --- |",
        ]
    )
    for cell in capacity["cells"]:
        lines.append(
            f"| {_markdown_cell(cell['domain'])} | "
            f"{_markdown_cell(cell['target_key'])} | "
            f"{cell['available_scenarios']} | "
            f"{cell['planned_no_repeat_required']} | "
            f"{_yes_no(cell['planned_no_repeat_sufficient'])} | "
            f"{cell['cyclic_reference_16_turn_no_repeat_required']} | "
            f"{_yes_no(cell['cyclic_reference_16_turn_no_repeat_sufficient'])} |"
        )

    lines.extend(
        [
            "",
            "## Restricted-peer nuisance design",
            "",
            "The restricted alternative preserves the anchor's target "
            "direction while changing one non-target attribute by 0.25. Both "
            "eligible non-target attributes and both peer-minus-anchor "
            "directions must be balanced within one observation in every "
            "multi-scenario target cell.",
            "",
            "| Domain | Target | Nuisance attributes | Directions | Joint "
            "combinations | Pass |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for cell in nuisance["cells"]:
        attribute_counts = ", ".join(
            f"{key}={value}"
            for key, value in cell["attribute_counts"].items()
        )
        direction_counts = ", ".join(
            f"{key}={value}"
            for key, value in cell["direction_counts"].items()
        )
        joint_counts = ", ".join(
            f"{key}={value}"
            for key, value in cell["joint_counts"].items()
        )
        lines.append(
            f"| {_markdown_cell(cell['domain'])} | "
            f"{_markdown_cell(cell['target_key'])} | "
            f"{_markdown_cell(attribute_counts)} | "
            f"{_markdown_cell(direction_counts)} | "
            f"{_markdown_cell(joint_counts)} | "
            f"{_yes_no(cell['passed'])} |"
        )

    lines.extend(
        [
            "",
            "## Neutral conversation-frame design",
            "",
            "The audit groups identical neutral bases before any canonical "
            "default or suggestion sentence is added. Multiple source-neutral "
            "frame families must be balanced within one in every selected "
            "domain-by-target cell. This is a machine design check, not a "
            "human naturalness or neutrality judgment.",
            "",
            f"- Frame families: **{conversation_frames['frame_family_count']}**",
            f"- Overall counts: "
            f"`{_markdown_cell(conversation_frames['frame_counts'])}`",
            f"- Assistant-agency candidates: "
            f"**{len(conversation_frames['assistant_agency_candidates'])}**",
            "",
            "| Domain | Target | Frame counts | Pass |",
            "| --- | --- | --- | --- |",
        ]
    )
    for cell in conversation_frames["cells"]:
        lines.append(
            f"| {_markdown_cell(cell['domain'])} | "
            f"{_markdown_cell(cell['target_key'])} | "
            f"{_markdown_cell(cell['frame_counts'])} | "
            f"{_yes_no(cell['balanced_within_one'])} |"
        )

    summary = probability["summary"]
    guardrails = probability["guardrails"]
    lines.extend(
        [
            "",
            "## Prospective simulator calibration",
            "",
            "Every probability is evaluated over all declared theta states, "
            "all Cartesian susceptibility profiles, both anchor directions, "
            "and both display orders. The legacy balanced/restricted estimands "
            "are explicitly order-averaged. Separately, every physical "
            "per-order probability for every mechanism must exceed the "
            "configured minimum matched probability.",
            "",
            f"The {probability['scenario_anchor_instance_count']} scenario-anchor "
            f"instances contain **{probability['unique_numeric_signature_count']}** "
            "unique numeric signatures "
            f"(repetition factor {probability['numeric_signature_repetition_factor']}).",
            "",
            "| Quantity | Minimum | Mean | Maximum | Guardrail | Pass |",
            "| --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for name, label, guardrail in (
        (
            "balanced_order_averaged_probability",
            "Order-averaged balanced anchor probability",
            guardrails["order_averaged_balanced_probability"],
        ),
        (
            "restricted_order_averaged_probability",
            "Order-averaged restricted anchor probability",
            guardrails["order_averaged_restricted_probability"],
        ),
    ):
        item = summary[name]
        boundary = (
            f"{guardrail['lower_inclusive']:.2f}–{guardrail['upper_inclusive']:.2f}"
        )
        lines.append(
            f"| {label} | {item['minimum']:.6f} | {item['mean']:.6f} | "
            f"{item['maximum']:.6f} | {boundary} for every averaged case | "
            f"{_yes_no(guardrail['passed'])} |"
        )
    for mechanism in ("balanced", "restricted", "default", "suggestion"):
        item = probability["physical_probability_summary"][mechanism]
        guardrail = guardrails["physical_mechanism_probabilities"][mechanism]
        lines.append(
            f"| Physical {mechanism} anchor probability | "
            f"{item['minimum']:.6f} | {item['mean']:.6f} | "
            f"{item['maximum']:.6f} | every order strictly between "
            f"{guardrail['minimum_matched_probability_exclusive']:.2f} and "
            f"{guardrail['complementary_maximum_probability_exclusive']:.2f} "
            "(both binary responses protected) | "
            f"{_yes_no(guardrail['passed'])} |"
        )
    for mechanism, label in (
        ("ranking", "Mean ranking increment"),
        ("default", "Mean default increment"),
        ("suggestion", "Mean suggestion increment"),
    ):
        item = summary[f"{mechanism}_increment"]
        guardrail = guardrails["mean_incremental_effects"][mechanism]
        boundary = (
            f"{guardrail['lower_inclusive']:.2f}–{guardrail['upper_inclusive']:.2f}"
        )
        lines.append(
            f"| {label} | {item['minimum']:.6f} | {item['mean']:.6f} | "
            f"{item['maximum']:.6f} | mean {boundary} | "
            f"{_yes_no(guardrail['passed'])} |"
        )

    lines.extend(
        [
            "",
            "## Review state and warnings",
            "",
            f"- Automated validation passed: "
            f"**{reviews['automated_validation'].get('passed', 0)} / "
            f"{reviews['scenario_count']}**",
            f"- Surface human review passed: "
            f"**{reviews['surface_human_review'].get('passed', 0)} / "
            f"{reviews['scenario_count']}**",
            f"- Scientific human review passed: "
            f"**{reviews['scientific_human_review'].get('passed', 0)} / "
            f"{reviews['scenario_count']}**",
            "- Version-bound human-review and pretest evidence verified: "
            "**no** (an evidence import contract is not implemented yet)",
            f"- Paper eligible: **{reviews['paper_eligible_count']} / "
            f"{reviews['scenario_count']}**",
            f"- Unresolved machine warnings: **{warnings['warning_count']}** "
            "(non-blocking for engineering; blocking for recorded scientific "
            "readiness)",
            f"- Exhaustive machine surfaces checked: "
            f"**{audit['machine_surface_rendering']['rendered_total_surface_count']} / "
            f"{audit['machine_surface_rendering']['expected_total_surface_count']}**",
            "",
        ]
    )
    if warnings["all"]:
        lines.extend(["### Warning candidates", ""])
        for warning in warnings["all"]:
            if warning["kind"] == "option_label_raw_word_count_difference":
                lines.append(
                    "- Raw label word-count difference: "
                    f"`{warning['scenario_id']}` / "
                    f"`{warning['presentation']}` differs by "
                    f"{warning['absolute_difference']} words."
                )
            elif warning["kind"] == (
                "option_label_raw_word_count_ratio_outside_range"
            ):
                lines.append(
                    "- Raw label word-count ratio outside 0.85–1.15: "
                    f"`{warning['scenario_id']}` / "
                    f"`{warning['presentation']}` has ratio "
                    f"{warning['first_to_second_ratio']}."
                )
            elif warning["kind"] == "rendered_surface_hygiene":
                lines.append(
                    "- Rendered-language issue: "
                    f"`{warning['scenario_id']}` / "
                    f"`{warning['presentation_mechanism']}` / "
                    f"`{warning['display_order']}` / "
                    f"`{warning['selected_role']}` / "
                    f"`{warning['message_role']}` / "
                    f"`{warning['issue']}`."
                )
            elif warning["kind"] in {
                "within_split_lexical_redundancy_candidate",
                "cross_split_lexical_overlap_candidate",
            }:
                label = (
                    "Within-split lexical redundancy candidate"
                    if warning["kind"]
                    == "within_split_lexical_redundancy_candidate"
                    else "Cross-split lexical overlap candidate"
                )
                lines.append(
                    f"- {label}: "
                    f"`{warning['scenario_a']}` and "
                    f"`{warning['scenario_b']}` "
                    f"(token Jaccard {warning['token_jaccard']:.3f})."
                )
            elif warning["kind"] == (
                "cross_split_exact_task_family_reuse_review_flag"
            ):
                lines.append(
                    "- Exact task-family reuse across splits: "
                    f"`{warning['domain']}` / `{warning['task_family']}` / "
                    f"`{', '.join(warning['splits'])}`."
                )
        lines.append("")

    lines.extend(["## Human-review instructions", ""])
    lines.extend(
        f"{index}. {item}"
        for index, item in enumerate(
            packet["instructions"],
            start=1,
        )
    )

    for scenario in packet["scenarios"]:
        lines.extend(
            [
                "",
                "---",
                "",
                f"## `{_markdown_cell(scenario['scenario_id'])}`",
                "",
                f"- Domain / task: `{_markdown_cell(scenario['domain'])}` / "
                f"`{_markdown_cell(scenario['task_family'])}`",
                f"- Target: `{_markdown_cell(scenario['target_key'])}` "
                f"(attribute {scenario['target_attribute']})",
                f"- Restricted peer nuisance: "
                f"`{_markdown_cell(scenario['nuisance_key'])}` "
                f"(attribute {scenario['nuisance_attribute']}, "
                f"direction {scenario['nuisance_direction']:+d})",
                f"- Status: `{_markdown_cell(scenario['status'])}`",
                f"- Recorded reviews: automated "
                f"`{_markdown_cell(scenario['review']['automated_validation'])}`, "
                f"surface "
                f"`{_markdown_cell(scenario['review']['surface_human_review'])}`, "
                f"scientific "
                f"`{_markdown_cell(scenario['review']['scientific_human_review'])}`",
                "",
                f"> **Prompt:** {_markdown_cell(scenario['prompt'])}",
                "",
                "Stored A–D entries define the noun-stem pool. Rendered A/B "
                "names below are reassigned by visible position.",
                "",
                "| Role | Stored name-pool entry | Visible label | Words | "
                "Features (reviewer aid; not model-visible) |",
                "| --- | --- | --- | ---: | --- |",
            ]
        )
        for option in scenario["options"]:
            lines.append(
                f"| `{_markdown_cell(option['role'])}` | "
                f"{_markdown_cell(option['stored_name_pool_entry'])} | "
                f"{_markdown_cell(option['label'])} | "
                f"{option['label_word_count']} | "
                f"`{_markdown_cell(option['features'])}` |"
            )
        if scenario["warnings"]:
            lines.extend(["", "**Scenario warnings:**"])
            for warning in scenario["warnings"]:
                if warning["kind"] == "option_label_raw_word_count_difference":
                    lines.append(
                        "- Raw label word-count difference of "
                        f"{warning['absolute_difference']} in "
                        f"`{warning['presentation']}`."
                    )
                elif warning["kind"] == (
                    "option_label_raw_word_count_ratio_outside_range"
                ):
                    lines.append(
                        "- Raw label word-count ratio of "
                        f"{warning['first_to_second_ratio']} in "
                        f"`{warning['presentation']}` is outside 0.85–1.15."
                    )
                elif warning["kind"] == "rendered_surface_hygiene":
                    lines.append(
                        "- Rendered-language issue "
                        f"`{warning['issue']}` in "
                        f"`{warning['presentation_mechanism']}` / "
                        f"`{warning['display_order']}` / "
                        f"`{warning['selected_role']}`."
                    )

        lines.extend(
            [
                "",
                "### Concise rendered review preview",
                "",
                f"The machine audit checked "
                f"{scenario['machine_surface_count']} exhaustive surfaces; "
                "the six representative surfaces below are the human preview.",
                "",
            ]
        )
        for example in scenario["rendered_examples"]:
            lines.extend(
                [
                    f"**{_markdown_cell(example['review_surface'])}**",
                    "",
                    f"> **Assistant:** {_markdown_cell(example['assistant_message'])}",
                    ">",
                    f"> **User:** {_markdown_cell(example['user_message'])}",
                    "",
                ]
            )
        lines.extend(
            [
                "### Independent review checklist",
                "",
                "- [ ] Surface wording is grammatical and natural.",
                "- [ ] Prompt, names, and labels are choice-neutral.",
                "- [ ] Every visible distinction maps to the declared feature "
                "role or is held constant.",
                "- [ ] Balanced and restricted tradeoffs are plausible and "
                "non-dominating.",
                "- [ ] Default, suggestion, restriction, and ordering differ "
                "only through their declared treatment.",
                "- [ ] Near-duplicate and raw word-count warnings were reviewed.",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "AUDIT_POLICY",
    "AUDIT_SCHEMA_VERSION",
    "build_scenario_calibration_audit",
    "render_blinded_surface_review_markdown",
    "render_scenario_calibration_markdown",
]
