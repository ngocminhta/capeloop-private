"""Machine-readable scientific stage-gate diagnostics.

Gate reports expose computed checks but never promote a smoke/pilot run into a
paper claim. ``claim_status`` stays ``"not_claimed"`` until researchers attach a
frozen analysis protocol and explicitly review the retained evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping
import math


@dataclass(frozen=True, slots=True)
class GateCriterion:
    criterion_id: str
    description: str
    passed: bool | None
    observed: Any
    requirement: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "criterion_id": self.criterion_id,
            "description": self.description,
            "passed": self.passed,
            "observed": self.observed,
            "requirement": self.requirement,
        }


@dataclass(frozen=True, slots=True)
class GateReport:
    gate_id: str
    title: str
    criteria: tuple[GateCriterion, ...]
    evidence_scope: str = "diagnostic"
    claim_status: str = "not_claimed"

    @property
    def computed_status(self) -> str:
        decisions = tuple(criterion.passed for criterion in self.criteria)
        if not decisions or any(decision is None for decision in decisions):
            return "incomplete"
        return (
            "meets_computational_checks" if all(decisions) else "does_not_meet_checks"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "gate_id": self.gate_id,
            "title": self.title,
            "evidence_scope": self.evidence_scope,
            "computed_status": self.computed_status,
            "claim_status": self.claim_status,
            "criteria": [criterion.to_dict() for criterion in self.criteria],
        }


def _finite_mean(values: Iterable[float]) -> float | None:
    material = tuple(float(value) for value in values if math.isfinite(float(value)))
    return None if not material else math.fsum(material) / len(material)


def gate_1_from_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    material_gap: float = 0.01,
    required_mechanisms: int = 2,
    required_mechanism_ids: Iterable[str] = (
        "balanced",
        "restricted",
        "ranking",
        "default",
        "suggested",
    ),
    required_domains: Iterable[str] = ("travel", "writing"),
    exact_updater_id: str = "exact_action_aware",
    oracle_tolerance: float = 1e-10,
    same_response_audit_passed: bool | None = None,
    held_out_paraphrase_ready: bool | None = None,
    held_out_paraphrase_verified: bool | None = None,
) -> GateReport:
    """Evaluate whether Experiment A can identify provenance calibration.

    This is deliberately an outcome-neutral readiness gate.  It checks the
    controlled design, exact-reference implementation, and held-out surface
    coverage, but it never requires an evaluated updater to deviate from the
    oracle. ``held_out_paraphrase_verified`` is retained as a compatibility
    alias for callers using the earlier artifact name.
    """

    if (
        isinstance(material_gap, bool)
        or not isinstance(material_gap, (int, float))
        or not math.isfinite(float(material_gap))
        or material_gap < 0
    ):
        raise ValueError("material_gap must be a finite non-negative number")
    if (
        isinstance(required_mechanisms, bool)
        or not isinstance(required_mechanisms, int)
        or required_mechanisms <= 0
    ):
        raise ValueError("required_mechanisms must be a positive integer")
    if (
        isinstance(oracle_tolerance, bool)
        or not isinstance(oracle_tolerance, (int, float))
        or not math.isfinite(float(oracle_tolerance))
        or oracle_tolerance < 0
    ):
        raise ValueError("oracle_tolerance must be a finite non-negative number")
    if (
        held_out_paraphrase_ready is not None
        and held_out_paraphrase_verified is not None
        and held_out_paraphrase_ready != held_out_paraphrase_verified
    ):
        raise ValueError("held-out paraphrase readiness aliases disagree")
    material = tuple(rows)
    controlled = [
        row for row in material if row.get("response_mode") == "controlled_anchor"
    ]
    expected_domains = tuple(sorted(set(required_domains)))
    expected_mechanisms = tuple(sorted(set(required_mechanism_ids)))
    nonbalanced = tuple(
        mechanism for mechanism in expected_mechanisms if mechanism != "balanced"
    )
    if not expected_domains:
        raise ValueError("required_domains cannot be empty")
    exact_rows = [
        row for row in controlled if row.get("updater_id") == exact_updater_id
    ]
    observed_domains = tuple(
        sorted(
            {
                str(row.get("domain"))
                for row in exact_rows
                if row.get("domain") is not None
            }
        )
    )
    observed_mechanisms = tuple(
        sorted(
            {
                str(row.get("mechanism"))
                for row in exact_rows
                if row.get("mechanism") is not None
            }
        )
    )
    covered_cells = {
        (str(row.get("domain")), str(row.get("mechanism")))
        for row in exact_rows
        if row.get("domain") is not None and row.get("mechanism") is not None
    }
    expected_cells = {
        (domain, mechanism)
        for domain in expected_domains
        for mechanism in expected_mechanisms
    }
    coverage_complete = expected_cells <= covered_cells

    def exact_update(row: Mapping[str, Any]) -> float | None:
        value = row.get(
            "exact_anchor_directional_log_odds_update",
            row.get("exact_log_odds_update"),
        )
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            return None
        return float(value)

    def system_update(row: Mapping[str, Any]) -> float | None:
        value = row.get(
            "anchor_directional_log_odds_update",
            row.get("log_odds_update"),
        )
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            return None
        return float(value)

    oracle_residuals: list[float] = []
    oracle_acues: list[float] = []
    oracle_rows_valid = bool(exact_rows)
    for row in exact_rows:
        expected = exact_update(row)
        observed = system_update(row)
        acue = row.get("exact_acue")
        if (
            expected is None
            or observed is None
            or isinstance(acue, bool)
            or not isinstance(acue, (int, float))
            or not math.isfinite(float(acue))
        ):
            oracle_rows_valid = False
            continue
        oracle_residuals.append(abs(observed - expected))
        oracle_acues.append(abs(float(acue)))
    oracle_consistent = (
        None
        if not exact_rows
        else (
            oracle_rows_valid
            and len(oracle_residuals) == len(exact_rows)
            and max(oracle_residuals, default=math.inf) <= oracle_tolerance
            and max(oracle_acues, default=math.inf) <= oracle_tolerance
        )
    )

    def matched_set_id(row: Mapping[str, Any]) -> tuple[Any, ...]:
        explicit = (
            row.get("user_id"),
            row.get("domain"),
            row.get("target_attribute"),
            row.get("anchor_direction"),
            row.get("prior_stratum"),
            row.get("prior_strength"),
        )
        if all(value is not None for value in explicit):
            return explicit
        trial_id = row.get("trial_id")
        mechanism = row.get("mechanism")
        response_mode = row.get("response_mode")
        if (
            isinstance(trial_id, str)
            and isinstance(mechanism, str)
            and isinstance(response_mode, str)
        ):
            suffix = f":{mechanism}:{response_mode}"
            if trial_id.endswith(suffix):
                return (trial_id[: -len(suffix)],)
        return (trial_id,)

    by_set: dict[
        tuple[Any, ...],
        dict[str, tuple[str, float]],
    ] = {}
    for row in exact_rows:
        value = exact_update(row)
        mechanism = row.get("mechanism")
        domain = row.get("domain")
        if (
            value is None
            or not isinstance(mechanism, str)
            or not isinstance(domain, str)
        ):
            continue
        by_set.setdefault(matched_set_id(row), {})[mechanism] = (
            domain,
            value,
        )
    separation_values: dict[tuple[str, str], list[float]] = {
        (domain, mechanism): []
        for domain in expected_domains
        for mechanism in nonbalanced
    }
    for cells in by_set.values():
        balanced = cells.get("balanced")
        if balanced is None:
            continue
        balanced_domain, balanced_update = balanced
        for mechanism in nonbalanced:
            comparison = cells.get(mechanism)
            if comparison is None or comparison[0] != balanced_domain:
                continue
            separation_values[(balanced_domain, mechanism)].append(
                abs(comparison[1] - balanced_update)
            )
    mean_separations = {
        mechanism: {
            domain: _finite_mean(separation_values[(domain, mechanism)])
            for domain in expected_domains
        }
        for mechanism in nonbalanced
    }
    qualifying_mechanisms = tuple(
        mechanism
        for mechanism, domain_values in mean_separations.items()
        if all(
            value is not None and value > material_gap
            for value in domain_values.values()
        )
    )
    separation_estimable = coverage_complete and all(
        separation_values[(domain, mechanism)]
        for domain in expected_domains
        for mechanism in nonbalanced
    )
    held_out_ready = (
        held_out_paraphrase_ready
        if held_out_paraphrase_ready is not None
        else held_out_paraphrase_verified
    )
    return GateReport(
        gate_id="gate-1",
        title="Identifiable causal-provenance calibration",
        criteria=(
            GateCriterion(
                "same-response-invariance",
                (
                    "The matched controlled-anchor design holds the selected "
                    "response, prior, user, and anchor fixed across provenance "
                    "mechanisms."
                ),
                same_response_audit_passed,
                {"same_response_audit_passed": same_response_audit_passed},
                "the machine-readable same-response audit passes",
            ),
            GateCriterion(
                "exact-oracle-self-consistency",
                (
                    "The exact updater reproduces the exact action-aware "
                    "posterior used as its reference."
                ),
                oracle_consistent,
                {
                    "exact_updater_id": exact_updater_id,
                    "row_count": len(exact_rows),
                    "tolerance": oracle_tolerance,
                    "maximum_exact_acue": (max(oracle_acues) if oracle_acues else None),
                    "maximum_log_odds_residual": (
                        max(oracle_residuals) if oracle_residuals else None
                    ),
                },
                f"all exact self-residuals <= {oracle_tolerance}",
            ),
            GateCriterion(
                "warranted-update-separation",
                (
                    "The declared response model warrants different update "
                    "strengths for non-balanced provenance mechanisms while "
                    "the controlled response is held fixed."
                ),
                (
                    None
                    if not separation_estimable
                    else len(qualifying_mechanisms) >= required_mechanisms
                ),
                {
                    "reference_basis": "exact_action_aware",
                    "balanced_reference": True,
                    "material_threshold": material_gap,
                    "qualifying_mechanisms": list(qualifying_mechanisms),
                    "required": required_mechanisms,
                    "mean_absolute_paired_separation_by_domain": (mean_separations),
                },
                (
                    f"mean paired exact-update separation from balanced > "
                    f"{material_gap} in both domains for at least "
                    f"{required_mechanisms} mechanisms"
                ),
            ),
            GateCriterion(
                "design-cell-coverage",
                "Every preregistered domain-by-mechanism cell is represented.",
                (coverage_complete if exact_rows else None),
                {
                    "required_domains": list(expected_domains),
                    "required_mechanisms": list(expected_mechanisms),
                    "observed_domains": list(observed_domains),
                    "observed_mechanisms": list(observed_mechanisms),
                    "missing_cells": [
                        {"domain": domain, "mechanism": mechanism}
                        for domain, mechanism in sorted(expected_cells - covered_cells)
                    ],
                },
                "all required domain-by-mechanism cells",
            ),
            GateCriterion(
                "held-out-controlled-paraphrase-readiness",
                (
                    "Held-out controlled paraphrases cover the declared "
                    "surface design and preserve selected-option and visible-"
                    "context semantics."
                ),
                held_out_ready,
                {"coverage_and_invariance_verified": held_out_ready},
                "complete held-out coverage and structural invariance",
            ),
        ),
    )


def gate_2_and_3_hierarchy_from_trajectories(
    rows: Iterable[Mapping[str, Any]],
    *,
    target_updater_ids: Iterable[str] | None = None,
    inferential_evidence: Mapping[str, Any] | None = None,
    selection_noninferiority_margin: float = 0.02,
    net_harm_margin: float = 0.02,
) -> tuple[GateReport, GateReport, GateReport]:
    """Return behavioral, legibility, and nested net-harm decisions.

    Version-5 inference evidence supplies one-sided complete-user sign-flip
    tests and frozen within-model multiplicity decisions. Older interval-only
    evidence remains readable so existing run artifacts and callers fail
    conservatively rather than becoming invalid.
    """

    if isinstance(selection_noninferiority_margin, bool) or (
        not isinstance(selection_noninferiority_margin, (int, float))
        or not math.isfinite(float(selection_noninferiority_margin))
        or selection_noninferiority_margin < 0.0
    ):
        raise ValueError(
            "selection_noninferiority_margin must be finite and non-negative"
        )
    if isinstance(net_harm_margin, bool) or (
        not isinstance(net_harm_margin, (int, float))
        or not math.isfinite(float(net_harm_margin))
        or net_harm_margin < 0.0
    ):
        raise ValueError("net_harm_margin must be finite and non-negative")
    if isinstance(inferential_evidence, Mapping):

        def verify_evidence_margin(name: str, frozen: float) -> None:
            if name not in inferential_evidence:
                return
            raw = inferential_evidence[name]
            if (
                isinstance(raw, bool)
                or not isinstance(raw, (int, float))
                or not math.isfinite(float(raw))
                or float(raw) < 0.0
            ):
                raise ValueError(
                    f"inferential_evidence.{name} must be finite and non-negative"
                )
            if not math.isclose(
                float(raw),
                frozen,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"inferential_evidence.{name} does not match the frozen "
                    "caller/config value"
                )

        verify_evidence_margin(
            "selection_noninferiority_margin",
            float(selection_noninferiority_margin),
        )
        verify_evidence_margin(
            "net_harm_margin",
            float(net_harm_margin),
        )
    material = tuple(rows)
    targets = None if target_updater_ids is None else set(target_updater_ids)
    soft = [
        row
        for row in material
        if row.get("policy_id") == "soft_profile_conditioned"
        and row.get("initial_profile") == "incorrect"
        and (targets is None or row.get("updater_id") in targets)
    ]
    cases = [row for row in soft if bool(row.get("is_self_confirming"))]
    self_confirming = len(cases)
    mechanisms = {
        str(mechanism) for row in soft for mechanism in row.get("mechanisms", ())
    }

    def inference_metric(metric_id: str) -> Mapping[str, Any] | None:
        if not isinstance(inferential_evidence, Mapping):
            return None
        metrics = inferential_evidence.get("metrics")
        if not isinstance(metrics, Mapping):
            return None
        value = metrics.get(metric_id)
        return value if isinstance(value, Mapping) else None

    def policy_inference_metric(
        policy_id: str,
        metric_id: str,
    ) -> Mapping[str, Any] | None:
        if not isinstance(inferential_evidence, Mapping):
            return None
        policies = inferential_evidence.get("policy_specific_metrics")
        if not isinstance(policies, Mapping):
            return None
        policy = policies.get(policy_id)
        if not isinstance(policy, Mapping):
            return None
        value = policy.get(metric_id)
        return value if isinstance(value, Mapping) else None

    multiplicity_payload = (
        inferential_evidence.get("multiplicity")
        if isinstance(inferential_evidence, Mapping)
        else None
    )
    enforce_multiplicity = bool(
        isinstance(multiplicity_payload, Mapping)
        or (
            isinstance(inferential_evidence, Mapping)
            and inferential_evidence.get("analysis")
            == "experiment-b-clustered-randomization-v5"
        )
    )

    def multiplicity_decision(
        claim_id: str,
    ) -> tuple[bool | None, Mapping[str, Any] | None]:
        if not enforce_multiplicity:
            return None, None
        if not isinstance(multiplicity_payload, Mapping):
            return False, None
        decisions = multiplicity_payload.get("claim_decisions")
        if not isinstance(decisions, Mapping):
            return False, None
        decision = decisions.get(claim_id)
        if not isinstance(decision, Mapping):
            return False, None
        status = decision.get("status")
        if status in {"not_computed", "not_estimable"}:
            return None, decision
        rejected = decision.get("multiplicity_reject")
        return (rejected if isinstance(rejected, bool) else False), decision

    def adequate(metric: Mapping[str, Any] | None) -> bool | None:
        if metric is None or metric.get("adequacy_status") == "not_computed":
            return None
        return bool(metric.get("adequate", False))

    def lower_above_zero(
        metric: Mapping[str, Any] | None,
    ) -> bool | None:
        if metric is None:
            return None
        lower = metric.get("lower")
        if not isinstance(lower, (int, float)) or not math.isfinite(float(lower)):
            return None
        return float(lower) > 0.0

    def lower_above(
        metric: Mapping[str, Any] | None,
        threshold: float,
    ) -> bool | None:
        if metric is None:
            return None
        lower = metric.get("lower")
        if not isinstance(lower, (int, float)) or not math.isfinite(float(lower)):
            return None
        return float(lower) > threshold

    def upper_at_or_below(
        metric: Mapping[str, Any] | None,
        threshold: float,
    ) -> bool | None:
        if metric is None:
            return None
        upper = metric.get("upper")
        if not isinstance(upper, (int, float)) or not math.isfinite(float(upper)):
            return None
        return float(upper) <= threshold

    def directional_test(
        metric: Mapping[str, Any] | None,
    ) -> Mapping[str, Any] | None:
        if metric is None:
            return None
        value = metric.get("directional_test")
        return value if isinstance(value, Mapping) else None

    def directional_adequate(
        metric: Mapping[str, Any] | None,
    ) -> bool | None:
        test = directional_test(metric)
        if test is None:
            return adequate(metric)
        status = test.get("adequacy_status")
        if status == "not_computed":
            return None
        return bool(test.get("adequate", False))

    def directional_pass(
        metric: Mapping[str, Any] | None,
        *,
        alternative: str,
        threshold: float,
    ) -> bool | None:
        test = directional_test(metric)
        if test is not None:
            status = test.get("adequacy_status")
            if status == "not_computed":
                return None
            if status != "adequate":
                return False
            declared_alternative = test.get("alternative")
            declared_margin = test.get("null_margin")
            if declared_alternative != alternative:
                return False
            if (
                isinstance(declared_margin, bool)
                or not isinstance(declared_margin, (int, float))
                or not math.isfinite(float(declared_margin))
                or not math.isclose(
                    float(declared_margin),
                    float(threshold),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                return False
            passed = test.get("passed")
            return passed if isinstance(passed, bool) else None
        if alternative == "greater":
            return lower_above(metric, threshold)
        if alternative == "less":
            return upper_at_or_below(metric, threshold)
        raise ValueError("unknown directional alternative")

    error_amplification_interval = inference_metric("error_amplification_ratio")
    excess_confidence_interval = inference_metric(
        "mean_cumulative_excess_confidence_log_odds"
    )
    excess_confidence_contrast_interval = inference_metric(
        "soft_minus_balanced_excess_confidence_log_odds"
    )
    soft_policy_excess_confidence_interval = (
        policy_inference_metric(
            "soft_profile_conditioned",
            "mean_cumulative_excess_confidence_log_odds",
        )
        or excess_confidence_interval
    )
    soft_policy_error_amplification_interval = (
        policy_inference_metric(
            "soft_profile_conditioned",
            "error_amplification_ratio",
        )
        or error_amplification_interval
    )
    soft_policy_partial_reinforcement_interval = policy_inference_metric(
        "soft_profile_conditioned",
        "reinforcement_event_rate",
    )
    information_deficit_interval = inference_metric(
        "action_aware_information_gain_deficit"
    )
    disconfirmation_deficit_interval = inference_metric(
        "disconfirmation_evidence_deficit_log_odds"
    )
    visible_divergence_interval = inference_metric("visible_action_divergence_rate")
    choice_divergence_interval = inference_metric("observed_choice_divergence_rate")
    later_action_interval = inference_metric("later_action_influence_rate")
    self_confirming_interval = inference_metric("self_confirming_profile_rate")
    soft_gap_interval = policy_inference_metric(
        "soft_profile_conditioned",
        "same_history_attribution_gap",
    )
    soft_balanced_gap_interval = inference_metric("soft_minus_balanced_attribution_gap")
    selection_interval = inference_metric("evidence_selection_cost")
    net_harm_interval = inference_metric("soft_minus_balanced_terminal_error")
    gate_3_intervals = {
        "soft_same_history_attribution_gap": soft_gap_interval,
        "soft_minus_balanced_attribution_gap": soft_balanced_gap_interval,
        "evidence_selection_cost": selection_interval,
    }
    adequate_gate_3 = tuple(
        directional_adequate(interval) for interval in gate_3_intervals.values()
    )
    supporting_harm_signals = {
        "error_amplification": directional_pass(
            soft_policy_error_amplification_interval,
            alternative="greater",
            threshold=1.0,
        ),
        "absolute_excess_confidence": directional_pass(
            soft_policy_excess_confidence_interval,
            alternative="greater",
            threshold=0.0,
        ),
        "partial_reinforcement": directional_pass(
            soft_policy_partial_reinforcement_interval,
            alternative="greater",
            threshold=0.0,
        ),
        "information_deficit_lower_gt_0": lower_above_zero(
            information_deficit_interval
        ),
        "disconfirmation_deficit_lower_gt_0": lower_above_zero(
            disconfirmation_deficit_interval
        ),
    }
    gate_2 = GateReport(
        gate_id="gate-2",
        title="Conditional behavioral feedback amplification",
        criteria=(
            GateCriterion(
                "soft-mechanism-coverage",
                "The soft treatment visibly uses a presentation channel.",
                len(mechanisms) >= 1 if soft else None,
                {
                    "mechanisms_among_eligible_trajectories": (sorted(mechanisms)),
                },
                "at least one of ranking, default, suggestion",
            ),
            GateCriterion(
                "soft-alternatives",
                "Counter-profile alternatives remain available.",
                (
                    all(
                        bool(row.get("counter_profile_available", False))
                        for row in soft
                    )
                    if soft
                    else None
                ),
                {"eligible_soft_trajectories": len(soft)},
                "all eligible soft trajectories retain both directions",
            ),
            GateCriterion(
                "visible-action-divergence",
                "Profile conditioning changes the visible action stream.",
                directional_pass(
                    visible_divergence_interval,
                    alternative="greater",
                    threshold=0.0,
                ),
                visible_divergence_interval,
                (
                    "one-sided complete-user randomization decision > 0; "
                    "clustered interval is sensitivity evidence"
                ),
            ),
            GateCriterion(
                "natural-choice-divergence",
                "The changed actions alter at least some natural responses.",
                directional_pass(
                    choice_divergence_interval,
                    alternative="greater",
                    threshold=0.0,
                ),
                choice_divergence_interval,
                (
                    "one-sided complete-user randomization decision > 0; "
                    "clustered interval is sensitivity evidence"
                ),
            ),
            GateCriterion(
                "later-action-influence",
                "Updated memory changes a later action.",
                directional_pass(
                    later_action_interval,
                    alternative="greater",
                    threshold=0.0,
                ),
                later_action_interval,
                (
                    "one-sided complete-user randomization decision > 0; "
                    "clustered interval is sensitivity evidence"
                ),
            ),
            GateCriterion(
                "continuous-endpoint-cluster-adequacy",
                (
                    "The paired relative-confidence endpoint uses enough "
                    "independent latent-user clusters."
                ),
                directional_adequate(excess_confidence_contrast_interval),
                excess_confidence_contrast_interval,
                "the declared paired CEC-contrast directional analysis is adequate",
            ),
            GateCriterion(
                "relative-confidence-penalty",
                (
                    "Soft conditioning retains more wrong-direction confidence "
                    "than the paired balanced policy. This is a relative "
                    "confidence penalty, not by itself reinforcement."
                ),
                directional_pass(
                    excess_confidence_contrast_interval,
                    alternative="greater",
                    threshold=0.0,
                ),
                {
                    "relative_confidence_penalty": (
                        excess_confidence_contrast_interval
                    ),
                    "absolute_excess_confidence": (
                        soft_policy_excess_confidence_interval
                    ),
                    "error_amplification": (soft_policy_error_amplification_interval),
                    "partial_reinforcement": (
                        soft_policy_partial_reinforcement_interval
                    ),
                    "supporting_noncontrolling_signals": (supporting_harm_signals),
                    "secondary_strict_self_confirmation": {
                        "case_count": self_confirming,
                        "interval": self_confirming_interval,
                        "controls_gate": False,
                    },
                },
                (
                    "soft-minus-balanced CEC one-sided complete-user "
                    "randomization decision > 0; interval is sensitivity evidence"
                ),
            ),
            *(
                (
                    GateCriterion(
                        "within-model-multiplicity-control",
                        (
                            "The Gate 2 intersection-union claim survives the "
                            "frozen post-Gate-3 Holm family within this model."
                        ),
                        multiplicity_decision(
                            "conditional_behavioral_feedback_amplification"
                        )[0],
                        multiplicity_decision(
                            "conditional_behavioral_feedback_amplification"
                        )[1],
                        (
                            "Gate 3 primary IUT rejects and the Gate 2 composite "
                            "is rejected by the fixed three-claim Holm family"
                        ),
                    ),
                )
                if enforce_multiplicity
                else ()
            ),
        ),
    )
    gate_3 = GateReport(
        gate_id="gate-3",
        title="Policy-conditioned evidential legibility",
        criteria=(
            GateCriterion(
                "soft-same-history-attribution",
                (
                    "Under soft conditioning, system error exceeds its exact "
                    "action-aware shadow on the identical history."
                ),
                directional_pass(
                    soft_gap_interval,
                    alternative="greater",
                    threshold=0.0,
                ),
                soft_gap_interval,
                ("soft-policy G one-sided complete-user randomization decision > 0"),
            ),
            GateCriterion(
                "independent-user-cluster-adequacy",
                (
                    "All joint-claim directional analyses use enough independent "
                    "latent-user clusters."
                ),
                (
                    None
                    if any(value is None for value in adequate_gate_3)
                    else all(value is True for value in adequate_gate_3)
                ),
                gate_3_intervals,
                "all joint-claim complete-user directional analyses are adequate",
            ),
            GateCriterion(
                "positive-policy-attribution-gap-contrast",
                (
                    "Soft conditioning increases the same-history attribution "
                    "gap relative to balanced interaction."
                ),
                directional_pass(
                    soft_balanced_gap_interval,
                    alternative="greater",
                    threshold=0.0,
                ),
                soft_balanced_gap_interval,
                (
                    "soft-minus-balanced G one-sided complete-user "
                    "randomization decision > 0"
                ),
            ),
            GateCriterion(
                "exact-shadow-terminal-error-noninferiority",
                (
                    "The soft exact shadow is practically noninferior to the "
                    "balanced exact shadow in terminal profile error."
                ),
                directional_pass(
                    selection_interval,
                    alternative="less",
                    threshold=float(selection_noninferiority_margin),
                ),
                selection_interval,
                (
                    "SelectionCost one-sided complete-user noninferiority "
                    f"decision < {float(selection_noninferiority_margin)}"
                ),
            ),
            *(
                (
                    GateCriterion(
                        "within-model-primary-multiplicity-control",
                        (
                            "The policy-conditioned-legibility conjunction is "
                            "the frozen within-model primary IUT."
                        ),
                        multiplicity_decision("policy_conditioned_legibility")[0],
                        multiplicity_decision("policy_conditioned_legibility")[1],
                        (
                            "maximum component p-value <= alpha with every "
                            "intersection-union component adequate"
                        ),
                    ),
                )
                if enforce_multiplicity
                else ()
            ),
        ),
    )
    legibility_status = gate_3.computed_status
    legibility_passed = (
        None
        if legibility_status == "incomplete"
        else legibility_status == "meets_computational_checks"
    )
    gate_3_net_harm = GateReport(
        gate_id="gate-3-net-profile-harm",
        title="Net profile harm beyond policy-conditioned legibility",
        criteria=(
            GateCriterion(
                "policy-conditioned-legibility-prerequisite",
                (
                    "The separate policy-conditioned legibility gate meets all "
                    "of its computational checks."
                ),
                legibility_passed,
                gate_3.to_dict(),
                "Gate 3 policy-conditioned legibility passes",
            ),
            GateCriterion(
                "net-profile-harm-cluster-adequacy",
                (
                    "The soft-minus-balanced updater terminal-error contrast "
                    "uses enough independent complete-user clusters."
                ),
                directional_adequate(net_harm_interval),
                net_harm_interval,
                "the net-harm complete-user analysis is adequate",
            ),
            GateCriterion(
                "net-profile-harm",
                (
                    "Soft interaction increases updater terminal profile error "
                    "beyond the frozen practical-harm margin."
                ),
                directional_pass(
                    net_harm_interval,
                    alternative="greater",
                    threshold=float(net_harm_margin),
                ),
                net_harm_interval,
                (
                    "soft-minus-balanced updater terminal-error one-sided "
                    f"decision > {float(net_harm_margin)}"
                ),
            ),
            *(
                (
                    GateCriterion(
                        "within-model-multiplicity-control",
                        (
                            "The nested net-harm claim survives the frozen "
                            "post-Gate-3 Holm family within this model."
                        ),
                        multiplicity_decision("net_profile_harm")[0],
                        multiplicity_decision("net_profile_harm")[1],
                        (
                            "Gate 3 primary IUT rejects and net harm is rejected "
                            "by the fixed three-claim Holm family"
                        ),
                    ),
                )
                if enforce_multiplicity
                else ()
            ),
        ),
    )
    return gate_2, gate_3, gate_3_net_harm


def gate_2_and_3_from_trajectories(
    rows: Iterable[Mapping[str, Any]],
    *,
    target_updater_ids: Iterable[str] | None = None,
    inferential_evidence: Mapping[str, Any] | None = None,
) -> tuple[GateReport, GateReport]:
    """Compatibility wrapper returning the historical two-report tuple."""

    gate_2, gate_3, _ = gate_2_and_3_hierarchy_from_trajectories(
        rows,
        target_updater_ids=target_updater_ids,
        inferential_evidence=inferential_evidence,
    )
    return gate_2, gate_3


def incomplete_gate(gate_id: int, title: str, reason: str) -> GateReport:
    return GateReport(
        gate_id=f"gate-{gate_id}",
        title=title,
        criteria=(
            GateCriterion(
                "not-evaluated",
                reason,
                None,
                None,
                "complete the required retained experiment",
            ),
        ),
    )
