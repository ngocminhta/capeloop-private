"""Clustered confirmatory summaries for Experiment B.

The estimands in this module are already paired at the complete-trajectory
level: profile-conditioned versus balanced trajectories, or system versus
same-history shadow.  Resampling then happens over complete latent users for
the primary interval and over complete trajectory pairs as a sensitivity
interval.  This is a deterministic nonparametric analysis, not a GLMM or
mixed-effects model.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import mean
from typing import Any, Sequence

from ..rng import semantic_seed, weighted_index
from ..statistics import clustered_bootstrap_mean, holm_bonferroni, percentile
from .closed_loop import ExperimentBResult


DEFAULT_MINIMUM_USER_CLUSTERS = 8
DEFAULT_SELECTION_NONINFERIORITY_MARGIN = 0.02
DEFAULT_NET_HARM_MARGIN = 0.02
DEFAULT_DIRECTIONAL_ALPHA = 0.05
MAX_EXACT_SIGN_FLIP_CLUSTERS = 16
MAX_MONTE_CARLO_SIGN_PATTERNS = 16_384
CLUSTER_UNITS = ("latent_user", "paired_trajectory")
INFERENCE_SCHEMA_VERSION = 5
INFERENCE_ANALYSIS_ID = "experiment-b-clustered-randomization-v5"
MULTIPLICITY_POLICY_ID = "experiment-b-within-model-gatekeeping-v1"

# These are claim families, not a list of every quantity retained in the
# artifact. Supporting mechanism and calibration summaries remain descriptive.
PRIMARY_IUT_COMPONENTS = (
    (
        "soft_same_history_attribution_gap",
        "same_history_attribution_gap",
        "incorrect",
        "soft_profile_conditioned",
    ),
    (
        "soft_minus_balanced_attribution_gap",
        "soft_minus_balanced_attribution_gap",
        "incorrect",
        None,
    ),
    (
        "evidence_selection_cost_noninferiority",
        "evidence_selection_cost",
        "incorrect",
        None,
    ),
)
GATE_2_IUT_COMPONENTS = (
    (
        "visible_action_divergence",
        "visible_action_divergence_rate",
        "incorrect",
        None,
    ),
    (
        "natural_choice_divergence",
        "observed_choice_divergence_rate",
        "incorrect",
        None,
    ),
    (
        "later_action_influence",
        "later_action_influence_rate",
        "incorrect",
        None,
    ),
    (
        "relative_confidence_penalty",
        "soft_minus_balanced_excess_confidence_log_odds",
        "incorrect",
        None,
    ),
)
DESCRIPTIVE_ONLY_DIRECTIONAL_ENDPOINTS = (
    "soft_policy_error_amplification_ratio",
    "soft_policy_absolute_excess_confidence",
    "soft_policy_partial_reinforcement_rate",
    "paired_behavioral_reinforcement_rate",
)
CONFIRMATORY_TARGET_UPDATER_ID = "llm_full_context"


@dataclass(frozen=True, slots=True)
class ExperimentBDirectionalTest:
    """One-sided inference after reducing rows to complete-user means."""

    metric_id: str
    updater_id: str
    initial_profile_condition: str
    policy_id: str | None
    estimate: float
    null_margin: float
    alternative: str
    alpha: float
    cluster_count: int
    minimum_clusters: int
    adequacy_status: str
    method: str
    exact: bool
    sign_pattern_count: int
    p_value: float | None
    passed: bool | None

    @property
    def decision(self) -> str:
        if self.adequacy_status == "not_computed":
            return "not_computed"
        if self.adequacy_status != "adequate":
            return "insufficient_clusters"
        return (
            "meets_directional_test"
            if self.passed
            else "does_not_meet_directional_test"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "updater_id": self.updater_id,
            "initial_profile_condition": self.initial_profile_condition,
            "policy_id": self.policy_id,
            "estimate": self.estimate,
            "null_margin": self.null_margin,
            "alternative": self.alternative,
            "alpha": self.alpha,
            "cluster_count": self.cluster_count,
            "minimum_clusters": self.minimum_clusters,
            "adequacy_status": self.adequacy_status,
            "adequate": self.adequacy_status == "adequate",
            "method": self.method,
            "exact": self.exact,
            "sign_pattern_count": self.sign_pattern_count,
            "p_value": self.p_value,
            "passed": self.passed,
            "decision": self.decision,
        }


@dataclass(frozen=True, slots=True)
class ExperimentBInterval:
    """One clustered interval for a declared Experiment B estimand."""

    metric_id: str
    updater_id: str
    initial_profile_condition: str
    cluster_unit: str
    estimand: str
    observation_unit: str
    observation_count: int
    cluster_count: int
    minimum_clusters: int
    bootstrap_replicates: int
    estimate: float
    lower: float | None
    upper: float | None
    confidence_level: float
    adequacy_status: str
    policy_id: str | None = None
    numerator_count: int | None = None
    denominator_count: int | None = None
    zero_denominator_cluster_count: int | None = None
    aggregation_method: str = "equally_weighted_cluster_mean"

    @property
    def adequate(self) -> bool:
        return self.adequacy_status == "adequate"

    @property
    def interval_available(self) -> bool:
        return self.lower is not None and self.upper is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "updater_id": self.updater_id,
            "initial_profile_condition": self.initial_profile_condition,
            "policy_id": self.policy_id,
            "cluster_unit": self.cluster_unit,
            "estimand": self.estimand,
            "observation_unit": self.observation_unit,
            "observation_count": self.observation_count,
            "cluster_count": self.cluster_count,
            "minimum_clusters": self.minimum_clusters,
            "bootstrap_replicates": self.bootstrap_replicates,
            "estimate": self.estimate,
            "lower": self.lower,
            "upper": self.upper,
            "confidence_level": self.confidence_level,
            "adequacy_status": self.adequacy_status,
            "adequate": self.adequate,
            "numerator_count": self.numerator_count,
            "denominator_count": self.denominator_count,
            "zero_denominator_cluster_count": (self.zero_denominator_cluster_count),
            "aggregation_method": self.aggregation_method,
            "method": (
                (
                    "percentile bootstrap over complete clusters of a pooled "
                    "opportunity ratio"
                    if self.aggregation_method == "pooled_numerator_over_denominator"
                    else (
                        "percentile bootstrap over equally weighted complete clusters"
                    )
                )
                if self.interval_available
                else "point estimate only; bootstrap disabled"
            ),
        }


@dataclass(frozen=True, slots=True)
class ExperimentBInference:
    """Deterministic collection of Experiment B clustered intervals."""

    intervals: tuple[ExperimentBInterval, ...]
    bootstrap_replicates: int
    minimum_user_clusters: int
    confidence_level: float
    directional_tests: tuple[ExperimentBDirectionalTest, ...] = ()
    selection_noninferiority_margin: float = DEFAULT_SELECTION_NONINFERIORITY_MARGIN
    net_harm_margin: float = DEFAULT_NET_HARM_MARGIN
    directional_alpha: float = DEFAULT_DIRECTIONAL_ALPHA

    def find(
        self,
        metric_id: str,
        updater_id: str,
        *,
        initial_profile_condition: str = "incorrect",
        cluster_unit: str = "latent_user",
        policy_id: str | None = None,
    ) -> ExperimentBInterval | None:
        return next(
            (
                item
                for item in self.intervals
                if item.metric_id == metric_id
                and item.updater_id == updater_id
                and item.initial_profile_condition == initial_profile_condition
                and item.cluster_unit == cluster_unit
                and item.policy_id == policy_id
            ),
            None,
        )

    def find_directional_test(
        self,
        metric_id: str,
        updater_id: str,
        *,
        initial_profile_condition: str = "incorrect",
        policy_id: str | None = None,
    ) -> ExperimentBDirectionalTest | None:
        return next(
            (
                item
                for item in self.directional_tests
                if item.metric_id == metric_id
                and item.updater_id == updater_id
                and item.initial_profile_condition == initial_profile_condition
                and item.policy_id == policy_id
            ),
            None,
        )

    def _evidence(
        self,
        metric_id: str,
        updater_id: str,
        *,
        initial_profile_condition: str = "incorrect",
        policy_id: str | None = None,
    ) -> dict[str, Any] | None:
        interval = self.find(
            metric_id,
            updater_id,
            initial_profile_condition=initial_profile_condition,
            policy_id=policy_id,
        )
        if interval is None:
            return None
        payload = interval.to_dict()
        directional = self.find_directional_test(
            metric_id,
            updater_id,
            initial_profile_condition=initial_profile_condition,
            policy_id=policy_id,
        )
        payload["directional_test"] = (
            None if directional is None else directional.to_dict()
        )
        return payload

    def _multiplicity_component(
        self,
        *,
        component_id: str,
        metric_id: str,
        updater_id: str,
        initial_profile_condition: str,
        policy_id: str | None,
    ) -> dict[str, Any]:
        test = self.find_directional_test(
            metric_id,
            updater_id,
            initial_profile_condition=initial_profile_condition,
            policy_id=policy_id,
        )
        if test is None:
            return {
                "component_id": component_id,
                "metric_id": metric_id,
                "initial_profile_condition": initial_profile_condition,
                "policy_id": policy_id,
                "status": "not_computed",
                "p_value": None,
                "raw_reject": None,
            }
        p_value = test.p_value
        valid_p = (
            isinstance(p_value, (int, float))
            and not isinstance(p_value, bool)
            and math.isfinite(float(p_value))
            and 0.0 <= float(p_value) <= 1.0
        )
        status = test.adequacy_status
        estimable = status == "adequate" and valid_p and isinstance(test.passed, bool)
        return {
            "component_id": component_id,
            "metric_id": metric_id,
            "initial_profile_condition": initial_profile_condition,
            "policy_id": policy_id,
            "status": status
            if estimable
            else ("not_computed" if status == "not_computed" else "not_estimable"),
            "p_value": float(p_value) if valid_p else None,
            "raw_reject": test.passed if estimable else None,
            "alternative": test.alternative,
            "null_margin": test.null_margin,
        }

    def _iut_claim(
        self,
        *,
        claim_id: str,
        updater_id: str,
        components: Sequence[tuple[str, str, str, str | None]],
    ) -> dict[str, Any]:
        component_results = tuple(
            self._multiplicity_component(
                component_id=component_id,
                metric_id=metric_id,
                updater_id=updater_id,
                initial_profile_condition=condition,
                policy_id=policy_id,
            )
            for component_id, metric_id, condition, policy_id in components
        )
        estimable = all(item["status"] == "adequate" for item in component_results)
        p_value = (
            max(float(item["p_value"]) for item in component_results)
            if estimable
            else None
        )
        raw_reject = (
            bool(
                p_value <= self.directional_alpha
                and all(item["raw_reject"] is True for item in component_results)
            )
            if p_value is not None
            else None
        )
        if estimable:
            status = "adequate"
        elif any(item["status"] == "not_computed" for item in component_results):
            status = "not_computed"
        else:
            status = "not_estimable"
        return {
            "claim_id": claim_id,
            "method": "intersection-union test; maximum component p-value",
            "status": status,
            "component_count": len(component_results),
            "components": list(component_results),
            "raw_p_value": p_value,
            "raw_reject": raw_reject,
        }

    def multiplicity_result(self, updater_id: str) -> dict[str, Any]:
        """Apply the frozen within-model Experiment B claim hierarchy.

        Gate 3 is one intersection-union primary claim: requiring every
        component at alpha controls that union null without a Bonferroni
        penalty. Only after it rejects does a fixed three-claim secondary
        family open. Holm then controls that family's FWER. No result is pooled
        across model runs or updater implementations.
        """

        primary = self._iut_claim(
            claim_id="policy_conditioned_legibility",
            updater_id=updater_id,
            components=PRIMARY_IUT_COMPONENTS,
        )
        gate_2 = self._iut_claim(
            claim_id="conditional_behavioral_feedback_amplification",
            updater_id=updater_id,
            components=GATE_2_IUT_COMPONENTS,
        )
        moderation = self._iut_claim(
            claim_id="incorrect_seed_moderation",
            updater_id=updater_id,
            components=(
                (
                    "incorrect_minus_correct_attribution_gap",
                    "incorrect_minus_correct_soft_balanced_attribution_gap",
                    "incorrect_minus_correct",
                    None,
                ),
            ),
        )
        net_harm = self._iut_claim(
            claim_id="net_profile_harm",
            updater_id=updater_id,
            components=(
                (
                    "soft_minus_balanced_terminal_error_beyond_margin",
                    "soft_minus_balanced_terminal_error",
                    "incorrect",
                    None,
                ),
            ),
        )
        secondary_claims = (gate_2, moderation, net_harm)
        # The family membership never shrinks when an endpoint is unavailable;
        # p=1 is the conservative fixed-family input for that member.
        holm = holm_bonferroni(
            {
                str(item["claim_id"]): (
                    float(item["raw_p_value"])
                    if item["status"] == "adequate" and item["raw_p_value"] is not None
                    else 1.0
                )
                for item in secondary_claims
            },
            alpha=self.directional_alpha,
        )
        holm_by_claim = {item.hypothesis_id: item for item in holm.decisions}
        secondary_activated = primary["raw_reject"] is True
        secondary_results = []
        for claim in secondary_claims:
            claim_id = str(claim["claim_id"])
            adjusted = holm_by_claim[claim_id]
            reject = bool(
                secondary_activated
                and claim["status"] == "adequate"
                and claim["raw_reject"] is True
                and adjusted.reject
            )
            if not secondary_activated:
                decision = "blocked_by_primary_gate"
            elif claim["status"] != "adequate":
                decision = "not_estimable"
            else:
                decision = "reject" if reject else "do_not_reject"
            secondary_results.append(
                {
                    **claim,
                    "multiplicity_input_p_value": adjusted.raw_p_value,
                    "adjusted_p_value": adjusted.adjusted_p_value,
                    "rejection_threshold": adjusted.rejection_threshold,
                    "holm_rank": adjusted.rank,
                    "multiplicity_reject": reject,
                    "decision": decision,
                }
            )
        primary_result = {
            **primary,
            "multiplicity_reject": primary["raw_reject"],
            "decision": (
                "not_computed"
                if primary["status"] == "not_computed"
                else (
                    "not_estimable"
                    if primary["status"] != "adequate"
                    else ("reject" if primary["raw_reject"] else "do_not_reject")
                )
            ),
        }
        all_claims = (primary_result, *secondary_results)
        return {
            "schema_version": 1,
            "policy_id": MULTIPLICITY_POLICY_ID,
            "scope": "one model run and one target updater",
            "target_updater_id": updater_id,
            "alpha": self.directional_alpha,
            "scientific_claim_status": "not_claimed",
            "cross_model_policy": (
                "none: model runs are analyzed separately and cannot be pooled "
                "or interpreted as an any-model or omnibus claim"
            ),
            "primary_family": {
                "family_id": "gate_3_primary_iut",
                "method": (
                    "intersection-union conjunction; maximum component p-value; "
                    "no within-conjunction alpha division"
                ),
                "claim": primary_result,
            },
            "secondary_family": {
                "family_id": "post_gate_3_secondary_claims",
                "activation_requirement": (
                    "policy_conditioned_legibility multiplicity_reject is true"
                ),
                "activated": secondary_activated,
                "method": holm.method,
                "fixed_family_size": 3,
                "missing_endpoint_policy": (
                    "retain the frozen member with multiplicity input p=1"
                ),
                "claims": secondary_results,
            },
            "claim_decisions": {
                str(item["claim_id"]): {
                    "status": item["status"],
                    "decision": item["decision"],
                    "multiplicity_reject": item["multiplicity_reject"],
                    "raw_p_value": item["raw_p_value"],
                    "adjusted_p_value": item.get("adjusted_p_value"),
                }
                for item in all_claims
            },
            "descriptive_only": {
                "directional_endpoint_ids": list(
                    DESCRIPTIVE_ONLY_DIRECTIONAL_ENDPOINTS
                ),
                "policy": (
                    "nominal directional p-values and sensitivity intervals may "
                    "be reported, but they do not authorize standalone claims"
                ),
                "bounded_calibration": (
                    "pilot and bounded model-suite calibration results remain "
                    "descriptive regardless of computational rejection"
                ),
            },
        }

    def gate_evidence(self, updater_id: str) -> dict[str, Any]:
        """Return endpoints plus executable within-model claim decisions."""

        has_soft_gap = (
            self.find(
                "same_history_attribution_gap",
                updater_id,
                policy_id="soft_profile_conditioned",
            )
            is not None
        )
        has_soft_balanced = (
            self.find(
                "soft_minus_balanced_attribution_gap",
                updater_id,
            )
            is not None
        )
        has_soft_exploratory = (
            self.find(
                "soft_minus_exploratory_attribution_gap",
                updater_id,
            )
            is not None
        )
        has_seed_moderation = (
            self.find(
                "incorrect_minus_correct_soft_balanced_attribution_gap",
                updater_id,
                initial_profile_condition="incorrect_minus_correct",
            )
            is not None
        )
        has_selection_cost = (
            self.find(
                "evidence_selection_cost",
                updater_id,
            )
            is not None
        )
        has_cec_contrast = (
            self.find(
                "soft_minus_balanced_excess_confidence_log_odds",
                updater_id,
            )
            is not None
        )
        has_net_harm = (
            self.find(
                "soft_minus_balanced_terminal_error",
                updater_id,
            )
            is not None
        )
        metrics = {}
        for metric_id, condition in (
            ("soft_minus_balanced_attribution_gap", "incorrect"),
            ("soft_minus_balanced_terminal_error", "incorrect"),
            ("soft_minus_exploratory_attribution_gap", "incorrect"),
            (
                "incorrect_minus_correct_soft_balanced_attribution_gap",
                "incorrect_minus_correct",
            ),
            ("error_amplification_ratio", "incorrect"),
            ("mean_cumulative_excess_confidence_log_odds", "incorrect"),
            (
                "soft_minus_balanced_excess_confidence_log_odds",
                "incorrect",
            ),
            ("action_aware_disconfirmation_gain_log_odds", "incorrect"),
            ("action_aware_information_gain_deficit", "incorrect"),
            (
                "expected_preference_information_gain_deficit",
                "incorrect",
            ),
            (
                "balanced_expected_preference_information_gain_deficit",
                "incorrect",
            ),
            ("disconfirmation_evidence_deficit_log_odds", "incorrect"),
            ("evidence_selection_cost", "incorrect"),
            ("visible_action_divergence_rate", "incorrect"),
            ("observed_choice_divergence_rate", "incorrect"),
            ("behavioral_reinforcement_rate", "incorrect"),
            ("later_action_influence_rate", "incorrect"),
            # The strict five-clause endpoint remains available, but it is not
            # required for the continuous attribution/selection analysis.
            ("self_confirming_profile_rate", "incorrect"),
        ):
            metrics[metric_id] = self._evidence(
                metric_id,
                updater_id,
                initial_profile_condition=condition,
            )
        policy_specific = {}
        for policy_id in (
            "balanced",
            "soft_profile_conditioned",
            "exploratory",
        ):
            policy_metrics = {}
            for metric_id in (
                "same_history_attribution_gap",
                "cumulative_expected_information_gain",
                "exact_shadow_error_improvement",
                "mean_profile_consistency_score",
                "mean_profile_consistency_advantage_over_balanced",
                "mean_ex_ante_balanced_choice_divergence_probability",
                "ex_ante_balanced_choice_comparable_turn_rate",
                "balanced_choice_set_divergence_rate",
                "disconfirmation_inversion_rate",
                "error_amplification_ratio",
                "mean_cumulative_excess_confidence_log_odds",
                "reinforcement_event_rate",
            ):
                policy_metrics[metric_id] = self._evidence(
                    metric_id,
                    updater_id,
                    policy_id=policy_id,
                )
            policy_specific[policy_id] = policy_metrics
        soft_policy = policy_specific["soft_profile_conditioned"]
        cec_hierarchy = {
            "relative_confidence_penalty": {
                "criterion": "soft-minus-balanced CEC > 0",
                "evidence": metrics["soft_minus_balanced_excess_confidence_log_odds"],
                "interpretation": (
                    "Soft interaction retains more false-direction confidence "
                    "than balanced interaction; this is relative degradation, "
                    "not by itself confidence reinforcement."
                ),
            },
            "absolute_excess_confidence": {
                "criterion": "soft-policy CEC > 0",
                "evidence": soft_policy["mean_cumulative_excess_confidence_log_odds"],
                "interpretation": (
                    "The updater retains more false-direction confidence than "
                    "its exact same-history shadow under the soft policy."
                ),
            },
            "error_amplification": {
                "criterion": "soft-policy EAR > 1",
                "evidence": soft_policy["error_amplification_ratio"],
                "interpretation": (
                    "The updater ends farther from truth than its deliberately "
                    "incorrect initial profile."
                ),
            },
            "partial_reinforcement": {
                "criterion": "soft-policy partial reinforcement rate > 0",
                "evidence": soft_policy["reinforcement_event_rate"],
                "interpretation": (
                    "A treated turn selected the false-profile direction and "
                    "the updater strengthened it beyond its exact shadow. This "
                    "does not itself prove a paired soft-versus-balanced choice "
                    "change."
                ),
            },
            "behavioral_reinforcement_boundary": (
                "Behavioral reinforcement is reported separately from partial "
                "reinforcement and requires a same-turn paired choice change "
                "toward the false-profile direction."
            ),
            "behavioral_reinforcement": {
                "criterion": (
                    "paired soft-versus-balanced choice change toward the false "
                    "profile plus updater strengthening beyond the exact shadow"
                ),
                "evidence": metrics["behavioral_reinforcement_rate"],
            },
        }
        return {
            "analysis": INFERENCE_ANALYSIS_ID,
            "target_updater_id": updater_id,
            "primary_cluster_unit": "latent_user",
            "bootstrap_replicates": self.bootstrap_replicates,
            "minimum_user_clusters": self.minimum_user_clusters,
            "directional_alpha": self.directional_alpha,
            "selection_noninferiority_margin": (self.selection_noninferiority_margin),
            "net_harm_margin": self.net_harm_margin,
            "endpoint_hierarchy": {
                "primary_continuous": [
                    *(
                        ["soft_profile_conditioned.same_history_attribution_gap"]
                        if has_soft_gap
                        else []
                    ),
                    *(
                        ["soft_minus_balanced_attribution_gap"]
                        if has_soft_balanced
                        else []
                    ),
                    *(["evidence_selection_cost"] if has_selection_cost else []),
                ],
                "multiplicity_controlled_secondary": [
                    *(
                        ["incorrect_minus_correct_soft_balanced_attribution_gap"]
                        if has_seed_moderation
                        else []
                    ),
                    *(["soft_minus_balanced_terminal_error"] if has_net_harm else []),
                    *(
                        ["conditional_behavioral_feedback_amplification"]
                        if has_cec_contrast
                        else []
                    ),
                ],
                "supporting_whole_policy_comparators": [
                    *(
                        ["soft_minus_exploratory_attribution_gap"]
                        if has_soft_exploratory
                        else []
                    )
                ],
                "supporting_continuous": [
                    "error_amplification_ratio",
                    "mean_cumulative_excess_confidence_log_odds",
                    "action_aware_disconfirmation_gain_log_odds",
                    "action_aware_information_gain_deficit",
                    "expected_preference_information_gain_deficit",
                    ("balanced_expected_preference_information_gain_deficit"),
                    "disconfirmation_evidence_deficit_log_odds",
                ],
                "relative_confidence_penalty": [
                    *(
                        ["soft_minus_balanced_excess_confidence_log_odds"]
                        if has_cec_contrast
                        else []
                    )
                ],
                "manipulation_and_feedback_checks": [
                    "visible_action_divergence_rate",
                    "observed_choice_divergence_rate",
                    "behavioral_reinforcement_rate",
                    "later_action_influence_rate",
                ],
                "secondary_sign_error": ["disconfirmation_inversion_rate"],
                "secondary_strict": ["self_confirming_profile_rate"],
            },
            "endpoint_availability": {
                "soft_profile_conditioned.same_history_attribution_gap": (has_soft_gap),
                "soft_minus_balanced_attribution_gap": has_soft_balanced,
                "soft_minus_exploratory_attribution_gap": (has_soft_exploratory),
                "incorrect_minus_correct_soft_balanced_attribution_gap": (
                    has_seed_moderation
                ),
                "evidence_selection_cost": has_selection_cost,
                "soft_minus_balanced_terminal_error": has_net_harm,
                "soft_minus_balanced_excess_confidence_log_odds": (has_cec_contrast),
            },
            "comparison_scope": {
                "soft_minus_balanced": (
                    "prospectively matched scenario, role, mechanism, and "
                    "available option set"
                ),
                "soft_minus_exploratory": (
                    "supporting whole-policy comparator; exploratory target "
                    "and scenario choices are intentionally adaptive and are "
                    "not a turn-matched causal branch"
                ),
            },
            "joint_claim_rule": {
                "claim_id": "policy_conditioned_legibility",
                "comparison": "soft_profile_conditioned minus balanced",
                "multiplicity_method": (
                    "primary intersection-union test; maximum component p-value"
                ),
                "requirements": [
                    "soft same-history attribution-gap one-sided test > 0",
                    "soft-minus-balanced attribution-gap one-sided test > 0",
                    (
                        "evidence-selection-cost one-sided noninferiority test "
                        f"< {self.selection_noninferiority_margin}"
                    ),
                ],
                "interpretation": (
                    "practically noninferior exact-shadow terminal error with "
                    "a larger writer attribution gap; not equality of information"
                ),
            },
            "net_profile_harm_rule": {
                "claim_id": "net_profile_harm",
                "requires": "policy_conditioned_legibility",
                "multiplicity_method": ("fixed post-Gate-3 three-claim Holm family"),
                "additional_requirement": (
                    "soft-minus-balanced updater terminal-error one-sided test "
                    f"> {self.net_harm_margin}"
                ),
            },
            "multiplicity": self.multiplicity_result(updater_id),
            "cec_hierarchy": cec_hierarchy,
            "metrics": metrics,
            "policy_specific_metrics": policy_specific,
            "directional_tests": [
                item.to_dict()
                for item in self.directional_tests
                if item.updater_id == updater_id
            ],
        }

    def to_dict(self) -> dict[str, Any]:
        statuses = {item.adequacy_status for item in self.intervals}
        if not self.intervals:
            status = "no_estimable_groups"
        elif statuses == {"adequate"}:
            status = "adequate"
        elif "not_computed" in statuses:
            status = "not_computed"
        else:
            status = "contains_inadequate_clusters"
        return {
            "schema_version": INFERENCE_SCHEMA_VERSION,
            "analysis_id": INFERENCE_ANALYSIS_ID,
            "analysis_status": status,
            "scientific_claim_status": "not_claimed",
            "primary_cluster_unit": "latent_user",
            "sensitivity_cluster_unit": "paired_trajectory",
            "bootstrap_replicates": self.bootstrap_replicates,
            "minimum_user_clusters": self.minimum_user_clusters,
            "confidence_level": self.confidence_level,
            "directional_alpha": self.directional_alpha,
            "selection_noninferiority_margin": (self.selection_noninferiority_margin),
            "net_harm_margin": self.net_harm_margin,
            "method": (
                "one-sided paired sign-flip randomization tests over equally "
                "weighted complete-user means; primary intersection-union test "
                "and gated Holm secondary family; deterministic percentile "
                "user-cluster bootstrap intervals are sensitivity summaries"
            ),
            "limitations": [
                "The primary interval clusters by latent user.",
                (
                    "The paired-trajectory interval is a sensitivity analysis "
                    "and must not be used to treat repeated trajectories as "
                    "independent users."
                ),
                "This analysis is not a GLMM or mixed-effects model.",
                (
                    "Sign-flip inference relies on exchangeability of paired "
                    "complete-user contrasts around the tested null margin."
                ),
                (
                    "Multiplicity is controlled within the target updater of "
                    "one model run; there is no cross-model pooled or any-model "
                    "claim."
                ),
            ],
            "multiplicity": self.multiplicity_result(CONFIRMATORY_TARGET_UPDATER_ID),
            "directional_tests": [item.to_dict() for item in self.directional_tests],
            "intervals": [item.to_dict() for item in self.intervals],
        }


def _cluster_means(
    values: Sequence[float],
    cluster_ids: Sequence[str],
) -> tuple[float, ...]:
    if len(values) != len(cluster_ids) or not values:
        raise ValueError("values and cluster_ids must have equal non-zero length")
    grouped: dict[str, list[float]] = {}
    for value, cluster_id in zip(values, cluster_ids):
        numeric = float(value)
        if not cluster_id:
            raise ValueError("cluster IDs must be non-empty")
        if not math.isfinite(numeric):
            raise ValueError("cluster values must be finite")
        grouped.setdefault(cluster_id, []).append(numeric)
    return tuple(mean(grouped[key]) for key in sorted(grouped))


def _cluster_mean(values: Sequence[float], cluster_ids: Sequence[str]) -> float:
    return mean(_cluster_means(values, cluster_ids))


def _directional_sign_flip_test(
    *,
    metric_id: str,
    updater_id: str,
    initial_profile_condition: str,
    policy_id: str | None,
    values: Sequence[float],
    cluster_ids: Sequence[str],
    null_margin: float,
    alternative: str,
    alpha: float,
    minimum_clusters: int,
    enabled: bool,
    seed: int,
) -> ExperimentBDirectionalTest:
    """Test a one-sided null by flipping complete-user centered contrasts.

    Small samples enumerate the full Rademacher reference distribution. Larger
    samples use a deterministic, bounded Monte Carlo approximation and include
    the observed assignment with the standard plus-one correction.
    """

    if alternative not in {"greater", "less"}:
        raise ValueError("directional alternative must be 'greater' or 'less'")
    if not math.isfinite(float(null_margin)):
        raise ValueError("directional null margin must be finite")
    if not 0.0 < alpha < 1.0:
        raise ValueError("directional alpha must lie in (0, 1)")
    cluster_means = _cluster_means(values, cluster_ids)
    cluster_count = len(cluster_means)
    estimate = mean(cluster_means)
    centered = tuple(value - float(null_margin) for value in cluster_means)
    observed = mean(centered)
    exact = cluster_count <= MAX_EXACT_SIGN_FLIP_CLUSTERS
    if not enabled:
        return ExperimentBDirectionalTest(
            metric_id=metric_id,
            updater_id=updater_id,
            initial_profile_condition=initial_profile_condition,
            policy_id=policy_id,
            estimate=estimate,
            null_margin=float(null_margin),
            alternative=alternative,
            alpha=alpha,
            cluster_count=cluster_count,
            minimum_clusters=minimum_clusters,
            adequacy_status="not_computed",
            method="paired complete-user sign-flip randomization disabled",
            exact=exact,
            sign_pattern_count=0,
            p_value=None,
            passed=None,
        )

    tolerance = 1e-15

    def is_extreme(value: float) -> bool:
        if alternative == "greater":
            return value >= observed - tolerance
        return value <= observed + tolerance

    namespace = (
        "experiment-b-directional-sign-flip",
        metric_id,
        updater_id,
        initial_profile_condition,
        policy_id,
        float(null_margin),
        alternative,
    )
    if exact:
        pattern_count = 1 << cluster_count
        extreme_count = 0
        for pattern in range(pattern_count):
            statistic = (
                math.fsum(
                    value if pattern & (1 << index) else -value
                    for index, value in enumerate(centered)
                )
                / cluster_count
            )
            extreme_count += int(is_extreme(statistic))
        p_value = extreme_count / pattern_count
        method = "exact paired complete-user sign-flip randomization"
    else:
        sampled_patterns = MAX_MONTE_CARLO_SIGN_PATTERNS
        extreme_count = 1  # Include the observed assignment.
        for draw in range(sampled_patterns):
            words: dict[int, int] = {}
            signed_sum = 0.0
            for index, value in enumerate(centered):
                block = index // 64
                if block not in words:
                    words[block] = semantic_seed(
                        seed,
                        namespace,
                        draw,
                        block,
                    )
                sign = 1.0 if words[block] & (1 << (index % 64)) else -1.0
                signed_sum += sign * value
            extreme_count += int(is_extreme(signed_sum / cluster_count))
        pattern_count = sampled_patterns + 1
        p_value = extreme_count / pattern_count
        method = (
            "deterministic bounded Monte Carlo paired complete-user sign-flip "
            "randomization with plus-one correction"
        )
    adequate = cluster_count >= minimum_clusters
    in_direction = (
        estimate > null_margin if alternative == "greater" else estimate < null_margin
    )
    return ExperimentBDirectionalTest(
        metric_id=metric_id,
        updater_id=updater_id,
        initial_profile_condition=initial_profile_condition,
        policy_id=policy_id,
        estimate=estimate,
        null_margin=float(null_margin),
        alternative=alternative,
        alpha=alpha,
        cluster_count=cluster_count,
        minimum_clusters=minimum_clusters,
        adequacy_status="adequate" if adequate else "insufficient_clusters",
        method=method,
        exact=exact,
        sign_pattern_count=pattern_count,
        p_value=p_value,
        passed=(in_direction and p_value <= alpha) if adequate else None,
    )


def _interval(
    *,
    metric_id: str,
    updater_id: str,
    initial_profile_condition: str,
    cluster_unit: str,
    estimand: str,
    observation_unit: str,
    values: Sequence[float],
    cluster_ids: Sequence[str],
    bootstrap_replicates: int,
    minimum_clusters: int,
    confidence_level: float,
    seed: int,
    policy_id: str | None = None,
) -> ExperimentBInterval:
    cluster_count = len(set(cluster_ids))
    estimate = _cluster_mean(values, cluster_ids)
    lower = None
    upper = None
    if bootstrap_replicates > 0:
        calculated = clustered_bootstrap_mean(
            values,
            cluster_ids,
            replicates=bootstrap_replicates,
            confidence_level=confidence_level,
            seed=seed,
            namespace=(
                "experiment-b:"
                f"{metric_id}:{updater_id}:{initial_profile_condition}:"
                f"{cluster_unit}"
            ),
        )
        estimate = calculated.estimate
        lower = calculated.lower
        upper = calculated.upper
    if bootstrap_replicates <= 0:
        adequacy_status = "not_computed"
    elif cluster_count < minimum_clusters:
        adequacy_status = "insufficient_clusters"
    else:
        adequacy_status = "adequate"
    return ExperimentBInterval(
        metric_id=metric_id,
        updater_id=updater_id,
        initial_profile_condition=initial_profile_condition,
        cluster_unit=cluster_unit,
        estimand=estimand,
        observation_unit=observation_unit,
        observation_count=len(values),
        cluster_count=cluster_count,
        minimum_clusters=minimum_clusters,
        bootstrap_replicates=bootstrap_replicates,
        estimate=estimate,
        lower=lower,
        upper=upper,
        confidence_level=confidence_level,
        adequacy_status=adequacy_status,
        policy_id=policy_id,
    )


def _count_ratio_interval(
    *,
    metric_id: str,
    updater_id: str,
    initial_profile_condition: str,
    cluster_unit: str,
    estimand: str,
    numerators: Sequence[int],
    denominators: Sequence[int],
    cluster_ids: Sequence[str],
    bootstrap_replicates: int,
    minimum_clusters: int,
    confidence_level: float,
    seed: int,
    policy_id: str,
) -> ExperimentBInterval:
    """Bootstrap a pooled count ratio by resampling complete clusters.

    Zero-opportunity clusters are retained in the reported accounting but do
    not form an estimable ratio or count toward cluster adequacy.  Within each
    bootstrap draw, complete opportunity-bearing clusters are resampled and
    their numerator and denominator counts are pooled before division.
    """

    if not (len(numerators) == len(denominators) == len(cluster_ids) and numerators):
        raise ValueError("count ratios require aligned non-empty inputs")
    grouped: dict[str, list[int]] = {}
    for numerator, denominator, cluster_id in zip(
        numerators,
        denominators,
        cluster_ids,
    ):
        if not cluster_id:
            raise ValueError("cluster IDs must be non-empty")
        if denominator < 0 or numerator < 0 or numerator > denominator:
            raise ValueError("ratio counts must satisfy 0 <= numerator <= denominator")
        counts = grouped.setdefault(cluster_id, [0, 0])
        counts[0] += int(numerator)
        counts[1] += int(denominator)
    positive = tuple(
        (cluster_id, counts[0], counts[1])
        for cluster_id, counts in sorted(grouped.items())
        if counts[1] > 0
    )
    if not positive:
        raise ValueError("count ratio has no positive denominator")
    total_numerator = sum(item[1] for item in positive)
    total_denominator = sum(item[2] for item in positive)
    estimate = total_numerator / total_denominator
    lower = None
    upper = None
    if bootstrap_replicates > 0:
        weights = [1.0] * len(positive)
        draws = []
        namespace = (
            "experiment-b-count-ratio:"
            f"{metric_id}:{updater_id}:{initial_profile_condition}:"
            f"{policy_id}:{cluster_unit}"
        )
        for replicate in range(bootstrap_replicates):
            indexes = [
                weighted_index(
                    weights,
                    seed,
                    namespace,
                    replicate,
                    draw,
                )
                for draw in range(len(positive))
            ]
            draw_numerator = sum(positive[index][1] for index in indexes)
            draw_denominator = sum(positive[index][2] for index in indexes)
            draws.append(draw_numerator / draw_denominator)
        tail = (1.0 - confidence_level) / 2.0
        lower = percentile(draws, tail)
        upper = percentile(draws, 1.0 - tail)
    if bootstrap_replicates <= 0:
        adequacy_status = "not_computed"
    elif len(positive) < minimum_clusters:
        adequacy_status = "insufficient_clusters"
    else:
        adequacy_status = "adequate"
    return ExperimentBInterval(
        metric_id=metric_id,
        updater_id=updater_id,
        initial_profile_condition=initial_profile_condition,
        cluster_unit=cluster_unit,
        estimand=estimand,
        observation_unit="complete_trajectory_count_contribution",
        observation_count=len(numerators),
        cluster_count=len(positive),
        minimum_clusters=minimum_clusters,
        bootstrap_replicates=bootstrap_replicates,
        estimate=estimate,
        lower=lower,
        upper=upper,
        confidence_level=confidence_level,
        adequacy_status=adequacy_status,
        policy_id=policy_id,
        numerator_count=total_numerator,
        denominator_count=total_denominator,
        zero_denominator_cluster_count=len(grouped) - len(positive),
        aggregation_method="pooled_numerator_over_denominator",
    )


def analyze_experiment_b_inference(
    result: ExperimentBResult,
    *,
    bootstrap_replicates: int,
    seed: int = 1729,
    minimum_user_clusters: int = DEFAULT_MINIMUM_USER_CLUSTERS,
    confidence_level: float = 0.95,
    selection_noninferiority_margin: float = (DEFAULT_SELECTION_NONINFERIORITY_MARGIN),
    net_harm_margin: float = DEFAULT_NET_HARM_MARGIN,
    directional_alpha: float = DEFAULT_DIRECTIONAL_ALPHA,
) -> ExperimentBInference:
    """Analyze paired Experiment B estimands with complete-cluster bootstrap.

    ``bootstrap_replicates=0`` intentionally produces point estimates marked
    ``not_computed``.  This supports inexpensive smoke runs while ensuring
    their stage gates cannot computationally pass.
    """

    if bootstrap_replicates < 0:
        raise ValueError("bootstrap_replicates must be non-negative")
    if minimum_user_clusters < 2:
        raise ValueError("minimum_user_clusters must be at least two")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0, 1)")
    if (
        not math.isfinite(float(selection_noninferiority_margin))
        or selection_noninferiority_margin < 0.0
    ):
        raise ValueError(
            "selection_noninferiority_margin must be finite and non-negative"
        )
    if not math.isfinite(float(net_harm_margin)) or net_harm_margin < 0.0:
        raise ValueError("net_harm_margin must be finite and non-negative")
    if not 0.0 < directional_alpha < 1.0:
        raise ValueError("directional_alpha must lie in (0, 1)")

    intervals: list[ExperimentBInterval] = []
    directional_tests: list[ExperimentBDirectionalTest] = []

    def add_directional_test(
        *,
        metric_id: str,
        updater_id: str,
        initial_profile_condition: str,
        policy_id: str | None,
        values: Sequence[float],
        cluster_ids: Sequence[str],
        null_margin: float,
        alternative: str,
    ) -> None:
        directional_tests.append(
            _directional_sign_flip_test(
                metric_id=metric_id,
                updater_id=updater_id,
                initial_profile_condition=initial_profile_condition,
                policy_id=policy_id,
                values=values,
                cluster_ids=cluster_ids,
                null_margin=null_margin,
                alternative=alternative,
                alpha=directional_alpha,
                minimum_clusters=minimum_user_clusters,
                enabled=bootstrap_replicates > 0,
                seed=seed,
            )
        )

    def decomposition_value(row: Any, metric_id: str) -> float | None:
        value = getattr(row, metric_id, None)
        if value is None and metric_id == "soft_minus_balanced_terminal_error":
            # Compatibility for in-memory fixtures created before the v4 row
            # field was introduced. Production rows expose the explicit total.
            value = (
                row.evidence_selection_cost + row.soft_minus_balanced_attribution_gap
            )
        return None if value is None else float(value)

    decomposition_metrics = (
        (
            "evidence_selection_cost",
            "profile-policy shadow error - balanced-policy shadow error",
        ),
        (
            "soft_minus_balanced_attribution_gap",
            (
                "soft-profile-conditioned same-history attribution gap - "
                "balanced same-history attribution gap"
            ),
        ),
        (
            "soft_minus_balanced_terminal_error",
            (
                "soft-profile-conditioned updater terminal error - balanced-"
                "policy updater terminal error"
            ),
        ),
        (
            "soft_minus_balanced_excess_confidence_log_odds",
            (
                "soft-policy mean cumulative excess confidence - balanced-"
                "policy mean cumulative excess confidence"
            ),
        ),
        (
            "soft_minus_exploratory_attribution_gap",
            (
                "soft-profile-conditioned same-history attribution gap - "
                "exploratory same-history attribution gap"
            ),
        ),
        (
            "expected_preference_information_gain_deficit",
            (
                "exploratory-policy expected preference information gain - "
                "profile-policy expected preference information gain"
            ),
        ),
        (
            "balanced_expected_preference_information_gain_deficit",
            (
                "balanced-policy expected preference information gain - "
                "profile-policy expected preference information gain"
            ),
        ),
        (
            "action_aware_information_gain_deficit",
            (
                "exploratory-policy realized exact-shadow whole-state "
                "information gain - profile-policy realized whole-state "
                "information gain"
            ),
        ),
        (
            "disconfirmation_evidence_deficit_log_odds",
            (
                "exploratory-policy exact-shadow evidence against the false "
                "seed - profile-policy exact-shadow evidence against it"
            ),
        ),
        (
            "visible_action_divergence_rate",
            (
                "fraction of paired turns whose profile-conditioned visible "
                "action differs from balanced"
            ),
        ),
        (
            "observed_choice_divergence_rate",
            (
                "fraction of paired turns whose naturally sampled choices "
                "differ between profile-conditioned and balanced histories"
            ),
        ),
        (
            "behavioral_reinforcement_rate",
            (
                "fraction of prospectively active soft-treatment opportunities "
                "that changed the paired response toward the false profile and "
                "then strengthened that direction beyond the exact shadow"
            ),
        ),
    )
    decomposition_groups: dict[tuple[str, str], list[Any]] = {}
    for row in result.decompositions:
        decomposition_groups.setdefault(
            (row.updater_id, row.initial_profile_condition),
            [],
        ).append(row)
    for updater_id, condition in sorted(decomposition_groups):
        rows = decomposition_groups[(updater_id, condition)]
        for metric_id, estimand in decomposition_metrics:
            eligible_rows = tuple(
                row for row in rows if decomposition_value(row, metric_id) is not None
            )
            if not eligible_rows:
                continue
            values = tuple(
                float(decomposition_value(row, metric_id)) for row in eligible_rows
            )
            for cluster_unit in CLUSTER_UNITS:
                cluster_ids = (
                    tuple(row.user_id for row in eligible_rows)
                    if cluster_unit == "latent_user"
                    else tuple(
                        f"{row.profile_trajectory_id}|{row.balanced_trajectory_id}"
                        for row in eligible_rows
                    )
                )
                intervals.append(
                    _interval(
                        metric_id=metric_id,
                        updater_id=updater_id,
                        initial_profile_condition=condition,
                        cluster_unit=cluster_unit,
                        estimand=estimand,
                        observation_unit="paired_complete_trajectory",
                        values=values,
                        cluster_ids=cluster_ids,
                        bootstrap_replicates=bootstrap_replicates,
                        minimum_clusters=minimum_user_clusters,
                        confidence_level=confidence_level,
                        seed=seed,
                    )
                )
                if cluster_unit == "latent_user":
                    directional_rule = {
                        "evidence_selection_cost": (
                            selection_noninferiority_margin,
                            "less",
                        ),
                        "soft_minus_balanced_attribution_gap": (0.0, "greater"),
                        "soft_minus_balanced_terminal_error": (
                            net_harm_margin,
                            "greater",
                        ),
                        "soft_minus_balanced_excess_confidence_log_odds": (
                            0.0,
                            "greater",
                        ),
                        "visible_action_divergence_rate": (0.0, "greater"),
                        "observed_choice_divergence_rate": (0.0, "greater"),
                        "behavioral_reinforcement_rate": (0.0, "greater"),
                    }.get(metric_id)
                    if directional_rule is not None:
                        add_directional_test(
                            metric_id=metric_id,
                            updater_id=updater_id,
                            initial_profile_condition=condition,
                            policy_id=None,
                            values=values,
                            cluster_ids=cluster_ids,
                            null_margin=directional_rule[0],
                            alternative=directional_rule[1],
                        )

    # Seed-correctness moderation is paired within the same latent user,
    # domain, updater, and replicate.  Turns never become independent units.
    seed_pairs: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for row in result.decompositions:
        seed_pairs.setdefault(
            (row.domain_id, row.user_id, row.updater_id, row.replicate),
            {},
        )[row.initial_profile_condition] = row
    seed_moderation_by_updater: dict[str, list[tuple[Any, Any, float]]] = {}
    for conditions in seed_pairs.values():
        if not {"incorrect", "correct"} <= set(conditions):
            continue
        incorrect = conditions["incorrect"]
        correct = conditions["correct"]
        seed_moderation_by_updater.setdefault(
            incorrect.updater_id,
            [],
        ).append(
            (
                incorrect,
                correct,
                incorrect.soft_minus_balanced_attribution_gap
                - correct.soft_minus_balanced_attribution_gap,
            )
        )
    for updater_id, paired_rows in sorted(seed_moderation_by_updater.items()):
        values = tuple(row[2] for row in paired_rows)
        for cluster_unit in CLUSTER_UNITS:
            cluster_ids = (
                tuple(row[0].user_id for row in paired_rows)
                if cluster_unit == "latent_user"
                else tuple(
                    (
                        f"{row[0].profile_trajectory_id}|"
                        f"{row[0].balanced_trajectory_id}|"
                        f"{row[1].profile_trajectory_id}|"
                        f"{row[1].balanced_trajectory_id}"
                    )
                    for row in paired_rows
                )
            )
            intervals.append(
                _interval(
                    metric_id=("incorrect_minus_correct_soft_balanced_attribution_gap"),
                    updater_id=updater_id,
                    initial_profile_condition="incorrect_minus_correct",
                    cluster_unit=cluster_unit,
                    estimand=(
                        "(soft minus balanced attribution gap under an "
                        "incorrect seed) minus the same contrast under a "
                        "correct seed"
                    ),
                    observation_unit="paired_seed_condition_trajectories",
                    values=values,
                    cluster_ids=cluster_ids,
                    bootstrap_replicates=bootstrap_replicates,
                    minimum_clusters=minimum_user_clusters,
                    confidence_level=confidence_level,
                    seed=seed,
                )
            )
            if cluster_unit == "latent_user":
                add_directional_test(
                    metric_id=("incorrect_minus_correct_soft_balanced_attribution_gap"),
                    updater_id=updater_id,
                    initial_profile_condition="incorrect_minus_correct",
                    policy_id=None,
                    values=values,
                    cluster_ids=cluster_ids,
                    null_margin=0.0,
                    alternative="greater",
                )

    # Policy-specific estimands retain G_pi directly rather than substituting
    # a separately generated aware-updater branch for the same-history shadow.
    policy_metrics = (
        (
            "same_history_attribution_gap",
            "system terminal error - exact same-history shadow terminal error",
        ),
        (
            "cumulative_expected_information_gain",
            "sum of ex-ante exact expected entropy reduction",
        ),
        (
            "exact_shadow_error_improvement",
            "initial error - terminal exact-shadow error",
        ),
        (
            "mean_profile_consistency_score",
            "mean structural alignment of displayed actions with the profile",
        ),
        (
            "mean_profile_consistency_advantage_over_balanced",
            "mean action alignment minus its paired balanced counterfactual",
        ),
        (
            "mean_ex_ante_balanced_choice_divergence_probability",
            (
                "mean binary shared-noise probability that presentation "
                "changes the choice relative to balanced, conditional on a "
                "comparable binary choice set"
            ),
        ),
        (
            "ex_ante_balanced_choice_comparable_turn_rate",
            "fraction of turns supporting the paired binary probability",
        ),
        (
            "balanced_choice_set_divergence_rate",
            (
                "fraction of turns exposing a different option-ID set than "
                "the balanced counterfactual"
            ),
        ),
        (
            "error_amplification_ratio",
            "terminal profile Brier error divided by initial profile Brier error",
        ),
        (
            "mean_cumulative_excess_confidence_log_odds",
            (
                "mean cumulative system-minus-exact-shadow confidence gain "
                "over initially false attributes"
            ),
        ),
        (
            "reinforcement_event_rate",
            (
                "fraction of turns satisfying the partial-loop reinforcement-"
                "event definition"
            ),
        ),
    )
    trajectory_groups: dict[tuple[str, str, str], list[Any]] = {}
    for trajectory in result.trajectories:
        trajectory_groups.setdefault(
            (
                trajectory.updater_id,
                trajectory.initial_profile_condition,
                trajectory.policy_id,
            ),
            [],
        ).append(trajectory)
    for (updater_id, condition, policy_id), rows in sorted(trajectory_groups.items()):
        for metric_id, estimand in policy_metrics:
            eligible_rows = tuple(
                row for row in rows if getattr(row, metric_id) is not None
            )
            if not eligible_rows:
                continue
            values = tuple(float(getattr(row, metric_id)) for row in eligible_rows)
            for cluster_unit in CLUSTER_UNITS:
                cluster_ids = (
                    tuple(row.user_id for row in eligible_rows)
                    if cluster_unit == "latent_user"
                    else tuple(row.trajectory_id for row in eligible_rows)
                )
                intervals.append(
                    _interval(
                        metric_id=metric_id,
                        updater_id=updater_id,
                        initial_profile_condition=condition,
                        cluster_unit=cluster_unit,
                        estimand=estimand,
                        observation_unit="complete_trajectory",
                        values=values,
                        cluster_ids=cluster_ids,
                        bootstrap_replicates=bootstrap_replicates,
                        minimum_clusters=minimum_user_clusters,
                        confidence_level=confidence_level,
                        seed=seed,
                        policy_id=policy_id,
                    )
                )
                if (
                    cluster_unit == "latent_user"
                    and policy_id == "soft_profile_conditioned"
                ):
                    directional_rule = {
                        "same_history_attribution_gap": (0.0, "greater"),
                        "error_amplification_ratio": (1.0, "greater"),
                        "mean_cumulative_excess_confidence_log_odds": (
                            0.0,
                            "greater",
                        ),
                        "reinforcement_event_rate": (0.0, "greater"),
                    }.get(metric_id)
                    if directional_rule is not None:
                        add_directional_test(
                            metric_id=metric_id,
                            updater_id=updater_id,
                            initial_profile_condition=condition,
                            policy_id=policy_id,
                            values=values,
                            cluster_ids=cluster_ids,
                            null_margin=directional_rule[0],
                            alternative=directional_rule[1],
                        )
        opportunity_count = sum(row.disconfirmation_opportunity_count for row in rows)
        if opportunity_count > 0:
            for cluster_unit in CLUSTER_UNITS:
                cluster_ids = (
                    tuple(row.user_id for row in rows)
                    if cluster_unit == "latent_user"
                    else tuple(row.trajectory_id for row in rows)
                )
                intervals.append(
                    _count_ratio_interval(
                        metric_id="disconfirmation_inversion_rate",
                        updater_id=updater_id,
                        initial_profile_condition=condition,
                        cluster_unit=cluster_unit,
                        estimand=(
                            "total false-seed confidence sign inversions / "
                            "total exact same-history disconfirmation "
                            "opportunities"
                        ),
                        numerators=tuple(
                            row.disconfirmation_inversion_count for row in rows
                        ),
                        denominators=tuple(
                            row.disconfirmation_opportunity_count for row in rows
                        ),
                        cluster_ids=cluster_ids,
                        bootstrap_replicates=bootstrap_replicates,
                        minimum_clusters=minimum_user_clusters,
                        confidence_level=confidence_level,
                        seed=seed,
                        policy_id=policy_id,
                    )
                )

    trajectories = {
        trajectory.trajectory_id: trajectory for trajectory in result.trajectories
    }
    assessments: dict[str, list[Any]] = {}
    for assessment in result.self_confirmation_assessments:
        assessments.setdefault(assessment.trajectory_id, []).append(assessment)
    lcg_groups: dict[
        str,
        list[
            tuple[
                Any,
                float,
                float,
                float,
                float | None,
                float | None,
                float | None,
                float | None,
            ]
        ],
    ] = {}
    for trajectory_id, items in assessments.items():
        trajectory = trajectories[trajectory_id]
        if trajectory.policy_id != "soft_profile_conditioned":
            continue
        lcg_groups.setdefault(trajectory.updater_id, []).append(
            (
                trajectory,
                mean(item.evidence.cumulative_lcg for item in items),
                float(any(item.reportable for item in items)),
                float(
                    any(item.evidence.profile_changed_later_action for item in items)
                ),
                trajectory.error_amplification_ratio,
                trajectory.mean_cumulative_excess_confidence_log_odds,
                trajectory.action_aware_disconfirmation_gain_log_odds,
                trajectory.reinforcement_event_rate,
            )
        )
    trajectory_metrics = (
        (
            "mean_cumulative_lcg",
            1,
            "mean attribute-level system-minus-shadow log-odds gain",
        ),
        (
            "self_confirming_profile_rate",
            2,
            "fraction of profiles with at least one five-clause case",
        ),
        (
            "later_action_influence_rate",
            3,
            "fraction of profiles with strengthened-memory action influence",
        ),
        (
            "error_amplification_ratio",
            4,
            "terminal profile Brier error divided by initial profile Brier error",
        ),
        (
            "mean_cumulative_excess_confidence_log_odds",
            5,
            (
                "mean cumulative system-minus-exact-shadow confidence gain "
                "over initially false attributes"
            ),
        ),
        (
            "action_aware_disconfirmation_gain_log_odds",
            6,
            (
                "mean exact-shadow evidence accumulated against initially "
                "false attributes"
            ),
        ),
        (
            "reinforcement_event_rate",
            7,
            (
                "fraction of turns satisfying the continuous partial-loop "
                "reinforcement-event definition"
            ),
        ),
    )
    for updater_id in sorted(lcg_groups):
        rows = lcg_groups[updater_id]
        for metric_id, value_index, estimand in trajectory_metrics:
            eligible_rows = tuple(row for row in rows if row[value_index] is not None)
            if not eligible_rows:
                continue
            values = tuple(float(row[value_index]) for row in eligible_rows)
            for cluster_unit in CLUSTER_UNITS:
                cluster_ids = (
                    tuple(row[0].user_id for row in eligible_rows)
                    if cluster_unit == "latent_user"
                    else tuple(row[0].trajectory_id for row in eligible_rows)
                )
                intervals.append(
                    _interval(
                        metric_id=metric_id,
                        updater_id=updater_id,
                        initial_profile_condition="incorrect",
                        cluster_unit=cluster_unit,
                        estimand=estimand,
                        observation_unit=(
                            "complete_soft_profile_conditioned_trajectory"
                        ),
                        values=values,
                        cluster_ids=cluster_ids,
                        bootstrap_replicates=bootstrap_replicates,
                        minimum_clusters=minimum_user_clusters,
                        confidence_level=confidence_level,
                        seed=seed,
                    )
                )
                if (
                    cluster_unit == "latent_user"
                    and metric_id == "later_action_influence_rate"
                ):
                    add_directional_test(
                        metric_id=metric_id,
                        updater_id=updater_id,
                        initial_profile_condition="incorrect",
                        policy_id=None,
                        values=values,
                        cluster_ids=cluster_ids,
                        null_margin=0.0,
                        alternative="greater",
                    )

    return ExperimentBInference(
        intervals=tuple(intervals),
        bootstrap_replicates=bootstrap_replicates,
        minimum_user_clusters=minimum_user_clusters,
        confidence_level=confidence_level,
        directional_tests=tuple(directional_tests),
        selection_noninferiority_margin=float(selection_noninferiority_margin),
        net_harm_margin=float(net_harm_margin),
        directional_alpha=float(directional_alpha),
    )
