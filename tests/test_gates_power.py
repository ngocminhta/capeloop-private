from __future__ import annotations

import unittest

from cape_loop.gates import (
    gate_1_from_rows,
    gate_2_and_3_from_trajectories,
    gate_2_and_3_hierarchy_from_trajectories,
)
from cape_loop.power import benjamini_hochberg, paired_pilot_power


class GateTests(unittest.TestCase):
    def test_gate_reports_remain_result_claim_free(self) -> None:
        rows = []
        for domain in ("travel", "writing"):
            warranted_updates = {
                "balanced": 0.20,
                "restricted": 0.45,
                "ranking": 0.30,
                "default": 0.35,
                "suggested": 0.40,
            }
            for mechanism, warranted_update in warranted_updates.items():
                rows.append(
                    {
                        "trial_id": (
                            f"{domain}:user-1:prior-flat:{mechanism}:controlled_anchor"
                        ),
                        "user_id": "user-1",
                        "response_mode": "controlled_anchor",
                        "domain": domain,
                        "target_attribute": 0,
                        "anchor_direction": 1,
                        "prior_stratum": "flat",
                        "prior_strength": 0.0,
                        "mechanism": mechanism,
                        "updater_id": "exact_action_aware",
                        "exact_acue": 0.0,
                        "anchor_directional_log_odds_update": warranted_update,
                        "exact_anchor_directional_log_odds_update": (warranted_update),
                    }
                )
        report = gate_1_from_rows(
            rows,
            same_response_audit_passed=True,
            held_out_paraphrase_ready=True,
        )
        self.assertEqual(report.computed_status, "meets_computational_checks")
        self.assertEqual(report.claim_status, "not_claimed")
        self.assertNotIn(
            "aware-beats-unaware",
            {criterion.criterion_id for criterion in report.criteria},
        )
        self.assertNotIn(
            "exact-oracle-miscalibration",
            {criterion.criterion_id for criterion in report.criteria},
        )

    def test_closed_loop_gate_requires_complete_clauses(self) -> None:
        row = {
            "policy_id": "soft_profile_conditioned",
            "initial_profile": "incorrect",
            "updater_id": "full-context",
            "mechanisms": ["ranking", "default", "suggestion"],
            "counter_profile_available": True,
            "cumulative_lcg": 0.4,
            "profile_changed_later_action": True,
            "is_self_confirming": True,
            "attribution_cost": 0.2,
        }
        gate_2, gate_3 = gate_2_and_3_from_trajectories([row])
        self.assertEqual(gate_2.computed_status, "incomplete")
        self.assertEqual(gate_3.computed_status, "incomplete")

        def interval(
            metric_id: str,
            lower: float,
            upper: float | None = None,
        ) -> dict[str, object]:
            return {
                "metric_id": metric_id,
                "cluster_unit": "latent_user",
                "cluster_count": 8,
                "minimum_clusters": 8,
                "adequacy_status": "adequate",
                "adequate": True,
                "lower": lower,
                "upper": lower + 0.2 if upper is None else upper,
            }

        gate_2, gate_3 = gate_2_and_3_from_trajectories(
            [row],
            inferential_evidence={
                "metrics": {
                    "error_amplification_ratio": interval(
                        "error_amplification_ratio",
                        1.1,
                    ),
                    "mean_cumulative_excess_confidence_log_odds": interval(
                        "mean_cumulative_excess_confidence_log_odds",
                        0.1,
                    ),
                    "soft_minus_balanced_excess_confidence_log_odds": interval(
                        "soft_minus_balanced_excess_confidence_log_odds",
                        0.1,
                    ),
                    "action_aware_information_gain_deficit": interval(
                        "action_aware_information_gain_deficit",
                        0.1,
                    ),
                    "disconfirmation_evidence_deficit_log_odds": interval(
                        "disconfirmation_evidence_deficit_log_odds",
                        0.1,
                    ),
                    "visible_action_divergence_rate": interval(
                        "visible_action_divergence_rate",
                        0.1,
                    ),
                    "observed_choice_divergence_rate": interval(
                        "observed_choice_divergence_rate",
                        0.1,
                    ),
                    "later_action_influence_rate": interval(
                        "later_action_influence_rate",
                        0.1,
                    ),
                    "self_confirming_profile_rate": interval(
                        "self_confirming_profile_rate",
                        0.05,
                    ),
                    "profile_attribution_cost": interval(
                        "profile_attribution_cost",
                        0.08,
                    ),
                    "soft_minus_balanced_attribution_gap": interval(
                        "soft_minus_balanced_attribution_gap",
                        0.08,
                    ),
                    "evidence_selection_cost": interval(
                        "evidence_selection_cost",
                        -0.25,
                        -0.05,
                    ),
                },
                "policy_specific_metrics": {
                    "soft_profile_conditioned": {
                        "same_history_attribution_gap": interval(
                            "same_history_attribution_gap",
                            0.08,
                        ),
                    }
                },
            },
        )
        self.assertEqual(gate_2.computed_status, "meets_computational_checks")
        self.assertEqual(gate_3.computed_status, "meets_computational_checks")
        self.assertEqual(gate_2.claim_status, "not_claimed")

    def test_closed_loop_gate_rejects_inadequate_or_crossing_zero_interval(
        self,
    ) -> None:
        row = {
            "policy_id": "soft_profile_conditioned",
            "initial_profile": "incorrect",
            "updater_id": "llm_full_context",
            "mechanisms": ["ranking"],
            "counter_profile_available": True,
            "cumulative_lcg": 0.4,
            "profile_changed_later_action": True,
            "is_self_confirming": True,
            "attribution_cost": 0.2,
        }
        adequate = {
            "adequacy_status": "adequate",
            "adequate": True,
            "lower": 0.1,
            "upper": 0.3,
        }
        inadequate = {
            "adequacy_status": "insufficient_clusters",
            "adequate": False,
            "lower": 0.1,
            "upper": 0.3,
        }
        crossing = {
            "adequacy_status": "adequate",
            "adequate": True,
            "lower": -0.01,
            "upper": 0.3,
        }
        gate_2, gate_3 = gate_2_and_3_from_trajectories(
            [row],
            inferential_evidence={
                "metrics": {
                    "error_amplification_ratio": adequate,
                    "mean_cumulative_excess_confidence_log_odds": inadequate,
                    "soft_minus_balanced_excess_confidence_log_odds": (inadequate),
                    "action_aware_information_gain_deficit": adequate,
                    "disconfirmation_evidence_deficit_log_odds": adequate,
                    "visible_action_divergence_rate": crossing,
                    "observed_choice_divergence_rate": adequate,
                    "later_action_influence_rate": adequate,
                    "self_confirming_profile_rate": adequate,
                    "profile_attribution_cost": crossing,
                    "soft_minus_balanced_attribution_gap": crossing,
                    "evidence_selection_cost": crossing,
                },
                "policy_specific_metrics": {
                    "soft_profile_conditioned": {
                        "same_history_attribution_gap": crossing,
                    }
                },
            },
        )
        self.assertEqual(gate_2.computed_status, "does_not_meet_checks")
        self.assertEqual(gate_3.computed_status, "does_not_meet_checks")

    def test_legibility_and_net_harm_are_separate_randomization_gates(
        self,
    ) -> None:
        row = {
            "policy_id": "soft_profile_conditioned",
            "initial_profile": "incorrect",
            "updater_id": "llm_full_context",
            "mechanisms": ["ranking"],
            "counter_profile_available": True,
            "is_self_confirming": False,
        }

        def evidence(
            metric_id: str,
            *,
            passed: bool,
            estimate: float,
            margin: float,
            alternative: str,
        ) -> dict[str, object]:
            return {
                "metric_id": metric_id,
                "cluster_unit": "latent_user",
                "cluster_count": 8,
                "minimum_clusters": 8,
                "adequacy_status": "adequate",
                "adequate": True,
                "estimate": estimate,
                "lower": estimate - 0.20,
                "upper": estimate + 0.20,
                "directional_test": {
                    "adequacy_status": "adequate",
                    "adequate": True,
                    "alternative": alternative,
                    "null_margin": margin,
                    "p_value": 1 / 256,
                    "passed": passed,
                },
            }

        metrics = {
            "soft_minus_balanced_attribution_gap": evidence(
                "soft_minus_balanced_attribution_gap",
                passed=True,
                estimate=0.10,
                margin=0.0,
                alternative="greater",
            ),
            "evidence_selection_cost": evidence(
                "evidence_selection_cost",
                passed=True,
                estimate=0.01,
                margin=0.02,
                alternative="less",
            ),
            "soft_minus_balanced_terminal_error": evidence(
                "soft_minus_balanced_terminal_error",
                passed=False,
                estimate=0.015,
                margin=0.02,
                alternative="greater",
            ),
        }
        policy_metrics = {
            "soft_profile_conditioned": {
                "same_history_attribution_gap": evidence(
                    "same_history_attribution_gap",
                    passed=True,
                    estimate=0.12,
                    margin=0.0,
                    alternative="greater",
                )
            }
        }
        _, legibility, net_harm = gate_2_and_3_hierarchy_from_trajectories(
            [row],
            inferential_evidence={
                "selection_noninferiority_margin": 0.02,
                "net_harm_margin": 0.02,
                "metrics": metrics,
                "policy_specific_metrics": policy_metrics,
            },
        )
        self.assertEqual(
            legibility.computed_status,
            "meets_computational_checks",
        )
        self.assertEqual(
            net_harm.computed_status,
            "does_not_meet_checks",
        )
        net_criterion = next(
            item
            for item in net_harm.criteria
            if item.criterion_id == "net-profile-harm"
        )
        self.assertFalse(net_criterion.passed)

        metrics["soft_minus_balanced_terminal_error"] = evidence(
            "soft_minus_balanced_terminal_error",
            passed=True,
            estimate=0.08,
            margin=0.02,
            alternative="greater",
        )
        _, legibility, net_harm = gate_2_and_3_hierarchy_from_trajectories(
            [row],
            inferential_evidence={
                "selection_noninferiority_margin": 0.02,
                "net_harm_margin": 0.02,
                "metrics": metrics,
                "policy_specific_metrics": policy_metrics,
            },
        )
        self.assertEqual(
            legibility.computed_status,
            "meets_computational_checks",
        )
        self.assertEqual(
            net_harm.computed_status,
            "meets_computational_checks",
        )

        metrics["evidence_selection_cost"]["directional_test"][  # type: ignore[index]
            "alternative"
        ] = "greater"
        _, legibility, _ = gate_2_and_3_hierarchy_from_trajectories(
            [row],
            inferential_evidence={
                "selection_noninferiority_margin": 0.02,
                "net_harm_margin": 0.02,
                "metrics": metrics,
                "policy_specific_metrics": policy_metrics,
            },
        )
        self.assertEqual(
            legibility.computed_status,
            "does_not_meet_checks",
        )

        with self.assertRaisesRegex(ValueError, "must be finite"):
            gate_2_and_3_hierarchy_from_trajectories(
                [row],
                inferential_evidence={
                    "selection_noninferiority_margin": float("nan"),
                    "net_harm_margin": 0.02,
                    "metrics": metrics,
                    "policy_specific_metrics": policy_metrics,
                },
            )

        with self.assertRaisesRegex(ValueError, "does not match the frozen"):
            gate_2_and_3_hierarchy_from_trajectories(
                [row],
                selection_noninferiority_margin=0.03,
                net_harm_margin=0.02,
                inferential_evidence={
                    "selection_noninferiority_margin": 0.02,
                    "net_harm_margin": 0.02,
                    "metrics": metrics,
                    "policy_specific_metrics": policy_metrics,
                },
            )
        with self.assertRaisesRegex(ValueError, "does not match the frozen"):
            gate_2_and_3_hierarchy_from_trajectories(
                [row],
                selection_noninferiority_margin=0.02,
                net_harm_margin=0.03,
                inferential_evidence={
                    "selection_noninferiority_margin": 0.02,
                    "net_harm_margin": 0.02,
                    "metrics": metrics,
                    "policy_specific_metrics": policy_metrics,
                },
            )

    def test_v5_gates_require_the_frozen_multiplicity_decisions(self) -> None:
        row = {
            "policy_id": "soft_profile_conditioned",
            "initial_profile": "incorrect",
            "updater_id": "llm_full_context",
            "mechanisms": ["ranking"],
            "counter_profile_available": True,
        }

        def evidence(metric_id: str, alternative: str, margin: float) -> dict:
            return {
                "metric_id": metric_id,
                "adequacy_status": "adequate",
                "adequate": True,
                "lower": 0.1,
                "upper": 0.2,
                "directional_test": {
                    "adequacy_status": "adequate",
                    "adequate": True,
                    "alternative": alternative,
                    "null_margin": margin,
                    "p_value": 0.01,
                    "passed": True,
                },
            }

        metrics = {
            "visible_action_divergence_rate": evidence(
                "visible_action_divergence_rate", "greater", 0.0
            ),
            "observed_choice_divergence_rate": evidence(
                "observed_choice_divergence_rate", "greater", 0.0
            ),
            "later_action_influence_rate": evidence(
                "later_action_influence_rate", "greater", 0.0
            ),
            "soft_minus_balanced_excess_confidence_log_odds": evidence(
                "soft_minus_balanced_excess_confidence_log_odds", "greater", 0.0
            ),
            "soft_minus_balanced_attribution_gap": evidence(
                "soft_minus_balanced_attribution_gap", "greater", 0.0
            ),
            "evidence_selection_cost": evidence(
                "evidence_selection_cost", "less", 0.02
            ),
            "soft_minus_balanced_terminal_error": evidence(
                "soft_minus_balanced_terminal_error", "greater", 0.02
            ),
        }
        policy_metrics = {
            "soft_profile_conditioned": {
                "same_history_attribution_gap": evidence(
                    "same_history_attribution_gap", "greater", 0.0
                )
            }
        }
        decisions = {
            "policy_conditioned_legibility": {
                "status": "adequate",
                "multiplicity_reject": True,
            },
            "conditional_behavioral_feedback_amplification": {
                "status": "adequate",
                "multiplicity_reject": False,
            },
            "net_profile_harm": {
                "status": "adequate",
                "multiplicity_reject": False,
            },
        }
        gate_2, gate_3, net_harm = gate_2_and_3_hierarchy_from_trajectories(
            [row],
            inferential_evidence={
                "analysis": "experiment-b-clustered-randomization-v5",
                "selection_noninferiority_margin": 0.02,
                "net_harm_margin": 0.02,
                "metrics": metrics,
                "policy_specific_metrics": policy_metrics,
                "multiplicity": {"claim_decisions": decisions},
            },
        )
        self.assertEqual(gate_3.computed_status, "meets_computational_checks")
        self.assertEqual(gate_2.computed_status, "does_not_meet_checks")
        self.assertEqual(net_harm.computed_status, "does_not_meet_checks")


class PowerTests(unittest.TestCase):
    def test_power_and_multiplicity_are_deterministic(self) -> None:
        first = paired_pilot_power([0.2, 0.3, 0.1, 0.25], [8], simulations=40, seed=3)
        second = paired_pilot_power([0.2, 0.3, 0.1, 0.25], [8], simulations=40, seed=3)
        self.assertEqual(first, second)
        decisions = benjamini_hochberg({"a": 0.001, "b": 0.02, "c": 0.8})
        self.assertTrue(decisions["a"])
        self.assertFalse(decisions["c"])


if __name__ == "__main__":
    unittest.main()
