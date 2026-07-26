from __future__ import annotations

import unittest

from cape_loop.gates import gate_1_from_rows, gate_2_and_3_from_trajectories
from cape_loop.power import benjamini_hochberg, paired_pilot_power


class GateTests(unittest.TestCase):
    def test_gate_reports_remain_result_claim_free(self) -> None:
        rows = []
        for domain in ("travel", "writing"):
            for mechanism in ("default", "suggested"):
                rows.extend(
                    [
                        {
                            "response_mode": "naturally_sampled",
                            "domain": domain,
                            "mechanism": mechanism,
                            "updater_id": "fitted_action_aware",
                            "brier": 0.1,
                        },
                        {
                            "response_mode": "naturally_sampled",
                            "domain": domain,
                            "mechanism": mechanism,
                            "updater_id": "fitted_action_unaware",
                            "brier": 0.2,
                        },
                        {
                            "response_mode": "naturally_sampled",
                            "domain": domain,
                            "mechanism": mechanism,
                            "updater_id": "full_context_blind",
                            "brier": 0.2,
                        },
                    ]
                )
        report = gate_1_from_rows(
            rows,
            held_out_paraphrase_verified=True,
        )
        self.assertEqual(report.computed_status, "meets_computational_checks")
        self.assertEqual(report.claim_status, "not_claimed")

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

        def interval(metric_id: str, lower: float) -> dict[str, object]:
            return {
                "metric_id": metric_id,
                "cluster_unit": "latent_user",
                "cluster_count": 8,
                "minimum_clusters": 8,
                "adequacy_status": "adequate",
                "adequate": True,
                "lower": lower,
                "upper": lower + 0.2,
            }

        gate_2, gate_3 = gate_2_and_3_from_trajectories(
            [row],
            inferential_evidence={
                "metrics": {
                    "mean_cumulative_lcg": interval(
                        "mean_cumulative_lcg",
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
                }
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
                    "mean_cumulative_lcg": inadequate,
                    "self_confirming_profile_rate": adequate,
                    "profile_attribution_cost": crossing,
                }
            },
        )
        self.assertEqual(gate_2.computed_status, "does_not_meet_checks")
        self.assertEqual(gate_3.computed_status, "does_not_meet_checks")


class PowerTests(unittest.TestCase):
    def test_power_and_multiplicity_are_deterministic(self) -> None:
        first = paired_pilot_power(
            [0.2, 0.3, 0.1, 0.25], [8], simulations=40, seed=3
        )
        second = paired_pilot_power(
            [0.2, 0.3, 0.1, 0.25], [8], simulations=40, seed=3
        )
        self.assertEqual(first, second)
        decisions = benjamini_hochberg({"a": 0.001, "b": 0.02, "c": 0.8})
        self.assertTrue(decisions["a"])
        self.assertFalse(decisions["c"])


if __name__ == "__main__":
    unittest.main()
