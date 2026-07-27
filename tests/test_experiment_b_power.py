from __future__ import annotations

from types import SimpleNamespace
import unittest

from cape_loop.power import (
    EXPERIMENT_B_POWER_MAXIMUM_SIMULATIONS,
    bounded_experiment_b_simulations,
    experiment_b_pilot_interactions,
    experiment_b_pilot_power,
    format_experiment_b_power_summary,
)


TARGET = "full_context_blind"
REFERENCE = "fitted_action_aware"
TREATED_POLICY = "soft_profile_conditioned"
REFERENCE_POLICY = "balanced"
FOCAL_PROFILE = "incorrect"
REFERENCE_PROFILE = "correct"


def _pilot_rows(
    effects: dict[str, float],
) -> tuple[SimpleNamespace, ...]:
    rows = []
    for user_id, effect in effects.items():
        for domain_id in ("travel", "writing"):
            for replicate in range(2):
                for updater_id in (TARGET, REFERENCE):
                    for policy_id in (TREATED_POLICY, REFERENCE_POLICY):
                        for initial_profile in (
                            FOCAL_PROFILE,
                            REFERENCE_PROFILE,
                        ):
                            terminal_error = 0.20
                            if (
                                updater_id == TARGET
                                and policy_id == TREATED_POLICY
                                and initial_profile == FOCAL_PROFILE
                            ):
                                terminal_error += effect
                            trajectory_id = (
                                f"experiment-b:{domain_id}:{user_id}:"
                                f"{initial_profile}:replicate-{replicate}:"
                                f"{policy_id}:{updater_id}"
                            )
                            rows.append(
                                SimpleNamespace(
                                    trajectory_id=trajectory_id,
                                    crn_key=(
                                        f"experiment-b:{domain_id}:{user_id}:"
                                        f"{initial_profile}:"
                                        f"replicate-{replicate}"
                                    ),
                                    user_id=user_id,
                                    domain_id=domain_id,
                                    updater_id=updater_id,
                                    policy_id=policy_id,
                                    initial_profile_condition=initial_profile,
                                    terminal_error=terminal_error,
                                )
                            )
    return tuple(rows)


class ExperimentBPowerTests(unittest.TestCase):
    def test_complete_user_three_way_interaction_is_exact_and_stable(self) -> None:
        effects = {
            "test-user-0001": 0.02,
            "test-user-0002": -0.01,
            "test-user-0003": 0.05,
        }
        rows = _pilot_rows(effects)
        first = experiment_b_pilot_interactions(
            rows,
            target_updater_id=TARGET,
        )
        second = experiment_b_pilot_interactions(
            tuple(reversed(rows)),
            target_updater_id=TARGET,
        )

        self.assertEqual(first, second)
        self.assertEqual(first.contributing_trajectory_count, 96)
        self.assertEqual(first.excluded_users, ())
        self.assertEqual(
            {
                row.user_id: row.stratum_count
                for row in first.eligible_users
            },
            {user_id: 4 for user_id in effects},
        )
        for row in first.eligible_users:
            self.assertAlmostEqual(
                row.interaction,
                effects[row.user_id],
            )
        self.assertEqual(len(first.pilot_input_sha256), 64)

    def test_incomplete_crossed_user_is_reported_not_partially_used(self) -> None:
        rows = list(
            _pilot_rows(
                {
                    "test-user-complete": 0.03,
                    "test-user-incomplete": 0.04,
                }
            )
        )
        rows = [
            row
            for row in rows
            if not (
                row.user_id == "test-user-incomplete"
                and row.domain_id == "travel"
                and row.crn_key.endswith("replicate-0")
                and row.updater_id == TARGET
                and row.policy_id == TREATED_POLICY
                and row.initial_profile_condition == FOCAL_PROFILE
            )
        ]
        pilot = experiment_b_pilot_interactions(
            rows,
            target_updater_id=TARGET,
        )

        self.assertEqual(
            [row.user_id for row in pilot.eligible_users],
            ["test-user-complete"],
        )
        self.assertEqual(pilot.excluded_users[0][0], "test-user-incomplete")
        self.assertIn("1 required cell(s) missing", pilot.excluded_users[0][1])

    def test_power_artifact_is_deterministic_bounded_and_reports_mc_error(
        self,
    ) -> None:
        rows = _pilot_rows(
            {
                "test-user-0001": 0.01,
                "test-user-0002": 0.03,
                "test-user-0003": -0.01,
                "test-user-0004": 0.05,
            }
        )
        first = experiment_b_pilot_power(
            rows,
            target_updater_id=TARGET,
            sample_sizes=(4, 8),
            simulations=40,
            seed=19,
        )
        second = experiment_b_pilot_power(
            rows,
            target_updater_id=TARGET,
            sample_sizes=(4, 8),
            simulations=40,
            seed=19,
        )

        self.assertEqual(first, second)
        self.assertEqual(first["status"], "estimated_from_configured_pilot")
        self.assertEqual(first["artifact_role"], "pilot_design_evidence")
        self.assertEqual(first["scientific_claim_status"], "not_claimed")
        self.assertEqual(first["pilot"]["eligible_user_count"], 4)
        self.assertEqual(
            [point["sample_size"] for point in first["points"]],
            [4, 8],
        )
        for point in first["points"]:
            self.assertGreaterEqual(point["monte_carlo_lower"], 0.0)
            self.assertLessEqual(point["monte_carlo_upper"], 1.0)
            self.assertGreaterEqual(point["monte_carlo_standard_error"], 0.0)
        rendered = format_experiment_b_power_summary(first)
        self.assertIn("Updater", first["estimand"])
        self.assertIn("95% MC interval", rendered)
        self.assertIn("not empirical evidence", rendered)

        self.assertEqual(bounded_experiment_b_simulations(0), 200)
        self.assertEqual(
            bounded_experiment_b_simulations(
                EXPERIMENT_B_POWER_MAXIMUM_SIMULATIONS + 1
            ),
            EXPERIMENT_B_POWER_MAXIMUM_SIMULATIONS,
        )
        with self.assertRaisesRegex(ValueError, "simulations"):
            experiment_b_pilot_power(
                rows,
                target_updater_id=TARGET,
                simulations=EXPERIMENT_B_POWER_MAXIMUM_SIMULATIONS + 1,
            )

    def test_missing_frozen_target_is_explicitly_not_estimable(self) -> None:
        payload = experiment_b_pilot_power(
            (),
            target_updater_id=None,
            simulations=20,
            sample_sizes=(4,),
        )

        self.assertEqual(payload["status"], "not_estimable")
        self.assertIsNone(payload["decision"]["selected_user_count"])
        self.assertEqual(payload["points"], [])
        self.assertIn(
            "neither llm_full_context nor full_context_blind",
            payload["reason"],
        )


if __name__ == "__main__":
    unittest.main()
