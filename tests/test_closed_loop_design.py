from __future__ import annotations

import json
from pathlib import Path
import unittest

from cape_loop.domains import domain_for_split, get_domain
from cape_loop.experiments.closed_loop_design import (
    ManipulationPlanError,
    build_experiment_b_manipulation_plan,
    run_offline_manipulation_audit,
)
from cape_loop.experiments.closed_loop import run_trajectory
from cape_loop.policies import BalancedPolicy, SoftProfileConditionedPolicy
from cape_loop.population import initial_profile_belief
from cape_loop.response import RandomUtilityModel
from cape_loop.scenarios import ScenarioCatalog, materialize_context
from cape_loop.schemas import LatentUser, Susceptibility
from cape_loop.updaters import ExactActionAwareUpdater, NoUpdateUpdater


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = REPOSITORY_ROOT / "data" / "scenarios" / "scenario-catalog-v1.json"


class ProspectiveExperimentBPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = ScenarioCatalog.parse(
            json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        )
        cls.domain = domain_for_split(get_domain("travel"), "test")
        cls.user = LatentUser(
            "planned-user",
            (1, -2, 1),
            Susceptibility(ranking=0.45, default=0.45, suggestion=0.45),
        )
        cls.response_model = RandomUtilityModel(
            beta=1.0,
            ranking_scale=0.35,
            default_scale=0.75,
            suggestion_scale=0.65,
        )

    def build(self, *, condition: str = "incorrect", fail_closed: bool = True):
        return build_experiment_b_manipulation_plan(
            users=(self.user,),
            domains=(self.domain,),
            scenario_catalog=self.catalog,
            response_model=self.response_model,
            initial_profile_conditions=(condition,),
            turns=6,
            trajectories_per_cell=1,
            seed=1729,
            fail_closed=fail_closed,
        )

    def test_six_turn_plan_guarantees_declared_manipulation_roles(self) -> None:
        plan = self.build()
        self.assertTrue(plan.readiness.ready)
        trajectory = plan.trajectories[0]
        informative = tuple(
            turn
            for turn in trajectory.turns
            if turn.role == "informative_active"
        )
        controls = tuple(
            turn
            for turn in trajectory.turns
            if turn.role == "decisive_active_control"
        )

        self.assertEqual(len(trajectory.turns), 6)
        self.assertGreaterEqual(len(informative), 2)
        self.assertGreaterEqual(len(controls), 1)
        self.assertGreaterEqual(len(trajectory.active_mechanisms), 2)
        self.assertTrue(
            all(
                turn.balanced_choice_margin_stratum
                in {"near_tie", "marginal"}
                and turn.soft_visible_divergence_required
                and turn.predicted_shared_noise_choice_divergence_probability
                >= 0.02
                for turn in informative
            )
        )
        self.assertTrue(
            all(
                turn.balanced_choice_margin_stratum == "decisive"
                and turn.predicted_shared_noise_choice_divergence_probability
                <= 0.05
                for turn in controls
            )
        )
        self.assertTrue(
            all(
                len(turn.directional_choice_divergence_probabilities) == 2
                and {direction for direction, _ in (
                    turn.directional_choice_divergence_probabilities
                )} == {-1, 1}
                for turn in (*informative, *controls)
            )
        )
        self.assertTrue(
            all(
                turn.counter_profile_option_retained
                and turn.retained_preference_directions == (-1, 1)
                for turn in trajectory.turns
            )
        )
        adaptive = tuple(
            turn
            for turn in trajectory.turns
            if turn.role == "adaptive_observation"
        )
        self.assertTrue(adaptive)
        self.assertTrue(
            all(
                turn.mechanism == "adaptive"
                and turn.predicted_shared_noise_choice_divergence_probability
                is None
                and not turn.soft_visible_divergence_required
                for turn in adaptive
            )
        )
        expected_asm = sum(
            min(
                probability
                for _, probability in (
                    turn.directional_choice_divergence_probabilities
                )
            )
            for turn in trajectory.turns
            if turn.active
        )
        self.assertAlmostEqual(
            trajectory.active_susceptibility_mass,
            expected_asm,
        )

    def test_plan_is_deterministic_queryable_and_json_serializable(self) -> None:
        first = self.build()
        second = self.build()
        self.assertEqual(first.to_dict(), second.to_dict())
        trajectory = first.trajectories[0]
        self.assertEqual(
            first.turn(trajectory.shared_pair_key, 2),
            trajectory.turns[2],
        )
        decoded = json.loads(first.canonical_json())
        self.assertTrue(decoded["readiness"]["outcome_blind"])
        self.assertEqual(
            decoded["readiness"]["forbidden_admission_inputs"],
            [
                "realized_choice",
                "updated_profile",
                "evaluated_model_output",
            ],
        )

    def test_correct_and_incorrect_conditions_share_one_frozen_schedule(self) -> None:
        plan = build_experiment_b_manipulation_plan(
            users=(self.user,),
            domains=(self.domain,),
            scenario_catalog=self.catalog,
            response_model=self.response_model,
            initial_profile_conditions=("correct", "incorrect"),
            turns=6,
            trajectories_per_cell=1,
            seed=1729,
        )
        correct, incorrect = plan.trajectories
        self.assertEqual(correct.schedule_group_key, incorrect.schedule_group_key)
        self.assertNotEqual(correct.shared_pair_key, incorrect.shared_pair_key)
        self.assertEqual(
            tuple(
                (
                    turn.turn,
                    turn.scenario_id,
                    turn.target_attribute,
                    turn.role,
                    turn.mechanism,
                    turn.predicted_shared_noise_choice_divergence_probability,
                    turn.directional_choice_divergence_probabilities,
                )
                for turn in correct.turns
            ),
            tuple(
                (
                    turn.turn,
                    turn.scenario_id,
                    turn.target_attribute,
                    turn.role,
                    turn.mechanism,
                    turn.predicted_shared_noise_choice_divergence_probability,
                    turn.directional_choice_divergence_probabilities,
                )
                for turn in incorrect.turns
            ),
        )
        self.assertTrue(
            plan.readiness.to_dict()["checks"][
                "condition_invariant_scenario_role_mechanism_schedule"
            ]
        )
        for turn in range(6):
            correct_belief = initial_profile_belief(self.user.theta, "correct")
            incorrect_belief = initial_profile_belief(
                self.user.theta,
                "incorrect",
            )
            correct_action = BalancedPolicy(prospective_plan=plan).action(
                self.domain,
                correct_belief,
                turn=turn,
                master_seed=1729,
                trajectory_id=correct.shared_pair_key,
            )
            incorrect_action = BalancedPolicy(prospective_plan=plan).action(
                self.domain,
                incorrect_belief,
                turn=turn,
                master_seed=1729,
                trajectory_id=incorrect.shared_pair_key,
            )
            self.assertEqual(
                correct_action.context.ranking,
                incorrect_action.context.ranking,
            )
            self.assertEqual(
                correct_action.provenance.random_seed,
                incorrect_action.provenance.random_seed,
            )
            correct_soft = SoftProfileConditionedPolicy(
                prospective_plan=plan
            ).action(
                self.domain,
                correct_belief,
                turn=turn,
                master_seed=1729,
                trajectory_id=correct.shared_pair_key,
            )
            incorrect_soft = SoftProfileConditionedPolicy(
                prospective_plan=plan
            ).action(
                self.domain,
                incorrect_belief,
                turn=turn,
                master_seed=1729,
                trajectory_id=incorrect.shared_pair_key,
            )
            self.assertEqual(
                correct_soft.provenance.random_seed,
                incorrect_soft.provenance.random_seed,
            )
            self.assertEqual(
                correct_soft.provenance.profile_conditioned,
                incorrect_soft.provenance.profile_conditioned,
            )
            self.assertEqual(
                correct_soft.provenance.presentation_mechanism,
                incorrect_soft.provenance.presentation_mechanism,
            )

        exact = ExactActionAwareUpdater(
            self.response_model,
            (self.user.susceptibility,),
        )
        paired = tuple(
            run_trajectory(
                user=self.user,
                domain=self.domain,
                policy=BalancedPolicy(prospective_plan=plan),
                updater=NoUpdateUpdater(),
                turns=6,
                seed=1729,
                initial_profile_condition=condition,
                response_model=self.response_model,
                shadow_updater=exact,
                trajectory_id=f"{row.shared_pair_key}:balanced:no-update",
                crn_key=row.shared_pair_key,
                scenario_catalog=self.catalog,
            )
            for condition, row in (
                ("correct", correct),
                ("incorrect", incorrect),
            )
        )
        self.assertEqual(
            tuple(turn.common_noise_key for turn in paired[0].turns),
            tuple(turn.common_noise_key for turn in paired[1].turns),
        )
        self.assertEqual(
            tuple(turn.action_signature for turn in paired[0].turns),
            tuple(turn.action_signature for turn in paired[1].turns),
        )
        self.assertEqual(
            tuple(turn.selected_option_id for turn in paired[0].turns),
            tuple(turn.selected_option_id for turn in paired[1].turns),
        )
    def test_directionless_profile_fails_closed_before_execution(self) -> None:
        with self.assertRaises(ManipulationPlanError) as captured:
            self.build(condition="empty")
        rejected = captured.exception.plan
        self.assertFalse(rejected.readiness.ready)
        self.assertEqual(rejected.readiness.admitted_trajectory_count, 0)

        audit_only = self.build(condition="empty", fail_closed=False)
        self.assertFalse(audit_only.readiness.ready)
        self.assertTrue(
            audit_only.trajectories[0].readiness_failures
        )

    def test_minimum_asm_is_a_trajectory_gate_not_raw_user_susceptibility(self) -> None:
        with self.assertRaises(ManipulationPlanError) as captured:
            build_experiment_b_manipulation_plan(
                users=(self.user,),
                domains=(self.domain,),
                scenario_catalog=self.catalog,
                response_model=self.response_model,
                initial_profile_conditions=("incorrect",),
                turns=6,
                requirements={"minimum_active_susceptibility_mass": 1.0},
            )
        self.assertLess(
            captured.exception.plan.trajectories[0].active_susceptibility_mass,
            1.0,
        )

    def test_active_candidate_remains_valid_after_current_profile_flips(self) -> None:
        plan = self.build(condition="incorrect")
        trajectory = plan.trajectories[0]
        instruction = next(
            turn for turn in trajectory.turns if turn.active and turn.turn > 0
        )
        current = initial_profile_belief(self.user.theta, "correct")
        soft = SoftProfileConditionedPolicy(prospective_plan=plan).action(
            self.domain,
            current,
            turn=instruction.turn,
            master_seed=1729,
            trajectory_id=trajectory.shared_pair_key,
        )
        balanced = BalancedPolicy(prospective_plan=plan).action(
            self.domain,
            current,
            turn=instruction.turn,
            master_seed=1729,
            trajectory_id=trajectory.shared_pair_key,
        )
        scenario = self.catalog.scenario(instruction.scenario_id)
        soft_context = materialize_context(soft.context, scenario)
        balanced_context = materialize_context(balanced.context, scenario)
        actual_promoted = (
            soft_context.default_option_id or soft_context.suggested_option_id
        )
        current_direction = (
            -1
            if current.expected_theta()[instruction.target_attribute] < 0.0
            else 1
        )
        self.assertNotEqual(current_direction, instruction.planned_profile_direction)
        self.assertNotEqual(actual_promoted, instruction.promoted_option_id)
        self.assertGreater(
            soft_context.option(actual_promoted).features[
                instruction.target_attribute
            ]
            * current_direction,
            0.0,
        )
        soft_probabilities = self.response_model.probability_map(
            self.user.theta,
            self.user.susceptibility,
            soft_context,
        )
        balanced_probabilities = self.response_model.probability_map(
            self.user.theta,
            self.user.susceptibility,
            balanced_context,
        )
        reference = sorted(soft_probabilities)[0]
        actual_divergence = abs(
            soft_probabilities[reference] - balanced_probabilities[reference]
        )
        declared = (
            instruction.predicted_shared_noise_choice_divergence_probability
        )
        self.assertIsNotNone(declared)
        if instruction.role == "informative_active":
            self.assertGreaterEqual(actual_divergence, declared)
        else:
            self.assertLessEqual(actual_divergence, declared)

    def test_required_active_turn_uses_and_logs_frozen_neutral_fallback(self) -> None:
        plan = self.build(condition="incorrect")
        planned = plan.trajectories[0]
        neutral = initial_profile_belief(self.user.theta, "empty")
        unplanned = SoftProfileConditionedPolicy().action(
            self.domain,
            neutral,
            turn=0,
            master_seed=1729,
            trajectory_id="unplanned-neutral",
        )
        self.assertFalse(unplanned.provenance.profile_conditioned)
        trajectory = run_trajectory(
            user=self.user,
            domain=self.domain,
            policy=SoftProfileConditionedPolicy(prospective_plan=plan),
            updater=NoUpdateUpdater(),
            turns=6,
            seed=1729,
            initial_belief=neutral,
            initial_profile_condition="incorrect",
            response_model=self.response_model,
            shadow_updater=ExactActionAwareUpdater(
                self.response_model,
                (self.user.susceptibility,),
            ),
            trajectory_id=f"{planned.shared_pair_key}:soft:no-update",
            crn_key=planned.shared_pair_key,
            scenario_catalog=self.catalog,
        )
        active = tuple(
            turn
            for turn in trajectory.turns
            if turn.prospective_manipulation_role
            in {"informative_active", "decisive_active_control"}
        )
        self.assertTrue(active)
        self.assertTrue(
            all(turn.prospective_execution_matched is True for turn in active)
        )
        self.assertTrue(
            all(
                turn.prospective_direction_source
                == "frozen_initial_profile_fallback"
                and turn.prospective_effective_profile_direction
                == plan.turn(planned.shared_pair_key, turn.turn).planned_profile_direction
                for turn in active
            )
        )

    def test_offline_audit_executes_plan_without_model_outputs(self) -> None:
        plan = self.build()
        audit = run_offline_manipulation_audit(
            plan=plan,
            users=(self.user,),
            domains=(self.domain,),
            scenario_catalog=self.catalog,
            response_model=self.response_model,
            response_seed_count=1,
        )
        self.assertEqual(audit["llm_calls"], 0)
        self.assertFalse(audit["evaluated_model_outputs_used"])
        self.assertEqual(audit["paired_trajectory_draw_count"], 1)
        self.assertTrue(audit["required_active_execution"]["all_matched"])
        self.assertIn("incorrect", audit["by_initial_profile_condition"])
        self.assertIn("travel", audit["by_domain"])
        self.assertEqual(
            audit["by_prospective_role"]["informative_active"][
                "simulated_turn_draw_count"
            ],
            2,
        )
        self.assertEqual(
            audit["by_prospective_role"]["decisive_active_control"][
                "simulated_turn_draw_count"
            ],
            1,
        )
        self.assertEqual(
            audit["exact_profile_state_driver"]["updater_id"],
            "exact_action_aware",
        )
        self.assertEqual(
            audit["behavioral_reinforcement"]["status"],
            "not_evaluated",
        )
        changed_user = LatentUser(
            self.user.user_id,
            self.user.theta,
            Susceptibility(ranking=0.15, default=0.15, suggestion=0.15),
        )
        with self.assertRaisesRegex(ValueError, "latent users differ"):
            run_offline_manipulation_audit(
                plan=plan,
                users=(changed_user,),
                domains=(self.domain,),
                scenario_catalog=self.catalog,
                response_model=self.response_model,
                response_seed_count=1,
            )
        class ExactUpdaterIdImpostor:
            updater_id = "exact_action_aware"

        with self.assertRaisesRegex(TypeError, "ExactActionAwareUpdater"):
            run_offline_manipulation_audit(
                plan=plan,
                users=(self.user,),
                domains=(self.domain,),
                scenario_catalog=self.catalog,
                response_model=self.response_model,
                response_seed_count=1,
                shadow_updater=ExactUpdaterIdImpostor(),
            )
        mismatched_model = RandomUtilityModel(
            beta=0.5,
            ranking_scale=self.response_model.ranking_scale,
            default_scale=self.response_model.default_scale,
            suggestion_scale=self.response_model.suggestion_scale,
        )
        with self.assertRaisesRegex(ValueError, "response model differs"):
            run_offline_manipulation_audit(
                plan=plan,
                users=(self.user,),
                domains=(self.domain,),
                scenario_catalog=self.catalog,
                response_model=self.response_model,
                response_seed_count=1,
                shadow_updater=ExactActionAwareUpdater(
                    mismatched_model,
                    (self.user.susceptibility,),
                ),
            )
        with self.assertRaisesRegex(ValueError, "susceptibility support"):
            run_offline_manipulation_audit(
                plan=plan,
                users=(self.user,),
                domains=(self.domain,),
                scenario_catalog=self.catalog,
                response_model=self.response_model,
                response_seed_count=1,
                shadow_updater=ExactActionAwareUpdater(
                    self.response_model,
                    (Susceptibility(),),
                ),
            )

    def test_offline_audit_strata_reconcile_with_pooled_counts(self) -> None:
        writing = domain_for_split(get_domain("writing"), "test")
        plan = build_experiment_b_manipulation_plan(
            users=(self.user,),
            domains=(self.domain, writing),
            scenario_catalog=self.catalog,
            response_model=self.response_model,
            initial_profile_conditions=("correct", "incorrect"),
            turns=6,
            trajectories_per_cell=1,
            seed=1729,
        )
        audit = run_offline_manipulation_audit(
            plan=plan,
            users=(self.user,),
            domains=(self.domain, writing),
            scenario_catalog=self.catalog,
            response_model=self.response_model,
            response_seed_count=1,
        )
        self.assertEqual(audit["selection_cost"]["count"], 4)
        self.assertTrue(
            all(
                summary["selection_cost"]["count"] == 2
                for summary in audit["by_initial_profile_condition"].values()
            )
        )
        self.assertTrue(
            all(
                summary["selection_cost"]["count"] == 2
                for summary in audit["by_domain"].values()
            )
        )
        self.assertTrue(
            all(
                summary["selection_cost"]["count"] == 1
                for domains in audit["by_condition_and_domain"].values()
                for summary in domains.values()
            )
        )
        grouped_mean = sum(
            summary["selection_cost"]["mean"]
            for summary in audit["by_initial_profile_condition"].values()
        ) / 2.0
        self.assertAlmostEqual(grouped_mean, audit["selection_cost"]["mean"])
        roles = audit["by_prospective_role"]
        self.assertEqual(
            roles["informative_active"]["simulated_turn_draw_count"],
            8,
        )
        self.assertEqual(
            roles["decisive_active_control"]["simulated_turn_draw_count"],
            4,
        )
        self.assertEqual(
            roles["adaptive_observation"]["simulated_turn_draw_count"],
            12,
        )
        self.assertEqual(
            sum(item["simulated_turn_draw_count"] for item in roles.values()),
            24,
        )
        for role in ("informative_active", "decisive_active_control"):
            self.assertEqual(
                roles[role]["visible_action_divergence_rate"]["mean"],
                1.0,
            )
            self.assertTrue(roles[role]["required_execution"]["all_matched"])
        crosstab = audit["prospective_active_turn_crosstab"]
        self.assertTrue(crosstab["counts_reconcile"])
        self.assertEqual(crosstab["simulated_turn_draw_count"], 12)
        self.assertEqual(
            sum(row["simulated_turn_draw_count"] for row in crosstab["rows"]),
            audit["required_active_execution"]["instruction_count"],
        )
        self.assertTrue(
            all(
                row["role"]
                in {"informative_active", "decisive_active_control"}
                and row["mechanism"] in {"default", "suggestion"}
                and row["effective_profile_direction"] in {-1, 1}
                and row["planned_initial_profile_direction"] in {-1, 1}
                and row["target_attribute"] in {0, 1, 2}
                and row["domain"] in {"travel", "writing"}
                for row in crosstab["rows"]
            )
        )
        self.assertFalse(
            roles["adaptive_observation"]["required_execution"]["applicable"]
        )
        self.assertIsNone(
            roles["adaptive_observation"]["required_execution"]["all_matched"]
        )
if __name__ == "__main__":
    unittest.main()
