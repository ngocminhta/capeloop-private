from __future__ import annotations

from types import SimpleNamespace
import unittest

from cape_loop.config import AppConfig, ExperimentSection
from cape_loop.experiments.closed_loop import DecompositionRow
from cape_loop.experiments.closed_loop_inference import (
    analyze_experiment_b_inference,
)


def _synthetic_result(users: int) -> SimpleNamespace:
    decompositions = []
    trajectories = []
    assessments = []
    for index in range(users):
        user_id = f"user-{index:02d}"
        profile_id = f"profile-{index:02d}"
        balanced_id = f"balanced-{index:02d}"
        decompositions.append(
            DecompositionRow(
                domain_id="travel",
                user_id=user_id,
                initial_profile_condition="incorrect",
                updater_id="llm_full_context",
                replicate=0,
                profile_trajectory_id=profile_id,
                balanced_trajectory_id=balanced_id,
                evidence_selection_cost=0.10 + index / 1000,
                profile_attribution_cost=0.25 + index / 1000,
                balanced_attribution_cost=0.05,
                self_confirmation_interaction=0.20 + index / 1000,
                soft_minus_balanced_excess_confidence_log_odds=(0.15 + index / 1000),
                exploratory_attribution_cost=0.10,
                expected_preference_information_gain_deficit=0.08,
                balanced_expected_preference_information_gain_deficit=-0.02,
            )
        )
        decompositions.append(
            DecompositionRow(
                domain_id="travel",
                user_id=user_id,
                initial_profile_condition="correct",
                updater_id="llm_full_context",
                replicate=0,
                profile_trajectory_id=f"correct-profile-{index:02d}",
                balanced_trajectory_id=f"correct-balanced-{index:02d}",
                evidence_selection_cost=0.01,
                profile_attribution_cost=0.10,
                balanced_attribution_cost=0.05,
                self_confirmation_interaction=0.05,
                soft_minus_balanced_excess_confidence_log_odds=0.01,
                exploratory_attribution_cost=0.08,
            )
        )
        trajectories.append(
            SimpleNamespace(
                trajectory_id=profile_id,
                user_id=user_id,
                updater_id="llm_full_context",
                policy_id="soft_profile_conditioned",
                initial_profile_condition="incorrect",
                same_history_attribution_gap=0.25 + index / 1000,
                cumulative_expected_information_gain=0.40,
                exact_shadow_error_improvement=0.30,
                mean_profile_consistency_score=0.50,
                mean_profile_consistency_advantage_over_balanced=0.25,
                mean_ex_ante_balanced_choice_divergence_probability=0.10,
                ex_ante_balanced_choice_comparable_turn_rate=1.0,
                balanced_choice_set_divergence_rate=0.0,
                disconfirmation_inversion_rate=0.20,
                disconfirmation_opportunity_count=5,
                disconfirmation_inversion_count=1,
                error_amplification_ratio=1.20 + index / 1000,
                mean_cumulative_excess_confidence_log_odds=(0.30 + index / 1000),
                action_aware_disconfirmation_gain_log_odds=(0.15 + index / 1000),
                reinforcement_event_rate=0.25,
            )
        )
        for attribute in range(3):
            assessments.append(
                SimpleNamespace(
                    trajectory_id=profile_id,
                    reportable=True,
                    evidence=SimpleNamespace(
                        cumulative_lcg=0.30 + attribute / 100,
                        profile_changed_later_action=True,
                    ),
                )
            )
    return SimpleNamespace(
        decompositions=tuple(decompositions),
        trajectories=tuple(trajectories),
        self_confirmation_assessments=tuple(assessments),
    )


class ExperimentBInferenceTests(unittest.TestCase):
    def test_closed_loop_config_allows_confirmatory_bootstraps(self) -> None:
        config = AppConfig(
            experiment=ExperimentSection(
                kind="closed_loop",
                mechanisms=("ranking", "default", "suggestion"),
                response_modes=("naturally_sampled",),
                policies=("balanced", "soft_profile_conditioned"),
                bootstrap_replicates=2_000,
            )
        )
        config.validate_experiment_contract()

    def test_user_and_trajectory_clustered_intervals_are_deterministic(
        self,
    ) -> None:
        result = _synthetic_result(8)
        first = analyze_experiment_b_inference(
            result,
            bootstrap_replicates=80,
            seed=91,
        )
        second = analyze_experiment_b_inference(
            result,
            bootstrap_replicates=80,
            seed=91,
        )
        self.assertEqual(first.to_dict(), second.to_dict())
        artifact = first.to_dict()
        self.assertEqual(artifact["schema_version"], 5)
        self.assertEqual(
            artifact["analysis_id"],
            "experiment-b-clustered-randomization-v5",
        )
        self.assertEqual(artifact["selection_noninferiority_margin"], 0.02)
        self.assertEqual(artifact["net_harm_margin"], 0.02)

        attribution = first.find(
            "same_history_attribution_gap",
            "llm_full_context",
            policy_id="soft_profile_conditioned",
        )
        self.assertIsNotNone(attribution)
        assert attribution is not None
        self.assertTrue(attribution.adequate)
        self.assertEqual(attribution.cluster_count, 8)
        self.assertGreater(attribution.lower, 0.0)

        policy_gap = first.find(
            "same_history_attribution_gap",
            "llm_full_context",
            policy_id="soft_profile_conditioned",
        )
        self.assertIsNotNone(policy_gap)
        explicit_contrast = first.find(
            "soft_minus_balanced_attribution_gap",
            "llm_full_context",
        )
        self.assertIsNotNone(explicit_contrast)
        total_effect = first.find(
            "soft_minus_balanced_terminal_error",
            "llm_full_context",
        )
        self.assertIsNotNone(total_effect)
        assert total_effect is not None
        self.assertAlmostEqual(total_effect.estimate, 0.307)

        attribution_test = first.find_directional_test(
            "soft_minus_balanced_attribution_gap",
            "llm_full_context",
        )
        self.assertIsNotNone(attribution_test)
        assert attribution_test is not None
        self.assertTrue(attribution_test.exact)
        self.assertEqual(attribution_test.sign_pattern_count, 256)
        self.assertAlmostEqual(attribution_test.p_value, 1 / 256)
        self.assertTrue(attribution_test.passed)
        selection_test = first.find_directional_test(
            "evidence_selection_cost",
            "llm_full_context",
        )
        self.assertIsNotNone(selection_test)
        assert selection_test is not None
        self.assertEqual(selection_test.alternative, "less")
        self.assertEqual(selection_test.null_margin, 0.02)
        self.assertFalse(selection_test.passed)
        cec_contrast = first.find(
            "soft_minus_balanced_excess_confidence_log_odds",
            "llm_full_context",
        )
        self.assertIsNotNone(cec_contrast)
        assert cec_contrast is not None
        self.assertGreater(cec_contrast.lower, 0.0)
        seed_moderation = first.find(
            "incorrect_minus_correct_soft_balanced_attribution_gap",
            "llm_full_context",
            initial_profile_condition="incorrect_minus_correct",
        )
        self.assertIsNotNone(seed_moderation)
        seed_moderation_test = first.find_directional_test(
            "incorrect_minus_correct_soft_balanced_attribution_gap",
            "llm_full_context",
            initial_profile_condition="incorrect_minus_correct",
        )
        self.assertIsNotNone(seed_moderation_test)
        assert seed_moderation_test is not None
        self.assertEqual(seed_moderation_test.alternative, "greater")
        self.assertEqual(seed_moderation_test.null_margin, 0.0)
        self.assertTrue(seed_moderation_test.exact)

        trajectory = first.find(
            "evidence_selection_cost",
            "llm_full_context",
            cluster_unit="paired_trajectory",
        )
        self.assertIsNotNone(trajectory)
        assert trajectory is not None
        self.assertEqual(trajectory.observation_unit, "paired_complete_trajectory")
        self.assertEqual(trajectory.cluster_count, 8)

        lcg = first.find("mean_cumulative_lcg", "llm_full_context")
        rate = first.find(
            "self_confirming_profile_rate",
            "llm_full_context",
        )
        self.assertIsNotNone(lcg)
        self.assertIsNotNone(rate)
        assert lcg is not None and rate is not None
        self.assertGreater(lcg.lower, 0.0)
        self.assertGreater(rate.lower, 0.0)

        gate_evidence = first.gate_evidence("llm_full_context")
        self.assertNotIn("mean_cumulative_lcg", gate_evidence["metrics"])
        hierarchy = gate_evidence["cec_hierarchy"]
        self.assertIn("relative_confidence_penalty", hierarchy)
        self.assertIn("absolute_excess_confidence", hierarchy)
        self.assertIn("error_amplification", hierarchy)
        self.assertIn("partial_reinforcement", hierarchy)
        endpoints = gate_evidence["endpoint_hierarchy"]
        self.assertIn(
            "soft_profile_conditioned.same_history_attribution_gap",
            endpoints["primary_continuous"],
        )
        self.assertNotIn(
            "soft_minus_exploratory_attribution_gap",
            endpoints["primary_continuous"],
        )
        self.assertIn(
            "soft_minus_exploratory_attribution_gap",
            endpoints["supporting_whole_policy_comparators"],
        )
        soft_metrics = gate_evidence["policy_specific_metrics"][
            "soft_profile_conditioned"
        ]
        self.assertIsNotNone(soft_metrics["mean_cumulative_excess_confidence_log_odds"])
        self.assertIsNotNone(soft_metrics["error_amplification_ratio"])
        self.assertIsNotNone(soft_metrics["reinforcement_event_rate"])
        multiplicity = gate_evidence["multiplicity"]
        self.assertEqual(
            multiplicity["policy_id"],
            "experiment-b-within-model-gatekeeping-v1",
        )
        self.assertEqual(
            multiplicity["primary_family"]["claim"]["raw_p_value"],
            1.0,
        )
        self.assertFalse(multiplicity["secondary_family"]["activated"])
        self.assertTrue(
            all(
                item["decision"] == "blocked_by_primary_gate"
                for item in multiplicity["secondary_family"]["claims"]
            )
        )

    def test_frozen_secondary_family_uses_holm_after_primary_iut(self) -> None:
        inference = analyze_experiment_b_inference(
            _synthetic_result(8),
            bootstrap_replicates=20,
            selection_noninferiority_margin=0.15,
            net_harm_margin=0.25,
        )
        result = inference.multiplicity_result("llm_full_context")
        primary = result["primary_family"]["claim"]
        self.assertEqual(
            primary["method"], ("intersection-union test; maximum component p-value")
        )
        self.assertEqual(primary["component_count"], 3)
        self.assertTrue(primary["multiplicity_reject"])
        secondary = result["secondary_family"]
        self.assertTrue(secondary["activated"])
        self.assertEqual(secondary["fixed_family_size"], 3)
        decisions = {item["claim_id"]: item for item in secondary["claims"]}
        self.assertTrue(decisions["incorrect_seed_moderation"]["multiplicity_reject"])
        self.assertTrue(decisions["net_profile_harm"]["multiplicity_reject"])
        self.assertFalse(
            decisions["conditional_behavioral_feedback_amplification"][
                "multiplicity_reject"
            ]
        )
        self.assertAlmostEqual(
            decisions["incorrect_seed_moderation"]["adjusted_p_value"],
            3 / 256,
        )
        self.assertIn("none:", result["cross_model_policy"])

    def test_small_cluster_and_disabled_bootstrap_are_not_adequate(self) -> None:
        small = analyze_experiment_b_inference(
            _synthetic_result(7),
            bootstrap_replicates=20,
        )
        interval = small.find(
            "soft_minus_balanced_attribution_gap",
            "llm_full_context",
        )
        self.assertIsNotNone(interval)
        assert interval is not None
        self.assertEqual(interval.adequacy_status, "insufficient_clusters")
        self.assertFalse(interval.adequate)

        disabled = analyze_experiment_b_inference(
            _synthetic_result(8),
            bootstrap_replicates=0,
        )
        no_interval = disabled.find(
            "evidence_selection_cost",
            "llm_full_context",
        )
        self.assertIsNotNone(no_interval)
        assert no_interval is not None
        self.assertEqual(no_interval.adequacy_status, "not_computed")
        self.assertIsNone(no_interval.lower)
        self.assertEqual(
            disabled.to_dict()["analysis_status"],
            "not_computed",
        )
        disabled_test = disabled.find_directional_test(
            "evidence_selection_cost",
            "llm_full_context",
        )
        self.assertIsNotNone(disabled_test)
        assert disabled_test is not None
        self.assertEqual(disabled_test.decision, "not_computed")

    def test_directional_margins_are_validated_and_recorded(self) -> None:
        inference = analyze_experiment_b_inference(
            _synthetic_result(8),
            bootstrap_replicates=20,
            selection_noninferiority_margin=0.15,
            net_harm_margin=0.25,
        )
        selection = inference.find_directional_test(
            "evidence_selection_cost",
            "llm_full_context",
        )
        net_harm = inference.find_directional_test(
            "soft_minus_balanced_terminal_error",
            "llm_full_context",
        )
        self.assertIsNotNone(selection)
        self.assertIsNotNone(net_harm)
        assert selection is not None and net_harm is not None
        self.assertEqual(selection.null_margin, 0.15)
        self.assertEqual(net_harm.null_margin, 0.25)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            analyze_experiment_b_inference(
                _synthetic_result(8),
                bootstrap_replicates=20,
                selection_noninferiority_margin=-0.01,
            )

    def test_dir_uses_pooled_counts_and_retains_denominator_support(self) -> None:
        result = _synthetic_result(2)
        result.trajectories[0].disconfirmation_opportunity_count = 1
        result.trajectories[0].disconfirmation_inversion_count = 1
        result.trajectories[1].disconfirmation_opportunity_count = 9
        result.trajectories[1].disconfirmation_inversion_count = 0
        inference = analyze_experiment_b_inference(
            result,
            bootstrap_replicates=0,
            minimum_user_clusters=2,
        )
        interval = inference.find(
            "disconfirmation_inversion_rate",
            "llm_full_context",
            policy_id="soft_profile_conditioned",
        )
        self.assertIsNotNone(interval)
        assert interval is not None
        self.assertAlmostEqual(interval.estimate, 0.1)
        self.assertEqual(interval.numerator_count, 1)
        self.assertEqual(interval.denominator_count, 10)
        self.assertEqual(interval.zero_denominator_cluster_count, 0)


if __name__ == "__main__":
    unittest.main()
