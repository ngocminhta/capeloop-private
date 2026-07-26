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
            )
        )
        trajectories.append(
            SimpleNamespace(
                trajectory_id=profile_id,
                user_id=user_id,
                updater_id="llm_full_context",
                policy_id="soft_profile_conditioned",
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

        attribution = first.find(
            "profile_attribution_cost",
            "llm_full_context",
        )
        self.assertIsNotNone(attribution)
        assert attribution is not None
        self.assertTrue(attribution.adequate)
        self.assertEqual(attribution.cluster_count, 8)
        self.assertGreater(attribution.lower, 0.0)

        trajectory = first.find(
            "profile_attribution_cost",
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

    def test_small_cluster_and_disabled_bootstrap_are_not_adequate(self) -> None:
        small = analyze_experiment_b_inference(
            _synthetic_result(7),
            bootstrap_replicates=20,
        )
        interval = small.find(
            "self_confirmation_interaction",
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
            "profile_attribution_cost",
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


if __name__ == "__main__":
    unittest.main()
