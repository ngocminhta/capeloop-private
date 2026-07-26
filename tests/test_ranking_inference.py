from __future__ import annotations

import random
import unittest
from types import SimpleNamespace

from cape_loop.experiments.evaluation import (
    EvaluationRow,
    TerminalBatteryScore,
    analyze_rankings,
    build_clustered_ranking_samples,
)
from cape_loop.runner import _gate_5_for_c
from cape_loop.statistics import (
    inferential_partial_order,
    paired_system_difference_intervals,
    paired_system_regime_shift_intervals,
    tied_evaluation_selection_regret,
)


class RankingInferenceTests(unittest.TestCase):
    @staticmethod
    def _row(
        *,
        split: str,
        regime: str,
        user_id: str,
        domain_id: str,
        replicate: int,
        updater_id: str,
        error: float,
    ) -> EvaluationRow:
        score = TerminalBatteryScore(
            profile_brier=error,
            behavioral_accuracy=1.0 - error,
            tie_excluded_behavioral_accuracy=1.0 - error,
            fractional_behavioral_accuracy=1.0 - error,
            cross_context_accuracy=1.0 - error,
            mean_intrinsic_regret=error,
            predicted_option_ids=("selected",),
            predicted_utility_tie_count=0.0,
            intrinsic_utility_tie_count=0.0,
            evaluated_item_count=1,
        )
        stable_id = (
            f"{split}:{regime}:{user_id}:{domain_id}:{replicate}:{updater_id}"
        )
        return EvaluationRow(
            split=split,
            regime=regime,
            replicate=replicate,
            user_id=user_id,
            domain_id=domain_id,
            updater_id=updater_id,
            profile_error=error,
            behavioral_accuracy=score.behavioral_accuracy,
            cross_context_accuracy=score.cross_context_accuracy,
            intrinsic_regret=score.mean_intrinsic_regret,
            history_digest=stable_id,
            event_signatures=(stable_id,),
            battery_id=f"battery:{domain_id}",
            battery_digest=f"digest:{domain_id}",
            predicted_option_ids=score.predicted_option_ids,
            score_basis="test",
            system_projection_score=score,
        )

    @classmethod
    def _complete_ranking_rows(cls) -> tuple[EvaluationRow, ...]:
        rows = []
        regimes_by_split = {
            "development": (
                "fixed_balanced",
                "fixed_biased",
                "endogenous_closed_loop",
            ),
            "test": ("endogenous_closed_loop",),
        }
        users_by_split = {
            "development": ("dev-1", "dev-2"),
            "test": ("test-1", "test-2"),
        }
        system_effect = {
            "fixed_balanced": {"a": 0.00, "b": 0.20},
            "fixed_biased": {"a": 0.05, "b": 0.15},
            "endogenous_closed_loop": {"a": 0.22, "b": 0.02},
        }
        for split, regimes in regimes_by_split.items():
            for regime in regimes:
                for user_index, user_id in enumerate(users_by_split[split]):
                    for domain_index, domain_id in enumerate(
                        ("travel", "workflow")
                    ):
                        for replicate in (0, 1):
                            for updater_id in ("a", "b"):
                                error = (
                                    0.05
                                    + 0.04 * user_index
                                    + 0.01 * domain_index
                                    + 0.005 * replicate
                                    + system_effect[regime][updater_id]
                                )
                                rows.append(
                                    cls._row(
                                        split=split,
                                        regime=regime,
                                        user_id=user_id,
                                        domain_id=domain_id,
                                        replicate=replicate,
                                        updater_id=updater_id,
                                        error=error,
                                    )
                                )
        return tuple(rows)

    def test_partial_order_uses_paired_uncertainty(self) -> None:
        samples = {
            "a": (0.10, 0.11, 0.09, 0.10),
            "b": (0.30, 0.31, 0.29, 0.30),
            "c": (0.305, 0.295, 0.31, 0.29),
        }
        intervals = paired_system_difference_intervals(
            samples,
            replicates=200,
            seed=11,
            tie_tolerance=1e-3,
        )
        tiers = inferential_partial_order(tuple(samples), intervals)
        self.assertEqual(tiers[0], ("a",))
        self.assertEqual(set(tiers[1]), {"b", "c"})

    def test_joint_paired_reversal_requires_opposite_resolved_orders(self) -> None:
        clear = paired_system_regime_shift_intervals(
            {
                "a": (0.10, 0.11, 0.09, 0.10),
                "b": (0.30, 0.31, 0.29, 0.30),
            },
            {
                "a": (0.30, 0.31, 0.29, 0.30),
                "b": (0.10, 0.11, 0.09, 0.10),
            },
            replicates=200,
            seed=23,
            tie_tolerance=1e-3,
        )[0]
        self.assertTrue(clear.credible_reversal)
        self.assertEqual(
            clear.reversal_relation,
            "first_better_open_second_better_closed",
        )

        uncertain = paired_system_regime_shift_intervals(
            {
                "a": (0.06, 0.14, 0.08, 0.12, 0.09, 0.11),
                "b": (0.10, 0.10, 0.10, 0.10, 0.10, 0.10),
            },
            {
                "a": (0.14, 0.06, 0.12, 0.08, 0.11, 0.09),
                "b": (0.10, 0.10, 0.10, 0.10, 0.10, 0.10),
            },
            replicates=400,
            seed=23,
            tie_tolerance=1e-3,
        )[0]
        self.assertFalse(uncertain.credible_reversal)
        self.assertEqual(
            uncertain.reversal_relation,
            "no_credible_reversal",
        )

    def test_esr_returns_tied_selection_sets_and_range(self) -> None:
        result = tied_evaluation_selection_regret(
            {"a": 0.1, "b": 0.1000001, "c": 0.2},
            {"a": 0.2, "b": 0.3, "c": 0.2000001},
            {"a": 0.25, "b": 0.4, "c": 0.1},
            tie_tolerance=1e-3,
        )
        self.assertEqual(result["open_selected_set"], ("a", "b"))
        self.assertEqual(result["closed_selected_set"], ("a", "c"))
        self.assertLessEqual(
            result["evaluation_selection_regret_min"],
            result["evaluation_selection_regret"],
        )
        self.assertGreaterEqual(
            result["evaluation_selection_regret_max"],
            result["evaluation_selection_regret"],
        )

    def test_experiment_c_ranking_is_invariant_to_input_row_order(self) -> None:
        rows = self._complete_ranking_rows()
        shuffled = list(rows)
        random.Random(73).shuffle(shuffled)
        expected = analyze_rankings(
            rows,
            updater_ids=("a", "b"),
            bootstrap_replicates=100,
            seed=19,
        )
        observed = analyze_rankings(
            tuple(shuffled),
            updater_ids=("a", "b"),
            bootstrap_replicates=100,
            seed=19,
        )
        self.assertEqual(observed, expected)

    def test_experiment_c_bootstrap_inputs_are_complete_user_clusters(
        self,
    ) -> None:
        rows = self._complete_ranking_rows()
        clustered = build_clustered_ranking_samples(
            rows,
            split="development",
            regime="fixed_balanced",
            updater_ids=("a", "b"),
        )
        self.assertEqual(clustered.cluster_ids, ("dev-1", "dev-2"))
        self.assertEqual(
            clustered.component_layout,
            (
                ("travel", 0),
                ("travel", 1),
                ("workflow", 0),
                ("workflow", 1),
            ),
        )
        self.assertTrue(
            all(len(members) == 4 for members in clustered.member_keys)
        )
        # Each downstream sample is one complete-user mean, not one entry per
        # domain/replicate row. A bootstrap index therefore cannot split these
        # four trajectory components apart.
        self.assertEqual(
            {system: len(values) for system, values in clustered.system_samples},
            {"a": 2, "b": 2},
        )
        self.assertAlmostEqual(
            clustered.errors_by_system["a"][0],
            (0.05 + 0.055 + 0.06 + 0.065) / 4.0,
        )

        analysis = analyze_rankings(
            rows,
            updater_ids=("a", "b"),
            bootstrap_replicates=20,
            seed=19,
        )
        self.assertEqual(analysis.inference_unit, "complete_latent_user_cluster")
        self.assertEqual(analysis.development_cluster_count, 2)
        self.assertEqual(analysis.test_cluster_count, 2)
        self.assertEqual(
            analysis.credible_pairwise_reversals,
            ("a|b",),
        )
        self.assertEqual(analysis.open_partial_order[0], ("a",))
        self.assertEqual(analysis.closed_partial_order[0], ("b",))
        self.assertEqual(
            dict(analysis.evaluation_selection_regret)["selection_basis"],
            "paired development error-difference confidence-set top tiers",
        )

    def test_gate_5_does_not_promote_an_uncertain_near_tie(self) -> None:
        rows = []
        open_differences = (
            -0.040,
            -0.030,
            -0.020,
            -0.010,
            0.009,
            0.020,
            0.030,
            0.033,
        )
        for split, users in (
            ("development", tuple(f"dev-{index}" for index in range(8))),
            ("test", tuple(f"test-{index}" for index in range(8))),
        ):
            regimes = (
                (
                    "fixed_balanced",
                    open_differences,
                ),
                (
                    "fixed_biased",
                    tuple(0.0 for _ in open_differences),
                ),
                (
                    "endogenous_closed_loop",
                    tuple(-value for value in open_differences),
                ),
            )
            if split == "test":
                regimes = (
                    (
                        "endogenous_closed_loop",
                        tuple(0.0 for _ in open_differences),
                    ),
                )
            for regime, differences in regimes:
                for user_id, difference in zip(users, differences):
                    rows.append(
                        self._row(
                            split=split,
                            regime=regime,
                            user_id=user_id,
                            domain_id="travel",
                            replicate=0,
                            updater_id="a",
                            error=0.20 + difference,
                        )
                    )
                    rows.append(
                        self._row(
                            split=split,
                            regime=regime,
                            user_id=user_id,
                            domain_id="travel",
                            replicate=0,
                            updater_id="b",
                            error=0.20,
                        )
                    )
        analysis = analyze_rankings(
            tuple(rows),
            updater_ids=("a", "b"),
            bootstrap_replicates=500,
            seed=59,
            tie_tolerance=1e-3,
        )
        self.assertEqual(analysis.open_closed_kendall_tau, -1.0)
        self.assertEqual(analysis.credible_pairwise_reversals, ())
        self.assertEqual(set(analysis.open_partial_order[0]), {"a", "b"})
        self.assertEqual(set(analysis.closed_partial_order[0]), {"a", "b"})
        gate = _gate_5_for_c(SimpleNamespace(rankings=analysis))
        self.assertEqual(gate.computed_status, "does_not_meet_checks")
        observed = gate.criteria[0].observed
        self.assertTrue(observed["low_rank_agreement_descriptive"])
        self.assertFalse(
            observed["kendall_tau_is_gate_sufficient_without_interval"]
        )

    def test_experiment_c_rejects_missing_or_duplicate_stable_keys(self) -> None:
        rows = self._complete_ranking_rows()
        missing = tuple(
            row
            for row in rows
            if not (
                row.split == "development"
                and row.regime == "fixed_balanced"
                and row.user_id == "dev-1"
                and row.domain_id == "travel"
                and row.replicate == 0
                and row.updater_id == "b"
            )
        )
        with self.assertRaisesRegex(ValueError, "not aligned"):
            build_clustered_ranking_samples(
                missing,
                split="development",
                regime="fixed_balanced",
                updater_ids=("a", "b"),
            )

        incomplete_cluster = tuple(
            row
            for row in rows
            if not (
                row.split == "development"
                and row.regime == "fixed_balanced"
                and row.user_id == "dev-1"
                and row.domain_id == "workflow"
                and row.replicate == 1
            )
        )
        with self.assertRaisesRegex(ValueError, "complete observed"):
            build_clustered_ranking_samples(
                incomplete_cluster,
                split="development",
                regime="fixed_balanced",
                updater_ids=("a", "b"),
            )

        duplicate = rows + (rows[0],)
        with self.assertRaisesRegex(ValueError, "duplicate ranking row"):
            build_clustered_ranking_samples(
                duplicate,
                split="development",
                regime="fixed_balanced",
                updater_ids=("a", "b"),
            )


if __name__ == "__main__":
    unittest.main()
