from __future__ import annotations

import math
import unittest
from unittest.mock import patch

from cape_loop.beliefs import MarginalPreferenceBelief, PreferenceBelief
from cape_loop.domains import TRAVEL
from cape_loop.experiments.provenance import (
    ExperimentAConfirmatoryResult,
    compare_experiment_a_raw_calibrated,
    estimate_oracle_update_slopes,
    experiment_a_matched_set_id,
    experiment_a_mechanism_contrasts,
    experiment_a_updater_mechanism_interaction,
    fit_experiment_a_marginal_ols,
    fitted_evidence_strength_ordering,
    run_provenance_audit,
)
from cape_loop.statistics import (
    MarginalForecast,
    clustered_bootstrap_mean,
    compare_raw_and_calibrated_forecasts,
    fit_cluster_robust_ols,
    holm_bonferroni,
    paired_cluster_contrast,
    paired_cluster_interaction,
    simulate_paired_cluster_power,
)
from cape_loop.updaters import FittedActionAwareUpdater, NoUpdateUpdater


class StatisticsInfrastructureTests(unittest.TestCase):
    def test_complete_cluster_bootstrap_equal_weights_clusters(self) -> None:
        first = clustered_bootstrap_mean(
            [0.0, 0.0, 10.0],
            ["participant-a", "participant-a", "participant-b"],
            replicates=50,
            seed=9,
        )
        second = clustered_bootstrap_mean(
            [0.0, 0.0, 10.0],
            ["participant-a", "participant-a", "participant-b"],
            replicates=50,
            seed=9,
        )
        self.assertEqual(first, second)
        self.assertEqual(first.estimate, 5.0)
        self.assertEqual(first.cluster_count, 2)

    def test_paired_contrast_and_interaction_preserve_pairing(self) -> None:
        contrast = paired_cluster_contrast(
            [3.0, 5.0, 7.0],
            [1.0, 2.0, 3.0],
            ["u1", "u2", "u3"],
            contrast_id="treated-minus-reference",
            first_label="treated",
            second_label="reference",
            replicates=40,
            seed=4,
        )
        self.assertEqual(contrast.estimate, 3.0)
        interaction = paired_cluster_interaction(
            [5.0, 7.0],
            [2.0, 3.0],
            [4.0, 4.0],
            [3.0, 3.0],
            ["u1", "u2"],
            contrast_id="did",
            first_label="aware",
            second_label="blind",
            treated_label="default",
            reference_label="balanced",
            replicates=30,
            seed=5,
        )
        self.assertEqual(interaction.estimate, 2.5)

    def test_holm_adjustment_is_step_down_and_serializable(self) -> None:
        result = holm_bonferroni(
            {"primary": 0.01, "secondary": 0.03, "null": 0.7},
            alpha=0.05,
        )
        decisions = {
            item.hypothesis_id: item for item in result.decisions
        }
        self.assertTrue(decisions["primary"].reject)
        self.assertFalse(decisions["secondary"].reject)
        self.assertFalse(decisions["null"].reject)
        self.assertAlmostEqual(
            decisions["primary"].adjusted_p_value,
            0.03,
        )
        self.assertEqual(result.to_dict()["method"][:4], "Holm")

    def test_raw_calibrated_scores_use_valid_one_vs_rest_reliability(self) -> None:
        forecasts = (
            MarginalForecast(
                "r1",
                "u1",
                (0.6, 0.3, 0.1),
                (0.8, 0.15, 0.05),
                0,
            ),
            MarginalForecast(
                "r2",
                "u2",
                (0.6, 0.3, 0.1),
                (0.4, 0.4, 0.2),
                1,
            ),
        )
        comparison = compare_raw_and_calibrated_forecasts(
            forecasts,
            bin_count=5,
        )
        self.assertEqual(len(comparison.scores), 4)
        self.assertEqual(len(comparison.reliability_bins), 10)
        for variant in ("raw", "calibrated"):
            self.assertEqual(
                sum(
                    row.prediction_count
                    for row in comparison.reliability_bins
                    if row.variant == variant
                ),
                6,
            )
        self.assertIn(
            "calibrated_minus_raw_mean_brier",
            comparison.to_dict()["summary"],
        )

    def test_cluster_robust_ols_and_power_artifact_are_deterministic(self) -> None:
        design = [
            [1.0, float(index)]
            for index in range(8)
        ]
        outcomes = [1.0 + 2.0 * row[1] for row in design]
        clusters = [
            "u1",
            "u1",
            "u2",
            "u2",
            "u3",
            "u3",
            "u4",
            "u4",
        ]
        regression = fit_cluster_robust_ols(
            design,
            outcomes,
            clusters,
            ("intercept", "x"),
        )
        self.assertAlmostEqual(regression.coefficients[0].estimate, 1.0)
        self.assertAlmostEqual(regression.coefficients[1].estimate, 2.0)
        self.assertIn("not a mixed-effects model", regression.inference_note)

        first = simulate_paired_cluster_power(
            (0.1, 0.2, 0.3, -0.1),
            (8, 12),
            estimand="updater-by-policy interaction",
            simulations=40,
            seed=7,
        )
        second = simulate_paired_cluster_power(
            (0.1, 0.2, 0.3, -0.1),
            (8, 12),
            estimand="updater-by-policy interaction",
            simulations=40,
            seed=7,
        )
        self.assertEqual(first, second)
        self.assertEqual(first.to_dict()["seed"], 7)


class ExperimentAConfirmatoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        aware = FittedActionAwareUpdater()
        no_update = NoUpdateUpdater()
        cls.result = run_provenance_audit(
            domains=(TRAVEL,),
            updaters={
                aware.updater_id: aware,
                no_update.updater_id: no_update,
            },
            response_modes=("controlled_anchor",),
            seed=31,
        )

    def test_oracle_slope_and_fitted_evidence_ordering(self) -> None:
        slopes = estimate_oracle_update_slopes(
            self.result.rows,
            replicates=40,
            seed=2,
        )
        by_updater = {item.updater_id: item for item in slopes}
        self.assertAlmostEqual(
            by_updater["fitted_action_aware"].slope,
            1.0,
            places=9,
        )
        self.assertAlmostEqual(by_updater["no_update"].slope, 0.0)
        self.assertIsNotNone(
            by_updater["fitted_action_aware"].slope_interval
        )
        mechanism_slopes = {
            item.mechanism: item
            for item in by_updater[
                "fitted_action_aware"
            ].mechanism_slopes
        }
        self.assertEqual(
            set(mechanism_slopes),
            {"balanced", "default", "ranking", "restricted", "suggested"},
        )
        self.assertTrue(
            all(
                item.slope is not None
                and math.isclose(item.slope, 1.0, abs_tol=1e-9)
                for mechanism, item in mechanism_slopes.items()
                if mechanism != "restricted"
            )
        )
        self.assertIsNone(mechanism_slopes["restricted"].slope)
        self.assertIn(
            "not_estimable",
            mechanism_slopes["restricted"].inference_status,
        )
        self.assertTrue(
            all(
                item.slope_interval is not None
                for mechanism, item in mechanism_slopes.items()
                if mechanism != "restricted"
            )
        )
        exact_slopes = self.result.exact_oracle_update_slopes(
            replicates=40,
            seed=2,
        )
        self.assertTrue(
            all(
                item.reference_basis == "exact_action_aware"
                for item in exact_slopes
            )
        )
        self.assertTrue(
            all(math.isfinite(item.slope) for item in exact_slopes)
        )

        baseline = fitted_evidence_strength_ordering(self.result.rows)
        self.assertEqual(
            {label for label, _ in baseline.aggregate_strengths},
            {"balanced", "default", "ranking", "restricted", "suggested"},
        )
        self.assertEqual(baseline.volunteered_control_coverage, 0)
        volunteered = {
            experiment_a_matched_set_id(row): 2.0
            for row in self.result.rows
        }
        with_control = fitted_evidence_strength_ordering(
            self.result.rows,
            volunteered_strengths=volunteered,
        )
        self.assertEqual(
            with_control.volunteered_control_coverage,
            len(with_control.matched_sets),
        )
        self.assertIn(
            "volunteered",
            dict(with_control.aggregate_strengths),
        )

    def test_oracle_bootstrap_reuses_belief_derived_row_updates(self) -> None:
        original_marginals = PreferenceBelief.marginals
        marginal_calls = 0

        def counted_marginals(
            belief: PreferenceBelief,
        ) -> MarginalPreferenceBelief:
            nonlocal marginal_calls
            marginal_calls += 1
            return original_marginals(belief)

        selected_count = sum(
            row.response_mode == "controlled_anchor"
            for row in self.result.rows
        )
        with patch.object(
            PreferenceBelief,
            "marginals",
            new=counted_marginals,
        ):
            slopes = estimate_oracle_update_slopes(
                self.result.rows,
                replicates=20,
                seed=17,
                reference_basis="exact_action_aware",
            )

        self.assertTrue(slopes)
        # Each selected row contributes two sign masses to the exact reference
        # and two to the system update. Bootstrap replication must not trigger
        # any additional belief marginalization.
        self.assertEqual(marginal_calls, 4 * selected_count)

    def test_mechanism_contrast_interaction_and_marginal_model(self) -> None:
        contrasts = experiment_a_mechanism_contrasts(
            self.result.rows,
            first_mechanism="default",
            second_mechanism="balanced",
            metric="update_magnitude",
            replicates=30,
            seed=4,
        )
        self.assertEqual(
            {item.contrast_id.split(":")[2] for item in contrasts},
            {"fitted_action_aware", "no_update"},
        )
        residual_contrasts = experiment_a_mechanism_contrasts(
            self.result.rows,
            first_mechanism="default",
            second_mechanism="balanced",
            metric="calibration_residual",
            updater_id="fitted_action_aware",
            replicates=30,
            seed=4,
        )
        self.assertEqual(len(residual_contrasts), 1)
        self.assertIn(
            ":calibration_residual:fitted_action_aware:",
            residual_contrasts[0].contrast_id,
        )
        interaction = experiment_a_updater_mechanism_interaction(
            self.result.rows,
            first_updater="fitted_action_aware",
            second_updater="no_update",
            treated_mechanism="default",
            reference_mechanism="balanced",
            metric="update_magnitude",
            replicates=30,
            seed=4,
        )
        self.assertGreater(interaction.pair_count, 0)
        regression = fit_experiment_a_marginal_ols(
            self.result.rows,
            outcome="acue",
            response_mode="controlled_anchor",
        )
        self.assertEqual(regression.cluster_count, 2)
        self.assertIn("marginal OLS", regression.model_label)

    def test_marginal_model_includes_executable_prior_strength_factor(
        self,
    ) -> None:
        aware = FittedActionAwareUpdater()
        no_update = NoUpdateUpdater()
        crossed = run_provenance_audit(
            domains=(TRAVEL,),
            updaters={
                aware.updater_id: aware,
                no_update.updater_id: no_update,
            },
            prior_strengths=(0.0, 0.35, 0.7),
            response_modes=("controlled_anchor",),
            seed=37,
        )
        regression = fit_experiment_a_marginal_ols(
            crossed.rows,
            outcome="acue",
            response_mode="controlled_anchor",
        )
        self.assertIn(
            "prior_strength",
            {coefficient.name for coefficient in regression.coefficients},
        )
        self.assertIn("+ prior_strength", regression.model_label)

    def test_raw_calibrated_integration_and_result_bundle(self) -> None:
        theta_by_user = {
            "audit-user-negative": (-2, -1, 1),
            "audit-user-positive": (2, 1, -1),
        }
        comparison = compare_experiment_a_raw_calibrated(
            self.result.rows,
            self.result.rows,
            true_theta_by_user=theta_by_user,
            bin_count=4,
        )
        self.assertAlmostEqual(
            comparison.raw_mean_brier,
            comparison.calibrated_mean_brier,
        )
        bundle = ExperimentAConfirmatoryResult(
            oracle_update_slopes=self.result.oracle_update_slopes(
                replicates=20,
                seed=1,
            ),
            evidence_strength=self.result.evidence_strength_analysis(),
            raw_calibrated_comparison=comparison,
            notes=(
                "The mixed-effects primary model remains an external analysis.",
            ),
        )
        serialized = bundle.to_dict()
        self.assertEqual(serialized["independent_unit"], "complete latent user")
        self.assertIsNotNone(serialized["raw_calibrated_comparison"])


if __name__ == "__main__":
    unittest.main()
