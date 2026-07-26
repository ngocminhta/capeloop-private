from __future__ import annotations

import unittest

from cape_loop.config import load_config
from cape_loop.population import add_prior_uncertainty
from cape_loop.response import RandomUtilityModel, RuleBasedResponseModel
from cape_loop.sensitivity import (
    PhaseCriterion,
    classify_phase_point,
    evaluate_grid,
    infer_axis_boundaries,
    response_model_at,
    sensitivity_grid,
)
from cape_loop.beliefs import PreferenceBelief


class SensitivityTests(unittest.TestCase):
    def test_grid_is_explicit_and_model_scales_noise(self) -> None:
        points = sensitivity_grid(
            decision_noise_values=[0.5, 1.0],
            presentation_multipliers=[1.0],
            profile_strength_values=[0.8],
            trajectory_lengths=[4, 8],
        )
        self.assertEqual(len(points), 4)
        low_noise = response_model_at(
            points[0],
            beta=1.0,
            rank_scale=0.3,
            default_scale=0.8,
            suggestion_scale=0.6,
        )
        self.assertGreater(low_noise.beta, 1.0)
        rows = evaluate_grid(points, lambda point: {"metric": point.decision_noise})
        self.assertEqual(len(rows), len(points))

    def test_checked_in_sensitivity_config_loads(self) -> None:
        config = load_config("configs/sensitivity.toml")
        self.assertEqual(config.experiment.kind, "sensitivity")
        self.assertEqual(len(config.sensitivity.decision_noise_values), 3)

    def test_full_grid_has_independent_axes_and_alternative_family(self) -> None:
        points = sensitivity_grid(
            decision_noise_values=[1.0],
            presentation_multipliers=[1.0],
            rank_multipliers=[0.5, 1.5],
            default_multipliers=[1.0],
            suggestion_multipliers=[1.0],
            profile_strength_values=[0.8],
            prior_uncertainty_values=[0.0, 0.4],
            trajectory_lengths=[4],
            response_model_families=["random_utility", "rule_based"],
            rule_noise_values=[0.1, 0.2],
        )
        # Two rank levels × two prior levels × (one logit + two rule points).
        self.assertEqual(len(points), 12)
        models = [
            response_model_at(
                point,
                beta=1.0,
                rank_scale=0.3,
                default_scale=0.8,
                suggestion_scale=0.6,
            )
            for point in points
        ]
        self.assertTrue(any(isinstance(model, RandomUtilityModel) for model in models))
        self.assertTrue(any(isinstance(model, RuleBasedResponseModel) for model in models))
        self.assertEqual(
            load_config("configs/sensitivity_full.toml").sensitivity.response_model_families,
            ("random_utility", "rule_based"),
        )

    def test_prior_uncertainty_and_phase_boundaries_are_explicit(self) -> None:
        point = PreferenceBelief.point_mass((2, 2, 2))
        mixed = add_prior_uncertainty(point, 0.5)
        self.assertLess(mixed.probability((2, 2, 2)), 1.0)
        criteria = (
            PhaseCriterion("selection", "selection_cost", "gt", 0.0),
            PhaseCriterion("calibration", "ece", "le", 0.1),
        )
        rows = [
            {
                "point_id": "a",
                "decision_noise": 0.5,
                "profile_strength": 0.8,
                "selection_cost": 0.2,
                "ece": 0.05,
            },
            {
                "point_id": "b",
                "decision_noise": 1.0,
                "profile_strength": 0.8,
                "selection_cost": -0.1,
                "ece": 0.05,
            },
        ]
        self.assertTrue(classify_phase_point(rows[0], criteria)["joint_region"])
        boundaries = infer_axis_boundaries(
            rows,
            criteria,
            axis="decision_noise",
        )
        self.assertEqual(boundaries[0]["passing_coordinates"], [0.5])


if __name__ == "__main__":
    unittest.main()
