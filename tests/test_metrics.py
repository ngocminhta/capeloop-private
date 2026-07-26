from __future__ import annotations

import math
import unittest

from cape_loop.beliefs import MarginalPreferenceBelief, PreferenceBelief
from cape_loop.metrics import (
    SelfConfirmationEvidence,
    action_conditioned_update_error,
    attribution_cost,
    false_confidence_gain,
    information_gain,
    laundered_confidence_gain,
    marginal_brier,
    selection_cost,
    self_confirmation_interaction,
    update_direction_accuracy,
)
from cape_loop.statistics import (
    bootstrap_ranks,
    evaluation_selection_regret,
    kendall_tau_b,
    pairwise_reversal_probability,
    pairwise_reversal_and_tie_probability,
    ranks_from_errors,
)


def belief_with_first_row(
    row: tuple[float, float, float, float],
) -> PreferenceBelief:
    uniform = (0.25, 0.25, 0.25, 0.25)
    return MarginalPreferenceBelief((row, uniform, uniform)).independent_joint()


class BeliefMetricTests(unittest.TestCase):
    def test_information_gain_is_entropy_reduction_and_can_be_negative(self) -> None:
        uniform = PreferenceBelief.uniform()
        concentrated = PreferenceBelief.point_mass((-2, -2, -2))
        self.assertAlmostEqual(
            information_gain(uniform, concentrated),
            math.log(64.0),
        )
        self.assertAlmostEqual(
            information_gain(concentrated, uniform),
            -math.log(64.0),
        )

    def test_acue_is_increment_error_not_endpoint_error(self) -> None:
        prior = PreferenceBelief.uniform()
        aware = belief_with_first_row((0.10, 0.15, 0.25, 0.50))
        same_increment = action_conditioned_update_error(
            prior,
            aware,
            prior,
            aware,
        )
        no_update = action_conditioned_update_error(
            prior,
            prior,
            prior,
            aware,
        )
        self.assertAlmostEqual(same_increment, 0.0)
        self.assertAlmostEqual(no_update, 0.50)
        self.assertEqual(
            update_direction_accuracy(prior, prior, prior, aware),
            0.0,
        )
        self.assertEqual(
            update_direction_accuracy(prior, aware, prior, aware),
            1.0,
        )

    def test_brier_confidence_and_decomposition_formulas(self) -> None:
        prior = PreferenceBelief.uniform()
        system = belief_with_first_row((0.05, 0.10, 0.25, 0.60))
        shadow = belief_with_first_row((0.15, 0.20, 0.30, 0.35))
        self.assertLess(marginal_brier(system, (2, -1, 1)), 0.75)
        system_gain = false_confidence_gain(
            prior,
            system,
            attribute=0,
            wrong_direction=1,
        )
        shadow_gain = false_confidence_gain(
            prior,
            shadow,
            attribute=0,
            wrong_direction=1,
        )
        self.assertAlmostEqual(
            laundered_confidence_gain(
                prior,
                system,
                prior,
                shadow,
                attribute=0,
                wrong_direction=1,
            ),
            system_gain - shadow_gain,
        )
        self.assertAlmostEqual(selection_cost(0.4, 0.3), 0.1)
        self.assertAlmostEqual(attribution_cost(0.6, 0.4), 0.2)
        self.assertAlmostEqual(
            self_confirmation_interaction(0.7, 0.4, 0.4, 0.3),
            0.2,
        )


class ProposalPredicateTests(unittest.TestCase):
    def test_self_confirmation_requires_all_five_clauses(self) -> None:
        complete = SelfConfirmationEvidence(
            remains_materially_wrong=True,
            wrong_mass_increased=True,
            cumulative_lcg=0.31,
            profile_changed_later_action=True,
            shadow_gained_equivalent_confidence=False,
            lcg_threshold=0.25,
        )
        self.assertTrue(complete.is_self_confirming)
        self.assertTrue(all(complete.clauses().values()))

        for field in (
            "remains_materially_wrong",
            "wrong_mass_increased",
            "profile_changed_later_action",
        ):
            values = {
                "remains_materially_wrong": True,
                "wrong_mass_increased": True,
                "cumulative_lcg": 0.31,
                "profile_changed_later_action": True,
                "shadow_gained_equivalent_confidence": False,
            }
            values[field] = False
            self.assertFalse(SelfConfirmationEvidence(**values).is_self_confirming)
        self.assertFalse(
            SelfConfirmationEvidence(
                True,
                True,
                0.25,
                True,
                False,
            ).is_self_confirming
        )
        self.assertFalse(
            SelfConfirmationEvidence(
                True,
                True,
                0.31,
                True,
                True,
            ).is_self_confirming
        )


class RankingMetricTests(unittest.TestCase):
    def test_ranks_tau_bootstrap_and_reversal_are_reproducible(self) -> None:
        open_errors = {
            "a": (0.10, 0.12, 0.11, 0.13),
            "b": (0.20, 0.22, 0.21, 0.23),
            "c": (0.30, 0.32, 0.31, 0.33),
        }
        closed_errors = {
            "a": (0.31, 0.30, 0.32, 0.29),
            "b": (0.20, 0.21, 0.19, 0.22),
            "c": (0.10, 0.11, 0.09, 0.12),
        }
        ranks = ranks_from_errors(
            {name: sum(values) / len(values) for name, values in open_errors.items()}
        )
        self.assertEqual(ranks, {"a": 1.0, "b": 2.0, "c": 3.0})
        self.assertAlmostEqual(
            kendall_tau_b(
                {name: sum(values) for name, values in open_errors.items()},
                {name: sum(values) for name, values in closed_errors.items()},
            ),
            -1.0,
        )

        first = bootstrap_ranks(open_errors, replicates=50, seed=41)
        second = bootstrap_ranks(open_errors, replicates=50, seed=41)
        self.assertEqual(first, second)
        reversals_first = pairwise_reversal_probability(
            open_errors,
            closed_errors,
            replicates=50,
            seed=41,
        )
        reversals_second = pairwise_reversal_probability(
            open_errors,
            closed_errors,
            replicates=50,
            seed=41,
        )
        self.assertEqual(reversals_first, reversals_second)
        self.assertTrue(all(value == 1.0 for value in reversals_first.values()))
        reversals, ties = pairwise_reversal_and_tie_probability(
            {"a": (0.1, 0.1), "b": (0.1, 0.1)},
            {"a": (0.2, 0.2), "b": (0.3, 0.3)},
            replicates=20,
            seed=41,
            tie_tolerance=1e-6,
        )
        self.assertEqual(reversals["a|b"], 0.0)
        self.assertEqual(ties["a|b"], 1.0)
        with self.assertRaises(ValueError):
            bootstrap_ranks(open_errors, replicates=0, seed=41)
        with self.assertRaises(ValueError):
            bootstrap_ranks({"a": (0.1, 0.2)}, replicates=10, seed=41)

    def test_evaluation_selection_regret_uses_closed_test_error(self) -> None:
        result = evaluation_selection_regret(
            {"open-best": 0.1, "closed-best": 0.2},
            {"open-best": 0.4, "closed-best": 0.2},
            {"open-best": 0.5, "closed-best": 0.15},
        )
        self.assertEqual(result["open_selected"], "open-best")
        self.assertEqual(result["closed_selected"], "closed-best")
        self.assertAlmostEqual(
            float(result["evaluation_selection_regret"]),
            0.35,
        )


if __name__ == "__main__":
    unittest.main()
