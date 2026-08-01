from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import unittest

from cape_loop.beliefs import MarginalPreferenceBelief, PreferenceBelief
from cape_loop.domains import TRAVEL
from cape_loop.experiments.hypothesis_estimands import (
    H7_VALID_LEARNING_RETENTION_FRACTION,
    VolunteeredPreferenceUpdate,
    analyze_experiment_a_hypotheses,
    analyze_h1,
    analyze_h2,
    analyze_h7_closed_loop,
    analyze_h7_experiment_a,
)
from cape_loop.experiments.provenance import run_provenance_audit
from cape_loop.metrics import action_conditioned_update_error
from cape_loop.updaters import NoUpdateUpdater


def _belief_with_directional_mass(
    *,
    attribute: int,
    direction: int,
    mass: float,
) -> PreferenceBelief:
    negative_mass = mass if direction < 0 else 1.0 - mass
    positive_mass = 1.0 - negative_mass
    target = (
        negative_mass / 2.0,
        negative_mass / 2.0,
        positive_mass / 2.0,
        positive_mass / 2.0,
    )
    uniform = (0.25, 0.25, 0.25, 0.25)
    marginals = [uniform, uniform, uniform]
    marginals[attribute] = target
    return PreferenceBelief.from_marginals(
        MarginalPreferenceBelief(tuple(marginals))  # type: ignore[arg-type]
    )


class ProposalHypothesisEstimandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        no_update = NoUpdateUpdater()
        base = run_provenance_audit(
            domains=(TRAVEL,),
            updaters={no_update.updater_id: no_update},
            response_modes=("controlled_anchor",),
            seed=41,
        )
        rows = []
        for source in base.rows:
            aware_mass = 0.72 if source.mechanism == "balanced" else 0.58
            full_mass = 0.75 if source.mechanism == "balanced" else 0.76
            mitigation_mass = (
                0.72 if source.mechanism == "balanced" else 0.60
            )
            unaware_mass = (
                0.75 if source.mechanism == "balanced" else 0.82
            )
            aware = _belief_with_directional_mass(
                attribute=source.target_attribute,
                direction=source.anchor_direction,
                mass=aware_mass,
            )
            for updater_id, mass in (
                ("llm_full_context", full_mass),
                ("fitted_action_unaware", unaware_mass),
                ("llm_provenance_aware", mitigation_mass),
            ):
                posterior = _belief_with_directional_mass(
                    attribute=source.target_attribute,
                    direction=source.anchor_direction,
                    mass=mass,
                )
                rows.append(
                    replace(
                        source,
                        updater_id=updater_id,
                        posterior=posterior,
                        fitted_aware_posterior=aware,
                        exact_posterior=aware,
                        acue=action_conditioned_update_error(
                            source.prior,
                            posterior,
                            source.prior,
                            aware,
                        ),
                    )
                )
        cls.rows = tuple(rows)
        cls.volunteered = tuple(
            VolunteeredPreferenceUpdate(
                case_id=f"volunteered-{user_id}",
                user_id=user_id,
                updater_id=updater_id,
                directional_log_odds_update=update,
            )
            for user_id in (
                "audit-user-negative",
                "audit-user-positive",
            )
            for updater_id, update in (
                ("llm_full_context", 1.0),
                ("llm_provenance_aware", 0.85),
            )
        )

    def test_h1_h2_h7_estimands_are_deterministic_and_proposal_aligned(
        self,
    ) -> None:
        first = analyze_experiment_a_hypotheses(
            self.rows,
            volunteered_updates=self.volunteered,
            replicates=80,
            seed=13,
        )
        second = analyze_experiment_a_hypotheses(
            self.rows,
            volunteered_updates=self.volunteered,
            replicates=80,
            seed=13,
        )
        self.assertEqual(first, second)
        self.assertIs(first.h1.criterion_met, True)
        self.assertEqual(
            {item.mechanism for item in first.h1.estimands},
            {"restricted", "default", "suggested"},
        )
        self.assertTrue(
            all(
                item.directional_update.interval.lower > 0.0
                and item.update_strength.interval.lower > 0.0
                for item in first.h1.estimands
            )
        )
        self.assertIs(first.h2.criterion_met, True)
        self.assertEqual(len(first.h2.qualifying_mechanisms), 3)
        self.assertTrue(
            all(
                item.aware_minus_unaware_distance.interval.lower > 0.0
                for item in first.h2.estimands
            )
        )
        self.assertIs(first.h7.criterion_met, True)
        self.assertIs(
            first.h7.balanced_valid_learning.criterion_met,
            True,
        )
        self.assertIs(
            first.h7.volunteered_valid_learning.criterion_met,
            True,
        )
        serialized = first.to_dict()
        self.assertEqual(serialized["claim_status"], "not_claimed")
        self.assertEqual(
            serialized["frozen_decision_constants"][
                "h7_valid_learning_retention_fraction"
            ],
            H7_VALID_LEARNING_RETENTION_FRACTION,
        )

    def test_h1_is_not_inferred_from_acue_alone(self) -> None:
        under_updating = []
        for row in self.rows:
            if row.updater_id != "llm_full_context":
                continue
            posterior = _belief_with_directional_mass(
                attribute=row.target_attribute,
                direction=row.anchor_direction,
                mass=0.52,
            )
            under_updating.append(
                replace(
                    row,
                    posterior=posterior,
                    acue=action_conditioned_update_error(
                        row.prior,
                        posterior,
                        row.prior,
                        row.fitted_aware_posterior,
                    ),
                )
            )
        self.assertTrue(all(row.acue > 0.0 for row in under_updating))
        result = analyze_h1(
            under_updating,
            replicates=40,
            seed=7,
        )
        self.assertIs(result.criterion_met, False)
        self.assertTrue(
            all(
                item.directional_update.estimate < 0.0
                for item in result.estimands
            )
        )

    def test_h2_requires_at_least_two_qualifying_mechanisms(self) -> None:
        mixed = []
        for row in self.rows:
            if (
                row.updater_id == "llm_full_context"
                and row.mechanism in {"default", "suggested"}
            ):
                posterior = _belief_with_directional_mass(
                    attribute=row.target_attribute,
                    direction=row.anchor_direction,
                    mass=0.59,
                )
                mixed.append(
                    replace(
                        row,
                        posterior=posterior,
                        acue=action_conditioned_update_error(
                            row.prior,
                            posterior,
                            row.prior,
                            row.fitted_aware_posterior,
                        ),
                    )
                )
            else:
                mixed.append(row)
        result = analyze_h2(mixed, replicates=40, seed=5)
        self.assertEqual(result.qualifying_mechanisms, ("restricted",))
        self.assertIs(result.criterion_met, False)

    def test_h7_does_not_impute_missing_volunteered_positive_control(
        self,
    ) -> None:
        result = analyze_h7_experiment_a(
            self.rows,
            replicates=40,
            seed=11,
        )
        self.assertIs(result.balanced_valid_learning.criterion_met, True)
        self.assertIsNone(
            result.volunteered_valid_learning.criterion_met
        )
        self.assertIn(
            "must not invent",
            result.volunteered_valid_learning.missing_reason or "",
        )
        self.assertIsNone(result.criterion_met)

    def test_h7_closed_loop_superiority_uses_matched_profile_rates(
        self,
    ) -> None:
        trajectories = []
        assessments = []
        for user_id in ("u1", "u2"):
            for updater_id, error, shadow_error, reportable in (
                ("llm_full_context", 0.5, 0.1, True),
                ("llm_provenance_aware", 0.2, 0.1, False),
            ):
                trajectory_id = f"{user_id}:{updater_id}"
                trajectories.append(
                    SimpleNamespace(
                        trajectory_id=trajectory_id,
                        crn_key=f"paired:{user_id}",
                        user_id=user_id,
                        domain_id="travel",
                        updater_id=updater_id,
                        policy_id="soft_profile_conditioned",
                        initial_profile_condition="incorrect",
                        terminal_error=error,
                        terminal_shadow_error=shadow_error,
                    )
                )
                assessments.append(
                    SimpleNamespace(
                        trajectory_id=trajectory_id,
                        reportable=reportable,
                    )
                )
        result = analyze_h7_closed_loop(
            SimpleNamespace(
                trajectories=tuple(trajectories),
                self_confirmation_assessments=tuple(assessments),
            ),
            replicates=40,
            seed=17,
        )
        self.assertIs(result.criterion_met, True)
        self.assertGreater(
            result.attribution_error_reduction.interval.lower,  # type: ignore[union-attr]
            0.0,
        )
        self.assertGreater(
            result.self_confirming_profile_rate_reduction.interval.lower,  # type: ignore[union-attr]
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
