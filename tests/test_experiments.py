from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import unittest

from cape_loop.beliefs import (
    MarginalPreferenceBelief,
    PreferenceBelief,
    THETA_STATES,
    THETA_VALUES,
)
from cape_loop.conversation_surfaces import load_conversation_bank
from cape_loop.domains import TRAVEL
from cape_loop.elicitation import MECHANISMS, build_matched_anchor_set
from cape_loop.experiments import (
    assess_self_confirmation,
    build_terminal_battery,
    generate_fixed_history,
    replay_history,
    run_experiment_b,
    run_experiment_c,
    run_provenance_audit,
    run_trajectory,
    summarize_prospective_strata_occupancy,
)
from cape_loop.experiments.closed_loop import DecompositionRow
from cape_loop.experiments.provenance import (
    audit_same_response_provenance,
    build_experiment_a_control_battery,
)
from cape_loop.metrics import marginal_brier, marginal_l1
from cape_loop.native import EpisodicMemoryUpdater
from cape_loop.policies import (
    BalancedPolicy,
    ExploratoryPolicy,
    HardFilterPolicy,
    SoftProfileConditionedPolicy,
)
from cape_loop.population import initial_profile_belief
from cape_loop.response import RandomUtilityModel
from cape_loop.scenarios import load_scenario_catalog
from cape_loop.schemas import LatentUser, ProfileUpdate, Susceptibility
from cape_loop.updaters import (
    ExactActionAwareUpdater,
    FittedActionAwareUpdater,
    FittedActionUnawareUpdater,
    FullContextBlindUpdater,
    LLMReplayUpdater,
    NoUpdateUpdater,
    ProvenanceAwareUpdater,
    ProvenanceDiscountUpdater,
    ResponseOnlyUpdater,
    UpdateResult,
    UpdaterState,
    UpdateViewKind,
    build_updater_registry,
    make_update_view,
)


class SameSignStrengtheningUpdater:
    """Fixture that increases wrong mass without changing its policy-facing sign."""

    updater_id = "same_sign_strengthening"
    view_kind = UpdateViewKind.FULL_CONTEXT

    def __init__(self, strengthened: PreferenceBelief) -> None:
        self.strengthened = strengthened

    def initial_state(self, prior: PreferenceBelief) -> UpdaterState:
        return UpdaterState(self.updater_id, prior)

    def update(self, state: UpdaterState, view: object) -> UpdateResult:
        event_id = getattr(view, "event_id")
        next_state = UpdaterState(
            self.updater_id,
            self.strengthened,
            turn=state.turn + 1,
            event_ids=state.event_ids + (event_id,),
        )
        return UpdateResult(
            state=next_state,
            profile_update=ProfileUpdate(
                updater_id=self.updater_id,
                belief_before=state.belief.probabilities,
                belief_after=self.strengthened.probabilities,
                written_delta=("strengthen existing signs",),
            ),
        )


class StrengthenThenWeakenUpdater:
    """Fixture whose only action difference occurs below the seeded wrong mass."""

    updater_id = "strengthen_then_weaken"
    view_kind = UpdateViewKind.FULL_CONTEXT

    def __init__(
        self,
        strong: PreferenceBelief,
        weak: PreferenceBelief,
    ) -> None:
        self.strong = strong
        self.weak = weak

    def initial_state(self, prior: PreferenceBelief) -> UpdaterState:
        return UpdaterState(self.updater_id, prior)

    def update(self, state: UpdaterState, view: object) -> UpdateResult:
        # Turn 0 strengthens, turns 1/2 weaken below the seed, and turn 3
        # strengthens only after its action has already been selected.
        belief = self.strong if state.turn in (0, 3) else self.weak
        event_id = getattr(view, "event_id")
        next_state = UpdaterState(
            self.updater_id,
            belief,
            turn=state.turn + 1,
            event_ids=state.event_ids + (event_id,),
        )
        return UpdateResult(
            state=next_state,
            profile_update=ProfileUpdate(
                updater_id=self.updater_id,
                belief_before=state.belief.probabilities,
                belief_after=belief.probabilities,
                written_delta=("scheduled",),
            ),
        )


def user_fixture(user_id: str = "experiment-user") -> LatentUser:
    return LatentUser(
        user_id,
        (2, -1, 1),
        Susceptibility(ranking=0.35, default=0.80, suggestion=0.65),
        )


class WrongEventHistoryUpdater:
    """Fixture that returns a valid-length but incorrect consumed-event history."""

    updater_id = "wrong_event_history"
    view_kind = UpdateViewKind.FULL_CONTEXT

    def initial_state(self, prior: PreferenceBelief) -> UpdaterState:
        return UpdaterState(self.updater_id, prior)

    def update(self, state: UpdaterState, view: object) -> UpdateResult:
        event_id = str(getattr(view, "event_id"))
        next_state = UpdaterState(
            self.updater_id,
            state.belief,
            turn=state.turn + 1,
            event_ids=state.event_ids + (f"wrong:{event_id}",),
        )
        return UpdateResult(
            state=next_state,
            profile_update=ProfileUpdate(
                updater_id=self.updater_id,
                belief_before=state.belief.probabilities,
                belief_after=state.belief.probabilities,
                written_delta=("malformed event history",),
            ),
        )


class MatchedProvenanceTests(unittest.TestCase):
    def test_anchor_identity_attributes_and_probability_threshold(self) -> None:
        user = user_fixture()
        response = RandomUtilityModel()
        matched = build_matched_anchor_set(
            TRAVEL,
            target_attribute=0,
            anchor_direction=-1,
            scenario_id="invariant-anchor",
        )
        matched.validate_invariants()
        anchor = matched.anchor()
        self.assertEqual(set(matched.contexts), set(MECHANISMS))
        for context in matched.contexts.values():
            self.assertIs(context.option(matched.anchor_option_id), anchor)
            self.assertEqual(
                context.option(matched.anchor_option_id).features,
                anchor.features,
            )
        probabilities = matched.choice_probabilities(user, response)
        self.assertTrue(all(0.05 < value < 0.95 for value in probabilities.values()))
        self.assertTrue(
            matched.eligible(user, response, minimum_probability=0.05)
        )

    def test_matched_anchor_position_is_counterbalanced_with_ranking_reversed(
        self,
    ) -> None:
        matched = build_matched_anchor_set(
            TRAVEL,
            scenario_id="counterbalanced-anchor",
        )
        for anchor_first in (False, True):
            with self.subTest(anchor_first=anchor_first):
                reordered = matched.with_anchor_position(
                    anchor_first=anchor_first
                )
                expected_position = 0 if anchor_first else 1
                for mechanism, context in reordered.contexts.items():
                    self.assertEqual(
                        context.ranking.index(reordered.anchor_option_id),
                        (
                            1 - expected_position
                            if mechanism == "ranking"
                            else expected_position
                        ),
                    )
                self.assertEqual(
                    reordered.context("ranking").ranking,
                    tuple(reversed(reordered.context("balanced").ranking)),
                )
                reordered.validate_invariants()

    def test_update_views_enforce_context_and_provenance_boundaries(self) -> None:
        matched = build_matched_anchor_set(TRAVEL, scenario_id="views")
        context = matched.context("default")
        observation = matched.observation()
        action = SoftProfileConditionedPolicy().action(
            TRAVEL,
            PreferenceBelief.uniform(),
            turn=1,
            master_seed=4,
            trajectory_id="view-action",
        )
        response_only = make_update_view(
            UpdateViewKind.RESPONSE_ONLY,
            context,
            observation,
            action.provenance,
            event_id="response-only",
        )
        full_context = make_update_view(
            UpdateViewKind.FULL_CONTEXT,
            context,
            observation,
            action.provenance,
            event_id="full-context",
        )
        aware = make_update_view(
            UpdateViewKind.PROVENANCE_AWARE,
            context,
            observation,
            action.provenance,
            event_id="aware",
        )
        self.assertIsNone(response_only.context)
        self.assertIsNone(response_only.provenance)
        self.assertIs(full_context.context, context)
        self.assertIsNone(full_context.provenance)
        self.assertIs(aware.provenance, action.provenance)
        self.assertNotIn("policy_id", context.to_dict())

        with self.assertRaises(ValueError):
            ResponseOnlyUpdater().update(
                ResponseOnlyUpdater().initial_state(PreferenceBelief.uniform()),
                full_context,
            )

    def test_all_structured_updater_variants_follow_one_protocol(self) -> None:
        identifiers = (
            "no_update",
            "exact_action_aware",
            "fitted_action_aware",
            "fitted_action_unaware",
            "response_only",
            "full_context_blind",
            "provenance_discount",
            "provenance_aware",
            "conservative",
        )
        registry = build_updater_registry(identifiers)
        matched = build_matched_anchor_set(TRAVEL, scenario_id="all-updaters")
        context = matched.context("suggested")
        observation = matched.observation()
        provenance = SoftProfileConditionedPolicy().action(
            TRAVEL,
            PreferenceBelief.uniform(),
            turn=2,
            master_seed=5,
            trajectory_id="all-updaters",
        ).provenance
        prior = PreferenceBelief.uniform()
        for identifier, updater in registry.items():
            view = make_update_view(
                updater.view_kind,
                context,
                observation,
                provenance,
                event_id=f"event:{identifier}",
            )
            result = updater.update(updater.initial_state(prior), view)
            self.assertEqual(result.state.updater_id, identifier)
            self.assertEqual(result.state.turn, 1)
            self.assertEqual(result.state.event_ids, (f"event:{identifier}",))
            self.assertAlmostEqual(sum(result.state.belief.probabilities), 1.0)
            self.assertEqual(result.profile_update.updater_id, identifier)
            serialized = result.to_dict()["state"]
            if identifier == "exact_action_aware":
                self.assertIn("joint_belief", serialized)
                self.assertEqual(
                    len(serialized["joint_belief"]["probabilities"]),
                    64 * 3,
                )
        self.assertEqual(
            registry["no_update"].update(
                registry["no_update"].initial_state(prior),
                make_update_view(
                    registry["no_update"].view_kind,
                    context,
                    observation,
                    provenance,
                    event_id="no-update-check",
                ),
            ).state.belief,
            prior,
        )


class ExperimentATests(unittest.TestCase):
    def test_catalog_scenarios_and_anchor_positions_are_crossed(self) -> None:
        path = Path("data/scenarios/scenario-catalog-v1.json")
        catalog = load_scenario_catalog(
            path,
            expected_sha256=sha256(path.read_bytes()).hexdigest(),
        ).catalog
        users = tuple(
            LatentUser(
                f"allocation-user-{index:02d}",
                (2, -1, 1),
                Susceptibility(0.35, 0.45, 0.65),
            )
            for index in range(16)
        )
        result = run_provenance_audit(
            users=users,
            domains=(TRAVEL,),
            updaters={"no_update": NoUpdateUpdater()},
            mechanisms=("balanced",),
            response_modes=("controlled_anchor",),
            seed=1729,
            scenario_catalog=catalog,
        )

        by_user_target: dict[tuple[str, int], list[object]] = {}
        direction_counts: dict[tuple[int, str, int], int] = {}
        for row in result.rows:
            by_user_target.setdefault(
                (row.user_id, row.target_attribute),
                [],
            ).append(row)
            key = (
                row.target_attribute,
                row.context.scenario_id,
                row.anchor_direction,
            )
            direction_counts[key] = direction_counts.get(key, 0) + 1

        for paired in by_user_target.values():
            self.assertEqual(len(paired), 2)
            self.assertEqual(
                len({row.context.scenario_id for row in paired}),
                1,
            )
            self.assertEqual(
                {
                    row.context.ranking.index(row.selected_option_id)
                    for row in paired
                },
                {0, 1},
            )

        for target in range(3):
            scenario_ids = {
                scenario.scenario_id
                for scenario in catalog.eligible("travel", "test", target)
            }
            for direction in (-1, 1):
                counts = [
                    direction_counts.get((target, scenario_id, direction), 0)
                    for scenario_id in scenario_ids
                ]
                self.assertGreater(min(counts), 0)
                self.assertLessEqual(max(counts) - min(counts), 1)

    def test_same_response_audit_covers_ranking_and_natural_language_reply(
        self,
    ) -> None:
        catalog_path = Path("data/scenarios/scenario-catalog-v1.json")
        catalog = load_scenario_catalog(
            catalog_path,
            expected_sha256=sha256(catalog_path.read_bytes()).hexdigest(),
        ).catalog
        bank = load_conversation_bank(
            Path("data/scenarios/conversation-templates-v1.json")
        )
        result = run_provenance_audit(
            users=(user_fixture("same-response-user"),),
            domains=(TRAVEL,),
            updaters={"no_update": NoUpdateUpdater()},
            mechanisms=MECHANISMS,
            response_modes=("controlled_anchor",),
            minimum_probability=0.001,
            seed=23,
            scenario_catalog=catalog,
            conversation_bank=bank,
        )

        audit = result.same_response_audit()
        self.assertTrue(audit.passed)
        self.assertEqual(audit.failed_cell_count, 0)
        self.assertEqual(audit.required_mechanisms, MECHANISMS)
        self.assertEqual(len(audit.cells), 6)
        for cell in audit.cells:
            self.assertEqual(set(cell.observed_mechanisms), set(MECHANISMS))
            self.assertEqual(cell.selected_option_id, cell.anchor_option_id)
            self.assertRegex(cell.local_user_reply or "", r"^I choose .+\.$")
            self.assertTrue(all(cell.checks.values()))

        by_matched_set: dict[str, dict[str, object]] = {}
        for row in result.controlled_rows:
            key = (
                f"{row.user_id}:{row.target_attribute}:"
                f"{row.anchor_direction}:{row.prior_stratum}"
            )
            by_matched_set.setdefault(key, {})[row.mechanism] = row
        for mechanism_rows in by_matched_set.values():
            balanced = mechanism_rows["balanced"]
            ranking = mechanism_rows["ranking"]
            self.assertEqual(
                ranking.context.ranking,
                tuple(reversed(balanced.context.ranking)),
            )
            self.assertEqual(
                ranking.selected_option_id,
                balanced.selected_option_id,
            )
            self.assertEqual(
                ranking.observation.surface_response,
                balanced.observation.surface_response,
            )

        missing_ranking = audit_same_response_provenance(
            tuple(row for row in result.rows if row.mechanism != "ranking")
        )
        self.assertFalse(missing_ranking.passed)
        self.assertTrue(
            all(
                "mechanism_coverage" in cell.failed_checks
                for cell in missing_ranking.cells
            )
        )

        source = result.rows[0]
        other_option_id = next(
            option_id
            for option_id in source.context.option_ids
            if option_id != source.anchor_option_id
        )
        corrupted = replace(
            source,
            anchor_option_id=other_option_id,
            observation=replace(
                source.observation,
                surface_response="I choose an inconsistent local reply.",
            ),
            prior=PreferenceBelief.point_mass(user_fixture().theta),
        )
        corrupted_rows = (corrupted,) + result.rows[1:]
        corrupted_audit = audit_same_response_provenance(corrupted_rows)
        failed = next(
            cell
            for cell in corrupted_audit.cells
            if cell.matched_set_id
            == (
                f"{source.user_id}|{source.domain_id}|"
                f"attribute-{source.target_attribute}|"
                f"direction-{source.anchor_direction:+d}|"
                f"prior-{source.prior_stratum}|controlled_anchor"
            )
        )
        self.assertTrue(
            {
                "selected_anchor_identical",
                "local_user_reply_identical",
                "prior_invariant",
                "anchor_invariant",
            }
            <= set(failed.failed_checks)
        )
        serialized = audit.to_dict()
        self.assertEqual(
            serialized["analysis"],
            "experiment_a_same_response_audit",
        )
        self.assertTrue(serialized["passed"])

    def test_scenario_is_shared_across_users_but_trial_ids_are_unique(
        self,
    ) -> None:
        first = user_fixture()
        second = LatentUser(
            "second-audit-user",
            (-2, 1, 2),
            Susceptibility(0.35, 0.45, 0.65),
        )
        result = run_provenance_audit(
            users=(first, second),
            domains=(TRAVEL,),
            updaters={"no_update": NoUpdateUpdater()},
            mechanisms=("balanced",),
            response_modes=("controlled_anchor",),
            seed=19,
        )
        by_cell: dict[tuple[int, int], list[object]] = {}
        for row in result.rows:
            by_cell.setdefault(
                (row.target_attribute, row.anchor_direction),
                [],
            ).append(row)
        for rows in by_cell.values():
            self.assertEqual(len(rows), 2)
            self.assertEqual(
                len({row.context.scenario_id for row in rows}),
                1,
            )
            self.assertEqual(
                len({row.trial_id for row in rows}),
                2,
            )

    def test_prior_strength_factor_preserves_matched_context_and_response(self) -> None:
        result = run_provenance_audit(
            users=(user_fixture(),),
            domains=(TRAVEL,),
            updaters={"no_update": NoUpdateUpdater()},
            prior_strengths=(0.0, 0.7),
            mechanisms=("balanced",),
            response_modes=("naturally_sampled",),
            seed=17,
        )
        self.assertEqual(len(result.rows), 12)
        self.assertEqual(
            {row.prior_strength for row in result.rows},
            {0.0, 0.7},
        )
        grouped: dict[tuple[int, int], list[object]] = {}
        for row in result.rows:
            grouped.setdefault(
                (row.target_attribute, row.anchor_direction),
                [],
            ).append(row)
        for paired in grouped.values():
            self.assertEqual(len(paired), 2)
            first, second = paired
            self.assertEqual(first.context, second.context)
            self.assertEqual(first.observation, second.observation)
            self.assertNotEqual(first.prior, second.prior)
            self.assertNotEqual(first.trial_id, second.trial_id)
        self.assertEqual(
            [item["prior_strength"] for item in result.summary()["prior_strata"]],
            [0.0, 0.7],
        )

    def test_truth_aligned_prior_contains_no_hidden_joint_information(self) -> None:
        result = run_provenance_audit(
            users=(user_fixture(),),
            domains=(TRAVEL,),
            updaters={"no_update": NoUpdateUpdater()},
            prior_strengths=(0.7,),
            mechanisms=("balanced",),
            response_modes=("controlled_anchor",),
            seed=17,
        )
        row = result.rows[0]
        marginals = row.prior.marginals()
        reconstructed = marginals.independent_joint()
        for observed, expected in zip(
            row.prior.probabilities,
            reconstructed.probabilities,
        ):
            self.assertAlmostEqual(observed, expected, places=12)
        for theta, probability in zip(
            THETA_STATES,
            row.prior.probabilities,
        ):
            expected = 1.0
            for attribute, value in enumerate(theta):
                expected *= marginals.marginal(attribute)[
                    THETA_VALUES.index(value)
                ]
            self.assertAlmostEqual(probability, expected)

        llm = LLMReplayUpdater(
            "llm_full_context",
            UpdateViewKind.FULL_CONTEXT,
            object(),  # type: ignore[arg-type]
        )
        llm_state = llm.initial_state(row.prior)
        request = llm.build_request(
            llm_state,
            make_update_view(
                llm.view_kind,
                row.context,
                row.observation,
                row.provenance,
                event_id="prior-information-audit",
            ),
        )
        visible = request.payload["prior"]
        visible_rows = tuple(
            tuple(
                float(visible[f"attribute_{attribute + 1}"][label])
                for label in ("-2", "-1", "+1", "+2")
            )
            for attribute in range(3)
        )
        prompt_reconstruction = PreferenceBelief.from_marginals(
            MarginalPreferenceBelief(visible_rows)  # type: ignore[arg-type]
        )
        exact_state = ExactActionAwareUpdater().initial_state(row.prior)
        for observed, expected in zip(
            prompt_reconstruction.probabilities,
            row.prior.probabilities,
        ):
            self.assertAlmostEqual(observed, expected, places=12)
        self.assertEqual(llm_state.belief, row.prior)
        for observed, expected in zip(
            exact_state.belief.probabilities,
            prompt_reconstruction.probabilities,
        ):
            self.assertAlmostEqual(observed, expected, places=12)
        exact_theta_prior = (
            exact_state.joint_belief.theta_belief()  # type: ignore[union-attr]
        )
        for observed, expected in zip(
            exact_theta_prior.probabilities,
            prompt_reconstruction.probabilities,
        ):
            self.assertAlmostEqual(observed, expected, places=12)

    def test_fixed_control_battery_covers_all_proposal_controls(self) -> None:
        first = build_experiment_a_control_battery()
        second = build_experiment_a_control_battery()
        self.assertEqual(first, second)
        self.assertEqual(len(first.battery_sha256), 64)
        self.assertEqual(
            {case.polarity for case in first.cases},
            {"positive", "negative"},
        )
        self.assertEqual(
            {case.signal_kind for case in first.cases},
            {
                "explicit_volunteered_preference",
                "repeated_balanced_cross_context_choices",
                "direct_correction",
                "indifferent_response",
                "random_choice",
                "target_nondistinguishing_response",
            },
        )
        self.assertIn("not_scored", first.to_dict()["status"])

    def test_controlled_and_natural_rows_retain_complete_causal_chain(self) -> None:
        updaters = {
            updater.updater_id: updater
            for updater in (
                ResponseOnlyUpdater(),
                ProvenanceDiscountUpdater(),
            )
        }
        result = run_provenance_audit(
            users=(user_fixture(),),
            domains=(TRAVEL,),
            updaters=updaters,
            mechanisms=("balanced", "restricted"),
            response_modes=("controlled_anchor", "naturally_sampled"),
            seed=13,
        )
        self.assertEqual(len(result.rows), 48)
        self.assertEqual(len(result.controlled_rows), 24)
        self.assertEqual(len(result.natural_rows), 24)
        self.assertFalse(result.excluded)
        for row in result.rows:
            self.assertEqual(row.context.domain, row.domain_id)
            self.assertEqual(
                row.observation.selected_option_id,
                row.selected_option_id,
            )
            self.assertNotIn("policy_id", row.context.to_dict())
            self.assertAlmostEqual(
                row.brier - row.fitted_aware_brier,
                row.excess_brier,
            )
            self.assertAlmostEqual(
                row.brier,
                marginal_brier(row.posterior, user_fixture().theta),
            )
            self.assertAlmostEqual(
                row.update_magnitude,
                marginal_l1(row.prior, row.posterior) / 3.0,
            )
            # Structured choice is sampled first; absent text cannot add an
            # unsupported general-preference claim.
            self.assertIsNone(row.observation.surface_response)

        self.assertEqual(
            {row.anchor_direction for row in result.rows},
            {-1, 1},
        )
        grouped: dict[tuple[str, int, int, str], set[str]] = {}
        for row in result.controlled_rows:
            grouped.setdefault(
                (
                    row.user_id,
                    row.target_attribute,
                    row.anchor_direction,
                    row.updater_id,
                ),
                set(),
            ).add(row.selected_option_id)
        self.assertTrue(all(len(selected) == 1 for selected in grouped.values()))
        natural_noise_keys: dict[tuple[str, int, int], set[str]] = {}
        for row in result.natural_rows:
            natural_noise_keys.setdefault(
                (row.user_id, row.target_attribute, row.anchor_direction),
                set(),
            ).add(row.observation.choice_noise_key)
        self.assertTrue(
            all(
                len(keys)
                == len(
                    {
                        row.mechanism
                        for row in result.natural_rows
                    }
                )
                for keys in natural_noise_keys.values()
            )
        )

    def test_balanced_and_unapplied_soft_policy_share_neutral_ranking(self) -> None:
        belief = PreferenceBelief.uniform()
        common_key = "paired-neutral-policy"
        balanced = BalancedPolicy()
        soft = SoftProfileConditionedPolicy()
        candidate = None
        for seed in range(100):
            action = soft.action(
                TRAVEL,
                belief,
                turn=0,
                master_seed=seed,
                trajectory_id=common_key,
            )
            if not action.provenance.profile_conditioned:
                candidate = (seed, action)
                break
        self.assertIsNotNone(candidate)
        assert candidate is not None
        seed, soft_action = candidate
        balanced_action = balanced.action(
            TRAVEL,
            belief,
            turn=0,
            master_seed=seed,
            trajectory_id=common_key,
        )
        self.assertEqual(soft_action.context.ranking, balanced_action.context.ranking)
        self.assertEqual(
            soft_action.context.default_option_id,
            balanced_action.context.default_option_id,
        )
        self.assertEqual(
            soft_action.context.suggested_option_id,
            balanced_action.context.suggested_option_id,
        )


class ClosedLoopTests(unittest.TestCase):
    def test_decomposition_rejects_inconsistent_raw_operands(self) -> None:
        with self.assertRaisesRegex(ValueError, "decomposition identity failed"):
            DecompositionRow(
                domain_id="travel",
                user_id="user-1",
                initial_profile_condition="incorrect",
                updater_id="llm_full_context",
                replicate=0,
                profile_trajectory_id="soft",
                balanced_trajectory_id="balanced",
                evidence_selection_cost=0.1,
                profile_attribution_cost=0.2,
                balanced_attribution_cost=0.1,
                self_confirmation_interaction=0.1,
                soft_terminal_error=0.5,
                balanced_terminal_error=0.1,
                soft_terminal_shadow_error=0.1,
                balanced_terminal_shadow_error=0.0,
            )

    def test_trajectory_fails_closed_when_updater_history_differs_from_shadow(
        self,
    ) -> None:
        with self.assertRaisesRegex(RuntimeError, "consume the same event"):
            run_trajectory(
                user=user_fixture(),
                domain=TRAVEL,
                policy=BalancedPolicy(),
                updater=WrongEventHistoryUpdater(),
                turns=1,
                seed=19,
            )

    def test_same_sign_strengthening_is_not_action_influence(self) -> None:
        user = user_fixture("same-sign-user")
        initial = initial_profile_belief(
            user.theta, "incorrect", profile_strength=0.80
        )
        strengthened = initial_profile_belief(
            user.theta, "incorrect", profile_strength=0.95
        )
        trajectory = run_trajectory(
            user=user,
            domain=TRAVEL,
            policy=HardFilterPolicy(),
            updater=SameSignStrengtheningUpdater(strengthened),
            turns=6,
            seed=61,
            initial_belief=initial,
            initial_profile_condition="incorrect",
        )
        self.assertEqual(trajectory.preference_dimension_coverage, 0.0)
        self.assertIsNone(trajectory.turns_to_full_preference_coverage)
        self.assertTrue(
            all(
                turn.action_signature
                == turn.unstrengthened_action_signatures[attribute]
                for turn in trajectory.turns
                for attribute in range(3)
            )
        )
        assessments = assess_self_confirmation(
            trajectory,
            materially_wrong_mass=0.5,
            lcg_threshold=0.0,
        )
        self.assertTrue(
            all(
                not item.evidence.profile_changed_later_action
                for item in assessments
            )
        )
        self.assertFalse(any(item.reportable for item in assessments))

    def test_weakened_profile_is_not_strengthening_action_influence(self) -> None:
        user = user_fixture("strengthen-then-weaken-user")
        initial = initial_profile_belief(
            user.theta, "incorrect", profile_strength=0.80
        )
        strong = initial_profile_belief(
            user.theta, "incorrect", profile_strength=0.95
        )
        weak = initial_profile_belief(
            user.theta, "incorrect", profile_strength=0.60
        )
        trajectory = run_trajectory(
            user=user,
            domain=TRAVEL,
            policy=SoftProfileConditionedPolicy(),
            updater=StrengthenThenWeakenUpdater(strong, weak),
            turns=4,
            seed=1,
            initial_belief=initial,
            initial_profile_condition="incorrect",
            trajectory_id="x",
            crn_key="x",
        )
        assessment = assess_self_confirmation(
            trajectory,
            lcg_threshold=0.0,
        )[0]
        action_turn = trajectory.turns[3]
        self.assertLess(
            action_turn.wrong_mass_before[0],
            assessment.initial_wrong_mass,
        )
        self.assertTrue(
            action_turn.profile_attribute_influenced_action[0]
        )
        self.assertFalse(
            assessment.evidence.profile_changed_later_action
        )
        self.assertFalse(assessment.reportable)

    def test_fixed_theta_same_history_shadow_and_crn_reproducibility(self) -> None:
        user = user_fixture()
        kwargs = {
            "user": user,
            "domain": TRAVEL,
            "policy": SoftProfileConditionedPolicy(),
            "updater": FullContextBlindUpdater(),
            "turns": 5,
            "seed": 23,
            "initial_profile_condition": "incorrect",
            "response_model": RandomUtilityModel(),
            "crn_key": "paired-user-twin",
        }
        first = run_trajectory(
            **kwargs,
            trajectory_id="trajectory-first",
        )
        second = run_trajectory(
            **kwargs,
            trajectory_id="trajectory-second",
        )
        self.assertTrue(first.same_history_shadow)
        self.assertTrue(second.same_history_shadow)
        self.assertTrue(all(turn.theta_snapshot == user.theta for turn in first.turns))
        self.assertEqual(
            [turn.context_id for turn in first.turns],
            [turn.context_id for turn in second.turns],
        )
        self.assertEqual(
            [turn.selected_option_id for turn in first.turns],
            [turn.selected_option_id for turn in second.turns],
        )
        self.assertEqual(
            [turn.common_noise_key for turn in first.turns],
            [turn.common_noise_key for turn in second.turns],
        )
        self.assertNotIn("theta", first.audit_record.to_dict())
        retained = first.to_dict()
        self.assertIsNotNone(retained["terminal_shadow_joint_belief"])
        self.assertIsNotNone(
            retained["turns"][0]["shadow_joint_before"]
        )
        self.assertIsNotNone(
            retained["turns"][0]["shadow_joint_after"]
        )
        first_turn = first.turns[0]
        self.assertEqual(
            first_turn.information_gain_state_space,
            "theta_psi_joint",
        )
        self.assertAlmostEqual(
            first_turn.action_aware_information_gain,
            first_turn.shadow_joint_before.entropy()
            - first_turn.shadow_joint_after.entropy(),
        )
        self.assertEqual(
            first.ex_ante_preference_strengths_by_attribute,
            (2, 1, 1),
        )
        self.assertEqual(
            first.ex_ante_preference_strength_strata_by_attribute,
            ("strong", "weak", "weak"),
        )
        self.assertEqual(
            first.ex_ante_user_preference_strength_stratum,
            "mixed",
        )
        occupancy = summarize_prospective_strata_occupancy((first,))
        self.assertEqual(occupancy["strata_assignment_timing"], "before_natural_response")
        self.assertEqual(occupancy["unique_user_count"], 1)
        self.assertEqual(occupancy["trajectory_count"], 1)
        self.assertEqual(occupancy["turn_count"], 5)
        self.assertTrue(occupancy["coverage_flags"]["weak_attribute_present"])
        self.assertTrue(occupancy["coverage_flags"]["strong_attribute_present"])
        self.assertEqual(
            sum(occupancy["balanced_choice_margin_stratum_counts"].values()),
            5,
        )
        self.assertEqual(len(occupancy["cells"]), 1)
        self.assertEqual(
            occupancy["paper_mechanism_coverage_rule"][
                "minimum_informative_visible_divergences_per_soft_trajectory"
            ],
            2,
        )
        response = RandomUtilityModel()
        for turn in first.turns:
            with self.subTest(turn=turn.turn):
                self.assertEqual(
                    turn.ex_ante_target_preference_strength,
                    abs(user.theta[turn.target_attribute]),
                )
                self.assertEqual(
                    turn.ex_ante_target_preference_strength_stratum,
                    (
                        "strong"
                        if abs(user.theta[turn.target_attribute]) == 2
                        else "weak"
                    ),
                )
                balanced = BalancedPolicy().action(
                    TRAVEL,
                    turn.belief_before,
                    turn=turn.turn,
                    master_seed=23,
                    trajectory_id="paired-user-twin",
                )
                probabilities = sorted(
                    response.probabilities(
                        user.theta,
                        user.susceptibility,
                        balanced.context,
                    ),
                    reverse=True,
                )
                self.assertAlmostEqual(
                    turn.ex_ante_balanced_choice_probability_margin,
                    probabilities[0] - probabilities[1],
                )
                self.assertIn(
                    turn.ex_ante_balanced_choice_margin_stratum,
                    {"near_tie", "marginal", "decisive"},
                )

    def test_policy_action_uses_profile_not_latent_user(self) -> None:
        policy = SoftProfileConditionedPolicy()
        negative_profile = PreferenceBelief.from_marginals(
            __import__(
                "cape_loop.beliefs",
                fromlist=["MarginalPreferenceBelief"],
            ).MarginalPreferenceBelief(
                (
                    (0.45, 0.45, 0.05, 0.05),
                    (0.25, 0.25, 0.25, 0.25),
                    (0.25, 0.25, 0.25, 0.25),
                )
            )
        )
        positive_profile = PreferenceBelief.from_marginals(
            __import__(
                "cape_loop.beliefs",
                fromlist=["MarginalPreferenceBelief"],
            ).MarginalPreferenceBelief(
                (
                    (0.05, 0.05, 0.45, 0.45),
                    (0.25, 0.25, 0.25, 0.25),
                    (0.25, 0.25, 0.25, 0.25),
                )
            )
        )
        negative = policy.action(
            TRAVEL,
            negative_profile,
            turn=0,
            master_seed=8,
            trajectory_id="policy-boundary",
        )
        repeated = policy.action(
            TRAVEL,
            negative_profile,
            turn=0,
            master_seed=8,
            trajectory_id="policy-boundary",
        )
        positive = policy.action(
            TRAVEL,
            positive_profile,
            turn=0,
            master_seed=8,
            trajectory_id="policy-boundary",
        )
        self.assertEqual(negative, repeated)
        self.assertNotEqual(negative.signature(), positive.signature())

    def test_exploratory_policy_bounds_target_exposure(self) -> None:
        policy = ExploratoryPolicy()
        counts = [0, 0, 0]
        for turn in range(12):
            action = policy.action(
                TRAVEL,
                PreferenceBelief.uniform(),
                turn=turn,
                master_seed=8,
                trajectory_id="balanced-exploration",
                target_counts=tuple(counts),
            )
            target = action.context.target_attribute
            self.assertIsNotNone(target)
            counts[int(target)] += 1
            self.assertLessEqual(max(counts) - min(counts), 1)
        self.assertEqual(counts, [4, 4, 4])

        for turn in range(12, 16):
            action = policy.action(
                TRAVEL,
                PreferenceBelief.uniform(),
                turn=turn,
                master_seed=8,
                trajectory_id="balanced-exploration",
                target_counts=tuple(counts),
            )
            target = action.context.target_attribute
            self.assertIsNotNone(target)
            counts[int(target)] += 1
            self.assertLessEqual(max(counts) - min(counts), 1)
        self.assertEqual(sorted(counts), [5, 5, 6])
        self.assertEqual(max(counts), 6)

    def test_exploratory_counts_reach_both_production_policy_callers(self) -> None:
        entropy_skewed = PreferenceBelief.from_marginals(
            MarginalPreferenceBelief(
                (
                    (0.49, 0.49, 0.01, 0.01),
                    (0.40, 0.40, 0.10, 0.10),
                    (0.25, 0.25, 0.25, 0.25),
                )
            )
        )
        trajectory = run_trajectory(
            user=user_fixture(),
            domain=TRAVEL,
            policy=ExploratoryPolicy(),
            updater=NoUpdateUpdater(),
            turns=12,
            seed=8,
            initial_belief=entropy_skewed,
            trajectory_id="balanced-exploration-closed-loop",
        )
        closed_loop_counts = [
            sum(turn.target_attribute == target for turn in trajectory.turns)
            for target in range(3)
        ]
        self.assertEqual(closed_loop_counts, [4, 4, 4])

        history = generate_fixed_history(
            user=user_fixture(),
            domain=TRAVEL,
            policy=ExploratoryPolicy(),
            turns=16,
            seed=8,
            reference_belief=entropy_skewed,
            history_id="balanced-exploration-fixed-history",
        )
        fixed_history_counts = [
            sum(
                event.context.target_attribute == target
                for event in history.events
            )
            for target in range(3)
        ]
        self.assertEqual(sorted(fixed_history_counts), [5, 5, 6])
        self.assertEqual(max(fixed_history_counts), 6)

    def test_experiment_b_crossing_decomposition_and_report_predicate(self) -> None:
        updater_list = (
            FittedActionAwareUpdater(),
            FullContextBlindUpdater(),
        )
        updaters = {updater.updater_id: updater for updater in updater_list}
        policies = {
            "balanced": BalancedPolicy(),
            "soft_profile_conditioned": SoftProfileConditionedPolicy(),
            "exploratory": ExploratoryPolicy(),
        }
        result = run_experiment_b(
            users=(user_fixture(),),
            domains=(TRAVEL,),
            updaters=updaters,
            policies=policies,
            initial_profile_conditions=("incorrect",),
            turns=4,
            trajectories_per_cell=1,
            seed=31,
        )
        self.assertEqual(len(result.trajectories), 6)
        self.assertEqual(len(result.decompositions), 2)
        self.assertEqual(len(result.self_confirmation_assessments), 18)
        for trajectory in result.trajectories:
            self.assertTrue(trajectory.same_history_shadow)
            self.assertGreaterEqual(
                trajectory.terminal_shadow_to_system_marginal_kl,
                0.0,
            )
            self.assertEqual(trajectory.preference_dimension_coverage, 1.0)
            self.assertEqual(
                trajectory.turns_to_full_preference_coverage,
                3,
            )
            self.assertGreater(trajectory.displayed_option_diversity, 0.0)
            self.assertGreaterEqual(trajectory.selected_option_count, 1)
            self.assertGreaterEqual(
                trajectory.profile_conditioned_exposure_rate,
                0.0,
            )
            self.assertGreaterEqual(trajectory.presentation_mechanism_count, 1)
            self.assertGreaterEqual(
                trajectory.presentation_mechanism_evenness,
                0.0,
            )
            self.assertAlmostEqual(
                trajectory.error_amplification_ratio,
                trajectory.terminal_error / trajectory.initial_error,
            )
            self.assertEqual(
                len(trajectory.reinforcement_event_flags()),
                len(trajectory.turns),
            )
            self.assertGreaterEqual(
                trajectory.profile_aligned_treatment_opportunities,
                trajectory.reinforcement_event_count,
            )
            self.assertGreaterEqual(
                trajectory.cumulative_expected_information_gain,
                0.0,
            )
            self.assertGreaterEqual(
                trajectory.mean_profile_consistency_score,
                -1.0,
            )
            self.assertLessEqual(
                trajectory.mean_profile_consistency_score,
                1.0,
            )
            self.assertLessEqual(
                trajectory.disconfirmation_inversion_count,
                trajectory.disconfirmation_opportunity_count,
            )
            if trajectory.disconfirmation_opportunity_count == 0:
                self.assertIsNone(trajectory.disconfirmation_inversion_rate)
        for row in result.decompositions:
            self.assertIsNotNone(row.exploratory_trajectory_id)
            self.assertIsNotNone(
                row.action_aware_information_gain_deficit
            )
            self.assertIsNotNone(
                row.disconfirmation_evidence_deficit_log_odds
            )
            self.assertIsNotNone(
                row.balanced_action_aware_information_gain_deficit
            )
            self.assertIsNotNone(
                row.balanced_disconfirmation_evidence_deficit_log_odds
            )
            self.assertIsNotNone(row.exploratory_attribution_cost)
            self.assertIsNotNone(
                row.soft_minus_exploratory_attribution_gap
            )
            self.assertAlmostEqual(
                row.soft_minus_balanced_attribution_gap,
                row.self_confirmation_interaction,
            )
        for reported in result.reportable_self_confirming:
            self.assertTrue(all(reported.evidence.clauses().values()))
        self.assertTrue(
            all(
                isinstance(assessment.false_stable, bool)
                for assessment in result.self_confirmation_assessments
            )
        )

    def test_disconfirmation_inversion_uses_exact_opportunities(self) -> None:
        trajectory = run_trajectory(
            user=user_fixture(),
            domain=TRAVEL,
            policy=BalancedPolicy(),
            updater=FullContextBlindUpdater(),
            turns=2,
            seed=43,
            initial_profile_condition="incorrect",
        )
        attribute = trajectory.initially_false_attributes[0]
        modified_turns = []
        for index, turn in enumerate(trajectory.turns):
            shadow_gain = [0.0, 0.0, 0.0]
            system_gain = [0.0, 0.0, 0.0]
            if index == 0:
                shadow_gain[attribute] = -0.20
                system_gain[attribute] = 0.10
            modified_turns.append(
                replace(
                    turn,
                    shadow_false_confidence_gain=tuple(shadow_gain),
                    system_false_confidence_gain=tuple(system_gain),
                )
            )
        inverted = replace(trajectory, turns=tuple(modified_turns))
        self.assertEqual(inverted.disconfirmation_opportunity_count, 1)
        self.assertEqual(inverted.disconfirmation_inversion_count, 1)
        self.assertEqual(inverted.disconfirmation_inversion_rate, 1.0)
        self.assertEqual(
            inverted.disconfirmation_inversion_turn_flags(),
            (True, False),
        )

        no_opportunity = replace(
            trajectory,
            turns=tuple(
                replace(
                    turn,
                    shadow_false_confidence_gain=(0.0, 0.0, 0.0),
                    system_false_confidence_gain=(0.0, 0.0, 0.0),
                )
                for turn in trajectory.turns
            ),
        )
        self.assertEqual(no_opportunity.disconfirmation_opportunity_count, 0)
        self.assertIsNone(no_opportunity.disconfirmation_inversion_rate)


class FixedReplayAndEvaluationTests(unittest.TestCase):
    def test_static_history_is_replayed_identically_across_updaters(self) -> None:
        history = generate_fixed_history(
            user=user_fixture(),
            domain=TRAVEL,
            policy=BalancedPolicy(),
            turns=5,
            seed=17,
            history_id="canonical-static-history",
        )
        first = replay_history(history, ResponseOnlyUpdater())
        second = replay_history(history, FittedActionUnawareUpdater())
        self.assertEqual(first.history_digest, second.history_digest)
        self.assertEqual(first.event_signatures, second.event_signatures)
        self.assertEqual(
            [
                (
                    item.context.to_dict(),
                    item.provenance.to_dict(),
                    item.observation.to_dict(),
                )
                for item in first.audit_record.interactions
            ],
            [
                (
                    item.context.to_dict(),
                    item.provenance.to_dict(),
                    item.observation.to_dict(),
                )
                for item in second.audit_record.interactions
            ],
        )

    def test_terminal_battery_and_ranking_analysis_are_system_independent(
        self,
    ) -> None:
        dev = user_fixture("development-user")
        test = LatentUser(
            "test-user",
            (-1, 2, -2),
            Susceptibility(ranking=0.2, default=0.5, suggestion=0.7),
        )
        updaters = {
            updater.updater_id: updater
            for updater in (
                ResponseOnlyUpdater(),
                ProvenanceAwareUpdater(),
                EpisodicMemoryUpdater(),
            )
        }
        kwargs = {
            "development_users": (dev,),
            "test_users": (test,),
            "domains": (TRAVEL,),
            "updaters": updaters,
            "turns": 4,
            "trajectories_per_cell": 2,
            "seed": 29,
            "bootstrap_replicates": 20,
        }
        first = run_experiment_c(**kwargs)
        second = run_experiment_c(**kwargs)
        first.assert_static_replay_identity()
        first.assert_terminal_battery_identity()
        self.assertEqual(len(first.fixed_histories), 8)
        self.assertEqual(len(first.replay_results), 24)
        self.assertEqual(len(first.endogenous_trajectories), 12)
        self.assertEqual(len(first.rows), 36)
        self.assertEqual(first.rankings, second.rankings)
        self.assertEqual(
            {row.battery_digest for row in first.rows},
            {first.terminal_batteries[0].battery_digest},
        )
        for row in first.rows:
            self.assertEqual(
                row.system_projection_score.evaluated_item_count,
                len(first.terminal_batteries[0].items),
            )
            self.assertGreaterEqual(
                row.system_projection_score.predicted_utility_tie_count,
                0.0,
            )
            self.assertGreaterEqual(
                row.system_projection_score.intrinsic_utility_tie_count,
                0.0,
            )
            self.assertGreaterEqual(
                row.system_projection_score.fractional_behavioral_accuracy,
                0.0,
            )
            self.assertLessEqual(
                row.system_projection_score.fractional_behavioral_accuracy,
                row.system_projection_score.behavioral_accuracy,
            )
            self.assertIsNotNone(row.system_projection_score.profile_ece)
            self.assertEqual(
                row.system_projection_score.profile_calibration_prediction_count,
                3,
            )
            self.assertEqual(
                sum(
                    item.prediction_count
                    for item in row.system_projection_score.profile_reliability_bins
                ),
                3,
            )
            self.assertIsNotNone(row.ranking_score)
            self.assertIsNotNone(row.ranking_score.profile_ece)
        native_rows = [
            row for row in first.rows if row.updater_id == "episodic_memory"
        ]
        self.assertTrue(native_rows)
        self.assertTrue(
            all(len(row.native_decoder_evaluations) == 2 for row in native_rows)
        )
        self.assertTrue(
            all(
                row.score_basis == "mean_of_two_blinded_native_decoders"
                for row in native_rows
            )
        )
        for row in native_rows:
            decoder_mean = sum(
                evaluation.score.profile_brier
                for evaluation in row.native_decoder_evaluations
            ) / 2
            self.assertAlmostEqual(row.profile_error, decoder_mean)
            self.assertEqual(row.predicted_option_ids, ())
        self.assertTrue(
            all(
                row.score_basis == "system_structured_projection"
                for row in first.rows
                if row.updater_id != "episodic_memory"
            )
        )

        static_row = next(
            row for row in first.rows if row.regime == "fixed_balanced"
        )
        tampered = replace(static_row, history_digest="not-a-real-history")
        tampered_result = replace(
            first,
            rows=tuple(
                tampered if row is static_row else row for row in first.rows
            ),
        )
        with self.assertRaises(AssertionError):
            tampered_result.assert_static_replay_identity()

    def test_terminal_battery_definition_does_not_take_system_or_policy(self) -> None:
        first = build_terminal_battery(TRAVEL)
        second = build_terminal_battery(TRAVEL)
        self.assertEqual(first, second)
        self.assertTrue(
            all(item.context.default_option_id is None for item in first.items)
        )
        self.assertTrue(
            all(item.context.suggested_option_id is None for item in first.items)
        )
        training_ids = {
            option.option_id
            for option in TRAVEL.option_pool + TRAVEL.isolated_options
        }
        training_features = {
            tuple(option.features)
            for option in TRAVEL.option_pool + TRAVEL.isolated_options
        }
        self.assertFalse(
            {
                option.option_id
                for item in first.items
                for option in item.context.options
            }
            & training_ids
        )
        self.assertFalse(
            {
                tuple(option.features)
                for item in first.items
                for option in item.context.options
            }
            & training_features
        )


if __name__ == "__main__":
    unittest.main()
