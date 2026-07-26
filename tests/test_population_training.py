from __future__ import annotations

import unittest

from cape_loop.beliefs import PreferenceBelief
from cape_loop.domains import (
    dialogue_template_id,
    domain_for_split,
    get_domain,
    option_template_id,
    scenario_family_id,
)
from cape_loop.elicitation import build_matched_anchor_set
from cape_loop.metrics import marginal_brier
from cape_loop.population import (
    generate_users,
    initial_profile_belief,
    susceptibility_group_id,
    theta_group_id,
    user_state_record,
    wrong_directions,
)
from cape_loop.response import RandomUtilityModel
from cape_loop.schemas import (
    LatentUser,
    Observation,
    PolicyProvenance,
    Susceptibility,
)
from cape_loop.splits import build_split_manifest
from cape_loop.training import (
    fit_model_bundle,
    generate_training_examples,
    held_out_response_scores,
)
from cape_loop.updaters import (
    FittedActionAwareUpdater,
    FittedActionUnawareUpdater,
    make_update_view,
)


class PopulationTests(unittest.TestCase):
    def test_split_domains_are_feature_matched_but_surface_disjoint(self) -> None:
        base = get_domain("travel")
        variants = {
            split: domain_for_split(base, split)
            for split in ("train", "development", "test")
        }
        option_ids = {
            split: {
                option.option_id
                for option in (
                    *domain.option_pool,
                    *domain.isolated_options,
                )
            }
            for split, domain in variants.items()
        }
        self.assertFalse(option_ids["train"] & option_ids["development"])
        self.assertFalse(option_ids["train"] & option_ids["test"])
        self.assertFalse(option_ids["development"] & option_ids["test"])
        for split, domain in variants.items():
            self.assertTrue(
                all(
                    option.option_id.startswith(
                        option_template_id("travel", split) + ":"
                    )
                    for option in (
                        *domain.option_pool,
                        *domain.isolated_options,
                    )
                )
            )
            self.assertEqual(
                tuple(option.features for option in domain.option_pool),
                tuple(option.features for option in base.option_pool),
            )

    def test_training_generator_consumes_split_surface_families(self) -> None:
        manifest = build_split_manifest(seed=31)
        users = generate_users(
            domain_id="shared",
            count=2,
            split="development",
            manifest=manifest,
            seed=31,
        )
        domain = domain_for_split(get_domain("travel"), "development")
        rows = generate_training_examples(
            domain,
            users,
            RandomUtilityModel(),
            count=4,
            seed=31,
            split="development",
        )
        self.assertTrue(
            all(
                row.context.wording_template
                == dialogue_template_id("travel", "development")
                for row in rows
            )
        )
        self.assertTrue(
            all(
                row.context.scenario_id.startswith(
                    scenario_family_id("travel", "development") + ":"
                )
                for row in rows
            )
        )

    def test_groups_and_generated_users_are_split_disjoint(self) -> None:
        manifest = build_split_manifest(seed=17)
        by_split = {
            split: generate_users(
                domain_id="travel",
                count=5,
                split=split,
                manifest=manifest,
                seed=17,
            )
            for split in ("train", "development", "test")
        }
        for split, users in by_split.items():
            for user in users:
                self.assertEqual(manifest.theta_groups[theta_group_id(user.theta)], split)
                self.assertEqual(
                    manifest.susceptibility_groups[
                        susceptibility_group_id(user.susceptibility)
                    ],
                    split,
                )
        theta_sets = {
            split: {user.theta for user in users}
            for split, users in by_split.items()
        }
        self.assertFalse(theta_sets["train"] & theta_sets["test"])
        record = user_state_record(
            by_split["test"][0], domain_id="travel", split="test"
        )
        self.assertEqual(record["schema_version"], 1)
        self.assertEqual(record["split"], "test")

    def test_initial_seed_semantics(self) -> None:
        truth = (-2, 1, 2)
        correct = initial_profile_belief(truth, "correct")
        incorrect = initial_profile_belief(truth, "incorrect")
        empty = initial_profile_belief(truth, "empty")
        self.assertGreater(correct.sign_mass(0, -1), 0.79)
        self.assertGreater(incorrect.sign_mass(0, 1), 0.79)
        self.assertAlmostEqual(empty.sign_mass(0, -1), 0.5)
        self.assertEqual(wrong_directions(truth), (1, -1, -1))


class TrainingTests(unittest.TestCase):
    def test_training_crosses_each_user_with_every_mechanism(self) -> None:
        users = (
            LatentUser(
                "negative",
                (-2, -2, -2),
                Susceptibility(0.2, 0.3, 0.4),
            ),
            LatentUser(
                "positive",
                (2, 2, 2),
                Susceptibility(0.5, 0.6, 0.7),
            ),
        )
        examples = generate_training_examples(
            get_domain("travel"),
            users,
            RandomUtilityModel(),
            count=8,
            seed=11,
        )

        def mechanism(example: object) -> str:
            context = getattr(example, "context")
            target = context.target_attribute
            if context.default_option_id is not None:
                return "default"
            if context.suggested_option_id is not None:
                return "suggested"
            directions = {
                -1 if option.features[target] < 0 else 1
                for option in context.options
            }
            return "balanced" if directions == {-1, 1} else "restricted"

        by_theta = {
            user.theta: frozenset(
                mechanism(example)
                for example in examples
                if example.theta == user.theta
            )
            for user in users
        }
        self.assertEqual(
            set(by_theta.values()),
            {
                frozenset(
                    {"balanced", "restricted", "default", "suggested"}
                )
            },
        )

    def test_training_is_deterministic_and_serializable(self) -> None:
        manifest = build_split_manifest(seed=22)
        users = generate_users(
            domain_id="travel",
            count=16,
            split="train",
            manifest=manifest,
            seed=22,
        )
        domain = get_domain("travel")
        response = RandomUtilityModel(
            beta=1.0,
            ranking_scale=0.35,
            default_scale=0.8,
            suggestion_scale=0.65,
        )
        first = generate_training_examples(
            domain, users, response, count=96, seed=22
        )
        second = generate_training_examples(
            domain, users, response, count=96, seed=22
        )
        self.assertEqual(first, second)
        bundle = fit_model_bundle(
            first, seed=22, fit_steps=80, learning_rate=0.03
        )
        payload = bundle.to_dict()
        self.assertEqual(payload["training_examples"], 96)
        scores = held_out_response_scores(bundle, first[-16:])
        self.assertTrue(scores["aware_option_nll"] >= 0)
        self.assertFalse(scores["scores_share_outcome_space"])

    def test_fitted_aware_beats_unaware_on_identifying_holdout(self) -> None:
        levels = (0.5, 1.0, 1.5)
        manifest = build_split_manifest(
            seed=77,
            susceptibility_levels=levels,
        )
        train_users = generate_users(
            domain_id="identifying",
            count=36,
            split="train",
            manifest=manifest,
            susceptibility_levels=levels,
            seed=77,
        )
        test_users = generate_users(
            domain_id="identifying",
            count=12,
            split="test",
            manifest=manifest,
            susceptibility_levels=levels,
            seed=77,
        )
        domain = get_domain("travel")
        response = RandomUtilityModel(
            beta=1.5,
            ranking_scale=1.0,
            default_scale=5.0,
            suggestion_scale=5.0,
        )
        fitted = fit_model_bundle(
            generate_training_examples(
                domain,
                train_users,
                response,
                count=384,
                seed=77,
            ),
            seed=77,
            fit_steps=300,
            learning_rate=0.03,
        )
        updaters = (
            FittedActionAwareUpdater(fitted.aware),
            FittedActionUnawareUpdater(fitted.unaware),
        )
        errors = [0.0, 0.0]
        units = 0
        prior = PreferenceBelief.uniform()
        provenance = PolicyProvenance("identifying-fixture", "v1")
        for user_index, user in enumerate(test_users):
            for attribute in range(3):
                matched = build_matched_anchor_set(
                    domain,
                    target_attribute=attribute,
                    anchor_direction=(
                        -1 if (user_index + attribute) % 2 else 1
                    ),
                    scenario_id=f"heldout:{user_index}:{attribute}",
                )
                for mechanism in ("default", "suggested"):
                    context = matched.context(mechanism)
                    probabilities = response.probabilities(
                        user.theta,
                        user.susceptibility,
                        context,
                    )
                    for option, probability in zip(
                        context.options,
                        probabilities,
                    ):
                        observation = Observation(option.option_id)
                        for updater_index, updater in enumerate(updaters):
                            view = make_update_view(
                                updater.view_kind,
                                context,
                                observation,
                                provenance,
                                event_id=(
                                    f"{user_index}:{attribute}:{mechanism}:"
                                    f"{option.option_id}:{updater_index}"
                                ),
                            )
                            posterior = updater.update(
                                updater.initial_state(prior),
                                view,
                            ).state.belief
                            errors[updater_index] += (
                                probability
                                * marginal_brier(posterior, user.theta)
                            )
                    units += 1
        aware_error, unaware_error = (
            value / units for value in errors
        )
        self.assertLess(aware_error, unaware_error - 0.001)


if __name__ == "__main__":
    unittest.main()
