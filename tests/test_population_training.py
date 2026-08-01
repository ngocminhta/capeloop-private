from __future__ import annotations

from collections import Counter
import unittest

from cape_loop.beliefs import PreferenceBelief
from cape_loop.config import AppConfig, ConfigError, PopulationSection
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
    joint_cross_balance_score,
    susceptibility_grid,
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
from cape_loop.splits import (
    BALANCED_SUSCEPTIBILITY_POLICY,
    BALANCED_THETA_POLICY,
    LEGACY_SUSCEPTIBILITY_POLICY,
    LEGACY_THETA_POLICY,
    build_split_manifest,
    orthogonal_susceptibility_group_order,
    orthogonal_theta_group_order,
)
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
    @staticmethod
    def _susceptibility_tuple(user: LatentUser) -> tuple[float, float, float]:
        susceptibility = user.susceptibility
        return (
            susceptibility.ranking,
            susceptibility.default,
            susceptibility.suggestion,
        )

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

    def test_legacy_susceptibility_policy_is_exactly_backwards_compatible(
        self,
    ) -> None:
        implicit = build_split_manifest(seed=1729)
        explicit = build_split_manifest(
            seed=1729,
            susceptibility_policy=LEGACY_SUSCEPTIBILITY_POLICY,
        )
        self.assertEqual(implicit, explicit)
        self.assertEqual(implicit.to_dict()["schema_version"], 1)
        self.assertNotIn("susceptibility_policy", implicit.to_dict())
        users = generate_users(
            domain_id="shared",
            count=8,
            split="test",
            manifest=explicit,
            seed=1729,
        )
        self.assertEqual(
            tuple(self._susceptibility_tuple(user) for user in users),
            (
                (0.45, 0.85, 0.15),
                (0.85, 0.85, 0.45),
                (0.85, 0.45, 0.15),
                (0.45, 0.85, 0.15),
                (0.85, 0.85, 0.45),
                (0.85, 0.45, 0.15),
                (0.85, 0.85, 0.45),
                (0.45, 0.85, 0.15),
            ),
        )
        self.assertEqual(
            tuple(user.theta for user in users),
            (
                (2, 1, 1),
                (-1, -1, 1),
                (1, -2, 2),
                (-1, 2, -2),
                (1, 2, -1),
                (2, 1, 1),
                (1, -2, 2),
                (1, -2, 2),
            ),
        )

    def test_balanced_theta_policy_partitions_four_cubed_grid_orthogonally(
        self,
    ) -> None:
        values = (-2, -1, 1, 2)
        manifest = build_split_manifest(
            seed=1729,
            theta_values=values,
            theta_policy=BALANCED_THETA_POLICY,
        )
        manifest.assert_disjoint()
        self.assertEqual(manifest.to_dict()["schema_version"], 2)
        self.assertEqual(
            manifest.to_dict()["theta_policy"],
            BALANCED_THETA_POLICY,
        )
        split_profiles = {}
        for split, expected_support, expected_repetitions in (
            ("train", 32, 8),
            ("development", 16, 4),
            ("test", 16, 4),
        ):
            profiles = {
                tuple(int(value) for value in group_id.split(","))
                for group_id, assigned_split in (
                    manifest.theta_groups.items()
                )
                if assigned_split == split
            }
            split_profiles[split] = profiles
            self.assertEqual(len(profiles), expected_support)
            for coordinate in range(3):
                self.assertEqual(
                    Counter(profile[coordinate] for profile in profiles),
                    Counter(
                        {
                            value: expected_repetitions
                            for value in values
                        }
                    ),
                )
            expected_pair_repetitions = expected_support // 16
            for first, second in ((0, 1), (0, 2), (1, 2)):
                self.assertEqual(
                    Counter(
                        (profile[first], profile[second])
                        for profile in profiles
                    ),
                    Counter(
                        {
                            (left, right): expected_pair_repetitions
                            for left in values
                            for right in values
                        }
                    ),
                )
        self.assertFalse(
            split_profiles["train"] & split_profiles["development"]
        )
        self.assertFalse(split_profiles["train"] & split_profiles["test"])
        self.assertFalse(
            split_profiles["development"] & split_profiles["test"]
        )
        self.assertEqual(
            len(set().union(*split_profiles.values())),
            64,
        )

    def test_balanced_policy_partitions_three_cubed_grid_orthogonally(
        self,
    ) -> None:
        levels = (0.15, 0.45, 0.85)
        manifest = build_split_manifest(
            seed=1729,
            susceptibility_levels=levels,
            susceptibility_policy=BALANCED_SUSCEPTIBILITY_POLICY,
        )
        manifest.assert_disjoint()
        self.assertEqual(manifest.to_dict()["schema_version"], 2)
        self.assertEqual(
            manifest.to_dict()["susceptibility_policy"],
            BALANCED_SUSCEPTIBILITY_POLICY,
        )
        split_profiles = {}
        for split in ("train", "development", "test"):
            profiles = {
                (
                    susceptibility.ranking,
                    susceptibility.default,
                    susceptibility.suggestion,
                )
                for susceptibility in susceptibility_grid(levels)
                if manifest.susceptibility_groups[
                    susceptibility_group_id(susceptibility)
                ]
                == split
            }
            split_profiles[split] = profiles
            self.assertEqual(len(profiles), 9)
            for coordinate in range(3):
                self.assertEqual(
                    Counter(profile[coordinate] for profile in profiles),
                    Counter({level: 3 for level in levels}),
                )
            for first, second in ((0, 1), (0, 2), (1, 2)):
                self.assertEqual(
                    Counter(
                        (profile[first], profile[second])
                        for profile in profiles
                    ),
                    Counter(
                        (left, right)
                        for left in levels
                        for right in levels
                    ),
                )
        self.assertFalse(
            split_profiles["train"] & split_profiles["development"]
        )
        self.assertFalse(split_profiles["train"] & split_profiles["test"])
        self.assertFalse(
            split_profiles["development"] & split_profiles["test"]
        )
        self.assertEqual(
            len(set().union(*split_profiles.values())),
            27,
        )

    def test_balanced_policy_uses_deterministic_blocked_round_robin(
        self,
    ) -> None:
        levels = (0.15, 0.45, 0.85)
        manifest = build_split_manifest(
            seed=1729,
            susceptibility_levels=levels,
            susceptibility_policy=BALANCED_SUSCEPTIBILITY_POLICY,
        )
        for count in (4, 8, 10, 16, 18, 20, 24, 32):
            first = generate_users(
                domain_id="shared",
                count=count,
                split="test",
                manifest=manifest,
                susceptibility_levels=levels,
                seed=1729,
            )
            second = generate_users(
                domain_id="shared",
                count=count,
                split="test",
                manifest=manifest,
                susceptibility_levels=levels,
                seed=1729,
            )
            self.assertEqual(first, second)
            profile_counts = Counter(
                self._susceptibility_tuple(user) for user in first
            )
            support_counts = tuple(
                profile_counts.get(profile, 0)
                for profile in {
                    self._susceptibility_tuple(user)
                    for user in generate_users(
                        domain_id="shared",
                        count=9,
                        split="test",
                        manifest=manifest,
                        susceptibility_levels=levels,
                        seed=1729,
                    )
                }
            )
            self.assertLessEqual(max(support_counts) - min(support_counts), 1)
            if count >= 9:
                self.assertEqual(len(profile_counts), 9)
            for coordinate in range(3):
                marginal = Counter(
                    self._susceptibility_tuple(user)[coordinate]
                    for user in first
                )
                counts = tuple(marginal[level] for level in levels)
                self.assertLessEqual(max(counts) - min(counts), 1)
        exactly_balanced = generate_users(
            domain_id="shared",
            count=18,
            split="test",
            manifest=manifest,
            susceptibility_levels=levels,
            seed=1729,
        )
        expected_mean = sum(levels) / len(levels)
        for coordinate in range(3):
            self.assertAlmostEqual(
                sum(
                    self._susceptibility_tuple(user)[coordinate]
                    for user in exactly_balanced
                )
                / len(exactly_balanced),
                expected_mean,
            )

    def test_joint_balanced_policy_balances_theta_and_susceptibility_prefixes(
        self,
    ) -> None:
        theta_values = (-2, -1, 1, 2)
        susceptibility_levels = (0.15, 0.45, 0.85)
        manifest = build_split_manifest(
            seed=1729,
            theta_values=theta_values,
            theta_policy=BALANCED_THETA_POLICY,
            susceptibility_levels=susceptibility_levels,
            susceptibility_policy=BALANCED_SUSCEPTIBILITY_POLICY,
        )
        for split, count in (
            ("train", 24),
            ("development", 8),
            ("test", 4),
            ("test", 8),
            ("test", 10),
            ("test", 16),
            ("test", 20),
            ("test", 24),
            ("test", 32),
        ):
            with self.subTest(split=split, count=count):
                first = generate_users(
                    domain_id="shared",
                    count=count,
                    split=split,
                    manifest=manifest,
                    susceptibility_levels=susceptibility_levels,
                    seed=1729,
                )
                second = generate_users(
                    domain_id="shared",
                    count=count,
                    split=split,
                    manifest=manifest,
                    susceptibility_levels=susceptibility_levels,
                    seed=1729,
                )
                self.assertEqual(first, second)
                theta_counts = Counter(user.theta for user in first)
                theta_support = (
                    32 if split == "train" else 16
                )
                if count >= theta_support:
                    self.assertEqual(len(theta_counts), theta_support)
                for coordinate in range(3):
                    marginal = Counter(
                        user.theta[coordinate] for user in first
                    )
                    counts = tuple(
                        marginal[value] for value in theta_values
                    )
                    self.assertLessEqual(max(counts) - min(counts), 1)
                for coordinate in range(3):
                    marginal = Counter(
                        self._susceptibility_tuple(user)[coordinate]
                        for user in first
                    )
                    counts = tuple(
                        marginal[level]
                        for level in susceptibility_levels
                    )
                    self.assertLessEqual(max(counts) - min(counts), 1)
                for user in first:
                    self.assertEqual(
                        manifest.theta_groups[theta_group_id(user.theta)],
                        split,
                    )
                    self.assertEqual(
                        manifest.susceptibility_groups[
                            susceptibility_group_id(user.susceptibility)
                        ],
                        split,
                    )
        exactly_balanced = generate_users(
            domain_id="shared",
            count=16,
            split="test",
            manifest=manifest,
            susceptibility_levels=susceptibility_levels,
            seed=1729,
        )
        for coordinate in range(3):
            self.assertEqual(
                Counter(
                    user.theta[coordinate] for user in exactly_balanced
                ),
                Counter({value: 4 for value in theta_values}),
            )

    def test_joint_optimizer_reduces_cross_coordinate_contingency_imbalance(
        self,
    ) -> None:
        theta_values = (-2, -1, 1, 2)
        susceptibility_levels = (0.15, 0.45, 0.85)
        horizons = (4, 8, 10, 16, 20, 24, 32)

        def collision_count(
            theta_sequence: tuple[tuple[int, int, int], ...],
            susceptibility_sequence: tuple[Susceptibility, ...],
        ) -> int:
            total = 0
            for horizon in horizons:
                for theta_coordinate in range(3):
                    for susceptibility_name in (
                        "ranking",
                        "default",
                        "suggestion",
                    ):
                        cells = Counter(
                            (
                                theta_sequence[index][theta_coordinate],
                                getattr(
                                    susceptibility_sequence[index],
                                    susceptibility_name,
                                ),
                            )
                            for index in range(horizon)
                        )
                        total += sum(
                            count * (count - 1) // 2
                            for count in cells.values()
                        )
            return total

        for seed in (1729, 271828, 314159):
            with self.subTest(seed=seed):
                theta_support = tuple(
                    tuple(int(value) for value in group_id.split(","))
                    for group_id in orthogonal_theta_group_order(
                        theta_values,
                        seed=seed,
                        split="test",
                    )
                )
                susceptibility_support = tuple(
                    Susceptibility(
                        *(
                            float(value)
                            for value in group_id.split(",")
                        )
                    )
                    for group_id in orthogonal_susceptibility_group_order(
                        susceptibility_levels,
                        seed=seed,
                        split="test",
                    )
                )
                simple_theta = tuple(
                    theta_support[index % len(theta_support)]
                    for index in range(32)
                )
                simple_susceptibility = tuple(
                    susceptibility_support[
                        index % len(susceptibility_support)
                    ]
                    for index in range(32)
                )
                manifest = build_split_manifest(
                    seed=seed,
                    theta_policy=BALANCED_THETA_POLICY,
                    susceptibility_policy=(
                        BALANCED_SUSCEPTIBILITY_POLICY
                    ),
                )
                optimized_users = generate_users(
                    domain_id="shared",
                    count=32,
                    split="test",
                    manifest=manifest,
                    seed=seed,
                )
                optimized_theta = tuple(
                    user.theta for user in optimized_users
                )
                optimized_susceptibility = tuple(
                    user.susceptibility for user in optimized_users
                )
                simple_score = joint_cross_balance_score(
                    simple_theta,
                    simple_susceptibility,
                )
                optimized_score = joint_cross_balance_score(
                    optimized_theta,
                    optimized_susceptibility,
                )
                self.assertLess(optimized_score[0], simple_score[0])
                self.assertLess(optimized_score[1], simple_score[1])
                self.assertLess(optimized_score[2], simple_score[2])
                self.assertLess(
                    collision_count(
                        optimized_theta,
                        optimized_susceptibility,
                    ),
                    collision_count(
                        simple_theta,
                        simple_susceptibility,
                    ),
                )

    def test_joint_optimizer_preserves_single_v2_allocations(self) -> None:
        susceptibility_only = build_split_manifest(
            seed=1729,
            susceptibility_policy=BALANCED_SUSCEPTIBILITY_POLICY,
        )
        susceptibility_users = generate_users(
            domain_id="shared",
            count=8,
            split="test",
            manifest=susceptibility_only,
            seed=1729,
        )
        self.assertEqual(
            tuple(user.theta for user in susceptibility_users),
            (
                (-1, 2, -2),
                (1, -2, -2),
                (1, 2, -1),
                (-1, 2, -2),
                (2, 1, 1),
                (-1, -1, 1),
                (-2, 1, -2),
                (-2, -2, 2),
            ),
        )
        theta_only = build_split_manifest(
            seed=1729,
            theta_policy=BALANCED_THETA_POLICY,
        )
        theta_users = generate_users(
            domain_id="shared",
            count=8,
            split="test",
            manifest=theta_only,
            seed=1729,
        )
        self.assertEqual(
            tuple(
                self._susceptibility_tuple(user)
                for user in theta_users
            ),
            (
                (0.85, 0.45, 0.15),
                (0.85, 0.45, 0.15),
                (0.85, 0.45, 0.15),
                (0.15, 0.85, 0.45),
                (0.85, 0.45, 0.15),
                (0.85, 0.85, 0.45),
                (0.45, 0.85, 0.15),
                (0.45, 0.85, 0.15),
            ),
        )

    def test_population_policy_is_opt_in_and_validated(self) -> None:
        legacy = AppConfig()
        self.assertEqual(
            legacy.population.susceptibility_policy,
            LEGACY_SUSCEPTIBILITY_POLICY,
        )
        self.assertEqual(
            legacy.population.theta_policy,
            LEGACY_THETA_POLICY,
        )
        self.assertNotIn("population", legacy.to_dict())
        balanced = AppConfig.parse(
            {
                "schema_version": 1,
                "population": {
                    "susceptibility_policy": (
                        BALANCED_SUSCEPTIBILITY_POLICY
                    )
                },
            }
        )
        self.assertEqual(
            balanced.population,
            PopulationSection(BALANCED_SUSCEPTIBILITY_POLICY),
        )
        self.assertEqual(
            balanced.population.theta_policy,
            LEGACY_THETA_POLICY,
        )
        self.assertEqual(
            balanced.to_dict()["population"]["susceptibility_policy"],
            BALANCED_SUSCEPTIBILITY_POLICY,
        )
        with self.assertRaisesRegex(
            ConfigError,
            "population.susceptibility_policy",
        ):
            AppConfig.parse(
                {
                    "schema_version": 1,
                    "population": {"susceptibility_policy": "unknown"},
                }
            )
        fully_balanced = AppConfig.parse(
            {
                "schema_version": 1,
                "population": {
                    "susceptibility_policy": (
                        BALANCED_SUSCEPTIBILITY_POLICY
                    ),
                    "theta_policy": BALANCED_THETA_POLICY,
                },
            }
        )
        self.assertEqual(
            fully_balanced.population.theta_policy,
            BALANCED_THETA_POLICY,
        )
        with self.assertRaisesRegex(
            ConfigError,
            "population.theta_policy",
        ):
            AppConfig.parse(
                {
                    "schema_version": 1,
                    "population": {"theta_policy": "unknown"},
                }
            )

    def test_balanced_policy_rejects_non_three_level_designs(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly three unique"):
            build_split_manifest(
                seed=1729,
                susceptibility_levels=(0.2, 0.8),
                susceptibility_policy=BALANCED_SUSCEPTIBILITY_POLICY,
            )
        with self.assertRaisesRegex(ValueError, "exactly four unique"):
            build_split_manifest(
                seed=1729,
                theta_values=(-2, -1, 1),
                theta_policy=BALANCED_THETA_POLICY,
            )

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
            count=10,
            seed=11,
        )

        def mechanism(example: object) -> str:
            context = getattr(example, "context")
            target = context.target_attribute
            if context.default_option_id is not None:
                return "default"
            if context.suggested_option_id is not None:
                return "suggested"
            if ":ranking:" in context.context_id:
                return "ranking"
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
                    {
                        "balanced",
                        "restricted",
                        "ranking",
                        "default",
                        "suggested",
                    }
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
