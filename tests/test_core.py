from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
import math
import unittest

from cape_loop.domains import DOMAINS, TRAVEL, WRITING, get_domain
from cape_loop.rng import gumbel, semantic_seed, uniform, weighted_choice
from cape_loop.response import (
    RandomUtilityModel,
    RuleBasedResponseModel,
    intrinsic_utility,
    regret,
)
from cape_loop.schemas import (
    InteractionContext,
    InteractionRecord,
    LatentUser,
    Observation,
    Option,
    PolicyProvenance,
    ProfileUpdate,
    Susceptibility,
    TrajectoryRecord,
)


def balanced_context(
    *,
    context_id: str = "ctx",
    ranking_reversed: bool = False,
    default: str | None = None,
    suggestion: str | None = None,
) -> InteractionContext:
    negative, positive = TRAVEL.isolated_pair(0)
    ranking = (
        (positive.option_id, negative.option_id)
        if ranking_reversed
        else (negative.option_id, positive.option_id)
    )
    return InteractionContext(
        context_id=context_id,
        options=(negative, positive),
        ranking=ranking,
        domain="travel",
        scenario_id="price-pair",
        turn_id="turn-0",
        default_option_id=default,
        suggested_option_id=suggestion,
        target_attribute=0,
    )


class SchemaTests(unittest.TestCase):
    def test_latent_types_are_validated_and_immutable(self) -> None:
        user = LatentUser(
            "user-1",
            (-2, -1, 2),
            Susceptibility(ranking=0.3, default=0.5, suggestion=1.0),
        )
        with self.assertRaises(FrozenInstanceError):
            user.theta = (2, 2, 2)  # type: ignore[misc]
        with self.assertRaises(ValueError):
            LatentUser("bad", (0, 1, 2))
        with self.assertRaises(ValueError):
            Susceptibility(default=-0.1)
        with self.assertRaises(ValueError):
            Option("bad", (1.0, 2.0))

    def test_context_validates_references_and_keeps_provenance_separate(self) -> None:
        context = balanced_context()
        option_ids = context.option_ids
        with self.assertRaises(ValueError):
            InteractionContext(
                "bad-ranking",
                context.options,
                (option_ids[0], option_ids[0]),
            )
        with self.assertRaises(ValueError):
            InteractionContext(
                "bad-default",
                context.options,
                context.ranking,
                default_option_id="not-displayed",
            )

        context_fields = {field.name for field in fields(InteractionContext)}
        self.assertNotIn("policy_id", context_fields)
        self.assertNotIn("profile_snapshot", context_fields)

        provenance = PolicyProvenance(
            "balanced",
            "v1",
            (("prefers_budget", 0.2), ("prefers_premium", 0.8)),
            random_seed=7,
        )
        self.assertEqual(
            provenance.to_dict()["profile_snapshot"]["prefers_premium"],
            0.8,
        )
        self.assertNotIn("policy_id", context.to_dict())

    def test_auditable_record_chain_serializes_without_truth(self) -> None:
        context = balanced_context()
        observation = Observation(context.option_ids[0], "The first one.")
        provenance = PolicyProvenance("balanced", "v1")
        update = ProfileUpdate(
            updater_id="exact-aware",
            belief_before=(0.5, 0.5),
            belief_after=(0.6, 0.4),
            written_delta=("observed choice",),
        )
        event = InteractionRecord(
            "record-0",
            context,
            provenance,
            observation,
            update,
        )
        trajectory = TrajectoryRecord(
            "trajectory-0",
            "user-0",
            "travel",
            (event,),
        )
        payload = trajectory.to_dict()
        self.assertEqual(
            payload["interactions"][0]["policy_provenance"]["policy_id"],
            "balanced",
        )
        self.assertNotIn("theta", payload)

        with self.assertRaises(ValueError):
            InteractionRecord(
                "invalid",
                context,
                provenance,
                Observation("not-displayed"),
            )


class DomainTests(unittest.TestCase):
    def test_both_domains_have_stable_isolated_pairs(self) -> None:
        self.assertEqual(DOMAINS, (TRAVEL, WRITING))
        self.assertIs(get_domain("travel"), TRAVEL)
        for domain in DOMAINS:
            self.assertEqual(len(domain.attributes), 3)
            self.assertEqual(len(domain.option_pool), 8)
            self.assertEqual(len(domain.isolated_options), 6)
            self.assertEqual(len({o.option_id for o in domain.option_pool}), 8)
            for attribute in range(3):
                negative, positive = domain.isolated_pair(attribute)
                self.assertEqual(negative.features[attribute], -0.5)
                self.assertEqual(positive.features[attribute], 0.5)
                for nuisance in set(range(3)) - {attribute}:
                    self.assertEqual(negative.features[nuisance], 0.0)
                    self.assertEqual(positive.features[nuisance], 0.0)


class SemanticRngTests(unittest.TestCase):
    def test_rng_is_deterministic_keyed_and_order_stable_for_mappings(self) -> None:
        first = semantic_seed(11, "user", {"a": 1, "b": 2})
        second = semantic_seed(11, "user", {"b": 2, "a": 1})
        self.assertEqual(first, second)
        self.assertNotEqual(first, semantic_seed(11, "other"))

        value = uniform(11, "choice")
        self.assertGreater(value, 0.0)
        self.assertLess(value, 1.0)
        self.assertTrue(math.isfinite(gumbel(11, "choice")))
        self.assertEqual(
            weighted_choice(("a", "b"), (0.0, 1.0), 11, "weighted"),
            "b",
        )

    def test_option_keyed_gumbels_are_common_across_orderings(self) -> None:
        model = RandomUtilityModel()
        susceptibility = Susceptibility()
        standard = balanced_context(context_id="condition-a")
        reversed_order = InteractionContext(
            context_id="condition-b",
            options=tuple(reversed(standard.options)),
            ranking=tuple(reversed(standard.ranking)),
            domain=standard.domain,
            target_attribute=0,
        )
        theta = (1, -1, 2)
        standard_choice = model.sample_choice(
            theta, susceptibility, standard, 19, noise_key=("twin", 0)
        )
        reversed_choice = model.sample_choice(
            theta, susceptibility, reversed_order, 19, noise_key=("twin", 0)
        )
        self.assertEqual(standard_choice, reversed_choice)


class ResponseTests(unittest.TestCase):
    def test_multinomial_logit_is_normalized_and_presentation_sensitive(self) -> None:
        theta = (-1, 1, 2)
        susceptibility = Susceptibility(ranking=0.7, default=1.0, suggestion=0.8)
        base = balanced_context()
        target = base.option_ids[1]
        treated = balanced_context(
            context_id="treated",
            ranking_reversed=True,
            default=target,
            suggestion=target,
        )
        model = RandomUtilityModel()
        base_map = model.probability_map(theta, susceptibility, base)
        treated_map = model.probability_map(theta, susceptibility, treated)
        self.assertAlmostEqual(sum(base_map.values()), 1.0)
        self.assertAlmostEqual(sum(treated_map.values()), 1.0)
        self.assertGreater(treated_map[target], base_map[target])

    def test_presentation_never_enters_intrinsic_welfare(self) -> None:
        negative, positive = TRAVEL.isolated_pair(0)
        theta = (-2, 1, 1)
        expected_negative = intrinsic_utility(theta, negative)
        expected_positive = intrinsic_utility(theta, positive)
        self.assertGreater(expected_negative, expected_positive)

        strong_compliance = Susceptibility(
            ranking=20.0,
            default=20.0,
            suggestion=20.0,
        )
        context = balanced_context(
            ranking_reversed=True,
            default=positive.option_id,
            suggestion=positive.option_id,
        )
        model = RandomUtilityModel()
        self.assertGreater(
            model.probability_map(theta, strong_compliance, context)[
                positive.option_id
            ],
            0.999,
        )
        # Welfare is unchanged even when presentation nearly forces the worse item.
        self.assertEqual(intrinsic_utility(theta, negative), expected_negative)
        self.assertEqual(intrinsic_utility(theta, positive), expected_positive)
        self.assertAlmostEqual(
            regret(theta, positive, TRAVEL.option_pool),
            max(intrinsic_utility(theta, o) for o in TRAVEL.option_pool)
            - expected_positive,
        )

    def test_rule_based_robustness_model_is_normalized(self) -> None:
        model = RuleBasedResponseModel(decision_noise=0.2)
        probabilities = model.probabilities(
            (2, 1, -1),
            Susceptibility(default=0.5),
            balanced_context(),
        )
        self.assertAlmostEqual(sum(probabilities), 1.0)
        self.assertTrue(all(probability > 0.0 for probability in probabilities))


if __name__ == "__main__":
    unittest.main()

