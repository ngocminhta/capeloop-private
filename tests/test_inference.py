from __future__ import annotations

import math
import unittest

from cape_loop.beliefs import (
    JointThetaPsiBelief,
    MarginalPreferenceBelief,
    PreferenceBelief,
    THETA_STATES,
)
from cape_loop.domains import TRAVEL
from cape_loop.fitting import (
    AdamConfig,
    AwareConditionalLogitModel,
    ChoiceTrainingExample,
    PARAMETER_COUNT,
    SemanticDirectionTrainingExample,
    UnawareSemanticDirectionModel,
    fit_aware_conditional_logit,
    fit_unaware_semantic_direction,
)
from cape_loop.inference import (
    KnownSusceptibilityLikelihood,
    exact_aware_update,
    theta_bayes_update,
)
from cape_loop.response import RandomUtilityModel
from cape_loop.schemas import (
    InteractionContext,
    Observation,
    Option,
    Susceptibility,
)


def make_context(
    context_id: str = "inference-context",
    *,
    positive_first: bool = True,
    default_direction: int = 0,
    suggestion_direction: int = 0,
) -> InteractionContext:
    negative, positive = TRAVEL.isolated_pair(0)
    by_direction = {-1: negative, 1: positive}
    ranking = (
        (positive.option_id, negative.option_id)
        if positive_first
        else (negative.option_id, positive.option_id)
    )
    return InteractionContext(
        context_id=context_id,
        options=(negative, positive),
        ranking=ranking,
        domain="travel",
        target_attribute=0,
        default_option_id=(
            None
            if default_direction == 0
            else by_direction[default_direction].option_id
        ),
        suggested_option_id=(
            None
            if suggestion_direction == 0
            else by_direction[suggestion_direction].option_id
        ),
    )


class BeliefTests(unittest.TestCase):
    def test_theta_state_space_and_marginals(self) -> None:
        self.assertEqual(len(THETA_STATES), 64)
        self.assertEqual(len(set(THETA_STATES)), 64)
        belief = PreferenceBelief.uniform()
        self.assertAlmostEqual(sum(belief.probabilities), 1.0)
        for attribute in range(3):
            self.assertEqual(belief.marginal(attribute), (0.25,) * 4)
            self.assertAlmostEqual(belief.sign_mass(attribute, 1), 0.5)
            self.assertAlmostEqual(belief.sign_mass(attribute, -1), 0.5)
        self.assertAlmostEqual(belief.entropy(), math.log(64.0))

    def test_marginal_joint_round_trip_and_joint_psi_marginals(self) -> None:
        marginals = MarginalPreferenceBelief.from_weights(
            (
                (1.0, 2.0, 3.0, 4.0),
                (4.0, 3.0, 2.0, 1.0),
                (1.0, 1.0, 1.0, 1.0),
            )
        )
        joint = marginals.independent_joint()
        for attribute in range(3):
            for actual, expected in zip(
                joint.marginal(attribute), marginals.marginal(attribute)
            ):
                self.assertAlmostEqual(actual, expected)

        psi = (Susceptibility(), Susceptibility(default=1.0))
        theta_psi = JointThetaPsiBelief.from_independent(
            joint,
            psi,
            (1.0, 3.0),
        )
        self.assertAlmostEqual(sum(theta_psi.probabilities), 1.0)
        self.assertEqual(
            tuple(round(value, 10) for value in theta_psi.susceptibility_marginal()),
            (0.25, 0.75),
        )
        for actual, expected in zip(
            theta_psi.theta_belief().probabilities, joint.probabilities
        ):
            self.assertAlmostEqual(actual, expected)


class ExactInferenceTests(unittest.TestCase):
    def test_exact_posterior_normalizes_and_matches_hand_enumeration(self) -> None:
        context = make_context(
            positive_first=True,
            default_direction=-1,
            suggestion_direction=1,
        )
        observation = Observation(context.option_ids[1])
        susceptibilities = (
            Susceptibility(ranking=0.0, default=0.0, suggestion=0.0),
            Susceptibility(ranking=0.4, default=0.8, suggestion=1.1),
        )
        prior = JointThetaPsiBelief.from_independent(
            PreferenceBelief.uniform(),
            susceptibilities,
            (0.25, 0.75),
        )
        model = RandomUtilityModel(
            beta=0.7,
            ranking_scale=0.9,
            default_scale=1.2,
            suggestion_scale=0.8,
        )
        posterior = exact_aware_update(prior, context, observation, model)
        self.assertAlmostEqual(sum(posterior.probabilities), 1.0)

        selected_index = context.option_ids.index(observation.selected_option_id)
        unnormalized: list[float] = []
        for theta_index, theta in enumerate(THETA_STATES):
            for psi_index, psi in enumerate(susceptibilities):
                raw_scores = []
                for option in context.options:
                    intrinsic = 0.7 * sum(
                        coefficient * feature
                        for coefficient, feature in zip(theta, option.features)
                    )
                    presentation = 0.0
                    if context.rank(option.option_id) == 0:
                        presentation += 0.9 * psi.ranking
                    if context.default_option_id == option.option_id:
                        presentation += 1.2 * psi.default
                    if context.suggested_option_id == option.option_id:
                        presentation += 0.8 * psi.suggestion
                    raw_scores.append(intrinsic + presentation)
                maximum = max(raw_scores)
                exp_scores = [math.exp(score - maximum) for score in raw_scores]
                likelihood = exp_scores[selected_index] / sum(exp_scores)
                flat_index = theta_index * len(susceptibilities) + psi_index
                unnormalized.append(prior.probabilities[flat_index] * likelihood)
        normalizer = sum(unnormalized)
        expected = [weight / normalizer for weight in unnormalized]
        for actual, hand_value in zip(posterior.probabilities, expected):
            self.assertAlmostEqual(actual, hand_value, places=13)

    def test_generic_theta_update_matches_single_psi_exact_marginal(self) -> None:
        context = make_context(
            positive_first=False,
            default_direction=1,
        )
        observation = Observation(context.option_ids[0])
        susceptibility = Susceptibility(ranking=0.3, default=0.8)
        response = RandomUtilityModel(beta=0.9)
        theta_prior = PreferenceBelief.uniform()
        joint_prior = JointThetaPsiBelief.from_independent(
            theta_prior,
            (susceptibility,),
        )
        exact = exact_aware_update(
            joint_prior,
            context,
            observation,
            response,
        ).theta_belief()
        generic = theta_bayes_update(
            theta_prior,
            KnownSusceptibilityLikelihood(response, susceptibility),
            context,
            observation,
        )
        for actual, expected in zip(generic.probabilities, exact.probabilities):
            self.assertAlmostEqual(actual, expected, places=14)


class FittedModelTests(unittest.TestCase):
    @staticmethod
    def _aware_identifying_fixture() -> (
        tuple[ChoiceTrainingExample, ...]
    ):
        truth = AwareConditionalLogitModel((0.65, 0.45, 0.75, -0.35))
        examples: list[ChoiceTrainingExample] = []
        counter = 0
        for theta_value in (-2, -1, 1, 2):
            theta = (theta_value, 1, -1)
            for positive_first in (False, True):
                for default_direction in (-1, 0, 1):
                    for suggestion_direction in (-1, 0, 1):
                        context = make_context(
                            f"fit-{counter}",
                            positive_first=positive_first,
                            default_direction=default_direction,
                            suggestion_direction=suggestion_direction,
                        )
                        counter += 1
                        probabilities = truth.probabilities(theta, context)
                        # Fractional-count equivalence via positive example weights
                        # makes this deterministic and close to the population MLE.
                        total = 200.0
                        for option, probability in zip(
                            context.options, probabilities
                        ):
                            examples.append(
                                ChoiceTrainingExample(
                                    theta,
                                    context,
                                    Observation(option.option_id),
                                    weight=max(probability * total, 1e-9),
                                )
                            )
        return tuple(examples)

    def test_aware_adam_recovers_identified_coefficients(self) -> None:
        fitted = fit_aware_conditional_logit(
            self._aware_identifying_fixture(),
            AdamConfig(learning_rate=0.04, epochs=900),
        )
        expected = (0.65, 0.45, 0.75, -0.35)
        self.assertEqual(len(fitted.parameters), PARAMETER_COUNT)
        for actual, target in zip(fitted.parameters, expected):
            self.assertAlmostEqual(actual, target, delta=0.035)

        round_trip = AwareConditionalLogitModel.from_json(fitted.to_json())
        self.assertEqual(round_trip, fitted)

    def test_unaware_model_has_matched_capacity_and_ignores_provenance(self) -> None:
        model = UnawareSemanticDirectionModel((0.8, -0.1, 0.2, -0.3))
        theta = (2, -1, 1)
        balanced = make_context("balanced")
        positive = TRAVEL.directional_option(0, 1)
        restricted_peer = Option(
            "travel_price_pos_peer",
            (0.5, 0.0, 0.0),
            "another premium option",
            "travel",
        )
        restricted = InteractionContext(
            "restricted",
            (positive, restricted_peer),
            (restricted_peer.option_id, positive.option_id),
            domain="travel",
            default_option_id=positive.option_id,
            suggested_option_id=positive.option_id,
            target_attribute=0,
        )
        balanced_observation = Observation(
            TRAVEL.directional_option(0, 1).option_id
        )
        restricted_observation = Observation(positive.option_id)
        self.assertEqual(len(model.parameters), PARAMETER_COUNT)
        self.assertAlmostEqual(
            model.likelihood(theta, balanced, balanced_observation),
            model.likelihood(theta, restricted, restricted_observation),
        )
        self.assertEqual(
            UnawareSemanticDirectionModel.from_json(model.to_json()),
            model,
        )

    def test_unaware_adam_recovers_semantic_response_fixture(self) -> None:
        truth = UnawareSemanticDirectionModel((0.55, -0.12, 0.18, -0.25))
        examples: list[SemanticDirectionTrainingExample] = []
        for theta in THETA_STATES:
            for target_attribute in range(3):
                positive = truth.positive_probability(theta, target_attribute)
                examples.append(
                    SemanticDirectionTrainingExample(
                        theta,
                        target_attribute,
                        1,
                        weight=max(positive * 100.0, 1e-9),
                    )
                )
                examples.append(
                    SemanticDirectionTrainingExample(
                        theta,
                        target_attribute,
                        -1,
                        weight=max((1.0 - positive) * 100.0, 1e-9),
                    )
                )
        fitted = fit_unaware_semantic_direction(
            examples,
            AdamConfig(learning_rate=0.04, epochs=700),
        )
        expected = truth.parameters
        for actual, target in zip(fitted.parameters, expected):
            self.assertAlmostEqual(actual, target, delta=0.025)


if __name__ == "__main__":
    unittest.main()

