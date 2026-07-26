"""Exact and fitted-likelihood Bayesian preference inference."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

from .beliefs import (
    JointThetaPsiBelief,
    PreferenceBelief,
    THETA_STATES,
)
from .response import RandomUtilityModel
from .schemas import InteractionContext, Observation, Susceptibility, Theta


class ThetaLikelihoodModel(Protocol):
    """Interface consumed by :class:`ThetaBayesUpdater`."""

    def likelihood(
        self,
        theta: Theta,
        context: InteractionContext,
        observation: Observation,
    ) -> float: ...


def _validated_likelihood(value: float) -> float:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("likelihoods must be finite and non-negative")
    return value


def exact_aware_update(
    prior: JointThetaPsiBelief,
    context: InteractionContext,
    observation: Observation,
    response_model: RandomUtilityModel,
) -> JointThetaPsiBelief:
    """One Bayes-optimal update under the declared response model."""

    if observation.selected_option_id not in context.option_ids:
        raise ValueError("observation must select a displayed option")
    weights: list[float] = []
    for theta_index, theta in enumerate(THETA_STATES):
        for psi_index, susceptibility in enumerate(prior.susceptibilities):
            flat_index = (
                theta_index * len(prior.susceptibilities) + psi_index
            )
            likelihood = _validated_likelihood(
                response_model.likelihood(
                    theta,
                    susceptibility,
                    context,
                    observation,
                )
            )
            weights.append(prior.probabilities[flat_index] * likelihood)
    return JointThetaPsiBelief.from_weights(prior.susceptibilities, weights)


def theta_bayes_update(
    prior: PreferenceBelief,
    likelihood_model: ThetaLikelihoodModel,
    context: InteractionContext,
    observation: Observation,
) -> PreferenceBelief:
    """Generic Bayes update over theta from any declared likelihood model."""

    if observation.selected_option_id not in context.option_ids:
        raise ValueError("observation must select a displayed option")
    weights = tuple(
        prior_probability
        * _validated_likelihood(
            likelihood_model.likelihood(theta, context, observation)
        )
        for theta, prior_probability in zip(THETA_STATES, prior.probabilities)
    )
    return PreferenceBelief.from_weights(weights)


# A verb-first alias that reads naturally at call sites.
update_theta_belief = theta_bayes_update


@dataclass(frozen=True, slots=True)
class KnownSusceptibilityLikelihood:
    """Adapter from the declared response model to a theta-only likelihood."""

    response_model: RandomUtilityModel
    susceptibility: Susceptibility

    def likelihood(
        self,
        theta: Theta,
        context: InteractionContext,
        observation: Observation,
    ) -> float:
        return self.response_model.likelihood(
            theta,
            self.susceptibility,
            context,
            observation,
        )


@dataclass(frozen=True, slots=True)
class MixtureSusceptibilityLikelihood:
    """Population likelihood with a fixed susceptibility mixture."""

    response_model: RandomUtilityModel
    susceptibilities: tuple[Susceptibility, ...]
    probabilities: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.susceptibilities:
            raise ValueError("susceptibilities cannot be empty")
        if len(self.susceptibilities) != len(self.probabilities):
            raise ValueError("one probability is required per susceptibility")
        if any(
            not math.isfinite(probability) or probability < 0.0
            for probability in self.probabilities
        ):
            raise ValueError("mixture probabilities must be finite and non-negative")
        total = math.fsum(self.probabilities)
        if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("mixture probabilities must sum to one")

    def likelihood(
        self,
        theta: Theta,
        context: InteractionContext,
        observation: Observation,
    ) -> float:
        return math.fsum(
            probability
            * self.response_model.likelihood(
                theta,
                susceptibility,
                context,
                observation,
            )
            for susceptibility, probability in zip(
                self.susceptibilities, self.probabilities
            )
        )


@dataclass(frozen=True, slots=True)
class ExactAwareUpdater:
    """Immutable sequential exact filter over theta and susceptibility."""

    response_model: RandomUtilityModel
    belief: JointThetaPsiBelief

    def updated(
        self,
        context: InteractionContext,
        observation: Observation,
    ) -> ExactAwareUpdater:
        posterior = exact_aware_update(
            self.belief,
            context,
            observation,
            self.response_model,
        )
        return ExactAwareUpdater(self.response_model, posterior)

    # Familiar synonym for callers that still treat the immutable result as state.
    update = updated


@dataclass(frozen=True, slots=True)
class ThetaBayesUpdater:
    """Immutable sequential theta filter for fitted or rule-based likelihoods."""

    likelihood_model: ThetaLikelihoodModel
    belief: PreferenceBelief

    def updated(
        self,
        context: InteractionContext,
        observation: Observation,
    ) -> ThetaBayesUpdater:
        posterior = theta_bayes_update(
            self.belief,
            self.likelihood_model,
            context,
            observation,
        )
        return ThetaBayesUpdater(self.likelihood_model, posterior)

    update = updated

