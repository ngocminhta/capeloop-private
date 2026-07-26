"""Declared user-response models and presentation-free welfare functions."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from .rng import gumbel, weighted_choice
from .schemas import (
    InteractionContext,
    Observation,
    Option,
    Susceptibility,
    Theta,
    validate_theta,
)


def intrinsic_utility(theta: Theta, option: Option) -> float:
    """Presentation-independent welfare utility ``theta · phi(option)``."""

    canonical_theta = validate_theta(theta)
    return math.fsum(
        coefficient * feature
        for coefficient, feature in zip(canonical_theta, option.features)
    )


def regret(
    theta: Theta,
    selected_option: Option,
    full_option_pool: tuple[Option, ...],
) -> float:
    """Intrinsic regret against the complete feasible pool."""

    if not full_option_pool:
        raise ValueError("full_option_pool cannot be empty")
    optimum = max(intrinsic_utility(theta, option) for option in full_option_pool)
    selected = intrinsic_utility(theta, selected_option)
    # Avoid a negative signed zero or tiny floating error.
    return max(0.0, optimum - selected)


def _softmax(logits: tuple[float, ...]) -> tuple[float, ...]:
    if not logits:
        raise ValueError("logits cannot be empty")
    maximum = max(logits)
    exponentials = tuple(math.exp(value - maximum) for value in logits)
    total = math.fsum(exponentials)
    return tuple(value / total for value in exponentials)


@dataclass(frozen=True, slots=True)
class RandomUtilityModel:
    """Finite multinomial-logit response model.

    Susceptibility values are user-specific logit bonuses.  The scale fields are
    global declared-model coefficients and default to one.
    """

    beta: float = 0.8
    ranking_scale: float = 1.0
    default_scale: float = 1.0
    suggestion_scale: float = 1.0

    def __post_init__(self) -> None:
        for field_name in (
            "beta",
            "ranking_scale",
            "default_scale",
            "suggestion_scale",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{field_name} must be numeric")
            numeric = float(value)
            if not math.isfinite(numeric) or numeric < 0.0:
                raise ValueError(f"{field_name} must be finite and non-negative")
            object.__setattr__(self, field_name, numeric)

    def presentation_utility(
        self,
        susceptibility: Susceptibility,
        context: InteractionContext,
        option_id: str,
    ) -> float:
        """Presentation bonus; it must never be used by welfare metrics."""

        # Position one is the declared rank treatment.  Other positions are the
        # reference category, keeping the coefficient directly interpretable.
        rank_bonus = (
            self.ranking_scale * susceptibility.ranking
            if context.rank(option_id) == 0
            else 0.0
        )
        default_bonus = (
            self.default_scale * susceptibility.default
            if context.default_option_id == option_id
            else 0.0
        )
        suggestion_bonus = (
            self.suggestion_scale * susceptibility.suggestion
            if context.suggested_option_id == option_id
            else 0.0
        )
        return rank_bonus + default_bonus + suggestion_bonus

    def logits(
        self,
        theta: Theta,
        susceptibility: Susceptibility,
        context: InteractionContext,
    ) -> tuple[float, ...]:
        validate_theta(theta)
        return tuple(
            self.beta * intrinsic_utility(theta, option)
            + self.presentation_utility(
                susceptibility, context, option.option_id
            )
            for option in context.options
        )

    def probabilities(
        self,
        theta: Theta,
        susceptibility: Susceptibility,
        context: InteractionContext,
    ) -> tuple[float, ...]:
        return _softmax(self.logits(theta, susceptibility, context))

    def probability_map(
        self,
        theta: Theta,
        susceptibility: Susceptibility,
        context: InteractionContext,
    ) -> dict[str, float]:
        probabilities = self.probabilities(theta, susceptibility, context)
        return {
            option.option_id: probability
            for option, probability in zip(context.options, probabilities)
        }

    def likelihood(
        self,
        theta: Theta,
        susceptibility: Susceptibility,
        context: InteractionContext,
        observation: Observation | str,
    ) -> float:
        selected = (
            observation.selected_option_id
            if isinstance(observation, Observation)
            else observation
        )
        try:
            index = context.option_ids.index(selected)
        except ValueError as exc:
            raise ValueError("selected option is not displayed") from exc
        return self.probabilities(theta, susceptibility, context)[index]

    def sample_choice(
        self,
        theta: Theta,
        susceptibility: Susceptibility,
        context: InteractionContext,
        master_seed: int,
        noise_key: Any | None = None,
    ) -> str:
        """Sample through option-keyed Gumbels for branch-consistent CRN."""

        semantic_key = context.context_id if noise_key is None else noise_key
        logits = self.logits(theta, susceptibility, context)
        perturbed = tuple(
            score
            + gumbel(
                master_seed,
                "random_utility",
                semantic_key,
                option.option_id,
            )
            for score, option in zip(logits, context.options)
        )
        winner = max(range(len(perturbed)), key=perturbed.__getitem__)
        return context.options[winner].option_id

    def sample(
        self,
        theta: Theta,
        susceptibility: Susceptibility,
        context: InteractionContext,
        master_seed: int,
        noise_key: Any | None = None,
    ) -> Observation:
        selected = self.sample_choice(
            theta, susceptibility, context, master_seed, noise_key
        )
        semantic_key = context.context_id if noise_key is None else noise_key
        return Observation(
            selected_option_id=selected,
            choice_noise_key=str(semantic_key),
        )


# Concise public synonym matching the paper's terminology.
ResponseModel = RandomUtilityModel


@dataclass(frozen=True, slots=True)
class RuleBasedResponseModel:
    """Noisy utility-maximizing robustness model without Gumbel assumptions."""

    decision_noise: float = 0.15
    beta: float = 0.8
    ranking_scale: float = 1.0
    default_scale: float = 1.0
    suggestion_scale: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.decision_noise <= 1.0:
            raise ValueError("decision_noise must be in [0, 1]")
        # Reuse the primary model's coefficient validation.
        RandomUtilityModel(
            beta=self.beta,
            ranking_scale=self.ranking_scale,
            default_scale=self.default_scale,
            suggestion_scale=self.suggestion_scale,
        )

    def _scores(
        self,
        theta: Theta,
        susceptibility: Susceptibility,
        context: InteractionContext,
    ) -> tuple[float, ...]:
        declared = RandomUtilityModel(
            beta=self.beta,
            ranking_scale=self.ranking_scale,
            default_scale=self.default_scale,
            suggestion_scale=self.suggestion_scale,
        )
        return declared.logits(theta, susceptibility, context)

    def probabilities(
        self,
        theta: Theta,
        susceptibility: Susceptibility,
        context: InteractionContext,
    ) -> tuple[float, ...]:
        scores = self._scores(theta, susceptibility, context)
        maximum = max(scores)
        winners = tuple(
            index
            for index, score in enumerate(scores)
            if math.isclose(score, maximum, rel_tol=0.0, abs_tol=1e-12)
        )
        random_mass = self.decision_noise / len(scores)
        maximizing_mass = (1.0 - self.decision_noise) / len(winners)
        return tuple(
            random_mass + (maximizing_mass if index in winners else 0.0)
            for index in range(len(scores))
        )

    def likelihood(
        self,
        theta: Theta,
        susceptibility: Susceptibility,
        context: InteractionContext,
        observation: Observation | str,
    ) -> float:
        selected = (
            observation.selected_option_id
            if isinstance(observation, Observation)
            else observation
        )
        try:
            index = context.option_ids.index(selected)
        except ValueError as exc:
            raise ValueError("selected option is not displayed") from exc
        return self.probabilities(theta, susceptibility, context)[index]

    def sample_choice(
        self,
        theta: Theta,
        susceptibility: Susceptibility,
        context: InteractionContext,
        master_seed: int,
        noise_key: Any | None = None,
    ) -> str:
        semantic_key = context.context_id if noise_key is None else noise_key
        return weighted_choice(
            context.option_ids,
            self.probabilities(theta, susceptibility, context),
            master_seed,
            "rule_based_response",
            semantic_key,
        )

    def sample(
        self,
        theta: Theta,
        susceptibility: Susceptibility,
        context: InteractionContext,
        master_seed: int,
        noise_key: Any | None = None,
    ) -> Observation:
        """Sample an observation through the same interface as the primary model."""

        selected = self.sample_choice(
            theta,
            susceptibility,
            context,
            master_seed,
            noise_key,
        )
        semantic_key = context.context_id if noise_key is None else noise_key
        return Observation(
            selected_option_id=selected,
            choice_noise_key=str(semantic_key),
        )
