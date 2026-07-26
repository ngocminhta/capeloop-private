"""Dependency-free fitted action-aware and action-unaware likelihood models."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Callable

from .schemas import (
    InteractionContext,
    NUM_ATTRIBUTES,
    Observation,
    Option,
    Theta,
    validate_theta,
)


PARAMETER_NAMES: tuple[str, ...] = (
    "intrinsic",
    "ranking",
    "default",
    "suggestion",
)
PARAMETER_COUNT = len(PARAMETER_NAMES)


def _validate_parameters(parameters: tuple[float, ...]) -> tuple[float, ...]:
    values = tuple(parameters)
    if len(values) != PARAMETER_COUNT:
        raise ValueError(f"models require exactly {PARAMETER_COUNT} parameters")
    result = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"parameters[{index}] must be numeric")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("parameters must be finite")
        result.append(numeric)
    return tuple(result)


def _dot(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return math.fsum(x * y for x, y in zip(left, right))


def _softmax(logits: tuple[float, ...]) -> tuple[float, ...]:
    maximum = max(logits)
    exponentials = tuple(math.exp(value - maximum) for value in logits)
    total = math.fsum(exponentials)
    return tuple(value / total for value in exponentials)


def aware_candidate_features(
    theta: Theta,
    context: InteractionContext,
    option: Option,
) -> tuple[float, float, float, float]:
    """Four conditional-logit features for a displayed candidate."""

    validate_theta(theta)
    intrinsic = math.fsum(
        coefficient * feature
        for coefficient, feature in zip(theta, option.features)
    )
    return (
        intrinsic,
        1.0 if context.rank(option.option_id) == 0 else 0.0,
        1.0 if context.default_option_id == option.option_id else 0.0,
        1.0 if context.suggested_option_id == option.option_id else 0.0,
    )


@dataclass(frozen=True, slots=True)
class ChoiceTrainingExample:
    """One labeled action-aware choice."""

    theta: Theta
    context: InteractionContext
    observation: Observation
    weight: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "theta", validate_theta(self.theta))
        if self.observation.selected_option_id not in self.context.option_ids:
            raise ValueError("training observation must select a displayed option")
        if not math.isfinite(self.weight) or self.weight <= 0.0:
            raise ValueError("training weight must be finite and positive")


@dataclass(frozen=True, slots=True)
class AwareConditionalLogitModel:
    """Learned ``P(Y | theta, C)`` with four auditable coefficients."""

    parameters: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "parameters",
            _validate_parameters(self.parameters),
        )

    def logits(
        self,
        theta: Theta,
        context: InteractionContext,
    ) -> tuple[float, ...]:
        return tuple(
            _dot(
                self.parameters,
                aware_candidate_features(theta, context, option),
            )
            for option in context.options
        )

    def probabilities(
        self,
        theta: Theta,
        context: InteractionContext,
    ) -> tuple[float, ...]:
        return _softmax(self.logits(theta, context))

    def likelihood(
        self,
        theta: Theta,
        context: InteractionContext,
        observation: Observation,
    ) -> float:
        try:
            selected_index = context.option_ids.index(
                observation.selected_option_id
            )
        except ValueError as exc:
            raise ValueError("observation must select a displayed option") from exc
        return self.probabilities(theta, context)[selected_index]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": "aware_conditional_logit",
            "parameter_names": list(PARAMETER_NAMES),
            "parameters": list(self.parameters),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> AwareConditionalLogitModel:
        if payload.get("model_type") != "aware_conditional_logit":
            raise ValueError("not an aware conditional-logit payload")
        return cls(tuple(payload["parameters"]))

    @classmethod
    def from_json(cls, payload: str) -> AwareConditionalLogitModel:
        decoded = json.loads(payload)
        if not isinstance(decoded, dict):
            raise ValueError("serialized model must decode to an object")
        return cls.from_dict(decoded)


def semantic_choice_label(
    context: InteractionContext,
    observation: Observation,
) -> int:
    """Map a selected item to the signed target-direction response ``{-1,+1}``."""

    if context.target_attribute is None:
        raise ValueError("semantic direction requires context.target_attribute")
    option = context.option(observation.selected_option_id)
    value = option.features[context.target_attribute]
    if value == 0.0:
        raise ValueError("selected option is neutral on the target attribute")
    return 1 if value > 0.0 else -1


def _oriented_theta_features(
    theta: Theta,
    target_attribute: int,
) -> tuple[float, float, float, float]:
    validate_theta(theta)
    if not 0 <= target_attribute < NUM_ATTRIBUTES:
        raise ValueError(f"target_attribute must be in [0, {NUM_ATTRIBUTES})")
    return (
        float(theta[target_attribute]),
        float(theta[(target_attribute + 1) % NUM_ATTRIBUTES]),
        float(theta[(target_attribute + 2) % NUM_ATTRIBUTES]),
        1.0,
    )


@dataclass(frozen=True, slots=True)
class SemanticDirectionTrainingExample:
    """A context-marginalized response label for the action-unaware model."""

    theta: Theta
    target_attribute: int
    label: int
    weight: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "theta", validate_theta(self.theta))
        if (
            isinstance(self.target_attribute, bool)
            or not isinstance(self.target_attribute, int)
            or not 0 <= self.target_attribute < NUM_ATTRIBUTES
        ):
            raise ValueError(
                f"target_attribute must be in [0, {NUM_ATTRIBUTES})"
            )
        if self.label not in (-1, 1):
            raise ValueError("semantic label must be -1 or +1")
        if not math.isfinite(self.weight) or self.weight <= 0.0:
            raise ValueError("training weight must be finite and positive")

    @classmethod
    def from_choice(
        cls,
        example: ChoiceTrainingExample,
    ) -> SemanticDirectionTrainingExample:
        if example.context.target_attribute is None:
            raise ValueError("choice context has no target_attribute")
        return cls(
            theta=example.theta,
            target_attribute=example.context.target_attribute,
            label=semantic_choice_label(example.context, example.observation),
            weight=example.weight,
        )


@dataclass(frozen=True, slots=True)
class UnawareSemanticDirectionModel:
    """Learned ``P(Z | theta)`` over canonical signed choice labels.

    The context is used solely to decode the selected option into ``Z`` and is
    never part of the evidence model.  Four parameters match the aware model's
    capacity.
    """

    parameters: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "parameters",
            _validate_parameters(self.parameters),
        )

    def positive_probability(self, theta: Theta, target_attribute: int) -> float:
        logit = _dot(
            self.parameters,
            _oriented_theta_features(theta, target_attribute),
        )
        if logit >= 0.0:
            return 1.0 / (1.0 + math.exp(-logit))
        exponential = math.exp(logit)
        return exponential / (1.0 + exponential)

    def label_probability(
        self,
        theta: Theta,
        target_attribute: int,
        label: int,
    ) -> float:
        if label not in (-1, 1):
            raise ValueError("label must be -1 or +1")
        positive = self.positive_probability(theta, target_attribute)
        return positive if label > 0 else 1.0 - positive

    def likelihood(
        self,
        theta: Theta,
        context: InteractionContext,
        observation: Observation,
    ) -> float:
        if context.target_attribute is None:
            raise ValueError("unaware likelihood requires target_attribute")
        return self.label_probability(
            theta,
            context.target_attribute,
            semantic_choice_label(context, observation),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": "unaware_semantic_direction",
            "parameter_names": [
                "target_theta",
                "next_theta",
                "remaining_theta",
                "intercept",
            ],
            "parameters": list(self.parameters),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
    ) -> UnawareSemanticDirectionModel:
        if payload.get("model_type") != "unaware_semantic_direction":
            raise ValueError("not an unaware semantic-direction payload")
        return cls(tuple(payload["parameters"]))

    @classmethod
    def from_json(cls, payload: str) -> UnawareSemanticDirectionModel:
        decoded = json.loads(payload)
        if not isinstance(decoded, dict):
            raise ValueError("serialized model must decode to an object")
        return cls.from_dict(decoded)


@dataclass(frozen=True, slots=True)
class AdamConfig:
    learning_rate: float = 0.03
    epochs: int = 600
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    l2: float = 0.0
    gradient_clip: float | None = 10.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if isinstance(self.epochs, bool) or not isinstance(self.epochs, int):
            raise TypeError("epochs must be an integer")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if not 0.0 <= self.beta1 < 1.0 or not 0.0 <= self.beta2 < 1.0:
            raise ValueError("Adam beta values must be in [0, 1)")
        if not math.isfinite(self.epsilon) or self.epsilon <= 0.0:
            raise ValueError("epsilon must be finite and positive")
        if not math.isfinite(self.l2) or self.l2 < 0.0:
            raise ValueError("l2 must be finite and non-negative")
        if self.gradient_clip is not None and (
            not math.isfinite(self.gradient_clip) or self.gradient_clip <= 0.0
        ):
            raise ValueError("gradient_clip must be finite and positive or None")


GradientFunction = Callable[[tuple[float, ...]], tuple[float, tuple[float, ...]]]


def _adam(
    gradient_function: GradientFunction,
    config: AdamConfig,
    initial_parameters: tuple[float, ...] | None,
) -> tuple[float, float, float, float]:
    parameters = list(
        _validate_parameters(
            (0.0, 0.0, 0.0, 0.0)
            if initial_parameters is None
            else initial_parameters
        )
    )
    first_moment = [0.0] * PARAMETER_COUNT
    second_moment = [0.0] * PARAMETER_COUNT

    for step in range(1, config.epochs + 1):
        _, gradient = gradient_function(tuple(parameters))
        if len(gradient) != PARAMETER_COUNT:
            raise ValueError("gradient function returned the wrong shape")
        if config.gradient_clip is not None:
            norm = math.sqrt(math.fsum(value * value for value in gradient))
            if norm > config.gradient_clip:
                scale = config.gradient_clip / norm
                gradient = tuple(value * scale for value in gradient)

        for index, derivative in enumerate(gradient):
            if not math.isfinite(derivative):
                raise ValueError("non-finite training gradient")
            first_moment[index] = (
                config.beta1 * first_moment[index]
                + (1.0 - config.beta1) * derivative
            )
            second_moment[index] = (
                config.beta2 * second_moment[index]
                + (1.0 - config.beta2) * derivative * derivative
            )
            corrected_first = first_moment[index] / (1.0 - config.beta1**step)
            corrected_second = second_moment[index] / (1.0 - config.beta2**step)
            parameters[index] -= (
                config.learning_rate
                * corrected_first
                / (math.sqrt(corrected_second) + config.epsilon)
            )
    return (
        parameters[0],
        parameters[1],
        parameters[2],
        parameters[3],
    )


def fit_aware_conditional_logit(
    examples: tuple[ChoiceTrainingExample, ...] | list[ChoiceTrainingExample],
    config: AdamConfig = AdamConfig(),
    initial_parameters: tuple[float, ...] | None = None,
) -> AwareConditionalLogitModel:
    """Fit the four-parameter action-aware model by full-batch Adam."""

    training = tuple(examples)
    if not training:
        raise ValueError("at least one training example is required")
    if not all(isinstance(example, ChoiceTrainingExample) for example in training):
        raise TypeError("examples must contain ChoiceTrainingExample objects")
    total_weight = math.fsum(example.weight for example in training)

    def objective(parameters: tuple[float, ...]) -> tuple[float, tuple[float, ...]]:
        gradient = [0.0] * PARAMETER_COUNT
        loss = 0.0
        for example in training:
            candidate_features = tuple(
                aware_candidate_features(example.theta, example.context, option)
                for option in example.context.options
            )
            probabilities = _softmax(
                tuple(
                    _dot(parameters, features)
                    for features in candidate_features
                )
            )
            selected_index = example.context.option_ids.index(
                example.observation.selected_option_id
            )
            loss -= example.weight * math.log(
                max(probabilities[selected_index], 1e-300)
            )
            expected = tuple(
                math.fsum(
                    probability * features[parameter_index]
                    for probability, features in zip(
                        probabilities, candidate_features
                    )
                )
                for parameter_index in range(PARAMETER_COUNT)
            )
            selected_features = candidate_features[selected_index]
            for parameter_index in range(PARAMETER_COUNT):
                gradient[parameter_index] += example.weight * (
                    expected[parameter_index]
                    - selected_features[parameter_index]
                )

        loss /= total_weight
        for parameter_index in range(PARAMETER_COUNT):
            gradient[parameter_index] /= total_weight
            loss += 0.5 * config.l2 * parameters[parameter_index] ** 2
            gradient[parameter_index] += (
                config.l2 * parameters[parameter_index]
            )
        return loss, tuple(gradient)

    learned = _adam(objective, config, initial_parameters)
    return AwareConditionalLogitModel(learned)


def fit_unaware_semantic_direction(
    examples: (
        tuple[SemanticDirectionTrainingExample, ...]
        | list[SemanticDirectionTrainingExample]
    ),
    config: AdamConfig = AdamConfig(),
    initial_parameters: tuple[float, ...] | None = None,
) -> UnawareSemanticDirectionModel:
    """Fit context-free ``P(Z | theta)`` with the same four-parameter capacity."""

    training = tuple(examples)
    if not training:
        raise ValueError("at least one training example is required")
    if not all(
        isinstance(example, SemanticDirectionTrainingExample)
        for example in training
    ):
        raise TypeError(
            "examples must contain SemanticDirectionTrainingExample objects"
        )
    total_weight = math.fsum(example.weight for example in training)

    def objective(parameters: tuple[float, ...]) -> tuple[float, tuple[float, ...]]:
        gradient = [0.0] * PARAMETER_COUNT
        loss = 0.0
        for example in training:
            features = _oriented_theta_features(
                example.theta, example.target_attribute
            )
            logit = _dot(parameters, features)
            if logit >= 0.0:
                positive = 1.0 / (1.0 + math.exp(-logit))
            else:
                exponential = math.exp(logit)
                positive = exponential / (1.0 + exponential)
            target = 1.0 if example.label > 0 else 0.0
            probability = positive if target else 1.0 - positive
            loss -= example.weight * math.log(max(probability, 1e-300))
            residual = positive - target
            for parameter_index, feature in enumerate(features):
                gradient[parameter_index] += (
                    example.weight * residual * feature
                )

        loss /= total_weight
        for parameter_index in range(PARAMETER_COUNT):
            gradient[parameter_index] /= total_weight
            loss += 0.5 * config.l2 * parameters[parameter_index] ** 2
            gradient[parameter_index] += (
                config.l2 * parameters[parameter_index]
            )
        return loss, tuple(gradient)

    learned = _adam(objective, config, initial_parameters)
    return UnawareSemanticDirectionModel(learned)


# Compact aliases for experiment configuration code.
FittedAwareModel = AwareConditionalLogitModel
FittedUnawareModel = UnawareSemanticDirectionModel
fit_aware = fit_aware_conditional_logit
fit_unaware = fit_unaware_semantic_direction


def model_from_dict(
    payload: dict[str, Any],
) -> AwareConditionalLogitModel | UnawareSemanticDirectionModel:
    model_type = payload.get("model_type")
    if model_type == "aware_conditional_logit":
        return AwareConditionalLogitModel.from_dict(payload)
    if model_type == "unaware_semantic_direction":
        return UnawareSemanticDirectionModel.from_dict(payload)
    raise ValueError(f"unknown fitted model type: {model_type!r}")

