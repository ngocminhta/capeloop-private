"""Randomized simulator training data and fitted likelihood bundles."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

from .calibration import CalibrationExample, fit_temperature
from .domains import (
    DATA_SPLITS,
    DomainSpec,
    dialogue_template_id,
    scenario_family_id,
)
from .elicitation import MECHANISMS, build_matched_anchor_set
from .fitting import (
    AdamConfig,
    AwareConditionalLogitModel,
    ChoiceTrainingExample,
    SemanticDirectionTrainingExample,
    UnawareSemanticDirectionModel,
    fit_aware_conditional_logit,
    fit_unaware_semantic_direction,
    semantic_choice_label,
)
from .response import RandomUtilityModel
from .schemas import LatentUser, Observation


@dataclass(frozen=True, slots=True)
class FittedModelBundle:
    aware: AwareConditionalLogitModel
    unaware: UnawareSemanticDirectionModel
    training_examples: int
    training_seed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "training_examples": self.training_examples,
            "training_seed": self.training_seed,
            "aware": self.aware.to_dict(),
            "unaware": self.unaware.to_dict(),
        }


def generate_training_examples(
    domain: DomainSpec,
    users: Sequence[LatentUser],
    response_model: RandomUtilityModel,
    *,
    count: int,
    seed: int,
    split: str = "train",
) -> tuple[ChoiceTrainingExample, ...]:
    """Generate a balanced randomized training log across provenance conditions."""

    if not users:
        raise ValueError("at least one training user is required")
    if count <= 0:
        raise ValueError("training count must be positive")
    if split not in DATA_SPLITS:
        raise ValueError(f"split must be one of {DATA_SPLITS}")
    examples = []
    for index in range(count):
        # Cycle users inside each target × mechanism × direction block. This
        # crosses every user with provenance conditions instead of assigning a
        # permanent mechanism through two matching modulo schedules.
        user = users[index % len(users)]
        cell = index // len(users)
        mechanism = MECHANISMS[cell % len(MECHANISMS)]
        target = (cell // len(MECHANISMS)) % 3
        direction = (
            -1
            if (cell // (len(MECHANISMS) * 3)) % 2 == 0
            else 1
        )
        matched = build_matched_anchor_set(
            domain,
            target_attribute=target,
            anchor_direction=direction,
            scenario_id=(
                f"{scenario_family_id(domain.domain_id, split)}:{index}"
            ),
            wording_template=dialogue_template_id(
                domain.domain_id,
                split,
            ),
            turn=index,
        )
        context = matched.context(mechanism)
        selected = response_model.sample_choice(
            user.theta,
            user.susceptibility,
            context,
            seed,
            noise_key=(split, domain.domain_id, user.user_id, index),
        )
        examples.append(
            ChoiceTrainingExample(
                theta=user.theta,
                context=context,
                observation=Observation(
                    selected_option_id=selected,
                    choice_noise_key=f"{split}:{index}",
                ),
            )
        )
    return tuple(examples)


def fit_model_bundle(
    examples: Sequence[ChoiceTrainingExample],
    *,
    seed: int,
    fit_steps: int = 600,
    learning_rate: float = 0.03,
    l2: float = 0.001,
) -> FittedModelBundle:
    training = tuple(examples)
    optimizer = AdamConfig(
        learning_rate=learning_rate,
        epochs=fit_steps,
        l2=l2,
    )
    aware = fit_aware_conditional_logit(training, optimizer)
    semantic = tuple(
        SemanticDirectionTrainingExample.from_choice(example)
        for example in training
    )
    unaware = fit_unaware_semantic_direction(semantic, optimizer)
    return FittedModelBundle(
        aware=aware,
        unaware=unaware,
        training_examples=len(training),
        training_seed=seed,
    )


def held_out_response_scores(
    bundle: FittedModelBundle,
    examples: Sequence[ChoiceTrainingExample],
) -> dict[str, float]:
    """Report declared response-space NLLs without treating them as comparable.

    The aware score is over displayed option identity and the unaware score over
    the canonical signed semantic response. They are diagnostics with different
    outcome supports; preference-posterior proper scores are used for Gate 1.
    """

    material = tuple(examples)
    if not material:
        raise ValueError("held-out examples cannot be empty")
    aware_nll = 0.0
    unaware_nll = 0.0
    for example in material:
        aware_nll -= math.log(
            max(
                bundle.aware.likelihood(
                    example.theta, example.context, example.observation
                ),
                1e-15,
            )
        )
        unaware_nll -= math.log(
            max(
                bundle.unaware.likelihood(
                    example.theta, example.context, example.observation
                ),
                1e-15,
            )
        )
    return {
        "aware_option_nll": aware_nll / len(material),
        "unaware_semantic_nll": unaware_nll / len(material),
        "scores_share_outcome_space": False,
    }


def held_out_aware_reliability(
    bundle: FittedModelBundle,
    examples: Sequence[ChoiceTrainingExample],
    *,
    bins: int = 10,
) -> dict[str, Any]:
    """Compute fixed-width option-level reliability on held-out interactions.

    Every displayed option contributes one Bernoulli forecast: the fitted
    action-aware choice probability and whether that option was selected. This
    makes the calibration support explicit and avoids treating the unrelated
    action-unaware semantic score as directly comparable.
    """

    material = tuple(examples)
    if not material:
        raise ValueError("held-out examples cannot be empty")
    if isinstance(bins, bool) or not isinstance(bins, int) or bins <= 0:
        raise ValueError("bins must be a positive integer")
    buckets: list[list[tuple[float, float]]] = [
        [] for _ in range(bins)
    ]
    for example in material:
        probabilities = bundle.aware.probabilities(
            example.theta,
            example.context,
        )
        for option, probability in zip(
            example.context.options,
            probabilities,
        ):
            index = min(int(float(probability) * bins), bins - 1)
            buckets[index].append(
                (
                    float(probability),
                    float(
                        option.option_id
                        == example.observation.selected_option_id
                    ),
                )
            )
    total = sum(len(bucket) for bucket in buckets)
    rows = []
    weighted_gap = 0.0
    for index, bucket in enumerate(buckets):
        lower = index / bins
        upper = (index + 1) / bins
        if bucket:
            mean_probability = math.fsum(
                probability for probability, _ in bucket
            ) / len(bucket)
            empirical_rate = math.fsum(
                outcome for _, outcome in bucket
            ) / len(bucket)
            absolute_gap = abs(mean_probability - empirical_rate)
            weighted_gap += len(bucket) * absolute_gap
        else:
            mean_probability = None
            empirical_rate = None
            absolute_gap = None
        rows.append(
            {
                "bin": index,
                "lower": lower,
                "upper": upper,
                "count": len(bucket),
                "mean_probability": mean_probability,
                "empirical_rate": empirical_rate,
                "absolute_gap": absolute_gap,
            }
        )
    return {
        "aware_option_ece": weighted_gap / total,
        "aware_option_forecasts": total,
        "aware_reliability_bins": rows,
    }


def temperature_calibrate_model_bundle(
    bundle: FittedModelBundle,
    examples: Sequence[ChoiceTrainingExample],
) -> tuple[FittedModelBundle, dict[str, Any]]:
    """Fit development-only temperatures and return calibrated likelihoods.

    Temperature scaling a conditional logit is equivalent to dividing all of
    its logits—and therefore its coefficient vector—by the fitted temperature.
    The aware and unaware outcome spaces receive separate temperatures.
    """

    material = tuple(examples)
    if not material:
        raise ValueError("calibration examples cannot be empty")
    aware_examples = []
    unaware_examples = []
    for example in material:
        aware_probabilities = bundle.aware.probabilities(
            example.theta,
            example.context,
        )
        aware_examples.append(
            CalibrationExample(
                aware_probabilities,
                example.context.option_ids.index(
                    example.observation.selected_option_id
                ),
                "development",
            )
        )
        if example.context.target_attribute is None:
            raise ValueError("calibration context lacks target_attribute")
        positive = bundle.unaware.positive_probability(
            example.theta,
            example.context.target_attribute,
        )
        label = semantic_choice_label(
            example.context,
            example.observation,
        )
        unaware_examples.append(
            CalibrationExample(
                (1.0 - positive, positive),
                1 if label > 0 else 0,
                "development",
            )
        )

    aware_calibration = fit_temperature(aware_examples)
    unaware_calibration = fit_temperature(unaware_examples)
    calibrated = FittedModelBundle(
        aware=AwareConditionalLogitModel(
            tuple(
                parameter / aware_calibration.temperature
                for parameter in bundle.aware.parameters
            )
        ),
        unaware=UnawareSemanticDirectionModel(
            tuple(
                parameter / unaware_calibration.temperature
                for parameter in bundle.unaware.parameters
            )
        ),
        training_examples=bundle.training_examples,
        training_seed=bundle.training_seed,
    )
    return calibrated, {
        "schema_version": 1,
        "kind": "temperature",
        "aware": aware_calibration.to_dict(),
        "unaware": unaware_calibration.to_dict(),
    }
