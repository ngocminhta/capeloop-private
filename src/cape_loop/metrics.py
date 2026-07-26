"""Belief, causal-decomposition, welfare, and gate metrics."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

from .beliefs import (
    JointThetaPsiBelief,
    MarginalPreferenceBelief,
    PreferenceBelief,
    THETA_VALUES,
)
from .schemas import Theta, validate_theta


BeliefLike = PreferenceBelief | MarginalPreferenceBelief


def _marginals(belief: BeliefLike) -> MarginalPreferenceBelief:
    return belief if isinstance(belief, MarginalPreferenceBelief) else belief.marginals()


def marginal_brier(belief: BeliefLike, truth: Theta) -> float:
    """Mean four-class Brier score across the three preference attributes."""

    theta = validate_theta(truth)
    marginals = _marginals(belief)
    return math.fsum(
        math.fsum(
            (probability - (1.0 if value == theta[attribute] else 0.0)) ** 2
            for probability, value in zip(marginals.marginal(attribute), THETA_VALUES)
        )
        for attribute in range(3)
    ) / 3.0


def joint_nll(belief: PreferenceBelief, truth: Theta, *, epsilon: float = 1e-15) -> float:
    return -math.log(max(belief.probability(validate_theta(truth)), epsilon))


def marginal_kl(
    reference: BeliefLike,
    candidate: BeliefLike,
    *,
    epsilon: float = 1e-15,
) -> float:
    """Mean marginal KL; never mislabeled as a joint divergence."""

    first, second = _marginals(reference), _marginals(candidate)
    total = 0.0
    for attribute in range(3):
        for p, q in zip(first.marginal(attribute), second.marginal(attribute)):
            if p:
                total += p * math.log(p / max(q, epsilon))
    return total / 3.0


def marginal_l1(first: BeliefLike, second: BeliefLike) -> float:
    a, b = _marginals(first), _marginals(second)
    return math.fsum(
        abs(x - y)
        for attribute in range(3)
        for x, y in zip(a.marginal(attribute), b.marginal(attribute))
    )


def action_conditioned_update_error(
    system_before: BeliefLike,
    system_after: BeliefLike,
    aware_before: BeliefLike,
    aware_after: BeliefLike,
) -> float:
    """L1 distance between marginal probability increments (ACUE)."""

    sb, sa = _marginals(system_before), _marginals(system_after)
    ab, aa = _marginals(aware_before), _marginals(aware_after)
    return math.fsum(
        abs((sa.probabilities[d][v] - sb.probabilities[d][v]) - (aa.probabilities[d][v] - ab.probabilities[d][v]))
        for d in range(3)
        for v in range(4)
    )


def update_direction_accuracy(
    system_before: BeliefLike,
    system_after: BeliefLike,
    aware_before: BeliefLike,
    aware_after: BeliefLike,
    *,
    tolerance: float = 1e-9,
) -> float | None:
    """Sign agreement with aware marginal increments, excluding aware ties."""

    accuracy, _, _ = update_direction_accuracy_details(
        system_before,
        system_after,
        aware_before,
        aware_after,
        tolerance=tolerance,
    )
    return accuracy


def update_direction_accuracy_details(
    system_before: BeliefLike,
    system_after: BeliefLike,
    aware_before: BeliefLike,
    aware_after: BeliefLike,
    *,
    tolerance: float = 1e-9,
) -> tuple[float | None, int, int]:
    """Return accuracy plus evaluated/excluded component counts."""

    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    sb, sa = _marginals(system_before), _marginals(system_after)
    ab, aa = _marginals(aware_before), _marginals(aware_after)
    agreements = []
    excluded = 0
    for dimension in range(3):
        for value in range(4):
            system_delta = sa.probabilities[dimension][value] - sb.probabilities[dimension][value]
            aware_delta = aa.probabilities[dimension][value] - ab.probabilities[dimension][value]
            if abs(aware_delta) <= tolerance:
                excluded += 1
                continue
            same_direction = (
                system_delta > tolerance
                if aware_delta > tolerance
                else system_delta < -tolerance
            )
            agreements.append(1.0 if same_direction else 0.0)
    accuracy = (
        None
        if not agreements
        else math.fsum(agreements) / len(agreements)
    )
    return accuracy, len(agreements), excluded


def information_gain(
    before: PreferenceBelief | JointThetaPsiBelief,
    after: PreferenceBelief | JointThetaPsiBelief,
) -> float:
    """Declared action-aware information gain: ``H(before) - H(after)``."""

    if type(before) is not type(after):
        raise TypeError("information-gain states must use the same state space")
    return before.entropy() - after.entropy()


def entropy_change(before: BeliefLike, after: BeliefLike) -> float:
    return before.entropy() - after.entropy()


def clipped_logit(probability: float, *, epsilon: float = 1e-6) -> float:
    p = min(max(probability, epsilon), 1.0 - epsilon)
    return math.log(p / (1.0 - p))


def false_confidence_gain(
    before: BeliefLike,
    after: BeliefLike,
    *,
    attribute: int,
    wrong_direction: int,
) -> float:
    return clipped_logit(after.sign_mass(attribute, wrong_direction)) - clipped_logit(
        before.sign_mass(attribute, wrong_direction)
    )


def laundered_confidence_gain(
    system_before: BeliefLike,
    system_after: BeliefLike,
    shadow_before: BeliefLike,
    shadow_after: BeliefLike,
    *,
    attribute: int,
    wrong_direction: int,
) -> float:
    return false_confidence_gain(
        system_before,
        system_after,
        attribute=attribute,
        wrong_direction=wrong_direction,
    ) - false_confidence_gain(
        shadow_before,
        shadow_after,
        attribute=attribute,
        wrong_direction=wrong_direction,
    )


def selection_cost(profile_policy_shadow_error: float, balanced_shadow_error: float) -> float:
    return profile_policy_shadow_error - balanced_shadow_error


def attribution_cost(system_error: float, same_history_shadow_error: float) -> float:
    return system_error - same_history_shadow_error


def self_confirmation_interaction(
    profile_system_error: float,
    profile_aware_error: float,
    balanced_system_error: float,
    balanced_aware_error: float,
) -> float:
    return (profile_system_error - profile_aware_error) - (
        balanced_system_error - balanced_aware_error
    )


@dataclass(frozen=True, slots=True)
class SelfConfirmationEvidence:
    """The five proposal clauses required before labeling an episode."""

    remains_materially_wrong: bool
    wrong_mass_increased: bool
    cumulative_lcg: float
    profile_changed_later_action: bool
    shadow_gained_equivalent_confidence: bool
    lcg_threshold: float = 0.25
    shadow_equivalence_tolerance: float = 0.05

    @property
    def is_self_confirming(self) -> bool:
        return (
            self.remains_materially_wrong
            and self.wrong_mass_increased
            and self.cumulative_lcg > self.lcg_threshold
            and self.profile_changed_later_action
            and not self.shadow_gained_equivalent_confidence
        )

    def clauses(self) -> dict[str, bool]:
        return {
            "remains_materially_wrong": self.remains_materially_wrong,
            "wrong_mass_increased": self.wrong_mass_increased,
            "lcg_exceeds_threshold": self.cumulative_lcg > self.lcg_threshold,
            "profile_changed_later_action": self.profile_changed_later_action,
            "shadow_did_not_gain_equivalent_confidence": (
                not self.shadow_gained_equivalent_confidence
            ),
        }


def mean_or_nan(values: Iterable[float]) -> float:
    material = tuple(value for value in values if math.isfinite(value))
    return math.nan if not material else math.fsum(material) / len(material)
