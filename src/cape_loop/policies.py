"""Interaction policies that never receive latent user truth."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Protocol

from .beliefs import PreferenceBelief
from .domains import DomainSpec, dialogue_template_id, scenario_family_id
from .elicitation import build_matched_anchor_set
from .rng import semantic_seed, uniform
from .schemas import InteractionContext, PolicyProvenance


@dataclass(frozen=True, slots=True)
class PolicyAction:
    context: InteractionContext
    provenance: PolicyProvenance

    def signature(self) -> tuple[object, ...]:
        """Fields through which a profile can change what the user observes."""

        return (
            tuple(option.option_id for option in self.context.options),
            self.context.ranking,
            self.context.default_option_id,
            self.context.suggested_option_id,
            self.context.wording_template,
            self.context.prompt,
        )


class InteractionPolicy(Protocol):
    policy_id: str
    policy_version: str

    def action(
        self,
        domain: DomainSpec,
        belief: PreferenceBelief,
        *,
        turn: int,
        master_seed: int,
        trajectory_id: str,
        target_counts: tuple[int, int, int] | None = None,
    ) -> PolicyAction: ...


def _snapshot(belief: PreferenceBelief) -> tuple[tuple[str, float], ...]:
    return tuple(
        (f"attribute_{index + 1}", value)
        for index, value in enumerate(belief.expected_theta())
    )


def _rank_pair(
    negative_id: str,
    positive_id: str,
    *,
    master_seed: int,
    key: tuple[object, ...],
) -> tuple[str, str]:
    if uniform(master_seed, "policy-rank", key) < 0.5:
        return (negative_id, positive_id)
    return (positive_id, negative_id)


def _prospective_turn(
    plan: object | None,
    trajectory_id: str,
    turn: int,
) -> object | None:
    """Read one outcome-blind instruction without coupling policy modules."""

    if plan is None:
        return None
    lookup = getattr(plan, "turn", None)
    if not callable(lookup):
        raise TypeError("prospective manipulation plan must expose turn(key, index)")
    instruction = lookup(trajectory_id, turn)
    target = getattr(instruction, "target_attribute", None)
    if isinstance(target, bool) or not isinstance(target, int) or target not in range(3):
        raise ValueError("prospective turn target_attribute must be 0, 1, or 2")
    return instruction


def _prospective_schedule_key(
    plan: object | None,
    trajectory_id: str,
) -> str:
    """Use one exogenous-randomization key across profile conditions."""

    if plan is None:
        return trajectory_id
    lookup = getattr(plan, "schedule_key", None)
    if not callable(lookup):
        return trajectory_id
    key = lookup(trajectory_id)
    if not isinstance(key, str) or not key:
        raise ValueError("prospective schedule key must be a non-empty string")
    return key


@dataclass(frozen=True, slots=True)
class BalancedPolicy:
    policy_id: str = "balanced"
    policy_version: str = "v1"
    prospective_plan: object | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def action(
        self,
        domain: DomainSpec,
        belief: PreferenceBelief,
        *,
        turn: int,
        master_seed: int,
        trajectory_id: str,
        target_counts: tuple[int, int, int] | None = None,
    ) -> PolicyAction:
        planned = _prospective_turn(
            self.prospective_plan,
            trajectory_id,
            turn,
        )
        schedule_key = _prospective_schedule_key(
            self.prospective_plan,
            trajectory_id,
        )
        target = (
            turn % 3
            if planned is None
            else int(getattr(planned, "target_attribute"))
        )
        negative, positive = domain.isolated_pair(target)
        ranking = _rank_pair(
            negative.option_id,
            positive.option_id,
            master_seed=master_seed,
            key=(schedule_key, turn, target, "shared-neutral-ranking"),
        )
        context = InteractionContext(
            context_id=f"{trajectory_id}:{turn}:{self.policy_id}",
            options=(negative, positive),
            ranking=ranking,
            domain=domain.domain_id,
            scenario_id=(
                f"{scenario_family_id(domain.domain_id, 'test')}:"
                f"dimension:{target}"
            ),
            turn_id=str(turn),
            wording_template=dialogue_template_id(
                domain.domain_id,
                "test",
            ),
            question_type="choice",
            target_attribute=target,
        )
        provenance = PolicyProvenance(
            policy_id=self.policy_id,
            policy_version=(
                self.policy_version
                if planned is None
                else "v3-condition-matched-schedule"
            ),
            profile_snapshot=_snapshot(belief),
            random_seed=semantic_seed(
                master_seed, schedule_key, turn, self.policy_id
            ),
            presentation_mechanism="balanced",
            profile_conditioned=False,
        )
        return PolicyAction(context, provenance)


@dataclass(frozen=True, slots=True)
class SoftProfileConditionedPolicy:
    policy_id: str = "soft_profile_conditioned"
    policy_version: str = "v2-neutral-profile-tie"
    conditioning_strength: float | None = None
    prospective_plan: object | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        strength = self.conditioning_strength
        if strength is None:
            return
        if (
            isinstance(strength, bool)
            or not isinstance(strength, (int, float))
            or not math.isfinite(float(strength))
            or not 0.0 <= float(strength) <= 1.0
        ):
            raise ValueError("conditioning_strength must lie in [0, 1]")

    def action(
        self,
        domain: DomainSpec,
        belief: PreferenceBelief,
        *,
        turn: int,
        master_seed: int,
        trajectory_id: str,
        target_counts: tuple[int, int, int] | None = None,
    ) -> PolicyAction:
        planned = _prospective_turn(
            self.prospective_plan,
            trajectory_id,
            turn,
        )
        schedule_key = _prospective_schedule_key(
            self.prospective_plan,
            trajectory_id,
        )
        target = (
            turn % 3
            if planned is None
            else int(getattr(planned, "target_attribute"))
        )
        negative, positive = domain.isolated_pair(target)
        expected = belief.expected_theta()[target]
        current_direction = (
            -1
            if expected < -1e-12
            else 1
            if expected > 1e-12
            else 0
        )
        # A three-block Latin-square rotation crosses every attribute with every
        # presentation channel over nine turns instead of confounding channel
        # identity with ``target = turn % 3``.
        planned_role = None if planned is None else getattr(planned, "role", None)
        forced_active = planned_role in {
            "informative_active",
            "decisive_active_control",
        }
        direction = current_direction
        if forced_active and direction == 0:
            frozen_direction = getattr(planned, "planned_profile_direction", None)
            if frozen_direction not in {-1, 1}:
                raise ValueError(
                    "a prospectively active neutral-profile turn requires a "
                    "frozen planned_profile_direction"
                )
            direction = int(frozen_direction)
        preferred = negative if direction < 0 else positive
        other = positive if preferred is negative else negative
        mechanism = (
            str(getattr(planned, "mechanism"))
            if forced_active
            else ("ranking", "default", "suggestion")[
                ((turn % 3) + (turn // 3)) % 3
            ]
        )
        if forced_active and mechanism not in {"default", "suggestion"}:
            raise ValueError(
                "prospectively active soft turns require default or suggestion"
            )
        if forced_active:
            application_probability = 1.0
        elif direction == 0:
            # An exactly directionless profile supplies no profile-consistent
            # option to promote. Keep the visible action neutral rather than
            # manufacturing a negative-direction treatment.
            application_probability = 0.0
        elif self.conditioning_strength is None:
            # Preserve the ordinary Experiment B/C protocol. Those runs use
            # profile confidence to make conditioning soft and adaptive.
            confidence = belief.sign_mass(target, direction)
            application_probability = 0.15 + 0.80 * max(
                0.0, min(1.0, (confidence - 0.5) / 0.5)
            )
        else:
            # Sensitivity assigns an exogenous multiplier to the legacy
            # adaptive propensity. Lambda=0 is a neutral-action control and
            # lambda=1 exactly preserves the ordinary soft policy. Retaining
            # the confidence term lets accumulated profile changes alter
            # later actions, which the strict self-confirmation definition
            # requires.
            confidence = belief.sign_mass(target, direction)
            legacy_probability = 0.15 + 0.80 * max(
                0.0, min(1.0, (confidence - 0.5) / 0.5)
            )
            application_probability = (
                float(self.conditioning_strength)
                * legacy_probability
            )
        apply_profile = forced_active
        if not forced_active:
            apply_profile = uniform(
                master_seed,
                "soft-profile-application",
                schedule_key,
                turn,
                target,
                mechanism,
            ) < application_probability
        if mechanism == "ranking" and apply_profile:
            ranking = (preferred.option_id, other.option_id)
            default_id = None
            suggestion_id = None
        else:
            ranking = _rank_pair(
                negative.option_id,
                positive.option_id,
                master_seed=master_seed,
                key=(
                    schedule_key,
                    turn,
                    target,
                    "shared-neutral-ranking",
                ),
            )
            default_id = (
                preferred.option_id
                if mechanism == "default" and apply_profile
                else None
            )
            suggestion_id = (
                preferred.option_id
                if mechanism == "suggestion" and apply_profile
                else None
            )
        context = InteractionContext(
            context_id=f"{trajectory_id}:{turn}:{self.policy_id}:{mechanism}",
            options=(negative, positive),
            ranking=ranking,
            domain=domain.domain_id,
            scenario_id=(
                f"{scenario_family_id(domain.domain_id, 'test')}:"
                f"dimension:{target}"
            ),
            turn_id=str(turn),
            default_option_id=default_id,
            suggested_option_id=suggestion_id,
            # This is a surface-template identifier, not an audit label. The
            # visible ranking/default/suggestion fields carry the treatment.
            wording_template=dialogue_template_id(
                domain.domain_id,
                "test",
            ),
            question_type="choice",
            target_attribute=target,
        )
        return PolicyAction(
            context,
            PolicyProvenance(
                policy_id=self.policy_id,
                policy_version=(
                    "v5-condition-matched-active-turns"
                    if planned is not None
                    else self.policy_version
                    if (
                        self.conditioning_strength is None
                        or float(self.conditioning_strength) == 1.0
                    )
                    else "v3-conditioning-strength"
                ),
                profile_snapshot=_snapshot(belief),
                random_seed=semantic_seed(
                    master_seed, schedule_key, turn, self.policy_id
                ),
                presentation_mechanism=(
                    mechanism if apply_profile else "balanced"
                ),
                profile_conditioned=apply_profile,
            ),
        )


@dataclass(frozen=True, slots=True)
class ExploratoryPolicy:
    policy_id: str = "exploratory"
    policy_version: str = "v3-balanced-coverage-shared-neutral-ranking"

    def action(
        self,
        domain: DomainSpec,
        belief: PreferenceBelief,
        *,
        turn: int,
        master_seed: int,
        trajectory_id: str,
        target_counts: tuple[int, int, int] | None = None,
    ) -> PolicyAction:
        if target_counts is None:
            completed, remainder = divmod(turn, 3)
            target_counts = tuple(
                completed + int(index < remainder) for index in range(3)
            )
        if (
            len(target_counts) != 3
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in target_counts
            )
            or sum(target_counts) != turn
        ):
            raise ValueError(
                "exploratory target_counts must be three non-negative "
                "integers summing to turn"
            )
        marginal_entropies = tuple(
            -sum(p * __import__("math").log(p) for p in belief.marginal(index) if p > 0)
            for index in range(3)
        )
        minimum_exposure = min(target_counts)
        eligible = tuple(
            index for index, count in enumerate(target_counts)
            if count == minimum_exposure
        )
        target = max(
            eligible,
            key=lambda index: (marginal_entropies[index], -index),
        )
        negative, positive = domain.isolated_pair(target)
        ranking = _rank_pair(
            negative.option_id,
            positive.option_id,
            master_seed=master_seed,
            key=(trajectory_id, turn, target, "shared-neutral-ranking"),
        )
        context = InteractionContext(
            context_id=f"{trajectory_id}:{turn}:{self.policy_id}",
            options=(negative, positive),
            ranking=ranking,
            domain=domain.domain_id,
            scenario_id=(
                f"{scenario_family_id(domain.domain_id, 'test')}:"
                f"explore:{target}"
            ),
            turn_id=str(turn),
            wording_template=dialogue_template_id(
                domain.domain_id,
                "test",
            ),
            question_type="choice",
            target_attribute=target,
        )
        return PolicyAction(
            context,
            PolicyProvenance(
                policy_id=self.policy_id,
                policy_version=self.policy_version,
                profile_snapshot=_snapshot(belief),
                random_seed=semantic_seed(
                    master_seed, trajectory_id, turn, self.policy_id
                ),
                presentation_mechanism="target_selection",
                profile_conditioned=True,
            ),
        )


@dataclass(frozen=True, slots=True)
class FixedBiasPolicy:
    """Mildly biased logger whose reference profile is updater-independent."""

    bias_direction: int = -1
    policy_id: str = "fixed_bias"
    policy_version: str = "v1"

    def __post_init__(self) -> None:
        if self.bias_direction not in (-1, 1):
            raise ValueError("bias_direction must be -1 or +1")

    def action(
        self,
        domain: DomainSpec,
        belief: PreferenceBelief,
        *,
        turn: int,
        master_seed: int,
        trajectory_id: str,
        target_counts: tuple[int, int, int] | None = None,
    ) -> PolicyAction:
        target = turn % 3
        negative, positive = domain.isolated_pair(target)
        biased = negative if self.bias_direction < 0 else positive
        other = positive if self.bias_direction < 0 else negative
        context = InteractionContext(
            context_id=f"{trajectory_id}:{turn}:{self.policy_id}",
            options=(negative, positive),
            ranking=(biased.option_id, other.option_id),
            domain=domain.domain_id,
            scenario_id=(
                f"{scenario_family_id(domain.domain_id, 'test')}:"
                f"fixed-bias:{target}"
            ),
            turn_id=str(turn),
            default_option_id=biased.option_id,
            wording_template=dialogue_template_id(
                domain.domain_id,
                "test",
            ),
            question_type="choice",
            target_attribute=target,
        )
        # The current evaluated belief is logged for audit but does not affect the
        # fixed reference action above.
        return PolicyAction(
            context,
            PolicyProvenance(
                policy_id=self.policy_id,
                policy_version=self.policy_version,
                profile_snapshot=_snapshot(belief),
                random_seed=semantic_seed(
                    master_seed, trajectory_id, turn, self.policy_id
                ),
                presentation_mechanism="default",
                profile_conditioned=False,
            ),
        )


@dataclass(frozen=True, slots=True)
class HardFilterPolicy:
    """Secondary stress policy; never used for the main soft-loop claim."""

    policy_id: str = "hard_filter"
    policy_version: str = "v1-stress"

    def action(
        self,
        domain: DomainSpec,
        belief: PreferenceBelief,
        *,
        turn: int,
        master_seed: int,
        trajectory_id: str,
        target_counts: tuple[int, int, int] | None = None,
    ) -> PolicyAction:
        target = turn % 3
        direction = -1 if belief.expected_theta()[target] <= 0 else 1
        matched = build_matched_anchor_set(
            domain,
            target_attribute=target,
            anchor_direction=direction,
            scenario_id=(
                f"{scenario_family_id(domain.domain_id, 'test')}:"
                f"{trajectory_id}:hard:{turn}"
            ),
            wording_template=dialogue_template_id(
                domain.domain_id,
                "test",
            ),
            turn=turn,
        )
        context = matched.context("restricted")
        return PolicyAction(
            context,
            PolicyProvenance(
                policy_id=self.policy_id,
                policy_version=self.policy_version,
                profile_snapshot=_snapshot(belief),
                random_seed=semantic_seed(
                    master_seed, trajectory_id, turn, self.policy_id
                ),
                presentation_mechanism="restriction",
                profile_conditioned=True,
            ),
        )


POLICY_FACTORIES = {
    "balanced": BalancedPolicy,
    "soft_profile_conditioned": SoftProfileConditionedPolicy,
    "exploratory": ExploratoryPolicy,
    "fixed_bias": FixedBiasPolicy,
    "hard_filter": HardFilterPolicy,
}


def build_policy(policy_id: str) -> InteractionPolicy:
    try:
        return POLICY_FACTORIES[policy_id]()
    except KeyError as exc:
        raise KeyError(f"unknown interaction policy: {policy_id}") from exc
