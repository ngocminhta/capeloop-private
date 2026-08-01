"""Experiment B closed-loop trajectories and causal error decomposition."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..beliefs import (
    THETA_STATES,
    JointThetaPsiBelief,
    MarginalPreferenceBelief,
    PreferenceBelief,
)
from ..conversation_surfaces import ConversationTemplateBank
from ..domains import DOMAINS, DomainSpec
from ..inference import exact_aware_update
from ..metrics import (
    SelfConfirmationEvidence,
    attribution_cost,
    false_confidence_gain,
    information_gain,
    laundered_confidence_gain,
    marginal_brier,
    marginal_kl,
    selection_cost,
    self_confirmation_interaction,
)
from ..policies import (
    BalancedPolicy,
    ExploratoryPolicy,
    InteractionPolicy,
    PolicyAction,
    SoftProfileConditionedPolicy,
)
from ..population import (
    INITIAL_PROFILE_KINDS,
    add_prior_uncertainty,
)
from ..population import (
    initial_profile_belief as _population_initial_profile_belief,
)
from ..response import RandomUtilityModel, regret
from ..scenarios import ScenarioCatalog, ScenarioSpec, materialize_context
from ..schemas import (
    InteractionRecord,
    InteractionContext,
    LatentUser,
    Observation,
    Susceptibility,
    Theta,
    TrajectoryRecord,
)
from ..updaters import (
    ExactActionAwareUpdater,
    ProfileUpdater,
    build_updater_registry,
    make_update_view,
)

INITIAL_PROFILE_CONDITIONS = INITIAL_PROFILE_KINDS
BALANCED_CHOICE_MARGIN_THRESHOLDS = (0.20, 0.50)
MIN_INFORMATIVE_SOFT_VISIBLE_TURNS_PER_TRAJECTORY = 2
MIN_INFORMATIVE_SOFT_USERS_PER_DOMAIN = 2


def _preference_strength_stratum(value: int) -> str:
    """Classify the declared latent magnitude without using an observation."""

    magnitude = abs(value)
    if magnitude == 1:
        return "weak"
    if magnitude == 2:
        return "strong"
    raise ValueError("preference strength requires latent magnitude 1 or 2")


def _user_preference_strength_stratum(theta: Theta) -> str:
    strata = {_preference_strength_stratum(value) for value in theta}
    if len(strata) == 1:
        return next(iter(strata))
    return "mixed"


def _balanced_choice_probability_margin(
    probabilities: Sequence[float],
) -> float:
    """Return the top-two probability gap before the natural response is drawn."""

    material = tuple(float(value) for value in probabilities)
    if len(material) < 2:
        raise ValueError("balanced-choice margin requires at least two options")
    if any(
        not math.isfinite(value) or not 0.0 <= value <= 1.0
        for value in material
    ):
        raise ValueError("balanced-choice probabilities must lie in [0, 1]")
    if not math.isclose(
        math.fsum(material),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("balanced-choice probabilities must sum to one")
    first, second = sorted(material, reverse=True)[:2]
    return first - second


def _balanced_choice_margin_stratum(margin: float) -> str:
    """Prospectively bin a probability gap as near-tie, marginal, or decisive."""

    numeric = float(margin)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError("balanced-choice margin must lie in [0, 1]")
    near_tie, decisive = BALANCED_CHOICE_MARGIN_THRESHOLDS
    if numeric < near_tie:
        return "near_tie"
    if numeric < decisive:
        return "marginal"
    return "decisive"


def _expected_action_aware_information_gain(
    prior: JointThetaPsiBelief,
    context: InteractionContext,
    response_model: RandomUtilityModel,
) -> float:
    """Expected preference-state entropy reduction before observing a response.

    The expectation uses only the exact shadow's current joint belief, the
    proposed visible action, and the declared response model.  It therefore
    measures the ex-ante diagnostic value of the action rather than the
    realized entropy change produced by one sampled response.
    """

    psi_count = len(prior.susceptibilities)
    expected_posterior_theta_entropy = 0.0
    predictive_total = 0.0
    for option in context.options:
        observation = Observation(selected_option_id=option.option_id)
        predictive_probability = math.fsum(
            prior.probabilities[theta_index * psi_count + psi_index]
            * response_model.likelihood(
                theta,
                susceptibility,
                context,
                observation,
            )
            for theta_index, theta in enumerate(THETA_STATES)
            for psi_index, susceptibility in enumerate(
                prior.susceptibilities
            )
        )
        posterior = exact_aware_update(
            prior,
            context,
            observation,
            response_model,
        )
        predictive_total += predictive_probability
        expected_posterior_theta_entropy += (
            predictive_probability * posterior.theta_belief().entropy()
        )
    if not math.isclose(
        predictive_total,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise RuntimeError("predictive response probabilities must sum to one")
    # Expected information gain is non-negative in exact arithmetic.  Reject
    # material violations and clamp only floating-point underflow at zero.
    gain = prior.theta_belief().entropy() - expected_posterior_theta_entropy
    if gain < -1e-10:
        raise RuntimeError("expected information gain became materially negative")
    return max(0.0, gain)


def _profile_consistency_score(
    belief: PreferenceBelief,
    context: InteractionContext,
) -> float:
    """Structural alignment of one visible action with the current profile.

    The score averages the signed alignment of the option-set composition,
    first-ranked option, default, and suggestion with the profile's current
    direction on the target attribute.  It lies in ``[-1, 1]``: positive
    favors the profile, negative favors its opposite, and zero is neutral or
    undefined because the current profile has no directional expectation.
    This is an action descriptor, not an outcome and not a claim that the
    action changed user behavior.
    """

    attribute = context.target_attribute
    if attribute is None:
        raise ValueError("profile consistency requires a target attribute")
    expected = belief.expected_theta()[attribute]
    if math.isclose(expected, 0.0, rel_tol=0.0, abs_tol=1e-12):
        return 0.0
    profile_direction = 1 if expected > 0.0 else -1

    def alignment(option_id: str) -> float | None:
        value = context.option(option_id).features[attribute]
        if value == 0.0:
            return None
        return float(profile_direction * (1 if value > 0.0 else -1))

    option_alignments = tuple(
        value
        for option in context.options
        if (value := alignment(option.option_id)) is not None
    )
    components: list[float] = []
    if option_alignments:
        components.append(math.fsum(option_alignments) / len(option_alignments))
    if context.ranking:
        ranked = alignment(context.ranking[0])
        if ranked is not None:
            components.append(ranked)
    for option_id in (
        context.default_option_id,
        context.suggested_option_id,
    ):
        if option_id is None:
            continue
        promoted = alignment(option_id)
        if promoted is not None:
            components.append(promoted)
    if not components:
        return 0.0
    return math.fsum(components) / len(components)


def _binary_shared_noise_choice_divergence_probability(
    actual_context: InteractionContext,
    actual_probabilities: Sequence[float],
    balanced_context: InteractionContext,
    balanced_probabilities: Sequence[float],
) -> float | None:
    """Ex-ante probability that paired binary common-noise choices differ.

    For the same two option IDs under the monotone shared-noise coupling used
    by both response simulators, the probability of different winners is the
    absolute change in either option's choice probability. The random-utility
    model realizes this with shared Gumbels; the rule-based model uses a shared
    inverse-CDF draw. Actions with different choice sets remain null.
    """

    if (
        len(actual_context.options) != 2
        or len(balanced_context.options) != 2
        or set(actual_context.option_ids) != set(balanced_context.option_ids)
    ):
        return None
    actual = dict(zip(actual_context.option_ids, actual_probabilities))
    balanced = dict(zip(balanced_context.option_ids, balanced_probabilities))
    reference_id = sorted(actual)[0]
    return abs(actual[reference_id] - balanced[reference_id])


def initial_profile_belief(
    theta: Theta,
    condition: str,
    *,
    strength: float = 0.80,
    prior_uncertainty: float = 0.0,
) -> PreferenceBelief:
    """Use the repository's canonical crossed initial-profile constructor."""

    return add_prior_uncertainty(
        _population_initial_profile_belief(
            theta,
            condition,
            profile_strength=strength,
        ),
        prior_uncertainty,
    )


def wrong_directions(theta: Theta) -> tuple[int, int, int]:
    return tuple(-1 if value > 0 else 1 for value in theta)  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class ClosedLoopTurn:
    turn: int
    event_id: str
    context_id: str
    scenario_id: str
    policy_id: str
    target_attribute: int
    ex_ante_target_preference_strength: int
    ex_ante_target_preference_strength_stratum: str
    ex_ante_balanced_target_attribute: int
    ex_ante_balanced_choice_probability_margin: float
    ex_ante_balanced_choice_margin_stratum: str
    selected_option_id: str
    common_noise_key: str
    belief_before: PreferenceBelief
    belief_after: PreferenceBelief
    shadow_before: PreferenceBelief
    shadow_after: PreferenceBelief
    joint_belief_before: JointThetaPsiBelief | None
    joint_belief_after: JointThetaPsiBelief | None
    shadow_joint_before: JointThetaPsiBelief | None
    shadow_joint_after: JointThetaPsiBelief | None
    wrong_mass_before: tuple[float, float, float]
    wrong_mass_after: tuple[float, float, float]
    shadow_wrong_mass_before: tuple[float, float, float]
    shadow_wrong_mass_after: tuple[float, float, float]
    system_false_confidence_gain: tuple[float, float, float]
    shadow_false_confidence_gain: tuple[float, float, float]
    laundered_confidence_gain: tuple[float, float, float]
    expected_action_aware_information_gain: float
    action_aware_information_gain: float
    information_gain_state_space: str
    profile_consistency_score: float
    balanced_profile_consistency_score: float
    ex_ante_balanced_choice_divergence_probability: float | None
    intrinsic_regret: float
    profile_influenced_action: bool
    profile_attribute_influenced_action: tuple[bool, bool, bool]
    action_signature: tuple[object, ...]
    balanced_action_signature: tuple[object, ...]
    unstrengthened_action_signatures: tuple[
        tuple[object, ...],
        tuple[object, ...],
        tuple[object, ...],
    ]
    native_state_before: Mapping[str, Any] | None
    native_state_after: Mapping[str, Any] | None
    theta_snapshot: Theta
    prospective_manipulation_role: str | None = None
    prospective_presentation_mechanism: str | None = None
    prospective_predicted_choice_divergence_probability: float | None = None
    prospective_execution_matched: bool | None = None
    prospective_effective_profile_direction: int | None = None
    prospective_direction_source: str | None = None

    def to_dict(self, *, include_truth: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "turn": self.turn,
            "event_id": self.event_id,
            "context_id": self.context_id,
            "scenario_id": self.scenario_id,
            "policy_id": self.policy_id,
            "target_attribute": self.target_attribute,
            "ex_ante_target_preference_strength": (
                self.ex_ante_target_preference_strength
            ),
            "ex_ante_target_preference_strength_stratum": (
                self.ex_ante_target_preference_strength_stratum
            ),
            "ex_ante_balanced_target_attribute": (
                self.ex_ante_balanced_target_attribute
            ),
            "ex_ante_balanced_choice_probability_margin": (
                self.ex_ante_balanced_choice_probability_margin
            ),
            "ex_ante_balanced_choice_margin_stratum": (
                self.ex_ante_balanced_choice_margin_stratum
            ),
            "selected_option_id": self.selected_option_id,
            "common_noise_key": self.common_noise_key,
            "wrong_mass_before": list(self.wrong_mass_before),
            "wrong_mass_after": list(self.wrong_mass_after),
            "shadow_wrong_mass_before": list(self.shadow_wrong_mass_before),
            "shadow_wrong_mass_after": list(self.shadow_wrong_mass_after),
            "system_false_confidence_gain": list(self.system_false_confidence_gain),
            "shadow_false_confidence_gain": list(self.shadow_false_confidence_gain),
            "laundered_confidence_gain": list(self.laundered_confidence_gain),
            "expected_action_aware_information_gain": (
                self.expected_action_aware_information_gain
            ),
            "action_aware_information_gain": self.action_aware_information_gain,
            "information_gain_state_space": self.information_gain_state_space,
            "profile_consistency_score": self.profile_consistency_score,
            "balanced_profile_consistency_score": (
                self.balanced_profile_consistency_score
            ),
            "profile_consistency_advantage_over_balanced": (
                self.profile_consistency_score
                - self.balanced_profile_consistency_score
            ),
            "ex_ante_balanced_choice_divergence_probability": (
                self.ex_ante_balanced_choice_divergence_probability
            ),
            "intrinsic_regret": self.intrinsic_regret,
            "profile_influenced_action": self.profile_influenced_action,
            "profile_attribute_influenced_action": list(
                self.profile_attribute_influenced_action
            ),
            "action_signature": self.action_signature,
            "balanced_action_signature": self.balanced_action_signature,
            "visible_action_diverged_from_balanced": (
                self.action_signature != self.balanced_action_signature
            ),
            "unstrengthened_action_signatures": list(
                self.unstrengthened_action_signatures
            ),
            "belief_before": self.belief_before.to_dict(),
            "belief_after": self.belief_after.to_dict(),
            "shadow_before": self.shadow_before.to_dict(),
            "shadow_after": self.shadow_after.to_dict(),
            "joint_belief_before": (
                None
                if self.joint_belief_before is None
                else self.joint_belief_before.to_dict()
            ),
            "joint_belief_after": (
                None
                if self.joint_belief_after is None
                else self.joint_belief_after.to_dict()
            ),
            "shadow_joint_before": (
                None
                if self.shadow_joint_before is None
                else self.shadow_joint_before.to_dict()
            ),
            "shadow_joint_after": (
                None
                if self.shadow_joint_after is None
                else self.shadow_joint_after.to_dict()
            ),
            "native_state_before": self.native_state_before,
            "native_state_after": self.native_state_after,
            "prospective_manipulation_role": (
                self.prospective_manipulation_role
            ),
            "prospective_presentation_mechanism": (
                self.prospective_presentation_mechanism
            ),
            "prospective_predicted_choice_divergence_probability": (
                self.prospective_predicted_choice_divergence_probability
            ),
            "prospective_execution_matched": (
                self.prospective_execution_matched
            ),
            "prospective_effective_profile_direction": (
                self.prospective_effective_profile_direction
            ),
            "prospective_direction_source": self.prospective_direction_source,
        }
        if include_truth:
            result["theta_snapshot"] = list(self.theta_snapshot)
        return result


@dataclass(frozen=True, slots=True)
class ClosedLoopTrajectory:
    trajectory_id: str
    crn_key: str
    schedule_group_key: str
    user_id: str
    theta: Theta
    susceptibility: Susceptibility
    domain_id: str
    updater_id: str
    policy_id: str
    initial_profile_condition: str
    initial_belief: PreferenceBelief
    terminal_belief: PreferenceBelief
    terminal_shadow_belief: PreferenceBelief
    terminal_joint_belief: JointThetaPsiBelief | None
    terminal_shadow_joint_belief: JointThetaPsiBelief | None
    terminal_opaque_state: object | None
    direction_tolerance: float
    turns: tuple[ClosedLoopTurn, ...]
    audit_record: TrajectoryRecord

    def __post_init__(self) -> None:
        if not self.turns:
            raise ValueError("a closed-loop trajectory requires at least one turn")
        if any(turn.theta_snapshot != self.theta for turn in self.turns):
            raise ValueError("latent preference changed within a trajectory")
        event_ids = tuple(turn.event_id for turn in self.turns)
        if event_ids != tuple(
            interaction.record_id for interaction in self.audit_record.interactions
        ):
            raise ValueError("turns and audit interactions are misaligned")
        if self.audit_record.trajectory_id != self.trajectory_id:
            raise ValueError("audit record trajectory ID differs")
        if (
            not math.isfinite(self.direction_tolerance)
            or self.direction_tolerance < 0
        ):
            raise ValueError(
                "direction_tolerance must be finite and non-negative"
            )

    @property
    def terminal_error(self) -> float:
        return marginal_brier(self.terminal_belief, self.theta)

    @property
    def initial_error(self) -> float:
        return marginal_brier(self.initial_belief, self.theta)

    @property
    def error_amplification_ratio(self) -> float | None:
        """Terminal error divided by initial error.

        The ratio is undefined for an exactly correct point-mass seed. Keeping
        that case null avoids manufacturing an infinite value.
        """

        if self.initial_error <= 1e-15:
            return None
        return self.terminal_error / self.initial_error

    @property
    def terminal_shadow_error(self) -> float:
        return marginal_brier(self.terminal_shadow_belief, self.theta)

    @property
    def same_history_attribution_gap(self) -> float:
        """System error minus exact-shadow error on the identical history."""

        return self.terminal_error - self.terminal_shadow_error

    @property
    def exact_shadow_error_improvement(self) -> float:
        """Initial error minus terminal exact-shadow error; positive is better."""

        return self.initial_error - self.terminal_shadow_error

    @property
    def terminal_shadow_to_system_marginal_kl(self) -> float:
        """Terminal divergence from the same-history exact shadow."""

        return marginal_kl(
            self.terminal_shadow_belief,
            self.terminal_belief,
        )

    @property
    def preference_dimension_coverage(self) -> float:
        covered: set[int] = set()
        for interaction in self.audit_record.interactions:
            attribute = interaction.context.target_attribute
            if attribute is None:
                continue
            directions = {
                1 if option.features[attribute] > 0.0 else -1
                for option in interaction.context.options
                if option.features[attribute] != 0.0
            }
            if directions == {-1, 1}:
                covered.add(attribute)
        return len(covered) / 3.0

    @property
    def turns_to_full_preference_coverage(self) -> int | None:
        covered: set[int] = set()
        for index, interaction in enumerate(
            self.audit_record.interactions,
            start=1,
        ):
            attribute = interaction.context.target_attribute
            if attribute is not None:
                directions = {
                    1 if option.features[attribute] > 0.0 else -1
                    for option in interaction.context.options
                    if option.features[attribute] != 0.0
                }
                if directions == {-1, 1}:
                    covered.add(attribute)
            if len(covered) == 3:
                return index
        return None

    @property
    def profile_conditioned_exposure_rate(self) -> float:
        return math.fsum(
            interaction.provenance.profile_conditioned
            for interaction in self.audit_record.interactions
        ) / len(self.audit_record.interactions)

    @property
    def ex_ante_preference_strengths_by_attribute(
        self,
    ) -> tuple[int, int, int]:
        return tuple(abs(value) for value in self.theta)  # type: ignore[return-value]

    @property
    def ex_ante_preference_strength_strata_by_attribute(
        self,
    ) -> tuple[str, str, str]:
        return tuple(  # type: ignore[return-value]
            _preference_strength_stratum(value) for value in self.theta
        )

    @property
    def ex_ante_user_preference_strength_stratum(self) -> str:
        return _user_preference_strength_stratum(self.theta)

    @property
    def ex_ante_balanced_choice_mean_probability_margin(self) -> float:
        return math.fsum(
            turn.ex_ante_balanced_choice_probability_margin
            for turn in self.turns
        ) / len(self.turns)

    @property
    def ex_ante_balanced_choice_margin_stratum_counts(
        self,
    ) -> dict[str, int]:
        counts = {"near_tie": 0, "marginal": 0, "decisive": 0}
        for turn in self.turns:
            counts[turn.ex_ante_balanced_choice_margin_stratum] += 1
        return counts

    @property
    def presentation_mechanism_count(self) -> int:
        return len(
            {
                interaction.provenance.presentation_mechanism
                for interaction in self.audit_record.interactions
            }
        )

    @property
    def presentation_mechanism_evenness(self) -> float:
        counts: dict[str, int] = {}
        for interaction in self.audit_record.interactions:
            mechanism = interaction.provenance.presentation_mechanism
            counts[mechanism] = counts.get(mechanism, 0) + 1
        if len(counts) <= 1:
            return 0.0
        total = len(self.audit_record.interactions)
        entropy = -math.fsum(
            (count / total) * math.log(count / total) for count in counts.values()
        )
        return entropy / math.log(len(counts))

    @property
    def displayed_option_diversity(self) -> float:
        """Fraction of the domain's six isolated options ever displayed."""

        displayed = {
            option.features
            for interaction in self.audit_record.interactions
            for option in interaction.context.options
            if sum(value != 0.0 for value in option.features) == 1
        }
        return min(1.0, len(displayed) / 6.0)

    @property
    def selected_option_count(self) -> int:
        return len({turn.selected_option_id for turn in self.turns})

    @property
    def cumulative_lcg(self) -> tuple[float, float, float]:
        return tuple(
            math.fsum(turn.laundered_confidence_gain[attribute] for turn in self.turns)
            for attribute in range(3)
        )  # type: ignore[return-value]

    @property
    def cumulative_information_gain(self) -> float:
        return math.fsum(turn.action_aware_information_gain for turn in self.turns)

    @property
    def cumulative_expected_information_gain(self) -> float:
        return math.fsum(
            turn.expected_action_aware_information_gain for turn in self.turns
        )

    @property
    def mean_profile_consistency_score(self) -> float:
        return math.fsum(
            turn.profile_consistency_score for turn in self.turns
        ) / len(self.turns)

    @property
    def mean_profile_consistency_advantage_over_balanced(self) -> float:
        return math.fsum(
            turn.profile_consistency_score
            - turn.balanced_profile_consistency_score
            for turn in self.turns
        ) / len(self.turns)

    @property
    def mean_ex_ante_balanced_choice_divergence_probability(
        self,
    ) -> float | None:
        """Mean choice-change probability on comparable binary turns only."""

        values = tuple(
            turn.ex_ante_balanced_choice_divergence_probability
            for turn in self.turns
            if turn.ex_ante_balanced_choice_divergence_probability is not None
        )
        if not values:
            return None
        return math.fsum(values) / len(values)

    @property
    def ex_ante_balanced_choice_comparable_turn_count(self) -> int:
        return sum(
            turn.ex_ante_balanced_choice_divergence_probability is not None
            for turn in self.turns
        )

    @property
    def ex_ante_balanced_choice_comparable_turn_rate(self) -> float:
        return (
            self.ex_ante_balanced_choice_comparable_turn_count
            / len(self.turns)
        )

    @property
    def balanced_choice_set_divergence_count(self) -> int:
        return sum(
            set(turn.action_signature[0])
            != set(turn.balanced_action_signature[0])
            for turn in self.turns
        )

    @property
    def balanced_choice_set_divergence_rate(self) -> float:
        return self.balanced_choice_set_divergence_count / len(self.turns)

    @property
    def initially_false_attributes(self) -> tuple[int, ...]:
        """Attributes whose seed assigns majority mass to the wrong sign."""

        wrong = wrong_directions(self.theta)
        return tuple(
            attribute
            for attribute in range(3)
            if self.initial_belief.sign_mass(
                attribute,
                wrong[attribute],
            )
            > 0.5 + 1e-9
        )

    @property
    def mean_cumulative_excess_confidence_log_odds(
        self,
    ) -> float | None:
        """Mean cumulative LCG over attributes initially seeded false."""

        attributes = self.initially_false_attributes
        if not attributes:
            return None
        cumulative = self.cumulative_lcg
        return math.fsum(cumulative[attribute] for attribute in attributes) / len(
            attributes
        )

    @property
    def action_aware_disconfirmation_gain_log_odds(self) -> float | None:
        """Mean exact-shadow evidence against initially false attributes."""

        attributes = self.initially_false_attributes
        if not attributes:
            return None
        return math.fsum(
            -math.fsum(
                turn.shadow_false_confidence_gain[attribute]
                for turn in self.turns
            )
            for attribute in attributes
        ) / len(attributes)

    def profile_aligned_treatment_flags(self) -> tuple[bool, ...]:
        """Whether each visible treatment promoted the initially false sign."""

        wrong = wrong_directions(self.theta)
        false_attributes = set(self.initially_false_attributes)
        flags = []
        for turn, interaction in zip(
            self.turns,
            self.audit_record.interactions,
        ):
            attribute = turn.target_attribute
            provenance = interaction.provenance
            context = interaction.context
            promoted_option_id: str | None = None
            if (
                attribute in false_attributes
                and provenance.profile_conditioned
                and turn.action_signature
                != turn.balanced_action_signature
            ):
                if provenance.presentation_mechanism == "ranking":
                    promoted_option_id = context.ranking[0]
                elif provenance.presentation_mechanism == "default":
                    promoted_option_id = context.default_option_id
                elif provenance.presentation_mechanism == "suggestion":
                    promoted_option_id = context.suggested_option_id
            if promoted_option_id is None:
                flags.append(False)
                continue
            promoted_value = context.option(
                promoted_option_id
            ).features[attribute]
            promoted_direction = (
                1 if promoted_value > 0.0 else -1 if promoted_value < 0.0 else 0
            )
            flags.append(promoted_direction == wrong[attribute])
        return tuple(flags)

    def reinforcement_event_flags(
        self,
        *,
        tolerance: float | None = None,
    ) -> tuple[bool, ...]:
        """Partial-loop events under the review's four-clause definition."""

        threshold = (
            self.direction_tolerance if tolerance is None else float(tolerance)
        )
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError(
                "reinforcement-event tolerance must be finite and non-negative"
            )
        wrong = wrong_directions(self.theta)
        treatment_flags = self.profile_aligned_treatment_flags()
        events = []
        for treatment_applied, turn, interaction in zip(
            treatment_flags,
            self.turns,
            self.audit_record.interactions,
        ):
            attribute = turn.target_attribute
            selected_value = interaction.context.option(
                turn.selected_option_id
            ).features[attribute]
            selected_direction = (
                1 if selected_value > 0.0 else -1 if selected_value < 0.0 else 0
            )
            events.append(
                treatment_applied
                and selected_direction == wrong[attribute]
                and turn.system_false_confidence_gain[attribute] > threshold
                and turn.laundered_confidence_gain[attribute] > threshold
            )
        return tuple(events)

    def disconfirmation_inversion_counts(
        self,
        *,
        tolerance: float | None = None,
    ) -> tuple[int, int]:
        """Return exact-disconfirmation opportunities and sign inversions.

        An opportunity is an initially false attribute-turn on which the exact
        same-history shadow reduces confidence in the false sign.  It becomes
        an inversion when the evaluated updater instead increases confidence
        in that same false sign.  The definition intentionally does not
        require a profile-conditioned action or a behavior change; those are
        separate feedback-loop conditions.
        """

        threshold = (
            self.direction_tolerance if tolerance is None else float(tolerance)
        )
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError(
                "disconfirmation-inversion tolerance must be finite and "
                "non-negative"
            )
        opportunities = 0
        inversions = 0
        for turn in self.turns:
            for attribute in self.initially_false_attributes:
                if turn.shadow_false_confidence_gain[attribute] < -threshold:
                    opportunities += 1
                    if turn.system_false_confidence_gain[attribute] > threshold:
                        inversions += 1
        return opportunities, inversions

    def disconfirmation_inversion_turn_flags(
        self,
        *,
        tolerance: float | None = None,
    ) -> tuple[bool, ...]:
        """Whether any initially false attribute inverted on each turn."""

        threshold = (
            self.direction_tolerance if tolerance is None else float(tolerance)
        )
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError(
                "disconfirmation-inversion tolerance must be finite and "
                "non-negative"
            )
        attributes = self.initially_false_attributes
        return tuple(
            any(
                turn.shadow_false_confidence_gain[attribute] < -threshold
                and turn.system_false_confidence_gain[attribute] > threshold
                for attribute in attributes
            )
            for turn in self.turns
        )

    @property
    def disconfirmation_opportunity_count(self) -> int:
        return self.disconfirmation_inversion_counts()[0]

    @property
    def disconfirmation_inversion_count(self) -> int:
        return self.disconfirmation_inversion_counts()[1]

    @property
    def disconfirmation_inversion_rate(self) -> float | None:
        opportunities, inversions = self.disconfirmation_inversion_counts()
        if opportunities == 0:
            return None
        return inversions / opportunities

    @property
    def disconfirmation_inversion_turn_rate(self) -> float | None:
        if not self.initially_false_attributes:
            return None
        return sum(self.disconfirmation_inversion_turn_flags()) / len(self.turns)

    @property
    def reinforcement_event_count(self) -> int:
        return sum(self.reinforcement_event_flags())

    @property
    def profile_aligned_treatment_opportunities(self) -> int:
        return sum(self.profile_aligned_treatment_flags())

    @property
    def reinforcement_event_rate(self) -> float | None:
        if not self.initially_false_attributes:
            return None
        return self.reinforcement_event_count / len(self.turns)

    @property
    def total_regret(self) -> float:
        return math.fsum(turn.intrinsic_regret for turn in self.turns)

    @property
    def same_history_shadow(self) -> bool:
        """The shadow consumes each actual event exactly once."""

        return tuple(turn.event_id for turn in self.turns) == tuple(
            interaction.record_id for interaction in self.audit_record.interactions
        ) and len(self.turns) == len(self.audit_record.interactions)

    def to_dict(self, *, include_truth: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "trajectory_id": self.trajectory_id,
            "crn_key": self.crn_key,
            "schedule_group_key": self.schedule_group_key,
            "user_id": self.user_id,
            "domain_id": self.domain_id,
            "updater_id": self.updater_id,
            "policy_id": self.policy_id,
            "initial_profile_condition": self.initial_profile_condition,
            "initial_belief": self.initial_belief.to_dict(),
            "terminal_belief": self.terminal_belief.to_dict(),
            "terminal_shadow_belief": self.terminal_shadow_belief.to_dict(),
            "terminal_joint_belief": (
                None
                if self.terminal_joint_belief is None
                else self.terminal_joint_belief.to_dict()
            ),
            "terminal_shadow_joint_belief": (
                None
                if self.terminal_shadow_joint_belief is None
                else self.terminal_shadow_joint_belief.to_dict()
            ),
            "terminal_native_state": _opaque_state_payload(self.terminal_opaque_state),
            "initial_error": self.initial_error,
            "terminal_error": self.terminal_error,
            "terminal_shadow_error": self.terminal_shadow_error,
            "same_history_attribution_gap": self.same_history_attribution_gap,
            "exact_shadow_error_improvement": (
                self.exact_shadow_error_improvement
            ),
            "error_amplification_ratio": self.error_amplification_ratio,
            "terminal_shadow_to_system_marginal_kl": (
                self.terminal_shadow_to_system_marginal_kl
            ),
            "preference_dimension_coverage": (self.preference_dimension_coverage),
            "turns_to_full_preference_coverage": (
                self.turns_to_full_preference_coverage
            ),
            "displayed_option_diversity": self.displayed_option_diversity,
            "selected_option_count": self.selected_option_count,
            "profile_conditioned_exposure_rate": (
                self.profile_conditioned_exposure_rate
            ),
            "ex_ante_preference_strengths_by_attribute": list(
                self.ex_ante_preference_strengths_by_attribute
            ),
            "ex_ante_preference_strength_strata_by_attribute": list(
                self.ex_ante_preference_strength_strata_by_attribute
            ),
            "ex_ante_user_preference_strength_stratum": (
                self.ex_ante_user_preference_strength_stratum
            ),
            "ex_ante_balanced_choice_mean_probability_margin": (
                self.ex_ante_balanced_choice_mean_probability_margin
            ),
            "ex_ante_balanced_choice_margin_stratum_counts": dict(
                self.ex_ante_balanced_choice_margin_stratum_counts
            ),
            "presentation_mechanism_count": (self.presentation_mechanism_count),
            "presentation_mechanism_evenness": (self.presentation_mechanism_evenness),
            "cumulative_lcg": list(self.cumulative_lcg),
            "cumulative_information_gain": self.cumulative_information_gain,
            "cumulative_expected_information_gain": (
                self.cumulative_expected_information_gain
            ),
            "mean_profile_consistency_score": (
                self.mean_profile_consistency_score
            ),
            "mean_profile_consistency_advantage_over_balanced": (
                self.mean_profile_consistency_advantage_over_balanced
            ),
            "mean_ex_ante_balanced_choice_divergence_probability": (
                self.mean_ex_ante_balanced_choice_divergence_probability
            ),
            "ex_ante_balanced_choice_comparable_turn_count": (
                self.ex_ante_balanced_choice_comparable_turn_count
            ),
            "ex_ante_balanced_choice_comparable_turn_rate": (
                self.ex_ante_balanced_choice_comparable_turn_rate
            ),
            "balanced_choice_set_divergence_count": (
                self.balanced_choice_set_divergence_count
            ),
            "balanced_choice_set_divergence_rate": (
                self.balanced_choice_set_divergence_rate
            ),
            "initially_false_attributes": list(
                self.initially_false_attributes
            ),
            "mean_cumulative_excess_confidence_log_odds": (
                self.mean_cumulative_excess_confidence_log_odds
            ),
            "action_aware_disconfirmation_gain_log_odds": (
                self.action_aware_disconfirmation_gain_log_odds
            ),
            "profile_aligned_treatment_opportunities": (
                self.profile_aligned_treatment_opportunities
            ),
            "reinforcement_event_count": self.reinforcement_event_count,
            "reinforcement_event_rate": self.reinforcement_event_rate,
            "disconfirmation_inversion_tolerance": self.direction_tolerance,
            "disconfirmation_opportunity_count": (
                self.disconfirmation_opportunity_count
            ),
            "disconfirmation_inversion_count": (
                self.disconfirmation_inversion_count
            ),
            "disconfirmation_inversion_rate": (
                self.disconfirmation_inversion_rate
            ),
            "disconfirmation_inversion_turn_rate": (
                self.disconfirmation_inversion_turn_rate
            ),
            "total_regret": self.total_regret,
            "same_history_shadow": self.same_history_shadow,
            "turns": [turn.to_dict(include_truth=include_truth) for turn in self.turns],
            "audit_record": self.audit_record.to_dict(),
        }
        if include_truth:
            result["theta"] = list(self.theta)
            result["susceptibility"] = self.susceptibility.to_dict()
        return result


def _profile_snapshot(
    belief: PreferenceBelief,
) -> tuple[tuple[str, float], ...]:
    return tuple(
        (f"attribute_{index + 1}", value)
        for index, value in enumerate(belief.expected_theta())
    )


def _reset_attribute_to_initial(
    current: PreferenceBelief,
    initial: PreferenceBelief,
    attribute: int,
) -> PreferenceBelief:
    """Counterfactual profile with one attribute's update history removed.

    Policies consume only marginal expectations/entropies. Reconstructing an
    independent joint from the mixed marginals therefore preserves every
    policy-visible quantity while resetting the selected attribute exactly.
    """

    rows = list(current.marginals().probabilities)
    rows[attribute] = initial.marginal(attribute)
    marginals = MarginalPreferenceBelief((rows[0], rows[1], rows[2]))
    return PreferenceBelief.from_marginals(marginals)


def _opaque_state_payload(value: object | None) -> Mapping[str, Any] | None:
    if value is None:
        return None
    to_dict = getattr(value, "to_dict", None)
    if not callable(to_dict):
        return None
    payload = to_dict()
    if not isinstance(payload, Mapping):
        raise TypeError("opaque updater state to_dict() must return a mapping")
    return payload


def run_trajectory(
    *,
    user: LatentUser,
    domain: DomainSpec,
    policy: InteractionPolicy,
    updater: ProfileUpdater,
    turns: int,
    seed: int,
    response_seed: int | None = None,
    initial_belief: PreferenceBelief | None = None,
    initial_profile_condition: str = "empty",
    profile_strength: float = 0.80,
    prior_uncertainty: float = 0.0,
    response_model: RandomUtilityModel | None = None,
    shadow_updater: ExactActionAwareUpdater | None = None,
    direction_tolerance: float = 1e-9,
    trajectory_id: str | None = None,
    crn_key: str | None = None,
    scenario_catalog: ScenarioCatalog | None = None,
    conversation_bank: ConversationTemplateBank | None = None,
    data_split: str = "test",
) -> ClosedLoopTrajectory:
    """Run an endogenous loop with an exact aware same-history shadow.

    The user object is read but never mutated.  Choice Gumbels are keyed by the
    caller-controlled ``crn_key`` and turn, so overlapping options are paired
    across counterfactual policy/updater branches.
    """

    if turns <= 0:
        raise ValueError("turns must be positive")
    if not math.isfinite(direction_tolerance) or direction_tolerance < 0:
        raise ValueError("direction_tolerance must be finite and non-negative")
    if domain.domain_id not in {option.domain for option in domain.option_pool}:
        raise ValueError("domain option pool is internally inconsistent")
    declared_response = response_model or RandomUtilityModel()
    declared_response_seed = seed if response_seed is None else response_seed
    if (
        isinstance(declared_response_seed, bool)
        or not isinstance(declared_response_seed, int)
        or declared_response_seed < 0
    ):
        raise ValueError("response_seed must be a non-negative integer")
    initial = (
        initial_profile_belief(
            user.theta,
            initial_profile_condition,
            strength=profile_strength,
            prior_uncertainty=prior_uncertainty,
        )
        if initial_belief is None
        else initial_belief
    )
    identifier = trajectory_id or (
        f"{domain.domain_id}:{user.user_id}:{policy.policy_id}:"
        f"{updater.updater_id}:{initial_profile_condition}"
    )
    common_key = crn_key or identifier
    evaluated_state = updater.initial_state(initial)
    shadow = shadow_updater or ExactActionAwareUpdater(declared_response)
    shadow_state = shadow.initial_state(initial)
    wrong = wrong_directions(user.theta)
    traces: list[ClosedLoopTurn] = []
    interactions: list[InteractionRecord] = []
    scenario_occurrences = [0, 0, 0]
    target_occurrences = [0, 0, 0]
    prospective_plan = getattr(policy, "prospective_plan", None)
    if prospective_plan is not None and scenario_catalog is None:
        raise ValueError(
            "prospective manipulation execution requires a scenario catalog"
        )
    schedule_group_key = common_key
    if prospective_plan is not None:
        schedule_lookup = getattr(prospective_plan, "schedule_key", None)
        if not callable(schedule_lookup):
            raise TypeError(
                "prospective manipulation plan must expose schedule_key(key)"
            )
        schedule_group_key = schedule_lookup(common_key)
        if not isinstance(schedule_group_key, str) or not schedule_group_key:
            raise ValueError(
                "prospective manipulation schedule key must be non-empty"
            )

    def with_catalog(
        action: PolicyAction,
        *,
        advance: bool,
        preferred_scenario: ScenarioSpec | None = None,
    ) -> PolicyAction:
        if scenario_catalog is None:
            return action
        target = action.context.target_attribute
        if target is None:
            raise ValueError("catalog-backed policy action requires a target attribute")
        if (
            preferred_scenario is not None
            and preferred_scenario.domain == domain.domain_id
            and preferred_scenario.split == data_split
            and preferred_scenario.target_attribute == target
        ):
            scenario = preferred_scenario
        else:
            occurrence = scenario_occurrences[target]
            scenario = scenario_catalog.select_cycle(
                domain=domain.domain_id,
                split=data_split,
                target_attribute=target,
                seed=seed,
                cycle_key=("closed-loop", common_key),
                occurrence_index=occurrence,
            )
            if advance:
                scenario_occurrences[target] += 1
        return PolicyAction(
            context=materialize_context(action.context, scenario),
            provenance=action.provenance,
        )

    for turn in range(turns):
        planned_turn = (
            None
            if prospective_plan is None
            else prospective_plan.turn(common_key, turn)
        )
        # The actual action uses only the updater's current public profile.
        counts_before = tuple(target_occurrences)
        raw_action = policy.action(
            domain,
            evaluated_state.belief,
            turn=turn,
            master_seed=seed,
            trajectory_id=common_key,
            target_counts=counts_before,
        )
        actual_target = raw_action.context.target_attribute
        if actual_target is None:
            raise ValueError("closed-loop policy action requires a target attribute")
        if (
            planned_turn is not None
            and actual_target != planned_turn.target_attribute
        ):
            raise RuntimeError(
                "policy target differs from the prospective manipulation plan"
            )
        target_occurrences[actual_target] += 1
        planned_scenario = (
            None
            if planned_turn is None or scenario_catalog is None
            else scenario_catalog.scenario(planned_turn.scenario_id)
        )
        action = with_catalog(
            raw_action,
            advance=True,
            preferred_scenario=planned_scenario,
        )
        actual_scenario = (
            None
            if scenario_catalog is None
            else scenario_catalog.scenario(action.context.scenario_id)
        )
        # Evaluator-only, per-attribute counterfactuals remove the accumulated
        # update to one profile dimension while preserving all other
        # policy-visible marginals and semantic randomness. They reuse the
        # actual scenario whenever the target is unchanged and never consume
        # the trajectory's scenario schedule. This measures whether
        # strengthening that attribute—not stimulus drift or merely seeding it
        # wrong—changed the subsequent action.
        counterfactual_actions = tuple(
            with_catalog(
                policy.action(
                    domain,
                    _reset_attribute_to_initial(
                        evaluated_state.belief,
                        initial,
                        attribute,
                    ),
                    turn=turn,
                    master_seed=seed,
                    trajectory_id=common_key,
                    target_counts=counts_before,
                ),
                advance=False,
                preferred_scenario=actual_scenario,
            )
            for attribute in range(3)
        )
        action_signature = action.signature()
        balanced_counterfactual_action = with_catalog(
            BalancedPolicy(prospective_plan=prospective_plan).action(
                domain,
                evaluated_state.belief,
                turn=turn,
                master_seed=seed,
                trajectory_id=common_key,
                target_counts=counts_before,
            ),
            advance=False,
            preferred_scenario=actual_scenario,
        )
        balanced_action_signature = (
            balanced_counterfactual_action.signature()
        )
        balanced_target = (
            balanced_counterfactual_action.context.target_attribute
        )
        if balanced_target is None:
            raise ValueError(
                "balanced counterfactual requires a target attribute"
            )
        balanced_probabilities = declared_response.probabilities(
            user.theta,
            user.susceptibility,
            balanced_counterfactual_action.context,
        )
        actual_probabilities = declared_response.probabilities(
            user.theta,
            user.susceptibility,
            action.context,
        )
        balanced_probability_margin = (
            _balanced_choice_probability_margin(balanced_probabilities)
        )
        balanced_margin_stratum = _balanced_choice_margin_stratum(
            balanced_probability_margin
        )
        target_preference_strength = abs(user.theta[actual_target])
        target_preference_strength_stratum = (
            _preference_strength_stratum(user.theta[actual_target])
        )
        counterfactual_signatures = tuple(
            candidate.signature() for candidate in counterfactual_actions
        )
        attribute_influence = tuple(
            action_signature != signature for signature in counterfactual_signatures
        )
        profile_influenced = any(attribute_influence)
        shadow_joint_for_action = shadow_state.joint_belief
        if shadow_joint_for_action is None:
            raise RuntimeError(
                "exact shadow requires a joint belief before every action"
            )
        expected_action_information_gain = (
            _expected_action_aware_information_gain(
                shadow_joint_for_action,
                action.context,
                shadow.response_model,
            )
        )
        profile_consistency = _profile_consistency_score(
            evaluated_state.belief,
            action.context,
        )
        balanced_profile_consistency = _profile_consistency_score(
            evaluated_state.belief,
            balanced_counterfactual_action.context,
        )
        ex_ante_choice_divergence_probability = (
            _binary_shared_noise_choice_divergence_probability(
                action.context,
                actual_probabilities,
                balanced_counterfactual_action.context,
                balanced_probabilities,
            )
        )
        prospective_execution_matched: bool | None = None
        prospective_effective_profile_direction: int | None = None
        prospective_direction_source: str | None = None
        if planned_turn is not None:
            active_role = planned_turn.role in {
                "informative_active",
                "decisive_active_control",
            }
            if policy.policy_id == "soft_profile_conditioned" and active_role:
                current_expected = evaluated_state.belief.expected_theta()[
                    actual_target
                ]
                current_direction = (
                    -1
                    if current_expected < -1e-12
                    else 1
                    if current_expected > 1e-12
                    else 0
                )
                frozen_direction = planned_turn.planned_profile_direction
                effective_direction = (
                    current_direction
                    if current_direction != 0
                    else frozen_direction
                )
                if effective_direction not in {-1, 1}:
                    raise RuntimeError(
                        "required prospective active turn has neither a current "
                        "nor frozen profile direction"
                    )
                prospective_effective_profile_direction = effective_direction
                prospective_direction_source = (
                    "current_profile"
                    if current_direction != 0
                    else "frozen_initial_profile_fallback"
                )
                promoted_option_id = (
                    action.context.default_option_id
                    if planned_turn.mechanism == "default"
                    else action.context.suggested_option_id
                    if planned_turn.mechanism == "suggestion"
                    else None
                )
                promotion_matches_effective_profile = (
                    promoted_option_id is not None
                    and action.context.option(promoted_option_id).features[
                        actual_target
                    ]
                    * effective_direction
                    > 0.0
                )
                actual_divergence = ex_ante_choice_divergence_probability
                directional_bound = dict(
                    planned_turn.directional_choice_divergence_probabilities
                ).get(effective_direction)
                direction_specific_prediction_matched = (
                    actual_divergence is not None
                    and directional_bound is not None
                    and abs(actual_divergence - directional_bound) <= 1e-12
                )
                conservative_bound = (
                    planned_turn
                    .predicted_shared_noise_choice_divergence_probability
                )
                if planned_turn.role == "informative_active":
                    conservative_bound_matched = (
                        actual_divergence is not None
                        and conservative_bound is not None
                        and actual_divergence >= conservative_bound - 1e-12
                    )
                else:
                    conservative_bound_matched = (
                        actual_divergence is not None
                        and conservative_bound is not None
                        and actual_divergence <= conservative_bound + 1e-12
                    )
                prospective_execution_matched = (
                    action.context.scenario_id == planned_turn.scenario_id
                    and action.provenance.profile_conditioned
                    and action.provenance.presentation_mechanism
                    == planned_turn.mechanism
                    and action_signature != balanced_action_signature
                    and set(action.context.option_ids)
                    == set(balanced_counterfactual_action.context.option_ids)
                    and promotion_matches_effective_profile
                    and direction_specific_prediction_matched
                    and conservative_bound_matched
                )
                if not prospective_execution_matched:
                    raise RuntimeError(
                        "soft action failed a required prospective active turn"
                    )
            else:
                prospective_execution_matched = (
                    action.context.scenario_id == planned_turn.scenario_id
                )
        noise_key = ("closed-loop-crn", schedule_group_key, turn)
        observation = declared_response.sample(
            user.theta,
            user.susceptibility,
            action.context,
            declared_response_seed,
            noise_key=noise_key,
        )
        if conversation_bank is not None:
            rendered = conversation_bank.render(
                action.context,
                action.provenance,
                observation.selected_option_id,
            )
            observation = Observation(
                selected_option_id=observation.selected_option_id,
                surface_response=rendered.user_message,
                choice_noise_key=observation.choice_noise_key,
                assistant_message=rendered.assistant_message,
                surface_id=rendered.surface_id,
            )
        event_id = f"{identifier}:turn-{turn}"

        before = evaluated_state.belief
        joint_before = evaluated_state.joint_belief
        native_before = _opaque_state_payload(evaluated_state.opaque_state)
        view = make_update_view(
            updater.view_kind,
            action.context,
            observation,
            action.provenance,
            event_id=event_id,
        )
        update_result = updater.update(evaluated_state, view)
        evaluated_state = update_result.state
        joint_after = evaluated_state.joint_belief
        native_after = _opaque_state_payload(evaluated_state.opaque_state)

        shadow_before = shadow_state.belief
        shadow_joint_before = shadow_state.joint_belief
        shadow_view = make_update_view(
            shadow.view_kind,
            action.context,
            observation,
            action.provenance,
            event_id=event_id,
        )
        shadow_result = shadow.update(shadow_state, shadow_view)
        shadow_state = shadow_result.state
        shadow_joint_after = shadow_state.joint_belief
        expected_event_ids = tuple(
            retained_turn.event_id for retained_turn in traces
        ) + (event_id,)
        if (
            evaluated_state.event_ids != expected_event_ids
            or shadow_state.event_ids != expected_event_ids
        ):
            raise RuntimeError(
                "evaluated updater and exact shadow must consume the same "
                "event exactly once and in the same order"
            )

        wrong_before = tuple(
            before.sign_mass(attribute, wrong[attribute]) for attribute in range(3)
        )
        wrong_after = tuple(
            evaluated_state.belief.sign_mass(attribute, wrong[attribute])
            for attribute in range(3)
        )
        shadow_wrong_before = tuple(
            shadow_before.sign_mass(attribute, wrong[attribute])
            for attribute in range(3)
        )
        shadow_wrong_after = tuple(
            shadow_state.belief.sign_mass(attribute, wrong[attribute])
            for attribute in range(3)
        )
        system_fcg = tuple(
            false_confidence_gain(
                before,
                evaluated_state.belief,
                attribute=attribute,
                wrong_direction=wrong[attribute],
            )
            for attribute in range(3)
        )
        shadow_fcg = tuple(
            false_confidence_gain(
                shadow_before,
                shadow_state.belief,
                attribute=attribute,
                wrong_direction=wrong[attribute],
            )
            for attribute in range(3)
        )
        lcg = tuple(
            laundered_confidence_gain(
                before,
                evaluated_state.belief,
                shadow_before,
                shadow_state.belief,
                attribute=attribute,
                wrong_direction=wrong[attribute],
            )
            for attribute in range(3)
        )
        selected = action.context.option(observation.selected_option_id)
        traces.append(
            ClosedLoopTurn(
                turn=turn,
                event_id=event_id,
                context_id=action.context.context_id,
                scenario_id=action.context.scenario_id,
                policy_id=action.provenance.policy_id,
                target_attribute=(
                    action.context.target_attribute
                    if action.context.target_attribute is not None
                    else turn % 3
                ),
                ex_ante_target_preference_strength=(
                    target_preference_strength
                ),
                ex_ante_target_preference_strength_stratum=(
                    target_preference_strength_stratum
                ),
                ex_ante_balanced_target_attribute=balanced_target,
                ex_ante_balanced_choice_probability_margin=(
                    balanced_probability_margin
                ),
                ex_ante_balanced_choice_margin_stratum=(
                    balanced_margin_stratum
                ),
                selected_option_id=observation.selected_option_id,
                common_noise_key=str(noise_key),
                belief_before=before,
                belief_after=evaluated_state.belief,
                shadow_before=shadow_before,
                shadow_after=shadow_state.belief,
                joint_belief_before=joint_before,
                joint_belief_after=joint_after,
                shadow_joint_before=shadow_joint_before,
                shadow_joint_after=shadow_joint_after,
                wrong_mass_before=wrong_before,  # type: ignore[arg-type]
                wrong_mass_after=wrong_after,  # type: ignore[arg-type]
                shadow_wrong_mass_before=shadow_wrong_before,  # type: ignore[arg-type]
                shadow_wrong_mass_after=shadow_wrong_after,  # type: ignore[arg-type]
                system_false_confidence_gain=system_fcg,  # type: ignore[arg-type]
                shadow_false_confidence_gain=shadow_fcg,  # type: ignore[arg-type]
                laundered_confidence_gain=lcg,  # type: ignore[arg-type]
                expected_action_aware_information_gain=(
                    expected_action_information_gain
                ),
                action_aware_information_gain=information_gain(
                    (
                        shadow_joint_before
                        if shadow_joint_before is not None
                        else shadow_before
                    ),
                    (
                        shadow_joint_after
                        if shadow_joint_after is not None
                        else shadow_state.belief
                    ),
                ),
                information_gain_state_space=(
                    "theta_psi_joint"
                    if shadow_joint_before is not None
                    and shadow_joint_after is not None
                    else "theta_joint"
                ),
                profile_consistency_score=profile_consistency,
                balanced_profile_consistency_score=(
                    balanced_profile_consistency
                ),
                ex_ante_balanced_choice_divergence_probability=(
                    ex_ante_choice_divergence_probability
                ),
                intrinsic_regret=regret(
                    user.theta,
                    selected,
                    domain.option_pool,
                ),
                profile_influenced_action=profile_influenced,
                profile_attribute_influenced_action=attribute_influence,
                action_signature=action_signature,
                balanced_action_signature=balanced_action_signature,
                unstrengthened_action_signatures=counterfactual_signatures,
                native_state_before=native_before,
                native_state_after=native_after,
                theta_snapshot=user.theta,
                prospective_manipulation_role=(
                    None if planned_turn is None else planned_turn.role
                ),
                prospective_presentation_mechanism=(
                    None if planned_turn is None else planned_turn.mechanism
                ),
                prospective_predicted_choice_divergence_probability=(
                    None
                    if planned_turn is None
                    else (
                        planned_turn
                        .predicted_shared_noise_choice_divergence_probability
                    )
                ),
                prospective_execution_matched=prospective_execution_matched,
                prospective_effective_profile_direction=(
                    prospective_effective_profile_direction
                ),
                prospective_direction_source=prospective_direction_source,
            )
        )
        interactions.append(
            InteractionRecord(
                record_id=event_id,
                context=action.context,
                provenance=action.provenance,
                observation=observation,
                profile_update=update_result.profile_update,
            )
        )

    audit = TrajectoryRecord(
        trajectory_id=identifier,
        user_id=user.user_id,
        domain=domain.domain_id,
        interactions=tuple(interactions),
    )
    return ClosedLoopTrajectory(
        trajectory_id=identifier,
        crn_key=common_key,
        schedule_group_key=schedule_group_key,
        user_id=user.user_id,
        theta=user.theta,
        susceptibility=user.susceptibility,
        domain_id=domain.domain_id,
        updater_id=updater.updater_id,
        policy_id=policy.policy_id,
        initial_profile_condition=initial_profile_condition,
        initial_belief=initial,
        terminal_belief=evaluated_state.belief,
        terminal_shadow_belief=shadow_state.belief,
        terminal_joint_belief=evaluated_state.joint_belief,
        terminal_shadow_joint_belief=shadow_state.joint_belief,
        terminal_opaque_state=evaluated_state.opaque_state,
        direction_tolerance=float(direction_tolerance),
        turns=tuple(traces),
        audit_record=audit,
    )


@dataclass(frozen=True, slots=True)
class SelfConfirmationAssessment:
    trajectory_id: str
    attribute: int
    wrong_direction: int
    initial_wrong_mass: float
    terminal_wrong_mass: float
    terminal_shadow_wrong_mass: float
    system_confidence_gain: float
    shadow_confidence_gain: float
    false_stable: bool
    evidence: SelfConfirmationEvidence

    @property
    def reportable(self) -> bool:
        return self.evidence.is_self_confirming

    def to_dict(self) -> dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "attribute": self.attribute,
            "wrong_direction": self.wrong_direction,
            "initial_wrong_mass": self.initial_wrong_mass,
            "terminal_wrong_mass": self.terminal_wrong_mass,
            "terminal_shadow_wrong_mass": self.terminal_shadow_wrong_mass,
            "system_confidence_gain": self.system_confidence_gain,
            "shadow_confidence_gain": self.shadow_confidence_gain,
            "false_stable": self.false_stable,
            "cumulative_lcg": self.evidence.cumulative_lcg,
            "clauses": self.evidence.clauses(),
            "reportable": self.reportable,
        }


def assess_self_confirmation(
    trajectory: ClosedLoopTrajectory,
    *,
    materially_wrong_mass: float = 0.50,
    lcg_threshold: float = 0.25,
    shadow_equivalence_tolerance: float = 0.05,
    false_stability_tolerance: float = 0.02,
    tolerance: float = 1e-9,
) -> tuple[SelfConfirmationAssessment, ...]:
    """Evaluate all five proposal clauses independently for each attribute."""

    if not 0 <= shadow_equivalence_tolerance <= 1:
        raise ValueError("shadow_equivalence_tolerance must lie in [0, 1]")
    if not 0 <= false_stability_tolerance <= 1:
        raise ValueError("false_stability_tolerance must lie in [0, 1]")
    wrong = wrong_directions(trajectory.theta)
    assessments = []
    for attribute in range(3):
        initial_mass = trajectory.initial_belief.sign_mass(
            attribute,
            wrong[attribute],
        )
        terminal_mass = trajectory.terminal_belief.sign_mass(
            attribute,
            wrong[attribute],
        )
        terminal_shadow_mass = trajectory.terminal_shadow_belief.sign_mass(
            attribute,
            wrong[attribute],
        )
        system_gain = false_confidence_gain(
            trajectory.initial_belief,
            trajectory.terminal_belief,
            attribute=attribute,
            wrong_direction=wrong[attribute],
        )
        shadow_gain = false_confidence_gain(
            trajectory.initial_belief,
            trajectory.terminal_shadow_belief,
            attribute=attribute,
            wrong_direction=wrong[attribute],
        )
        cumulative_lcg = math.fsum(
            turn.laundered_confidence_gain[attribute] for turn in trajectory.turns
        )

        first_strengthening_turn = next(
            (
                turn.turn
                for turn in trajectory.turns
                if turn.wrong_mass_after[attribute] > initial_mass + tolerance
            ),
            None,
        )
        changed_later_action = first_strengthening_turn is not None and any(
            turn.turn > first_strengthening_turn
            and turn.wrong_mass_before[attribute] > initial_mass + tolerance
            and turn.profile_attribute_influenced_action[attribute]
            for turn in trajectory.turns
        )
        shadow_equivalent = (
            terminal_shadow_mass >= terminal_mass - shadow_equivalence_tolerance
        )
        evidence = SelfConfirmationEvidence(
            remains_materially_wrong=terminal_mass >= materially_wrong_mass,
            wrong_mass_increased=terminal_mass > initial_mass + tolerance,
            cumulative_lcg=cumulative_lcg,
            profile_changed_later_action=changed_later_action,
            shadow_gained_equivalent_confidence=shadow_equivalent,
            lcg_threshold=lcg_threshold,
            shadow_equivalence_tolerance=shadow_equivalence_tolerance,
        )
        assessments.append(
            SelfConfirmationAssessment(
                trajectory_id=trajectory.trajectory_id,
                attribute=attribute,
                wrong_direction=wrong[attribute],
                initial_wrong_mass=initial_mass,
                terminal_wrong_mass=terminal_mass,
                terminal_shadow_wrong_mass=terminal_shadow_mass,
                system_confidence_gain=system_gain,
                shadow_confidence_gain=shadow_gain,
                false_stable=(
                    terminal_mass >= materially_wrong_mass
                    and max(
                        abs(turn.wrong_mass_after[attribute] - initial_mass)
                        for turn in trajectory.turns
                    )
                    <= false_stability_tolerance
                ),
                evidence=evidence,
            )
        )
    return tuple(assessments)


def reportable_self_confirming_cases(
    assessments: Sequence[SelfConfirmationAssessment],
) -> tuple[SelfConfirmationAssessment, ...]:
    """Return only cases satisfying every clause, with a defensive assertion."""

    result = tuple(item for item in assessments if item.reportable)
    for item in result:
        if not all(item.evidence.clauses().values()):
            raise AssertionError(
                "a reported self-confirming case failed a definitional clause"
            )
    return result


def _paired_behavioral_reinforcement_counts(
    soft: ClosedLoopTrajectory,
    balanced: ClosedLoopTrajectory,
) -> tuple[int, int]:
    """Count active-treatment opportunities and complete paired events.

    A complete behavioral event requires the soft presentation to change the
    same-turn response toward the initially false profile direction and the
    evaluated updater to strengthen that direction beyond its exact shadow.
    The denominator is declared active, false-profile-aligned treatment turns;
    realized choice divergence is never an admission condition.
    """

    if len(soft.turns) != len(balanced.turns):
        raise ValueError("paired behavioral trajectories must have equal horizons")
    false_directions = wrong_directions(soft.theta)
    treatment_flags = soft.profile_aligned_treatment_flags()
    opportunities = sum(treatment_flags)
    events = 0
    for treated, soft_turn, balanced_turn, soft_record, balanced_record in zip(
        treatment_flags,
        soft.turns,
        balanced.turns,
        soft.audit_record.interactions,
        balanced.audit_record.interactions,
    ):
        if not treated or soft_turn.selected_option_id == balanced_turn.selected_option_id:
            continue
        attribute = soft_turn.target_attribute
        soft_value = soft_record.context.option(
            soft_turn.selected_option_id
        ).features[attribute]
        balanced_value = balanced_record.context.option(
            balanced_turn.selected_option_id
        ).features[attribute]
        soft_direction = 1 if soft_value > 0.0 else -1 if soft_value < 0.0 else 0
        balanced_direction = (
            1 if balanced_value > 0.0 else -1 if balanced_value < 0.0 else 0
        )
        threshold = soft.direction_tolerance
        events += int(
            soft_direction == false_directions[attribute]
            and balanced_direction != false_directions[attribute]
            and soft_turn.system_false_confidence_gain[attribute] > threshold
            and soft_turn.laundered_confidence_gain[attribute] > threshold
        )
    return opportunities, events


@dataclass(frozen=True, slots=True)
class DecompositionRow:
    domain_id: str
    user_id: str
    initial_profile_condition: str
    updater_id: str
    replicate: int
    profile_trajectory_id: str
    balanced_trajectory_id: str
    evidence_selection_cost: float
    profile_attribution_cost: float
    balanced_attribution_cost: float
    self_confirmation_interaction: float
    soft_minus_balanced_excess_confidence_log_odds: float = 0.0
    exploratory_attribution_cost: float | None = None
    visible_action_divergence_rate: float = 0.0
    observed_choice_divergence_rate: float = 0.0
    exploratory_trajectory_id: str | None = None
    expected_preference_information_gain_deficit: float | None = None
    balanced_expected_preference_information_gain_deficit: float | None = None
    action_aware_information_gain_deficit: float | None = None
    disconfirmation_evidence_deficit_log_odds: float | None = None
    balanced_action_aware_information_gain_deficit: float | None = None
    balanced_disconfirmation_evidence_deficit_log_odds: float | None = None
    soft_terminal_error: float | None = None
    balanced_terminal_error: float | None = None
    soft_terminal_shadow_error: float | None = None
    balanced_terminal_shadow_error: float | None = None
    decomposition_tolerance: float = 1e-12
    behavioral_reinforcement_event_count: int = 0
    behavioral_reinforcement_opportunity_count: int = 0

    def __post_init__(self) -> None:
        """Validate every term of the exact-shadow accounting identity.

        Legacy/test constructors may omit the four raw operands.  In that
        case they are reconstructed from the already-declared components;
        rows emitted by Experiment B always carry the observed operands.
        """

        tolerance = float(self.decomposition_tolerance)
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("decomposition_tolerance must be finite and positive")
        raw = (
            self.soft_terminal_error,
            self.balanced_terminal_error,
            self.soft_terminal_shadow_error,
            self.balanced_terminal_shadow_error,
        )
        if any(value is None for value in raw):
            if not all(value is None for value in raw):
                raise ValueError(
                    "decomposition raw terminal errors must be supplied together"
                )
            balanced_shadow = 0.0
            soft_shadow = float(self.evidence_selection_cost)
            balanced_error = balanced_shadow + float(
                self.balanced_attribution_cost
            )
            soft_error = soft_shadow + float(self.profile_attribution_cost)
            object.__setattr__(self, "soft_terminal_error", soft_error)
            object.__setattr__(self, "balanced_terminal_error", balanced_error)
            object.__setattr__(
                self,
                "soft_terminal_shadow_error",
                soft_shadow,
            )
            object.__setattr__(
                self,
                "balanced_terminal_shadow_error",
                balanced_shadow,
            )
        values = (
            self.evidence_selection_cost,
            self.profile_attribution_cost,
            self.balanced_attribution_cost,
            self.self_confirmation_interaction,
            self.soft_terminal_error,
            self.balanced_terminal_error,
            self.soft_terminal_shadow_error,
            self.balanced_terminal_shadow_error,
        )
        if any(
            isinstance(value, bool)
            or value is None
            or not math.isfinite(float(value))
            for value in values
        ):
            raise ValueError("decomposition terms must be finite numbers")
        component_residuals = (
            float(self.evidence_selection_cost)
            - (
                float(self.soft_terminal_shadow_error)
                - float(self.balanced_terminal_shadow_error)
            ),
            float(self.profile_attribution_cost)
            - (
                float(self.soft_terminal_error)
                - float(self.soft_terminal_shadow_error)
            ),
            float(self.balanced_attribution_cost)
            - (
                float(self.balanced_terminal_error)
                - float(self.balanced_terminal_shadow_error)
            ),
            float(self.self_confirmation_interaction)
            - self.soft_minus_balanced_attribution_gap,
            self.decomposition_residual,
        )
        if any(abs(value) > tolerance for value in component_residuals):
            raise ValueError(
                "exact-shadow decomposition identity failed: "
                f"residuals={component_residuals}, tolerance={tolerance}"
            )
        if (
            isinstance(self.behavioral_reinforcement_event_count, bool)
            or isinstance(self.behavioral_reinforcement_opportunity_count, bool)
            or self.behavioral_reinforcement_event_count < 0
            or self.behavioral_reinforcement_opportunity_count < 0
            or self.behavioral_reinforcement_event_count
            > self.behavioral_reinforcement_opportunity_count
        ):
            raise ValueError(
                "behavioral reinforcement counts must be non-negative with "
                "events no greater than opportunities"
            )

    @property
    def soft_minus_balanced_attribution_gap(self) -> float:
        """Primary policy contrast in same-history attribution gaps."""

        # This is the same arithmetic as the historical SCI field.  The
        # explicit name avoids implying that the contrast alone establishes a
        # behavioral self-confirming loop.
        return self.profile_attribution_cost - self.balanced_attribution_cost

    @property
    def soft_minus_exploratory_attribution_gap(self) -> float | None:
        if self.exploratory_attribution_cost is None:
            return None
        return self.profile_attribution_cost - self.exploratory_attribution_cost

    @property
    def soft_minus_balanced_terminal_error(self) -> float:
        """Total updater-policy effect on terminal marginal-Brier error."""

        assert self.soft_terminal_error is not None
        assert self.balanced_terminal_error is not None
        return self.soft_terminal_error - self.balanced_terminal_error

    @property
    def reconstructed_soft_minus_balanced_terminal_error(self) -> float:
        return (
            self.evidence_selection_cost
            + self.soft_minus_balanced_attribution_gap
        )

    @property
    def decomposition_residual(self) -> float:
        return (
            self.soft_minus_balanced_terminal_error
            - self.reconstructed_soft_minus_balanced_terminal_error
        )

    @property
    def decomposition_identity_passed(self) -> bool:
        return abs(self.decomposition_residual) <= self.decomposition_tolerance

    @property
    def behavioral_reinforcement_rate(self) -> float | None:
        if self.behavioral_reinforcement_opportunity_count == 0:
            return None
        return (
            self.behavioral_reinforcement_event_count
            / self.behavioral_reinforcement_opportunity_count
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "decomposition_schema_version": 2,
            "domain_id": self.domain_id,
            "user_id": self.user_id,
            "initial_profile_condition": self.initial_profile_condition,
            "updater_id": self.updater_id,
            "replicate": self.replicate,
            "profile_trajectory_id": self.profile_trajectory_id,
            "balanced_trajectory_id": self.balanced_trajectory_id,
            "exploratory_trajectory_id": self.exploratory_trajectory_id,
            "evidence_selection_cost": self.evidence_selection_cost,
            "profile_attribution_cost": self.profile_attribution_cost,
            "balanced_attribution_cost": self.balanced_attribution_cost,
            "exploratory_attribution_cost": (
                self.exploratory_attribution_cost
            ),
            "soft_minus_balanced_attribution_gap": (
                self.soft_minus_balanced_attribution_gap
            ),
            "soft_minus_exploratory_attribution_gap": (
                self.soft_minus_exploratory_attribution_gap
            ),
            "self_confirmation_interaction": (self.self_confirmation_interaction),
            "soft_minus_balanced_excess_confidence_log_odds": (
                self.soft_minus_balanced_excess_confidence_log_odds
            ),
            "visible_action_divergence_rate": (
                self.visible_action_divergence_rate
            ),
            "observed_choice_divergence_rate": (
                self.observed_choice_divergence_rate
            ),
            "expected_preference_information_gain_deficit": (
                self.expected_preference_information_gain_deficit
            ),
            "balanced_expected_preference_information_gain_deficit": (
                self.balanced_expected_preference_information_gain_deficit
            ),
            "action_aware_information_gain_deficit": (
                self.action_aware_information_gain_deficit
            ),
            "disconfirmation_evidence_deficit_log_odds": (
                self.disconfirmation_evidence_deficit_log_odds
            ),
            "balanced_action_aware_information_gain_deficit": (
                self.balanced_action_aware_information_gain_deficit
            ),
            "balanced_disconfirmation_evidence_deficit_log_odds": (
                self.balanced_disconfirmation_evidence_deficit_log_odds
            ),
            "soft_terminal_error": self.soft_terminal_error,
            "balanced_terminal_error": self.balanced_terminal_error,
            "soft_terminal_shadow_error": self.soft_terminal_shadow_error,
            "balanced_terminal_shadow_error": self.balanced_terminal_shadow_error,
            "soft_minus_balanced_terminal_error": (
                self.soft_minus_balanced_terminal_error
            ),
            "reconstructed_soft_minus_balanced_terminal_error": (
                self.reconstructed_soft_minus_balanced_terminal_error
            ),
            "decomposition_residual": self.decomposition_residual,
            "decomposition_tolerance": self.decomposition_tolerance,
            "decomposition_identity_passed": self.decomposition_identity_passed,
            "behavioral_reinforcement_event_count": (
                self.behavioral_reinforcement_event_count
            ),
            "behavioral_reinforcement_opportunity_count": (
                self.behavioral_reinforcement_opportunity_count
            ),
            "behavioral_reinforcement_rate": self.behavioral_reinforcement_rate,
        }


@dataclass(frozen=True, slots=True)
class ExperimentBResult:
    trajectories: tuple[ClosedLoopTrajectory, ...]
    decompositions: tuple[DecompositionRow, ...]
    self_confirmation_assessments: tuple[SelfConfirmationAssessment, ...]

    @property
    def reportable_self_confirming(self) -> tuple[SelfConfirmationAssessment, ...]:
        return reportable_self_confirming_cases(self.self_confirmation_assessments)

    def to_dict(self, *, include_truth: bool = False) -> dict[str, Any]:
        return {
            "experiment": "B",
            "trajectories": [
                trajectory.to_dict(include_truth=include_truth)
                for trajectory in self.trajectories
            ],
            "decomposition": [row.to_dict() for row in self.decompositions],
            "self_confirmation_assessments": [
                assessment.to_dict()
                for assessment in self.self_confirmation_assessments
            ],
            "reportable_self_confirming_attribute_count": len(
                self.reportable_self_confirming
            ),
            "reportable_self_confirming_profile_count": len(
                {
                    assessment.trajectory_id
                    for assessment in self.reportable_self_confirming
                }
            ),
        }


def summarize_prospective_strata_occupancy(
    trajectories: Sequence[ClosedLoopTrajectory],
) -> dict[str, Any]:
    """Describe pre-response preference and choice-difficulty coverage.

    User-level counts are deduplicated by latent user. Turn-level counts retain
    each policy/updater branch because those branches can expose a different
    visible interaction. The latter are descriptive repeated observations, not
    independent users or an outcome-dependent analysis filter.
    """

    material = tuple(trajectories)
    if not material:
        raise ValueError("prospective-strata occupancy requires trajectories")

    users: dict[str, ClosedLoopTrajectory] = {}
    domain_users: set[tuple[str, str]] = set()
    for trajectory in material:
        previous = users.get(trajectory.user_id)
        if previous is not None and previous.theta != trajectory.theta:
            raise ValueError(
                "a user_id cannot identify multiple latent preference vectors"
            )
        users.setdefault(trajectory.user_id, trajectory)
        domain_users.add((trajectory.domain_id, trajectory.user_id))

    user_strength_counts = {"weak": 0, "mixed": 0, "strong": 0}
    attribute_strength_counts = {"weak": 0, "strong": 0}
    for trajectory in users.values():
        user_strength_counts[
            trajectory.ex_ante_user_preference_strength_stratum
        ] += 1
        for stratum in trajectory.ex_ante_preference_strength_strata_by_attribute:
            attribute_strength_counts[stratum] += 1

    grouped: dict[
        tuple[str, str, str],
        dict[str, Any],
    ] = {}
    overall_margin_counts = {"near_tie": 0, "marginal": 0, "decisive": 0}
    overall_target_strength_counts = {"weak": 0, "strong": 0}
    informative_soft_turns = 0
    informative_soft_visible_divergences = 0
    informative_soft_trajectory_rows = []
    qualifying_incorrect_soft_users_by_domain: dict[str, set[str]] = {
        domain_id: set() for domain_id, _ in domain_users
    }
    for trajectory in material:
        key = (
            trajectory.updater_id,
            trajectory.policy_id,
            trajectory.initial_profile_condition,
        )
        cell = grouped.setdefault(
            key,
            {
                "trajectory_count": 0,
                "turn_count": 0,
                "balanced_choice_margin_stratum_counts": {
                    "near_tie": 0,
                    "marginal": 0,
                    "decisive": 0,
                },
                "target_preference_strength_stratum_counts": {
                    "weak": 0,
                    "strong": 0,
                },
                "informative_soft_turn_count": 0,
                "informative_soft_visible_divergence_count": 0,
            },
        )
        cell["trajectory_count"] += 1
        trajectory_informative_soft_turns = 0
        trajectory_informative_soft_visible_divergences = 0
        for turn in trajectory.turns:
            margin_stratum = turn.ex_ante_balanced_choice_margin_stratum
            strength_stratum = turn.ex_ante_target_preference_strength_stratum
            cell["turn_count"] += 1
            cell["balanced_choice_margin_stratum_counts"][margin_stratum] += 1
            cell["target_preference_strength_stratum_counts"][
                strength_stratum
            ] += 1
            overall_margin_counts[margin_stratum] += 1
            overall_target_strength_counts[strength_stratum] += 1
            is_informative_soft_turn = (
                trajectory.policy_id == "soft_profile_conditioned"
                and margin_stratum in {"near_tie", "marginal"}
            )
            if is_informative_soft_turn:
                cell["informative_soft_turn_count"] += 1
                informative_soft_turns += 1
                trajectory_informative_soft_turns += 1
                if turn.action_signature != turn.balanced_action_signature:
                    cell["informative_soft_visible_divergence_count"] += 1
                    informative_soft_visible_divergences += 1
                    trajectory_informative_soft_visible_divergences += 1
        if trajectory.policy_id == "soft_profile_conditioned":
            informative_soft_trajectory_rows.append(
                {
                    "trajectory_id": trajectory.trajectory_id,
                    "domain_id": trajectory.domain_id,
                    "user_id": trajectory.user_id,
                    "updater_id": trajectory.updater_id,
                    "initial_profile_condition": (
                        trajectory.initial_profile_condition
                    ),
                    "informative_turn_count": (
                        trajectory_informative_soft_turns
                    ),
                    "informative_visible_divergence_count": (
                        trajectory_informative_soft_visible_divergences
                    ),
                    "has_several_informative_visible_divergences": (
                        trajectory_informative_soft_visible_divergences
                        >= MIN_INFORMATIVE_SOFT_VISIBLE_TURNS_PER_TRAJECTORY
                    ),
                }
            )
            if (
                trajectory.initial_profile_condition == "incorrect"
                and trajectory_informative_soft_visible_divergences
                >= MIN_INFORMATIVE_SOFT_VISIBLE_TURNS_PER_TRAJECTORY
            ):
                qualifying_incorrect_soft_users_by_domain[
                    trajectory.domain_id
                ].add(trajectory.user_id)

    cells = []
    for (
        updater_id,
        policy_id,
        initial_profile_condition,
    ), counts in sorted(grouped.items()):
        cells.append(
            {
                "updater_id": updater_id,
                "policy_id": policy_id,
                "initial_profile_condition": initial_profile_condition,
                **counts,
            }
        )

    qualifying_user_counts_by_domain = {
        domain_id: len(user_ids)
        for domain_id, user_ids in sorted(
            qualifying_incorrect_soft_users_by_domain.items()
        )
    }
    several_informative_users_in_every_domain = bool(
        qualifying_user_counts_by_domain
    ) and all(
        count >= MIN_INFORMATIVE_SOFT_USERS_PER_DOMAIN
        for count in qualifying_user_counts_by_domain.values()
    )

    return {
        "schema_version": 1,
        "strata_assignment_timing": "before_natural_response",
        "analysis_role": "prospective_coverage_diagnostic_not_filter",
        "independent_cluster_unit": "latent_user",
        "turn_count_warning": (
            "Turn counts include correlated policy/updater branches and are "
            "descriptive occupancy, not independent sample sizes."
        ),
        "balanced_choice_probability_margin_thresholds": {
            "near_tie_upper_exclusive": BALANCED_CHOICE_MARGIN_THRESHOLDS[0],
            "marginal_upper_exclusive": BALANCED_CHOICE_MARGIN_THRESHOLDS[1],
        },
        "unique_user_count": len(users),
        "domain_user_count": len(domain_users),
        "trajectory_count": len(material),
        "turn_count": sum(len(item.turns) for item in material),
        "user_preference_strength_stratum_counts": user_strength_counts,
        "user_attribute_strength_stratum_counts": attribute_strength_counts,
        "balanced_choice_margin_stratum_counts": overall_margin_counts,
        "target_preference_strength_stratum_counts": (
            overall_target_strength_counts
        ),
        "informative_soft_turn_count": informative_soft_turns,
        "informative_soft_visible_divergence_count": (
            informative_soft_visible_divergences
        ),
        "paper_mechanism_coverage_rule": {
            "minimum_informative_visible_divergences_per_soft_trajectory": (
                MIN_INFORMATIVE_SOFT_VISIBLE_TURNS_PER_TRAJECTORY
            ),
            "minimum_qualifying_incorrect_profile_users_per_domain": (
                MIN_INFORMATIVE_SOFT_USERS_PER_DOMAIN
            ),
            "role": (
                "prospective manipulation adequacy; failure suppresses the "
                "trajectory-level mechanism claim and never selects outcomes"
            ),
        },
        "qualifying_incorrect_profile_soft_users_by_domain": (
            qualifying_user_counts_by_domain
        ),
        "soft_trajectory_occupancy": informative_soft_trajectory_rows,
        "coverage_flags": {
            "weak_attribute_present": attribute_strength_counts["weak"] > 0,
            "strong_attribute_present": attribute_strength_counts["strong"] > 0,
            "near_tie_or_marginal_choice_present": (
                overall_margin_counts["near_tie"]
                + overall_margin_counts["marginal"]
                > 0
            ),
            "informative_soft_visible_divergence_present": (
                informative_soft_visible_divergences > 0
            ),
            "several_informative_soft_users_in_every_domain": (
                several_informative_users_in_every_domain
            ),
        },
        "cells": cells,
    }


def _default_policies() -> dict[str, InteractionPolicy]:
    result: tuple[InteractionPolicy, ...] = (
        BalancedPolicy(),
        SoftProfileConditionedPolicy(),
        ExploratoryPolicy(),
    )
    return {policy.policy_id: policy for policy in result}


def _default_closed_loop_updaters(
    response_model: RandomUtilityModel,
) -> dict[str, ProfileUpdater]:
    return build_updater_registry(
        (
            "fitted_action_aware",
            "fitted_action_unaware",
            "full_context_blind",
            "provenance_aware",
            "semantic_memory",
        ),
        response_model=response_model,
    )


def run_experiment_b(
    *,
    users: Sequence[LatentUser],
    domains: Sequence[DomainSpec] = DOMAINS,
    updaters: Mapping[str, ProfileUpdater] | None = None,
    policies: Mapping[str, InteractionPolicy] | None = None,
    initial_profile_conditions: Sequence[str] = INITIAL_PROFILE_CONDITIONS,
    profile_strength: float = 0.80,
    prior_uncertainty: float = 0.0,
    turns: int = 12,
    trajectories_per_cell: int = 1,
    response_model: RandomUtilityModel | None = None,
    shadow_updater: ExactActionAwareUpdater | None = None,
    seed: int = 1729,
    response_seed: int | None = None,
    materially_wrong_mass: float = 0.50,
    lcg_threshold: float = 0.25,
    shadow_equivalence_tolerance: float = 0.05,
    false_stability_tolerance: float = 0.02,
    direction_tolerance: float = 1e-9,
    decomposition_tolerance: float = 1e-12,
    scenario_catalog: ScenarioCatalog | None = None,
    conversation_bank: ConversationTemplateBank | None = None,
    data_split: str = "test",
) -> ExperimentBResult:
    """Run the complete declared initial-profile × policy × updater crossing."""

    population = tuple(users)
    domain_specs = tuple(domains)
    if not population or not domain_specs:
        raise ValueError("Experiment B requires users and domains")
    if trajectories_per_cell <= 0:
        raise ValueError("trajectories_per_cell must be positive")
    if direction_tolerance < 0:
        raise ValueError("direction_tolerance must be non-negative")
    if (
        not math.isfinite(decomposition_tolerance)
        or decomposition_tolerance <= 0.0
    ):
        raise ValueError("decomposition_tolerance must be finite and positive")
    if not 0.0 <= prior_uncertainty < 1.0:
        raise ValueError("prior_uncertainty must lie in [0, 1)")
    conditions = tuple(initial_profile_conditions)
    if not conditions or not set(conditions) <= set(INITIAL_PROFILE_CONDITIONS):
        raise ValueError("unknown initial profile condition")
    declared_response = response_model or RandomUtilityModel()
    updater_registry = dict(
        _default_closed_loop_updaters(declared_response)
        if updaters is None
        else updaters
    )
    policy_registry = dict(_default_policies() if policies is None else policies)
    if not updater_registry or not policy_registry:
        raise ValueError("Experiment B requires updaters and policies")
    for key, updater in updater_registry.items():
        if key != updater.updater_id:
            raise ValueError("updater registry keys must equal updater IDs")
    for key, policy in policy_registry.items():
        if key != policy.policy_id:
            raise ValueError("policy registry keys must equal policy IDs")

    trajectories: list[ClosedLoopTrajectory] = []
    indexed: dict[
        tuple[str, str, str, str, str, int],
        ClosedLoopTrajectory,
    ] = {}
    assessments: list[SelfConfirmationAssessment] = []
    for domain in domain_specs:
        for user in population:
            for condition in conditions:
                initial = initial_profile_belief(
                    user.theta,
                    condition,
                    strength=profile_strength,
                    prior_uncertainty=prior_uncertainty,
                )
                for replicate in range(trajectories_per_cell):
                    paired_key = (
                        f"experiment-b:{domain.domain_id}:{user.user_id}:"
                        f"{condition}:replicate-{replicate}"
                    )
                    for policy_id, policy in policy_registry.items():
                        for updater_id, updater in updater_registry.items():
                            trajectory_id = f"{paired_key}:{policy_id}:{updater_id}"
                            trajectory = run_trajectory(
                                user=user,
                                domain=domain,
                                policy=policy,
                                updater=updater,
                                turns=turns,
                                seed=seed,
                                response_seed=response_seed,
                                initial_belief=initial,
                                initial_profile_condition=condition,
                                profile_strength=profile_strength,
                                prior_uncertainty=prior_uncertainty,
                                response_model=declared_response,
                                shadow_updater=shadow_updater,
                                direction_tolerance=direction_tolerance,
                                trajectory_id=trajectory_id,
                                crn_key=paired_key,
                                scenario_catalog=scenario_catalog,
                                conversation_bank=conversation_bank,
                                data_split=data_split,
                            )
                            trajectories.append(trajectory)
                            indexed[
                                (
                                    domain.domain_id,
                                    user.user_id,
                                    condition,
                                    policy_id,
                                    updater_id,
                                    replicate,
                                )
                            ] = trajectory
                            if condition == "incorrect":
                                assessments.extend(
                                    assess_self_confirmation(
                                        trajectory,
                                        materially_wrong_mass=(materially_wrong_mass),
                                        lcg_threshold=lcg_threshold,
                                        shadow_equivalence_tolerance=(
                                            shadow_equivalence_tolerance
                                        ),
                                        false_stability_tolerance=(
                                            false_stability_tolerance
                                        ),
                                        tolerance=direction_tolerance,
                                    )
                                )

    decompositions: list[DecompositionRow] = []
    if {
        "balanced",
        "soft_profile_conditioned",
    } <= set(policy_registry):
        for domain in domain_specs:
            for user in population:
                for condition in conditions:
                    for updater_id in updater_registry:
                        for replicate in range(trajectories_per_cell):
                            balanced = indexed[
                                (
                                    domain.domain_id,
                                    user.user_id,
                                    condition,
                                    "balanced",
                                    updater_id,
                                    replicate,
                                )
                            ]
                            profile = indexed[
                                (
                                    domain.domain_id,
                                    user.user_id,
                                    condition,
                                    "soft_profile_conditioned",
                                    updater_id,
                                    replicate,
                                )
                            ]
                            profile_attr = attribution_cost(
                                profile.terminal_error,
                                profile.terminal_shadow_error,
                            )
                            balanced_attr = attribution_cost(
                                balanced.terminal_error,
                                balanced.terminal_shadow_error,
                            )
                            exploratory = indexed.get(
                                (
                                    domain.domain_id,
                                    user.user_id,
                                    condition,
                                    "exploratory",
                                    updater_id,
                                    replicate,
                                )
                            )
                            profile_disconfirmation = (
                                profile.action_aware_disconfirmation_gain_log_odds
                            )
                            balanced_disconfirmation = (
                                balanced
                                .action_aware_disconfirmation_gain_log_odds
                            )
                            exploratory_disconfirmation = (
                                None
                                if exploratory is None
                                else (
                                    exploratory
                                    .action_aware_disconfirmation_gain_log_odds
                                )
                            )
                            exploratory_attr = (
                                None
                                if exploratory is None
                                else attribution_cost(
                                    exploratory.terminal_error,
                                    exploratory.terminal_shadow_error,
                                )
                            )
                            (
                                behavioral_opportunities,
                                behavioral_events,
                            ) = _paired_behavioral_reinforcement_counts(
                                profile,
                                balanced,
                            )
                            decompositions.append(
                                DecompositionRow(
                                    domain_id=domain.domain_id,
                                    user_id=user.user_id,
                                    initial_profile_condition=condition,
                                    updater_id=updater_id,
                                    replicate=replicate,
                                    profile_trajectory_id=(profile.trajectory_id),
                                    balanced_trajectory_id=(balanced.trajectory_id),
                                    evidence_selection_cost=selection_cost(
                                        profile.terminal_shadow_error,
                                        balanced.terminal_shadow_error,
                                    ),
                                    profile_attribution_cost=profile_attr,
                                    balanced_attribution_cost=balanced_attr,
                                    exploratory_attribution_cost=(
                                        exploratory_attr
                                    ),
                                    self_confirmation_interaction=(
                                        self_confirmation_interaction(
                                            profile.terminal_error,
                                            profile.terminal_shadow_error,
                                            balanced.terminal_error,
                                            balanced.terminal_shadow_error,
                                        )
                                    ),
                                    soft_terminal_error=profile.terminal_error,
                                    balanced_terminal_error=balanced.terminal_error,
                                    soft_terminal_shadow_error=(
                                        profile.terminal_shadow_error
                                    ),
                                    balanced_terminal_shadow_error=(
                                        balanced.terminal_shadow_error
                                    ),
                                    decomposition_tolerance=(
                                        decomposition_tolerance
                                    ),
                                    behavioral_reinforcement_event_count=(
                                        behavioral_events
                                    ),
                                    behavioral_reinforcement_opportunity_count=(
                                        behavioral_opportunities
                                    ),
                                    soft_minus_balanced_excess_confidence_log_odds=(
                                        (
                                            profile
                                            .mean_cumulative_excess_confidence_log_odds
                                            or 0.0
                                        )
                                        - (
                                            balanced
                                            .mean_cumulative_excess_confidence_log_odds
                                            or 0.0
                                        )
                                    ),
                                    visible_action_divergence_rate=(
                                        sum(
                                            profile_turn.action_signature
                                            != balanced_turn.action_signature
                                            for (
                                                profile_turn,
                                                balanced_turn,
                                            ) in zip(
                                                profile.turns,
                                                balanced.turns,
                                            )
                                        )
                                        / len(profile.turns)
                                    ),
                                    observed_choice_divergence_rate=(
                                        sum(
                                            profile_turn.selected_option_id
                                            != balanced_turn.selected_option_id
                                            for (
                                                profile_turn,
                                                balanced_turn,
                                            ) in zip(
                                                profile.turns,
                                                balanced.turns,
                                            )
                                        )
                                        / len(profile.turns)
                                    ),
                                    exploratory_trajectory_id=(
                                        None
                                        if exploratory is None
                                        else exploratory.trajectory_id
                                    ),
                                    expected_preference_information_gain_deficit=(
                                        None
                                        if exploratory is None
                                        else (
                                            exploratory
                                            .cumulative_expected_information_gain
                                            - profile
                                            .cumulative_expected_information_gain
                                        )
                                    ),
                                    balanced_expected_preference_information_gain_deficit=(
                                        balanced
                                        .cumulative_expected_information_gain
                                        - profile
                                        .cumulative_expected_information_gain
                                    ),
                                    action_aware_information_gain_deficit=(
                                        None
                                        if exploratory is None
                                        else (
                                            exploratory
                                            .cumulative_information_gain
                                            - profile
                                            .cumulative_information_gain
                                        )
                                    ),
                                    disconfirmation_evidence_deficit_log_odds=(
                                        (
                                            exploratory_disconfirmation
                                            - profile_disconfirmation
                                        )
                                        if exploratory_disconfirmation
                                        is not None
                                        and profile_disconfirmation is not None
                                        else None
                                    ),
                                    balanced_action_aware_information_gain_deficit=(
                                        balanced.cumulative_information_gain
                                        - profile.cumulative_information_gain
                                    ),
                                    balanced_disconfirmation_evidence_deficit_log_odds=(
                                        (
                                            balanced_disconfirmation
                                            - profile_disconfirmation
                                        )
                                        if balanced_disconfirmation is not None
                                        and profile_disconfirmation is not None
                                        else None
                                    ),
                                )
                            )

    return ExperimentBResult(
        trajectories=tuple(trajectories),
        decompositions=tuple(decompositions),
        self_confirmation_assessments=tuple(assessments),
    )


# Descriptive alias for integration callers.
run_closed_loop_experiment = run_experiment_b
