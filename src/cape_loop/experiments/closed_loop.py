"""Experiment B closed-loop trajectories and causal error decomposition."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from ..beliefs import (
    JointThetaPsiBelief,
    MarginalPreferenceBelief,
    PreferenceBelief,
)
from ..domains import DOMAINS, DomainSpec
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
    SoftProfileConditionedPolicy,
)
from ..population import (
    INITIAL_PROFILE_KINDS,
    add_prior_uncertainty,
    initial_profile_belief as _population_initial_profile_belief,
)
from ..response import RandomUtilityModel, regret
from ..schemas import (
    InteractionRecord,
    LatentUser,
    PolicyProvenance,
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
    policy_id: str
    target_attribute: int
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
    action_aware_information_gain: float
    information_gain_state_space: str
    intrinsic_regret: float
    profile_influenced_action: bool
    profile_attribute_influenced_action: tuple[bool, bool, bool]
    action_signature: tuple[object, ...]
    unstrengthened_action_signatures: tuple[
        tuple[object, ...],
        tuple[object, ...],
        tuple[object, ...],
    ]
    native_state_before: Mapping[str, Any] | None
    native_state_after: Mapping[str, Any] | None
    theta_snapshot: Theta

    def to_dict(self, *, include_truth: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "turn": self.turn,
            "event_id": self.event_id,
            "context_id": self.context_id,
            "policy_id": self.policy_id,
            "target_attribute": self.target_attribute,
            "selected_option_id": self.selected_option_id,
            "common_noise_key": self.common_noise_key,
            "wrong_mass_before": list(self.wrong_mass_before),
            "wrong_mass_after": list(self.wrong_mass_after),
            "shadow_wrong_mass_before": list(self.shadow_wrong_mass_before),
            "shadow_wrong_mass_after": list(self.shadow_wrong_mass_after),
            "system_false_confidence_gain": list(
                self.system_false_confidence_gain
            ),
            "shadow_false_confidence_gain": list(
                self.shadow_false_confidence_gain
            ),
            "laundered_confidence_gain": list(
                self.laundered_confidence_gain
            ),
            "action_aware_information_gain": self.action_aware_information_gain,
            "information_gain_state_space": self.information_gain_state_space,
            "intrinsic_regret": self.intrinsic_regret,
            "profile_influenced_action": self.profile_influenced_action,
            "profile_attribute_influenced_action": list(
                self.profile_attribute_influenced_action
            ),
            "action_signature": self.action_signature,
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
        }
        if include_truth:
            result["theta_snapshot"] = list(self.theta_snapshot)
        return result


@dataclass(frozen=True, slots=True)
class ClosedLoopTrajectory:
    trajectory_id: str
    crn_key: str
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

    @property
    def terminal_error(self) -> float:
        return marginal_brier(self.terminal_belief, self.theta)

    @property
    def terminal_shadow_error(self) -> float:
        return marginal_brier(self.terminal_shadow_belief, self.theta)

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
            (count / total) * math.log(count / total)
            for count in counts.values()
        )
        return entropy / math.log(len(counts))

    @property
    def displayed_option_diversity(self) -> float:
        """Fraction of the domain's six isolated options ever displayed."""

        displayed = {
            option.option_id
            for interaction in self.audit_record.interactions
            for option in interaction.context.options
        }
        return len(displayed) / 6.0

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
        return math.fsum(
            turn.action_aware_information_gain for turn in self.turns
        )

    @property
    def total_regret(self) -> float:
        return math.fsum(turn.intrinsic_regret for turn in self.turns)

    @property
    def same_history_shadow(self) -> bool:
        """The shadow consumes each actual event exactly once."""

        return (
            tuple(turn.event_id for turn in self.turns)
            == tuple(interaction.record_id for interaction in self.audit_record.interactions)
            and len(self.turns) == len(self.audit_record.interactions)
        )

    def to_dict(self, *, include_truth: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "trajectory_id": self.trajectory_id,
            "crn_key": self.crn_key,
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
            "terminal_native_state": _opaque_state_payload(
                self.terminal_opaque_state
            ),
            "terminal_error": self.terminal_error,
            "terminal_shadow_error": self.terminal_shadow_error,
            "terminal_shadow_to_system_marginal_kl": (
                self.terminal_shadow_to_system_marginal_kl
            ),
            "preference_dimension_coverage": (
                self.preference_dimension_coverage
            ),
            "turns_to_full_preference_coverage": (
                self.turns_to_full_preference_coverage
            ),
            "displayed_option_diversity": self.displayed_option_diversity,
            "selected_option_count": self.selected_option_count,
            "profile_conditioned_exposure_rate": (
                self.profile_conditioned_exposure_rate
            ),
            "presentation_mechanism_count": (
                self.presentation_mechanism_count
            ),
            "presentation_mechanism_evenness": (
                self.presentation_mechanism_evenness
            ),
            "cumulative_lcg": list(self.cumulative_lcg),
            "cumulative_information_gain": self.cumulative_information_gain,
            "total_regret": self.total_regret,
            "same_history_shadow": self.same_history_shadow,
            "turns": [
                turn.to_dict(include_truth=include_truth) for turn in self.turns
            ],
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
    initial_belief: PreferenceBelief | None = None,
    initial_profile_condition: str = "empty",
    profile_strength: float = 0.80,
    prior_uncertainty: float = 0.0,
    response_model: RandomUtilityModel | None = None,
    shadow_updater: ExactActionAwareUpdater | None = None,
    trajectory_id: str | None = None,
    crn_key: str | None = None,
) -> ClosedLoopTrajectory:
    """Run an endogenous loop with an exact aware same-history shadow.

    The user object is read but never mutated.  Choice Gumbels are keyed by the
    caller-controlled ``crn_key`` and turn, so overlapping options are paired
    across counterfactual policy/updater branches.
    """

    if turns <= 0:
        raise ValueError("turns must be positive")
    if domain.domain_id not in {option.domain for option in domain.option_pool}:
        raise ValueError("domain option pool is internally inconsistent")
    declared_response = response_model or RandomUtilityModel()
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

    for turn in range(turns):
        # The actual action uses only the updater's current public profile.
        action = policy.action(
            domain,
            evaluated_state.belief,
            turn=turn,
            master_seed=seed,
            trajectory_id=common_key,
        )
        # Evaluator-only, per-attribute counterfactuals remove the accumulated
        # update to one profile dimension while preserving all other
        # policy-visible marginals and semantic randomness. This measures
        # whether strengthening that attribute—not merely seeding it wrong—
        # changed the subsequent action.
        counterfactual_actions = tuple(
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
            )
            for attribute in range(3)
        )
        action_signature = action.signature()
        counterfactual_signatures = tuple(
            candidate.signature() for candidate in counterfactual_actions
        )
        attribute_influence = tuple(
            action_signature != signature
            for signature in counterfactual_signatures
        )
        profile_influenced = any(attribute_influence)
        noise_key = ("closed-loop-crn", common_key, turn)
        observation = declared_response.sample(
            user.theta,
            user.susceptibility,
            action.context,
            seed,
            noise_key=noise_key,
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

        wrong_before = tuple(
            before.sign_mass(attribute, wrong[attribute])
            for attribute in range(3)
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
                policy_id=action.provenance.policy_id,
                target_attribute=(
                    action.context.target_attribute
                    if action.context.target_attribute is not None
                    else turn % 3
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
                intrinsic_regret=regret(
                    user.theta,
                    selected,
                    domain.option_pool,
                ),
                profile_influenced_action=profile_influenced,
                profile_attribute_influenced_action=attribute_influence,
                action_signature=action_signature,
                unstrengthened_action_signatures=counterfactual_signatures,
                native_state_before=native_before,
                native_state_after=native_after,
                theta_snapshot=user.theta,
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
            turn.laundered_confidence_gain[attribute]
            for turn in trajectory.turns
        )

        first_strengthening_turn = next(
            (
                turn.turn
                for turn in trajectory.turns
                if turn.wrong_mass_after[attribute]
                > initial_mass + tolerance
            ),
            None,
        )
        changed_later_action = (
            first_strengthening_turn is not None
            and any(
                turn.turn > first_strengthening_turn
                and turn.wrong_mass_before[attribute]
                > initial_mass + tolerance
                and turn.profile_attribute_influenced_action[attribute]
                for turn in trajectory.turns
            )
        )
        shadow_equivalent = (
            terminal_shadow_mass
            >= terminal_mass - shadow_equivalence_tolerance
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
                        abs(
                            turn.wrong_mass_after[attribute]
                            - initial_mass
                        )
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain_id": self.domain_id,
            "user_id": self.user_id,
            "initial_profile_condition": self.initial_profile_condition,
            "updater_id": self.updater_id,
            "replicate": self.replicate,
            "profile_trajectory_id": self.profile_trajectory_id,
            "balanced_trajectory_id": self.balanced_trajectory_id,
            "evidence_selection_cost": self.evidence_selection_cost,
            "profile_attribution_cost": self.profile_attribution_cost,
            "balanced_attribution_cost": self.balanced_attribution_cost,
            "self_confirmation_interaction": (
                self.self_confirmation_interaction
            ),
        }


@dataclass(frozen=True, slots=True)
class ExperimentBResult:
    trajectories: tuple[ClosedLoopTrajectory, ...]
    decompositions: tuple[DecompositionRow, ...]
    self_confirmation_assessments: tuple[SelfConfirmationAssessment, ...]

    @property
    def reportable_self_confirming(self) -> tuple[SelfConfirmationAssessment, ...]:
        return reportable_self_confirming_cases(
            self.self_confirmation_assessments
        )

    def to_dict(self, *, include_truth: bool = False) -> dict[str, Any]:
        return {
            "experiment": "B",
            "trajectories": [
                trajectory.to_dict(include_truth=include_truth)
                for trajectory in self.trajectories
            ],
            "decomposition": [
                row.to_dict() for row in self.decompositions
            ],
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
    materially_wrong_mass: float = 0.50,
    lcg_threshold: float = 0.25,
    shadow_equivalence_tolerance: float = 0.05,
    false_stability_tolerance: float = 0.02,
    direction_tolerance: float = 1e-9,
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
                            trajectory_id = (
                                f"{paired_key}:{policy_id}:{updater_id}"
                            )
                            trajectory = run_trajectory(
                                user=user,
                                domain=domain,
                                policy=policy,
                                updater=updater,
                                turns=turns,
                                seed=seed,
                                initial_belief=initial,
                                initial_profile_condition=condition,
                                profile_strength=profile_strength,
                                prior_uncertainty=prior_uncertainty,
                                response_model=declared_response,
                                shadow_updater=shadow_updater,
                                trajectory_id=trajectory_id,
                                crn_key=paired_key,
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
                                        materially_wrong_mass=(
                                            materially_wrong_mass
                                        ),
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
                            decompositions.append(
                                DecompositionRow(
                                    domain_id=domain.domain_id,
                                    user_id=user.user_id,
                                    initial_profile_condition=condition,
                                    updater_id=updater_id,
                                    replicate=replicate,
                                    profile_trajectory_id=(
                                        profile.trajectory_id
                                    ),
                                    balanced_trajectory_id=(
                                        balanced.trajectory_id
                                    ),
                                    evidence_selection_cost=selection_cost(
                                        profile.terminal_shadow_error,
                                        balanced.terminal_shadow_error,
                                    ),
                                    profile_attribution_cost=profile_attr,
                                    balanced_attribution_cost=balanced_attr,
                                    self_confirmation_interaction=(
                                        self_confirmation_interaction(
                                            profile.terminal_error,
                                            profile.terminal_shadow_error,
                                            balanced.terminal_error,
                                            balanced.terminal_shadow_error,
                                        )
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
