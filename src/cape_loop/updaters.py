"""Auditable structured profile updaters with explicit information views.

The experiment harness constructs :class:`UpdateView` objects centrally.  An
updater therefore cannot quietly reach into a trajectory store, inspect latent
truth, or receive policy provenance in a nominally full-context condition.

The deterministic ``response_only`` and ``full_context_blind`` writers are
inspectable stand-ins for replayed external writers.  Provider-backed results
can use the same state/result contract after validation by ``llm_exchange``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Protocol, runtime_checkable

from .beliefs import (
    THETA_STATES,
    JointThetaPsiBelief,
    MarginalPreferenceBelief,
    PreferenceBelief,
)
from .domains import get_domain
from .fitting import AwareConditionalLogitModel, UnawareSemanticDirectionModel
from .inference import exact_aware_update, theta_bayes_update
from .llm_exchange import (
    ATTRIBUTES,
    VALUES,
    CompletionProvider,
    LLMRequest,
    LLMResponse,
    ReplayProvider,
)
from .response import RandomUtilityModel
from .schemas import (
    InteractionContext,
    Observation,
    Option,
    PolicyProvenance,
    ProfileUpdate,
    Susceptibility,
)


class UpdateViewKind(str, Enum):
    """The three permitted updater information projections."""

    RESPONSE_ONLY = "response_only"
    FULL_CONTEXT = "full_context"
    PROVENANCE_AWARE = "provenance_aware"


# Readable alias used in configuration and documentation.
InformationView = UpdateViewKind
MODEL_VISIBLE_OPTION_ALIAS_POLICY = "presented-option-position-alias-v1"


@dataclass(frozen=True, slots=True)
class UpdateView:
    """One event projected to exactly the information an updater may consume."""

    kind: UpdateViewKind
    event_id: str
    observation: Observation
    selected_option: Option
    target_attribute: int
    context: InteractionContext | None = None
    provenance: PolicyProvenance | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, UpdateViewKind):
            object.__setattr__(self, "kind", UpdateViewKind(self.kind))
        if not self.event_id:
            raise ValueError("event_id cannot be empty")
        if self.observation.selected_option_id != self.selected_option.option_id:
            raise ValueError("selected_option must match the observation")
        if not 0 <= self.target_attribute < 3:
            raise ValueError("target_attribute must be in [0, 3)")

        if self.kind is UpdateViewKind.RESPONSE_ONLY:
            if self.context is not None or self.provenance is not None:
                raise ValueError(
                    "response-only views cannot contain context or provenance"
                )
        elif self.kind is UpdateViewKind.FULL_CONTEXT:
            if self.context is None:
                raise ValueError("full-context views require visible context")
            if self.provenance is not None:
                raise ValueError("full-context views cannot contain policy provenance")
        else:
            if self.context is None or self.provenance is None:
                raise ValueError(
                    "provenance-aware views require context and provenance"
                )

        if self.context is not None:
            if self.observation.selected_option_id not in self.context.option_ids:
                raise ValueError("observation must select a displayed option")
            if (
                self.context.option(self.selected_option.option_id)
                != self.selected_option
            ):
                raise ValueError("selected_option differs from the visible context")
            if self.context.target_attribute != self.target_attribute:
                raise ValueError("target_attribute differs from the visible context")

    @property
    def selected_direction(self) -> int:
        value = self.selected_option.features[self.target_attribute]
        if value == 0.0:
            raise ValueError("the selected option is neutral on the target attribute")
        return 1 if value > 0.0 else -1


def make_update_view(
    kind: UpdateViewKind | str,
    context: InteractionContext,
    observation: Observation,
    provenance: PolicyProvenance | None = None,
    *,
    event_id: str | None = None,
) -> UpdateView:
    """Project a complete event into a declared, leakage-resistant view."""

    view_kind = UpdateViewKind(kind)
    if context.target_attribute is None:
        raise ValueError("updater views require context.target_attribute")
    selected = context.option(observation.selected_option_id)
    identifier = context.context_id if event_id is None else event_id
    if view_kind is UpdateViewKind.RESPONSE_ONLY:
        return UpdateView(
            view_kind,
            identifier,
            observation,
            selected,
            context.target_attribute,
        )
    if view_kind is UpdateViewKind.FULL_CONTEXT:
        return UpdateView(
            view_kind,
            identifier,
            observation,
            selected,
            context.target_attribute,
            context=context,
        )
    if provenance is None:
        raise ValueError("provenance-aware projection requires provenance")
    return UpdateView(
        view_kind,
        identifier,
        observation,
        selected,
        context.target_attribute,
        context=context,
        provenance=provenance,
    )


@dataclass(frozen=True, slots=True)
class UpdaterState:
    """Immutable public belief plus optional updater-private sequential state."""

    updater_id: str
    belief: PreferenceBelief
    turn: int = 0
    event_ids: tuple[str, ...] = ()
    joint_belief: JointThetaPsiBelief | None = None
    opaque_state: object | None = None

    def __post_init__(self) -> None:
        if not self.updater_id:
            raise ValueError("updater_id cannot be empty")
        if self.turn < 0:
            raise ValueError("turn cannot be negative")
        if self.turn != len(self.event_ids):
            raise ValueError("turn must equal the number of consumed event IDs")
        if len(set(self.event_ids)) != len(self.event_ids):
            raise ValueError("an updater cannot consume the same event twice")
        if self.joint_belief is not None:
            theta_belief = self.joint_belief.theta_belief()
            if any(
                not math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12)
                for a, b in zip(theta_belief.probabilities, self.belief.probabilities)
            ):
                raise ValueError("public belief must be the joint theta marginal")

    def to_dict(self) -> dict[str, Any]:
        """Serialize every reproducibility-relevant part of an updater state."""

        result: dict[str, Any] = {
            "updater_id": self.updater_id,
            "turn": self.turn,
            "event_ids": list(self.event_ids),
            "belief": self.belief.to_dict(),
        }
        if self.joint_belief is not None:
            result["joint_belief"] = self.joint_belief.to_dict()
        if self.opaque_state is not None:
            to_dict = getattr(self.opaque_state, "to_dict", None)
            if callable(to_dict):
                opaque_payload = to_dict()
                if not isinstance(opaque_payload, Mapping):
                    raise TypeError(
                        "opaque updater state to_dict() must return a mapping"
                    )
                result["opaque_state"] = dict(opaque_payload)
        return result


DiagnosticValue = float | int | str | bool


@dataclass(frozen=True, slots=True)
class UpdateResult:
    """A replayable state transition and its serializable audit record."""

    state: UpdaterState
    profile_update: ProfileUpdate
    diagnostics: tuple[tuple[str, DiagnosticValue], ...] = ()

    def __post_init__(self) -> None:
        keys = tuple(key for key, _ in self.diagnostics)
        if len(keys) != len(set(keys)):
            raise ValueError("diagnostic keys must be unique")
        if self.profile_update.updater_id != self.state.updater_id:
            raise ValueError("profile update and state updater IDs differ")

    def diagnostic(self, name: str, default: Any = None) -> Any:
        return dict(self.diagnostics).get(name, default)

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.to_dict(),
            "profile_update": self.profile_update.to_dict(),
            "diagnostics": dict(self.diagnostics),
        }


@runtime_checkable
class ProfileUpdater(Protocol):
    """Common protocol for structured and native-memory adapters."""

    updater_id: str
    view_kind: UpdateViewKind

    def initial_state(self, prior: PreferenceBelief) -> UpdaterState: ...

    def update(self, state: UpdaterState, view: UpdateView) -> UpdateResult: ...


# Concise public synonym.
Updater = ProfileUpdater


def _check_transition(
    updater_id: str,
    expected_view: UpdateViewKind,
    state: UpdaterState,
    view: UpdateView,
) -> None:
    if state.updater_id != updater_id:
        raise ValueError(f"state belongs to {state.updater_id!r}, not {updater_id!r}")
    if view.kind is not expected_view:
        raise ValueError(
            f"{updater_id} requires {expected_view.value}, got {view.kind.value}"
        )
    if view.event_id in state.event_ids:
        raise ValueError(f"event {view.event_id!r} has already been consumed")


def _profile_record(
    updater_id: str,
    before: PreferenceBelief,
    after: PreferenceBelief,
    *,
    delta: str,
    native_before: tuple[str, ...] = (),
    native_after: tuple[str, ...] = (),
) -> ProfileUpdate:
    return ProfileUpdate(
        updater_id=updater_id,
        belief_before=before.probabilities,
        belief_after=after.probabilities,
        native_memory_before=native_before,
        native_memory_after=native_after,
        written_delta=(delta,),
    )


def _next_state(
    state: UpdaterState,
    belief: PreferenceBelief,
    event_id: str,
    *,
    joint_belief: JointThetaPsiBelief | None = None,
    opaque_state: object | None = None,
) -> UpdaterState:
    return UpdaterState(
        updater_id=state.updater_id,
        belief=belief,
        turn=state.turn + 1,
        event_ids=state.event_ids + (event_id,),
        joint_belief=joint_belief,
        opaque_state=opaque_state,
    )


def _initial_state(updater_id: str, prior: PreferenceBelief) -> UpdaterState:
    return UpdaterState(updater_id=updater_id, belief=prior)


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _tempered_probability(probability: float, evidence_scale: float) -> float:
    """Temper binary evidence around chance without changing its direction."""

    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must lie in [0, 1]")
    if not math.isfinite(evidence_scale) or evidence_scale < 0.0:
        raise ValueError("evidence_scale must be finite and non-negative")
    clipped = min(max(probability, 1e-12), 1.0 - 1e-12)
    log_odds = math.log(clipped / (1.0 - clipped))
    return _sigmoid(evidence_scale * log_odds)


def _semantic_update(
    prior: PreferenceBelief,
    view: UpdateView,
    likelihood_model: UnawareSemanticDirectionModel,
    *,
    evidence_scale: float = 1.0,
) -> PreferenceBelief:
    label = view.selected_direction
    weights = []
    for theta, prior_probability in zip(THETA_STATES, prior.probabilities):
        raw = likelihood_model.label_probability(
            theta,
            view.target_attribute,
            label,
        )
        weights.append(prior_probability * _tempered_probability(raw, evidence_scale))
    return PreferenceBelief.from_weights(weights)


def _visible_evidence_weight(view: UpdateView) -> float:
    """Rule-based discount using only the context visible to the user."""

    if view.context is None:
        raise ValueError("visible evidence weights require context")
    context = view.context
    selected = view.observation.selected_option_id
    directions: set[int] = set()
    for option in context.options:
        feature = option.features[view.target_attribute]
        if feature != 0.0:
            directions.add(1 if feature > 0.0 else -1)

    weight = 1.0
    if len(directions) < 2:
        weight *= 0.08
    if context.rank(selected) == 0 and len(context.options) > 1:
        weight *= 0.80
    if context.default_option_id == selected:
        weight *= 0.35
    if context.suggested_option_id == selected:
        weight *= 0.45
    return weight


def _blind_context_evidence_weight(view: UpdateView) -> float:
    """Use option-set diagnosticity without interpreting presentation causes."""

    if view.context is None:
        raise ValueError("full-context evidence weights require context")
    directions = {
        1 if option.features[view.target_attribute] > 0.0 else -1
        for option in view.context.options
        if option.features[view.target_attribute] != 0.0
    }
    # A naive full-dialogue writer can react to whether both directions were
    # available while still ignoring why ranking/default/suggestion occurred.
    return 1.10 if len(directions) >= 2 else 0.65


def _provenance_evidence_weight(view: UpdateView) -> float:
    if view.provenance is None:
        raise ValueError("provenance evidence weights require provenance")
    weight = _visible_evidence_weight(view)
    directional = {
        "ranking",
        "default",
        "suggestion",
        "restriction",
    }
    if (
        view.provenance.profile_conditioned
        and view.provenance.presentation_mechanism in directional
    ):
        weight *= 0.75
        snapshot = dict(view.provenance.profile_snapshot)
        expected = snapshot.get(
            f"attribute_{view.target_attribute + 1}",
            snapshot.get(f"attribute_{view.target_attribute}", 0.0),
        )
        if expected and (1 if expected > 0.0 else -1) == view.selected_direction:
            weight *= 0.70
    return weight


@dataclass(frozen=True, slots=True)
class NoUpdateUpdater:
    updater_id: str = "no_update"
    view_kind: UpdateViewKind = UpdateViewKind.FULL_CONTEXT

    def initial_state(self, prior: PreferenceBelief) -> UpdaterState:
        return _initial_state(self.updater_id, prior)

    def update(self, state: UpdaterState, view: UpdateView) -> UpdateResult:
        _check_transition(self.updater_id, self.view_kind, state, view)
        next_state = _next_state(state, state.belief, view.event_id)
        return UpdateResult(
            next_state,
            _profile_record(
                self.updater_id,
                state.belief,
                state.belief,
                delta="no update",
            ),
            (("evidence_weight", 0.0),),
        )


DEFAULT_SUSCEPTIBILITY_SUPPORT: tuple[Susceptibility, ...] = (
    Susceptibility(),
    Susceptibility(ranking=0.35, default=0.80, suggestion=0.65),
    Susceptibility(ranking=0.70, default=1.20, suggestion=1.00),
)


@dataclass(frozen=True, slots=True)
class ExactActionAwareUpdater:
    response_model: RandomUtilityModel = RandomUtilityModel()
    susceptibilities: tuple[Susceptibility, ...] = DEFAULT_SUSCEPTIBILITY_SUPPORT
    susceptibility_weights: tuple[float, ...] | None = None
    updater_id: str = "exact_action_aware"
    view_kind: UpdateViewKind = UpdateViewKind.FULL_CONTEXT

    def __post_init__(self) -> None:
        if not self.susceptibilities:
            raise ValueError("exact updater requires susceptibility support")

    def initial_state(self, prior: PreferenceBelief) -> UpdaterState:
        joint = JointThetaPsiBelief.from_independent(
            prior,
            self.susceptibilities,
            self.susceptibility_weights,
        )
        return UpdaterState(
            updater_id=self.updater_id,
            belief=prior,
            joint_belief=joint,
        )

    def update(self, state: UpdaterState, view: UpdateView) -> UpdateResult:
        _check_transition(self.updater_id, self.view_kind, state, view)
        if state.joint_belief is None or view.context is None:
            raise ValueError("exact updater state/view is incomplete")
        joint = exact_aware_update(
            state.joint_belief,
            view.context,
            view.observation,
            self.response_model,
        )
        belief = joint.theta_belief()
        next_state = _next_state(
            state,
            belief,
            view.event_id,
            joint_belief=joint,
        )
        return UpdateResult(
            next_state,
            _profile_record(
                self.updater_id,
                state.belief,
                belief,
                delta=f"exact aware update from {view.event_id}",
            ),
            (("evidence_weight", 1.0), ("reference", True)),
        )


@dataclass(frozen=True, slots=True)
class FittedActionAwareUpdater:
    likelihood_model: AwareConditionalLogitModel = AwareConditionalLogitModel(
        (1.0, 0.35, 0.80, 0.65)
    )
    updater_id: str = "fitted_action_aware"
    view_kind: UpdateViewKind = UpdateViewKind.FULL_CONTEXT

    def initial_state(self, prior: PreferenceBelief) -> UpdaterState:
        return _initial_state(self.updater_id, prior)

    def update(self, state: UpdaterState, view: UpdateView) -> UpdateResult:
        _check_transition(self.updater_id, self.view_kind, state, view)
        if view.context is None:
            raise ValueError("fitted aware update requires visible context")
        belief = theta_bayes_update(
            state.belief,
            self.likelihood_model,
            view.context,
            view.observation,
        )
        next_state = _next_state(state, belief, view.event_id)
        return UpdateResult(
            next_state,
            _profile_record(
                self.updater_id,
                state.belief,
                belief,
                delta=f"fitted aware update from {view.event_id}",
            ),
            (("evidence_weight", 1.0), ("reference", True)),
        )


@dataclass(frozen=True, slots=True)
class FittedActionUnawareUpdater:
    likelihood_model: UnawareSemanticDirectionModel = UnawareSemanticDirectionModel(
        (1.0, 0.0, 0.0, 0.0)
    )
    updater_id: str = "fitted_action_unaware"
    view_kind: UpdateViewKind = UpdateViewKind.RESPONSE_ONLY

    def initial_state(self, prior: PreferenceBelief) -> UpdaterState:
        return _initial_state(self.updater_id, prior)

    def update(self, state: UpdaterState, view: UpdateView) -> UpdateResult:
        _check_transition(self.updater_id, self.view_kind, state, view)
        belief = _semantic_update(state.belief, view, self.likelihood_model)
        next_state = _next_state(state, belief, view.event_id)
        return UpdateResult(
            next_state,
            _profile_record(
                self.updater_id,
                state.belief,
                belief,
                delta=f"context-free fitted update from {view.event_id}",
            ),
            (("evidence_weight", 1.0), ("reference", True)),
        )


@dataclass(frozen=True, slots=True)
class ResponseOnlyUpdater:
    """Inspectable provenance-blind writer receiving only the chosen item."""

    likelihood_model: UnawareSemanticDirectionModel = UnawareSemanticDirectionModel(
        (1.35, 0.0, 0.0, 0.0)
    )
    updater_id: str = "response_only"
    view_kind: UpdateViewKind = UpdateViewKind.RESPONSE_ONLY

    def initial_state(self, prior: PreferenceBelief) -> UpdaterState:
        return _initial_state(self.updater_id, prior)

    def update(self, state: UpdaterState, view: UpdateView) -> UpdateResult:
        _check_transition(self.updater_id, self.view_kind, state, view)
        belief = _semantic_update(state.belief, view, self.likelihood_model)
        next_state = _next_state(state, belief, view.event_id)
        return UpdateResult(
            next_state,
            _profile_record(
                self.updater_id,
                state.belief,
                belief,
                delta=(
                    f"generalized selected direction {view.selected_direction:+d} "
                    f"on attribute {view.target_attribute}"
                ),
            ),
            (("evidence_weight", 1.0), ("provenance_conditioned", False)),
        )


@dataclass(frozen=True, slots=True)
class FullContextBlindUpdater:
    """Full-dialogue proxy using option-set context but ignoring its causes."""

    likelihood_model: UnawareSemanticDirectionModel = UnawareSemanticDirectionModel(
        (1.35, 0.0, 0.0, 0.0)
    )
    updater_id: str = "full_context_blind"
    view_kind: UpdateViewKind = UpdateViewKind.FULL_CONTEXT

    def initial_state(self, prior: PreferenceBelief) -> UpdaterState:
        return _initial_state(self.updater_id, prior)

    def update(self, state: UpdaterState, view: UpdateView) -> UpdateResult:
        _check_transition(self.updater_id, self.view_kind, state, view)
        weight = _blind_context_evidence_weight(view)
        belief = _semantic_update(
            state.belief,
            view,
            self.likelihood_model,
            evidence_scale=weight,
        )
        next_state = _next_state(state, belief, view.event_id)
        return UpdateResult(
            next_state,
            _profile_record(
                self.updater_id,
                state.belief,
                belief,
                delta=(
                    "full-context provenance-blind "
                    f"weight={weight:.6f} from {view.event_id}"
                ),
            ),
            (
                ("evidence_weight", weight),
                ("provenance_conditioned", False),
            ),
        )


@dataclass(frozen=True, slots=True)
class ProvenanceDiscountUpdater:
    """Simple rule baseline that discounts visible elicitation treatments."""

    likelihood_model: UnawareSemanticDirectionModel = UnawareSemanticDirectionModel(
        (1.35, 0.0, 0.0, 0.0)
    )
    updater_id: str = "provenance_discount"
    view_kind: UpdateViewKind = UpdateViewKind.FULL_CONTEXT

    def initial_state(self, prior: PreferenceBelief) -> UpdaterState:
        return _initial_state(self.updater_id, prior)

    def update(self, state: UpdaterState, view: UpdateView) -> UpdateResult:
        _check_transition(self.updater_id, self.view_kind, state, view)
        weight = _visible_evidence_weight(view)
        belief = _semantic_update(
            state.belief,
            view,
            self.likelihood_model,
            evidence_scale=weight,
        )
        next_state = _next_state(state, belief, view.event_id)
        return UpdateResult(
            next_state,
            _profile_record(
                self.updater_id,
                state.belief,
                belief,
                delta=f"visible-provenance weight={weight:.6f}",
            ),
            (("evidence_weight", weight), ("provenance_conditioned", True)),
        )


@dataclass(frozen=True, slots=True)
class ProvenanceAwareUpdater:
    """Diagnostic writer using visible context and structured policy metadata."""

    likelihood_model: UnawareSemanticDirectionModel = UnawareSemanticDirectionModel(
        (1.35, 0.0, 0.0, 0.0)
    )
    updater_id: str = "provenance_aware"
    view_kind: UpdateViewKind = UpdateViewKind.PROVENANCE_AWARE

    def initial_state(self, prior: PreferenceBelief) -> UpdaterState:
        return _initial_state(self.updater_id, prior)

    def update(self, state: UpdaterState, view: UpdateView) -> UpdateResult:
        _check_transition(self.updater_id, self.view_kind, state, view)
        weight = _provenance_evidence_weight(view)
        belief = _semantic_update(
            state.belief,
            view,
            self.likelihood_model,
            evidence_scale=weight,
        )
        next_state = _next_state(state, belief, view.event_id)
        return UpdateResult(
            next_state,
            _profile_record(
                self.updater_id,
                state.belief,
                belief,
                delta=f"structured-provenance weight={weight:.6f}",
            ),
            (("evidence_weight", weight), ("provenance_conditioned", True)),
        )


@dataclass(frozen=True, slots=True)
class ConservativeUpdater:
    """Threshold writer that declines weak one-event generalizations."""

    likelihood_model: UnawareSemanticDirectionModel = UnawareSemanticDirectionModel(
        (1.0, 0.0, 0.0, 0.0)
    )
    confidence_threshold: float = 0.68
    minimum_change: float = 0.04
    updater_id: str = "conservative"
    view_kind: UpdateViewKind = UpdateViewKind.FULL_CONTEXT

    def __post_init__(self) -> None:
        if not 0.5 < self.confidence_threshold < 1.0:
            raise ValueError("confidence_threshold must lie in (0.5, 1)")
        if not 0.0 <= self.minimum_change < 1.0:
            raise ValueError("minimum_change must lie in [0, 1)")

    def initial_state(self, prior: PreferenceBelief) -> UpdaterState:
        return _initial_state(self.updater_id, prior)

    def update(self, state: UpdaterState, view: UpdateView) -> UpdateResult:
        _check_transition(self.updater_id, self.view_kind, state, view)
        weight = _visible_evidence_weight(view)
        proposal = _semantic_update(
            state.belief,
            view,
            self.likelihood_model,
            evidence_scale=weight,
        )
        direction = view.selected_direction
        before_mass = state.belief.sign_mass(view.target_attribute, direction)
        after_mass = proposal.sign_mass(view.target_attribute, direction)
        accepted = (
            after_mass >= self.confidence_threshold
            and after_mass - before_mass >= self.minimum_change
        )
        belief = proposal if accepted else state.belief
        next_state = _next_state(state, belief, view.event_id)
        return UpdateResult(
            next_state,
            _profile_record(
                self.updater_id,
                state.belief,
                belief,
                delta=(
                    f"{'accepted' if accepted else 'deferred'} candidate "
                    f"mass={after_mass:.6f}"
                ),
            ),
            (
                ("accepted", accepted),
                ("candidate_direction_mass", after_mass),
                ("evidence_weight", weight),
            ),
        )


class LLMReplayUpdater:
    """Profile updater backed by a prompt-hash-bound completion provider.

    The historical class name is retained as a stable public API.  Its
    provider may be either the offline replay implementation or an explicitly
    authorized live adapter that returns the same validated ``LLMResponse``
    contract.
    """

    def __init__(
        self,
        updater_id: str,
        view_kind: UpdateViewKind,
        provider: CompletionProvider,
    ) -> None:
        if not updater_id:
            raise ValueError("LLM updater_id cannot be empty")
        self.updater_id = updater_id
        self.view_kind = view_kind
        self.provider = provider
        self._requests: dict[str, LLMRequest] = {}
        self._responses: dict[str, LLMResponse] = {}

    @property
    def requests(self) -> tuple[LLMRequest, ...]:
        return tuple(self._requests.values())

    @property
    def responses(self) -> tuple[LLMResponse, ...]:
        return tuple(self._responses.values())

    def initial_state(self, prior: PreferenceBelief) -> UpdaterState:
        return _initial_state(self.updater_id, prior)

    @staticmethod
    def _belief_payload(belief: PreferenceBelief) -> dict[str, Any]:
        marginals = belief.marginals().probabilities
        return {
            attribute: {
                value: probability
                for value, probability in zip(VALUES, marginals[index])
            }
            for index, attribute in enumerate(ATTRIBUTES)
        }

    @staticmethod
    def _profile_schema(domain_id: str) -> dict[str, Any]:
        """Describe the three latent dimensions without exposing math fields."""

        domain = get_domain(domain_id)
        result: dict[str, Any] = {}
        for index, attribute in enumerate(domain.attributes, start=1):
            result[f"attribute_{index}"] = {
                "name": attribute.key,
                "values": {
                    "-2": (f"strongly favors {attribute.negative_label}"),
                    "-1": (f"somewhat favors {attribute.negative_label}"),
                    "+1": (f"somewhat favors {attribute.positive_label}"),
                    "+2": (f"strongly favors {attribute.positive_label}"),
                },
            }
        return result

    @staticmethod
    def _visible_option(option: Option, alias: str) -> dict[str, str]:
        """Return readable material under a non-semantic per-view alias."""

        return {
            "option_id": alias,
            "description": option.label,
        }

    def build_request(
        self,
        state: UpdaterState,
        view: UpdateView,
    ) -> LLMRequest:
        """Build the exact request without calling the completion provider."""

        if view.context is None:
            alias_by_id = {
                view.selected_option.option_id: "selected_option",
            }
        else:
            alias_by_id = {
                option_id: f"presented_option_{index}"
                for index, option_id in enumerate(
                    view.context.ranking,
                    start=1,
                )
            }
        selected_alias = alias_by_id[view.observation.selected_option_id]
        observation: dict[str, Any] = {
            "selected_option": selected_alias,
            "user_message": view.observation.surface_response,
            "selected_option_record": self._visible_option(
                view.selected_option,
                selected_alias,
            ),
            "profile_schema": self._profile_schema(view.selected_option.domain),
        }
        context = None
        if view.context is not None:
            context = {
                # Audit identifiers are intentionally absent. They can encode
                # user, initial-profile, policy, or CRN labels that are not
                # part of the user-visible elicitation context.
                "domain": view.context.domain,
                "options": [
                    self._visible_option(
                        option,
                        alias_by_id[option.option_id],
                    )
                    for option in view.context.options
                ],
                "ranking": [
                    alias_by_id[option_id] for option_id in view.context.ranking
                ],
                "default": (
                    None
                    if view.context.default_option_id is None
                    else alias_by_id[view.context.default_option_id]
                ),
                "suggested_option": (
                    None
                    if view.context.suggested_option_id is None
                    else alias_by_id[view.context.suggested_option_id]
                ),
                "question_type": view.context.question_type,
            }
            if view.context.prompt is not None:
                context["task"] = view.context.prompt
            if view.observation.assistant_message is not None:
                context["conversation"] = [
                    {
                        "role": "assistant",
                        "content": view.observation.assistant_message,
                    },
                    {
                        "role": "user",
                        "content": view.observation.surface_response,
                    },
                ]
        draft = LLMRequest.build(
            request_id="content-addressed-request",
            updater_id=self.updater_id,
            view=self.view_kind.value,
            prior=self._belief_payload(state.belief),
            observation=observation,
            context=context,
            provenance=(None if view.provenance is None else view.provenance.to_dict()),
        )
        # Event IDs are audit identifiers and may encode condition/user labels.
        # The replay key is therefore derived only from model-visible prompt
        # material. Identical prompts intentionally share one response record.
        return LLMRequest(
            request_id=f"{self.updater_id}:{draft.prompt_sha256}",
            updater_id=draft.updater_id,
            view=draft.view,
            payload=draft.payload,
            system_instruction=draft.system_instruction,
            prompt_sha256=draft.prompt_sha256,
        )

    def update(self, state: UpdaterState, view: UpdateView) -> UpdateResult:
        _check_transition(self.updater_id, self.view_kind, state, view)
        request = self.build_request(state, view)
        existing = self._requests.get(request.request_id)
        if existing is not None and existing != request:
            raise ValueError(
                f"request ID collision with different prompt: {request.request_id}"
            )
        response = self.provider.complete(request)
        rows = tuple(
            tuple(float(response.beliefs[attribute][value]) for value in VALUES)
            for attribute in ATTRIBUTES
        )
        prior_rows = state.belief.marginals().probabilities
        raw_equals_prior = all(
            math.isclose(
                returned,
                prior,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            for returned_row, prior_row in zip(rows, prior_rows)
            for returned, prior in zip(returned_row, prior_row)
        )
        belief = (
            state.belief
            if raw_equals_prior
            else PreferenceBelief.from_marginals(
                MarginalPreferenceBelief(rows)  # type: ignore[arg-type]
            )
        )
        self._requests[request.request_id] = request
        self._responses[request.request_id] = response
        next_state = _next_state(state, belief, view.event_id)
        return UpdateResult(
            next_state,
            _profile_record(
                self.updater_id,
                state.belief,
                belief,
                delta=(f"external {response.model_id} response {request.request_id}"),
            ),
            (
                ("evidence_weight", 1.0),
                ("model_id", response.model_id),
                ("prompt_sha256", request.prompt_sha256),
                ("external_model", True),
                ("returned_prior_unchanged", raw_equals_prior),
                (
                    "execution_mode",
                    "replay" if isinstance(self.provider, ReplayProvider) else "live",
                ),
            ),
        )


def build_updater(
    updater_id: str,
    *,
    response_model: RandomUtilityModel | None = None,
    aware_model: AwareConditionalLogitModel | None = None,
    unaware_model: UnawareSemanticDirectionModel | None = None,
    susceptibilities: tuple[Susceptibility, ...] | None = None,
    susceptibility_weights: tuple[float, ...] | None = None,
    replay_provider: CompletionProvider | None = None,
) -> ProfileUpdater:
    """Construct a deterministic updater from a configuration identifier."""

    if updater_id == "no_update":
        return NoUpdateUpdater()
    if updater_id == "exact_action_aware":
        return ExactActionAwareUpdater(
            response_model or RandomUtilityModel(),
            susceptibilities or DEFAULT_SUSCEPTIBILITY_SUPPORT,
            susceptibility_weights,
        )
    if updater_id == "fitted_action_aware":
        return FittedActionAwareUpdater(
            aware_model or AwareConditionalLogitModel((1.0, 0.35, 0.80, 0.65))
        )
    if updater_id == "fitted_action_unaware":
        return FittedActionUnawareUpdater(
            unaware_model or UnawareSemanticDirectionModel((1.0, 0.0, 0.0, 0.0))
        )
    if updater_id == "response_only":
        return ResponseOnlyUpdater(
            unaware_model or UnawareSemanticDirectionModel((1.35, 0.0, 0.0, 0.0))
        )
    if updater_id == "full_context_blind":
        return FullContextBlindUpdater(
            unaware_model or UnawareSemanticDirectionModel((1.35, 0.0, 0.0, 0.0))
        )
    if updater_id == "provenance_discount":
        return ProvenanceDiscountUpdater(
            unaware_model or UnawareSemanticDirectionModel((1.35, 0.0, 0.0, 0.0))
        )
    if updater_id == "provenance_aware":
        return ProvenanceAwareUpdater(
            unaware_model or UnawareSemanticDirectionModel((1.35, 0.0, 0.0, 0.0))
        )
    if updater_id == "conservative":
        return ConservativeUpdater(
            unaware_model or UnawareSemanticDirectionModel((1.0, 0.0, 0.0, 0.0))
        )
    llm_views = {
        "llm_response_only": UpdateViewKind.RESPONSE_ONLY,
        "llm_full_context": UpdateViewKind.FULL_CONTEXT,
        "llm_provenance_aware": UpdateViewKind.PROVENANCE_AWARE,
    }
    if updater_id in llm_views:
        if replay_provider is None:
            raise ValueError(f"{updater_id} requires a configured completion provider")
        return LLMReplayUpdater(
            updater_id,
            llm_views[updater_id],
            replay_provider,
        )
    if updater_id in {
        "episodic_memory",
        "semantic_memory",
        "provenance_linked_memory",
    }:
        # The local import keeps the native module free to implement this same
        # protocol without a module-import cycle.
        from .native import build_native_updater

        return build_native_updater(updater_id)
    raise KeyError(f"unknown updater: {updater_id!r}")


def build_updater_registry(
    updater_ids: tuple[str, ...] | list[str],
    **kwargs: Any,
) -> dict[str, ProfileUpdater]:
    """Construct a unique, insertion-ordered updater registry."""

    result: dict[str, ProfileUpdater] = {}
    for updater_id in updater_ids:
        if updater_id in result:
            raise ValueError(f"duplicate updater ID: {updater_id!r}")
        result[updater_id] = build_updater(updater_id, **kwargs)
    return result


def updater_views(
    updaters: Mapping[str, ProfileUpdater],
) -> dict[str, str]:
    """Return the declared view manifest used by run metadata."""

    return {
        updater_id: updater.view_kind.value for updater_id, updater in updaters.items()
    }
