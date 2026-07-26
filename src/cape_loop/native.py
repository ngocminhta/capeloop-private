"""Inspectable native persistent-memory adapters and blinded decoder views."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping

from .beliefs import MarginalPreferenceBelief, PreferenceBelief, THETA_STATES
from .schemas import ProfileUpdate
from .updaters import (
    ProfileUpdater,
    UpdateResult,
    UpdaterState,
    UpdateView,
    UpdateViewKind,
)


NATIVE_MEMORY_KINDS = frozenset(
    {"episodic", "semantic", "provenance_linked"}
)


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    material = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


@dataclass(frozen=True, slots=True)
class NativeEpisode:
    """One memory episode containing no latent user information."""

    event_id: str
    selected_option_id: str
    target_attribute: int
    selected_direction: int
    visible_mechanisms: tuple[str, ...]
    displayed_directions: tuple[int, ...]
    evidence_weight: float
    surface_response: str | None = None
    provenance_policy_id: str | None = None
    provenance_profile_snapshot: tuple[tuple[str, float], ...] = ()
    provenance_presentation_mechanism: str | None = None
    provenance_profile_conditioned: bool | None = None

    def __post_init__(self) -> None:
        if not self.event_id or not self.selected_option_id:
            raise ValueError("episode identifiers cannot be empty")
        if not 0 <= self.target_attribute < 3:
            raise ValueError("target_attribute must be in [0, 3)")
        if self.selected_direction not in (-1, 1):
            raise ValueError("selected_direction must be -1 or +1")
        if any(direction not in (-1, 0, 1) for direction in self.displayed_directions):
            raise ValueError("displayed directions must be -1, 0, or +1")
        if not math.isfinite(self.evidence_weight) or self.evidence_weight < 0.0:
            raise ValueError("evidence_weight must be finite and non-negative")
        if (
            self.provenance_profile_conditioned is not None
            and not isinstance(self.provenance_profile_conditioned, bool)
        ):
            raise TypeError(
                "provenance_profile_conditioned must be a Boolean or None"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "selected_option_id": self.selected_option_id,
            "target_attribute": self.target_attribute,
            "selected_direction": self.selected_direction,
            "visible_mechanisms": list(self.visible_mechanisms),
            "displayed_directions": list(self.displayed_directions),
            "evidence_weight": self.evidence_weight,
            "surface_response": self.surface_response,
            "provenance_policy_id": self.provenance_policy_id,
            "provenance_profile_snapshot": dict(
                self.provenance_profile_snapshot
            ),
            "provenance_presentation_mechanism": (
                self.provenance_presentation_mechanism
            ),
            "provenance_profile_conditioned": (
                self.provenance_profile_conditioned
            ),
        }


@dataclass(frozen=True, slots=True)
class SemanticClaim:
    """A general preference claim linked to every supporting episode."""

    attribute: int
    direction: int
    confidence: float
    source_event_ids: tuple[str, ...]
    cumulative_evidence_weight: float

    def __post_init__(self) -> None:
        if not 0 <= self.attribute < 3:
            raise ValueError("claim attribute must be in [0, 3)")
        if self.direction not in (-1, 1):
            raise ValueError("claim direction must be -1 or +1")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("claim confidence must lie in [0, 1]")
        if not self.source_event_ids:
            raise ValueError("a semantic claim requires supporting event IDs")
        if len(set(self.source_event_ids)) != len(self.source_event_ids):
            raise ValueError("claim source event IDs must be unique")
        if (
            not math.isfinite(self.cumulative_evidence_weight)
            or self.cumulative_evidence_weight < 0.0
        ):
            raise ValueError("cumulative evidence weight must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "attribute": self.attribute,
            "direction": self.direction,
            "confidence": self.confidence,
            "source_event_ids": list(self.source_event_ids),
            "cumulative_evidence_weight": self.cumulative_evidence_weight,
        }


def _state_payload(
    memory_kind: str,
    base_belief: PreferenceBelief,
    episodes: tuple[NativeEpisode, ...],
    claims: tuple[SemanticClaim, ...],
    persona_belief: PreferenceBelief,
    persona_text: str,
) -> dict[str, Any]:
    return {
        "memory_kind": memory_kind,
        "base_belief": list(base_belief.probabilities),
        "episodes": [episode.to_dict() for episode in episodes],
        "claims": [claim.to_dict() for claim in claims],
        "persona_belief": list(persona_belief.probabilities),
        "persona_text": persona_text,
    }


@dataclass(frozen=True, slots=True)
class NativeMemoryState:
    """Immutable episodic/semantic state with a content-addressed identity."""

    memory_kind: str
    base_belief: PreferenceBelief
    episodes: tuple[NativeEpisode, ...]
    claims: tuple[SemanticClaim, ...]
    persona_belief: PreferenceBelief
    persona_text: str
    state_id: str = ""

    def __post_init__(self) -> None:
        if self.memory_kind not in NATIVE_MEMORY_KINDS:
            raise ValueError(f"unknown memory kind: {self.memory_kind!r}")
        if len({episode.event_id for episode in self.episodes}) != len(
            self.episodes
        ):
            raise ValueError("native episodes must have unique event IDs")
        supported = {episode.event_id for episode in self.episodes}
        for claim in self.claims:
            if not set(claim.source_event_ids) <= supported:
                raise ValueError("claim references an unavailable episode")
        expected = _canonical_digest(
            _state_payload(
                self.memory_kind,
                self.base_belief,
                self.episodes,
                self.claims,
                self.persona_belief,
                self.persona_text,
            )
        )
        if self.state_id and self.state_id != expected:
            raise ValueError("native state_id does not match its contents")
        object.__setattr__(self, "state_id", expected)

    @classmethod
    def empty(
        cls,
        memory_kind: str,
        prior: PreferenceBelief,
    ) -> NativeMemoryState:
        return cls(
            memory_kind=memory_kind,
            base_belief=prior,
            episodes=(),
            claims=(),
            persona_belief=prior,
            persona_text=_persona_text(prior),
        )

    @property
    def policy_belief(self) -> PreferenceBelief:
        """The only projection a policy is expected to consume."""

        return self.persona_belief

    def to_dict(self) -> dict[str, Any]:
        result = _state_payload(
            self.memory_kind,
            self.base_belief,
            self.episodes,
            self.claims,
            self.persona_belief,
            self.persona_text,
        )
        result["state_id"] = self.state_id
        return result


def _persona_text(belief: PreferenceBelief) -> str:
    parts = []
    for attribute, expected in enumerate(belief.expected_theta()):
        if abs(expected) < 0.05:
            direction = "uncertain"
        else:
            direction = "positive" if expected > 0.0 else "negative"
        confidence = max(
            belief.sign_mass(attribute, -1),
            belief.sign_mass(attribute, 1),
        )
        parts.append(
            f"attribute_{attribute + 1}={direction} ({confidence:.3f})"
        )
    return "; ".join(parts)


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _apply_episode(
    belief: PreferenceBelief,
    episode: NativeEpisode,
    *,
    strength: float = 1.25,
) -> PreferenceBelief:
    weights = []
    for theta, prior_probability in zip(THETA_STATES, belief.probabilities):
        probability = _sigmoid(
            strength
            * episode.evidence_weight
            * episode.selected_direction
            * theta[episode.target_attribute]
        )
        weights.append(prior_probability * probability)
    return PreferenceBelief.from_weights(weights)


def _replay_episodes(
    prior: PreferenceBelief,
    episodes: tuple[NativeEpisode, ...],
    *,
    strength: float,
) -> PreferenceBelief:
    belief = prior
    for episode in episodes:
        belief = _apply_episode(belief, episode, strength=strength)
    return belief


def _visible_mechanisms(view: UpdateView) -> tuple[str, ...]:
    if view.context is None:
        return ()
    context = view.context
    selected = view.observation.selected_option_id
    mechanisms: list[str] = []
    directions = {
        (1 if option.features[view.target_attribute] > 0.0 else -1)
        for option in context.options
        if option.features[view.target_attribute] != 0.0
    }
    if len(directions) < 2:
        mechanisms.append("restricted")
    if context.rank(selected) == 0:
        mechanisms.append("ranked_first")
    if context.default_option_id == selected:
        mechanisms.append("default")
    if context.suggested_option_id == selected:
        mechanisms.append("suggested")
    if not mechanisms:
        mechanisms.append("balanced")
    return tuple(mechanisms)


def _native_evidence_weight(view: UpdateView, *, provenance_aware: bool) -> float:
    mechanisms = set(_visible_mechanisms(view))
    weight = 1.0
    if "restricted" in mechanisms:
        weight *= 0.08
    if "ranked_first" in mechanisms:
        weight *= 0.80
    if "default" in mechanisms:
        weight *= 0.35
    if "suggested" in mechanisms:
        weight *= 0.45
    if provenance_aware:
        if view.provenance is None:
            raise ValueError("provenance-linked memory requires provenance")
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
                0.0,
            )
            if (
                expected
                and (1 if expected > 0.0 else -1)
                == view.selected_direction
            ):
                weight *= 0.70
    return weight


def _episode_from_view(
    view: UpdateView,
    *,
    evidence_weight: float,
    retain_provenance: bool,
) -> NativeEpisode:
    if view.context is None:
        raise ValueError("native memories require visible context")
    displayed = tuple(
        0
        if option.features[view.target_attribute] == 0.0
        else (1 if option.features[view.target_attribute] > 0.0 else -1)
        for option in view.context.options
    )
    provenance = view.provenance if retain_provenance else None
    return NativeEpisode(
        event_id=view.event_id,
        selected_option_id=view.observation.selected_option_id,
        target_attribute=view.target_attribute,
        selected_direction=view.selected_direction,
        visible_mechanisms=_visible_mechanisms(view),
        displayed_directions=displayed,
        evidence_weight=evidence_weight,
        surface_response=view.observation.surface_response,
        provenance_policy_id=(
            None if provenance is None else provenance.policy_id
        ),
        provenance_profile_snapshot=(
            () if provenance is None else provenance.profile_snapshot
        ),
        provenance_presentation_mechanism=(
            None
            if provenance is None
            else provenance.presentation_mechanism
        ),
        provenance_profile_conditioned=(
            None if provenance is None else provenance.profile_conditioned
        ),
    )


def _updated_claims(
    existing: tuple[SemanticClaim, ...],
    episode: NativeEpisode,
    belief: PreferenceBelief,
) -> tuple[SemanticClaim, ...]:
    retained = [
        claim
        for claim in existing
        if not (
            claim.attribute == episode.target_attribute
            and claim.direction == episode.selected_direction
        )
    ]
    matching = next(
        (
            claim
            for claim in existing
            if claim.attribute == episode.target_attribute
            and claim.direction == episode.selected_direction
        ),
        None,
    )
    sources = (
        (episode.event_id,)
        if matching is None
        else matching.source_event_ids + (episode.event_id,)
    )
    cumulative = episode.evidence_weight + (
        0.0 if matching is None else matching.cumulative_evidence_weight
    )
    retained.append(
        SemanticClaim(
            attribute=episode.target_attribute,
            direction=episode.selected_direction,
            confidence=belief.sign_mass(
                episode.target_attribute,
                episode.selected_direction,
            ),
            source_event_ids=sources,
            cumulative_evidence_weight=cumulative,
        )
    )
    return tuple(sorted(retained, key=lambda claim: (claim.attribute, claim.direction)))


class _NativeUpdaterBase:
    updater_id: str
    view_kind: UpdateViewKind
    memory_kind: str
    retain_provenance: bool
    discount_provenance: bool
    update_strength: float

    def initial_state(self, prior: PreferenceBelief) -> UpdaterState:
        memory = NativeMemoryState.empty(self.memory_kind, prior)
        return UpdaterState(
            updater_id=self.updater_id,
            belief=prior,
            opaque_state=memory,
        )

    def _validate(self, state: UpdaterState, view: UpdateView) -> NativeMemoryState:
        if state.updater_id != self.updater_id:
            raise ValueError("native updater received another updater's state")
        if view.kind is not self.view_kind:
            raise ValueError(
                f"{self.updater_id} requires {self.view_kind.value}, "
                f"got {view.kind.value}"
            )
        if view.event_id in state.event_ids:
            raise ValueError("native updater cannot consume an event twice")
        memory = state.opaque_state
        if not isinstance(memory, NativeMemoryState):
            raise TypeError("native updater state lacks NativeMemoryState")
        if memory.memory_kind != self.memory_kind:
            raise ValueError("native state memory kind does not match updater")
        return memory

    def update(self, state: UpdaterState, view: UpdateView) -> UpdateResult:
        memory = self._validate(state, view)
        weight = (
            _native_evidence_weight(view, provenance_aware=True)
            if self.discount_provenance
            else 1.0
        )
        episode = _episode_from_view(
            view,
            evidence_weight=weight,
            retain_provenance=self.retain_provenance,
        )
        episodes = memory.episodes + (episode,)

        if self.memory_kind == "episodic":
            # Raw episodes remain unconsolidated.  The policy projection is
            # inferred from the complete retained history at query time.
            belief = _replay_episodes(
                memory.base_belief,
                episodes,
                strength=self.update_strength,
            )
            claims: tuple[SemanticClaim, ...] = ()
        else:
            # Semantic consolidation is deliberately more conservative than
            # raw episodic query-time inference. This makes the two memory
            # systems behaviorally distinct, not merely differently serialized.
            belief = _apply_episode(
                memory.persona_belief,
                episode,
                strength=self.update_strength,
            )
            claims = _updated_claims(memory.claims, episode, belief)

        after_memory = NativeMemoryState(
            memory_kind=self.memory_kind,
            base_belief=memory.base_belief,
            episodes=episodes,
            claims=claims,
            persona_belief=belief,
            persona_text=_persona_text(belief),
        )
        next_state = UpdaterState(
            updater_id=self.updater_id,
            belief=belief,
            turn=state.turn + 1,
            event_ids=state.event_ids + (view.event_id,),
            opaque_state=after_memory,
        )
        profile_update = ProfileUpdate(
            updater_id=self.updater_id,
            belief_before=state.belief.probabilities,
            belief_after=belief.probabilities,
            native_memory_before=(memory.state_id, memory.persona_text),
            native_memory_after=(after_memory.state_id, after_memory.persona_text),
            written_delta=(
                f"stored episode {view.event_id} with evidence weight {weight:.6f}",
            ),
        )
        return UpdateResult(
            state=next_state,
            profile_update=profile_update,
            diagnostics=(
                ("evidence_weight", weight),
                ("native_state_id", after_memory.state_id),
                ("supporting_event_id", view.event_id),
            ),
        )


@dataclass(frozen=True, slots=True)
class EpisodicMemoryUpdater(_NativeUpdaterBase):
    updater_id: str = "episodic_memory"
    view_kind: UpdateViewKind = UpdateViewKind.FULL_CONTEXT
    memory_kind: str = "episodic"
    retain_provenance: bool = False
    discount_provenance: bool = False
    update_strength: float = 1.25


@dataclass(frozen=True, slots=True)
class SemanticMemoryUpdater(_NativeUpdaterBase):
    updater_id: str = "semantic_memory"
    view_kind: UpdateViewKind = UpdateViewKind.FULL_CONTEXT
    memory_kind: str = "semantic"
    retain_provenance: bool = False
    discount_provenance: bool = False
    update_strength: float = 0.90


@dataclass(frozen=True, slots=True)
class ProvenanceLinkedMemoryUpdater(_NativeUpdaterBase):
    updater_id: str = "provenance_linked_memory"
    view_kind: UpdateViewKind = UpdateViewKind.PROVENANCE_AWARE
    memory_kind: str = "provenance_linked"
    retain_provenance: bool = True
    discount_provenance: bool = True
    update_strength: float = 0.90


def build_native_updater(updater_id: str) -> ProfileUpdater:
    if updater_id == "episodic_memory":
        return EpisodicMemoryUpdater()
    if updater_id == "semantic_memory":
        return SemanticMemoryUpdater()
    if updater_id == "provenance_linked_memory":
        return ProvenanceLinkedMemoryUpdater()
    raise KeyError(f"unknown native updater: {updater_id!r}")


@dataclass(frozen=True, slots=True)
class BlindedDecoderView:
    """Versioned decoder material with no system identity or latent truth."""

    decoder_id: str
    pseudonymous_state_id: str
    payload_json: str

    def __post_init__(self) -> None:
        if self.decoder_id not in {"direct_semantic_v1", "history_evidence_v1"}:
            raise ValueError("unknown blinded decoder")
        if not self.pseudonymous_state_id:
            raise ValueError("pseudonymous state ID cannot be empty")
        payload = self.payload()
        _assert_blinded_payload(payload)

    def payload(self) -> dict[str, Any]:
        decoded = json.loads(self.payload_json)
        if not isinstance(decoded, dict):
            raise ValueError("decoder payload must be a JSON object")
        return decoded

    def to_dict(self) -> dict[str, Any]:
        return {
            "decoder_id": self.decoder_id,
            "pseudonymous_state_id": self.pseudonymous_state_id,
            "payload": self.payload(),
        }


_FORBIDDEN_DECODER_KEYS = frozenset(
    {
        "system",
        "system_id",
        "updater",
        "updater_id",
        "memory_kind",
        "latent_truth",
        "truth",
        "user_id",
    }
)


def _assert_blinded_payload(value: Any) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if str(key).lower() in _FORBIDDEN_DECODER_KEYS:
                raise ValueError(f"decoder payload leaks forbidden field {key!r}")
            _assert_blinded_payload(nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_blinded_payload(nested)


def blinded_decoder_views(
    state: NativeMemoryState,
) -> tuple[BlindedDecoderView, BlindedDecoderView]:
    """Create two meaningfully different, independently decodable views."""

    pseudonym = _canonical_digest(
        {"native_state_digest": state.state_id, "blinding_version": 1}
    )
    event_pseudonyms = {
        episode.event_id: f"event-{index:04d}"
        for index, episode in enumerate(state.episodes, start=1)
    }
    sign_confidence = [
        {
            "negative": state.persona_belief.sign_mass(attribute, -1),
            "positive": state.persona_belief.sign_mass(attribute, 1),
            "expected_preference": state.persona_belief.expected_theta()[attribute],
        }
        for attribute in range(3)
    ]
    direct_payload = {
        "schema_version": 1,
        "persona_summary": state.persona_text,
        "direction_confidence": sign_confidence,
        "semantic_claims": [
            {
                **claim.to_dict(),
                "source_event_ids": [
                    event_pseudonyms[event_id]
                    for event_id in claim.source_event_ids
                ],
            }
            for claim in state.claims
        ],
    }
    history_payload = {
        "schema_version": 1,
        "prior_marginals": [
            list(state.base_belief.marginal(attribute))
            for attribute in range(3)
        ],
        "episodes": [
            {
                "event_id": event_pseudonyms[episode.event_id],
                "target_attribute": episode.target_attribute,
                "selected_direction": episode.selected_direction,
                "visible_mechanisms": list(episode.visible_mechanisms),
                "displayed_directions": list(episode.displayed_directions),
                "evidence_weight": episode.evidence_weight,
            }
            for episode in state.episodes
        ],
    }
    encode = lambda payload: json.dumps(  # noqa: E731 - compact local encoder
        payload,
        sort_keys=True,
        separators=(",", ":"),
    )
    return (
        BlindedDecoderView(
            "direct_semantic_v1",
            pseudonym,
            encode(direct_payload),
        ),
        BlindedDecoderView(
            "history_evidence_v1",
            pseudonym,
            encode(history_payload),
        ),
    )


def _belief_from_sign_confidence(
    confidences: list[dict[str, Any]],
) -> PreferenceBelief:
    rows = []
    for entry in confidences:
        negative = float(entry["negative"])
        positive = float(entry["positive"])
        total = negative + positive
        if total <= 0.0:
            negative = positive = 0.5
        else:
            negative, positive = negative / total, positive / total
        expected = abs(float(entry.get("expected_preference", 0.0)))
        strong_share = min(max(expected / 2.0, 0.25), 0.75)
        rows.append(
            (
                negative * strong_share,
                negative * (1.0 - strong_share),
                positive * (1.0 - strong_share),
                positive * strong_share,
            )
        )
    marginals = MarginalPreferenceBelief(tuple(rows))  # type: ignore[arg-type]
    return marginals.independent_joint()


def decode_blinded_view(view: BlindedDecoderView) -> PreferenceBelief:
    """Deterministically decode either blinded representation."""

    payload = view.payload()
    if view.decoder_id == "direct_semantic_v1":
        confidences = payload.get("direction_confidence")
        if not isinstance(confidences, list) or len(confidences) != 3:
            raise ValueError("direct decoder requires three confidence entries")
        return _belief_from_sign_confidence(confidences)

    marginal_rows = payload.get("prior_marginals")
    if not isinstance(marginal_rows, list) or len(marginal_rows) != 3:
        raise ValueError("history decoder requires three prior marginals")
    prior = MarginalPreferenceBelief(
        tuple(tuple(float(value) for value in row) for row in marginal_rows)
    ).independent_joint()  # type: ignore[arg-type]
    belief = prior
    episodes = payload.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError("history decoder requires an episode list")
    for raw in episodes:
        episode = NativeEpisode(
            event_id=str(raw["event_id"]),
            selected_option_id="blinded-selection",
            target_attribute=int(raw["target_attribute"]),
            selected_direction=int(raw["selected_direction"]),
            visible_mechanisms=tuple(raw["visible_mechanisms"]),
            displayed_directions=tuple(int(x) for x in raw["displayed_directions"]),
            evidence_weight=float(raw["evidence_weight"]),
        )
        belief = _apply_episode(belief, episode)
    return belief


@dataclass(frozen=True, slots=True)
class NativeDecoderResult:
    decoder_id: str
    pseudonymous_state_id: str
    belief: PreferenceBelief


def decode_native_state(
    state: NativeMemoryState,
) -> tuple[NativeDecoderResult, NativeDecoderResult]:
    """Run both fixed decoders; callers must retain and report both."""

    views = blinded_decoder_views(state)
    results = tuple(
        NativeDecoderResult(
            decoder_id=view.decoder_id,
            pseudonymous_state_id=view.pseudonymous_state_id,
            belief=decode_blinded_view(view),
        )
        for view in views
    )
    return results  # type: ignore[return-value]
