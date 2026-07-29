"""Validated, immutable data records for CAPE-Loop.

The module deliberately separates what the user saw (:class:`InteractionContext`)
from why it was shown (:class:`PolicyProvenance`).  Simulator-only latent state is
also represented separately so that callers can enforce leakage boundaries.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any, TypeAlias


THETA_VALUES: tuple[int, ...] = (-2, -1, 1, 2)
NUM_ATTRIBUTES = 3

Theta: TypeAlias = tuple[int, int, int]
FeatureVector: TypeAlias = tuple[float, float, float]


def _require_nonempty_string(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _finite_float(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def validate_theta(theta: Sequence[int]) -> Theta:
    """Return a canonical theta tuple or raise for an invalid latent profile."""

    if len(theta) != NUM_ATTRIBUTES:
        raise ValueError(f"theta must have {NUM_ATTRIBUTES} entries")
    values: list[int] = []
    for index, value in enumerate(theta):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"theta[{index}] must be an integer")
        if value not in THETA_VALUES:
            raise ValueError(
                f"theta[{index}]={value!r}; expected one of {THETA_VALUES}"
            )
        values.append(value)
    return (values[0], values[1], values[2])


def validate_features(features: Sequence[float]) -> FeatureVector:
    """Return a canonical finite three-dimensional feature vector."""

    if len(features) != NUM_ATTRIBUTES:
        raise ValueError(f"features must have {NUM_ATTRIBUTES} entries")
    values = tuple(
        _finite_float(value, f"features[{index}]")
        for index, value in enumerate(features)
    )
    return (values[0], values[1], values[2])


@dataclass(frozen=True, slots=True, order=True)
class Susceptibility:
    """A user's fixed, numeric sensitivity to presentation mechanisms."""

    ranking: float = 0.0
    default: float = 0.0
    suggestion: float = 0.0

    def __post_init__(self) -> None:
        for field_name in ("ranking", "default", "suggestion"):
            value = _finite_float(getattr(self, field_name), field_name)
            if value < 0.0:
                raise ValueError(f"{field_name} susceptibility cannot be negative")
            object.__setattr__(self, field_name, value)

    def to_dict(self) -> dict[str, float]:
        return {
            "ranking": self.ranking,
            "default": self.default,
            "suggestion": self.suggestion,
        }


@dataclass(frozen=True, slots=True)
class LatentUser:
    """Simulator-only state.  It must never be passed to a deployed updater."""

    user_id: str
    theta: Theta
    susceptibility: Susceptibility = Susceptibility()

    def __post_init__(self) -> None:
        _require_nonempty_string(self.user_id, "user_id")
        object.__setattr__(self, "theta", validate_theta(self.theta))
        if not isinstance(self.susceptibility, Susceptibility):
            raise TypeError("susceptibility must be a Susceptibility")

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "theta": list(self.theta),
            "susceptibility": self.susceptibility.to_dict(),
        }


# A concise synonym useful in experiment code.
UserState = LatentUser


@dataclass(frozen=True, slots=True)
class Option:
    """A feasible domain option with three declared intrinsic attributes."""

    option_id: str
    features: FeatureVector
    label: str = ""
    domain: str = ""

    def __post_init__(self) -> None:
        _require_nonempty_string(self.option_id, "option_id")
        object.__setattr__(self, "features", validate_features(self.features))
        if not isinstance(self.label, str):
            raise TypeError("label must be a string")
        if not isinstance(self.domain, str):
            raise TypeError("domain must be a string")

    def to_dict(self) -> dict[str, Any]:
        return {
            "option_id": self.option_id,
            "features": list(self.features),
            "label": self.label,
            "domain": self.domain,
        }


@dataclass(frozen=True, slots=True)
class InteractionContext:
    """The complete action context visible to and capable of affecting the user."""

    context_id: str
    options: tuple[Option, ...]
    ranking: tuple[str, ...]
    domain: str = ""
    scenario_id: str = ""
    turn_id: str = ""
    default_option_id: str | None = None
    suggested_option_id: str | None = None
    wording_template: str = "neutral_choice"
    question_type: str = "choice"
    target_attribute: int | None = None
    prompt: str | None = None

    def __post_init__(self) -> None:
        _require_nonempty_string(self.context_id, "context_id")
        options = tuple(self.options)
        if not options:
            raise ValueError("options cannot be empty")
        if not all(isinstance(option, Option) for option in options):
            raise TypeError("options must contain only Option objects")
        object.__setattr__(self, "options", options)

        option_ids = tuple(option.option_id for option in options)
        if len(set(option_ids)) != len(option_ids):
            raise ValueError("option IDs must be unique within a context")

        ranking = tuple(self.ranking)
        if len(ranking) != len(option_ids) or set(ranking) != set(option_ids):
            raise ValueError("ranking must be an exact permutation of option IDs")
        object.__setattr__(self, "ranking", ranking)

        for field_name in ("domain", "scenario_id", "turn_id"):
            if not isinstance(getattr(self, field_name), str):
                raise TypeError(f"{field_name} must be a string")
        for field_name in ("wording_template", "question_type"):
            _require_nonempty_string(getattr(self, field_name), field_name)
        if self.prompt is not None:
            _require_nonempty_string(self.prompt, "prompt")
            if len(self.prompt) > 500:
                raise ValueError("prompt must contain at most 500 characters")
            if any(character in self.prompt for character in ("\x00", "\r", "\n")):
                raise ValueError(
                    "prompt must be one line without control characters"
                )

        for field_name in ("default_option_id", "suggested_option_id"):
            value = getattr(self, field_name)
            if value is not None and value not in option_ids:
                raise ValueError(f"{field_name} must name a displayed option or be None")

        if self.target_attribute is not None:
            if (
                isinstance(self.target_attribute, bool)
                or not isinstance(self.target_attribute, int)
                or not 0 <= self.target_attribute < NUM_ATTRIBUTES
            ):
                raise ValueError(
                    f"target_attribute must be in [0, {NUM_ATTRIBUTES}) or None"
                )

    @property
    def option_ids(self) -> tuple[str, ...]:
        return tuple(option.option_id for option in self.options)

    @property
    def default(self) -> str | None:
        return self.default_option_id

    @property
    def suggested_option(self) -> str | None:
        return self.suggested_option_id

    def option(self, option_id: str) -> Option:
        for candidate in self.options:
            if candidate.option_id == option_id:
                return candidate
        raise KeyError(option_id)

    def rank(self, option_id: str) -> int:
        try:
            return self.ranking.index(option_id)
        except ValueError as exc:
            raise KeyError(option_id) from exc

    def to_dict(self) -> dict[str, Any]:
        result = {
            "context_id": self.context_id,
            "domain": self.domain,
            "scenario_id": self.scenario_id,
            "turn_id": self.turn_id,
            "options": [option.to_dict() for option in self.options],
            "ranking": list(self.ranking),
            "default": self.default_option_id,
            "suggested_option": self.suggested_option_id,
            "wording_template": self.wording_template,
            "question_type": self.question_type,
            "target_attribute": self.target_attribute,
        }
        if self.prompt is not None:
            result["prompt"] = self.prompt
        return result


def _canonical_snapshot(
    value: Mapping[str, float] | Sequence[tuple[str, float]],
    name: str,
) -> tuple[tuple[str, float], ...]:
    pairs = value.items() if isinstance(value, Mapping) else value
    result: list[tuple[str, float]] = []
    seen: set[str] = set()
    for key, raw_value in pairs:
        _require_nonempty_string(key, f"{name} key")
        if key in seen:
            raise ValueError(f"{name} keys must be unique")
        seen.add(key)
        result.append((key, _finite_float(raw_value, f"{name}[{key!r}]")))
    return tuple(sorted(result))


@dataclass(frozen=True, slots=True)
class PolicyProvenance:
    """Internal record of why a context was produced.

    This is intentionally not embedded in :class:`InteractionContext`.
    """

    policy_id: str
    policy_version: str
    profile_snapshot: tuple[tuple[str, float], ...] = ()
    random_seed: int = 0
    config_digest: str = ""
    presentation_mechanism: str = "none"
    profile_conditioned: bool = False

    def __post_init__(self) -> None:
        _require_nonempty_string(self.policy_id, "policy_id")
        _require_nonempty_string(self.policy_version, "policy_version")
        snapshot = _canonical_snapshot(self.profile_snapshot, "profile_snapshot")
        object.__setattr__(self, "profile_snapshot", snapshot)
        if isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int):
            raise TypeError("random_seed must be an integer")
        if not isinstance(self.config_digest, str):
            raise TypeError("config_digest must be a string")
        allowed_mechanisms = {
            "none",
            "balanced",
            "ranking",
            "default",
            "suggestion",
            "restriction",
            "target_selection",
        }
        if self.presentation_mechanism not in allowed_mechanisms:
            raise ValueError(
                "presentation_mechanism must be one of "
                f"{sorted(allowed_mechanisms)}"
            )
        if not isinstance(self.profile_conditioned, bool):
            raise TypeError("profile_conditioned must be a Boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "profile_snapshot": dict(self.profile_snapshot),
            "random_seed": self.random_seed,
            "config_digest": self.config_digest,
            "presentation_mechanism": self.presentation_mechanism,
            "profile_conditioned": self.profile_conditioned,
        }


@dataclass(frozen=True, slots=True)
class Observation:
    """A structured choice sampled before any optional language verbalization."""

    selected_option_id: str
    surface_response: str | None = None
    choice_noise_key: str = ""
    assistant_message: str | None = None
    surface_id: str = ""

    def __post_init__(self) -> None:
        _require_nonempty_string(self.selected_option_id, "selected_option_id")
        if self.surface_response is not None and not isinstance(
            self.surface_response, str
        ):
            raise TypeError("surface_response must be a string or None")
        if not isinstance(self.choice_noise_key, str):
            raise TypeError("choice_noise_key must be a string")
        if self.assistant_message is not None:
            _require_nonempty_string(
                self.assistant_message,
                "assistant_message",
            )
            if self.surface_response is None or not self.surface_response.strip():
                raise ValueError(
                    "assistant_message requires a non-empty surface_response"
                )
            if not self.surface_id:
                raise ValueError("assistant_message requires surface_id")
        if not isinstance(self.surface_id, str):
            raise TypeError("surface_id must be a string")
        if self.surface_id and self.assistant_message is None:
            raise ValueError("surface_id requires assistant_message")
        for field_name in ("assistant_message", "surface_response"):
            value = getattr(self, field_name)
            if value is not None and (
                len(value) > 4000
                or any(character in value for character in ("\x00", "\r"))
            ):
                raise ValueError(
                    f"{field_name} must contain at most 4000 characters "
                    "without NUL or carriage returns"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_option": self.selected_option_id,
            "surface_response": self.surface_response,
            "choice_noise_key": self.choice_noise_key,
            "assistant_message": self.assistant_message,
            "surface_id": self.surface_id,
        }


def _probability_snapshot(
    values: Sequence[float],
    name: str,
) -> tuple[float, ...]:
    result = tuple(
        _finite_float(value, f"{name}[{index}]")
        for index, value in enumerate(values)
    )
    if any(value < 0.0 for value in result):
        raise ValueError(f"{name} cannot contain negative values")
    return result


@dataclass(frozen=True, slots=True)
class ProfileUpdate:
    """Serializable update record without coupling schemas to a belief class."""

    updater_id: str
    belief_before: tuple[float, ...] = ()
    belief_after: tuple[float, ...] = ()
    native_memory_before: tuple[str, ...] = ()
    native_memory_after: tuple[str, ...] = ()
    written_delta: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_nonempty_string(self.updater_id, "updater_id")
        object.__setattr__(
            self,
            "belief_before",
            _probability_snapshot(self.belief_before, "belief_before"),
        )
        object.__setattr__(
            self,
            "belief_after",
            _probability_snapshot(self.belief_after, "belief_after"),
        )
        for field_name in (
            "native_memory_before",
            "native_memory_after",
            "written_delta",
        ):
            value = tuple(getattr(self, field_name))
            if not all(isinstance(item, str) for item in value):
                raise TypeError(f"{field_name} must contain only strings")
            object.__setattr__(self, field_name, value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "updater_id": self.updater_id,
            "belief_before": list(self.belief_before),
            "belief_after": list(self.belief_after),
            "native_memory_before": list(self.native_memory_before),
            "native_memory_after": list(self.native_memory_after),
            "written_delta": list(self.written_delta),
        }


@dataclass(frozen=True, slots=True)
class InteractionRecord:
    """One auditable policy -> context -> observation -> update chain."""

    record_id: str
    context: InteractionContext
    provenance: PolicyProvenance
    observation: Observation
    profile_update: ProfileUpdate | None = None

    def __post_init__(self) -> None:
        _require_nonempty_string(self.record_id, "record_id")
        if not isinstance(self.context, InteractionContext):
            raise TypeError("context must be an InteractionContext")
        if not isinstance(self.provenance, PolicyProvenance):
            raise TypeError("provenance must be a PolicyProvenance")
        if not isinstance(self.observation, Observation):
            raise TypeError("observation must be an Observation")
        if self.observation.selected_option_id not in self.context.option_ids:
            raise ValueError("observation must select a displayed option")
        if self.profile_update is not None and not isinstance(
            self.profile_update, ProfileUpdate
        ):
            raise TypeError("profile_update must be a ProfileUpdate or None")

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "context": self.context.to_dict(),
            "policy_provenance": self.provenance.to_dict(),
            "observation": self.observation.to_dict(),
            "profile_update": (
                None if self.profile_update is None else self.profile_update.to_dict()
            ),
        }


@dataclass(frozen=True, slots=True)
class TrajectoryRecord:
    """A sequence of auditable interactions, excluding latent truth by design."""

    trajectory_id: str
    user_id: str
    domain: str
    interactions: tuple[InteractionRecord, ...]

    def __post_init__(self) -> None:
        _require_nonempty_string(self.trajectory_id, "trajectory_id")
        _require_nonempty_string(self.user_id, "user_id")
        _require_nonempty_string(self.domain, "domain")
        interactions = tuple(self.interactions)
        if not all(isinstance(item, InteractionRecord) for item in interactions):
            raise TypeError("interactions must contain InteractionRecord objects")
        record_ids = tuple(item.record_id for item in interactions)
        if len(set(record_ids)) != len(record_ids):
            raise ValueError("record IDs must be unique within a trajectory")
        object.__setattr__(self, "interactions", interactions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "user_id": self.user_id,
            "domain": self.domain,
            "interactions": [item.to_dict() for item in self.interactions],
        }
