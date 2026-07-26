"""Stage-gated, exactly paired correction-debt experiment infrastructure.

The runner compares an incorrect seed with an equally strong correct-seed
control under the same reinforcement schedule, explicit correction, and
balanced-recovery horizon.  Pair-level deltas are retained before aggregation;
the independent unit is therefore never an individual turn.

The shipped adapter is a transparent diagnostic reference over declared log
odds.  It demonstrates and tests the protocol without pretending to be an LLM
or native-memory result.  A real system can implement :class:`CorrectionAdapter`
and use the same runner and summaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from statistics import mean
from typing import Any, Iterable, Mapping, Protocol
import json
import math


SEED_CONDITIONS = ("false", "correct")


def _require_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _digest(value: Any) -> str:
    return sha256(_canonical(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class CorrectionStage:
    """One proposal-defined placement of the explicit correction."""

    stage_id: str
    reinforcing_interactions: int
    consolidate_before_correction: bool = False

    def __post_init__(self) -> None:
        _require_text(self.stage_id, "stage_id")
        if (
            isinstance(self.reinforcing_interactions, bool)
            or not isinstance(self.reinforcing_interactions, int)
            or self.reinforcing_interactions < 0
        ):
            raise ValueError(
                "reinforcing_interactions must be a non-negative integer"
            )
        if not isinstance(self.consolidate_before_correction, bool):
            raise TypeError("consolidate_before_correction must be Boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "reinforcing_interactions": self.reinforcing_interactions,
            "consolidate_before_correction": (
                self.consolidate_before_correction
            ),
        }


def default_correction_stages() -> tuple[CorrectionStage, ...]:
    return (
        CorrectionStage("before-reinforcement", 0),
        CorrectionStage("after-one-reinforcement", 1),
        CorrectionStage("after-repeated-reinforcement", 3),
        CorrectionStage(
            "after-recurrent-consolidation",
            3,
            consolidate_before_correction=True,
        ),
    )


@dataclass(frozen=True, slots=True)
class CorrectionProtocol:
    stages: tuple[CorrectionStage, ...] = default_correction_stages()
    max_balanced_turns: int = 8
    recovery_wrong_mass_threshold: float = 0.25
    seed_log_odds: float = 2.0
    reinforcement_evidence: float = 0.65
    explicit_correction_evidence: float = 1.75
    balanced_evidence_per_turn: float = 0.55
    consolidation_multiplier: float = 1.25
    protocol_version: str = "correction-debt-v1"

    def __post_init__(self) -> None:
        stages = tuple(self.stages)
        if not stages:
            raise ValueError("correction protocol requires at least one stage")
        if len({stage.stage_id for stage in stages}) != len(stages):
            raise ValueError("correction stage IDs must be unique")
        object.__setattr__(self, "stages", stages)
        if (
            isinstance(self.max_balanced_turns, bool)
            or not isinstance(self.max_balanced_turns, int)
            or self.max_balanced_turns <= 0
        ):
            raise ValueError("max_balanced_turns must be positive")
        threshold = _finite(
            self.recovery_wrong_mass_threshold,
            "recovery_wrong_mass_threshold",
        )
        if not 0.0 < threshold < 0.5:
            raise ValueError(
                "recovery_wrong_mass_threshold must lie strictly in (0, 0.5)"
            )
        object.__setattr__(
            self, "recovery_wrong_mass_threshold", threshold
        )
        for name in (
            "seed_log_odds",
            "reinforcement_evidence",
            "explicit_correction_evidence",
            "balanced_evidence_per_turn",
        ):
            value = _finite(getattr(self, name), name)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        multiplier = _finite(
            self.consolidation_multiplier, "consolidation_multiplier"
        )
        if multiplier < 1.0:
            raise ValueError("consolidation_multiplier must be at least one")
        object.__setattr__(self, "consolidation_multiplier", multiplier)
        _require_text(self.protocol_version, "protocol_version")

    @property
    def protocol_sha256(self) -> str:
        return _digest(self.to_dict(include_digest=False))

    def to_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": 1,
            "protocol_version": self.protocol_version,
            "stages": [stage.to_dict() for stage in self.stages],
            "max_balanced_turns": self.max_balanced_turns,
            "recovery_wrong_mass_threshold": (
                self.recovery_wrong_mass_threshold
            ),
            "seed_log_odds": self.seed_log_odds,
            "reinforcement_evidence": self.reinforcement_evidence,
            "explicit_correction_evidence": (
                self.explicit_correction_evidence
            ),
            "balanced_evidence_per_turn": self.balanced_evidence_per_turn,
            "consolidation_multiplier": self.consolidation_multiplier,
        }
        if include_digest:
            result["protocol_sha256"] = self.protocol_sha256
        return result


@dataclass(frozen=True, slots=True)
class CorrectionMeasurement:
    """System outputs needed for profile, text, behavior, and memory recovery."""

    wrong_profile_mass: float
    textual_direction: int
    behavioral_direction: int
    wrong_derived_memory_count: int
    state_sha256: str

    def __post_init__(self) -> None:
        mass = _finite(self.wrong_profile_mass, "wrong_profile_mass")
        if not 0.0 <= mass <= 1.0:
            raise ValueError("wrong_profile_mass must lie in [0, 1]")
        object.__setattr__(self, "wrong_profile_mass", mass)
        if self.textual_direction not in (-1, 1):
            raise ValueError("textual_direction must be -1 or +1")
        if self.behavioral_direction not in (-1, 1):
            raise ValueError("behavioral_direction must be -1 or +1")
        if (
            isinstance(self.wrong_derived_memory_count, bool)
            or not isinstance(self.wrong_derived_memory_count, int)
            or self.wrong_derived_memory_count < 0
        ):
            raise ValueError(
                "wrong_derived_memory_count must be a non-negative integer"
            )
        if (
            not isinstance(self.state_sha256, str)
            or len(self.state_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.state_sha256
            )
        ):
            raise ValueError("state_sha256 must be a lowercase SHA-256 digest")

    def to_dict(self) -> dict[str, Any]:
        return {
            "wrong_profile_mass": self.wrong_profile_mass,
            "textual_direction": self.textual_direction,
            "behavioral_direction": self.behavioral_direction,
            "wrong_derived_memory_count": self.wrong_derived_memory_count,
            "state_sha256": self.state_sha256,
        }


class CorrectionAdapter(Protocol):
    """Minimal adapter required to run a native or structured system."""

    adapter_id: str

    def initialize(
        self,
        *,
        pair_id: str,
        truth_direction: int,
        seed_condition: str,
        protocol: CorrectionProtocol,
    ) -> object: ...

    def reinforce(
        self,
        state: object,
        *,
        evidence_direction: int,
        evidence_strength: float,
        event_key: str,
    ) -> object: ...

    def consolidate(
        self,
        state: object,
        *,
        multiplier: float,
        event_key: str,
    ) -> object: ...

    def apply_explicit_correction(
        self,
        state: object,
        *,
        truth_direction: int,
        evidence_strength: float,
        correction_text: str,
        event_key: str,
    ) -> object: ...

    def balanced_recovery_step(
        self,
        state: object,
        *,
        truth_direction: int,
        evidence_strength: float,
        turn: int,
        common_evidence_key: str,
    ) -> object: ...

    def measure(
        self,
        state: object,
        *,
        truth_direction: int,
    ) -> CorrectionMeasurement: ...


@dataclass(frozen=True, slots=True)
class ReferenceCorrectionState:
    """Transparent state for the diagnostic reference adapter."""

    truth_direction: int
    profile_log_odds_correct: float
    textual_log_odds_correct: float
    behavioral_log_odds_correct: float
    derived_memory_directions: tuple[int, ...]
    event_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.truth_direction not in (-1, 1):
            raise ValueError("truth_direction must be -1 or +1")
        for name in (
            "profile_log_odds_correct",
            "textual_log_odds_correct",
            "behavioral_log_odds_correct",
        ):
            object.__setattr__(
                self, name, _finite(getattr(self, name), name)
            )
        if any(
            direction not in (-1, 1)
            for direction in self.derived_memory_directions
        ):
            raise ValueError("derived memory directions must be -1 or +1")
        if any(not isinstance(key, str) or not key for key in self.event_keys):
            raise ValueError("event keys must be nonempty strings")

    @property
    def state_sha256(self) -> str:
        return _digest(
            {
                "truth_direction": self.truth_direction,
                "profile_log_odds_correct": self.profile_log_odds_correct,
                "textual_log_odds_correct": self.textual_log_odds_correct,
                "behavioral_log_odds_correct": (
                    self.behavioral_log_odds_correct
                ),
                "derived_memory_directions": (
                    self.derived_memory_directions
                ),
                "event_keys": self.event_keys,
            }
        )


class ReferenceLogOddsCorrectionAdapter:
    """Declared, inspectable protocol reference; not an empirical model."""

    adapter_id = "reference_log_odds_correction_v1"

    @staticmethod
    def _require_state(state: object) -> ReferenceCorrectionState:
        if not isinstance(state, ReferenceCorrectionState):
            raise TypeError("reference adapter received an incompatible state")
        return state

    def initialize(
        self,
        *,
        pair_id: str,
        truth_direction: int,
        seed_condition: str,
        protocol: CorrectionProtocol,
    ) -> ReferenceCorrectionState:
        _require_text(pair_id, "pair_id")
        if truth_direction not in (-1, 1):
            raise ValueError("truth_direction must be -1 or +1")
        if seed_condition not in SEED_CONDITIONS:
            raise ValueError(f"seed_condition must be one of {SEED_CONDITIONS}")
        correct_sign = 1.0 if seed_condition == "correct" else -1.0
        seed_direction = (
            truth_direction if seed_condition == "correct" else -truth_direction
        )
        strength = protocol.seed_log_odds * correct_sign
        return ReferenceCorrectionState(
            truth_direction=truth_direction,
            profile_log_odds_correct=strength,
            textual_log_odds_correct=1.10 * strength,
            behavioral_log_odds_correct=0.85 * strength,
            derived_memory_directions=(seed_direction,),
            event_keys=(f"seed:{pair_id}:{seed_condition}",),
        )

    @staticmethod
    def _signed_evidence(
        state: ReferenceCorrectionState,
        evidence_direction: int,
        evidence_strength: float,
    ) -> float:
        if evidence_direction not in (-1, 1):
            raise ValueError("evidence_direction must be -1 or +1")
        strength = _finite(evidence_strength, "evidence_strength")
        if strength <= 0.0:
            raise ValueError("evidence_strength must be positive")
        return (
            strength
            if evidence_direction == state.truth_direction
            else -strength
        )

    def reinforce(
        self,
        state: object,
        *,
        evidence_direction: int,
        evidence_strength: float,
        event_key: str,
    ) -> ReferenceCorrectionState:
        current = self._require_state(state)
        _require_text(event_key, "event_key")
        signed = self._signed_evidence(
            current, evidence_direction, evidence_strength
        )
        return ReferenceCorrectionState(
            truth_direction=current.truth_direction,
            profile_log_odds_correct=(
                current.profile_log_odds_correct + signed
            ),
            textual_log_odds_correct=(
                current.textual_log_odds_correct + 1.15 * signed
            ),
            behavioral_log_odds_correct=(
                current.behavioral_log_odds_correct + 0.85 * signed
            ),
            derived_memory_directions=(
                current.derived_memory_directions + (evidence_direction,)
            ),
            event_keys=current.event_keys + (event_key,),
        )

    def consolidate(
        self,
        state: object,
        *,
        multiplier: float,
        event_key: str,
    ) -> ReferenceCorrectionState:
        current = self._require_state(state)
        factor = _finite(multiplier, "multiplier")
        if factor < 1.0:
            raise ValueError("consolidation multiplier must be at least one")
        _require_text(event_key, "event_key")
        return ReferenceCorrectionState(
            truth_direction=current.truth_direction,
            profile_log_odds_correct=(
                current.profile_log_odds_correct * factor
            ),
            textual_log_odds_correct=(
                current.textual_log_odds_correct * factor
            ),
            behavioral_log_odds_correct=(
                current.behavioral_log_odds_correct * factor
            ),
            derived_memory_directions=current.derived_memory_directions,
            event_keys=current.event_keys + (event_key,),
        )

    def apply_explicit_correction(
        self,
        state: object,
        *,
        truth_direction: int,
        evidence_strength: float,
        correction_text: str,
        event_key: str,
    ) -> ReferenceCorrectionState:
        current = self._require_state(state)
        if truth_direction != current.truth_direction:
            raise ValueError("explicit correction contradicts retained truth")
        _require_text(correction_text, "correction_text")
        _require_text(event_key, "event_key")
        signed = self._signed_evidence(
            current, truth_direction, evidence_strength
        )
        return ReferenceCorrectionState(
            truth_direction=current.truth_direction,
            profile_log_odds_correct=(
                current.profile_log_odds_correct + signed
            ),
            textual_log_odds_correct=(
                current.textual_log_odds_correct + 1.30 * signed
            ),
            behavioral_log_odds_correct=(
                current.behavioral_log_odds_correct + 0.90 * signed
            ),
            # Explicit correction is retained without deleting the causal
            # history. Persistence of superseded wrong memories is measurable.
            derived_memory_directions=(
                current.derived_memory_directions + (truth_direction,)
            ),
            event_keys=current.event_keys + (event_key,),
        )

    def balanced_recovery_step(
        self,
        state: object,
        *,
        truth_direction: int,
        evidence_strength: float,
        turn: int,
        common_evidence_key: str,
    ) -> ReferenceCorrectionState:
        current = self._require_state(state)
        if truth_direction != current.truth_direction:
            raise ValueError("balanced evidence contradicts retained truth")
        if isinstance(turn, bool) or not isinstance(turn, int) or turn <= 0:
            raise ValueError("balanced recovery turn must be positive")
        _require_text(common_evidence_key, "common_evidence_key")
        signed = self._signed_evidence(
            current, truth_direction, evidence_strength
        )
        return ReferenceCorrectionState(
            truth_direction=current.truth_direction,
            profile_log_odds_correct=(
                current.profile_log_odds_correct + signed
            ),
            textual_log_odds_correct=(
                current.textual_log_odds_correct + 0.90 * signed
            ),
            behavioral_log_odds_correct=(
                current.behavioral_log_odds_correct + 1.05 * signed
            ),
            derived_memory_directions=current.derived_memory_directions,
            event_keys=current.event_keys + (common_evidence_key,),
        )

    def measure(
        self,
        state: object,
        *,
        truth_direction: int,
    ) -> CorrectionMeasurement:
        current = self._require_state(state)
        if truth_direction != current.truth_direction:
            raise ValueError("measurement truth disagrees with state")
        log_odds = current.profile_log_odds_correct
        if log_odds >= 0.0:
            wrong_mass = 1.0 / (1.0 + math.exp(log_odds))
        else:
            exp_value = math.exp(log_odds)
            wrong_mass = 1.0 / (1.0 + exp_value)
        textual = truth_direction if current.textual_log_odds_correct >= 0 else -truth_direction
        behavioral = (
            truth_direction
            if current.behavioral_log_odds_correct >= 0
            else -truth_direction
        )
        wrong_memories = sum(
            direction != truth_direction
            for direction in current.derived_memory_directions
        )
        return CorrectionMeasurement(
            wrong_profile_mass=wrong_mass,
            textual_direction=textual,
            behavioral_direction=behavioral,
            wrong_derived_memory_count=wrong_memories,
            state_sha256=current.state_sha256,
        )


@dataclass(frozen=True, slots=True)
class CorrectionSnapshot:
    turn: int
    phase: str
    cumulative_corrective_evidence: float
    common_evidence_key: str
    measurement: CorrectionMeasurement

    def __post_init__(self) -> None:
        if (
            isinstance(self.turn, bool)
            or not isinstance(self.turn, int)
            or self.turn < 0
        ):
            raise ValueError("snapshot turn must be non-negative")
        if self.phase not in {"after_explicit_correction", "balanced_recovery"}:
            raise ValueError("unknown correction snapshot phase")
        evidence = _finite(
            self.cumulative_corrective_evidence,
            "cumulative_corrective_evidence",
        )
        if evidence <= 0.0:
            raise ValueError("cumulative corrective evidence must be positive")
        object.__setattr__(
            self, "cumulative_corrective_evidence", evidence
        )
        _require_text(self.common_evidence_key, "common_evidence_key")

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn": self.turn,
            "phase": self.phase,
            "cumulative_corrective_evidence": (
                self.cumulative_corrective_evidence
            ),
            "common_evidence_key": self.common_evidence_key,
            "measurement": self.measurement.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class CorrectionArmResult:
    pair_id: str
    stage_id: str
    seed_condition: str
    truth_direction: int
    adapter_id: str
    snapshots: tuple[CorrectionSnapshot, ...]
    recovery_wrong_mass_threshold: float
    profile_recovery_turn: int | None
    corrective_evidence_to_recovery: float | None
    textual_recovery_turn: int | None
    behavioral_recovery_turn: int | None
    recovery_error_auc: float
    terminal_wrong_profile_mass: float
    terminal_wrong_derived_memory_count: int

    def __post_init__(self) -> None:
        _require_text(self.pair_id, "pair_id")
        _require_text(self.stage_id, "stage_id")
        _require_text(self.adapter_id, "adapter_id")
        if self.seed_condition not in SEED_CONDITIONS:
            raise ValueError(f"seed_condition must be one of {SEED_CONDITIONS}")
        if self.truth_direction not in (-1, 1):
            raise ValueError("truth_direction must be -1 or +1")
        snapshots = tuple(self.snapshots)
        if not snapshots or snapshots[0].turn != 0:
            raise ValueError("correction arm must begin with turn-zero snapshot")
        if tuple(item.turn for item in snapshots) != tuple(range(len(snapshots))):
            raise ValueError("correction snapshots must use contiguous turns")
        if any(
            later.cumulative_corrective_evidence
            <= earlier.cumulative_corrective_evidence
            for earlier, later in zip(snapshots, snapshots[1:])
        ):
            raise ValueError("corrective evidence must increase every turn")
        object.__setattr__(self, "snapshots", snapshots)

    @property
    def recovery_censored(self) -> bool:
        return self.profile_recovery_turn is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "stage_id": self.stage_id,
            "seed_condition": self.seed_condition,
            "truth_direction": self.truth_direction,
            "adapter_id": self.adapter_id,
            "recovery_wrong_mass_threshold": (
                self.recovery_wrong_mass_threshold
            ),
            "profile_recovery_turn": self.profile_recovery_turn,
            "corrective_evidence_to_recovery": (
                self.corrective_evidence_to_recovery
            ),
            "textual_recovery_turn": self.textual_recovery_turn,
            "behavioral_recovery_turn": self.behavioral_recovery_turn,
            "recovery_error_auc": self.recovery_error_auc,
            "terminal_wrong_profile_mass": self.terminal_wrong_profile_mass,
            "terminal_wrong_derived_memory_count": (
                self.terminal_wrong_derived_memory_count
            ),
            "recovery_censored": self.recovery_censored,
            "snapshots": [item.to_dict() for item in self.snapshots],
        }


def _first_turn(
    snapshots: tuple[CorrectionSnapshot, ...],
    predicate: Any,
) -> int | None:
    return next(
        (snapshot.turn for snapshot in snapshots if predicate(snapshot)),
        None,
    )


def _run_arm(
    *,
    pair_id: str,
    truth_direction: int,
    seed_condition: str,
    stage: CorrectionStage,
    protocol: CorrectionProtocol,
    adapter: CorrectionAdapter,
    correction_text: str,
) -> CorrectionArmResult:
    state = adapter.initialize(
        pair_id=pair_id,
        truth_direction=truth_direction,
        seed_condition=seed_condition,
        protocol=protocol,
    )
    seed_direction = (
        truth_direction if seed_condition == "correct" else -truth_direction
    )
    for index in range(stage.reinforcing_interactions):
        state = adapter.reinforce(
            state,
            evidence_direction=seed_direction,
            evidence_strength=protocol.reinforcement_evidence,
            event_key=(
                f"reinforcement:{pair_id}:{stage.stage_id}:"
                f"{seed_condition}:{index}"
            ),
        )
    if stage.consolidate_before_correction:
        state = adapter.consolidate(
            state,
            multiplier=protocol.consolidation_multiplier,
            event_key=(
                f"consolidation:{pair_id}:{stage.stage_id}:"
                f"{seed_condition}"
            ),
        )
    correction_key = f"correction:{pair_id}:{stage.stage_id}"
    state = adapter.apply_explicit_correction(
        state,
        truth_direction=truth_direction,
        evidence_strength=protocol.explicit_correction_evidence,
        correction_text=correction_text,
        event_key=correction_key,
    )
    cumulative_evidence = protocol.explicit_correction_evidence
    snapshots = [
        CorrectionSnapshot(
            turn=0,
            phase="after_explicit_correction",
            cumulative_corrective_evidence=cumulative_evidence,
            common_evidence_key=correction_key,
            measurement=adapter.measure(
                state, truth_direction=truth_direction
            ),
        )
    ]
    for turn in range(1, protocol.max_balanced_turns + 1):
        # The evidence key intentionally omits seed condition: both arms receive
        # the same semantic recovery event and any adapter randomness can use it
        # as a common-random-number key.
        common_key = (
            f"balanced-recovery:{pair_id}:{stage.stage_id}:turn-{turn}"
        )
        state = adapter.balanced_recovery_step(
            state,
            truth_direction=truth_direction,
            evidence_strength=protocol.balanced_evidence_per_turn,
            turn=turn,
            common_evidence_key=common_key,
        )
        cumulative_evidence += protocol.balanced_evidence_per_turn
        snapshots.append(
            CorrectionSnapshot(
                turn=turn,
                phase="balanced_recovery",
                cumulative_corrective_evidence=cumulative_evidence,
                common_evidence_key=common_key,
                measurement=adapter.measure(
                    state, truth_direction=truth_direction
                ),
            )
        )
    retained = tuple(snapshots)
    threshold = protocol.recovery_wrong_mass_threshold
    recovery_turn = _first_turn(
        retained,
        lambda snapshot: (
            snapshot.measurement.wrong_profile_mass <= threshold
        ),
    )
    evidence_to_recovery = next(
        (
            snapshot.cumulative_corrective_evidence
            for snapshot in retained
            if snapshot.turn == recovery_turn
        ),
        None,
    )
    textual_turn = _first_turn(
        retained,
        lambda snapshot: (
            snapshot.measurement.textual_direction == truth_direction
        ),
    )
    behavioral_turn = _first_turn(
        retained,
        lambda snapshot: (
            snapshot.measurement.behavioral_direction == truth_direction
        ),
    )
    auc = math.fsum(
        (
            first.measurement.wrong_profile_mass
            + second.measurement.wrong_profile_mass
        )
        / 2.0
        * (second.turn - first.turn)
        for first, second in zip(retained, retained[1:])
    )
    terminal = retained[-1].measurement
    return CorrectionArmResult(
        pair_id=pair_id,
        stage_id=stage.stage_id,
        seed_condition=seed_condition,
        truth_direction=truth_direction,
        adapter_id=adapter.adapter_id,
        snapshots=retained,
        recovery_wrong_mass_threshold=threshold,
        profile_recovery_turn=recovery_turn,
        corrective_evidence_to_recovery=evidence_to_recovery,
        textual_recovery_turn=textual_turn,
        behavioral_recovery_turn=behavioral_turn,
        recovery_error_auc=auc,
        terminal_wrong_profile_mass=terminal.wrong_profile_mass,
        terminal_wrong_derived_memory_count=(
            terminal.wrong_derived_memory_count
        ),
    )


@dataclass(frozen=True, slots=True)
class CorrectionDebtPair:
    pair_id: str
    stage_id: str
    false_seed_recovery_turn: int | None
    correct_seed_recovery_turn: int | None
    recovery_turn_debt: float | None
    corrective_evidence_debt: float | None
    recovery_error_auc_debt: float
    terminal_profile_error_debt: float
    textual_recovery_turn_debt: float | None
    behavioral_recovery_turn_debt: float | None
    persistent_wrong_memory_debt: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "stage_id": self.stage_id,
            "false_seed_recovery_turn": self.false_seed_recovery_turn,
            "correct_seed_recovery_turn": self.correct_seed_recovery_turn,
            "recovery_turn_debt": self.recovery_turn_debt,
            "corrective_evidence_debt": self.corrective_evidence_debt,
            "recovery_error_auc_debt": self.recovery_error_auc_debt,
            "terminal_profile_error_debt": self.terminal_profile_error_debt,
            "textual_recovery_turn_debt": self.textual_recovery_turn_debt,
            "behavioral_recovery_turn_debt": (
                self.behavioral_recovery_turn_debt
            ),
            "persistent_wrong_memory_debt": (
                self.persistent_wrong_memory_debt
            ),
        }


def _nullable_delta(
    false_value: int | float | None,
    correct_value: int | float | None,
) -> float | None:
    if false_value is None or correct_value is None:
        return None
    return float(false_value) - float(correct_value)


@dataclass(frozen=True, slots=True)
class CorrectionDebtStageSummary:
    stage_id: str
    paired_unit_count: int
    recovery_censored_pair_count: int
    mean_recovery_turn_debt: float | None
    mean_corrective_evidence_debt: float | None
    mean_recovery_error_auc_debt: float
    mean_terminal_profile_error_debt: float
    mean_textual_recovery_turn_debt: float | None
    mean_behavioral_recovery_turn_debt: float | None
    mean_persistent_wrong_memory_debt: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "paired_unit_count": self.paired_unit_count,
            "recovery_censored_pair_count": (
                self.recovery_censored_pair_count
            ),
            "mean_recovery_turn_debt": self.mean_recovery_turn_debt,
            "mean_corrective_evidence_debt": (
                self.mean_corrective_evidence_debt
            ),
            "mean_recovery_error_auc_debt": (
                self.mean_recovery_error_auc_debt
            ),
            "mean_terminal_profile_error_debt": (
                self.mean_terminal_profile_error_debt
            ),
            "mean_textual_recovery_turn_debt": (
                self.mean_textual_recovery_turn_debt
            ),
            "mean_behavioral_recovery_turn_debt": (
                self.mean_behavioral_recovery_turn_debt
            ),
            "mean_persistent_wrong_memory_debt": (
                self.mean_persistent_wrong_memory_debt
            ),
        }


def _mean_optional(values: Iterable[float | None]) -> float | None:
    retained = tuple(value for value in values if value is not None)
    return None if not retained else mean(retained)


@dataclass(frozen=True, slots=True)
class CorrectionDebtResult:
    protocol: CorrectionProtocol
    adapter_id: str
    correction_text: str
    stage_gate_authorized: bool
    arms: tuple[CorrectionArmResult, ...]
    paired_debts: tuple[CorrectionDebtPair, ...]
    stage_summaries: tuple[CorrectionDebtStageSummary, ...]
    claim_status: str = "not_claimed"
    interpretation_boundary: str = (
        "The reference adapter is a protocol diagnostic. External or native "
        "system runs and frozen empirical review are required for paper claims."
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "experiment": "correction_debt",
            "protocol": self.protocol.to_dict(),
            "adapter_id": self.adapter_id,
            "correction_text": self.correction_text,
            "stage_gate_authorized": self.stage_gate_authorized,
            "claim_status": self.claim_status,
            "interpretation_boundary": self.interpretation_boundary,
            "arms": [item.to_dict() for item in self.arms],
            "paired_debts": [
                item.to_dict() for item in self.paired_debts
            ],
            "stage_summaries": [
                item.to_dict() for item in self.stage_summaries
            ],
        }


def run_correction_debt_experiment(
    *,
    pair_truth_directions: Mapping[str, int],
    stage_gate_authorized: bool,
    protocol: CorrectionProtocol | None = None,
    adapter: CorrectionAdapter | None = None,
    correction_text: str = (
        "I do not generally prefer the earlier profile direction. I chose "
        "those options because of the options and defaults shown to me."
    ),
) -> CorrectionDebtResult:
    """Run all stage × false/correct seed arms with exact semantic pairing."""

    if stage_gate_authorized is not True:
        raise ValueError(
            "correction debt is stage-gated; pass stage_gate_authorized=True "
            "only after the prerequisite evidence has been reviewed"
        )
    _require_text(correction_text, "correction_text")
    if not pair_truth_directions:
        raise ValueError("at least one paired unit is required")
    for pair_id, direction in pair_truth_directions.items():
        _require_text(pair_id, "pair_id")
        if direction not in (-1, 1):
            raise ValueError("pair truth directions must be -1 or +1")
    declared_protocol = protocol or CorrectionProtocol()
    declared_adapter: CorrectionAdapter = (
        adapter or ReferenceLogOddsCorrectionAdapter()
    )
    _require_text(declared_adapter.adapter_id, "adapter.adapter_id")

    arms: list[CorrectionArmResult] = []
    for stage in declared_protocol.stages:
        for pair_id, truth_direction in sorted(
            pair_truth_directions.items()
        ):
            for seed_condition in SEED_CONDITIONS:
                arms.append(
                    _run_arm(
                        pair_id=pair_id,
                        truth_direction=truth_direction,
                        seed_condition=seed_condition,
                        stage=stage,
                        protocol=declared_protocol,
                        adapter=declared_adapter,
                        correction_text=correction_text,
                    )
                )

    arm_lookup = {
        (arm.pair_id, arm.stage_id, arm.seed_condition): arm for arm in arms
    }
    if len(arm_lookup) != len(arms):
        raise AssertionError("correction runner generated duplicate arms")
    paired: list[CorrectionDebtPair] = []
    for stage in declared_protocol.stages:
        for pair_id in sorted(pair_truth_directions):
            false = arm_lookup[(pair_id, stage.stage_id, "false")]
            correct = arm_lookup[(pair_id, stage.stage_id, "correct")]
            if any(
                first.common_evidence_key != second.common_evidence_key
                for first, second in zip(false.snapshots, correct.snapshots)
            ):
                raise AssertionError("correction arms lost exact evidence pairing")
            paired.append(
                CorrectionDebtPair(
                    pair_id=pair_id,
                    stage_id=stage.stage_id,
                    false_seed_recovery_turn=false.profile_recovery_turn,
                    correct_seed_recovery_turn=correct.profile_recovery_turn,
                    recovery_turn_debt=_nullable_delta(
                        false.profile_recovery_turn,
                        correct.profile_recovery_turn,
                    ),
                    corrective_evidence_debt=_nullable_delta(
                        false.corrective_evidence_to_recovery,
                        correct.corrective_evidence_to_recovery,
                    ),
                    recovery_error_auc_debt=(
                        false.recovery_error_auc - correct.recovery_error_auc
                    ),
                    terminal_profile_error_debt=(
                        false.terminal_wrong_profile_mass
                        - correct.terminal_wrong_profile_mass
                    ),
                    textual_recovery_turn_debt=_nullable_delta(
                        false.textual_recovery_turn,
                        correct.textual_recovery_turn,
                    ),
                    behavioral_recovery_turn_debt=_nullable_delta(
                        false.behavioral_recovery_turn,
                        correct.behavioral_recovery_turn,
                    ),
                    persistent_wrong_memory_debt=(
                        false.terminal_wrong_derived_memory_count
                        - correct.terminal_wrong_derived_memory_count
                    ),
                )
            )

    summaries: list[CorrectionDebtStageSummary] = []
    for stage in declared_protocol.stages:
        rows = [row for row in paired if row.stage_id == stage.stage_id]
        summaries.append(
            CorrectionDebtStageSummary(
                stage_id=stage.stage_id,
                paired_unit_count=len(rows),
                recovery_censored_pair_count=sum(
                    row.recovery_turn_debt is None for row in rows
                ),
                mean_recovery_turn_debt=_mean_optional(
                    row.recovery_turn_debt for row in rows
                ),
                mean_corrective_evidence_debt=_mean_optional(
                    row.corrective_evidence_debt for row in rows
                ),
                mean_recovery_error_auc_debt=mean(
                    row.recovery_error_auc_debt for row in rows
                ),
                mean_terminal_profile_error_debt=mean(
                    row.terminal_profile_error_debt for row in rows
                ),
                mean_textual_recovery_turn_debt=_mean_optional(
                    row.textual_recovery_turn_debt for row in rows
                ),
                mean_behavioral_recovery_turn_debt=_mean_optional(
                    row.behavioral_recovery_turn_debt for row in rows
                ),
                mean_persistent_wrong_memory_debt=mean(
                    row.persistent_wrong_memory_debt for row in rows
                ),
            )
        )
    return CorrectionDebtResult(
        protocol=declared_protocol,
        adapter_id=declared_adapter.adapter_id,
        correction_text=correction_text,
        stage_gate_authorized=True,
        arms=tuple(arms),
        paired_debts=tuple(paired),
        stage_summaries=tuple(summaries),
    )

