"""Immutable Gate 4 review artifacts for externally collected native evidence.

The Experiment B runner deliberately finalizes a run before external decoder
judgments or recorded native-system actions exist.  This module preserves that
boundary: it verifies the completed run, validates separately supplied
evidence, and atomically publishes a new checksum-bound review directory from
a durable same-parent stage.  It never writes inside the source run.

The import is an admission and computation check only.  A successful review
therefore retains ``claim_status = "not_claimed"``.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from itertools import combinations
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence
import json
import math
import os
import shutil
import tempfile

from .artifacts import canonical_json, file_sha256, verify_run
from .decoder_study import (
    DecoderAnalysis,
    DecoderTruthLabel,
    ExternalDecoderJudgment,
    ExternalDecoderRequest,
    analyze_external_decoders,
    validate_external_decoder_import,
)
from .gates import GateCriterion, GateReport
from .file_lock import try_file_lock, unlock_file
from .heldout import (
    HeldOutTerminalItem,
    HeldOutTerminalOption,
    HeldOutTerminalSuite,
    TerminalAction,
    score_heldout_terminal_actions,
)
from .openrouter_decoder_collection import (
    OPENROUTER_COLLECTION_LOCKS,
    is_openrouter_decoder_collection,
    validate_openrouter_decoder_collection,
)
from .schemas import validate_theta


_NATIVE_UPDATERS = frozenset(
    {
        "episodic_memory",
        "semantic_memory",
        "provenance_linked_memory",
    }
)
_UPDATER_MEMORY_KIND = {
    "episodic_memory": "episodic",
    "semantic_memory": "semantic",
    "provenance_linked_memory": "provenance_linked",
}
_RECORDING_ATTESTATION = (
    "actions_emitted_by_named_native_system_not_reference_projection"
)
_ACTION_EXECUTION_MODES = frozenset({"recorded_live", "recorded_replay"})
_ACTION_ADAPTER_KIND = "native_end_to_end_recorded"
_ACTION_EVIDENCE_ORIGIN = "imported_native_system"
_GATE4_NATIVE_MODEL = "gpt-5.6-sol"
_GATE4_NATIVE_REASONING_EFFORT = "medium"
_GATE4_NATIVE_ORIGIN = "https://api.openai.com"
_GATE4_NATIVE_MAX_REQUESTS = 900
_GATE4_NATIVE_MAX_TOTAL_TOKENS = 6_000_000
_GATE4_NATIVE_MAX_OUTPUT_TOKENS = 4_096
_NATIVE_COLLECTION_FILES = {
    "native_collection_plan": "collection-plan.json",
    "native_action_requests": "requests.jsonl",
    "native_transport_attempts": "transport-attempts.jsonl",
    "native_provider_audit": "provider-audit.jsonl",
    "native_terminal_actions": "native-actions.jsonl",
    "native_execution_manifest": "execution-manifest.json",
}
_EXTERNAL_COLLECTION_FILES = {
    "decoder_collection_plan": "collection-plan.json",
    "decoder_transport_attempts": "transport-attempts.jsonl",
    "decoder_provider_audit": "provider-audit.jsonl",
    "decoder_judgments": "judgments.jsonl",
    "decoder_execution_manifest": "execution-manifest.json",
}
_EXTERNAL_COLLECTION_LOCKS = (
    ".external-decoder-command.lock",
    ".external-decoder-collection.lock",
)
_GATE4_EXTERNAL_MAX_REQUESTS = 900
_GATE4_EXTERNAL_MAX_TOTAL_TOKENS = 6_000_000
_GATE4_EXTERNAL_MAX_OUTPUT_TOKENS = 1_024
DIRECT_FIRST_PARTY_COLLECTION_PROVENANCE = (
    "validated_direct_first_party_collection"
)
OPENROUTER_COLLECTION_PROVENANCE = (
    "selected_openrouter_gateway_collection"
)
EXTERNAL_COLLECTION_PROVENANCE_MODES = frozenset(
    {
        DIRECT_FIRST_PARTY_COLLECTION_PROVENANCE,
        OPENROUTER_COLLECTION_PROVENANCE,
    }
)


def _require_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _require_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be Boolean")
    return value


def _validate_digest(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _digest_value(value: Any) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _file_digest(path: Path) -> str:
    return file_sha256(path)


@dataclass(frozen=True, slots=True)
class _FileSnapshot:
    path: Path
    material: bytes

    @property
    def sha256(self) -> str:
        return sha256(self.material).hexdigest()

    def manifest_entry(
        self,
        *,
        record_count: int | None = None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "filename": self.path.name,
            "sha256": self.sha256,
            "bytes": len(self.material),
        }
        if record_count is not None:
            result["record_count"] = record_count
        return result


def _snapshot_file(path: Path, *, name: str) -> _FileSnapshot:
    unresolved = Path(path)
    if unresolved.is_symlink() or not unresolved.is_file():
        raise ValueError(f"{name} must be a safe regular file")
    resolved = unresolved.resolve()
    return _FileSnapshot(resolved, resolved.read_bytes())


def _assert_snapshot_unchanged(
    supplied_path: Path,
    snapshot: _FileSnapshot,
    *,
    name: str,
) -> None:
    """Rebind a supplied path to the exact bytes admitted earlier."""

    try:
        if (
            supplied_path.is_symlink()
            or not supplied_path.is_file()
            or supplied_path.resolve() != snapshot.path
            or snapshot.path.is_symlink()
            or not snapshot.path.is_file()
            or snapshot.path.read_bytes() != snapshot.material
        ):
            raise ValueError(
                f"{name} changed while the Gate 4 import was running"
            )
    except OSError as exc:
        raise ValueError(
            f"{name} changed while the Gate 4 import was running"
        ) from exc


def _json_object_from_snapshot(
    snapshot: _FileSnapshot,
) -> dict[str, Any]:
    try:
        decoded = json.loads(snapshot.material.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{snapshot.path}: invalid JSON: {exc}") from exc
    if not isinstance(decoded, Mapping):
        raise ValueError(f"{snapshot.path}: expected a JSON object")
    return dict(decoded)


def _jsonl_objects_from_snapshot(
    snapshot: _FileSnapshot,
) -> tuple[dict[str, Any], ...]:
    try:
        text = snapshot.material.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{snapshot.path}: input is not UTF-8") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            decoded = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{snapshot.path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(decoded, Mapping):
            raise ValueError(
                f"{snapshot.path}:{line_number}: record must be an object"
            )
        rows.append(dict(decoded))
    if not rows:
        raise ValueError(f"{snapshot.path}: input cannot be empty")
    return tuple(rows)


def _decoder_requests_from_snapshot(
    snapshot: _FileSnapshot,
) -> tuple[ExternalDecoderRequest, ...]:
    rows = tuple(
        ExternalDecoderRequest.parse(raw)
        for raw in _jsonl_objects_from_snapshot(snapshot)
    )
    if len({row.request_id for row in rows}) != len(rows):
        raise ValueError(f"{snapshot.path}: duplicate decoder request IDs")
    return rows


def _decoder_judgments_from_snapshot(
    snapshot: _FileSnapshot,
) -> tuple[ExternalDecoderJudgment, ...]:
    return tuple(
        ExternalDecoderJudgment.parse(raw)
        for raw in _jsonl_objects_from_snapshot(snapshot)
    )


def _decoder_truth_from_snapshot(
    snapshot: _FileSnapshot,
) -> tuple[DecoderTruthLabel, ...]:
    rows = tuple(
        DecoderTruthLabel.parse(raw)
        for raw in _jsonl_objects_from_snapshot(snapshot)
    )
    if len({row.pseudonymous_state_id for row in rows}) != len(rows):
        raise ValueError(f"{snapshot.path}: duplicate decoder truth state IDs")
    return rows


def _validate_timestamp(value: object, name: str) -> str:
    text = _require_text(value, name)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} must include a UTC offset")
    return text


def _strict_fields(
    raw: Mapping[str, Any],
    allowed: set[str],
    *,
    name: str,
) -> None:
    if set(raw) != allowed:
        raise ValueError(
            f"{name} has missing or unknown fields: "
            + canonical_json(
                {
                    "missing": sorted(allowed - set(raw)),
                    "unknown": sorted(set(raw) - allowed),
                }
            )
        )


def _terminal_action_dict(action: TerminalAction) -> dict[str, Any]:
    return {
        "item_id": action.item_id,
        "item_sha256": action.item_sha256,
        "wording_template_id": action.wording_template_id,
        "question_type": action.question_type,
        "selected_option_id": action.selected_option_id,
        "declared_direction": action.declared_direction,
    }


def _parse_terminal_action(raw: Mapping[str, Any]) -> TerminalAction:
    allowed = {
        "item_id",
        "item_sha256",
        "wording_template_id",
        "question_type",
        "selected_option_id",
        "declared_direction",
    }
    _strict_fields(raw, allowed, name="terminal action")
    return TerminalAction(
        item_id=raw["item_id"],
        item_sha256=raw["item_sha256"],
        wording_template_id=raw["wording_template_id"],
        question_type=raw["question_type"],
        selected_option_id=raw["selected_option_id"],
        declared_direction=raw["declared_direction"],
    )


@dataclass(frozen=True, slots=True)
class NativeTerminalActionRecord:
    """One genuinely recorded native-system terminal-suite execution."""

    record_id: str
    trajectory_id: str
    domain_id: str
    updater_id: str
    evaluation_split: str
    adapter_kind: str
    evidence_origin: str
    native_state_id: str
    native_system_id: str
    native_system_version: str
    suite_id: str
    suite_sha256: str
    action_execution_mode: str
    execution_trace_sha256: str
    recorded_at: str
    recording_attestation: str
    actions: tuple[TerminalAction, ...]
    record_sha256: str = ""

    def __post_init__(self) -> None:
        for name in (
            "record_id",
            "trajectory_id",
            "domain_id",
            "updater_id",
            "native_system_id",
            "native_system_version",
            "suite_id",
        ):
            _require_text(getattr(self, name), name)
        if self.evaluation_split != "test":
            raise ValueError(
                "native terminal action evaluation_split must be test"
            )
        if self.adapter_kind != _ACTION_ADAPTER_KIND:
            raise ValueError(
                "adapter_kind must be native_end_to_end_recorded; reference, "
                "persona, and deterministic projection actions are ineligible"
            )
        if self.evidence_origin != _ACTION_EVIDENCE_ORIGIN:
            raise ValueError(
                "evidence_origin must be imported_native_system"
            )
        if self.updater_id not in _NATIVE_UPDATERS:
            raise ValueError("native action record has a non-native updater")
        _validate_digest(self.native_state_id, "native_state_id")
        _validate_digest(self.suite_sha256, "suite_sha256")
        _validate_digest(
            self.execution_trace_sha256,
            "execution_trace_sha256",
        )
        if self.action_execution_mode not in _ACTION_EXECUTION_MODES:
            raise ValueError(
                "action_execution_mode must be recorded_live or "
                "recorded_replay"
            )
        _validate_timestamp(self.recorded_at, "recorded_at")
        if self.recording_attestation != _RECORDING_ATTESTATION:
            raise ValueError(
                "recording_attestation must explicitly exclude reference and "
                "belief-projection actions"
            )
        material = tuple(self.actions)
        if not material:
            raise ValueError("native terminal action record cannot be empty")
        if len({action.item_id for action in material}) != len(material):
            raise ValueError("native terminal action item IDs must be unique")
        object.__setattr__(self, "actions", material)
        expected = _digest_value(self._binding_payload())
        if self.record_sha256 and self.record_sha256 != expected:
            raise ValueError(
                "native terminal action record digest does not match content"
            )
        object.__setattr__(self, "record_sha256", expected)

    @classmethod
    def build(
        cls,
        *,
        record_id: str,
        trajectory_id: str,
        domain_id: str,
        updater_id: str,
        native_state_id: str,
        native_system_id: str,
        native_system_version: str,
        suite_id: str,
        suite_sha256: str,
        action_execution_mode: str,
        execution_trace_sha256: str,
        recorded_at: str,
        actions: Iterable[TerminalAction],
    ) -> "NativeTerminalActionRecord":
        return cls(
            record_id=record_id,
            trajectory_id=trajectory_id,
            domain_id=domain_id,
            updater_id=updater_id,
            evaluation_split="test",
            adapter_kind=_ACTION_ADAPTER_KIND,
            evidence_origin=_ACTION_EVIDENCE_ORIGIN,
            native_state_id=native_state_id,
            native_system_id=native_system_id,
            native_system_version=native_system_version,
            suite_id=suite_id,
            suite_sha256=suite_sha256,
            action_execution_mode=action_execution_mode,
            execution_trace_sha256=execution_trace_sha256,
            recorded_at=recorded_at,
            recording_attestation=_RECORDING_ATTESTATION,
            actions=tuple(actions),
        )

    @classmethod
    def parse(
        cls,
        raw: Mapping[str, Any],
    ) -> "NativeTerminalActionRecord":
        allowed = {
            "schema_version",
            "record_id",
            "trajectory_id",
            "domain_id",
            "updater_id",
            "evaluation_split",
            "adapter_kind",
            "evidence_origin",
            "native_state_id",
            "native_system_id",
            "native_system_version",
            "suite_id",
            "suite_sha256",
            "action_execution_mode",
            "execution_trace_sha256",
            "recorded_at",
            "recording_attestation",
            "actions",
            "record_sha256",
        }
        _strict_fields(raw, allowed, name="native terminal action record")
        if raw["schema_version"] != 1:
            raise ValueError(
                "native terminal action schema_version must be 1"
            )
        action_rows = raw["actions"]
        if (
            not isinstance(action_rows, Sequence)
            or isinstance(action_rows, (str, bytes))
        ):
            raise ValueError("native terminal actions must be an array")
        actions = []
        for raw_action in action_rows:
            if not isinstance(raw_action, Mapping):
                raise ValueError("each terminal action must be an object")
            actions.append(_parse_terminal_action(raw_action))
        return cls(
            record_id=raw["record_id"],
            trajectory_id=raw["trajectory_id"],
            domain_id=raw["domain_id"],
            updater_id=raw["updater_id"],
            evaluation_split=raw["evaluation_split"],
            adapter_kind=raw["adapter_kind"],
            evidence_origin=raw["evidence_origin"],
            native_state_id=raw["native_state_id"],
            native_system_id=raw["native_system_id"],
            native_system_version=raw["native_system_version"],
            suite_id=raw["suite_id"],
            suite_sha256=raw["suite_sha256"],
            action_execution_mode=raw["action_execution_mode"],
            execution_trace_sha256=raw["execution_trace_sha256"],
            recorded_at=raw["recorded_at"],
            recording_attestation=raw["recording_attestation"],
            actions=tuple(actions),
            record_sha256=raw["record_sha256"],
        )

    def _binding_payload(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "trajectory_id": self.trajectory_id,
            "domain_id": self.domain_id,
            "updater_id": self.updater_id,
            "evaluation_split": self.evaluation_split,
            "adapter_kind": self.adapter_kind,
            "evidence_origin": self.evidence_origin,
            "native_state_id": self.native_state_id,
            "native_system_id": self.native_system_id,
            "native_system_version": self.native_system_version,
            "suite_id": self.suite_id,
            "suite_sha256": self.suite_sha256,
            "action_execution_mode": self.action_execution_mode,
            "execution_trace_sha256": self.execution_trace_sha256,
            "recorded_at": self.recorded_at,
            "recording_attestation": self.recording_attestation,
            "actions": [
                _terminal_action_dict(action) for action in self.actions
            ],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            **self._binding_payload(),
            "record_sha256": self.record_sha256,
        }


def read_native_terminal_action_records(
    path: str | Path,
) -> tuple[NativeTerminalActionRecord, ...]:
    source = Path(path)
    rows = _read_jsonl_objects(source)
    records = tuple(NativeTerminalActionRecord.parse(row) for row in rows)
    if len({record.record_id for record in records}) != len(records):
        raise ValueError(f"{source}: duplicate native action record IDs")
    if len({record.trajectory_id for record in records}) != len(records):
        raise ValueError(f"{source}: duplicate native action trajectories")
    return records


@dataclass(frozen=True, slots=True)
class DecoderSourceAssessment:
    decoder_instance_id: str
    decoder_family_id: str
    judgment_origin: str
    source_descriptor: str
    eligible_for_gate4: bool
    dependency_notes: str

    def __post_init__(self) -> None:
        for name in (
            "decoder_instance_id",
            "decoder_family_id",
            "judgment_origin",
            "source_descriptor",
            "dependency_notes",
        ):
            _require_text(getattr(self, name), name)
        _require_bool(self.eligible_for_gate4, "eligible_for_gate4")

    @classmethod
    def parse(cls, raw: Mapping[str, Any]) -> "DecoderSourceAssessment":
        allowed = {
            "decoder_instance_id",
            "decoder_family_id",
            "judgment_origin",
            "source_descriptor",
            "eligible_for_gate4",
            "dependency_notes",
        }
        _strict_fields(raw, allowed, name="decoder source assessment")
        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        return {
            "decoder_instance_id": self.decoder_instance_id,
            "decoder_family_id": self.decoder_family_id,
            "judgment_origin": self.judgment_origin,
            "source_descriptor": self.source_descriptor,
            "eligible_for_gate4": self.eligible_for_gate4,
            "dependency_notes": self.dependency_notes,
        }


@dataclass(frozen=True, slots=True)
class DecoderSourcePairAssessment:
    left_decoder_instance_id: str
    right_decoder_instance_id: str
    genuinely_distinct_for_claimed_scope: bool
    rationale: str

    def __post_init__(self) -> None:
        left = _require_text(
            self.left_decoder_instance_id,
            "left_decoder_instance_id",
        )
        right = _require_text(
            self.right_decoder_instance_id,
            "right_decoder_instance_id",
        )
        if left >= right:
            raise ValueError(
                "decoder source pair IDs must be distinct and sorted"
            )
        _require_bool(
            self.genuinely_distinct_for_claimed_scope,
            "genuinely_distinct_for_claimed_scope",
        )
        _require_text(self.rationale, "rationale")

    @classmethod
    def parse(
        cls,
        raw: Mapping[str, Any],
    ) -> "DecoderSourcePairAssessment":
        allowed = {
            "left_decoder_instance_id",
            "right_decoder_instance_id",
            "genuinely_distinct_for_claimed_scope",
            "rationale",
        }
        _strict_fields(raw, allowed, name="decoder source pair assessment")
        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        return {
            "left_decoder_instance_id": self.left_decoder_instance_id,
            "right_decoder_instance_id": self.right_decoder_instance_id,
            "genuinely_distinct_for_claimed_scope": (
                self.genuinely_distinct_for_claimed_scope
            ),
            "rationale": self.rationale,
        }


@dataclass(frozen=True, slots=True)
class DecoderSourceReview:
    """Responsible-researcher source-design determination."""

    review_id: str
    responsible_researcher_id: str
    reviewed_at: str
    requests_sha256: str
    judgments_sha256: str
    decision: str
    source_assessments: tuple[DecoderSourceAssessment, ...]
    pair_assessments: tuple[DecoderSourcePairAssessment, ...]
    review_sha256: str = ""

    def __post_init__(self) -> None:
        _require_text(self.review_id, "review_id")
        _require_text(
            self.responsible_researcher_id,
            "responsible_researcher_id",
        )
        _validate_timestamp(self.reviewed_at, "reviewed_at")
        _validate_digest(self.requests_sha256, "requests_sha256")
        _validate_digest(self.judgments_sha256, "judgments_sha256")
        if self.decision not in {
            "eligible_distinct_sources",
            "not_eligible",
        }:
            raise ValueError("unknown decoder source-review decision")
        sources = tuple(self.source_assessments)
        pairs = tuple(self.pair_assessments)
        if not sources:
            raise ValueError("source review must assess at least one source")
        source_ids = [
            assessment.decoder_instance_id for assessment in sources
        ]
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("source review contains duplicate source IDs")
        pair_keys = [
            (
                assessment.left_decoder_instance_id,
                assessment.right_decoder_instance_id,
            )
            for assessment in pairs
        ]
        if len(set(pair_keys)) != len(pair_keys):
            raise ValueError("source review contains duplicate source pairs")
        unknown_pair_ids = sorted(
            {
                item
                for pair in pair_keys
                for item in pair
                if item not in set(source_ids)
            }
        )
        if unknown_pair_ids:
            raise ValueError(
                "source review pair references unknown sources: "
                + ", ".join(unknown_pair_ids)
            )
        object.__setattr__(self, "source_assessments", sources)
        object.__setattr__(self, "pair_assessments", pairs)
        expected = _digest_value(self._binding_payload())
        if self.review_sha256 and self.review_sha256 != expected:
            raise ValueError("source review digest does not match content")
        object.__setattr__(self, "review_sha256", expected)

    @classmethod
    def build(
        cls,
        *,
        review_id: str,
        responsible_researcher_id: str,
        reviewed_at: str,
        requests_sha256: str,
        judgments_sha256: str,
        decision: str,
        source_assessments: Iterable[DecoderSourceAssessment],
        pair_assessments: Iterable[DecoderSourcePairAssessment],
    ) -> "DecoderSourceReview":
        return cls(
            review_id=review_id,
            responsible_researcher_id=responsible_researcher_id,
            reviewed_at=reviewed_at,
            requests_sha256=requests_sha256,
            judgments_sha256=judgments_sha256,
            decision=decision,
            source_assessments=tuple(source_assessments),
            pair_assessments=tuple(pair_assessments),
        )

    @classmethod
    def parse(cls, raw: Mapping[str, Any]) -> "DecoderSourceReview":
        allowed = {
            "schema_version",
            "review_id",
            "responsible_researcher_id",
            "reviewed_at",
            "requests_sha256",
            "judgments_sha256",
            "decision",
            "source_assessments",
            "pair_assessments",
            "review_sha256",
        }
        _strict_fields(raw, allowed, name="decoder source review")
        if raw["schema_version"] != 1:
            raise ValueError("decoder source review schema_version must be 1")
        source_rows = raw["source_assessments"]
        pair_rows = raw["pair_assessments"]
        if not isinstance(source_rows, list):
            raise ValueError("source_assessments must be an array")
        if not isinstance(pair_rows, list):
            raise ValueError("pair_assessments must be an array")
        if not all(isinstance(item, Mapping) for item in source_rows):
            raise ValueError("each source assessment must be an object")
        if not all(isinstance(item, Mapping) for item in pair_rows):
            raise ValueError("each pair assessment must be an object")
        return cls(
            review_id=raw["review_id"],
            responsible_researcher_id=raw["responsible_researcher_id"],
            reviewed_at=raw["reviewed_at"],
            requests_sha256=raw["requests_sha256"],
            judgments_sha256=raw["judgments_sha256"],
            decision=raw["decision"],
            source_assessments=tuple(
                DecoderSourceAssessment.parse(item) for item in source_rows
            ),
            pair_assessments=tuple(
                DecoderSourcePairAssessment.parse(item) for item in pair_rows
            ),
            review_sha256=raw["review_sha256"],
        )

    def _binding_payload(self) -> dict[str, Any]:
        return {
            "review_id": self.review_id,
            "responsible_researcher_id": self.responsible_researcher_id,
            "reviewed_at": self.reviewed_at,
            "requests_sha256": self.requests_sha256,
            "judgments_sha256": self.judgments_sha256,
            "decision": self.decision,
            "source_assessments": [
                item.to_dict() for item in self.source_assessments
            ],
            "pair_assessments": [
                item.to_dict() for item in self.pair_assessments
            ],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            **self._binding_payload(),
            "review_sha256": self.review_sha256,
        }


def read_decoder_source_review(path: str | Path) -> DecoderSourceReview:
    raw = _read_json_object(Path(path))
    return DecoderSourceReview.parse(raw)


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(decoded, Mapping):
        raise ValueError(f"{path}: expected a JSON object")
    return dict(decoded)


def _read_jsonl_objects(path: Path) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                decoded = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            if not isinstance(decoded, Mapping):
                raise ValueError(
                    f"{path}:{line_number}: record must be an object"
                )
            rows.append(dict(decoded))
    if not rows:
        raise ValueError(f"{path}: input cannot be empty")
    return tuple(rows)


def _parse_terminal_suite(raw: Mapping[str, Any]) -> HeldOutTerminalSuite:
    allowed = {
        "schema_version",
        "suite_id",
        "domain_id",
        "suite_sha256",
        "items",
    }
    _strict_fields(raw, allowed, name="held-out terminal suite")
    if raw["schema_version"] != 1:
        raise ValueError("held-out terminal suite schema_version must be 1")
    raw_items = raw["items"]
    if not isinstance(raw_items, list):
        raise ValueError("held-out terminal suite items must be an array")
    items: list[HeldOutTerminalItem] = []
    for raw_item in raw_items:
        if not isinstance(raw_item, Mapping):
            raise ValueError("held-out terminal item must be an object")
        item_allowed = {
            "item_id",
            "family_id",
            "domain_id",
            "scenario_family_id",
            "wording_template_id",
            "question_type",
            "prompt",
            "options",
            "target_attribute",
            "item_sha256",
        }
        _strict_fields(
            raw_item,
            item_allowed,
            name="held-out terminal item",
        )
        raw_options = raw_item["options"]
        if not isinstance(raw_options, list):
            raise ValueError("held-out terminal item options must be an array")
        options: list[HeldOutTerminalOption] = []
        for raw_option in raw_options:
            if not isinstance(raw_option, Mapping):
                raise ValueError("held-out terminal option must be an object")
            option_allowed = {
                "option_id",
                "label",
                "features",
                "fingerprint",
            }
            _strict_fields(
                raw_option,
                option_allowed,
                name="held-out terminal option",
            )
            features = raw_option["features"]
            if not isinstance(features, list):
                raise ValueError(
                    "held-out terminal option features must be an array"
                )
            option = HeldOutTerminalOption(
                option_id=raw_option["option_id"],
                label=raw_option["label"],
                features=tuple(features),
            )
            if raw_option["fingerprint"] != option.fingerprint:
                raise ValueError(
                    f"terminal option fingerprint mismatch for "
                    f"{option.option_id}"
                )
            options.append(option)
        item = HeldOutTerminalItem(
            item_id=raw_item["item_id"],
            family_id=raw_item["family_id"],
            domain_id=raw_item["domain_id"],
            scenario_family_id=raw_item["scenario_family_id"],
            wording_template_id=raw_item["wording_template_id"],
            question_type=raw_item["question_type"],
            prompt=raw_item["prompt"],
            options=tuple(options),
            target_attribute=raw_item["target_attribute"],
        )
        if raw_item["item_sha256"] != item.item_sha256:
            raise ValueError(
                f"terminal item digest mismatch for {item.item_id}"
            )
        items.append(item)
    return HeldOutTerminalSuite(
        suite_id=raw["suite_id"],
        domain_id=raw["domain_id"],
        items=tuple(items),
        suite_sha256=raw["suite_sha256"],
    )


def _native_state_id(
    state: Mapping[str, Any],
    *,
    updater_id: str,
) -> str:
    required = {
        "memory_kind",
        "base_belief",
        "episodes",
        "claims",
        "persona_belief",
        "persona_text",
        "state_id",
    }
    _strict_fields(state, required, name="terminal native state")
    state_id = _validate_digest(state["state_id"], "native state_id")
    if state["memory_kind"] != _UPDATER_MEMORY_KIND[updater_id]:
        raise ValueError("native state memory kind does not match updater")
    payload = {key: state[key] for key in required if key != "state_id"}
    material = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    if sha256(material).hexdigest() != state_id:
        raise ValueError("terminal native state digest does not match content")
    return state_id


def _counter_profile_available(row: Mapping[str, Any]) -> bool:
    audit = row.get("audit_record")
    if not isinstance(audit, Mapping):
        raise ValueError("trajectory audit_record must be an object")
    interactions = audit.get("interactions")
    if not isinstance(interactions, list) or not interactions:
        raise ValueError("trajectory audit interactions must be nonempty")
    for interaction in interactions:
        if not isinstance(interaction, Mapping):
            raise ValueError("trajectory audit interaction must be an object")
        context = interaction.get("context")
        if not isinstance(context, Mapping):
            raise ValueError("trajectory interaction context must be an object")
        target = context.get("target_attribute")
        if target is None:
            continue
        if (
            isinstance(target, bool)
            or not isinstance(target, int)
            or not 0 <= target < 3
        ):
            raise ValueError("trajectory target_attribute is invalid")
        options = context.get("options")
        if not isinstance(options, list):
            raise ValueError("trajectory context options must be an array")
        directions: set[int] = set()
        for option in options:
            if not isinstance(option, Mapping):
                raise ValueError("trajectory option must be an object")
            features = option.get("features")
            if not isinstance(features, list) or len(features) != 3:
                raise ValueError("trajectory option features are invalid")
            value = features[target]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError("trajectory option feature is invalid")
            if float(value) != 0.0:
                directions.add(1 if float(value) > 0.0 else -1)
        if directions != {-1, 1}:
            return False
    return True


def _eligible_trajectories(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    observed_ids: set[str] = set()
    for row in rows:
        trajectory_id = _require_text(
            row.get("trajectory_id"),
            "trajectory_id",
        )
        if trajectory_id in observed_ids:
            raise ValueError("duplicate Experiment B trajectory IDs")
        observed_ids.add(trajectory_id)
        updater_id = row.get("updater_id")
        if (
            updater_id in _NATIVE_UPDATERS
            and row.get("policy_id") == "soft_profile_conditioned"
            and row.get("initial_profile_condition") == "incorrect"
            and _counter_profile_available(row)
        ):
            result[trajectory_id] = row
    if not result:
        raise ValueError(
            "completed Experiment B run has no eligible Gate 4 native "
            "trajectories"
        )
    return result


def _matched_failure_cases(
    eligible: Mapping[str, Mapping[str, Any]],
    assessment_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in assessment_rows:
        trajectory_id = _require_text(
            row.get("trajectory_id"),
            "assessment trajectory_id",
        )
        attribute = row.get("attribute")
        if (
            isinstance(attribute, bool)
            or not isinstance(attribute, int)
            or not 0 <= attribute < 3
        ):
            raise ValueError("self-confirmation assessment attribute is invalid")
        key = (trajectory_id, attribute)
        if key in by_key:
            raise ValueError("duplicate self-confirmation assessment key")
        if not isinstance(row.get("reportable"), bool):
            raise ValueError("assessment reportable must be Boolean")
        lcg = row.get("cumulative_lcg")
        if (
            isinstance(lcg, bool)
            or not isinstance(lcg, (int, float))
            or not math.isfinite(float(lcg))
        ):
            raise ValueError("assessment cumulative_lcg must be finite")
        by_key[key] = row
    for trajectory_id in eligible:
        missing = [
            attribute
            for attribute in range(3)
            if (trajectory_id, attribute) not in by_key
        ]
        if missing:
            raise ValueError(
                f"eligible trajectory {trajectory_id!r} lacks "
                "self-confirmation assessments"
            )
    by_crn_updater = {
        (
            _require_text(row.get("crn_key"), "trajectory crn_key"),
            _require_text(row.get("updater_id"), "trajectory updater_id"),
        ): row
        for row in eligible.values()
    }
    cases: list[dict[str, Any]] = []
    for trajectory in eligible.values():
        if trajectory["updater_id"] != "semantic_memory":
            continue
        control = by_crn_updater.get(
            (trajectory["crn_key"], "provenance_linked_memory")
        )
        if control is None:
            continue
        for attribute in range(3):
            candidate = by_key[(trajectory["trajectory_id"], attribute)]
            control_assessment = by_key[
                (control["trajectory_id"], attribute)
            ]
            if candidate["reportable"] and (
                not control_assessment["reportable"]
                or float(control_assessment["cumulative_lcg"])
                < float(candidate["cumulative_lcg"])
            ):
                cases.append(
                    {
                        "blind_trajectory_id": trajectory["trajectory_id"],
                        "control_trajectory_id": control["trajectory_id"],
                        "attribute": attribute,
                    }
                )
    return cases


def _read_codebook(
    path: Path,
    requests: Sequence[ExternalDecoderRequest],
    trajectories: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    rows = _read_jsonl_objects(path)
    allowed = {
        "schema_version",
        "request_id",
        "pseudonymous_state_id",
        "trajectory_id",
        "updater_id",
        "domain_id",
        "evaluation_split",
        "native_state_id",
    }
    by_request = {request.request_id: request for request in requests}
    if len(rows) != len(by_request):
        raise ValueError("decoder codebook/request counts differ")
    seen_requests: set[str] = set()
    result = []
    for row in rows:
        _strict_fields(row, allowed, name="decoder codebook row")
        if row["schema_version"] != 1:
            raise ValueError("decoder codebook schema_version must be 1")
        request_id = _require_text(row["request_id"], "codebook request_id")
        if request_id in seen_requests:
            raise ValueError("duplicate decoder codebook request IDs")
        seen_requests.add(request_id)
        try:
            request = by_request[request_id]
        except KeyError as exc:
            raise ValueError(
                f"codebook references unknown request {request_id!r}"
            ) from exc
        if (
            row["pseudonymous_state_id"]
            != request.pseudonymous_state_id
            or row["evaluation_split"] != request.evaluation_split
        ):
            raise ValueError("decoder codebook/request binding mismatch")
        _validate_digest(row["native_state_id"], "codebook native_state_id")
        if row["evaluation_split"] == "test":
            trajectory_id = row["trajectory_id"]
            try:
                trajectory = trajectories[trajectory_id]
            except KeyError as exc:
                raise ValueError(
                    "test decoder codebook references unknown trajectory"
                ) from exc
            state = trajectory.get("terminal_native_state")
            if not isinstance(state, Mapping):
                raise ValueError(
                    "test decoder codebook trajectory lacks native state"
                )
            expected_state_id = _native_state_id(
                state,
                updater_id=trajectory["updater_id"],
            )
            if (
                row["native_state_id"] != expected_state_id
                or row["updater_id"] != trajectory["updater_id"]
                or row["domain_id"] != trajectory["domain_id"]
            ):
                raise ValueError(
                    "decoder codebook/trajectory binding mismatch"
                )
        result.append(dict(row))
    if seen_requests != set(by_request):
        raise ValueError("decoder codebook does not cover every request")
    return tuple(result)


def _validate_source_review(
    review: DecoderSourceReview,
    judgments: Sequence[ExternalDecoderJudgment],
    *,
    requests_sha256: str,
    judgments_sha256: str,
) -> dict[str, Any]:
    if (
        review.requests_sha256 != requests_sha256
        or review.judgments_sha256 != judgments_sha256
    ):
        raise ValueError(
            "decoder source review is not hash-bound to supplied materials"
        )
    if review.decision != "eligible_distinct_sources":
        raise ValueError("decoder source design was not approved as eligible")
    metadata: dict[str, tuple[str, str, str]] = {}
    judgments_by_request: dict[str, set[str]] = {}
    for row in judgments:
        item = (
            row.decoder_family_id,
            row.judgment_origin,
            row.source_descriptor,
        )
        previous = metadata.setdefault(row.decoder_instance_id, item)
        if previous != item:
            raise ValueError("decoder instance metadata is inconsistent")
        judgments_by_request.setdefault(row.request_id, set()).add(
            row.decoder_instance_id
        )
    assessed = {
        row.decoder_instance_id: row for row in review.source_assessments
    }
    if set(assessed) != set(metadata):
        raise ValueError(
            "source review must cover exactly the judgment source instances"
        )
    for instance_id, expected in metadata.items():
        row = assessed[instance_id]
        observed = (
            row.decoder_family_id,
            row.judgment_origin,
            row.source_descriptor,
        )
        if observed != expected:
            raise ValueError(
                f"source review metadata mismatch for {instance_id!r}"
            )
        if not row.eligible_for_gate4:
            raise ValueError(
                f"decoder source {instance_id!r} is not Gate 4 eligible"
            )
    expected_pairs = {
        tuple(sorted(pair))
        for instances in judgments_by_request.values()
        for pair in combinations(sorted(instances), 2)
    }
    reviewed_pairs = {
        (
            row.left_decoder_instance_id,
            row.right_decoder_instance_id,
        ): row
        for row in review.pair_assessments
    }
    if set(reviewed_pairs) != expected_pairs:
        raise ValueError(
            "source review must cover every co-occurring decoder source pair"
        )
    rejected = [
        pair
        for pair, row in reviewed_pairs.items()
        if not row.genuinely_distinct_for_claimed_scope
    ]
    if rejected:
        raise ValueError(
            "decoder source pairs were not reviewed as genuinely distinct: "
            + canonical_json(rejected)
        )
    return {
        "review_id": review.review_id,
        "review_sha256": review.review_sha256,
        "responsible_researcher_id": review.responsible_researcher_id,
        "reviewed_at": review.reviewed_at,
        "decision": review.decision,
        "reviewed_source_count": len(assessed),
        "reviewed_pair_count": len(reviewed_pairs),
        "all_sources_eligible": True,
        "all_cooccurring_pairs_genuinely_distinct_for_claimed_scope": True,
    }


def _validate_actions(
    records: Sequence[NativeTerminalActionRecord],
    eligible: Mapping[str, Mapping[str, Any]],
    suites: Mapping[str, HeldOutTerminalSuite],
) -> tuple[dict[str, Any], ...]:
    by_trajectory = {record.trajectory_id: record for record in records}
    if len(by_trajectory) != len(records):
        raise ValueError("duplicate native action trajectory records")
    if set(by_trajectory) != set(eligible):
        raise ValueError(
            "native action records must cover eligible trajectories exactly: "
            + canonical_json(
                {
                    "missing": sorted(set(eligible) - set(by_trajectory)),
                    "unexpected": sorted(
                        set(by_trajectory) - set(eligible)
                    ),
                }
            )
        )
    scored: list[dict[str, Any]] = []
    for trajectory_id, trajectory in sorted(eligible.items()):
        record = by_trajectory[trajectory_id]
        domain_id = _require_text(
            trajectory.get("domain_id"),
            "trajectory domain_id",
        )
        updater_id = _require_text(
            trajectory.get("updater_id"),
            "trajectory updater_id",
        )
        state = trajectory.get("terminal_native_state")
        if not isinstance(state, Mapping):
            raise ValueError("eligible trajectory lacks terminal native state")
        state_id = _native_state_id(state, updater_id=updater_id)
        try:
            suite = suites[domain_id]
        except KeyError as exc:
            raise ValueError(
                f"missing held-out terminal suite for {domain_id!r}"
            ) from exc
        if (
            record.domain_id != domain_id
            or record.updater_id != updater_id
            or record.native_state_id != state_id
            or record.suite_id != suite.suite_id
            or record.suite_sha256 != suite.suite_sha256
        ):
            raise ValueError(
                f"native action record binding mismatch for {trajectory_id}"
            )
        theta = validate_theta(trajectory.get("theta"))
        score = score_heldout_terminal_actions(
            suite,
            record.actions,
            theta,
        )
        scored.append(
            {
                "trajectory_id": trajectory_id,
                "record_id": record.record_id,
                "record_sha256": record.record_sha256,
                "domain_id": domain_id,
                "updater_id": updater_id,
                "native_state_id": state_id,
                "native_system_id": record.native_system_id,
                "native_system_version": record.native_system_version,
                "suite_id": suite.suite_id,
                "suite_sha256": suite.suite_sha256,
                "action_execution_mode": record.action_execution_mode,
                "execution_trace_sha256": record.execution_trace_sha256,
                "suite_binding_validated": True,
                **score.to_dict(),
            }
        )
    return tuple(scored)


def _gate_4_from_persisted_evidence(
    eligible: Mapping[str, Mapping[str, Any]],
    *,
    matched_failure_cases: Sequence[Mapping[str, Any]],
    externally_decoded_trajectory_ids: set[str],
    decoder_audit: Mapping[str, Any],
    source_review: Mapping[str, Any],
    action_scores: Sequence[Mapping[str, Any]],
) -> GateReport:
    native_ids = set(eligible)
    complete_state_ids = {
        trajectory_id
        for trajectory_id, row in eligible.items()
        if isinstance(row.get("terminal_native_state"), Mapping)
    }
    action_ids = {
        str(row["trajectory_id"]) for row in action_scores
    }
    external_ready = (
        decoder_audit.get("complete_coverage") is True
        and decoder_audit.get("source_design_eligible") is True
        and source_review.get("decision") == "eligible_distinct_sources"
        and source_review.get("all_sources_eligible") is True
        and source_review.get(
            "all_cooccurring_pairs_genuinely_distinct_for_claimed_scope"
        )
        is True
        and native_ids <= externally_decoded_trajectory_ids
    )
    action_ready = (
        native_ids <= action_ids
        and all(
            row.get("suite_binding_validated") is True
            and row.get("action_execution_mode")
            in _ACTION_EXECUTION_MODES
            for row in action_scores
        )
    )
    return GateReport(
        gate_id="gate-4",
        title="Native-system validity",
        evidence_scope="immutable_external_evidence_review",
        criteria=(
            GateCriterion(
                "native-loop-present",
                "At least one inspectable native memory-action loop was evaluated.",
                bool(native_ids),
                {"native_trajectories": len(native_ids)},
                "native_trajectories > 0",
            ),
            GateCriterion(
                "native-state-retained",
                "Complete terminal native states and transition events are retained.",
                complete_state_ids == native_ids if native_ids else None,
                {
                    "complete_terminal_state_trajectory_ids": sorted(
                        complete_state_ids
                    ),
                    "events_retained": True,
                    "source_run_checksum_verified": True,
                },
                "complete retained state and events for every native trajectory",
            ),
            GateCriterion(
                "independent-blinded-decoder-judgments",
                (
                    "Imported blinded judgments from at least two genuinely "
                    "distinct, independently reviewed decoder sources cover "
                    "every eligible native trajectory."
                ),
                external_ready if native_ids else None,
                {
                    "external_evidence_imported": True,
                    "externally_decoded_trajectory_ids": sorted(
                        externally_decoded_trajectory_ids
                    ),
                    "source_design_review": dict(source_review),
                    "deterministic_projections_count_as_external": False,
                },
                (
                    "validated imported coverage; blind to system/truth; "
                    "source-design eligible; genuinely distinct-source review; "
                    "deterministic native projections are ineligible"
                ),
            ),
            GateCriterion(
                "native-end-to-end-terminal-actions",
                (
                    "The native system itself produces recorded end-to-end "
                    "actions on the hash-bound held-out terminal suite."
                ),
                action_ready if native_ids else None,
                {
                    "genuine_native_action_trajectory_ids": sorted(
                        action_ids
                    ),
                    "validated_action_record_count": len(action_scores),
                    "reference_action_rows": 0,
                    "persona_or_structured_references_count_as_native_actions": (
                        False
                    ),
                },
                (
                    "imported native_end_to_end_recorded actions with exact "
                    "suite binding and recorded live/replay execution for "
                    "every eligible native trajectory"
                ),
            ),
            GateCriterion(
                "native-failure-observed",
                (
                    "A provenance-blind native soft loop has a five-clause "
                    "failure relative to a matched provenance-linked control."
                ),
                bool(matched_failure_cases) if native_ids else None,
                {
                    "matched_native_failure_cases": [
                        dict(item) for item in matched_failure_cases
                    ]
                },
                "at least one matched blind-versus-provenance-linked case",
            ),
        ),
    )


def _baseline_gate_4(path: Path) -> Mapping[str, Any]:
    report = _read_json_object(path)
    gates = report.get("gates")
    if not isinstance(gates, list):
        raise ValueError("source gate report has no gates array")
    matches = [
        gate
        for gate in gates
        if isinstance(gate, Mapping) and gate.get("gate_id") == "gate-4"
    ]
    if len(matches) != 1:
        raise ValueError("source gate report must contain exactly one Gate 4")
    if report.get("claim_status") != "not_claimed":
        raise ValueError("source run unexpectedly claims scientific status")
    return matches[0]


def _assert_baseline_consistency(
    baseline: Mapping[str, Any],
    recomputed: GateReport,
) -> None:
    baseline_rows = baseline.get("criteria")
    if not isinstance(baseline_rows, list):
        raise ValueError("source Gate 4 criteria are missing")
    baseline_by_id = {
        row.get("criterion_id"): row
        for row in baseline_rows
        if isinstance(row, Mapping)
    }
    recomputed_by_id = {
        row.criterion_id: row for row in recomputed.criteria
    }
    for criterion_id in (
        "native-loop-present",
        "native-state-retained",
        "native-failure-observed",
    ):
        baseline_row = baseline_by_id.get(criterion_id)
        if (
            not isinstance(baseline_row, Mapping)
            or baseline_row.get("passed")
            != recomputed_by_id[criterion_id].passed
        ):
            raise ValueError(
                f"recomputed {criterion_id} disagrees with source Gate 4"
            )


def _input_manifest_entry(
    path: Path,
    *,
    record_count: int | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "filename": path.name,
        "sha256": _file_digest(path),
        "bytes": path.stat().st_size,
    }
    if record_count is not None:
        result["record_count"] = record_count
    return result


def _write_text_durable(path: Path, value: str) -> None:
    """Create one staged text file and make its contents durable."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json_durable(path: Path, value: Any) -> None:
    _write_text_durable(path, canonical_json(value) + "\n")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _gate_review_lock_path(output: Path) -> Path:
    return output.parent / f".{output.name}.gate4-review.lock"


@contextmanager
def _hold_exclusive_gate_review_lock(output: Path) -> Iterable[None]:
    """Serialize publication attempts for one destination."""

    lock_path = _gate_review_lock_path(output)
    try:
        descriptor = os.open(
            lock_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
    except FileExistsError as exc:
        raise FileExistsError(
            f"gate-review output is locked: {output}"
        ) from exc
    os.close(descriptor)
    try:
        yield
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _source_run_binding(run_path: Path) -> dict[str, Any]:
    """Return the exact immutable source identity retained by a review."""

    manifest_path = run_path / "manifest.json"
    checksum_path = run_path / "SHA256SUMS"
    manifest_snapshot = _snapshot_file(
        manifest_path,
        name="source run manifest",
    )
    checksum_snapshot = _snapshot_file(
        checksum_path,
        name="source run checksum manifest",
    )
    manifest = _json_object_from_snapshot(manifest_snapshot)
    return {
        "run_id": manifest["run_id"],
        "run_manifest_sha256": manifest_snapshot.sha256,
        "run_checksum_manifest_sha256": checksum_snapshot.sha256,
        "config_sha256": manifest["config_sha256"],
        "source_sha256": manifest["source_sha256"],
        "verified_complete": True,
    }


def _assert_source_run_unchanged(
    supplied_path: Path,
    run_path: Path,
    source_run: Mapping[str, Any],
) -> None:
    try:
        same_path = (
            not supplied_path.is_symlink()
            and supplied_path.is_dir()
            and supplied_path.resolve() == run_path
        )
    except OSError:
        same_path = False
    if not same_path:
        raise ValueError(
            "source run path changed while the Gate 4 import was running"
        )
    valid, errors = verify_run(run_path)
    if not valid:
        raise ValueError(
            "source run changed while the Gate 4 import was running: "
            + "; ".join(errors)
        )
    if _source_run_binding(run_path) != dict(source_run):
        raise ValueError(
            "source run binding changed while the Gate 4 import was running"
        )


def _snapshot_collection_files(
    collection: Path,
    *,
    file_names: Mapping[str, str],
    lock_names: Sequence[str],
    label: str,
) -> dict[str, _FileSnapshot]:
    """Snapshot an exact, flat collector output while its locks are held."""

    required_names = set(file_names.values()) | set(lock_names)
    if collection.is_symlink() or not collection.is_dir():
        raise ValueError(f"{label} must remain a safe collection directory")
    actual_names = {item.name for item in collection.iterdir()}
    if actual_names != required_names:
        raise ValueError(
            f"{label} has missing or unexpected entries: "
            + canonical_json(
                {
                    "missing": sorted(required_names - actual_names),
                    "unexpected": sorted(actual_names - required_names),
                }
            )
        )
    snapshots = {
        key: _snapshot_file(
            collection / filename,
            name=f"{label} file {filename}",
        )
        for key, filename in file_names.items()
    }
    for filename in lock_names:
        lock_path = collection / filename
        if (
            lock_path.is_symlink()
            or not lock_path.is_file()
            or lock_path.resolve().parent != collection
        ):
            raise ValueError(f"{label} lacks its safe collection lock")
    return snapshots


def _assert_collection_unchanged(
    supplied_path: Path,
    collection: Path,
    snapshots: Mapping[str, _FileSnapshot],
    *,
    file_names: Mapping[str, str],
    lock_names: Sequence[str],
    label: str,
) -> None:
    try:
        same_path = (
            not supplied_path.is_symlink()
            and supplied_path.is_dir()
            and supplied_path.resolve() == collection
        )
    except OSError:
        same_path = False
    if not same_path:
        raise ValueError(
            f"{label} changed while the Gate 4 import was running"
        )
    current = _snapshot_collection_files(
        collection,
        file_names=file_names,
        lock_names=lock_names,
        label=label,
    )
    if set(current) != set(snapshots) or any(
        current[key].path != snapshots[key].path
        or current[key].material != snapshots[key].material
        for key in snapshots
    ):
        raise ValueError(
            f"{label} changed while the Gate 4 import was running"
        )


def _assert_collection_manifest_bindings(
    inputs: Mapping[str, Mapping[str, Any]],
    snapshots: Mapping[str, _FileSnapshot],
    *,
    label: str,
) -> None:
    for key, snapshot in snapshots.items():
        entry = inputs.get(key)
        if (
            not isinstance(entry, Mapping)
            or entry.get("filename") != snapshot.path.name
            or entry.get("sha256") != snapshot.sha256
            or entry.get("bytes") != len(snapshot.material)
        ):
            raise ValueError(
                f"{label} digest binding changed during validation"
            )


@contextmanager
def _hold_shared_collection_locks(
    locks: Sequence[tuple[str, Path]],
) -> Iterable[None]:
    """Hold collector-compatible shared locks through validation and output."""

    descriptors: list[tuple[int, str]] = []
    try:
        for label, lock_path in locks:
            if lock_path.is_symlink() or not lock_path.is_file():
                raise ValueError(f"{label} lacks its safe collection lock")
            descriptor = os.open(lock_path, os.O_RDWR)
            try:
                acquired = try_file_lock(descriptor, shared=True)
            except Exception:
                os.close(descriptor)
                raise
            if not acquired:
                os.close(descriptor)
                raise ValueError(
                    f"{label} is currently locked by a collector"
                )
            descriptors.append((descriptor, label))
        yield
    finally:
        for descriptor, _ in reversed(descriptors):
            try:
                unlock_file(descriptor)
            finally:
                os.close(descriptor)


def _validate_native_action_collection(
    collection_dir: str | Path,
    run_path: Path,
) -> tuple[
    tuple[NativeTerminalActionRecord, ...],
    dict[str, dict[str, Any]],
    dict[str, Any],
]:
    """Revalidate one complete OpenAI native-action collection."""

    unresolved_collection = Path(collection_dir)
    if unresolved_collection.is_symlink() or not unresolved_collection.is_dir():
        raise ValueError(
            "Gate 4 requires a complete native action collection directory; "
            "a standalone native-actions.jsonl file is ineligible"
        )
    collection = unresolved_collection.resolve()
    if collection == run_path or run_path in collection.parents:
        raise ValueError(
            "native action collection cannot equal or be inside the source run"
        )
    required_names = set(_NATIVE_COLLECTION_FILES.values())
    allowed_names = required_names | {".collection.lock"}
    actual_names = {item.name for item in collection.iterdir()}
    missing = required_names - actual_names
    unexpected = actual_names - allowed_names
    if missing or unexpected:
        raise ValueError(
            "native action collection has missing or unexpected entries: "
            + canonical_json(
                {
                    "missing": sorted(missing),
                    "unexpected": sorted(unexpected),
                }
            )
        )
    paths: dict[str, Path] = {}
    for key, filename in _NATIVE_COLLECTION_FILES.items():
        candidate = collection / filename
        if (
            candidate.is_symlink()
            or not candidate.is_file()
            or candidate.resolve().parent != collection
        ):
            raise ValueError(
                f"native action collection file is unsafe: {filename}"
            )
        paths[key] = candidate
    lock_path = collection / ".collection.lock"
    if lock_path.is_symlink() or not lock_path.is_file():
        raise ValueError(
            "native action collection lacks its safe collection lock"
        )

    # Local imports avoid the gate_review <-> native_action_provider import
    # cycle while reusing the collector's exact plan, journal, and audit
    # validators.
    from .native_action_provider import (
        MODEL_SELECTION_RESOLVED_ON,
        NATIVE_ACTION_SYSTEM_ID,
        OPENAI_NATIVE_ACTION_REFERENCE,
        OPENAI_STRUCTURED_OUTPUT_REFERENCE,
        OpenAINativeActionProvider,
        _DurableAttemptLedger,
        _audit_rows,
        _build_collection_plan,
        _collection_config,
        _validate_audit_record,
        build_native_action_requests,
    )
    from .openai_provider import OpenAIProviderConfig

    plan = _read_json_object(paths["native_collection_plan"])
    collection_config = plan.get("collection_config")
    if not isinstance(collection_config, Mapping):
        raise ValueError("native action collection plan lacks collection_config")
    if (
        collection_config.get("provider") != "openai"
        or collection_config.get("model") != _GATE4_NATIVE_MODEL
        or collection_config.get("reasoning_effort")
        != _GATE4_NATIVE_REASONING_EFFORT
        or collection_config.get("base_url") != _GATE4_NATIVE_ORIGIN
        or collection_config.get("endpoint")
        != f"{_GATE4_NATIVE_ORIGIN}/v1/responses"
        or collection_config.get("allow_custom_base_url") is not False
        or collection_config.get("official_origin_locked") is not True
    ):
        raise ValueError(
            "Gate 4 native evidence requires the implemented official "
            "OpenAI gpt-5.6-sol/medium adapter and origin"
        )
    try:
        provider_config = OpenAIProviderConfig(
            model=collection_config["model"],
            reasoning_effort=collection_config["reasoning_effort"],
            api_key_env=collection_config["api_key_env"],
            base_url=collection_config["base_url"],
            allow_custom_base_url=collection_config[
                "allow_custom_base_url"
            ],
            timeout_seconds=collection_config["timeout_seconds"],
            max_retries=collection_config["max_retries"],
            initial_backoff_seconds=collection_config[
                "initial_backoff_seconds"
            ],
            max_backoff_seconds=collection_config[
                "max_backoff_seconds"
            ],
            jitter_fraction=collection_config["jitter_fraction"],
            max_output_tokens=collection_config["max_output_tokens"],
            max_requests=collection_config["max_requests"],
            max_total_tokens=collection_config["max_total_tokens"],
            live_execution=False,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "native action collection has an invalid provider configuration"
        ) from exc
    if dict(collection_config) != _collection_config(provider_config):
        raise ValueError(
            "native action collection configuration is incomplete or altered"
        )
    if (
        provider_config.max_requests > _GATE4_NATIVE_MAX_REQUESTS
        or provider_config.max_total_tokens
        > _GATE4_NATIVE_MAX_TOTAL_TOKENS
        or provider_config.max_output_tokens
        > _GATE4_NATIVE_MAX_OUTPUT_TOKENS
    ):
        raise ValueError(
            "native action collection exceeds the approved Gate 4 ceilings "
            "(900 requests, 6000000 total tokens, 4096 output tokens)"
        )

    requests = build_native_action_requests(run_path)
    provider = OpenAINativeActionProvider(provider_config)
    prepared = tuple(provider.prepare(request) for request in requests)
    expected_plan = _build_collection_plan(
        run_path,
        provider_config,
        requests,
        prepared,
    )
    if plan != expected_plan:
        raise ValueError(
            "native action collection plan does not match the verified "
            "source run and implemented adapter"
        )
    if expected_plan.get("within_declared_budget") is not True:
        raise ValueError(
            "native action collection retry-expanded plan exceeds its "
            "declared hard budget"
        )
    expected_request_bytes = "".join(
        canonical_json(request.to_dict()) + "\n" for request in requests
    ).encode("utf-8")
    if paths["native_action_requests"].read_bytes() != expected_request_bytes:
        raise ValueError(
            "native action requests do not exactly match the verified source run"
        )

    request_by_id = {request.request_id: request for request in requests}
    prepared_by_id = {
        request.request_id: item
        for request, item in zip(requests, prepared)
    }
    ledger = _DurableAttemptLedger(
        paths["native_transport_attempts"],
        collection_plan_sha256=plan["plan_sha256"],
        collection_config_sha256=plan["collection_config_sha256"],
    )
    ledger.validate_bindings(request_by_id, provider)
    if ledger.unresolved_attempt_ids:
        raise ValueError(
            "native action collection has unresolved transport attempts"
        )
    embedded_audits = ledger.embedded_final_audits()
    if set(embedded_audits) != set(request_by_id):
        raise ValueError(
            "native action attempt journal lacks one final audit per request"
        )
    attempts_by_request: dict[str, list[tuple[int, bool]]] = {}
    for attempt_id, start in ledger.starts.items():
        settlement = ledger.settlements.get(attempt_id)
        if settlement is None:
            raise ValueError(
                "native action collection has an unsettled transport attempt"
            )
        attempt_started_text = _validate_timestamp(
            start.get("started_at"),
            "native action attempt started_at",
        )
        attempt_settled_text = _validate_timestamp(
            settlement.get("settled_at"),
            "native action attempt settled_at",
        )
        if datetime.fromisoformat(
            attempt_settled_text.replace("Z", "+00:00")
        ) < datetime.fromisoformat(
            attempt_started_text.replace("Z", "+00:00")
        ):
            raise ValueError(
                "native action attempt settled_at precedes started_at"
            )
        attempts_by_request.setdefault(start["request_id"], []).append(
            (
                start["attempt_ordinal"],
                isinstance(settlement.get("provider_audit"), Mapping),
            )
        )
    final_ordinal_by_request: dict[str, int] = {}
    for request_id, attempt_rows in attempts_by_request.items():
        ordered = sorted(attempt_rows)
        final_ordinals = [
            ordinal for ordinal, is_final in ordered if is_final
        ]
        if (
            len(final_ordinals) != 1
            or final_ordinals[0] != ordered[-1][0]
            or final_ordinals[0] > provider_config.max_retries + 1
        ):
            raise ValueError(
                "native action collection has attempts after or without its "
                f"final provider audit for {request_id}"
            )
        final_ordinal_by_request[request_id] = final_ordinals[0]

    audit_rows = _audit_rows(paths["native_provider_audit"])
    audits_by_request: dict[str, dict[str, Any]] = {}
    accepted_records: dict[str, NativeTerminalActionRecord] = {}
    for audit in audit_rows:
        request_id = audit.get("request_id")
        if (
            not isinstance(request_id, str)
            or request_id not in request_by_id
            or request_id in audits_by_request
        ):
            raise ValueError(
                "native action provider audit has an invalid request identity"
            )
        if audit.get("acceptance_status") != "accepted":
            raise ValueError(
                "Gate 4 cannot admit rejected native action provider responses"
            )
        record = _validate_audit_record(
            audit,
            request_by_id[request_id],
            prepared_by_id[request_id],
            provider_config,
        )
        if audit.get("attempts") != final_ordinal_by_request[request_id]:
            raise ValueError(
                "native action provider audit attempt count does not match "
                "its transport journal"
            )
        if dict(audit) != embedded_audits.get(request_id):
            raise ValueError(
                "native action provider audit differs from the attempt journal"
            )
        audits_by_request[request_id] = dict(audit)
        accepted_records[request_id] = record
    if set(audits_by_request) != set(request_by_id):
        raise ValueError(
            "native action provider audits do not cover requests exactly"
        )

    actions = read_native_terminal_action_records(
        paths["native_terminal_actions"]
    )
    actions_by_trajectory = {
        record.trajectory_id: record for record in actions
    }
    expected_actions = {
        request_by_id[request_id].trajectory_id: record
        for request_id, record in accepted_records.items()
    }
    if set(actions_by_trajectory) != set(expected_actions) or any(
        actions_by_trajectory[trajectory_id].to_dict()
        != expected_actions[trajectory_id].to_dict()
        for trajectory_id in expected_actions
    ):
        raise ValueError(
            "native action records differ from accepted provider audits"
        )

    execution_manifest = _read_json_object(
        paths["native_execution_manifest"]
    )
    expected_manifest_fields = {
        "schema_version",
        "workflow",
        "status",
        "claim_status",
        "source_run_id",
        "source_run_manifest_sha256",
        "source_run_checksums_sha256",
        "native_system_id",
        "native_system_version",
        "model",
        "reasoning_effort",
        "request_count",
        "action_record_count",
        "reused_request_count",
        "new_request_count",
        "request_budget_used",
        "transport_attempt_count",
        "token_budget_used",
        "budget_accounting_unit",
        "max_requests",
        "max_total_tokens",
        "credentials_retained",
        "collection_plan_file",
        "collection_plan_sha256",
        "collection_plan_file_sha256",
        "collection_config",
        "collection_config_sha256",
        "requests_sha256",
        "transport_attempts_sha256",
        "provider_audit_sha256",
        "native_actions_sha256",
        "transport_attempts_file",
        "native_actions_file",
        "provider_audit_file",
        "official_references",
        "resolved_on",
    }
    if set(execution_manifest) != expected_manifest_fields:
        raise ValueError(
            "native action execution manifest has missing or unknown fields"
        )
    attempt_count, token_count = ledger.accounting()
    manifest_expected = {
        "schema_version": 1,
        "workflow": "native_terminal_actions",
        "status": "complete",
        "claim_status": "not_claimed",
        "source_run_id": plan["source_run_id"],
        "source_run_manifest_sha256": plan[
            "source_run_manifest_sha256"
        ],
        "source_run_checksums_sha256": plan[
            "source_run_checksums_sha256"
        ],
        "native_system_id": NATIVE_ACTION_SYSTEM_ID,
        "native_system_version": (
            f"openai-responses:{_GATE4_NATIVE_MODEL}"
        ),
        "model": _GATE4_NATIVE_MODEL,
        "reasoning_effort": _GATE4_NATIVE_REASONING_EFFORT,
        "request_count": len(requests),
        "action_record_count": len(actions),
        "request_budget_used": attempt_count,
        "transport_attempt_count": attempt_count,
        "token_budget_used": token_count,
        "budget_accounting_unit": "actual_transport_attempt",
        "max_requests": provider_config.max_requests,
        "max_total_tokens": provider_config.max_total_tokens,
        "credentials_retained": False,
        "collection_plan_file": "collection-plan.json",
        "collection_plan_sha256": plan["plan_sha256"],
        "collection_plan_file_sha256": _file_digest(
            paths["native_collection_plan"]
        ),
        "collection_config": dict(collection_config),
        "collection_config_sha256": plan["collection_config_sha256"],
        "requests_sha256": _file_digest(paths["native_action_requests"]),
        "transport_attempts_sha256": _file_digest(
            paths["native_transport_attempts"]
        ),
        "provider_audit_sha256": _file_digest(
            paths["native_provider_audit"]
        ),
        "native_actions_sha256": _file_digest(
            paths["native_terminal_actions"]
        ),
        "transport_attempts_file": "transport-attempts.jsonl",
        "native_actions_file": "native-actions.jsonl",
        "provider_audit_file": "provider-audit.jsonl",
        "official_references": [
            OPENAI_NATIVE_ACTION_REFERENCE,
            OPENAI_STRUCTURED_OUTPUT_REFERENCE,
        ],
        "resolved_on": MODEL_SELECTION_RESOLVED_ON,
    }
    for name, expected in manifest_expected.items():
        if execution_manifest.get(name) != expected:
            raise ValueError(
                f"native action execution manifest mismatch for {name}"
            )
    reused = execution_manifest.get("reused_request_count")
    created = execution_manifest.get("new_request_count")
    if (
        not isinstance(reused, int)
        or isinstance(reused, bool)
        or reused < 0
        or not isinstance(created, int)
        or isinstance(created, bool)
        or created < 0
        or reused + created != len(requests)
        or attempt_count > provider_config.max_requests
        or token_count > provider_config.max_total_tokens
    ):
        raise ValueError(
            "native action execution manifest has invalid completion accounting"
        )

    inputs = {
        key: _input_manifest_entry(
            paths[key],
            record_count=(
                len(requests)
                if key == "native_action_requests"
                else len(ledger.starts) + len(ledger.settlements)
                if key == "native_transport_attempts"
                else len(audit_rows)
                if key == "native_provider_audit"
                else len(actions)
                if key == "native_terminal_actions"
                else None
            ),
        )
        for key in _NATIVE_COLLECTION_FILES
    }
    summary = {
        "collection_status": "complete",
        "source_run_id": plan["source_run_id"],
        "native_system_id": NATIVE_ACTION_SYSTEM_ID,
        "native_system_version": execution_manifest[
            "native_system_version"
        ],
        "provider": "openai",
        "model": _GATE4_NATIVE_MODEL,
        "reasoning_effort": _GATE4_NATIVE_REASONING_EFFORT,
        "official_origin": _GATE4_NATIVE_ORIGIN,
        "official_origin_locked": True,
        "request_count": len(requests),
        "transport_attempt_count": attempt_count,
        "provider_audit_count": len(audit_rows),
        "action_record_count": len(actions),
        "all_collection_files_digest_bound": True,
        "requests_rebuilt_from_verified_source": True,
        "collection_plan_rebuilt_and_validated": True,
        "attempt_journal_validated": True,
        "provider_audits_validated": True,
        "actions_match_accepted_provider_audits": True,
        "legacy_standalone_action_files_eligible": False,
    }
    return actions, inputs, summary


def _validate_external_provider_audit(
    audit: Mapping[str, Any],
    *,
    request: ExternalDecoderRequest,
    provider: Any,
    final_attempt_ordinal: int,
) -> ExternalDecoderJudgment:
    """Reconstruct one accepted decoder result from its retained audit."""

    from .decoder_study import external_decoder_judgment_from_response
    from .external_decoder_providers import (
        ExternalDecoderProviderResult,
        _parse_anthropic_response,
        _parse_gemini_response,
        _validate_resumed_audit,
    )
    from .llm_exchange import LLMResponse

    if audit.get("acceptance_status") != "accepted":
        raise ValueError(
            "Gate 4 cannot admit rejected external decoder responses"
        )
    judgment = _validate_resumed_audit(audit, request, provider)
    llm_raw = audit.get("llm_response")
    raw_response = audit.get("raw_response")
    usage = audit.get("usage")
    if (
        not isinstance(llm_raw, Mapping)
        or not isinstance(raw_response, Mapping)
        or not isinstance(usage, Mapping)
    ):
        raise ValueError(
            "external decoder audit lacks structured response evidence"
        )
    llm_response = LLMResponse.parse(llm_raw)
    _validate_digest(
        llm_response.raw_response_sha256,
        "external decoder raw_response_sha256",
    )
    regenerated = external_decoder_judgment_from_response(
        request,
        llm_response,
        decoder_instance_id=provider.config.decoder_instance_id,
        decoder_family_id=provider.config.source.decoder_family_id,
        source_descriptor=provider.config.source_descriptor,
    )
    if judgment.to_dict() != regenerated.to_dict():
        raise ValueError(
            "external decoder audit judgment does not match its LLM response"
        )
    parse_raw = (
        _parse_anthropic_response
        if provider.config.provider == "anthropic"
        else _parse_gemini_response
    )
    try:
        returned_model, response_id, payload, parsed_usage = parse_raw(
            raw_response,
            # NUL cannot occur in accepted provider identifiers and is
            # serialized escaped in JSON, so it acts as a nonmatching
            # redaction sentinel without retaining a credential.
            secret="\0",
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "external decoder audit raw response is not valid provider output"
        ) from exc
    if (
        audit.get("model_returned") != returned_model
        or audit.get("provider_response_id") != response_id
        or dict(usage) != dict(parsed_usage)
        or llm_response.model_id != returned_model
        or payload.get("beliefs") != llm_response.beliefs
    ):
        raise ValueError(
            "external decoder audit response identity, usage, or beliefs "
            "do not match its retained raw response"
        )
    attempts = audit.get("attempts")
    if (
        not isinstance(attempts, int)
        or isinstance(attempts, bool)
        or attempts != final_attempt_ordinal
    ):
        raise ValueError(
            "external decoder audit attempt count does not match its journal"
        )
    started_text = _validate_timestamp(
        audit.get("started_at"),
        "external decoder started_at",
    )
    completed_text = _validate_timestamp(
        audit.get("completed_at"),
        "external decoder completed_at",
    )
    started = datetime.fromisoformat(started_text.replace("Z", "+00:00"))
    completed = datetime.fromisoformat(completed_text.replace("Z", "+00:00"))
    if completed < started:
        raise ValueError(
            "external decoder audit completed_at precedes started_at"
        )
    server_request_id = audit.get("server_request_id")
    if server_request_id is not None:
        _require_text(server_request_id, "external decoder server_request_id")
    result = ExternalDecoderProviderResult(
        judgment=judgment,
        llm_response=llm_response,
        provider=provider.config.provider,
        model_requested=str(provider.config.model),
        model_returned=returned_model,
        provider_response_id=_require_text(
            response_id,
            "external decoder provider_response_id",
        ),
        usage=dict(usage),
        started_at=started_text,
        completed_at=completed_text,
        attempts=attempts,
        request_body_sha256=_validate_digest(
            audit.get("request_body_sha256"),
            "external decoder request_body_sha256",
        ),
        client_request_id=_require_text(
            audit.get("client_request_id"),
            "external decoder client_request_id",
        ),
        server_request_id=server_request_id,
        estimated_max_tokens=audit.get("estimated_max_tokens"),
        raw_response=dict(raw_response),
    )
    if dict(audit) != result.to_audit_record():
        raise ValueError(
            "external decoder provider audit has missing, unknown, or "
            "internally inconsistent fields"
        )
    return judgment


def _validate_external_decoder_collection(
    collection_dir: str | Path,
    *,
    run_path: Path,
    requests: Sequence[ExternalDecoderRequest],
    supplied_judgments: _FileSnapshot,
) -> tuple[
    tuple[ExternalDecoderJudgment, ...],
    dict[str, dict[str, Any]],
    dict[str, Any],
]:
    """Revalidate one complete official Anthropic/Gemini collection."""

    from .external_decoder_providers import (
        ANTHROPIC_DEFAULT_MODEL,
        ANTHROPIC_OFFICIAL_ORIGIN,
        GEMINI_DEFAULT_MODEL,
        GEMINI_OFFICIAL_ORIGIN,
        ExternalDecoderProvider,
        ExternalDecoderProviderConfig,
        _DurableAttemptLedger,
        _read_audits,
        plan_external_decoder_collection,
    )

    unresolved_collection = Path(collection_dir)
    if (
        unresolved_collection.is_symlink()
        or not unresolved_collection.is_dir()
    ):
        raise ValueError(
            "official external decoder evidence requires a complete "
            "distinct-decoder collection directory"
        )
    collection = unresolved_collection.resolve()
    if collection == run_path or run_path in collection.parents:
        raise ValueError(
            "external decoder collection cannot equal or be inside the "
            "source run"
        )
    required_names = set(_EXTERNAL_COLLECTION_FILES.values()) | set(
        _EXTERNAL_COLLECTION_LOCKS
    )
    actual_names = {item.name for item in collection.iterdir()}
    if actual_names != required_names:
        raise ValueError(
            "external decoder collection has missing or unexpected entries: "
            + canonical_json(
                {
                    "missing": sorted(required_names - actual_names),
                    "unexpected": sorted(actual_names - required_names),
                }
            )
        )
    paths: dict[str, Path] = {}
    for key, filename in _EXTERNAL_COLLECTION_FILES.items():
        candidate = collection / filename
        if (
            candidate.is_symlink()
            or not candidate.is_file()
            or candidate.resolve().parent != collection
        ):
            raise ValueError(
                f"external decoder collection file is unsafe: {filename}"
            )
        paths[key] = candidate
    for filename in _EXTERNAL_COLLECTION_LOCKS:
        lock_path = collection / filename
        if lock_path.is_symlink() or not lock_path.is_file():
            raise ValueError(
                "external decoder collection lacks its safe collection locks"
            )

    plan = _read_json_object(paths["decoder_collection_plan"])
    source_rows = plan.get("sources")
    if (
        not isinstance(source_rows, list)
        or len(source_rows) != 2
        or any(not isinstance(row, Mapping) for row in source_rows)
    ):
        raise ValueError(
            "external decoder collection plan must declare exactly two sources"
        )
    selected_models = {
        "anthropic": (ANTHROPIC_DEFAULT_MODEL, ANTHROPIC_OFFICIAL_ORIGIN),
        "google_gemini": (
            GEMINI_DEFAULT_MODEL,
            GEMINI_OFFICIAL_ORIGIN,
        ),
    }
    configs = []
    seen_providers: set[str] = set()
    for raw_source in source_rows:
        source = dict(raw_source)
        provider_name = source.get("provider")
        if (
            not isinstance(provider_name, str)
            or provider_name not in selected_models
            or provider_name in seen_providers
        ):
            raise ValueError(
                "external decoder collection does not contain the selected "
                "distinct provider pair"
            )
        seen_providers.add(provider_name)
        selected_model, official_origin = selected_models[provider_name]
        expected_endpoint = (
            f"{official_origin}/v1/messages"
            if provider_name == "anthropic"
            else (
                f"{official_origin}/v1beta/models/"
                f"{selected_model}:generateContent"
            )
        )
        if (
            source.get("model") != selected_model
            or source.get("official_origin_locked") is not True
            or source.get("endpoint") != expected_endpoint
        ):
            raise ValueError(
                "Gate 4 official decoder evidence requires the selected "
                "Anthropic Claude Sonnet 5 and Gemini 3.6 Flash models at "
                "their first-party origins"
            )
        try:
            config = ExternalDecoderProviderConfig(
                provider=provider_name,
                model=source["model"],
                api_key_env=source["api_key_env"],
                base_url=official_origin,
                allow_custom_base_url=False,
                timeout_seconds=source["timeout_seconds"],
                max_retries=source["max_retries"],
                initial_backoff_seconds=source[
                    "initial_backoff_seconds"
                ],
                max_backoff_seconds=source["max_backoff_seconds"],
                jitter_fraction=source["jitter_fraction"],
                max_output_tokens=source["max_output_tokens"],
                max_requests=source["max_requests"],
                max_total_tokens=source["max_total_tokens"],
                live_execution=False,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "external decoder collection has an invalid source "
                "configuration"
            ) from exc
        if (
            config.max_requests > _GATE4_EXTERNAL_MAX_REQUESTS
            or config.max_total_tokens
            > _GATE4_EXTERNAL_MAX_TOTAL_TOKENS
            or config.max_output_tokens
            > _GATE4_EXTERNAL_MAX_OUTPUT_TOKENS
        ):
            raise ValueError(
                "external decoder collection exceeds the approved per-source "
                "Gate 4 ceilings (900 requests, 6000000 total tokens, 1024 "
                "output tokens)"
            )
        configs.append(config)
    if seen_providers != set(selected_models):
        raise ValueError(
            "external decoder collection lacks the selected provider pair"
        )
    expected_plan = plan_external_decoder_collection(requests, configs)
    if plan != expected_plan:
        raise ValueError(
            "external decoder collection plan does not match the retained "
            "requests and implemented provider adapters"
        )

    providers = tuple(
        ExternalDecoderProvider(config)
        for config in sorted(configs, key=lambda item: item.provider)
    )
    provider_by_name = {
        provider.config.provider: provider for provider in providers
    }
    request_by_id = {request.request_id: request for request in requests}
    ledger = _DurableAttemptLedger(
        paths["decoder_transport_attempts"]
    )
    ledger.validate_bindings(provider_by_name, request_by_id)
    if ledger.unresolved_attempt_ids or set(ledger.starts) != set(
        ledger.settlements
    ):
        raise ValueError(
            "external decoder collection has unresolved transport attempts"
        )
    expected_keys = {
        (provider_name, request_id)
        for provider_name in provider_by_name
        for request_id in request_by_id
    }
    embedded_audits = ledger.embedded_final_audits()
    if set(embedded_audits) != expected_keys:
        raise ValueError(
            "external decoder attempt journal lacks one final audit for every "
            "provider/request pair"
        )
    final_ordinals: dict[tuple[str, str], int] = {}
    attempts_by_key: dict[tuple[str, str], list[tuple[int, bool]]] = {}
    for attempt_id, start in ledger.starts.items():
        settlement = ledger.settlements[attempt_id]
        attempt_started_text = _validate_timestamp(
            start.get("started_at"),
            "external decoder attempt started_at",
        )
        attempt_settled_text = _validate_timestamp(
            settlement.get("settled_at"),
            "external decoder attempt settled_at",
        )
        if datetime.fromisoformat(
            attempt_settled_text.replace("Z", "+00:00")
        ) < datetime.fromisoformat(
            attempt_started_text.replace("Z", "+00:00")
        ):
            raise ValueError(
                "external decoder attempt settled_at precedes started_at"
            )
        key = (start["provider"], start["request_id"])
        attempts_by_key.setdefault(key, []).append(
            (
                start["attempt_ordinal"],
                isinstance(settlement.get("provider_audit"), Mapping),
            )
        )
    for key, attempt_rows in attempts_by_key.items():
        ordered = sorted(attempt_rows)
        audited = [ordinal for ordinal, final in ordered if final]
        if len(audited) != 1 or audited[0] != ordered[-1][0]:
            raise ValueError(
                "external decoder collection has attempts after or without "
                f"its final provider audit for {key}"
            )
        provider = provider_by_name[key[0]]
        if audited[0] > provider.config.max_retries + 1:
            raise ValueError(
                "external decoder collection exceeds its retry policy"
            )
        final_ordinals[key] = audited[0]

    audits = _read_audits(paths["decoder_provider_audit"])
    if set(audits) != expected_keys:
        raise ValueError(
            "external decoder provider audits do not cover requests exactly"
        )
    judgments_by_key: dict[
        tuple[str, str],
        ExternalDecoderJudgment,
    ] = {}
    ordered_judgments: list[ExternalDecoderJudgment] = []
    for key, audit in audits.items():
        if audit != embedded_audits[key]:
            raise ValueError(
                "external decoder provider audit differs from the transport "
                "attempt journal"
            )
        provider = provider_by_name[key[0]]
        judgment = _validate_external_provider_audit(
            audit,
            request=request_by_id[key[1]],
            provider=provider,
            final_attempt_ordinal=final_ordinals[key],
        )
        judgment_key = (
            judgment.decoder_instance_id,
            judgment.request_id,
        )
        if judgment_key in judgments_by_key:
            raise ValueError(
                "external decoder collection contains duplicate judgments"
            )
        judgments_by_key[judgment_key] = judgment
        ordered_judgments.append(judgment)

    expected_judgment_bytes = "".join(
        canonical_json(judgment.to_dict()) + "\n"
        for judgment in ordered_judgments
    ).encode("utf-8")
    retained_judgment_bytes = paths["decoder_judgments"].read_bytes()
    if retained_judgment_bytes != expected_judgment_bytes:
        raise ValueError(
            "external decoder judgments differ from accepted provider audits"
        )
    if supplied_judgments.material != retained_judgment_bytes:
        raise ValueError(
            "supplied judgments must be byte-identical to the selected "
            "external decoder collection"
        )
    judgments = tuple(ordered_judgments)
    source_design = validate_external_decoder_import(
        requests,
        judgments,
        minimum_sources_per_request=2,
        require_distinct_families=True,
    )
    if (
        not source_design.complete_coverage
        or not source_design.source_design_eligible
    ):
        raise ValueError(
            "external decoder collection lacks complete distinct-source "
            "coverage"
        )

    execution_manifest = _read_json_object(
        paths["decoder_execution_manifest"]
    )
    expected_manifest_fields = {
        "schema_version",
        "kind",
        "status",
        "claim_status",
        "collection_plan_sha256",
        "judgments_sha256",
        "provider_audit_sha256",
        "transport_attempts_sha256",
        "execution_summary",
        "source_design_audit",
        "distinct_provider_model_families",
        "statistical_independence_claimed",
        "responsible_researcher_source_review_required",
        "credentials_retained",
    }
    if set(execution_manifest) != expected_manifest_fields:
        raise ValueError(
            "external decoder execution manifest has missing or unknown fields"
        )
    fixed_manifest = {
        "schema_version": 1,
        "kind": "distinct-external-decoder-collection",
        "status": "complete",
        "claim_status": "not_claimed",
        "collection_plan_sha256": _file_digest(
            paths["decoder_collection_plan"]
        ),
        "judgments_sha256": _file_digest(paths["decoder_judgments"]),
        "provider_audit_sha256": _file_digest(
            paths["decoder_provider_audit"]
        ),
        "transport_attempts_sha256": _file_digest(
            paths["decoder_transport_attempts"]
        ),
        "source_design_audit": source_design.to_dict(),
        "distinct_provider_model_families": True,
        "statistical_independence_claimed": False,
        "responsible_researcher_source_review_required": True,
        "credentials_retained": False,
    }
    for name, expected in fixed_manifest.items():
        if execution_manifest.get(name) != expected:
            raise ValueError(
                f"external decoder execution manifest mismatch for {name}"
            )
    execution_summary = execution_manifest.get("execution_summary")
    expected_summary_fields = {
        "schema_version",
        "request_count",
        "source_count",
        "judgment_count",
        "resumed_count",
        "executed_count",
        "transport_attempts_by_provider",
        "total_tokens_by_provider",
        "judgments_path",
        "audit_path",
        "attempt_path",
        "repaired_trailing_files",
    }
    if (
        not isinstance(execution_summary, Mapping)
        or set(execution_summary) != expected_summary_fields
    ):
        raise ValueError(
            "external decoder execution summary has missing or unknown fields"
        )
    attempts_by_provider = {
        provider_name: ledger.accounting_for(provider_name)[0]
        for provider_name in provider_by_name
    }
    tokens_by_provider = {
        provider_name: ledger.accounting_for(provider_name)[1]
        for provider_name in provider_by_name
    }
    expected_summary = {
        "schema_version": 1,
        "request_count": len(requests),
        "source_count": len(providers),
        "judgment_count": len(judgments),
        "transport_attempts_by_provider": attempts_by_provider,
        "total_tokens_by_provider": tokens_by_provider,
        "judgments_path": "judgments.jsonl",
        "audit_path": "provider-audit.jsonl",
        "attempt_path": "transport-attempts.jsonl",
    }
    for name, expected in expected_summary.items():
        if execution_summary.get(name) != expected:
            raise ValueError(
                f"external decoder execution summary mismatch for {name}"
            )
    resumed = execution_summary.get("resumed_count")
    executed = execution_summary.get("executed_count")
    repaired = execution_summary.get("repaired_trailing_files")
    allowed_repaired = {
        "judgments.jsonl",
        "provider-audit.jsonl",
        "transport-attempts.jsonl",
    }
    if (
        not isinstance(resumed, int)
        or isinstance(resumed, bool)
        or resumed < 0
        or not isinstance(executed, int)
        or isinstance(executed, bool)
        or executed < 0
        or resumed + executed != len(judgments)
        or not isinstance(repaired, list)
        or any(
            not isinstance(name, str) or name not in allowed_repaired
            for name in repaired
        )
        or len(set(repaired)) != len(repaired)
    ):
        raise ValueError(
            "external decoder execution summary has invalid completion or "
            "portable-path accounting"
        )
    for provider_name, provider in provider_by_name.items():
        if (
            attempts_by_provider[provider_name]
            > provider.config.max_requests
            or tokens_by_provider[provider_name]
            > provider.config.max_total_tokens
        ):
            raise ValueError(
                "external decoder execution exceeds its declared budget"
            )

    inputs = {
        key: _input_manifest_entry(
            paths[key],
            record_count=(
                len(ledger.starts) + len(ledger.settlements)
                if key == "decoder_transport_attempts"
                else len(audits)
                if key == "decoder_provider_audit"
                else len(judgments)
                if key == "decoder_judgments"
                else None
            ),
        )
        for key in _EXTERNAL_COLLECTION_FILES
    }
    summary = {
        "provenance_mode": DIRECT_FIRST_PARTY_COLLECTION_PROVENANCE,
        "collection_status": "complete",
        "providers": [
            {
                "provider": provider.config.provider,
                "model": provider.config.model,
                "decoder_instance_id": (
                    provider.config.decoder_instance_id
                ),
                "decoder_family_id": provider.config.source.decoder_family_id,
                "official_origin": provider.config.source.official_origin,
                "official_origin_locked": True,
                "transport_attempt_count": attempts_by_provider[
                    provider.config.provider
                ],
                "token_budget_used": tokens_by_provider[
                    provider.config.provider
                ],
            }
            for provider in providers
        ],
        "request_count": len(requests),
        "source_count": len(providers),
        "judgment_count": len(judgments),
        "all_collection_files_digest_bound": True,
        "plan_rebuilt_from_retained_requests": True,
        "attempt_journal_validated": True,
        "provider_audits_validated": True,
        "judgments_match_accepted_provider_audits": True,
        "standalone_automated_provider_files_eligible": False,
    }
    return judgments, inputs, summary


def validate_official_external_decoder_collection(
    collection_dir: str | Path,
    *,
    run_dir: str | Path,
    requests: Sequence[ExternalDecoderRequest],
    judgments_path: str | Path,
) -> tuple[
    tuple[ExternalDecoderJudgment, ...],
    dict[str, dict[str, Any]],
    dict[str, Any],
]:
    """Read-only validation of one complete first-party decoder collection."""

    supplied_run = Path(run_dir).absolute()
    supplied_collection = Path(collection_dir).absolute()
    supplied_judgments = Path(judgments_path).absolute()
    if supplied_run.is_symlink() or not supplied_run.is_dir():
        raise ValueError("source run must be a safe directory")
    if (
        supplied_collection.is_symlink()
        or not supplied_collection.is_dir()
    ):
        raise ValueError(
            "official external decoder evidence requires a complete "
            "distinct-decoder collection directory"
        )
    run_path = supplied_run.resolve()
    collection = supplied_collection.resolve()
    valid, errors = verify_run(run_path)
    if not valid:
        raise ValueError(
            "source run verification failed: " + "; ".join(errors)
        )
    source_run = _source_run_binding(run_path)
    judgment_snapshot = _snapshot_file(
        supplied_judgments,
        name="decoder judgments",
    )
    request_tuple = tuple(requests)
    locks = tuple(
        (
            label,
            collection / filename,
        )
        for label, filename in (
            (
                "external decoder command collection",
                _EXTERNAL_COLLECTION_LOCKS[0],
            ),
            (
                "external decoder journal collection",
                _EXTERNAL_COLLECTION_LOCKS[1],
            ),
        )
    )
    with _hold_shared_collection_locks(locks):
        snapshots = _snapshot_collection_files(
            collection,
            file_names=_EXTERNAL_COLLECTION_FILES,
            lock_names=_EXTERNAL_COLLECTION_LOCKS,
            label="external decoder collection",
        )
        judgments, inputs, summary = (
            _validate_external_decoder_collection(
                collection,
                run_path=run_path,
                requests=request_tuple,
                supplied_judgments=judgment_snapshot,
            )
        )
        _assert_snapshot_unchanged(
            supplied_judgments,
            judgment_snapshot,
            name="decoder judgments",
        )
        _assert_collection_unchanged(
            supplied_collection,
            collection,
            snapshots,
            file_names=_EXTERNAL_COLLECTION_FILES,
            lock_names=_EXTERNAL_COLLECTION_LOCKS,
            label="external decoder collection",
        )
        _assert_collection_manifest_bindings(
            inputs,
            snapshots,
            label="external decoder collection",
        )
        _assert_source_run_unchanged(
            supplied_run,
            run_path,
            source_run,
        )
        return judgments, inputs, summary


def validate_selected_external_decoder_collection(
    collection_dir: str | Path,
    *,
    run_dir: str | Path,
    requests: Sequence[ExternalDecoderRequest],
    judgments_path: str | Path,
    expected_provenance_mode: str | None = None,
) -> tuple[
    tuple[ExternalDecoderJudgment, ...],
    dict[str, dict[str, Any]],
    dict[str, Any],
]:
    """Validate the selected direct or shared-gateway decoder collection.

    When ``expected_provenance_mode`` is supplied, validation fails before
    admitting an artifact of the opposite collection kind. This preserves
    the semantic distinction carried by the two CLI collection flags.
    """

    is_openrouter = is_openrouter_decoder_collection(collection_dir)
    if expected_provenance_mode is not None:
        if expected_provenance_mode not in EXTERNAL_COLLECTION_PROVENANCE_MODES:
            raise ValueError(
                "external decoder collection provenance mode is invalid"
            )
        expected_openrouter = (
            expected_provenance_mode == OPENROUTER_COLLECTION_PROVENANCE
        )
        if is_openrouter is not expected_openrouter:
            expected_flag = (
                "--openrouter-collection-dir"
                if expected_openrouter
                else "--external-collection-dir"
            )
            actual = (
                "an OpenRouter shared-gateway collection"
                if is_openrouter
                else "a direct first-party collection"
            )
            raise ValueError(
                f"{expected_flag} does not match the supplied artifact: "
                f"detected {actual}"
            )

    if not is_openrouter:
        return validate_official_external_decoder_collection(
            collection_dir,
            run_dir=run_dir,
            requests=requests,
            judgments_path=judgments_path,
        )

    supplied_run = Path(run_dir).absolute()
    supplied_collection = Path(collection_dir).absolute()
    supplied_judgments = Path(judgments_path).absolute()
    if supplied_run.is_symlink() or not supplied_run.is_dir():
        raise ValueError("source run must be a safe directory")
    if (
        supplied_collection.is_symlink()
        or not supplied_collection.is_dir()
    ):
        raise ValueError(
            "selected OpenRouter decoder evidence requires a complete "
            "collection directory"
        )
    run_path = supplied_run.resolve()
    collection = supplied_collection.resolve()
    if collection == run_path or run_path in collection.parents:
        raise ValueError(
            "external decoder collection cannot equal or be inside the "
            "source run"
        )
    valid, errors = verify_run(run_path)
    if not valid:
        raise ValueError(
            "source run verification failed: " + "; ".join(errors)
        )
    source_run = _source_run_binding(run_path)
    judgment_snapshot = _snapshot_file(
        supplied_judgments,
        name="decoder judgments",
    )
    locks = tuple(
        (
            f"OpenRouter decoder collection lock {name}",
            collection / name,
        )
        for name in OPENROUTER_COLLECTION_LOCKS
    )
    with _hold_shared_collection_locks(locks):
        validated = validate_openrouter_decoder_collection(
            collection,
            requests=tuple(requests),
            judgments_path=supplied_judgments,
        )
        _assert_snapshot_unchanged(
            supplied_judgments,
            judgment_snapshot,
            name="decoder judgments",
        )
        repeated = validate_openrouter_decoder_collection(
            collection,
            requests=tuple(requests),
            judgments_path=supplied_judgments,
        )
        if repeated != validated:
            raise ValueError(
                "OpenRouter decoder collection changed during validation"
            )
        _assert_source_run_unchanged(
            supplied_run,
            run_path,
            source_run,
        )
        return validated


def import_native_gate_review(
    *,
    run_dir: str | Path,
    requests_path: str | Path,
    judgments_path: str | Path,
    truth_labels_path: str | Path,
    native_collection_dir: str | Path,
    source_review_path: str | Path,
    output_dir: str | Path,
    external_collection_dir: str | Path | None = None,
    external_collection_provenance_mode: str | None = None,
    allow_reviewed_generic_decoders: bool = False,
) -> dict[str, Any]:
    """Validate evidence and atomically publish an immutable review."""

    supplied_run = Path(run_dir).absolute()
    supplied_requests = Path(requests_path).absolute()
    supplied_judgments = Path(judgments_path).absolute()
    supplied_truth = Path(truth_labels_path).absolute()
    supplied_native = Path(native_collection_dir).absolute()
    supplied_review = Path(source_review_path).absolute()
    supplied_output = Path(output_dir).absolute()
    if supplied_run.is_symlink() or not supplied_run.is_dir():
        raise ValueError("source run must be a safe directory")
    if supplied_output.is_symlink() or supplied_output.exists():
        raise FileExistsError(
            f"gate-review output already exists: {supplied_output}"
        )
    if supplied_output.parent.is_symlink():
        raise ValueError("gate-review output parent cannot be a symlink")

    run_path = supplied_run.resolve()
    output = supplied_output.resolve()
    if output == Path(output.anchor):
        raise ValueError("gate-review output cannot be a filesystem root")
    if output == run_path or run_path in output.parents:
        raise ValueError(
            "gate-review output cannot be inside the completed source run"
        )
    if (
        external_collection_dir is not None
    ) == allow_reviewed_generic_decoders:
        raise ValueError(
            "choose exactly one external decoder provenance mode: provide "
            "external_collection_dir or set "
            "allow_reviewed_generic_decoders=True"
        )
    if (
        external_collection_dir is None
        and external_collection_provenance_mode is not None
    ):
        raise ValueError(
            "external decoder provenance mode requires a collection"
        )
    if (
        external_collection_provenance_mode is not None
        and external_collection_provenance_mode
        not in EXTERNAL_COLLECTION_PROVENANCE_MODES
    ):
        raise ValueError(
            "external decoder collection provenance mode is invalid"
        )
    unresolved_native = supplied_native
    if unresolved_native.is_symlink() or not unresolved_native.is_dir():
        raise ValueError(
            "Gate 4 requires a complete native action collection directory; "
            "a standalone native-actions.jsonl file is ineligible"
        )
    native_collection = unresolved_native.resolve()
    if output == native_collection or native_collection in output.parents:
        raise ValueError(
            "gate-review output cannot equal or be inside the native action "
            "collection"
        )
    external_collection: Path | None = None
    supplied_external: Path | None = None
    external_collection_is_openrouter = False
    locks: list[tuple[str, Path]] = []
    if external_collection_dir is not None:
        unresolved_external = Path(external_collection_dir).absolute()
        if (
            unresolved_external.is_symlink()
            or not unresolved_external.is_dir()
        ):
            raise ValueError(
                "official external decoder evidence requires a complete "
                "distinct-decoder collection directory"
            )
        external_collection = unresolved_external.resolve()
        if (
            output == external_collection
            or external_collection in output.parents
        ):
            raise ValueError(
                "gate-review output cannot equal or be inside the external "
                "decoder collection"
            )
        if external_collection == native_collection:
            raise ValueError(
                "external decoder and native action collections must differ"
            )
        supplied_external = unresolved_external
        external_collection_is_openrouter = is_openrouter_decoder_collection(
            external_collection
        )
        if external_collection_provenance_mode is not None:
            expected_openrouter = (
                external_collection_provenance_mode
                == OPENROUTER_COLLECTION_PROVENANCE
            )
            if external_collection_is_openrouter is not expected_openrouter:
                expected_flag = (
                    "--openrouter-collection-dir"
                    if expected_openrouter
                    else "--external-collection-dir"
                )
                actual = (
                    "an OpenRouter shared-gateway collection"
                    if external_collection_is_openrouter
                    else "a direct first-party collection"
                )
                raise ValueError(
                    f"{expected_flag} does not match the supplied artifact: "
                    f"detected {actual}"
                )
        external_lock_names = (
            OPENROUTER_COLLECTION_LOCKS
            if external_collection_is_openrouter
            else _EXTERNAL_COLLECTION_LOCKS
        )
        # Match the collector's outer-to-inner nesting order.
        locks.extend(
            (
                f"external decoder collection lock {name}",
                external_collection / name,
            )
            for name in external_lock_names
        )
    locks.append(
        (
            "native action collection",
            native_collection / ".collection.lock",
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with _hold_exclusive_gate_review_lock(output):
        if output.is_symlink() or output.exists():
            raise FileExistsError(
                f"gate-review output already exists: {output}"
            )
        with _hold_shared_collection_locks(locks):
            return _import_native_gate_review_locked(
                run_path=run_path,
                run_input=supplied_run,
                requests_path=supplied_requests,
                judgments_path=supplied_judgments,
                truth_labels_path=supplied_truth,
                native_collection_dir=native_collection,
                native_collection_input=supplied_native,
                source_review_path=supplied_review,
                output=output,
                external_collection_dir=external_collection,
                external_collection_input=supplied_external,
                external_collection_is_openrouter=(
                    external_collection_is_openrouter
                ),
                allow_reviewed_generic_decoders=(
                    allow_reviewed_generic_decoders
                ),
            )


def _import_native_gate_review_locked(
    *,
    run_path: Path,
    run_input: Path,
    requests_path: Path,
    judgments_path: Path,
    truth_labels_path: Path,
    native_collection_dir: Path,
    native_collection_input: Path,
    source_review_path: Path,
    output: Path,
    external_collection_dir: Path | None,
    external_collection_input: Path | None,
    external_collection_is_openrouter: bool,
    allow_reviewed_generic_decoders: bool,
) -> dict[str, Any]:
    """Validate locked collections and stage one immutable review."""

    if output.is_symlink() or output.exists():
        raise FileExistsError(f"gate-review output already exists: {output}")
    if output == run_path or run_path in output.parents:
        raise ValueError(
            "gate-review output cannot be inside the completed source run"
        )
    valid, errors = verify_run(run_path)
    if not valid:
        raise ValueError(
            "source run verification failed: " + "; ".join(errors)
        )
    source_run = _source_run_binding(run_path)
    config = _read_json_object(run_path / "config.resolved.json")
    experiment = config.get("experiment")
    artifacts = config.get("artifacts")
    if (
        not isinstance(experiment, Mapping)
        or experiment.get("kind") != "closed_loop"
    ):
        raise ValueError("Gate 4 native import requires Experiment B")
    if (
        not isinstance(artifacts, Mapping)
        or artifacts.get("retain_events") is not True
    ):
        raise ValueError(
            "Gate 4 native import requires retained Experiment B events"
        )

    request_snapshot = _snapshot_file(
        Path(requests_path),
        name="decoder requests",
    )
    judgment_snapshot = _snapshot_file(
        Path(judgments_path),
        name="decoder judgments",
    )
    truth_snapshot = _snapshot_file(
        Path(truth_labels_path),
        name="decoder truth labels",
    )
    review_snapshot = _snapshot_file(
        Path(source_review_path),
        name="decoder source review",
    )

    requests = _decoder_requests_from_snapshot(request_snapshot)
    labels = _decoder_truth_from_snapshot(truth_snapshot)
    source_review = DecoderSourceReview.parse(
        _json_object_from_snapshot(review_snapshot)
    )
    external_collection_inputs: dict[str, dict[str, Any]]
    external_collection_summary: dict[str, Any]
    external_collection_snapshots: dict[str, _FileSnapshot] = {}
    external_collection_validation: tuple[
        tuple[ExternalDecoderJudgment, ...],
        dict[str, dict[str, Any]],
        dict[str, Any],
    ] | None = None
    if external_collection_dir is not None:
        if external_collection_input is None:
            raise ValueError("external decoder collection path is missing")
        if external_collection_is_openrouter:
            external_collection_validation = (
                validate_openrouter_decoder_collection(
                    external_collection_dir,
                    requests=requests,
                    judgments_path=judgment_snapshot.path,
                )
            )
        else:
            external_collection_snapshots = _snapshot_collection_files(
                external_collection_dir,
                file_names=_EXTERNAL_COLLECTION_FILES,
                lock_names=_EXTERNAL_COLLECTION_LOCKS,
                label="external decoder collection",
            )
            external_collection_validation = (
                _validate_external_decoder_collection(
                    external_collection_dir,
                    run_path=run_path,
                    requests=requests,
                    supplied_judgments=judgment_snapshot,
                )
            )
        (
            judgments,
            external_collection_inputs,
            external_collection_summary,
        ) = external_collection_validation
        if not external_collection_is_openrouter:
            _assert_collection_unchanged(
                external_collection_input,
                external_collection_dir,
                external_collection_snapshots,
                file_names=_EXTERNAL_COLLECTION_FILES,
                lock_names=_EXTERNAL_COLLECTION_LOCKS,
                label="external decoder collection",
            )
            _assert_collection_manifest_bindings(
                external_collection_inputs,
                external_collection_snapshots,
                label="external decoder collection",
            )
    else:
        if not allow_reviewed_generic_decoders:
            raise ValueError(
                "reviewed generic decoder mode was not explicitly authorized"
            )
        judgments = _decoder_judgments_from_snapshot(judgment_snapshot)
        external_collection_inputs = {}
        external_collection_summary = {
            "provenance_mode": "reviewed_generic_import",
            "collection_status": "not_applicable",
            "official_provider_collection_validated": False,
            "responsible_researcher_review_required": True,
            "standalone_automated_provider_provenance_claimed": False,
        }
    native_collection_snapshots = _snapshot_collection_files(
        native_collection_dir,
        file_names=_NATIVE_COLLECTION_FILES,
        lock_names=(".collection.lock",),
        label="native action collection",
    )
    actions, native_collection_inputs, native_collection_summary = (
        _validate_native_action_collection(
            native_collection_dir,
            run_path,
        )
    )
    _assert_collection_unchanged(
        native_collection_input,
        native_collection_dir,
        native_collection_snapshots,
        file_names=_NATIVE_COLLECTION_FILES,
        lock_names=(".collection.lock",),
        label="native action collection",
    )
    _assert_collection_manifest_bindings(
        native_collection_inputs,
        native_collection_snapshots,
        label="native action collection",
    )

    retained_request_path = (
        run_path / "decoder" / "external-requests.jsonl"
    ).resolve()
    retained_truth_path = (
        run_path / "decoder" / "truth-labels.researcher-only.jsonl"
    ).resolve()
    retained_request_snapshot = (
        request_snapshot
        if request_snapshot.path == retained_request_path
        else _snapshot_file(
            retained_request_path,
            name="retained decoder requests",
        )
    )
    retained_truth_snapshot = (
        truth_snapshot
        if truth_snapshot.path == retained_truth_path
        else _snapshot_file(
            retained_truth_path,
            name="retained decoder truth labels",
        )
    )
    retained_requests = _decoder_requests_from_snapshot(
        retained_request_snapshot
    )
    retained_labels = _decoder_truth_from_snapshot(retained_truth_snapshot)
    if [item.to_dict() for item in requests] != [
        item.to_dict() for item in retained_requests
    ]:
        raise ValueError(
            "supplied decoder requests do not exactly match the source run"
        )
    if [item.to_dict() for item in labels] != [
        item.to_dict() for item in retained_labels
    ]:
        raise ValueError(
            "supplied decoder truth labels do not exactly match the source run"
        )
    request_splits = {request.evaluation_split for request in requests}
    if request_splits != {"development", "test"}:
        raise ValueError(
            "decoder material must contain exactly development and test splits"
        )
    if any(
        request.representation_id != "blinded-native-content-v1"
        or request.rubric_version != "native-profile-decoder-v1"
        for request in requests
    ):
        raise ValueError(
            "decoder requests do not use the admitted blinded native contract"
        )
    label_by_state = {
        label.pseudonymous_state_id: label for label in labels
    }
    if set(label_by_state) != {
        request.pseudonymous_state_id for request in requests
    }:
        raise ValueError(
            "decoder truth labels must cover request states exactly"
        )
    for request in requests:
        if (
            label_by_state[request.pseudonymous_state_id].evaluation_split
            != request.evaluation_split
        ):
            raise ValueError("decoder request/truth split mismatch")

    trajectory_rows = _read_jsonl_objects(
        run_path / "events" / "experiment-b-trajectories.jsonl"
    )
    trajectories = {
        _require_text(row.get("trajectory_id"), "trajectory_id"): row
        for row in trajectory_rows
    }
    if len(trajectories) != len(trajectory_rows):
        raise ValueError("duplicate Experiment B trajectory IDs")
    eligible = _eligible_trajectories(trajectory_rows)
    codebook = _read_codebook(
        run_path / "decoder" / "researcher-codebook.jsonl",
        requests,
        trajectories,
    )
    codebook_by_trajectory = {
        row["trajectory_id"]: row
        for row in codebook
        if row["evaluation_split"] == "test"
        and row["trajectory_id"] in eligible
    }
    if set(codebook_by_trajectory) != set(eligible):
        raise ValueError(
            "test decoder codebook does not cover eligible trajectories"
        )

    decoder_audit = validate_external_decoder_import(
        requests,
        judgments,
        minimum_sources_per_request=2,
        require_distinct_families=True,
    )
    if (
        not decoder_audit.complete_coverage
        or not decoder_audit.source_design_eligible
    ):
        raise ValueError(
            "external decoder material lacks complete eligible source coverage"
        )
    if any(
        not row.blind_to_system_identity or not row.blind_to_latent_truth
        for row in judgments
    ):
        raise ValueError("decoder judgments lack required blinding")
    decoder_analysis: DecoderAnalysis = analyze_external_decoders(
        requests,
        judgments,
        labels,
        evaluation_splits=("test",),
    )
    source_review_summary = _validate_source_review(
        source_review,
        judgments,
        requests_sha256=request_snapshot.sha256,
        judgments_sha256=judgment_snapshot.sha256,
    )
    judgments_by_request: dict[str, list[ExternalDecoderJudgment]] = {}
    for judgment in judgments:
        judgments_by_request.setdefault(
            judgment.request_id, []
        ).append(judgment)
    externally_decoded_ids = {
        trajectory_id
        for trajectory_id, row in codebook_by_trajectory.items()
        if len(
            {
                judgment.decoder_instance_id
                for judgment in judgments_by_request[row["request_id"]]
            }
        )
        >= 2
    }
    if externally_decoded_ids != set(eligible):
        raise ValueError(
            "external decoder judgments do not cover eligible trajectories"
        )

    suite_rows = _read_jsonl_objects(
        run_path / "events" / "experiment-b-held-out-terminal-suites.jsonl"
    )
    suites = tuple(_parse_terminal_suite(row) for row in suite_rows)
    if len({suite.domain_id for suite in suites}) != len(suites):
        raise ValueError("duplicate held-out terminal suite domains")
    suites_by_domain = {suite.domain_id: suite for suite in suites}
    action_scores = _validate_actions(
        actions,
        eligible,
        suites_by_domain,
    )

    assessments = _read_jsonl_objects(
        run_path / "metrics" / "experiment-b-self-confirmation.jsonl"
    )
    matched_failure_cases = _matched_failure_cases(
        eligible,
        assessments,
    )
    gate_4 = _gate_4_from_persisted_evidence(
        eligible,
        matched_failure_cases=matched_failure_cases,
        externally_decoded_trajectory_ids=externally_decoded_ids,
        decoder_audit=decoder_audit.to_dict(),
        source_review=source_review_summary,
        action_scores=action_scores,
    )
    baseline = _baseline_gate_4(
        run_path / "metrics" / "gate-report.json"
    )
    _assert_baseline_consistency(baseline, gate_4)

    inputs = {
        "decoder_requests": request_snapshot.manifest_entry(
            record_count=len(requests),
        ),
        **(
            external_collection_inputs
            if external_collection_inputs
            else {
                "decoder_judgments": judgment_snapshot.manifest_entry(
                    record_count=len(judgments),
                )
            }
        ),
        "decoder_truth_labels": truth_snapshot.manifest_entry(
            record_count=len(labels),
        ),
        **native_collection_inputs,
        "decoder_source_review": review_snapshot.manifest_entry(),
    }
    validation_summary = {
        "status": "import_validated",
        "source_run_verified": True,
        "source_run_experiment": "B",
        "source_run_mutated": False,
        "requests_match_retained_packet": True,
        "truth_labels_match_retained_packet": True,
        "evaluation_splits": ["development", "test"],
        "eligible_trajectory_ids": sorted(eligible),
        "external_decoder_evidence": {
            "import_status": "import_validated",
            "collection_provenance": external_collection_summary,
            "complete_coverage": True,
            "source_design_eligible": True,
            "blind_to_system_identity": True,
            "blind_to_latent_truth": True,
            "independent_source_reviewed": True,
            "eligible_trajectory_ids": sorted(externally_decoded_ids),
            "import_audit": decoder_audit.to_dict(),
            "source_review": source_review_summary,
            "analysis": decoder_analysis.to_dict(),
        },
        "native_terminal_action_evidence": {
            "import_status": "import_validated",
            "complete_coverage": True,
            "all_suite_bindings_validated": True,
            "reference_or_projection_actions_accepted": False,
            "collection_provenance": native_collection_summary,
            "scores": list(action_scores),
        },
        "source_gate4_baseline_consistent": True,
    }
    review_core = {
        "schema_version": 1,
        "artifact_kind": "gate4-native-evidence-review",
        "claim_status": "not_claimed",
        "source_run": source_run,
        "inputs": inputs,
        "validation_summary": validation_summary,
        "gate_4": gate_4.to_dict(),
        "interpretation_boundary": (
            "Passing import and computational checks does not establish a "
            "paper claim. Source distinctness is a responsible-researcher "
            "determination. The validated native provider collection and "
            + (
                "selected decoder provider collection remain auditable "
                "external evidence."
                if external_collection_dir is not None
                else (
                    "explicitly reviewed generic decoder import remain "
                    "auditable external evidence without an automated "
                    "provider-provenance assertion."
                )
            )
        ),
    }
    artifact_id = _digest_value(review_core)
    review_payload = {
        **review_core,
        "artifact_id": artifact_id,
    }

    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.",
            suffix=".staging",
            dir=output.parent,
        )
    )
    published = False
    try:
        review_output = stage / "gate-review.json"
        _write_json_durable(review_output, review_payload)
        artifact_manifest = {
            "schema_version": 1,
            "artifact_kind": "gate4-native-evidence-review",
            "artifact_id": artifact_id,
            "status": "complete",
            "claim_status": "not_claimed",
            "gate_review_sha256": _file_digest(review_output),
            "source_run": source_run,
        }
        manifest_output = stage / "manifest.json"
        _write_json_durable(manifest_output, artifact_manifest)
        checksum_lines = [
            f"{_file_digest(review_output)}  gate-review.json",
            f"{_file_digest(manifest_output)}  manifest.json",
        ]
        _write_text_durable(
            stage / "SHA256SUMS",
            "\n".join(checksum_lines) + "\n",
        )
        _fsync_directory(stage)

        _assert_source_run_unchanged(run_input, run_path, source_run)
        for supplied, snapshot, name in (
            (requests_path, request_snapshot, "decoder requests"),
            (judgments_path, judgment_snapshot, "decoder judgments"),
            (truth_labels_path, truth_snapshot, "decoder truth labels"),
            (
                source_review_path,
                review_snapshot,
                "decoder source review",
            ),
        ):
            _assert_snapshot_unchanged(
                supplied,
                snapshot,
                name=name,
            )
        _assert_collection_unchanged(
            native_collection_input,
            native_collection_dir,
            native_collection_snapshots,
            file_names=_NATIVE_COLLECTION_FILES,
            lock_names=(".collection.lock",),
            label="native action collection",
        )
        if external_collection_dir is not None:
            if external_collection_input is None:
                raise ValueError(
                    "external decoder collection path is missing"
                )
            if external_collection_is_openrouter:
                repeated_openrouter_validation = (
                    validate_openrouter_decoder_collection(
                        external_collection_dir,
                        requests=requests,
                        judgments_path=judgment_snapshot.path,
                    )
                )
                if (
                    repeated_openrouter_validation
                    != external_collection_validation
                ):
                    raise ValueError(
                        "OpenRouter decoder collection changed while the "
                        "Gate 4 review was being built"
                    )
            else:
                _assert_collection_unchanged(
                    external_collection_input,
                    external_collection_dir,
                    external_collection_snapshots,
                    file_names=_EXTERNAL_COLLECTION_FILES,
                    lock_names=_EXTERNAL_COLLECTION_LOCKS,
                    label="external decoder collection",
                )

        staged_valid, staged_errors = verify_gate_review(
            stage,
            source_run_dir=run_path,
        )
        if not staged_valid:
            raise ValueError(
                "staged Gate 4 review failed verification: "
                + "; ".join(staged_errors)
            )
        if output.is_symlink() or output.exists():
            raise FileExistsError(
                f"gate-review output already exists: {output}"
            )
        os.rename(stage, output)
        published = True
        _fsync_directory(output.parent)
    finally:
        if not published:
            if stage.is_symlink():
                stage.unlink()
            elif stage.exists():
                shutil.rmtree(stage)
    return {
        "artifact_id": artifact_id,
        "output_dir": str(output),
        "computed_status": gate_4.computed_status,
        "claim_status": "not_claimed",
        "eligible_trajectory_count": len(eligible),
        "decoder_judgment_count": len(judgments),
        "native_action_record_count": len(actions),
    }


def verify_gate_review(
    path: str | Path,
    *,
    source_run_dir: str | Path | None = None,
) -> tuple[bool, tuple[str, ...]]:
    """Verify review checksums, semantics, and optionally its source run."""

    supplied = Path(path)
    if supplied.is_symlink():
        return False, ("review directory cannot be a symlink",)
    root = supplied.resolve()
    if not root.is_dir():
        return False, ("review directory is missing",)
    errors: list[str] = []
    checksum_path = root / "SHA256SUMS"
    if checksum_path.is_symlink() or not checksum_path.is_file():
        return False, ("missing regular SHA256SUMS",)
    expected_files = {
        "SHA256SUMS",
        "gate-review.json",
        "manifest.json",
    }
    try:
        actual_entries = {item.name for item in root.iterdir()}
    except OSError as exc:
        return False, (f"cannot inspect review directory: {exc}",)
    if actual_entries != expected_files:
        errors.append("review directory has an unexpected file set")
    retained: set[str] = set()
    try:
        checksum_lines = checksum_path.read_text(
            encoding="utf-8"
        ).splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        return False, (f"invalid SHA256SUMS: {exc}",)
    for line_number, line in enumerate(checksum_lines, start=1):
        if not line.strip():
            continue
        try:
            expected, relative = line.split("  ", 1)
            _validate_digest(expected, "checksum")
        except ValueError as exc:
            errors.append(
                f"malformed checksum line {line_number}: {exc}"
            )
            continue
        relative_path = PurePosixPath(relative)
        if (
            not relative
            or relative_path.is_absolute()
            or relative_path.as_posix() != relative
            or "\\" in relative
            or "\x00" in relative
            or any(part in {"", ".", ".."} for part in relative_path.parts)
        ):
            errors.append(f"unsafe checksum path on line {line_number}")
            continue
        unresolved = root.joinpath(*relative_path.parts)
        if unresolved.is_symlink():
            errors.append(f"symlinked review artifact: {relative}")
            continue
        candidate = unresolved.resolve()
        if candidate == root or root not in candidate.parents:
            errors.append(f"checksum path escapes review: {relative}")
            continue
        if relative in retained:
            errors.append(f"duplicate checksum path on line {line_number}")
            continue
        retained.add(relative)
        if not candidate.is_file():
            errors.append(f"missing {relative}")
        elif _file_digest(candidate) != expected:
            errors.append(f"checksum mismatch: {relative}")
    actual = {
        item.relative_to(root).as_posix()
        for item in root.rglob("*")
        if item.is_file() and item != checksum_path
    }
    for unexpected in sorted(actual - retained):
        errors.append(f"unlisted artifact: {unexpected}")
    if retained != {"gate-review.json", "manifest.json"}:
        errors.append("review checksum manifest has unexpected file set")
    for item in root.rglob("*"):
        if item.is_symlink():
            errors.append(
                "unexpected review symlink: "
                + item.relative_to(root).as_posix()
            )

    manifest: Mapping[str, Any] | None = None
    review: Mapping[str, Any] | None = None
    try:
        manifest = _json_object_from_snapshot(
            _snapshot_file(
                root / "manifest.json",
                name="gate-review manifest",
            )
        )
    except (OSError, ValueError) as exc:
        errors.append(f"invalid manifest.json: {exc}")
    try:
        review = _json_object_from_snapshot(
            _snapshot_file(
                root / "gate-review.json",
                name="gate-review payload",
            )
        )
    except (OSError, ValueError) as exc:
        errors.append(f"invalid gate-review.json: {exc}")
    if manifest is not None:
        if (
            manifest.get("schema_version") != 1
            or manifest.get("artifact_kind")
            != "gate4-native-evidence-review"
            or manifest.get("status") != "complete"
            or manifest.get("claim_status") != "not_claimed"
        ):
            errors.append("invalid gate-review manifest semantics")
        review_path = root / "gate-review.json"
        if (
            review_path.is_file()
            and manifest.get("gate_review_sha256")
            != _file_digest(review_path)
        ):
            errors.append("manifest gate-review digest mismatch")
    if review is not None:
        artifact_id = review.get("artifact_id")
        core = dict(review)
        core.pop("artifact_id", None)
        if artifact_id != _digest_value(core):
            errors.append("gate-review artifact_id mismatch")
        if (
            review.get("schema_version") != 1
            or review.get("artifact_kind")
            != "gate4-native-evidence-review"
            or review.get("claim_status") != "not_claimed"
        ):
            errors.append("invalid gate-review claim semantics")
        gate = review.get("gate_4")
        if (
            not isinstance(gate, Mapping)
            or gate.get("gate_id") != "gate-4"
            or gate.get("claim_status") != "not_claimed"
        ):
            errors.append("Gate 4 claim_status must be not_claimed")
        if (
            manifest is not None
            and manifest.get("artifact_id") != artifact_id
        ):
            errors.append("manifest/review artifact_id mismatch")
        if not isinstance(review.get("source_run"), Mapping):
            errors.append("gate-review source binding is missing")
        if not isinstance(review.get("inputs"), Mapping):
            errors.append("gate-review input bindings are missing")
    if (
        manifest is not None
        and review is not None
        and manifest.get("source_run") != review.get("source_run")
    ):
        errors.append("manifest/review source binding mismatch")

    if source_run_dir is not None:
        supplied_source = Path(source_run_dir)
        if supplied_source.is_symlink() or not supplied_source.is_dir():
            errors.append("source run must be a safe directory")
        else:
            source = supplied_source.resolve()
            valid, source_errors = verify_run(source)
            if not valid:
                errors.append(
                    "source run verification failed: "
                    + "; ".join(source_errors)
                )
            else:
                try:
                    expected_source = _source_run_binding(source)
                    if (
                        review is None
                        or review.get("source_run") != expected_source
                    ):
                        errors.append(
                            "gate-review/source run binding mismatch"
                        )
                except (KeyError, OSError, ValueError) as exc:
                    errors.append(f"invalid source run binding: {exc}")
    return not errors, tuple(errors)
