"""Immutable Gate 4 review artifacts for externally collected native evidence.

The Experiment B runner deliberately finalizes a run before external decoder
judgments or recorded native-system actions exist.  This module preserves that
boundary: it verifies the completed run, validates separately supplied
evidence, and writes a new checksum-bound review directory.  It never writes
inside the source run.

The import is an admission and computation check only.  A successful review
therefore retains ``claim_status = "not_claimed"``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from itertools import combinations
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence
import json
import math

from .artifacts import canonical_json, verify_run
from .decoder_study import (
    DecoderAnalysis,
    DecoderTruthLabel,
    ExternalDecoderJudgment,
    ExternalDecoderRequest,
    analyze_external_decoders,
    read_decoder_truth_labels,
    read_external_decoder_judgments,
    read_external_decoder_requests,
    validate_external_decoder_import,
)
from .gates import GateCriterion, GateReport
from .heldout import (
    HeldOutTerminalItem,
    HeldOutTerminalOption,
    HeldOutTerminalSuite,
    TerminalAction,
    score_heldout_terminal_actions,
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
    return sha256(path.read_bytes()).hexdigest()


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


def import_native_gate_review(
    *,
    run_dir: str | Path,
    requests_path: str | Path,
    judgments_path: str | Path,
    truth_labels_path: str | Path,
    actions_path: str | Path,
    source_review_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Validate external native evidence and write a new immutable review."""

    run_path = Path(run_dir).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
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
    manifest_path = run_path / "manifest.json"
    manifest = _read_json_object(manifest_path)
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

    request_file = Path(requests_path).resolve()
    judgment_file = Path(judgments_path).resolve()
    truth_file = Path(truth_labels_path).resolve()
    action_file = Path(actions_path).resolve()
    review_file = Path(source_review_path).resolve()

    requests = read_external_decoder_requests(request_file)
    judgments = read_external_decoder_judgments(judgment_file)
    labels = read_decoder_truth_labels(truth_file)
    actions = read_native_terminal_action_records(action_file)
    source_review = read_decoder_source_review(review_file)

    retained_requests = read_external_decoder_requests(
        run_path / "decoder" / "external-requests.jsonl"
    )
    retained_labels = read_decoder_truth_labels(
        run_path / "decoder" / "truth-labels.researcher-only.jsonl"
    )
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
        requests_sha256=_file_digest(request_file),
        judgments_sha256=_file_digest(judgment_file),
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
        "decoder_requests": _input_manifest_entry(
            request_file,
            record_count=len(requests),
        ),
        "decoder_judgments": _input_manifest_entry(
            judgment_file,
            record_count=len(judgments),
        ),
        "decoder_truth_labels": _input_manifest_entry(
            truth_file,
            record_count=len(labels),
        ),
        "native_terminal_actions": _input_manifest_entry(
            action_file,
            record_count=len(actions),
        ),
        "decoder_source_review": _input_manifest_entry(review_file),
    }
    source_run = {
        "run_id": manifest["run_id"],
        "run_manifest_sha256": _file_digest(manifest_path),
        "run_checksum_manifest_sha256": _file_digest(
            run_path / "SHA256SUMS"
        ),
        "config_sha256": manifest["config_sha256"],
        "source_sha256": manifest["source_sha256"],
        "verified_complete": True,
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
            "determination, and recorded-action attestations remain auditable "
            "external evidence."
        ),
    }
    artifact_id = _digest_value(review_core)
    review_payload = {
        **review_core,
        "artifact_id": artifact_id,
    }

    output.mkdir(parents=True, exist_ok=False)
    review_output = output / "gate-review.json"
    review_output.write_text(
        canonical_json(review_payload) + "\n",
        encoding="utf-8",
    )
    artifact_manifest = {
        "schema_version": 1,
        "artifact_kind": "gate4-native-evidence-review",
        "artifact_id": artifact_id,
        "status": "complete",
        "claim_status": "not_claimed",
        "gate_review_sha256": _file_digest(review_output),
        "source_run": source_run,
    }
    manifest_output = output / "manifest.json"
    manifest_output.write_text(
        canonical_json(artifact_manifest) + "\n",
        encoding="utf-8",
    )
    checksum_lines = [
        f"{_file_digest(review_output)}  gate-review.json",
        f"{_file_digest(manifest_output)}  manifest.json",
    ]
    (output / "SHA256SUMS").write_text(
        "\n".join(checksum_lines) + "\n",
        encoding="utf-8",
    )
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
) -> tuple[bool, tuple[str, ...]]:
    """Verify an immutable Gate 4 review directory and its content bindings."""

    root = Path(path).resolve()
    errors: list[str] = []
    checksum_path = root / "SHA256SUMS"
    if not checksum_path.is_file():
        return False, ("missing SHA256SUMS",)
    retained: set[str] = set()
    for line_number, line in enumerate(
        checksum_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            expected, relative = line.split("  ", 1)
        except ValueError:
            errors.append(f"malformed checksum line {line_number}")
            continue
        try:
            _validate_digest(expected, "checksum")
        except ValueError as exc:
            errors.append(f"line {line_number}: {exc}")
            continue
        relative_path = PurePosixPath(relative)
        if (
            relative_path.is_absolute()
            or relative_path.as_posix() != relative
            or any(part in {"", ".", ".."} for part in relative_path.parts)
        ):
            errors.append(f"unsafe checksum path on line {line_number}")
            continue
        candidate = root.joinpath(*relative_path.parts).resolve()
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

    manifest: Mapping[str, Any] | None = None
    review: Mapping[str, Any] | None = None
    try:
        manifest = _read_json_object(root / "manifest.json")
    except (OSError, ValueError) as exc:
        errors.append(f"invalid manifest.json: {exc}")
    try:
        review = _read_json_object(root / "gate-review.json")
    except (OSError, ValueError) as exc:
        errors.append(f"invalid gate-review.json: {exc}")
    if manifest is not None:
        if (
            manifest.get("artifact_kind")
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
            review.get("artifact_kind")
            != "gate4-native-evidence-review"
            or review.get("claim_status") != "not_claimed"
        ):
            errors.append("invalid gate-review claim semantics")
        gate = review.get("gate_4")
        if not isinstance(gate, Mapping) or gate.get(
            "claim_status"
        ) != "not_claimed":
            errors.append("Gate 4 claim_status must be not_claimed")
        if (
            manifest is not None
            and manifest.get("artifact_id") != artifact_id
        ):
            errors.append("manifest/review artifact_id mismatch")
    return not errors, tuple(errors)
