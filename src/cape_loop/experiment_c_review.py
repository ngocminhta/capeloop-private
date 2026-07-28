"""Immutable Experiment C rescoring with external native-state decoders.

Experiment C is completed before external judgments exist.  The source run
therefore exports a blinded, content-addressed packet for every native terminal
state.  This module later verifies that completed run, admits exactly two
metadata-distinct decoder families with complete coverage, fits one
development-only calibrator per family, rescores the common terminal battery,
and writes a new checksum-bound review artifact.  The source run is never
modified.

Distinct family/source metadata is an eligibility check.  It is not evidence
that the two judgments are statistically independent.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence
import json
import math
import os
import shutil
import tempfile

from .artifacts import (
    RunArtifacts,
    canonical_json,
    file_sha256,
    verify_run,
)
from .beliefs import MarginalPreferenceBelief, PreferenceBelief
from .config import AppConfig
from .decoder_study import (
    DecoderTruthLabel,
    ExternalDecoderJudgment,
    ExternalDecoderRequest,
    analyze_external_decoders,
    build_blinded_native_decoder_request,
    fit_decoder_calibration,
    validate_external_decoder_import,
)
from .domains import domain_for_split, get_domain
from .experiments.evaluation import (
    EvaluationRow,
    ExperimentCResult,
    TerminalBattery,
    TerminalBatteryScore,
    TerminalReliabilityBin,
    analyze_rankings,
    build_terminal_battery,
    evaluate_terminal_battery,
    mean_terminal_battery_scores,
)
from .gates import GateCriterion, GateReport
from .gate_review import (
    EXTERNAL_COLLECTION_PROVENANCE_MODES,
    OPENROUTER_COLLECTION_PROVENANCE,
    validate_selected_external_decoder_collection,
)
from .openrouter_decoder_collection import (
    is_openrouter_decoder_collection,
)
from .native import NativeMemoryState
from .schemas import LatentUser


NATIVE_UPDATER_IDS = frozenset(
    {
        "episodic_memory",
        "semantic_memory",
        "provenance_linked_memory",
    }
)

PACKET_REQUESTS = "decoder/experiment-c-external-requests.jsonl"
PACKET_TRUTH = (
    "decoder/experiment-c-truth-labels.researcher-only.jsonl"
)
PACKET_CODEBOOK = "decoder/experiment-c-researcher-codebook.jsonl"
PACKET_MANIFEST = "decoder/experiment-c-external-design-manifest.json"

SOURCE_METRICS = "metrics/experiment-c.jsonl"
SOURCE_BATTERIES = "events/terminal-batteries.jsonl"
SOURCE_REPLAYS = "events/experiment-c-replays.jsonl"
SOURCE_ENDOGENOUS = "events/experiment-c-endogenous.jsonl"

REVIEW_KIND = "experiment-c-external-decoder-rescore"
RESCORE_BASIS = (
    "mean_of_exactly_two_calibrated_external_decoder_families"
)
INDEPENDENCE_CAVEAT = (
    "Exactly two distinct family, instance, and source-descriptor metadata "
    "values are required per request. This is a design-eligibility check, "
    "not proof that decoder judgments are statistically independent."
)

REVIEW_FILES = frozenset(
    {
        "inputs/external-requests.jsonl",
        "inputs/truth-labels.researcher-only.jsonl",
        "inputs/researcher-codebook.jsonl",
        "inputs/judgments.jsonl",
        "metrics/external-decoder-scores.jsonl",
        "metrics/experiment-c-rescored.jsonl",
        "metrics/calibration.json",
        "metrics/decoder-analysis.json",
        "metrics/experiment-c-rankings.json",
        "metrics/gate-5.json",
        "review.json",
        "manifest.json",
    }
)


def _digest(value: Any) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _file_digest(path: Path) -> str:
    return file_sha256(path)


def _validate_digest(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _strict_fields(
    raw: Mapping[str, Any],
    expected: set[str],
    *,
    name: str,
) -> None:
    if set(raw) != expected:
        raise ValueError(
            f"{name} has missing or unknown fields: "
            + canonical_json(
                {
                    "missing": sorted(expected - set(raw)),
                    "unknown": sorted(set(raw) - expected),
                }
            )
        )


def _safe_json_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{path}: expected a safe regular JSON file")
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(decoded, Mapping):
        raise ValueError(f"{path}: expected a JSON object")
    return dict(decoded)


def _jsonl_from_bytes(
    material: bytes,
    *,
    source: Path,
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    try:
        lines = material.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError(f"{source}: invalid UTF-8 input: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            decoded = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{source}:{line_number}: {exc}") from exc
        if not isinstance(decoded, Mapping):
            raise ValueError(
                f"{source}:{line_number}: record must be an object"
            )
        rows.append(dict(decoded))
    if not rows:
        raise ValueError(f"{source}: JSONL input cannot be empty")
    return tuple(rows)


def _safe_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{path}: expected a safe regular JSONL file")
    try:
        material = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"{path}: cannot read JSONL input: {exc}") from exc
    return _jsonl_from_bytes(material, source=path)


def _jsonl_material(rows: Iterable[Mapping[str, Any]]) -> str:
    return "".join(canonical_json(dict(row)) + "\n" for row in rows)


def _row_key(raw: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "split": raw["split"],
        "regime": raw["regime"],
        "replicate": raw["replicate"],
        "user_id": raw["user_id"],
        "domain_id": raw["domain_id"],
        "updater_id": raw["updater_id"],
    }


@dataclass(frozen=True, slots=True)
class ExperimentCDecoderCodebookRow:
    """Researcher-only binding from one blinded state to one exact C row."""

    request_id: str
    pseudonymous_state_id: str
    evaluation_split: str
    regime: str
    replicate: int
    user_id: str
    domain_id: str
    updater_id: str
    stable_row_key_sha256: str
    source_metric_row_sha256: str
    battery_id: str
    battery_digest: str
    terminal_state_id: str
    terminal_state_sha256: str
    source_state_file: str
    source_state_record_id: str
    source_state_record_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "request_id",
            "pseudonymous_state_id",
            "regime",
            "user_id",
            "domain_id",
            "updater_id",
            "battery_id",
            "terminal_state_id",
            "source_state_file",
            "source_state_record_id",
        ):
            _require_text(getattr(self, name), name)
        if self.evaluation_split not in {"development", "test"}:
            raise ValueError(
                "Experiment C decoder split must be development or test"
            )
        if self.regime not in {
            "fixed_balanced",
            "fixed_biased",
            "endogenous_closed_loop",
        }:
            raise ValueError("unknown Experiment C decoder regime")
        if (
            isinstance(self.replicate, bool)
            or not isinstance(self.replicate, int)
            or self.replicate < 0
        ):
            raise ValueError("replicate must be a nonnegative integer")
        if self.updater_id not in NATIVE_UPDATER_IDS:
            raise ValueError("codebook row is not a native updater")
        if self.source_state_file not in {
            SOURCE_REPLAYS,
            SOURCE_ENDOGENOUS,
        }:
            raise ValueError("codebook source_state_file is not admitted")
        if (
            self.regime == "endogenous_closed_loop"
        ) != (self.source_state_file == SOURCE_ENDOGENOUS):
            raise ValueError("codebook regime/source-state file mismatch")
        for name in (
            "stable_row_key_sha256",
            "source_metric_row_sha256",
            "battery_digest",
            "terminal_state_id",
            "terminal_state_sha256",
            "source_state_record_sha256",
        ):
            _validate_digest(getattr(self, name), name)
        if self.stable_row_key_sha256 != _digest(self.stable_key()):
            raise ValueError("stable row key digest mismatch")

    def stable_key(self) -> dict[str, Any]:
        return {
            "split": self.evaluation_split,
            "regime": self.regime,
            "replicate": self.replicate,
            "user_id": self.user_id,
            "domain_id": self.domain_id,
            "updater_id": self.updater_id,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "request_id": self.request_id,
            "pseudonymous_state_id": self.pseudonymous_state_id,
            "evaluation_split": self.evaluation_split,
            "regime": self.regime,
            "replicate": self.replicate,
            "user_id": self.user_id,
            "domain_id": self.domain_id,
            "updater_id": self.updater_id,
            "stable_row_key_sha256": self.stable_row_key_sha256,
            "source_metric_row_sha256": self.source_metric_row_sha256,
            "battery_id": self.battery_id,
            "battery_digest": self.battery_digest,
            "terminal_state_id": self.terminal_state_id,
            "terminal_state_sha256": self.terminal_state_sha256,
            "source_state_file": self.source_state_file,
            "source_state_record_id": self.source_state_record_id,
            "source_state_record_sha256": (
                self.source_state_record_sha256
            ),
        }

    @classmethod
    def parse(
        cls,
        raw: Mapping[str, Any],
    ) -> "ExperimentCDecoderCodebookRow":
        fields = {
            "schema_version",
            "request_id",
            "pseudonymous_state_id",
            "evaluation_split",
            "regime",
            "replicate",
            "user_id",
            "domain_id",
            "updater_id",
            "stable_row_key_sha256",
            "source_metric_row_sha256",
            "battery_id",
            "battery_digest",
            "terminal_state_id",
            "terminal_state_sha256",
            "source_state_file",
            "source_state_record_id",
            "source_state_record_sha256",
        }
        _strict_fields(raw, fields, name="Experiment C decoder codebook row")
        if raw["schema_version"] != 1:
            raise ValueError("codebook schema_version must be 1")
        return cls(
            request_id=raw["request_id"],
            pseudonymous_state_id=raw["pseudonymous_state_id"],
            evaluation_split=raw["evaluation_split"],
            regime=raw["regime"],
            replicate=raw["replicate"],
            user_id=raw["user_id"],
            domain_id=raw["domain_id"],
            updater_id=raw["updater_id"],
            stable_row_key_sha256=raw["stable_row_key_sha256"],
            source_metric_row_sha256=raw["source_metric_row_sha256"],
            battery_id=raw["battery_id"],
            battery_digest=raw["battery_digest"],
            terminal_state_id=raw["terminal_state_id"],
            terminal_state_sha256=raw["terminal_state_sha256"],
            source_state_file=raw["source_state_file"],
            source_state_record_id=raw["source_state_record_id"],
            source_state_record_sha256=raw[
                "source_state_record_sha256"
            ],
        )


def _source_run_identity(run: RunArtifacts) -> dict[str, Any]:
    manifest = _safe_json_object(run.path / "manifest.json")
    return {
        "run_id": manifest["run_id"],
        "config_sha256": manifest["config_sha256"],
        "source_sha256": manifest["source_sha256"],
    }


def write_experiment_c_decoder_packet(
    run: RunArtifacts,
    result: ExperimentCResult,
    *,
    development_users: Sequence[LatentUser],
    test_users: Sequence[LatentUser],
    events_retained: bool,
) -> dict[str, Any]:
    """Write the blinded C decoder packet before the run is finalized."""

    users = {
        user.user_id: (split, user)
        for split, population in (
            ("development", development_users),
            ("test", test_users),
        )
        for user in population
    }
    if len(users) != len(development_users) + len(test_users):
        raise ValueError("Experiment C decoder users are not split-disjoint")

    replay_by_key = {
        (replay.history_digest, replay.updater_id): replay
        for replay in result.replay_results
    }
    if len(replay_by_key) != len(result.replay_results):
        raise ValueError("Experiment C replay bindings are not unique")
    endogenous_by_id = {
        trajectory.trajectory_id: trajectory
        for trajectory in result.endogenous_trajectories
    }
    if len(endogenous_by_id) != len(result.endogenous_trajectories):
        raise ValueError("Experiment C endogenous IDs are not unique")

    requests: list[ExternalDecoderRequest] = []
    labels: list[DecoderTruthLabel] = []
    codebook: list[ExperimentCDecoderCodebookRow] = []
    source_rows = [
        {"schema_version": 1, **row.to_dict()} for row in result.rows
    ]
    for evaluation, metric_row in zip(result.rows, source_rows):
        if evaluation.updater_id not in NATIVE_UPDATER_IDS:
            continue
        try:
            split, user = users[evaluation.user_id]
        except KeyError as exc:
            raise ValueError(
                "native Experiment C row references an unknown user"
            ) from exc
        if split != evaluation.split:
            raise ValueError("native Experiment C row/user split mismatch")

        if evaluation.regime in {"fixed_balanced", "fixed_biased"}:
            try:
                replay = replay_by_key[
                    (evaluation.history_digest, evaluation.updater_id)
                ]
            except KeyError as exc:
                raise ValueError(
                    "native fixed-history row lacks an exact replay"
                ) from exc
            state = replay.terminal_state.opaque_state
            state_record = {
                "schema_version": 1,
                **replay.to_dict(),
            }
            state_file = SOURCE_REPLAYS
            state_record_id = replay.audit_record.trajectory_id
        else:
            trajectory_id = (
                f"experiment-c:{evaluation.split}:{evaluation.domain_id}:"
                f"{evaluation.user_id}:replicate-{evaluation.replicate}:"
                f"closed:{evaluation.updater_id}"
            )
            try:
                trajectory = endogenous_by_id[trajectory_id]
            except KeyError as exc:
                raise ValueError(
                    "native endogenous row lacks an exact trajectory"
                ) from exc
            state = trajectory.terminal_opaque_state
            state_record = {
                "schema_version": 1,
                **trajectory.to_dict(include_truth=True),
            }
            state_file = SOURCE_ENDOGENOUS
            state_record_id = trajectory.trajectory_id
        if not isinstance(state, NativeMemoryState):
            raise ValueError(
                "configured native Experiment C row lacks NativeMemoryState"
            )

        metric_sha256 = _digest(metric_row)
        stable_key = _row_key(metric_row)
        nonce = _digest(
            {
                "assignment_protocol": (
                    "experiment-c-external-decoder-v1"
                ),
                "stable_row_key_sha256": _digest(stable_key),
                "source_metric_row_sha256": metric_sha256,
            }
        )
        request = build_blinded_native_decoder_request(
            state,
            evaluation_split=evaluation.split,
            assignment_nonce=nonce,
        )
        requests.append(request)
        labels.append(
            DecoderTruthLabel(
                pseudonymous_state_id=request.pseudonymous_state_id,
                theta=user.theta,
                evaluation_split=evaluation.split,
            )
        )
        codebook.append(
            ExperimentCDecoderCodebookRow(
                request_id=request.request_id,
                pseudonymous_state_id=request.pseudonymous_state_id,
                evaluation_split=evaluation.split,
                regime=evaluation.regime,
                replicate=evaluation.replicate,
                user_id=evaluation.user_id,
                domain_id=evaluation.domain_id,
                updater_id=evaluation.updater_id,
                stable_row_key_sha256=_digest(stable_key),
                source_metric_row_sha256=metric_sha256,
                battery_id=evaluation.battery_id,
                battery_digest=evaluation.battery_digest,
                terminal_state_id=state.state_id,
                terminal_state_sha256=_digest(state.to_dict()),
                source_state_file=state_file,
                source_state_record_id=state_record_id,
                source_state_record_sha256=_digest(state_record),
            )
        )

    if len({item.request_id for item in requests}) != len(requests):
        raise ValueError("Experiment C decoder request IDs are not unique")
    if len({item.pseudonymous_state_id for item in labels}) != len(labels):
        raise ValueError("Experiment C decoder state pseudonyms are not unique")
    expected_native_rows = sum(
        row.updater_id in NATIVE_UPDATER_IDS for row in result.rows
    )
    if len(codebook) != expected_native_rows:
        raise ValueError("Experiment C decoder packet has incomplete coverage")

    request_rows = [item.to_dict() for item in requests]
    label_rows = [item.to_dict() for item in labels]
    codebook_rows = [item.to_dict() for item in codebook]
    request_path = run.write_jsonl(PACKET_REQUESTS, request_rows)
    truth_path = run.write_jsonl(PACKET_TRUTH, label_rows)
    codebook_path = run.write_jsonl(PACKET_CODEBOOK, codebook_rows)
    run_identity = _source_run_identity(run)
    manifest = {
        "schema_version": 1,
        "artifact_kind": "experiment-c-external-decoder-design",
        "status": (
            "ready_for_external_judgments"
            if requests and events_retained
            else "ineligible_no_retained_native_states"
            if requests
            else "not_applicable_no_native_updaters"
        ),
        "claim_status": "not_claimed",
        "source_run_identity": run_identity,
        "source_run_identity_sha256": _digest(run_identity),
        "source_metric_row_set_sha256": _digest(source_rows),
        "terminal_battery_set_sha256": _digest(
            [
                battery.to_dict()
                for battery in sorted(
                    result.terminal_batteries,
                    key=lambda item: item.domain_id,
                )
            ]
        ),
        "events_retained": events_retained,
        "request_count": len(request_rows),
        "development_request_count": sum(
            item.evaluation_split == "development" for item in requests
        ),
        "test_request_count": sum(
            item.evaluation_split == "test" for item in requests
        ),
        "native_metric_row_count": expected_native_rows,
        "one_request_per_native_metric_row": (
            len(requests) == expected_native_rows
        ),
        "minimum_external_sources_per_request": 2,
        "exact_external_sources_used_for_rescore": 2,
        "distinct_decoder_families_required": True,
        "development_only_per_family_calibration_required": True,
        "requests_blind_to_system_identity_and_truth": True,
        "truth_file_must_not_be_shared_with_decoders": PACKET_TRUTH,
        "metadata_eligibility_is_statistical_independence": False,
        "independence_caveat": INDEPENDENCE_CAVEAT,
        "files": {
            PACKET_REQUESTS: {
                "sha256": _file_digest(request_path),
                "record_count": len(request_rows),
            },
            PACKET_TRUTH: {
                "sha256": _file_digest(truth_path),
                "record_count": len(label_rows),
            },
            PACKET_CODEBOOK: {
                "sha256": _file_digest(codebook_path),
                "record_count": len(codebook_rows),
            },
        },
    }
    run.write_json(PACKET_MANIFEST, manifest)
    return manifest


def _decoder_requests(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[ExternalDecoderRequest, ...]:
    requests = tuple(ExternalDecoderRequest.parse(row) for row in rows)
    if len({item.request_id for item in requests}) != len(requests):
        raise ValueError("duplicate Experiment C decoder request IDs")
    return requests


def _decoder_labels(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[DecoderTruthLabel, ...]:
    labels = tuple(DecoderTruthLabel.parse(row) for row in rows)
    if len({item.pseudonymous_state_id for item in labels}) != len(labels):
        raise ValueError("duplicate Experiment C decoder truth labels")
    return labels


def _decoder_codebook(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[ExperimentCDecoderCodebookRow, ...]:
    codebook = tuple(
        ExperimentCDecoderCodebookRow.parse(row) for row in rows
    )
    for field in (
        "request_id",
        "pseudonymous_state_id",
        "stable_row_key_sha256",
        "source_metric_row_sha256",
    ):
        values = [getattr(item, field) for item in codebook]
        if len(set(values)) != len(values):
            raise ValueError(f"duplicate Experiment C codebook {field}")
    return codebook


def _validate_packet_manifest(
    run_path: Path,
    requests: Sequence[ExternalDecoderRequest],
    labels: Sequence[DecoderTruthLabel],
    codebook: Sequence[ExperimentCDecoderCodebookRow],
) -> dict[str, Any]:
    manifest = _safe_json_object(run_path / PACKET_MANIFEST)
    expected_fields = {
        "schema_version",
        "artifact_kind",
        "status",
        "claim_status",
        "source_run_identity",
        "source_run_identity_sha256",
        "source_metric_row_set_sha256",
        "terminal_battery_set_sha256",
        "events_retained",
        "request_count",
        "development_request_count",
        "test_request_count",
        "native_metric_row_count",
        "one_request_per_native_metric_row",
        "minimum_external_sources_per_request",
        "exact_external_sources_used_for_rescore",
        "distinct_decoder_families_required",
        "development_only_per_family_calibration_required",
        "requests_blind_to_system_identity_and_truth",
        "truth_file_must_not_be_shared_with_decoders",
        "metadata_eligibility_is_statistical_independence",
        "independence_caveat",
        "files",
    }
    _strict_fields(
        manifest,
        expected_fields,
        name="Experiment C decoder design manifest",
    )
    required_semantics = {
        "schema_version": 1,
        "artifact_kind": "experiment-c-external-decoder-design",
        "status": "ready_for_external_judgments",
        "claim_status": "not_claimed",
        "events_retained": True,
        "one_request_per_native_metric_row": True,
        "minimum_external_sources_per_request": 2,
        "exact_external_sources_used_for_rescore": 2,
        "distinct_decoder_families_required": True,
        "development_only_per_family_calibration_required": True,
        "requests_blind_to_system_identity_and_truth": True,
        "truth_file_must_not_be_shared_with_decoders": PACKET_TRUTH,
        "metadata_eligibility_is_statistical_independence": False,
        "independence_caveat": INDEPENDENCE_CAVEAT,
    }
    for name, expected in required_semantics.items():
        if manifest.get(name) != expected:
            raise ValueError(
                f"Experiment C decoder manifest mismatch for {name}"
            )
    source_identity = manifest["source_run_identity"]
    if not isinstance(source_identity, Mapping):
        raise ValueError("source_run_identity must be an object")
    if manifest["source_run_identity_sha256"] != _digest(source_identity):
        raise ValueError("source run identity digest mismatch")
    source_manifest = _safe_json_object(run_path / "manifest.json")
    expected_identity = {
        "run_id": source_manifest["run_id"],
        "config_sha256": source_manifest["config_sha256"],
        "source_sha256": source_manifest["source_sha256"],
    }
    if dict(source_identity) != expected_identity:
        raise ValueError("decoder packet is bound to another source run")
    expected_counts = {
        "request_count": len(requests),
        "development_request_count": sum(
            row.evaluation_split == "development" for row in requests
        ),
        "test_request_count": sum(
            row.evaluation_split == "test" for row in requests
        ),
        "native_metric_row_count": len(codebook),
    }
    for name, expected in expected_counts.items():
        if manifest[name] != expected:
            raise ValueError(f"decoder packet count mismatch for {name}")
    if len(requests) != len(labels) or len(requests) != len(codebook):
        raise ValueError("decoder packet files have different coverage")
    files = manifest["files"]
    if not isinstance(files, Mapping) or set(files) != {
        PACKET_REQUESTS,
        PACKET_TRUTH,
        PACKET_CODEBOOK,
    }:
        raise ValueError("decoder packet manifest file set is invalid")
    for relative, expected_count in (
        (PACKET_REQUESTS, len(requests)),
        (PACKET_TRUTH, len(labels)),
        (PACKET_CODEBOOK, len(codebook)),
    ):
        entry = files[relative]
        if (
            not isinstance(entry, Mapping)
            or set(entry) != {"sha256", "record_count"}
            or entry["sha256"] != _file_digest(run_path / relative)
            or entry["record_count"] != expected_count
        ):
            raise ValueError(f"decoder packet manifest mismatch for {relative}")
    return manifest


def _validate_source_rows(
    run_path: Path,
    packet_manifest: Mapping[str, Any],
    requests: Sequence[ExternalDecoderRequest],
    labels: Sequence[DecoderTruthLabel],
    codebook: Sequence[ExperimentCDecoderCodebookRow],
) -> tuple[tuple[dict[str, Any], ...], dict[str, TerminalBattery]]:
    metric_rows = _safe_jsonl(run_path / SOURCE_METRICS)
    if packet_manifest["source_metric_row_set_sha256"] != _digest(metric_rows):
        raise ValueError("source Experiment C metric row-set digest mismatch")
    metric_by_sha = {_digest(row): row for row in metric_rows}
    if len(metric_by_sha) != len(metric_rows):
        raise ValueError("source Experiment C metric rows are duplicated")
    native_rows = {
        digest: row
        for digest, row in metric_by_sha.items()
        if row.get("updater_id") in NATIVE_UPDATER_IDS
    }
    if set(native_rows) != {
        item.source_metric_row_sha256 for item in codebook
    }:
        raise ValueError(
            "decoder codebook does not cover native metric rows exactly"
        )

    request_by_id = {item.request_id: item for item in requests}
    label_by_state = {
        item.pseudonymous_state_id: item for item in labels
    }
    if set(request_by_id) != {item.request_id for item in codebook}:
        raise ValueError("decoder request/codebook coverage mismatch")
    if set(label_by_state) != {
        item.pseudonymous_state_id for item in codebook
    }:
        raise ValueError("decoder truth/codebook coverage mismatch")
    for item in codebook:
        row = native_rows[item.source_metric_row_sha256]
        if _row_key(row) != item.stable_key():
            raise ValueError("codebook stable row does not match metric row")
        if _digest(_row_key(row)) != item.stable_row_key_sha256:
            raise ValueError("metric stable row key digest mismatch")
        if (
            row.get("battery_id") != item.battery_id
            or row.get("battery_digest") != item.battery_digest
        ):
            raise ValueError("codebook battery binding mismatch")
        request = request_by_id[item.request_id]
        label = label_by_state[item.pseudonymous_state_id]
        if (
            request.pseudonymous_state_id != item.pseudonymous_state_id
            or request.evaluation_split != item.evaluation_split
            or label.evaluation_split != item.evaluation_split
        ):
            raise ValueError("request/truth/codebook split binding mismatch")

    replay_rows = _safe_jsonl(run_path / SOURCE_REPLAYS)
    endogenous_rows = _safe_jsonl(run_path / SOURCE_ENDOGENOUS)
    state_records: dict[tuple[str, str], dict[str, Any]] = {}
    for path, rows, id_field in (
        (SOURCE_REPLAYS, replay_rows, None),
        (SOURCE_ENDOGENOUS, endogenous_rows, "trajectory_id"),
    ):
        for row in rows:
            if id_field is None:
                audit = row.get("audit_record")
                if not isinstance(audit, Mapping):
                    raise ValueError("replay lacks audit_record")
                identifier = audit.get("trajectory_id")
            else:
                identifier = row.get(id_field)
            _require_text(identifier, f"{path} record ID")
            key = (path, str(identifier))
            if key in state_records:
                raise ValueError(f"duplicate source state record {key}")
            state_records[key] = row
    for item in codebook:
        try:
            state_record = state_records[
                (item.source_state_file, item.source_state_record_id)
            ]
        except KeyError as exc:
            raise ValueError("codebook source state record is missing") from exc
        if _digest(state_record) != item.source_state_record_sha256:
            raise ValueError("source state record digest mismatch")
        state = state_record.get("terminal_native_state")
        if not isinstance(state, Mapping):
            raise ValueError("source state record lacks terminal native state")
        if (
            state.get("state_id") != item.terminal_state_id
            or _digest(state) != item.terminal_state_sha256
        ):
            raise ValueError("terminal native state binding mismatch")

    battery_rows = _safe_jsonl(run_path / SOURCE_BATTERIES)
    batteries: dict[str, TerminalBattery] = {}
    canonical_battery_rows = []
    for raw in battery_rows:
        domain_id = _require_text(raw.get("domain_id"), "battery domain_id")
        battery = build_terminal_battery(
            domain_for_split(get_domain(domain_id), "test")
        )
        expected = {"schema_version": 1, **battery.to_dict()}
        if raw != expected:
            raise ValueError(
                f"persisted terminal battery changed for {domain_id}"
            )
        if domain_id in batteries:
            raise ValueError("duplicate terminal battery domain")
        batteries[domain_id] = battery
        canonical_battery_rows.append(battery.to_dict())
    if packet_manifest["terminal_battery_set_sha256"] != _digest(
        sorted(canonical_battery_rows, key=lambda row: row["domain_id"])
    ):
        raise ValueError("terminal battery set digest mismatch")
    for item in codebook:
        battery = batteries.get(item.domain_id)
        if (
            battery is None
            or battery.battery_id != item.battery_id
            or battery.battery_digest != item.battery_digest
        ):
            raise ValueError("codebook references a changed terminal battery")
    return metric_rows, batteries


def _strict_two_source_design(
    requests: Sequence[ExternalDecoderRequest],
    judgments: Sequence[ExternalDecoderJudgment],
    *,
    official_collection_inputs: Mapping[str, Any] | None,
    official_collection_summary: Mapping[str, Any] | None,
) -> dict[str, Any]:
    audit = validate_external_decoder_import(
        requests,
        judgments,
        minimum_sources_per_request=2,
        require_distinct_families=True,
    )
    if not audit.complete_coverage or not audit.source_design_eligible:
        raise ValueError("external decoder judgments have incomplete coverage")
    if any(row.judgment_origin != "external_model" for row in judgments):
        raise ValueError(
            "Experiment C external rescore requires caller-declared "
            "external_model judgments"
        )
    by_request: dict[str, list[ExternalDecoderJudgment]] = {}
    for judgment in judgments:
        by_request.setdefault(judgment.request_id, []).append(judgment)
    if set(by_request) != {request.request_id for request in requests}:
        raise ValueError("judgments do not cover requests exactly")
    for request_id, rows in by_request.items():
        if len(rows) != 2:
            raise ValueError(
                f"{request_id}: rescore requires exactly two judgments"
            )
        if (
            len({row.decoder_family_id for row in rows}) != 2
            or len({row.decoder_instance_id for row in rows}) != 2
            or len({row.source_descriptor for row in rows}) != 2
        ):
            raise ValueError(
                f"{request_id}: decoder family/source metadata is not distinct"
            )
    families = {row.decoder_family_id for row in judgments}
    instances = {row.decoder_instance_id for row in judgments}
    descriptors = {row.source_descriptor for row in judgments}
    if len(families) != 2 or len(instances) != 2 or len(descriptors) != 2:
        raise ValueError(
            "one fixed pair of exactly two decoder families, instances, and "
            "source descriptors must cover the full packet"
        )
    official = (
        official_collection_inputs is not None
        and official_collection_summary is not None
    )
    if (official_collection_inputs is None) != (
        official_collection_summary is None
    ):
        raise ValueError(
            "official decoder collection provenance must be complete"
        )
    provenance_mode = (
        official_collection_summary.get(
            "provenance_mode",
            "validated_direct_first_party_collection",
        )
        if official_collection_summary is not None
        else "reviewed_generic_judgments"
    )
    if provenance_mode not in {
        "validated_direct_first_party_collection",
        "selected_openrouter_gateway_collection",
        "reviewed_generic_judgments",
    }:
        raise ValueError("decoder collection provenance mode is invalid")
    openrouter_collection = (
        provenance_mode == "selected_openrouter_gateway_collection"
    )
    return {
        "import_audit": audit.to_dict(),
        "decoder_family_ids": sorted(families),
        "decoder_instance_ids": sorted(instances),
        "source_descriptors": sorted(descriptors),
        "metadata_design_eligible": True,
        "provenance_mode": provenance_mode,
        "provider_provenance_validated": (
            official and not openrouter_collection
        ),
        "gateway_provenance_validated": (
            official and openrouter_collection
        ),
        "first_party_origin_claimed": (
            official and not openrouter_collection
        ),
        "shared_gateway": openrouter_collection,
        "distinct_transport_origins": (
            official and not openrouter_collection
        ),
        "caller_declared_source_metadata_only": not official,
        "official_collection_inputs": (
            dict(official_collection_inputs)
            if official_collection_inputs is not None
            else None
        ),
        "official_collection_summary": (
            dict(official_collection_summary)
            if official_collection_summary is not None
            else None
        ),
        "statistical_independence_claimed": False,
        "independence_caveat": INDEPENDENCE_CAVEAT,
    }


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _terminal_score(raw: Mapping[str, Any]) -> TerminalBatteryScore:
    expected = {
        "profile_brier",
        "behavioral_accuracy",
        "tie_excluded_behavioral_accuracy",
        "fractional_behavioral_accuracy",
        "cross_context_accuracy",
        "mean_intrinsic_regret",
        "predicted_option_ids",
        "predicted_utility_tie_count",
        "intrinsic_utility_tie_count",
        "evaluated_item_count",
        "profile_ece",
        "profile_calibration_sample_unit",
        "profile_calibration_prediction_count",
        "profile_reliability_bins",
        "profile_calibration_interpretation",
    }
    _strict_fields(raw, expected, name="terminal battery score")
    if (
        raw["profile_calibration_sample_unit"]
        != "preference_attribute_forecast"
    ):
        raise ValueError("terminal score calibration unit changed")
    bins = raw["profile_reliability_bins"]
    if not isinstance(bins, Sequence) or isinstance(bins, (str, bytes)):
        raise ValueError("profile_reliability_bins must be an array")
    parsed_bins = []
    for row in bins:
        if not isinstance(row, Mapping):
            raise ValueError("profile reliability bin must be an object")
        _strict_fields(
            row,
            {
                "bin_index",
                "lower",
                "upper",
                "prediction_count",
                "mean_confidence",
                "empirical_accuracy",
            },
            name="profile reliability bin",
        )
        parsed_bins.append(
            TerminalReliabilityBin(
                bin_index=int(row["bin_index"]),
                lower=_finite_number(row["lower"], "bin lower"),
                upper=_finite_number(row["upper"], "bin upper"),
                prediction_count=int(row["prediction_count"]),
                mean_confidence=(
                    None
                    if row["mean_confidence"] is None
                    else _finite_number(
                        row["mean_confidence"], "mean_confidence"
                    )
                ),
                empirical_accuracy=(
                    None
                    if row["empirical_accuracy"] is None
                    else _finite_number(
                        row["empirical_accuracy"], "empirical_accuracy"
                    )
                ),
            )
        )
    predicted = raw["predicted_option_ids"]
    if not isinstance(predicted, Sequence) or isinstance(
        predicted, (str, bytes)
    ):
        raise ValueError("predicted_option_ids must be an array")
    return TerminalBatteryScore(
        profile_brier=_finite_number(raw["profile_brier"], "profile_brier"),
        behavioral_accuracy=_finite_number(
            raw["behavioral_accuracy"], "behavioral_accuracy"
        ),
        tie_excluded_behavioral_accuracy=(
            None
            if raw["tie_excluded_behavioral_accuracy"] is None
            else _finite_number(
                raw["tie_excluded_behavioral_accuracy"],
                "tie_excluded_behavioral_accuracy",
            )
        ),
        fractional_behavioral_accuracy=_finite_number(
            raw["fractional_behavioral_accuracy"],
            "fractional_behavioral_accuracy",
        ),
        cross_context_accuracy=(
            None
            if raw["cross_context_accuracy"] is None
            else _finite_number(
                raw["cross_context_accuracy"], "cross_context_accuracy"
            )
        ),
        mean_intrinsic_regret=_finite_number(
            raw["mean_intrinsic_regret"], "mean_intrinsic_regret"
        ),
        predicted_option_ids=tuple(
            _require_text(item, "predicted option ID") for item in predicted
        ),
        predicted_utility_tie_count=_finite_number(
            raw["predicted_utility_tie_count"],
            "predicted_utility_tie_count",
        ),
        intrinsic_utility_tie_count=_finite_number(
            raw["intrinsic_utility_tie_count"],
            "intrinsic_utility_tie_count",
        ),
        evaluated_item_count=int(raw["evaluated_item_count"]),
        profile_ece=(
            None
            if raw["profile_ece"] is None
            else _finite_number(raw["profile_ece"], "profile_ece")
        ),
        profile_reliability_bins=tuple(parsed_bins),
        profile_calibration_prediction_count=int(
            raw["profile_calibration_prediction_count"]
        ),
    )


def _evaluation_row_for_ranking(raw: Mapping[str, Any]) -> EvaluationRow:
    ranking = raw.get("ranking_score")
    projection = raw.get("system_projection_score")
    if not isinstance(ranking, Mapping) or not isinstance(projection, Mapping):
        raise ValueError("Experiment C row lacks terminal score objects")
    ranking_score = _terminal_score(ranking)
    if not math.isclose(
        _finite_number(raw.get("profile_error"), "profile_error"),
        ranking_score.profile_brier,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("Experiment C top-level/ranking profile error mismatch")
    signatures = raw.get("event_signatures")
    predicted = raw.get("predicted_option_ids")
    if (
        not isinstance(signatures, Sequence)
        or isinstance(signatures, (str, bytes))
        or not isinstance(predicted, Sequence)
        or isinstance(predicted, (str, bytes))
    ):
        raise ValueError("Experiment C row sequence fields are invalid")
    return EvaluationRow(
        split=_require_text(raw.get("split"), "split"),
        regime=_require_text(raw.get("regime"), "regime"),
        replicate=int(raw["replicate"]),
        user_id=_require_text(raw.get("user_id"), "user_id"),
        domain_id=_require_text(raw.get("domain_id"), "domain_id"),
        updater_id=_require_text(raw.get("updater_id"), "updater_id"),
        profile_error=ranking_score.profile_brier,
        behavioral_accuracy=_finite_number(
            raw.get("behavioral_accuracy"), "behavioral_accuracy"
        ),
        cross_context_accuracy=(
            None
            if raw.get("cross_context_accuracy") is None
            else _finite_number(
                raw.get("cross_context_accuracy"),
                "cross_context_accuracy",
            )
        ),
        intrinsic_regret=_finite_number(
            raw.get("intrinsic_regret"), "intrinsic_regret"
        ),
        history_digest=_validate_digest(
            raw.get("history_digest"), "history_digest"
        ),
        event_signatures=tuple(
            _validate_digest(item, "event signature") for item in signatures
        ),
        battery_id=_require_text(raw.get("battery_id"), "battery_id"),
        battery_digest=_validate_digest(
            raw.get("battery_digest"), "battery_digest"
        ),
        predicted_option_ids=tuple(
            _require_text(item, "predicted option ID") for item in predicted
        ),
        score_basis=_require_text(raw.get("score_basis"), "score_basis"),
        system_projection_score=_terminal_score(projection),
        ranking_score=ranking_score,
    )


def _gate_5_from_ranking(ranking: Any) -> GateReport:
    esr_payload = dict(ranking.evaluation_selection_regret)
    esr = esr_payload.get("evaluation_selection_regret")
    esr_min = esr_payload.get("evaluation_selection_regret_min")
    esr_interval_lower = esr_payload.get(
        "evaluation_selection_regret_interval_envelope_lower"
    )
    reversal = max(
        dict(ranking.pairwise_reversal_probabilities).values(),
        default=0.0,
    )
    tau = ranking.open_closed_kendall_tau
    credible_reversals = list(ranking.credible_pairwise_reversals)
    finite_tau = isinstance(tau, (int, float)) and math.isfinite(float(tau))
    low_rank_agreement_descriptive = finite_tau and float(tau) < 0.50
    substantial_esr = (
        isinstance(esr_min, (int, float))
        and math.isfinite(float(esr_min))
        and float(esr_min) > 0.01
        and isinstance(esr_interval_lower, (int, float))
        and math.isfinite(float(esr_interval_lower))
        and float(esr_interval_lower) > 0.01
    )
    evidence_available = (
        ranking.development_cluster_count > 0
        and ranking.test_cluster_count > 0
        and bool(ranking.pairwise_open_closed_shift_intervals)
    )
    return GateReport(
        gate_id="gate-5",
        title="Evaluation implication",
        criteria=(
            GateCriterion(
                "evaluation-implication-disjunction",
                (
                    "A joint-paired credible pairwise reversal or substantial "
                    "inferential-top-tier evaluation selection regret is present."
                ),
                (
                    (bool(credible_reversals) or substantial_esr)
                    if evidence_available
                    else None
                ),
                {
                    "kendall_tau_b": tau,
                    "low_rank_agreement_threshold": 0.50,
                    "low_rank_agreement_descriptive": (
                        low_rank_agreement_descriptive
                    ),
                    "kendall_tau_is_gate_sufficient_without_interval": False,
                    "credible_reversed_pairs": credible_reversals,
                    "credible_reversal_method": (
                        "joint paired complete-user open/closed error-"
                        "difference and difference-of-differences intervals"
                    ),
                    "maximum_pairwise_reversal_probability": reversal,
                    "reversal_probability_is_gate_sufficient": False,
                    "evaluation_selection_regret": esr,
                    "evaluation_selection_regret_min": esr_min,
                    "evaluation_selection_regret_interval_envelope_lower": (
                        esr_interval_lower
                    ),
                    "evaluation_selection_basis": esr_payload.get(
                        "selection_basis"
                    ),
                    "selection_regret_threshold": 0.01,
                },
                (
                    "joint-paired credible reversal OR every inferential-top-"
                    "tier ESR estimate and the paired test interval envelope "
                    "clear 0.01"
                ),
            ),
        ),
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical_json(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(_jsonl_material(rows))
        handle.flush()
        os.fsync(handle.fileno())


def _write_text_durable(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_review_tree(root: Path) -> None:
    directories = sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for directory in directories:
        _fsync_directory(directory)
    _fsync_directory(root)


def _source_regular_file(run_path: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if (
        not relative
        or pure.is_absolute()
        or pure.as_posix() != relative
        or "\\" in relative
        or "\x00" in relative
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ValueError(f"unsafe source-run path: {relative}")
    candidate = run_path.joinpath(*pure.parts)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(
            f"source-run binding requires a regular file: {relative}"
        )
    resolved = candidate.resolve()
    if resolved == run_path or run_path not in resolved.parents:
        raise ValueError(f"source-run path escapes its root: {relative}")
    return candidate


def _source_run_binding(run_path: Path) -> dict[str, Any]:
    manifest_path = _source_regular_file(run_path, "manifest.json")
    checksum_path = _source_regular_file(run_path, "SHA256SUMS")
    metrics_path = _source_regular_file(run_path, SOURCE_METRICS)
    batteries_path = _source_regular_file(run_path, SOURCE_BATTERIES)
    design_path = _source_regular_file(run_path, PACKET_MANIFEST)
    manifest = _safe_json_object(manifest_path)
    return {
        "run_id": manifest["run_id"],
        "run_manifest_sha256": _file_digest(manifest_path),
        "run_checksum_manifest_sha256": _file_digest(checksum_path),
        "config_sha256": manifest["config_sha256"],
        "source_sha256": manifest["source_sha256"],
        "experiment_c_metrics_sha256": _file_digest(metrics_path),
        "terminal_batteries_sha256": _file_digest(batteries_path),
        "decoder_design_manifest_sha256": _file_digest(design_path),
        "verified_complete": True,
    }


def _review_lock_path(output: Path) -> Path:
    return output.parent / f".{output.name}.external-rescore.lock"


def import_experiment_c_external_rescore(
    *,
    run_dir: str | Path,
    judgments_path: str | Path,
    output_dir: str | Path,
    external_collection_dir: str | Path | None = None,
    external_collection_provenance_mode: str | None = None,
    allow_reviewed_generic_decoders: bool = True,
) -> dict[str, Any]:
    """Import exactly two decoder families and atomically rerun C.

    ``external_collection_dir`` binds judgments to either a validated direct
    first-party collection or the repository-selected audited OpenRouter
    collection. The programmatic API retains a backwards-compatible generic
    path, but labels its family/source metadata as caller-declared. The CLI
    requires users to choose one provenance mode explicitly.
    """

    supplied_run = Path(run_dir)
    judgments_source = Path(judgments_path)
    supplied_output = Path(output_dir)
    supplied_collection = (
        Path(external_collection_dir)
        if external_collection_dir is not None
        else None
    )
    if supplied_collection is None and not allow_reviewed_generic_decoders:
        raise ValueError(
            "Experiment C import requires --external-collection-dir, "
            "--openrouter-collection-dir, or "
            "--allow-reviewed-generic-decoders"
        )
    if (
        supplied_collection is None
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
    if supplied_run.is_symlink():
        raise ValueError("source run cannot be a symlink")
    if judgments_source.is_symlink() or not judgments_source.is_file():
        raise ValueError("judgments must be a safe regular JSONL file")
    if supplied_output.is_symlink() or supplied_output.exists():
        raise FileExistsError(
            f"external rescore output already exists: {supplied_output}"
        )
    if supplied_output.parent.is_symlink():
        raise ValueError("external rescore output parent cannot be a symlink")

    run_path = supplied_run.resolve()
    judgments_resolved = judgments_source.resolve()
    output = supplied_output.resolve()
    collection = None
    if supplied_collection is not None:
        if supplied_collection.is_symlink() or not supplied_collection.is_dir():
            raise ValueError(
                "selected decoder collection must be a safe directory"
            )
        collection = supplied_collection.resolve()
        if external_collection_provenance_mode is not None:
            detected_openrouter = is_openrouter_decoder_collection(
                collection
            )
            expected_openrouter = (
                external_collection_provenance_mode
                == OPENROUTER_COLLECTION_PROVENANCE
            )
            if detected_openrouter is not expected_openrouter:
                expected_flag = (
                    "--openrouter-collection-dir"
                    if expected_openrouter
                    else "--external-collection-dir"
                )
                actual = (
                    "an OpenRouter shared-gateway collection"
                    if detected_openrouter
                    else "a direct first-party collection"
                )
                raise ValueError(
                    f"{expected_flag} does not match the supplied artifact: "
                    f"detected {actual}"
                )
    if not run_path.is_dir():
        raise ValueError("source run must be a safe directory")
    if output == Path(output.anchor):
        raise ValueError("external rescore output cannot be a filesystem root")
    if output == run_path or run_path in output.parents:
        raise ValueError(
            "external rescore output cannot be inside the immutable source run"
        )
    if output == judgments_resolved or judgments_resolved in output.parents:
        raise ValueError("external rescore output cannot contain its input")
    if collection is not None and (
        output == collection or collection in output.parents
    ):
        raise ValueError(
            "external rescore output cannot be inside the decoder collection"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = _review_lock_path(output)
    try:
        descriptor = os.open(
            lock_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
    except FileExistsError as exc:
        raise FileExistsError(
            f"external rescore output is locked: {output}"
        ) from exc
    os.close(descriptor)
    try:
        if output.is_symlink() or output.exists():
            raise FileExistsError(
                f"external rescore output already exists: {output}"
            )
        return _import_experiment_c_external_rescore_locked(
            run_path=run_path,
            run_input=supplied_run.absolute(),
            judgments_resolved=judgments_resolved,
            judgments_input=judgments_source.absolute(),
            output=output,
            external_collection=collection,
            external_collection_provenance_mode=(
                external_collection_provenance_mode
            ),
        )
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _import_experiment_c_external_rescore_locked(
    *,
    run_path: Path,
    run_input: Path,
    judgments_resolved: Path,
    judgments_input: Path,
    output: Path,
    external_collection: Path | None,
    external_collection_provenance_mode: str | None,
) -> dict[str, Any]:
    """Compute and stage one rescore while holding its exclusive lock."""

    valid, errors = verify_run(run_path)
    if not valid:
        raise ValueError(
            "source run verification failed: " + "; ".join(errors)
        )
    source_run = _source_run_binding(run_path)
    config_raw = _safe_json_object(run_path / "config.resolved.json")
    config = AppConfig.parse(config_raw)
    if config.experiment.kind != "evaluation_validity":
        raise ValueError("external C rescore requires Experiment C")
    if not config.artifacts.retain_events:
        raise ValueError(
            "external C rescore requires retained native terminal states"
        )

    request_rows = _safe_jsonl(run_path / PACKET_REQUESTS)
    truth_rows = _safe_jsonl(run_path / PACKET_TRUTH)
    codebook_rows = _safe_jsonl(run_path / PACKET_CODEBOOK)
    if judgments_resolved.is_symlink() or not judgments_resolved.is_file():
        raise ValueError("judgments must remain a safe regular JSONL file")
    judgment_material = judgments_resolved.read_bytes()
    judgment_rows = _jsonl_from_bytes(
        judgment_material,
        source=judgments_resolved,
    )
    requests = _decoder_requests(request_rows)
    labels = _decoder_labels(truth_rows)
    codebook = _decoder_codebook(codebook_rows)
    judgments = tuple(
        ExternalDecoderJudgment.parse(row) for row in judgment_rows
    )
    packet_manifest = _validate_packet_manifest(
        run_path, requests, labels, codebook
    )
    source_metric_rows, batteries = _validate_source_rows(
        run_path,
        packet_manifest,
        requests,
        labels,
        codebook,
    )
    official_validation: tuple[
        tuple[ExternalDecoderJudgment, ...],
        dict[str, dict[str, Any]],
        dict[str, Any],
    ] | None = None
    if external_collection is not None:
        official_validation = (
            validate_selected_external_decoder_collection(
                external_collection,
                run_dir=run_path,
                requests=requests,
                judgments_path=judgments_resolved,
                expected_provenance_mode=(
                    external_collection_provenance_mode
                ),
            )
        )
        if official_validation[0] != judgments:
            raise ValueError(
                "selected decoder collection judgments changed ordering or "
                "content during validation"
            )
    source_design = _strict_two_source_design(
        requests,
        judgments,
        official_collection_inputs=(
            official_validation[1]
            if official_validation is not None
            else None
        ),
        official_collection_summary=(
            official_validation[2]
            if official_validation is not None
            else None
        ),
    )

    calibration = fit_decoder_calibration(requests, judgments, labels)
    if calibration.fitted_split != "development" or any(
        calibrator.fitted_splits != ("development",)
        for _, calibrator in calibration.calibrators
    ):
        raise ValueError("decoder calibration used non-development labels")
    development_request_count = sum(
        request.evaluation_split == "development" for request in requests
    )
    if development_request_count <= 0:
        raise ValueError("decoder calibration lacks development requests")
    expected_examples = development_request_count * 3
    if any(
        calibrator.example_count != expected_examples
        for _, calibrator in calibration.calibrators
    ):
        raise ValueError(
            "decoder calibration coverage includes an unexpected split"
        )
    decoder_analysis = analyze_external_decoders(
        requests,
        judgments,
        labels,
        calibration=calibration,
        evaluation_splits=("test",),
    )

    request_by_id = {item.request_id: item for item in requests}
    label_by_state = {
        item.pseudonymous_state_id: item for item in labels
    }
    judgments_by_request: dict[str, list[ExternalDecoderJudgment]] = {}
    for judgment in judgments:
        judgments_by_request.setdefault(
            judgment.request_id, []
        ).append(judgment)
    external_score_rows: list[dict[str, Any]] = []
    averaged_by_metric_sha: dict[str, TerminalBatteryScore] = {}
    for binding in sorted(
        codebook,
        key=lambda item: (
            item.evaluation_split,
            item.regime,
            item.user_id,
            item.domain_id,
            item.replicate,
            item.updater_id,
        ),
    ):
        request = request_by_id[binding.request_id]
        label = label_by_state[binding.pseudonymous_state_id]
        battery = batteries[binding.domain_id]
        user = LatentUser(binding.user_id, label.theta)
        source_scores: list[TerminalBatteryScore] = []
        for judgment in sorted(
            judgments_by_request[request.request_id],
            key=lambda item: (
                item.decoder_family_id,
                item.decoder_instance_id,
            ),
        ):
            calibrator = calibration.for_family(
                judgment.decoder_family_id
            )
            calibrated_rows = tuple(
                calibrator.apply(row)
                for row in judgment.probabilities
            )
            belief = PreferenceBelief.from_marginals(
                MarginalPreferenceBelief(calibrated_rows)  # type: ignore[arg-type]
            )
            score = evaluate_terminal_battery(belief, user, battery)
            source_scores.append(score)
            external_score_rows.append(
                {
                    "schema_version": 1,
                    "source_metric_row_sha256": (
                        binding.source_metric_row_sha256
                    ),
                    "stable_row_key_sha256": (
                        binding.stable_row_key_sha256
                    ),
                    "request_id": request.request_id,
                    "request_sha256": request.request_sha256,
                    "pseudonymous_state_id": (
                        request.pseudonymous_state_id
                    ),
                    "evaluation_split": binding.evaluation_split,
                    "regime": binding.regime,
                    "replicate": binding.replicate,
                    "user_id": binding.user_id,
                    "domain_id": binding.domain_id,
                    "updater_id": binding.updater_id,
                    "battery_id": binding.battery_id,
                    "battery_digest": binding.battery_digest,
                    "decoder_instance_id": judgment.decoder_instance_id,
                    "decoder_family_id": judgment.decoder_family_id,
                    "source_descriptor": judgment.source_descriptor,
                    "judgment_origin": judgment.judgment_origin,
                    "blind_to_system_identity": (
                        judgment.blind_to_system_identity
                    ),
                    "blind_to_latent_truth": judgment.blind_to_latent_truth,
                    "calibration_fitted_split": "development",
                    "calibration_temperature": calibrator.temperature,
                    "calibration_example_count": (
                        calibrator.example_count
                    ),
                    "calibrated_marginals": [
                        {
                            label_name: probability
                            for label_name, probability in zip(
                                ("-2", "-1", "+1", "+2"),
                                row,
                            )
                        }
                        for row in calibrated_rows
                    ],
                    "terminal_score": score.to_dict(),
                }
            )
        averaged_by_metric_sha[binding.source_metric_row_sha256] = (
            mean_terminal_battery_scores(
                source_scores,
                required_score_count=2,
            )
        )

    if set(averaged_by_metric_sha) != {
        item.source_metric_row_sha256 for item in codebook
    }:
        raise ValueError("external scores do not cover native rows exactly")
    if len(external_score_rows) != 2 * len(codebook):
        raise ValueError("external score output is not exactly two per row")

    rescored_rows: list[dict[str, Any]] = []
    allowed_native_changes = {
        "profile_error",
        "behavioral_accuracy",
        "cross_context_accuracy",
        "intrinsic_regret",
        "predicted_option_ids",
        "score_basis",
        "ranking_score",
    }
    non_native_source: list[dict[str, Any]] = []
    non_native_rescored: list[dict[str, Any]] = []
    for source in source_metric_rows:
        source_sha = _digest(source)
        if source.get("updater_id") not in NATIVE_UPDATER_IDS:
            rescored = dict(source)
            non_native_source.append(source)
            non_native_rescored.append(rescored)
        else:
            try:
                score = averaged_by_metric_sha[source_sha]
            except KeyError as exc:
                raise ValueError(
                    "native source row lacks external mean score"
                ) from exc
            rescored = dict(source)
            rescored.update(
                {
                    "profile_error": score.profile_brier,
                    "behavioral_accuracy": score.behavioral_accuracy,
                    "cross_context_accuracy": (
                        score.cross_context_accuracy
                    ),
                    "intrinsic_regret": score.mean_intrinsic_regret,
                    "predicted_option_ids": [],
                    "score_basis": RESCORE_BASIS,
                    "ranking_score": score.to_dict(),
                }
            )
            changed = {
                key
                for key in set(source) | set(rescored)
                if source.get(key) != rescored.get(key)
            }
            if not changed <= allowed_native_changes:
                raise ValueError(
                    "external rescore changed protected native row fields"
                )
        rescored_rows.append(rescored)
    if non_native_source != non_native_rescored:
        raise ValueError("external rescore changed non-native rows")

    ranking_rows = tuple(
        _evaluation_row_for_ranking(row) for row in rescored_rows
    )
    updater_ids = tuple(config.experiment.updaters)
    if set(updater_ids) != {row.updater_id for row in ranking_rows}:
        raise ValueError(
            "source metric updater coverage differs from resolved config"
        )
    rankings = analyze_rankings(
        ranking_rows,
        updater_ids=updater_ids,
        bootstrap_replicates=config.experiment.bootstrap_replicates,
        seed=config.run.seed,
        tie_tolerance=config.thresholds.ranking_tie_tolerance,
    )
    gate_5 = _gate_5_from_ranking(rankings)

    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.",
            suffix=".staging",
            dir=output.parent,
        )
    )
    published = False
    try:
        retained_inputs = {
            "inputs/external-requests.jsonl": request_rows,
            "inputs/truth-labels.researcher-only.jsonl": truth_rows,
            "inputs/researcher-codebook.jsonl": codebook_rows,
            "inputs/judgments.jsonl": [
                item.to_dict()
                for item in sorted(
                    judgments,
                    key=lambda row: (
                        row.request_id,
                        row.decoder_family_id,
                        row.decoder_instance_id,
                    ),
                )
            ],
        }
        for relative, rows in retained_inputs.items():
            _write_jsonl(stage / relative, rows)
        _write_jsonl(
            stage / "metrics/external-decoder-scores.jsonl",
            external_score_rows,
        )
        _write_jsonl(
            stage / "metrics/experiment-c-rescored.jsonl",
            rescored_rows,
        )
        _write_json(
            stage / "metrics/calibration.json",
            calibration.to_dict(),
        )
        _write_json(
            stage / "metrics/decoder-analysis.json",
            decoder_analysis.to_dict(),
        )
        _write_json(
            stage / "metrics/experiment-c-rankings.json",
            rankings.to_dict(),
        )
        _write_json(stage / "metrics/gate-5.json", gate_5.to_dict())

        generated_paths = sorted(
            (
                path
                for path in stage.rglob("*")
                if path.is_file()
            ),
            key=lambda path: path.relative_to(stage).as_posix(),
        )
        retained_files = {
            path.relative_to(stage).as_posix(): {
                "sha256": _file_digest(path),
                "bytes": len(path.read_bytes()),
            }
            for path in generated_paths
        }
        review_core = {
            "schema_version": 1,
            "artifact_kind": REVIEW_KIND,
            "claim_status": "not_claimed",
            "source_run": source_run,
            "input_judgments": {
                "source_filename": judgments_resolved.name,
                "source_sha256": sha256(judgment_material).hexdigest(),
                "canonical_retained_sha256": retained_files[
                    "inputs/judgments.jsonl"
                ]["sha256"],
                "record_count": len(judgments),
            },
            "validation": {
                "source_run_verified": True,
                "source_run_mutated": False,
                "packet_manifest_validated": True,
                "request_truth_codebook_coverage_complete": True,
                "row_hashes_validated": True,
                "native_state_hashes_validated": True,
                "battery_hashes_validated": True,
                "non_native_rows_unchanged": True,
                "non_native_row_set_sha256": _digest(non_native_source),
                "exactly_two_external_scores_per_native_row": True,
                "development_only_calibration": True,
                "test_labels_used_for_calibration": False,
                "source_design": source_design,
            },
            "rescore": {
                "score_basis": RESCORE_BASIS,
                "native_row_count": len(codebook),
                "external_score_count": len(external_score_rows),
                "source_metric_row_count": len(source_metric_rows),
                "rescored_metric_row_count": len(rescored_rows),
                "common_terminal_battery_count": len(batteries),
                "ranking_analysis_rerun": True,
                "gate_5_rerun": True,
                "evaluation_selection_regret_rerun": True,
            },
            "calibration": calibration.to_dict(),
            "gate_5_computed_status": gate_5.computed_status,
            "retained_files": retained_files,
            "interpretation_boundary": (
                "This artifact validates input, calibration, rescoring, "
                "ranking, and Gate 5 computation while retaining "
                "claim_status=not_claimed. "
                + (
                    (
                        "Gateway provenance was audit-validated through "
                        "OpenRouter. The selected Claude/Gemini pair shares "
                        "one gateway; no direct first-party provider origin "
                        "or distinct transport-origin claim is made. "
                    )
                    if source_design["provenance_mode"]
                    == "selected_openrouter_gateway_collection"
                    else (
                        "Provider provenance was validated against the "
                        "complete selected first-party Anthropic/Gemini "
                        "collection. "
                    )
                    if official_validation is not None
                    else (
                        "Decoder family, instance, source, and "
                        "external_model origin are caller-declared metadata; "
                        "no provider provenance was validated. "
                    )
                )
                + INDEPENDENCE_CAVEAT
            ),
        }
        artifact_id = _digest(review_core)
        review = {**review_core, "artifact_id": artifact_id}
        _write_json(stage / "review.json", review)
        manifest = {
            "schema_version": 1,
            "artifact_kind": REVIEW_KIND,
            "artifact_id": artifact_id,
            "status": "complete",
            "claim_status": "not_claimed",
            "review_sha256": _file_digest(stage / "review.json"),
            "source_run": source_run,
            "retained_files": {
                **retained_files,
                "review.json": {
                    "sha256": _file_digest(stage / "review.json"),
                    "bytes": len((stage / "review.json").read_bytes()),
                },
            },
        }
        _write_json(stage / "manifest.json", manifest)
        checksummed = sorted(
            (
                path
                for path in stage.rglob("*")
                if path.is_file()
            ),
            key=lambda path: path.relative_to(stage).as_posix(),
        )
        _write_text_durable(
            stage / "SHA256SUMS",
            "".join(
                f"{_file_digest(path)}  "
                f"{path.relative_to(stage).as_posix()}\n"
                for path in checksummed
            ),
        )
        _fsync_review_tree(stage)

        if (
            run_input.is_symlink()
            or not run_input.is_dir()
            or run_input.resolve() != run_path
        ):
            raise ValueError("source run path changed while the import was running")
        valid, errors = verify_run(run_path)
        if not valid:
            raise ValueError(
                "source run changed while the import was running: "
                + "; ".join(errors)
            )
        if _source_run_binding(run_path) != source_run:
            raise ValueError(
                "source run binding changed while the import was running"
            )
        if (
            judgments_input.is_symlink()
            or not judgments_input.is_file()
            or judgments_input.resolve() != judgments_resolved
            or judgments_resolved.is_symlink()
            or judgments_resolved.read_bytes() != judgment_material
        ):
            raise ValueError("judgments changed while the import was running")
        if external_collection is not None:
            repeated_official_validation = (
                validate_selected_external_decoder_collection(
                    external_collection,
                    run_dir=run_path,
                    requests=requests,
                    judgments_path=judgments_resolved,
                    expected_provenance_mode=(
                        external_collection_provenance_mode
                    ),
                )
            )
            if repeated_official_validation != official_validation:
                raise ValueError(
                    "selected decoder collection changed while the import "
                    "was running"
                )

        staged_valid, staged_errors = (
            verify_experiment_c_external_rescore(
                stage,
                source_run_dir=run_path,
            )
        )
        if not staged_valid:
            raise ValueError(
                "staged external rescore failed verification: "
                + "; ".join(staged_errors)
            )
        if output.is_symlink() or output.exists():
            raise FileExistsError(
                f"external rescore output already exists: {output}"
            )
        os.rename(stage, output)
        published = True
        _fsync_directory(output.parent)
    finally:
        if not published and stage.exists():
            if stage.is_symlink():
                stage.unlink()
            else:
                shutil.rmtree(stage)
    return {
        "artifact_id": artifact_id,
        "output_dir": str(output),
        "claim_status": "not_claimed",
        "native_row_count": len(codebook),
        "external_score_count": len(external_score_rows),
        "decoder_family_ids": source_design["decoder_family_ids"],
        "provenance_mode": source_design["provenance_mode"],
        "provider_provenance_validated": source_design[
            "provider_provenance_validated"
        ],
        "gateway_provenance_validated": source_design[
            "gateway_provenance_validated"
        ],
        "gate_5_computed_status": gate_5.computed_status,
    }


def verify_experiment_c_external_rescore(
    path: str | Path,
    *,
    source_run_dir: str | Path | None = None,
) -> tuple[bool, tuple[str, ...]]:
    """Verify review checksums, semantic bindings, and optionally its source."""

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
    retained: set[str] = set()
    for line_number, line in enumerate(
        checksum_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            expected, relative = line.split("  ", 1)
            _validate_digest(expected, "checksum")
        except ValueError as exc:
            errors.append(f"malformed checksum line {line_number}: {exc}")
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
            errors.append(f"duplicate checksum path: {relative}")
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
    if retained != REVIEW_FILES:
        errors.append("review checksum manifest has an unexpected file set")
    for relative in sorted(actual - retained):
        errors.append(f"unlisted artifact: {relative}")
    unexpected_nodes = {
        item.relative_to(root).as_posix()
        for item in root.rglob("*")
        if item.is_symlink()
    }
    for relative in sorted(unexpected_nodes):
        errors.append(f"unexpected review symlink: {relative}")

    try:
        review = _safe_json_object(root / "review.json")
        manifest = _safe_json_object(root / "manifest.json")
    except ValueError as exc:
        errors.append(str(exc))
        return False, tuple(errors)
    core = dict(review)
    artifact_id = core.pop("artifact_id", None)
    if artifact_id != _digest(core):
        errors.append("review artifact_id mismatch")
    if (
        review.get("artifact_kind") != REVIEW_KIND
        or review.get("claim_status") != "not_claimed"
    ):
        errors.append("invalid review claim semantics")
    if (
        manifest.get("artifact_kind") != REVIEW_KIND
        or manifest.get("status") != "complete"
        or manifest.get("claim_status") != "not_claimed"
        or manifest.get("artifact_id") != artifact_id
        or manifest.get("review_sha256")
        != _file_digest(root / "review.json")
    ):
        errors.append("invalid review manifest semantics")
    validation = review.get("validation")
    if not isinstance(validation, Mapping) or any(
        validation.get(name) is not expected
        for name, expected in {
            "source_run_mutated": False,
            "request_truth_codebook_coverage_complete": True,
            "row_hashes_validated": True,
            "native_state_hashes_validated": True,
            "battery_hashes_validated": True,
            "non_native_rows_unchanged": True,
            "exactly_two_external_scores_per_native_row": True,
            "development_only_calibration": True,
            "test_labels_used_for_calibration": False,
        }.items()
    ):
        errors.append("review validation semantics are incomplete")
    source_design = (
        validation.get("source_design")
        if isinstance(validation, Mapping)
        else None
    )
    if not isinstance(source_design, Mapping):
        errors.append("review source-design provenance is missing")
    else:
        provenance_mode = source_design.get("provenance_mode")
        provider_validated = source_design.get(
            "provider_provenance_validated"
        )
        gateway_validated = source_design.get(
            "gateway_provenance_validated"
        )
        first_party = source_design.get("first_party_origin_claimed")
        shared_gateway = source_design.get("shared_gateway")
        distinct_origins = source_design.get(
            "distinct_transport_origins"
        )
        caller_declared = source_design.get(
            "caller_declared_source_metadata_only"
        )
        official_inputs = source_design.get("official_collection_inputs")
        official_summary = source_design.get(
            "official_collection_summary"
        )
        if provenance_mode in {
            "validated_direct_first_party_collection",
            "selected_openrouter_gateway_collection",
        }:
            is_openrouter = (
                provenance_mode
                == "selected_openrouter_gateway_collection"
            )
            if (
                provider_validated is not (not is_openrouter)
                or gateway_validated is not is_openrouter
                or first_party is not (not is_openrouter)
                or shared_gateway is not is_openrouter
                or distinct_origins is not (not is_openrouter)
                or caller_declared is not False
                or not isinstance(official_inputs, Mapping)
                or not official_inputs
                or not isinstance(official_summary, Mapping)
                or official_summary.get("provenance_mode")
                != provenance_mode
            ):
                errors.append(
                    "selected decoder collection provenance is incomplete"
                )
            if (
                is_openrouter
                and (
                    official_summary.get("gateway") != "openrouter"
                    or official_summary.get("shared_gateway") is not True
                    or official_summary.get("first_party_origin_claimed")
                    is not False
                    or official_summary.get(
                        "statistical_independence_claimed"
                    )
                    is not False
                )
            ):
                errors.append(
                    "OpenRouter decoder provenance boundary is incomplete"
                )
        elif provenance_mode == "reviewed_generic_judgments":
            if (
                provider_validated is not False
                or gateway_validated is not False
                or first_party is not False
                or shared_gateway is not False
                or distinct_origins is not False
                or caller_declared is not True
                or official_inputs is not None
                or official_summary is not None
            ):
                errors.append(
                    "generic decoder provenance boundary is inconsistent"
                )
        else:
            errors.append("review decoder provenance mode is invalid")
    try:
        _decoder_requests(
            _safe_jsonl(root / "inputs/external-requests.jsonl")
        )
        _decoder_labels(
            _safe_jsonl(root / "inputs/truth-labels.researcher-only.jsonl")
        )
        _decoder_codebook(
            _safe_jsonl(root / "inputs/researcher-codebook.jsonl")
        )
        judgments = tuple(
            ExternalDecoderJudgment.parse(row)
            for row in _safe_jsonl(root / "inputs/judgments.jsonl")
        )
        if not judgments:
            raise ValueError("retained judgments cannot be empty")
        score_rows = _safe_jsonl(
            root / "metrics/external-decoder-scores.jsonl"
        )
        rescored_rows = _safe_jsonl(
            root / "metrics/experiment-c-rescored.jsonl"
        )
        if len(score_rows) % 2:
            raise ValueError("external score count is not even")
        if any(
            row.get("updater_id") in NATIVE_UPDATER_IDS
            and row.get("score_basis") != RESCORE_BASIS
            for row in rescored_rows
        ):
            raise ValueError("native rescore row has the wrong score basis")
        calibration = _safe_json_object(root / "metrics/calibration.json")
        if (
            calibration.get("fitted_split") != "development"
            or any(
                raw.get("fitted_splits") != ["development"]
                for raw in calibration.get("calibrators", {}).values()
                if isinstance(raw, Mapping)
            )
        ):
            raise ValueError("retained calibration is not development-only")
        gate = _safe_json_object(root / "metrics/gate-5.json")
        if (
            gate.get("gate_id") != "gate-5"
            or gate.get("claim_status") != "not_claimed"
        ):
            raise ValueError("retained Gate 5 semantics are invalid")
    except (AttributeError, TypeError, ValueError) as exc:
        errors.append(f"invalid retained review content: {exc}")

    if source_run_dir is not None:
        supplied_source = Path(source_run_dir)
        if supplied_source.is_symlink():
            errors.append("source run cannot be a symlink")
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
                    if review.get("source_run") != expected_source:
                        errors.append("review/source run binding mismatch")
                except (OSError, ValueError) as exc:
                    errors.append(f"invalid source run binding: {exc}")
    return not errors, tuple(errors)
