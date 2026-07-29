"""Compact, checksum-bound analysis projections for completed runs.

The full event archives are the audit and replay record.  This module creates a
separate, immutable bundle containing only the row-level fields used for
statistical analysis.  It never modifies the source run.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from typing import Any, BinaryIO, Iterator, Mapping

from . import __version__
from .artifacts import (
    canonical_json,
    file_sha256,
    read_control_bytes,
    verify_run,
)


_BUNDLE_KIND = "cape-loop-compact-analysis-bundle"
_BUNDLE_FILES = frozenset(
    {"manifest.json", "analysis-rows.jsonl", "SHA256SUMS"}
)
_THETA_VALUES = (-2, -1, 1, 2)
_EXPERIMENT_BY_KIND = {
    "provenance_audit": "A",
    "closed_loop": "B",
    "evaluation_validity": "C",
}
_INTERNAL_INPUTS = {
    "A": "analysis/experiment-a-rows.jsonl",
    "B": "analysis/experiment-b-turns.jsonl",
    "C": "analysis/experiment-c-rows.jsonl",
}
_LEGACY_INPUTS = {
    "A": "events/experiment-a.jsonl",
    "B": "events/experiment-b-trajectories.jsonl",
    "C": "metrics/experiment-c.jsonl",
}
_ANALYSIS_UNITS = {
    "A": "updater_trial",
    "B": "retained_trajectory_turn",
    "C": "evaluation_row",
}
_A_FIELDS = frozenset(
    {
        "schema_version",
        "source_record_index",
        "trial_id",
        "user_id",
        "domain_id",
        "scenario_id",
        "updater_id",
        "mechanism",
        "prior_strength",
        "response_mode",
        "update_error",
    }
)
_B_FIELDS = frozenset(
    {
        "schema_version",
        "source_record_index",
        "source_turn_index",
        "trajectory_id",
        "user_id",
        "domain_id",
        "crn_key",
        "updater_id",
        "policy_id",
        "initial_profile_condition",
        "turn",
        "terminal_error",
        "retained_terminal_error",
        "same_history_shadow",
    }
)
_C_FIELDS = frozenset(
    {
        "schema_version",
        "source_record_index",
        "split",
        "regime",
        "replicate",
        "user_id",
        "domain_id",
        "updater_id",
        "profile_error",
        "behavioral_accuracy",
        "cross_context_accuracy",
        "intrinsic_regret",
        "score_basis",
        "history_digest",
        "battery_id",
        "battery_digest",
    }
)
_OUTCOME_DERIVATIONS = {
    "A": "retained Experiment A metrics.acue",
    "B": (
        "per-turn marginal Brier from retained belief_after "
        "against immutable trajectory theta"
    ),
    "C": "retained active ranking_score.profile_brier",
}
_BASE_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_kind",
        "status",
        "claim_status",
        "experiment",
        "analysis_unit",
        "row_schema_version",
        "row_count",
        "source_record_count",
        "configured_turns",
        "analysis_rows_file",
        "analysis_rows_sha256",
        "source_run_id",
        "source_manifest_sha256",
        "source_checksums_sha256",
        "source_config_file_sha256",
        "source_summary_file_sha256",
        "source_config_sha256",
        "source_tree_sha256",
        "source_input_file",
        "source_input_sha256",
        "source_input_is_runner_compact",
        "exporter_version",
        "exporter_source_sha256",
        "outcome_derivation",
    }
)


@dataclass(frozen=True, slots=True)
class CompactAnalysisBundle:
    """Published compact bundle metadata."""

    path: Path
    experiment: str
    source_run_id: str
    row_count: int
    analysis_rows_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "experiment": self.experiment,
            "source_run_id": self.source_run_id,
            "row_count": self.row_count,
            "analysis_rows_sha256": self.analysis_rows_sha256,
        }


class _HashingReader:
    """Binary reader proxy that hashes exactly the bytes consumed."""

    def __init__(self, handle: BinaryIO) -> None:
        self._handle = handle
        self._digest = sha256()

    def __iter__(self) -> _HashingReader:
        return self

    def __next__(self) -> bytes:
        line = next(self._handle)
        self._digest.update(line)
        return line

    @property
    def hexdigest(self) -> str:
        return self._digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_json_object(path: Path, label: str) -> Mapping[str, Any]:
    try:
        decoded = json.loads(read_control_bytes(path, label=label))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(decoded, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return decoded


def _parse_checksums(path: Path) -> dict[str, str]:
    try:
        text = read_control_bytes(path, label="SHA256SUMS").decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"SHA256SUMS is not UTF-8: {exc}") from exc
    result: dict[str, str] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise ValueError(f"SHA256SUMS contains blank line {line_number}")
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as exc:
            raise ValueError(
                f"malformed SHA256SUMS line {line_number}"
            ) from exc
        pure = PurePosixPath(relative)
        if (
            not _is_sha256(digest)
            or not relative
            or pure.is_absolute()
            or pure.as_posix() != relative
            or "\\" in relative
            or any(part in {"", ".", ".."} for part in pure.parts)
            or relative in result
        ):
            raise ValueError(f"invalid SHA256SUMS line {line_number}")
        result[relative] = digest
    if not result:
        raise ValueError("SHA256SUMS is empty")
    return result


def _required_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _required_integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
    ):
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _required_number(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _optional_number(value: Any, label: str) -> float | None:
    if value is None:
        return None
    return _required_number(value, label)


def _decode_jsonl_line(
    raw_line: bytes,
    *,
    path: Path,
    line_number: int,
) -> Mapping[str, Any]:
    if raw_line in {b"", b"\n", b"\r\n"}:
        raise ValueError(f"{path}:{line_number}: blank JSONL row")
    try:
        line = raw_line.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"{path}:{line_number}: row is not UTF-8"
        ) from exc
    try:
        decoded = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if not isinstance(decoded, Mapping):
        raise ValueError(f"{path}:{line_number}: row must be an object")
    return decoded


def _iter_hashed_jsonl(
    path: Path,
) -> tuple[Iterator[tuple[int, Mapping[str, Any]]], _HashingReader, BinaryIO]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"analysis source must be a regular file: {path}")
    handle = path.open("rb")
    hashing_reader = _HashingReader(handle)

    def rows() -> Iterator[tuple[int, Mapping[str, Any]]]:
        for line_number, raw_line in enumerate(hashing_reader, start=1):
            yield line_number, _decode_jsonl_line(
                raw_line,
                path=path,
                line_number=line_number,
            )

    return rows(), hashing_reader, handle


def _experiment_a_row(
    raw: Mapping[str, Any],
    source_record_index: int,
    *,
    compact: bool,
) -> dict[str, Any]:
    if compact:
        scenario_id = raw.get("scenario_id")
        update_error = raw.get("update_error")
    else:
        scenario_id = _required_mapping(
            raw.get("context"),
            "Experiment A context",
        ).get("scenario_id")
        update_error = _required_mapping(
            raw.get("metrics"),
            "Experiment A metrics",
        ).get("acue")
    row = {
        "schema_version": 1,
        "source_record_index": source_record_index,
        "trial_id": _required_text(raw.get("trial_id"), "trial_id"),
        "user_id": _required_text(raw.get("user_id"), "user_id"),
        "domain_id": _required_text(raw.get("domain_id"), "domain_id"),
        "scenario_id": _required_text(scenario_id, "scenario_id"),
        "updater_id": _required_text(raw.get("updater_id"), "updater_id"),
        "mechanism": _required_text(raw.get("mechanism"), "mechanism"),
        "prior_strength": _required_number(
            raw.get("prior_strength"),
            "prior_strength",
        ),
        "response_mode": _required_text(
            raw.get("response_mode"),
            "response_mode",
        ),
        "update_error": _required_number(update_error, "update_error"),
    }
    if raw.get("schema_version") != 1:
        raise ValueError("unsupported Experiment A row schema_version")
    if compact:
        retained_index = _required_integer(
            raw.get("source_record_index"),
            "source_record_index",
            minimum=1,
        )
        if retained_index != source_record_index:
            raise ValueError(
                "Experiment A compact source indexes are not contiguous"
            )
    return row


def _belief_marginals(value: Any, label: str) -> tuple[tuple[float, ...], ...]:
    belief = _required_mapping(value, label)
    rows = (
        belief.get("marginals")
        if "marginals" in belief
        else belief.get("probabilities")
    )
    if not isinstance(rows, list) or len(rows) != 3:
        raise ValueError(f"{label} must contain three marginal rows")
    validated: list[tuple[float, ...]] = []
    for attribute, raw_row in enumerate(rows):
        if not isinstance(raw_row, list) or len(raw_row) != 4:
            raise ValueError(
                f"{label} marginal {attribute} must contain four values"
            )
        probabilities = tuple(
            _required_number(
                probability,
                f"{label} marginal {attribute} probability",
            )
            for probability in raw_row
        )
        if (
            any(probability < 0.0 or probability > 1.0 for probability in probabilities)
            or abs(math.fsum(probabilities) - 1.0) > 1e-9
        ):
            raise ValueError(f"{label} contains an invalid probability vector")
        validated.append(probabilities)
    return tuple(validated)


def _marginal_brier(value: Any, theta: Any, label: str) -> float:
    if not isinstance(theta, list) or len(theta) != 3:
        raise ValueError(f"{label} theta must contain three values")
    theta_values = tuple(
        _required_integer(item, f"{label} theta", minimum=-2)
        for item in theta
    )
    if any(item not in _THETA_VALUES for item in theta_values):
        raise ValueError(f"{label} theta values are outside the declared grid")
    marginals = _belief_marginals(value, label)
    scores = []
    for attribute, probabilities in enumerate(marginals):
        expected_index = _THETA_VALUES.index(theta_values[attribute])
        scores.append(
            math.fsum(
                (
                    probability
                    - (1.0 if index == expected_index else 0.0)
                )
                ** 2
                for index, probability in enumerate(probabilities)
            )
        )
    return math.fsum(scores) / 3.0


def _experiment_b_compact_row(
    raw: Mapping[str, Any],
    *,
    expected_record_index: int,
    expected_turn_index: int,
) -> dict[str, Any]:
    if raw.get("schema_version") != 1:
        raise ValueError("unsupported Experiment B compact schema_version")
    source_record_index = _required_integer(
        raw.get("source_record_index"),
        "source_record_index",
        minimum=1,
    )
    source_turn_index = _required_integer(
        raw.get("source_turn_index"),
        "source_turn_index",
    )
    if (
        source_record_index != expected_record_index
        or source_turn_index != expected_turn_index
    ):
        raise ValueError("Experiment B compact indexes are not contiguous")
    same_history_shadow = raw.get("same_history_shadow")
    if same_history_shadow is not True:
        raise ValueError("Experiment B row lacks a same-history shadow")
    turn = _required_integer(raw.get("turn"), "turn", minimum=1)
    if turn != source_turn_index + 1:
        raise ValueError("Experiment B turn is not source_turn_index + 1")
    return {
        "schema_version": 1,
        "source_record_index": source_record_index,
        "source_turn_index": source_turn_index,
        "trajectory_id": _required_text(
            raw.get("trajectory_id"),
            "trajectory_id",
        ),
        "user_id": _required_text(raw.get("user_id"), "user_id"),
        "domain_id": _required_text(raw.get("domain_id"), "domain_id"),
        "crn_key": _required_text(raw.get("crn_key"), "crn_key"),
        "updater_id": _required_text(raw.get("updater_id"), "updater_id"),
        "policy_id": _required_text(raw.get("policy_id"), "policy_id"),
        "initial_profile_condition": _required_text(
            raw.get("initial_profile_condition"),
            "initial_profile_condition",
        ),
        "turn": turn,
        "terminal_error": _required_number(
            raw.get("terminal_error"),
            "terminal_error",
        ),
        "retained_terminal_error": _required_number(
            raw.get("retained_terminal_error"),
            "retained_terminal_error",
        ),
        "same_history_shadow": True,
    }


def _experiment_b_rows(
    raw: Mapping[str, Any],
    source_record_index: int,
    configured_turns: int,
) -> Iterator[dict[str, Any]]:
    if raw.get("schema_version") != 1:
        raise ValueError("unsupported Experiment B trajectory schema_version")
    if raw.get("same_history_shadow") is not True:
        raise ValueError("Experiment B trajectory lacks a same-history shadow")
    trajectory_id = _required_text(raw.get("trajectory_id"), "trajectory_id")
    turns = raw.get("turns")
    if not isinstance(turns, list) or len(turns) != configured_turns:
        raise ValueError(
            f"{trajectory_id} retained turn count differs from configuration"
        )
    retained_terminal_error = _required_number(
        raw.get("terminal_error"),
        "terminal_error",
    )
    terminal_error: float | None = None
    for source_turn_index, turn_value in enumerate(turns):
        turn = _required_mapping(turn_value, f"{trajectory_id} turn")
        observed_turn = _required_integer(
            turn.get("turn"),
            f"{trajectory_id} turn",
        )
        if observed_turn != source_turn_index:
            raise ValueError(f"{trajectory_id} turn indexes are not contiguous")
        if (
            "theta_snapshot" in turn
            and turn["theta_snapshot"] != raw.get("theta")
        ):
            raise ValueError(f"{trajectory_id} changes theta within trajectory")
        terminal_error = _marginal_brier(
            turn.get("belief_after"),
            raw.get("theta"),
            f"{trajectory_id} turn {source_turn_index}",
        )
        yield {
            "schema_version": 1,
            "source_record_index": source_record_index,
            "source_turn_index": source_turn_index,
            "trajectory_id": trajectory_id,
            "user_id": _required_text(raw.get("user_id"), "user_id"),
            "domain_id": _required_text(raw.get("domain_id"), "domain_id"),
            "crn_key": _required_text(raw.get("crn_key"), "crn_key"),
            "updater_id": _required_text(raw.get("updater_id"), "updater_id"),
            "policy_id": _required_text(raw.get("policy_id"), "policy_id"),
            "initial_profile_condition": _required_text(
                raw.get("initial_profile_condition"),
                "initial_profile_condition",
            ),
            "turn": source_turn_index + 1,
            "terminal_error": terminal_error,
            "retained_terminal_error": retained_terminal_error,
            "same_history_shadow": True,
        }
    assert terminal_error is not None
    tolerance = 1e-12 + 1e-9 * abs(retained_terminal_error)
    if abs(terminal_error - retained_terminal_error) > tolerance:
        raise ValueError(
            f"{trajectory_id} final turn error differs from terminal_error"
        )


def _experiment_c_row(
    raw: Mapping[str, Any],
    source_record_index: int,
    *,
    compact: bool,
) -> dict[str, Any]:
    if raw.get("schema_version") != 1:
        raise ValueError("unsupported Experiment C row schema_version")
    profile_error = _required_number(
        raw.get("profile_error"),
        "profile_error",
    )
    behavioral_accuracy = _required_number(
        raw.get("behavioral_accuracy"),
        "behavioral_accuracy",
    )
    cross_context_accuracy = _optional_number(
        raw.get("cross_context_accuracy"),
        "cross_context_accuracy",
    )
    intrinsic_regret = _required_number(
        raw.get("intrinsic_regret"),
        "intrinsic_regret",
    )
    if not compact:
        ranking_score = _required_mapping(
            raw.get("ranking_score"),
            "ranking_score",
        )
        retained_values = (
            (
                profile_error,
                _required_number(
                    ranking_score.get("profile_brier"),
                    "ranking_score.profile_brier",
                ),
            ),
            (
                behavioral_accuracy,
                _required_number(
                    ranking_score.get("behavioral_accuracy"),
                    "ranking_score.behavioral_accuracy",
                ),
            ),
            (
                intrinsic_regret,
                _required_number(
                    ranking_score.get("mean_intrinsic_regret"),
                    "ranking_score.mean_intrinsic_regret",
                ),
            ),
        )
        if any(
            abs(first - second) > 1e-12 + 1e-9 * abs(second)
            for first, second in retained_values
        ):
            raise ValueError(
                "Experiment C scalar scores differ from active ranking_score"
            )
        ranking_cross_context = _optional_number(
            ranking_score.get("cross_context_accuracy"),
            "ranking_score.cross_context_accuracy",
        )
        if (
            (cross_context_accuracy is None)
            != (ranking_cross_context is None)
            or (
                cross_context_accuracy is not None
                and ranking_cross_context is not None
                and abs(cross_context_accuracy - ranking_cross_context)
                > 1e-12 + 1e-9 * abs(ranking_cross_context)
            )
        ):
            raise ValueError(
                "Experiment C cross-context score differs from active "
                "ranking_score"
            )
    row = {
        "schema_version": 1,
        "source_record_index": source_record_index,
        "split": _required_text(raw.get("split"), "split"),
        "regime": _required_text(raw.get("regime"), "regime"),
        "replicate": _required_integer(raw.get("replicate"), "replicate"),
        "user_id": _required_text(raw.get("user_id"), "user_id"),
        "domain_id": _required_text(raw.get("domain_id"), "domain_id"),
        "updater_id": _required_text(raw.get("updater_id"), "updater_id"),
        "profile_error": profile_error,
        "behavioral_accuracy": behavioral_accuracy,
        "cross_context_accuracy": cross_context_accuracy,
        "intrinsic_regret": intrinsic_regret,
        "score_basis": _required_text(raw.get("score_basis"), "score_basis"),
        "history_digest": _required_text(
            raw.get("history_digest"),
            "history_digest",
        ),
        "battery_id": _required_text(raw.get("battery_id"), "battery_id"),
        "battery_digest": _required_text(
            raw.get("battery_digest"),
            "battery_digest",
        ),
    }
    if compact:
        retained_index = _required_integer(
            raw.get("source_record_index"),
            "source_record_index",
            minimum=1,
        )
        if retained_index != source_record_index:
            raise ValueError(
                "Experiment C compact source indexes are not contiguous"
            )
    return row


def _write_projected_rows(
    *,
    source_path: Path,
    destination: Path,
    experiment: str,
    compact_source: bool,
    configured_turns: int,
) -> tuple[int, int, str]:
    rows, hashing_reader, handle = _iter_hashed_jsonl(source_path)
    source_record_count = 0
    output_row_count = 0
    expected_b_record = 1
    expected_b_turn = 0
    try:
        with destination.open("w", encoding="utf-8", newline="\n") as output:
            for line_number, raw in rows:
                if experiment == "A":
                    source_record_count += 1
                    projected = (
                        _experiment_a_row(
                            raw,
                            source_record_count,
                            compact=compact_source,
                        ),
                    )
                elif experiment == "B" and compact_source:
                    source_record_index = _required_integer(
                        raw.get("source_record_index"),
                        "source_record_index",
                        minimum=1,
                    )
                    if source_record_index != expected_b_record:
                        if (
                            source_record_index == expected_b_record + 1
                            and expected_b_turn == configured_turns
                        ):
                            expected_b_record += 1
                            expected_b_turn = 0
                        else:
                            raise ValueError(
                                "Experiment B compact records are not contiguous"
                            )
                    projected = (
                        _experiment_b_compact_row(
                            raw,
                            expected_record_index=expected_b_record,
                            expected_turn_index=expected_b_turn,
                        ),
                    )
                    expected_b_turn += 1
                    if expected_b_turn > configured_turns:
                        raise ValueError(
                            "Experiment B compact trajectory has too many turns"
                        )
                    source_record_count = expected_b_record
                elif experiment == "B":
                    source_record_count += 1
                    projected = _experiment_b_rows(
                        raw,
                        source_record_count,
                        configured_turns,
                    )
                else:
                    source_record_count += 1
                    projected = (
                        _experiment_c_row(
                            raw,
                            source_record_count,
                            compact=compact_source,
                        ),
                    )
                for output_row in projected:
                    output.write(canonical_json(output_row) + "\n")
                    output_row_count += 1
            if (
                experiment == "B"
                and compact_source
                and expected_b_turn != configured_turns
            ):
                raise ValueError(
                    "Experiment B final compact trajectory has incomplete turns"
                )
    finally:
        handle.close()
    if source_record_count == 0 or output_row_count == 0:
        raise ValueError("analysis source contains no rows")
    return source_record_count, output_row_count, hashing_reader.hexdigest


def _ensure_external_destination(source: Path, destination: Path) -> Path:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"compact analysis output already exists: {destination}"
        )
    parent = destination.absolute().parent
    parent.mkdir(parents=True, exist_ok=True)
    resolved_parent = parent.resolve()
    if resolved_parent.is_symlink() or not resolved_parent.is_dir():
        raise ValueError("compact analysis output parent must be a directory")
    resolved_destination = resolved_parent / destination.name
    if (
        resolved_destination == source
        or source in resolved_destination.parents
    ):
        raise ValueError(
            "compact analysis output must remain outside the immutable source run"
        )
    return resolved_destination


def export_compact_analysis(
    run_dir: str | Path,
    output_dir: str | Path,
) -> CompactAnalysisBundle:
    """Export a compact analysis bundle without modifying ``run_dir``."""

    source = Path(run_dir).resolve()
    valid, errors = verify_run(source)
    if not valid:
        raise ValueError(
            "cannot export an unverified source run: " + "; ".join(errors)
        )
    destination = _ensure_external_destination(source, Path(output_dir))
    manifest_path = source / "manifest.json"
    config_path = source / "config.resolved.json"
    summary_path = source / "metrics" / "summary.json"
    checksums_path = source / "SHA256SUMS"
    source_manifest_bytes = read_control_bytes(
        manifest_path,
        label="source manifest.json",
    )
    source_config_bytes = read_control_bytes(
        config_path,
        label="source config.resolved.json",
    )
    source_summary_bytes = read_control_bytes(
        summary_path,
        label="source metrics/summary.json",
    )
    source_checksums_bytes = read_control_bytes(
        checksums_path,
        label="source SHA256SUMS",
    )
    source_manifest = _read_json_object(
        manifest_path,
        "source manifest.json",
    )
    source_config = _read_json_object(
        config_path,
        "source config.resolved.json",
    )
    source_summary = _read_json_object(
        summary_path,
        "source metrics/summary.json",
    )
    experiment_kind = _required_text(
        _required_mapping(
            source_config.get("experiment"),
            "source experiment config",
        ).get("kind"),
        "source experiment kind",
    )
    experiment = _EXPERIMENT_BY_KIND.get(experiment_kind)
    if experiment is None:
        raise ValueError(
            "compact analysis export supports Experiments A, B, and C only"
        )
    expected_summary_experiment = source_summary.get("experiment")
    if expected_summary_experiment != experiment:
        raise ValueError(
            "source summary experiment differs from resolved configuration"
        )
    source_checksums = _parse_checksums(checksums_path)
    internal_relative = _INTERNAL_INPUTS[experiment]
    compact_source = internal_relative in source_checksums
    input_relative = (
        internal_relative if compact_source else _LEGACY_INPUTS[experiment]
    )
    if input_relative not in source_checksums:
        raise ValueError(
            f"verified source run lacks analysis input {input_relative}"
        )
    input_path = source / input_relative
    configured_turns = _required_integer(
        _required_mapping(
            source_config.get("experiment"),
            "source experiment config",
        ).get("turns"),
        "source configured turns",
        minimum=1,
    )
    temporary: Path | None = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
        )
    )
    try:
        rows_path = temporary / "analysis-rows.jsonl"
        (
            source_record_count,
            row_count,
            observed_input_sha256,
        ) = _write_projected_rows(
            source_path=input_path,
            destination=rows_path,
            experiment=experiment,
            compact_source=compact_source,
            configured_turns=configured_turns,
        )
        if observed_input_sha256 != source_checksums[input_relative]:
            raise ValueError(
                f"analysis input changed while exporting: {input_relative}"
            )
        expected_source_count = {
            "A": source_summary.get("row_count"),
            "B": source_summary.get("trajectories"),
            "C": source_summary.get("evaluation_rows"),
        }[experiment]
        if (
            compact_source
            and experiment == "B"
            and source_record_count != expected_source_count
        ):
            raise ValueError(
                "Experiment B compact trajectory count differs from summary"
            )
        if experiment != "B" and source_record_count != expected_source_count:
            raise ValueError(
                f"Experiment {experiment} row count differs from summary"
            )
        if (
            experiment == "B"
            and row_count != int(expected_source_count) * configured_turns
        ):
            raise ValueError(
                "Experiment B compact turn count differs from summary/config"
            )
        rows_sha256 = file_sha256(rows_path)
        source_run_id = _required_text(
            source_manifest.get("run_id"),
            "source run_id",
        )
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "artifact_kind": _BUNDLE_KIND,
            "status": "complete",
            "claim_status": "not_claimed",
            "experiment": experiment,
            "analysis_unit": _ANALYSIS_UNITS[experiment],
            "row_schema_version": 1,
            "row_count": row_count,
            "source_record_count": source_record_count,
            "configured_turns": (
                configured_turns if experiment == "B" else None
            ),
            "analysis_rows_file": "analysis-rows.jsonl",
            "analysis_rows_sha256": rows_sha256,
            "source_run_id": source_run_id,
            "source_manifest_sha256": sha256(
                source_manifest_bytes
            ).hexdigest(),
            "source_checksums_sha256": sha256(
                source_checksums_bytes
            ).hexdigest(),
            "source_config_file_sha256": source_checksums[
                "config.resolved.json"
            ],
            "source_summary_file_sha256": source_checksums[
                "metrics/summary.json"
            ],
            "source_config_sha256": source_manifest.get("config_sha256"),
            "source_tree_sha256": source_manifest.get("source_sha256"),
            "source_input_file": input_relative,
            "source_input_sha256": source_checksums[input_relative],
            "source_input_is_runner_compact": compact_source,
            "exporter_version": __version__,
            "exporter_source_sha256": file_sha256(
                Path(__file__).resolve()
            ),
            "outcome_derivation": _OUTCOME_DERIVATIONS[experiment],
        }
        if experiment == "A":
            exclusion_relative = next(
                (
                    relative
                    for relative in (
                        "analysis/experiment-a-exclusions.jsonl",
                        "events/experiment-a-exclusions.jsonl",
                    )
                    if relative in source_checksums
                ),
                None,
            )
            if exclusion_relative is not None:
                manifest["source_exclusion_file"] = exclusion_relative
                manifest["source_exclusion_sha256"] = source_checksums[
                    exclusion_relative
                ]
        (temporary / "manifest.json").write_text(
            canonical_json(manifest) + "\n",
            encoding="utf-8",
        )
        checksum_lines = [
            f"{file_sha256(temporary / relative)}  {relative}"
            for relative in ("analysis-rows.jsonl", "manifest.json")
        ]
        (temporary / "SHA256SUMS").write_text(
            "\n".join(checksum_lines) + "\n",
            encoding="utf-8",
        )
        if (
            read_control_bytes(manifest_path, label="source manifest.json")
            != source_manifest_bytes
            or read_control_bytes(
                config_path,
                label="source config.resolved.json",
            )
            != source_config_bytes
            or read_control_bytes(
                summary_path,
                label="source metrics/summary.json",
            )
            != source_summary_bytes
            or read_control_bytes(
                checksums_path,
                label="source SHA256SUMS",
            )
            != source_checksums_bytes
        ):
            raise ValueError("source run controls changed while exporting")
        valid_bundle, bundle_errors = verify_compact_analysis(temporary)
        if not valid_bundle:
            raise ValueError(
                "generated compact bundle is invalid: "
                + "; ".join(bundle_errors)
            )
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(
                f"compact analysis output already exists: {destination}"
            )
        os.rename(temporary, destination)
        temporary = None
        return CompactAnalysisBundle(
            path=destination,
            experiment=experiment,
            source_run_id=source_run_id,
            row_count=row_count,
            analysis_rows_sha256=rows_sha256,
        )
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)


def verify_compact_analysis(
    bundle_dir: str | Path,
) -> tuple[bool, tuple[str, ...]]:
    """Verify a compact bundle's exact inventory, checksums, and row count."""

    bundle = Path(bundle_dir).resolve()
    errors: list[str] = []
    if not bundle.is_dir() or bundle.is_symlink():
        return False, ("compact bundle must be a regular directory",)
    entries = tuple(bundle.rglob("*"))
    for entry in entries:
        if entry.is_symlink():
            errors.append(
                "symbolic link not allowed: "
                + entry.relative_to(bundle).as_posix()
            )
    files = {
        entry.relative_to(bundle).as_posix()
        for entry in entries
        if entry.is_file() and not entry.is_symlink()
    }
    if files != _BUNDLE_FILES:
        errors.append(
            "compact bundle inventory must be exactly "
            + ", ".join(sorted(_BUNDLE_FILES))
        )
        return False, tuple(errors)
    try:
        checksums = _parse_checksums(bundle / "SHA256SUMS")
    except ValueError as exc:
        errors.append(str(exc))
        return False, tuple(errors)
    if set(checksums) != _BUNDLE_FILES - {"SHA256SUMS"}:
        errors.append("compact SHA256SUMS has an unexpected inventory")
    for relative, expected in checksums.items():
        if file_sha256(bundle / relative) != expected:
            errors.append(f"checksum mismatch: {relative}")
    try:
        manifest = _read_json_object(
            bundle / "manifest.json",
            "compact manifest.json",
        )
    except ValueError as exc:
        errors.append(str(exc))
        return False, tuple(errors)
    if manifest.get("schema_version") != 1:
        errors.append("unsupported compact manifest schema_version")
    if manifest.get("artifact_kind") != _BUNDLE_KIND:
        errors.append("unexpected compact artifact_kind")
    if manifest.get("status") != "complete":
        errors.append("compact manifest status is not complete")
    if manifest.get("claim_status") != "not_claimed":
        errors.append("compact claim_status must be not_claimed")
    experiment = manifest.get("experiment")
    if experiment not in {"A", "B", "C"}:
        errors.append("compact experiment must be A, B, or C")
    expected_manifest_fields = set(_BASE_MANIFEST_FIELDS)
    if (
        "source_exclusion_file" in manifest
        or "source_exclusion_sha256" in manifest
    ):
        expected_manifest_fields.update(
            {"source_exclusion_file", "source_exclusion_sha256"}
        )
    if set(manifest) != expected_manifest_fields:
        errors.append("compact manifest has an unexpected field set")
    for field in (
        "analysis_rows_sha256",
        "source_manifest_sha256",
        "source_checksums_sha256",
        "source_config_file_sha256",
        "source_summary_file_sha256",
        "source_config_sha256",
        "source_tree_sha256",
        "source_input_sha256",
        "exporter_source_sha256",
    ):
        if not _is_sha256(manifest.get(field)):
            errors.append(f"compact manifest {field} is not a SHA-256")
    if manifest.get("analysis_rows_sha256") != file_sha256(
        bundle / "analysis-rows.jsonl"
    ):
        errors.append("analysis_rows_sha256 does not match analysis-rows.jsonl")
    if manifest.get("analysis_rows_file") != "analysis-rows.jsonl":
        errors.append("compact analysis_rows_file is not canonical")
    if manifest.get("row_schema_version") != 1:
        errors.append("unsupported compact row_schema_version")
    if not isinstance(manifest.get("source_run_id"), str) or not manifest[
        "source_run_id"
    ]:
        errors.append("compact source_run_id must be a non-empty string")
    if not isinstance(manifest.get("exporter_version"), str) or not manifest[
        "exporter_version"
    ]:
        errors.append("compact exporter_version must be a non-empty string")
    if experiment in {"A", "B", "C"}:
        if manifest.get("analysis_unit") != _ANALYSIS_UNITS[experiment]:
            errors.append("compact analysis_unit differs from experiment")
        if (
            manifest.get("outcome_derivation")
            != _OUTCOME_DERIVATIONS[experiment]
        ):
            errors.append("compact outcome_derivation differs from experiment")
        compact_source = manifest.get("source_input_is_runner_compact")
        if not isinstance(compact_source, bool):
            errors.append(
                "compact source_input_is_runner_compact must be a boolean"
            )
        else:
            expected_input = (
                _INTERNAL_INPUTS[experiment]
                if compact_source
                else _LEGACY_INPUTS[experiment]
            )
            if manifest.get("source_input_file") != expected_input:
                errors.append(
                    "compact source_input_file differs from its source role"
                )
    else:
        compact_source = False
    exclusion_file = manifest.get("source_exclusion_file")
    exclusion_sha256 = manifest.get("source_exclusion_sha256")
    if exclusion_file is not None or exclusion_sha256 is not None:
        if experiment != "A" or exclusion_file not in {
            "analysis/experiment-a-exclusions.jsonl",
            "events/experiment-a-exclusions.jsonl",
        }:
            errors.append("compact source exclusion binding is invalid")
        if not _is_sha256(exclusion_sha256):
            errors.append(
                "compact source_exclusion_sha256 is not a SHA-256"
            )
    try:
        expected_rows = _required_integer(
            manifest.get("row_count"),
            "compact row_count",
            minimum=1,
        )
    except ValueError as exc:
        errors.append(str(exc))
        expected_rows = -1
    try:
        source_record_count = _required_integer(
            manifest.get("source_record_count"),
            "compact source_record_count",
            minimum=1,
        )
    except ValueError as exc:
        errors.append(str(exc))
        source_record_count = -1
    if experiment == "B":
        try:
            configured_turns = _required_integer(
                manifest.get("configured_turns"),
                "compact configured_turns",
                minimum=1,
            )
        except ValueError as exc:
            errors.append(str(exc))
            configured_turns = -1
        if expected_rows != source_record_count * configured_turns:
            errors.append(
                "compact Experiment B row count is not records * turns"
            )
    else:
        configured_turns = 1
        if manifest.get("configured_turns") is not None:
            errors.append("configured_turns must be null outside Experiment B")
        if expected_rows != source_record_count:
            errors.append(
                "compact row count differs from source_record_count"
            )
    observed_rows = 0
    observed_a_keys: set[tuple[str, str]] = set()
    observed_b_trajectory_ids: dict[int, str] = {}
    observed_b_invariants: dict[int, tuple[Any, ...]] = {}
    observed_c_keys: set[tuple[Any, ...]] = set()
    try:
        with (bundle / "analysis-rows.jsonl").open("rb") as handle:
            for observed_rows, raw_line in enumerate(handle, start=1):
                row = _decode_jsonl_line(
                    raw_line,
                    path=bundle / "analysis-rows.jsonl",
                    line_number=observed_rows,
                )
                if experiment == "A":
                    if set(row) != _A_FIELDS:
                        raise ValueError(
                            "compact Experiment A row has an invalid field set"
                        )
                    projected = _experiment_a_row(
                        row,
                        observed_rows,
                        compact=True,
                    )
                    if projected["response_mode"] not in {
                        "controlled_anchor",
                        "naturally_sampled",
                    }:
                        raise ValueError(
                            "compact Experiment A response_mode is invalid"
                        )
                    key = (
                        projected["trial_id"],
                        projected["updater_id"],
                    )
                    if key in observed_a_keys:
                        raise ValueError(
                            "compact Experiment A has a duplicate scientific key"
                        )
                    observed_a_keys.add(key)
                elif experiment == "B":
                    if set(row) != _B_FIELDS:
                        raise ValueError(
                            "compact Experiment B row has an invalid field set"
                        )
                    expected_record = (
                        (observed_rows - 1) // configured_turns + 1
                    )
                    expected_turn = (observed_rows - 1) % configured_turns
                    projected = _experiment_b_compact_row(
                        row,
                        expected_record_index=expected_record,
                        expected_turn_index=expected_turn,
                    )
                    trajectory_id = projected["trajectory_id"]
                    prior_id = observed_b_trajectory_ids.get(expected_record)
                    if prior_id is None:
                        if trajectory_id in observed_b_trajectory_ids.values():
                            raise ValueError(
                                "compact Experiment B trajectory_id is reused"
                            )
                        observed_b_trajectory_ids[
                            expected_record
                        ] = trajectory_id
                        observed_b_invariants[expected_record] = tuple(
                            projected[field]
                            for field in (
                                "user_id",
                                "domain_id",
                                "crn_key",
                                "updater_id",
                                "policy_id",
                                "initial_profile_condition",
                                "retained_terminal_error",
                                "same_history_shadow",
                            )
                        )
                    elif (
                        prior_id != trajectory_id
                        or observed_b_invariants[expected_record]
                        != tuple(
                            projected[field]
                            for field in (
                                "user_id",
                                "domain_id",
                                "crn_key",
                                "updater_id",
                                "policy_id",
                                "initial_profile_condition",
                                "retained_terminal_error",
                                "same_history_shadow",
                            )
                        )
                    ):
                        raise ValueError(
                            "compact Experiment B trajectory fields change"
                        )
                    if expected_turn == configured_turns - 1:
                        retained = projected["retained_terminal_error"]
                        tolerance = 1e-12 + 1e-9 * abs(retained)
                        if (
                            abs(projected["terminal_error"] - retained)
                            > tolerance
                        ):
                            raise ValueError(
                                "compact Experiment B final error is inconsistent"
                            )
                elif experiment == "C":
                    if set(row) != _C_FIELDS:
                        raise ValueError(
                            "compact Experiment C row has an invalid field set"
                        )
                    projected = _experiment_c_row(
                        row,
                        observed_rows,
                        compact=True,
                    )
                    key = tuple(
                        projected[field]
                        for field in (
                            "split",
                            "regime",
                            "replicate",
                            "user_id",
                            "domain_id",
                            "updater_id",
                        )
                    )
                    if key in observed_c_keys:
                        raise ValueError(
                            "compact Experiment C has a duplicate scientific key"
                        )
                    observed_c_keys.add(key)
                elif row.get("schema_version") != 1:
                    raise ValueError(
                        "compact analysis row has unsupported schema_version"
                    )
    except ValueError as exc:
        errors.append(str(exc))
    if observed_rows != expected_rows:
        errors.append(
            "compact analysis row count differs from manifest.row_count"
        )
    return not errors, tuple(errors)
