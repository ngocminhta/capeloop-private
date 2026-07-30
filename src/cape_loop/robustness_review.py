"""Immutable, offline cross-run review for Gate 6 robustness evidence.

One sensitivity run intentionally cannot establish replication across model
families or transfer to Experiment A's held-out surface paraphrases.  This
module joins those two evidence streams without mutating either source run and
without making provider calls.

Family and provider-source identities are caller declarations.  The importer
binds them to exact requested/returned model identities and retained provider
evidence, but it neither infers families from display labels nor claims that
two provider outputs are statistically independent.
"""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence
import json
import os
import re
import shutil
import tempfile

from .artifacts import canonical_json, file_sha256, verify_run
from .beliefs import MarginalPreferenceBelief, PreferenceBelief
from .config import AppConfig
from .gates import GateCriterion, GateReport
from .heldout import (
    HeldOutParaphraseCase,
    ParaphraseEvaluationRecord,
    build_default_paraphrase_suite,
    evaluate_gate1_paraphrase_transfer,
)
from .llm_exchange import ATTRIBUTES, VALUES, LLMResponse, ReplayProvider, read_responses
from .openai_provider import read_requests
from .provider_attempts import DurableProviderAttemptLedger
from .sensitivity import (
    sensitivity_breadth_coverage,
    sensitivity_grid,
)


DECLARATION_KIND = "gate6-cross-run-declaration"
REVIEW_KIND = "gate6-cross-run-review"
FULL_CONTEXT_UPDATER = "llm_full_context"
CRITERION_IDS = (
    "another-response-model",
    "broad-simulator-parameters",
    "both-domains",
    "multiple-llm-families",
    "natural-language-paraphrases",
    "exact-and-fitted-action-aware-references",
)
REVIEW_FILES = frozenset(
    {
        "declaration.json",
        "evidence/pairs.jsonl",
        "metrics/gate-6.json",
        "review.json",
        "manifest.json",
    }
)
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_UTC = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$"
)
_PROVIDER_FIELDS = frozenset(
    {
        "mode",
        "responses_file",
        "model_role",
        "model",
        "reasoning_effort",
        "api_key_env",
        "base_url",
        "allow_custom_base_url",
        "timeout_seconds",
        "max_retries",
        "max_output_tokens",
        "max_requests",
        "max_total_tokens",
        "journal_dir",
        "openrouter_upstream_provider",
        "openrouter_allow_fallbacks",
        "openrouter_require_parameters",
        "openrouter_data_collection",
        "openrouter_zdr",
        "openrouter_http_referer",
        "openrouter_app_title",
    }
)


def _digest(value: Any) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _file_digest(path: Path) -> str:
    return file_sha256(path)


def _require_text(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or "\r" in value
        or "\n" in value
    ):
        raise ValueError(f"{name} must be a nonempty single-line string")
    return value


def _require_digest(value: object, name: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _strict_fields(
    raw: Mapping[str, Any],
    expected: set[str] | frozenset[str],
    *,
    name: str,
) -> None:
    actual = set(raw)
    if actual != set(expected):
        raise ValueError(
            f"{name} has missing or unknown fields: "
            + canonical_json(
                {
                    "missing": sorted(set(expected) - actual),
                    "unknown": sorted(actual - set(expected)),
                }
            )
        )


def _assert_no_symlink_components(path: Path, *, name: str) -> None:
    # macOS exposes ordinary temporary paths through the system-level
    # ``/var -> /private/var`` alias. Resolve ancestors, but never accept a
    # caller-controlled symlink as the declared file/directory itself.
    if path.is_symlink():
        raise ValueError(f"{name} cannot be a symlink: {path}")


def _assert_safe_tree(path: Path, *, name: str) -> Path:
    _assert_no_symlink_components(path, name=name)
    resolved = path.resolve()
    if not resolved.is_dir():
        raise ValueError(f"{name} must be an existing directory")
    for item in resolved.rglob("*"):
        if item.is_symlink():
            raise ValueError(f"{name} contains a symlink: {item}")
    return resolved


def _safe_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{path}: expected a safe regular JSON file")
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(decoded, Mapping):
        raise ValueError(f"{path}: expected a JSON object")
    return dict(decoded)


def _safe_jsonl(path: Path, *, allow_empty: bool = False) -> tuple[dict[str, Any], ...]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{path}: expected a safe regular JSONL file")
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"{path}: invalid JSONL input: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            decoded = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
        if not isinstance(decoded, Mapping):
            raise ValueError(f"{path}:{line_number}: record must be an object")
        rows.append(dict(decoded))
    if not rows and not allow_empty:
        raise ValueError(f"{path}: JSONL input cannot be empty")
    return tuple(rows)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(canonical_json(dict(row)) + "\n" for row in rows),
        encoding="utf-8",
    )


def _file_bindings(root: Path, names: Sequence[str]) -> dict[str, dict[str, Any]]:
    bindings: dict[str, dict[str, Any]] = {}
    for name in names:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"required source artifact is missing or unsafe: {name}")
        material = path.read_bytes()
        bindings[name] = {
            "sha256": sha256(material).hexdigest(),
            "bytes": len(material),
        }
    return bindings


def _source_run_binding(
    root: Path,
    *,
    expected_kind: str,
    expected_reference: Mapping[str, Any],
    required_files: Sequence[str],
) -> tuple[dict[str, Any], AppConfig, dict[str, Any], dict[str, Any]]:
    valid, errors = verify_run(root)
    if not valid:
        raise ValueError(
            f"{root}: source run verification failed: " + "; ".join(errors)
        )
    manifest = _safe_json(root / "manifest.json")
    config_raw = _safe_json(root / "config.resolved.json")
    config = AppConfig.parse(config_raw).validated()
    summary = _safe_json(root / "metrics/summary.json")
    if manifest.get("status") != "complete":
        raise ValueError(f"{root}: source manifest is not complete")
    if config.experiment.kind != expected_kind:
        raise ValueError(
            f"{root}: expected experiment kind {expected_kind!r}, "
            f"found {config.experiment.kind!r}"
        )
    expected_experiment = (
        "sensitivity" if expected_kind == "sensitivity" else "A"
    )
    if summary.get("experiment") != expected_experiment:
        raise ValueError(f"{root}: summary experiment identity is inconsistent")
    if summary.get("scientific_claim_status") != "not_claimed":
        raise ValueError(f"{root}: source run has invalid claim semantics")
    if manifest.get("run_id") != expected_reference["run_id"]:
        raise ValueError(f"{root}: declaration run_id does not match source")
    checksum_digest = _file_digest(root / "SHA256SUMS")
    if checksum_digest != expected_reference["sha256sums_sha256"]:
        raise ValueError(
            f"{root}: declaration SHA256SUMS digest does not match source"
        )
    files = _file_bindings(root, required_files)
    binding = {
        "resolved_path": str(root),
        "run_id": manifest["run_id"],
        "run_manifest_sha256": _file_digest(root / "manifest.json"),
        "run_checksum_manifest_sha256": checksum_digest,
        "config_sha256": manifest["config_sha256"],
        "source_sha256": manifest["source_sha256"],
        "required_files": files,
        "verified_complete": True,
    }
    return binding, config, summary, config_raw


def _normalize_source_reference(
    raw: Mapping[str, Any],
    *,
    name: str,
    base: Path,
    require_source_path: bool,
) -> dict[str, Any]:
    _strict_fields(
        raw,
        {"path", "run_id", "sha256sums_sha256"},
        name=name,
    )
    source_path = Path(_require_text(raw.get("path"), f"{name}.path"))
    if not source_path.is_absolute():
        source_path = base / source_path
    resolved = (
        _assert_safe_tree(source_path, name=name)
        if require_source_path
        else source_path.resolve(strict=False)
    )
    return {
        "path": str(resolved),
        "run_id": _require_text(raw.get("run_id"), f"{name}.run_id"),
        "sha256sums_sha256": _require_digest(
            raw.get("sha256sums_sha256"),
            f"{name}.sha256sums_sha256",
        ),
    }


def _validate_model_binding(
    raw: Mapping[str, Any],
    *,
    name: str,
) -> dict[str, Any]:
    fields = {
        "provider_id",
        "provider_source_id",
        "requested_model_id",
        "response_model_id",
        "upstream_provider_id",
        "upstream_model_id",
    }
    _strict_fields(raw, fields, name=name)
    provider = _require_text(raw.get("provider_id"), f"{name}.provider_id")
    if provider not in {"openai", "openrouter"}:
        raise ValueError(f"{name}.provider_id must be openai or openrouter")
    upstream_provider = raw.get("upstream_provider_id")
    upstream_model = raw.get("upstream_model_id")
    for value, field in (
        (upstream_provider, "upstream_provider_id"),
        (upstream_model, "upstream_model_id"),
    ):
        if value is not None:
            _require_text(value, f"{name}.{field}")
    if provider == "openai" and (
        upstream_provider is not None or upstream_model is not None
    ):
        raise ValueError(
            f"{name}: first-party OpenAI bindings must use null upstream fields"
        )
    if provider == "openrouter" and (
        upstream_provider is None or upstream_model is None
    ):
        raise ValueError(
            f"{name}: OpenRouter bindings require exact returned upstream labels"
        )
    return {
        "provider_id": provider,
        "provider_source_id": _require_text(
            raw.get("provider_source_id"),
            f"{name}.provider_source_id",
        ),
        "requested_model_id": _require_text(
            raw.get("requested_model_id"),
            f"{name}.requested_model_id",
        ),
        "response_model_id": _require_text(
            raw.get("response_model_id"),
            f"{name}.response_model_id",
        ),
        "upstream_provider_id": upstream_provider,
        "upstream_model_id": upstream_model,
    }


def read_gate6_cross_run_declaration(
    path: str | Path,
    *,
    require_source_paths: bool = True,
) -> dict[str, Any]:
    """Read, strictly validate, and normalize one external review declaration."""

    source = Path(path)
    _assert_no_symlink_components(source, name="declaration")
    if source.is_symlink() or not source.is_file():
        raise ValueError("declaration must be a safe regular JSON file")
    raw = _safe_json(source)
    _strict_fields(
        raw,
        {
            "schema_version",
            "artifact_kind",
            "declaration_id",
            "review_authority",
            "statistical_independence_claimed",
            "pairs",
        },
        name="declaration",
    )
    if raw.get("schema_version") != 1 or raw.get("artifact_kind") != DECLARATION_KIND:
        raise ValueError("unsupported Gate 6 declaration schema")
    if raw.get("statistical_independence_claimed") is not False:
        raise ValueError(
            "the declaration must set statistical_independence_claimed=false"
        )
    authority = raw.get("review_authority")
    if not isinstance(authority, Mapping):
        raise ValueError("review_authority must be an object")
    _strict_fields(
        authority,
        {
            "responsible_researcher_id",
            "reviewed_at_utc",
            "preregistration_reference",
            "family_assignments_declared_before_outcome_review",
            "source_identities_reviewed",
        },
        name="review_authority",
    )
    reviewed_at = _require_text(
        authority.get("reviewed_at_utc"),
        "review_authority.reviewed_at_utc",
    )
    if _UTC.fullmatch(reviewed_at) is None:
        raise ValueError("reviewed_at_utc must be an ISO-8601 UTC timestamp")
    for field in (
        "family_assignments_declared_before_outcome_review",
        "source_identities_reviewed",
    ):
        if not isinstance(authority.get(field), bool):
            raise ValueError(f"review_authority.{field} must be Boolean")
    if authority.get("source_identities_reviewed") is not True:
        raise ValueError(
            "source_identities_reviewed must be true before importing evidence"
        )
    pair_rows = raw.get("pairs")
    if (
        not isinstance(pair_rows, Sequence)
        or isinstance(pair_rows, (str, bytes))
        or len(pair_rows) < 2
    ):
        raise ValueError("pairs must contain at least two explicit run pairs")
    normalized_pairs: list[dict[str, Any]] = []
    base = source.resolve().parent
    for index, item in enumerate(pair_rows):
        if not isinstance(item, Mapping):
            raise ValueError(f"pairs[{index}] must be an object")
        _strict_fields(
            item,
            {
                "pair_id",
                "family_id",
                "sensitivity_run",
                "experiment_a_run",
                "model_binding",
            },
            name=f"pairs[{index}]",
        )
        sensitivity = item.get("sensitivity_run")
        experiment_a = item.get("experiment_a_run")
        model = item.get("model_binding")
        if not all(isinstance(value, Mapping) for value in (sensitivity, experiment_a, model)):
            raise ValueError(f"pairs[{index}] contains a non-object binding")
        normalized_pairs.append(
            {
                "pair_id": _require_text(
                    item.get("pair_id"),
                    f"pairs[{index}].pair_id",
                ),
                "family_id": _require_text(
                    item.get("family_id"),
                    f"pairs[{index}].family_id",
                ),
                "sensitivity_run": _normalize_source_reference(
                    sensitivity,
                    name=f"pairs[{index}].sensitivity_run",
                    base=base,
                    require_source_path=require_source_paths,
                ),
                "experiment_a_run": _normalize_source_reference(
                    experiment_a,
                    name=f"pairs[{index}].experiment_a_run",
                    base=base,
                    require_source_path=require_source_paths,
                ),
                "model_binding": _validate_model_binding(
                    model,
                    name=f"pairs[{index}].model_binding",
                ),
            }
        )
    for field in ("pair_id", "family_id"):
        values = [row[field] for row in normalized_pairs]
        if len(values) != len(set(values)):
            raise ValueError(f"declaration {field} values must be unique")
    sensitivity_paths = [
        row["sensitivity_run"]["path"] for row in normalized_pairs
    ]
    experiment_a_paths = [
        row["experiment_a_run"]["path"] for row in normalized_pairs
    ]
    if len(set(sensitivity_paths)) != len(sensitivity_paths):
        raise ValueError("each family must use a distinct sensitivity run")
    if len(set(experiment_a_paths)) != len(experiment_a_paths):
        raise ValueError("each family must use a distinct Experiment A run")
    response_models = [
        row["model_binding"]["response_model_id"]
        for row in normalized_pairs
    ]
    if len(set(response_models)) != len(response_models):
        raise ValueError(
            "distinct family IDs must bind distinct actual response model IDs"
        )
    model_bindings = [
        canonical_json(row["model_binding"]) for row in normalized_pairs
    ]
    if len(set(model_bindings)) != len(model_bindings):
        raise ValueError("declared families cannot reuse an exact model/source binding")
    return {
        "schema_version": 1,
        "artifact_kind": DECLARATION_KIND,
        "declaration_id": _require_text(
            raw.get("declaration_id"),
            "declaration_id",
        ),
        "review_authority": {
            "responsible_researcher_id": _require_text(
                authority.get("responsible_researcher_id"),
                "review_authority.responsible_researcher_id",
            ),
            "reviewed_at_utc": reviewed_at,
            "preregistration_reference": _require_text(
                authority.get("preregistration_reference"),
                "review_authority.preregistration_reference",
            ),
            "family_assignments_declared_before_outcome_review": authority[
                "family_assignments_declared_before_outcome_review"
            ],
            "source_identities_reviewed": True,
        },
        "statistical_independence_claimed": False,
        "pairs": normalized_pairs,
    }


def _scientific_sensitivity_config(config_raw: Mapping[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(dict(config_raw))
    run = normalized.get("run")
    llm = normalized.get("llm")
    if not isinstance(run, Mapping) or not isinstance(llm, Mapping):
        raise ValueError("resolved configuration lacks run or llm sections")
    normalized["run"] = {
        key: value
        for key, value in run.items()
        if key not in {"name", "seed", "output_root"}
    }
    normalized["llm"] = {
        key: value for key, value in llm.items() if key not in _PROVIDER_FIELDS
    }
    return normalized


def _validate_exchange_manifest(
    manifest: Mapping[str, Any],
    *,
    requests: Sequence[Any],
    responses: Sequence[LLMResponse],
    config: AppConfig,
) -> None:
    _strict_fields(
        manifest,
        {
            "schema_version",
            "prompts_retained",
            "requests",
            "models",
            "execution_mode",
            "probability_calibration",
        },
        name="llm/exchange-manifest.json",
    )
    if (
        manifest.get("schema_version") != 1
        or manifest.get("prompts_retained") is not True
        or manifest.get("execution_mode") != config.llm.mode
        or manifest.get("probability_calibration") != config.llm.calibration
    ):
        raise ValueError("LLM exchange manifest semantics are inconsistent")
    expected_requests = [
        {
            "request_id": request.request_id,
            "updater_id": request.updater_id,
            "view": request.view,
            "prompt_sha256": request.prompt_sha256,
        }
        for request in requests
    ]
    if manifest.get("requests") != expected_requests:
        raise ValueError("LLM exchange manifest request bindings differ")
    if manifest.get("models") != sorted({item.model_id for item in responses}):
        raise ValueError("LLM exchange manifest model identities differ")


def _provider_evidence(
    root: Path,
    *,
    config: AppConfig,
    declared: Mapping[str, Any],
) -> tuple[dict[str, Any], tuple[Any, ...], tuple[LLMResponse, ...]]:
    if config.llm.mode not in {"openai", "openrouter"}:
        raise ValueError(
            f"{root}: cross-run review requires retained live provider evidence"
        )
    if config.llm.mode != declared["provider_id"]:
        raise ValueError(f"{root}: provider differs from declaration")
    required = (
        "llm/requests.jsonl",
        "llm/responses.jsonl",
        "llm/exchange-manifest.json",
        "llm/provider-manifest.json",
        "llm/provider-audit.jsonl",
        "llm/transport-attempts.jsonl",
        "models/llm-calibration.json",
    )
    _file_bindings(root, required)
    requests = read_requests(root / "llm/requests.jsonl")
    responses = read_responses(root / "llm/responses.jsonl")
    if not requests or not responses:
        raise ValueError(f"{root}: LLM exchange cannot be empty")
    ReplayProvider(responses).validate_coverage(requests)
    exchange = _safe_json(root / "llm/exchange-manifest.json")
    _validate_exchange_manifest(
        exchange,
        requests=requests,
        responses=responses,
        config=config,
    )
    calibration = _safe_json(root / "models/llm-calibration.json")
    if (
        calibration.get("schema_version") != 1
        or calibration.get("test_labels_used") is not False
    ):
        raise ValueError(f"{root}: invalid LLM calibration manifest")
    if config.llm.calibration == "none":
        if calibration.get("kind") != "none":
            raise ValueError(f"{root}: LLM calibration kind mismatch")
        raw_responses = responses
    else:
        if (
            calibration.get("kind") != "per-updater-temperature"
            or calibration.get("fitted_split") != "development"
        ):
            raise ValueError(f"{root}: LLM calibration is not development-only")
        raw_path = root / "llm/test-raw-responses.jsonl"
        if raw_path.is_symlink() or not raw_path.is_file():
            raise ValueError(f"{root}: calibrated run lacks test raw responses")
        raw_responses = read_responses(raw_path)
        if {item.request_id for item in raw_responses} != {
            item.request_id for item in responses
        }:
            raise ValueError(f"{root}: raw/calibrated response coverage differs")
        active_by_id = {item.request_id: item for item in responses}
        for raw in raw_responses:
            active = active_by_id[raw.request_id]
            if (
                raw.prompt_sha256 != active.prompt_sha256
                or raw.model_id != active.model_id
                or raw.raw_response_sha256 != active.raw_response_sha256
            ):
                raise ValueError(f"{root}: raw/calibrated response binding differs")

    response_models = {item.model_id for item in responses}
    if response_models != {declared["response_model_id"]}:
        raise ValueError(f"{root}: actual response model differs from declaration")
    full_context_requests = tuple(
        item for item in requests if item.updater_id == FULL_CONTEXT_UPDATER
    )
    if not full_context_requests:
        raise ValueError(f"{root}: no llm_full_context request evidence")

    provider_manifest = _safe_json(root / "llm/provider-manifest.json")
    if (
        provider_manifest.get("schema_version") != 1
        or provider_manifest.get("provider") != declared["provider_id"]
        or provider_manifest.get("model_requested")
        != declared["requested_model_id"]
        or provider_manifest.get("credentials_retained") is not False
    ):
        raise ValueError(f"{root}: provider manifest differs from declaration")
    audit_path = root / "llm/provider-audit.jsonl"
    attempts_path = root / "llm/transport-attempts.jsonl"
    if (
        provider_manifest.get("provider_audit_file")
        != "llm/provider-audit.jsonl"
        or provider_manifest.get("provider_audit_sha256")
        != _file_digest(audit_path)
        or provider_manifest.get("transport_attempts_file")
        != "llm/transport-attempts.jsonl"
        or provider_manifest.get("transport_attempts_sha256")
        != _file_digest(attempts_path)
    ):
        raise ValueError(f"{root}: provider manifest file bindings differ")

    if declared["provider_id"] == "openrouter":
        upstream_providers = provider_manifest.get(
            "upstream_providers_returned"
        )
        upstream_models = provider_manifest.get("upstream_models_returned")
        if (
            upstream_providers != [declared["upstream_provider_id"]]
            or upstream_models != [declared["upstream_model_id"]]
        ):
            raise ValueError(
                f"{root}: returned OpenRouter upstream labels differ from declaration"
            )

    audit_rows = _safe_jsonl(audit_path)
    audit_by_id: dict[str, dict[str, Any]] = {}
    raw_by_id = {item.request_id: item for item in raw_responses}
    for index, row in enumerate(audit_rows):
        if (
            row.get("schema_version") != 1
            or row.get("provider") != declared["provider_id"]
            or row.get("acceptance_status", "accepted") != "accepted"
            or row.get("model_requested") != declared["requested_model_id"]
            or row.get("model_returned") != declared["response_model_id"]
        ):
            raise ValueError(f"{root}: invalid provider audit row {index + 1}")
        replay_raw = row.get("replay_response")
        if not isinstance(replay_raw, Mapping):
            raise ValueError(f"{root}: provider audit lacks replay response")
        replay = LLMResponse.parse(replay_raw)
        if (
            row.get("request_id") != replay.request_id
            or row.get("prompt_sha256") != replay.prompt_sha256
            or row.get("raw_response_sha256") != replay.raw_response_sha256
        ):
            raise ValueError(f"{root}: provider audit/replay binding differs")
        expected_raw = raw_by_id.get(replay.request_id)
        if expected_raw is not None and expected_raw != replay:
            raise ValueError(f"{root}: provider audit differs from raw response")
        if replay.request_id in audit_by_id:
            raise ValueError(f"{root}: duplicate provider audit request ID")
        audit_by_id[replay.request_id] = row
    if not set(raw_by_id) <= set(audit_by_id):
        raise ValueError(f"{root}: provider audit does not cover test responses")
    if provider_manifest.get("requests_used") != len(audit_rows):
        raise ValueError(f"{root}: provider audit count differs from manifest")

    ledger = DurableProviderAttemptLedger(
        attempts_path,
        provider_name=declared["provider_id"],
        model_requested=declared["requested_model_id"],
    )
    if ledger.unresolved_attempt_ids:
        raise ValueError(f"{root}: unresolved provider transport attempt")
    event_count = len(ledger.starts) + len(ledger.settlements)
    if (
        provider_manifest.get("transport_attempt_event_count") != event_count
        or provider_manifest.get("transport_attempt_count") != len(ledger.starts)
    ):
        raise ValueError(f"{root}: provider transport count differs from manifest")
    settled_audits: dict[str, Mapping[str, Any]] = {}
    for settlement in ledger.settlements.values():
        embedded = settlement.get("provider_audit")
        if not isinstance(embedded, Mapping):
            continue
        request_id = embedded.get("request_id")
        if not isinstance(request_id, str) or request_id in settled_audits:
            raise ValueError(f"{root}: invalid settled provider audit identity")
        settled_audits[request_id] = embedded
    if set(audit_by_id) != set(settled_audits):
        raise ValueError(f"{root}: attempt settlements do not bind every audit")
    if any(
        canonical_json(audit_by_id[key]) != canonical_json(settled_audits[key])
        for key in audit_by_id
    ):
        raise ValueError(f"{root}: settled and retained provider audits differ")

    evidence = {
        "provider_id": declared["provider_id"],
        "provider_source_id": declared["provider_source_id"],
        "requested_model_id": declared["requested_model_id"],
        "response_model_id": declared["response_model_id"],
        "upstream_provider_id": declared["upstream_provider_id"],
        "upstream_model_id": declared["upstream_model_id"],
        "full_context_request_count": len(full_context_requests),
        "exchange_request_count": len(requests),
        "provider_audit_count": len(audit_rows),
        "physical_transport_attempt_count": len(ledger.starts),
        "provider_manifest_sha256": _file_digest(
            root / "llm/provider-manifest.json"
        ),
        "exchange_manifest_sha256": _file_digest(
            root / "llm/exchange-manifest.json"
        ),
        "responses_sha256": _file_digest(root / "llm/responses.jsonl"),
        "provider_audit_sha256": _file_digest(audit_path),
        "transport_attempts_sha256": _file_digest(attempts_path),
        "caller_declared_family_or_source_identity": True,
        "identity_inferred_from_display_label": False,
        "statistical_independence_established": False,
    }
    return evidence, requests, responses


def _gate_six_from_report(raw: Mapping[str, Any]) -> dict[str, Any]:
    if (
        raw.get("schema_version") != 1
        or raw.get("claim_status") != "not_claimed"
        or not isinstance(raw.get("gates"), Sequence)
    ):
        raise ValueError("invalid source gate report")
    matches = [
        gate
        for gate in raw["gates"]
        if isinstance(gate, Mapping) and gate.get("gate_id") == "gate-6"
    ]
    if len(matches) != 1:
        raise ValueError("source gate report must contain exactly one Gate 6")
    gate = dict(matches[0])
    criteria = gate.get("criteria")
    if not isinstance(criteria, Sequence):
        raise ValueError("source Gate 6 criteria must be an array")
    ids = [
        row.get("criterion_id") if isinstance(row, Mapping) else None
        for row in criteria
    ]
    if ids != list(CRITERION_IDS):
        raise ValueError("source Gate 6 criterion set or order differs")
    if (
        gate.get("claim_status") != "not_claimed"
        or gate.get("evidence_scope") != "diagnostic"
    ):
        raise ValueError("source Gate 6 claim semantics are invalid")
    return gate


def _tri_conjunction(values: Iterable[bool | None]) -> bool | None:
    material = tuple(values)
    if any(value is False for value in material):
        return False
    if material and all(value is True for value in material):
        return True
    return None


def _expected_sensitivity_points(config: AppConfig) -> tuple[Any, ...]:
    return sensitivity_grid(
        design=config.sensitivity.design,
        decision_noise_values=config.sensitivity.decision_noise_values,
        presentation_multipliers=config.sensitivity.presentation_multipliers,
        profile_conditioning_strength_values=(
            config.sensitivity.profile_conditioning_strength_values
        ),
        rank_multipliers=config.sensitivity.rank_multipliers,
        default_multipliers=config.sensitivity.default_multipliers,
        suggestion_multipliers=config.sensitivity.suggestion_multipliers,
        profile_strength_values=config.sensitivity.profile_strength_values,
        prior_uncertainty_values=config.sensitivity.prior_uncertainty_values,
        trajectory_lengths=config.sensitivity.trajectory_lengths,
        response_model_families=config.sensitivity.response_model_families,
        rule_noise_values=config.sensitivity.rule_noise_values,
    )


def _recompute_sensitivity_clauses(
    config: AppConfig,
    phase_rows: Sequence[Mapping[str, Any]],
    domain_rows: Sequence[Mapping[str, Any]],
    model_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, bool | None], bool | None, dict[str, Any]]:
    points = _expected_sensitivity_points(config)
    expected_ids = {point.point_id for point in points}
    phase_ids = [row.get("point_id") for row in phase_rows]
    model_ids = [row.get("point_id") for row in model_rows]
    if (
        len(phase_ids) != len(set(phase_ids))
        or len(model_ids) != len(set(model_ids))
        or set(phase_ids) != expected_ids
        or set(model_ids) != expected_ids
    ):
        raise ValueError("sensitivity point coverage differs from resolved grid")
    if any(
        row.get("phase_target_updater_id") != FULL_CONTEXT_UPDATER
        or row.get("phase_target_is_llm") is not True
        or row.get("phase_target_is_live_llm") is not True
        or row.get("llm_execution_mode") != config.llm.mode
        or row.get("criteria_complete") not in {True, False}
        or row.get("operational_joint_region") not in {True, False, None}
        for row in phase_rows
    ):
        raise ValueError("sensitivity phase rows are not live llm_full_context evidence")
    required_domains = set(config.experiment.domains)
    expected_domain_keys = {
        (point.point_id, domain)
        for point in points
        for domain in required_domains
    }
    domain_keys = [
        (row.get("point_id"), row.get("domain_id")) for row in domain_rows
    ]
    if (
        len(domain_keys) != len(set(domain_keys))
        or set(domain_keys) != expected_domain_keys
    ):
        raise ValueError("sensitivity domain-point coverage is incomplete")
    if any(
        row.get("phase_target_updater_id") != FULL_CONTEXT_UPDATER
        or row.get("phase_target_is_llm") is not True
        or row.get("llm_execution_mode") != config.llm.mode
        or row.get("criteria_complete") not in {True, False}
        or row.get("operational_joint_region") not in {True, False, None}
        for row in domain_rows
    ):
        raise ValueError("sensitivity domain phase rows are inconsistent")
    if any(
        not isinstance(row.get("fitted_models"), Mapping)
        or not isinstance(row.get("raw_fitted_models"), Mapping)
        for row in model_rows
    ):
        raise ValueError("sensitivity fitted-model evidence is incomplete")

    passing = tuple(
        row for row in phase_rows if row["operational_joint_region"] is True
    )
    required_response_families = {"random_utility", "rule_based"}
    passing_response_families = {
        row.get("response_model_family") for row in passing
    }
    another = (
        required_response_families <= passing_response_families
    )
    levels, survival, broad = sensitivity_breadth_coverage(
        points,
        passing,
    )
    passing_domains = {
        row.get("domain_id")
        for row in domain_rows
        if row.get("operational_joint_region") is True
    }
    both_domains = required_domains <= passing_domains
    fitted = (
        "fitted_action_aware" in config.experiment.updaters
        and len(model_rows) == len(points)
        and bool(passing)
    )
    if any(row["operational_joint_region"] is True for row in phase_rows):
        family_effect: bool | None = True
    elif all(
        row.get("criteria_complete") is True
        and row.get("operational_joint_region") is False
        for row in phase_rows
    ):
        family_effect = False
    else:
        family_effect = None
    values = {
        CRITERION_IDS[0]: another,
        CRITERION_IDS[1]: broad,
        CRITERION_IDS[2]: both_domains,
        CRITERION_IDS[5]: fitted,
    }
    observed = {
        "declared_points": len(points),
        "passing_points": len(passing),
        "passing_response_model_families": sorted(
            str(item) for item in passing_response_families
        ),
        "parameter_levels": levels,
        "passing_level_coverage": survival,
        "passing_domains": sorted(str(item) for item in passing_domains),
        "family_effect_survives": family_effect,
    }
    return values, family_effect, observed


def _sensitivity_evidence(
    pair: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    reference = pair["sensitivity_run"]
    root = Path(reference["path"])
    required = (
        "metrics/summary.json",
        "metrics/gate-report.json",
        "metrics/sensitivity-phase-points.jsonl",
        "metrics/sensitivity-phase-domains.jsonl",
        "models/sensitivity-fits.jsonl",
        "models/llm-calibration.json",
        "llm/requests.jsonl",
        "llm/responses.jsonl",
        "llm/exchange-manifest.json",
        "llm/provider-manifest.json",
        "llm/provider-audit.jsonl",
        "llm/transport-attempts.jsonl",
    )
    binding, config, summary, config_raw = _source_run_binding(
        root,
        expected_kind="sensitivity",
        expected_reference=reference,
        required_files=required,
    )
    if (
        summary.get("declared_points") != summary.get("completed_points")
        or not isinstance(summary.get("declared_points"), int)
        or summary["declared_points"] <= 0
    ):
        raise ValueError(f"{root}: sensitivity grid is incomplete")
    if FULL_CONTEXT_UPDATER not in config.experiment.updaters:
        raise ValueError(f"{root}: sensitivity run lacks llm_full_context")
    gate = _gate_six_from_report(_safe_json(root / "metrics/gate-report.json"))
    phase_rows = _safe_jsonl(
        root / "metrics/sensitivity-phase-points.jsonl"
    )
    domain_rows = _safe_jsonl(
        root / "metrics/sensitivity-phase-domains.jsonl"
    )
    model_rows = _safe_jsonl(root / "models/sensitivity-fits.jsonl")
    clause_values, family_effect, phase_observed = (
        _recompute_sensitivity_clauses(
            config,
            phase_rows,
            domain_rows,
            model_rows,
        )
    )
    if len(phase_rows) != summary["declared_points"]:
        raise ValueError(f"{root}: summary phase-point count differs")
    source_criteria = {
        row["criterion_id"]: row
        for row in gate["criteria"]
        if isinstance(row, Mapping)
    }
    for criterion_id, expected in clause_values.items():
        if source_criteria[criterion_id].get("passed") is not expected:
            raise ValueError(
                f"{root}: source Gate 6 {criterion_id} differs from recomputation"
            )
    if source_criteria[CRITERION_IDS[3]].get("passed") is not None:
        raise ValueError(f"{root}: per-run multiple-family clause must be incomplete")
    if source_criteria[CRITERION_IDS[4]].get("passed") is not None:
        raise ValueError(f"{root}: per-run paraphrase clause must be incomplete")
    provider, _, _ = _provider_evidence(
        root,
        config=config,
        declared=pair["model_binding"],
    )
    scientific = _scientific_sensitivity_config(config_raw)
    return (
        {
            "source_run": binding,
            "scientific_config_sha256": _digest(scientific),
            "source_gate_6_sha256": _file_digest(
                root / "metrics/gate-report.json"
            ),
            "source_gate_6_computed_status": gate.get("computed_status"),
            "criterion_results": clause_values,
            "phase_evidence": phase_observed,
            "family_effect_survives": family_effect,
            "provider_evidence": provider,
        },
        canonical_json(scientific),
    )


def _case_from_raw(raw: Mapping[str, Any]) -> HeldOutParaphraseCase:
    fields = {
        "schema_version",
        "case_id",
        "source_trial_id",
        "domain_id",
        "mechanism",
        "selected_option_id",
        "template_id",
        "family_id",
        "split",
        "template_sha256",
        "context_sha256",
        "surface_response",
        "binding_sha256",
    }
    _strict_fields(raw, fields, name="held-out paraphrase case")
    if raw.get("schema_version") != 1:
        raise ValueError("unsupported held-out paraphrase case schema")
    return HeldOutParaphraseCase(
        **{key: value for key, value in raw.items() if key != "schema_version"}
    )


def _paraphrase_record_from_raw(
    raw: Mapping[str, Any],
) -> ParaphraseEvaluationRecord:
    fields = {
        "schema_version",
        "case_id",
        "binding_sha256",
        "source_trial_id",
        "template_id",
        "family_id",
        "split",
        "domain_id",
        "mechanism",
        "updater_id",
        "brier",
        "belief_sha256",
    }
    _strict_fields(raw, fields, name="held-out paraphrase score")
    if raw.get("schema_version") != 1:
        raise ValueError("unsupported held-out paraphrase score schema")
    return ParaphraseEvaluationRecord(
        **{key: value for key, value in raw.items() if key != "schema_version"}
    )


def _response_belief_sha256(response: LLMResponse) -> str:
    rows = tuple(
        tuple(float(response.beliefs[attribute][value]) for value in VALUES)
        for attribute in ATTRIBUTES
    )
    belief = PreferenceBelief.from_marginals(
        MarginalPreferenceBelief(rows)  # type: ignore[arg-type]
    )
    return _digest(belief.to_dict())


def _paraphrase_evidence(
    root: Path,
    *,
    config: AppConfig,
    requests: Sequence[Any],
    responses: Sequence[LLMResponse],
) -> tuple[dict[str, Any], bool]:
    suite = build_default_paraphrase_suite()
    retained_suite = _safe_json(
        root / "models/held-out-paraphrase-suite.json"
    )
    if retained_suite != suite.to_dict():
        raise ValueError(f"{root}: retained paraphrase suite differs from fixed suite")
    case_rows = _safe_jsonl(
        root / "events/experiment-a-held-out-paraphrases.jsonl"
    )
    score_rows = _safe_jsonl(
        root / "metrics/experiment-a-held-out-paraphrase-scores.jsonl"
    )
    cases = tuple(_case_from_raw(row) for row in case_rows)
    scores = tuple(_paraphrase_record_from_raw(row) for row in score_rows)
    nonbalanced = {
        mechanism
        for mechanism in config.experiment.mechanisms
        if mechanism != "balanced"
    }
    recomputed = evaluate_gate1_paraphrase_transfer(
        cases,
        scores,
        suite=suite,
        required_mechanisms=max(1, min(2, len(nonbalanced))),
        required_domains=config.experiment.domains,
    )
    retained = _safe_json(
        root / "metrics/experiment-a-held-out-paraphrase-transfer.json"
    )
    if retained != recomputed.to_dict():
        raise ValueError(f"{root}: paraphrase transfer differs from recomputation")
    if not recomputed.complete:
        raise ValueError(f"{root}: held-out paraphrase transfer is incomplete")
    if recomputed.verified not in {True, False}:
        raise ValueError(f"{root}: held-out paraphrase result is unresolved")

    response_by_id = {item.request_id: item for item in responses}
    score_by_case = {
        score.case_id: score
        for score in scores
        if score.updater_id == FULL_CONTEXT_UPDATER
    }
    for case in cases:
        matches = [
            request
            for request in requests
            if request.updater_id == FULL_CONTEXT_UPDATER
            and request.payload.get("observation", {}).get("surface_response")
            == case.surface_response
            and request.payload.get("observation", {}).get("selected_option")
            == case.selected_option_id
            and isinstance(request.payload.get("context"), Mapping)
            and _digest(request.payload["context"]) == case.context_sha256
        ]
        if len(matches) != 1:
            raise ValueError(
                f"{root}: paraphrase case {case.case_id} does not bind exactly "
                "one llm_full_context request"
            )
        request = matches[0]
        response = response_by_id[request.request_id]
        score = score_by_case.get(case.case_id)
        if score is None or score.belief_sha256 != _response_belief_sha256(response):
            raise ValueError(
                f"{root}: paraphrase score is not bound to the provider response"
            )
    return (
        {
            "criterion": recomputed.to_dict(),
            "suite_sha256": suite.suite_sha256,
            "case_count": len(cases),
            "score_count": len(scores),
            "llm_full_context_scores_bound_to_responses": True,
        },
        bool(recomputed.verified),
    )


def _experiment_a_evidence(pair: Mapping[str, Any]) -> dict[str, Any]:
    reference = pair["experiment_a_run"]
    root = Path(reference["path"])
    required = (
        "metrics/summary.json",
        "metrics/gate-report.json",
        "models/held-out-paraphrase-suite.json",
        "events/experiment-a-held-out-paraphrases.jsonl",
        "metrics/experiment-a-held-out-paraphrase-scores.jsonl",
        "metrics/experiment-a-held-out-paraphrase-transfer.json",
        "models/llm-calibration.json",
        "llm/requests.jsonl",
        "llm/responses.jsonl",
        "llm/exchange-manifest.json",
        "llm/provider-manifest.json",
        "llm/provider-audit.jsonl",
        "llm/transport-attempts.jsonl",
    )
    binding, config, _, _ = _source_run_binding(
        root,
        expected_kind="provenance_audit",
        expected_reference=reference,
        required_files=required,
    )
    if (
        FULL_CONTEXT_UPDATER not in config.experiment.updaters
        or "fitted_action_aware" not in config.experiment.updaters
    ):
        raise ValueError(
            f"{root}: Experiment A must include llm_full_context and fitted_action_aware"
        )
    gate_report = _safe_json(root / "metrics/gate-report.json")
    if (
        gate_report.get("schema_version") != 1
        or gate_report.get("claim_status") != "not_claimed"
        or not any(
            isinstance(gate, Mapping) and gate.get("gate_id") == "gate-1"
            for gate in gate_report.get("gates", ())
        )
    ):
        raise ValueError(f"{root}: invalid Experiment A gate report")
    provider, requests, responses = _provider_evidence(
        root,
        config=config,
        declared=pair["model_binding"],
    )
    paraphrase, verified = _paraphrase_evidence(
        root,
        config=config,
        requests=requests,
        responses=responses,
    )
    return {
        "source_run": binding,
        "source_gate_report_sha256": _file_digest(
            root / "metrics/gate-report.json"
        ),
        "provider_evidence": provider,
        "held_out_paraphrase": paraphrase,
        "held_out_paraphrase_survives": verified,
    }


def _evaluate_pair(pair: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    sensitivity, scientific = _sensitivity_evidence(pair)
    experiment_a = _experiment_a_evidence(pair)
    if (
        sensitivity["provider_evidence"]["response_model_id"]
        != experiment_a["provider_evidence"]["response_model_id"]
        or sensitivity["provider_evidence"]["requested_model_id"]
        != experiment_a["provider_evidence"]["requested_model_id"]
        or sensitivity["provider_evidence"]["provider_id"]
        != experiment_a["provider_evidence"]["provider_id"]
        or sensitivity["provider_evidence"]["upstream_provider_id"]
        != experiment_a["provider_evidence"]["upstream_provider_id"]
        or sensitivity["provider_evidence"]["upstream_model_id"]
        != experiment_a["provider_evidence"]["upstream_model_id"]
    ):
        raise ValueError(
            f"{pair['pair_id']}: sensitivity and Experiment A model evidence differs"
        )
    return (
        {
            "schema_version": 1,
            "pair_id": pair["pair_id"],
            "family_id": pair["family_id"],
            "model_binding": dict(pair["model_binding"]),
            "sensitivity": sensitivity,
            "experiment_a": experiment_a,
            "validation": {
                "source_runs_verified_complete": True,
                "scientific_sensitivity_config_bound": True,
                "same_actual_model_evidence_across_pair": True,
                "held_out_paraphrase_recomputed": True,
                "held_out_paraphrase_scores_bound_to_responses": True,
                "family_identity_caller_declared": True,
                "source_identity_caller_declared": True,
                "family_inferred_from_display_labels": False,
                "statistical_independence_established": False,
                "provider_calls_made_by_review": False,
            },
        },
        scientific,
    )


def _aggregate_gate_six(
    pairs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    source_ids = (CRITERION_IDS[0], CRITERION_IDS[1], CRITERION_IDS[2], CRITERION_IDS[5])
    source_results = {
        criterion_id: _tri_conjunction(
            pair["sensitivity"]["criterion_results"][criterion_id]
            for pair in pairs
        )
        for criterion_id in source_ids
    }
    multiple_families = _tri_conjunction(
        pair["sensitivity"]["family_effect_survives"] for pair in pairs
    )
    paraphrases = _tri_conjunction(
        pair["experiment_a"]["held_out_paraphrase_survives"]
        for pair in pairs
    )
    by_family = {
        pair["family_id"]: {
            "pair_id": pair["pair_id"],
            "model_binding": pair["model_binding"],
            "sensitivity_criterion_results": pair["sensitivity"][
                "criterion_results"
            ],
            "family_effect_survives": pair["sensitivity"][
                "family_effect_survives"
            ],
            "held_out_paraphrase_survives": pair["experiment_a"][
                "held_out_paraphrase_survives"
            ],
        }
        for pair in pairs
    }
    report = GateReport(
        gate_id="gate-6",
        title="Robustness",
        evidence_scope="cross_run_diagnostic",
        claim_status="not_claimed",
        criteria=(
            GateCriterion(
                CRITERION_IDS[0],
                "The meaningful-region effect survives another response model.",
                source_results[CRITERION_IDS[0]],
                {"by_declared_family": by_family},
                "every paired sensitivity run passes another-response-model",
            ),
            GateCriterion(
                CRITERION_IDS[1],
                "The effect survives broad declared simulator parameters.",
                source_results[CRITERION_IDS[1]],
                {"by_declared_family": by_family},
                "every paired sensitivity run passes broad-simulator-parameters",
            ),
            GateCriterion(
                CRITERION_IDS[2],
                "The meaningful-region effect survives both study domains.",
                source_results[CRITERION_IDS[2]],
                {"by_declared_family": by_family},
                "every paired sensitivity run passes both-domains",
            ),
            GateCriterion(
                CRITERION_IDS[3],
                "The effect survives multiple caller-declared LLM families.",
                multiple_families,
                {
                    "declared_family_ids": sorted(by_family),
                    "distinct_exact_model_bindings": len(
                        {
                            canonical_json(pair["model_binding"])
                            for pair in pairs
                        }
                    ),
                    "family_effect_survives": {
                        pair["family_id"]: pair["sensitivity"][
                            "family_effect_survives"
                        ]
                        for pair in pairs
                    },
                    "family_identity_inferred_from_display_labels": False,
                    "statistical_independence_claimed": False,
                },
                (
                    "at least two distinct caller-declared family IDs with "
                    "distinct exact model bindings and surviving complete "
                    "LLM sensitivity evidence"
                ),
            ),
            GateCriterion(
                CRITERION_IDS[4],
                "The effect survives held-out natural-language paraphrases.",
                paraphrases,
                {
                    "held_out_paraphrase_survives": {
                        pair["family_id"]: pair["experiment_a"][
                            "held_out_paraphrase_survives"
                        ]
                        for pair in pairs
                    }
                },
                (
                    "every paired Experiment A run has a complete recomputed "
                    "held-out llm_full_context transfer result"
                ),
            ),
            GateCriterion(
                CRITERION_IDS[5],
                (
                    "The meaningful-region effect survives comparison with "
                    "exact and fitted action-aware references."
                ),
                source_results[CRITERION_IDS[5]],
                {"by_declared_family": by_family},
                (
                    "every paired sensitivity run passes "
                    "exact-and-fitted-action-aware-references"
                ),
            ),
        ),
    )
    return report.to_dict()


def _checksum_lines(root: Path) -> str:
    paths = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and path.name != "SHA256SUMS"
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    return "".join(
        f"{_file_digest(path)}  {path.relative_to(root).as_posix()}\n"
        for path in paths
    )


def _fsync_tree(root: Path) -> None:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            descriptor = os.open(path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    for directory in sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    descriptor = os.open(root, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def build_gate6_cross_run_review(
    *,
    declaration_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Validate paired runs and atomically materialize a Gate 6 review."""

    declaration = read_gate6_cross_run_declaration(declaration_path)
    output = Path(output_dir).absolute()
    _assert_no_symlink_components(output.parent, name="review output parent")
    if not output.parent.is_dir():
        raise ValueError("review output parent must already exist")
    if os.path.lexists(output):
        raise ValueError("review output must not already exist")
    for pair in declaration["pairs"]:
        for key in ("sensitivity_run", "experiment_a_run"):
            source = Path(pair[key]["path"])
            resolved_output = output.resolve(strict=False)
            if resolved_output == source or source in resolved_output.parents:
                raise ValueError("review output must be outside every source run")

    lock_path = output.parent / f".{output.name}.gate6-review.lock"
    try:
        descriptor = os.open(
            lock_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
    except FileExistsError as exc:
        raise FileExistsError(
            f"Gate 6 review output is locked: {output}"
        ) from exc
    os.close(descriptor)
    try:
        if os.path.lexists(output):
            raise ValueError("review output must not already exist")
        return _build_gate6_cross_run_review_locked(
            declaration=declaration,
            output=output,
        )
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _build_gate6_cross_run_review_locked(
    *,
    declaration: Mapping[str, Any],
    output: Path,
) -> dict[str, Any]:
    """Build one review while holding its sibling publication lock."""

    pair_evidence: list[dict[str, Any]] = []
    scientific_configs: list[str] = []
    for pair in declaration["pairs"]:
        evidence, scientific = _evaluate_pair(pair)
        pair_evidence.append(evidence)
        scientific_configs.append(scientific)
    if len(set(scientific_configs)) != 1:
        raise ValueError(
            "sensitivity scientific grids/configurations differ outside the "
            "allowed run identity, seed, output, provider, and model fields"
        )
    gate_six = _aggregate_gate_six(pair_evidence)

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    try:
        _write_json(temporary / "declaration.json", declaration)
        _write_jsonl(temporary / "evidence/pairs.jsonl", pair_evidence)
        _write_json(temporary / "metrics/gate-6.json", gate_six)
        retained = _file_bindings(
            temporary,
            (
                "declaration.json",
                "evidence/pairs.jsonl",
                "metrics/gate-6.json",
            ),
        )
        review_core = {
            "schema_version": 1,
            "artifact_kind": REVIEW_KIND,
            "claim_status": "not_claimed",
            "declaration_id": declaration["declaration_id"],
            "declaration_sha256": retained["declaration.json"]["sha256"],
            "scientific_sensitivity_config_sha256": _digest(
                json.loads(scientific_configs[0])
            ),
            "pair_count": len(pair_evidence),
            "declared_family_ids": sorted(
                pair["family_id"] for pair in pair_evidence
            ),
            "gate_6_computed_status": gate_six["computed_status"],
            "validation": {
                "source_runs_verified_complete": True,
                "source_runs_reverified_before_commit": True,
                "scientific_sensitivity_configs_matched": True,
                "model_and_provider_bindings_exact": True,
                "held_out_paraphrase_transfers_recomputed": True,
                "source_runs_mutated": False,
                "provider_calls_made": False,
                "statistical_independence_established": False,
            },
            "retained_files": retained,
            "interpretation_boundary": (
                "This is an offline computational review with "
                "claim_status=not_claimed. Family and provider-source IDs are "
                "responsible-researcher declarations, not inferences from "
                "display labels. Distinct bindings do not establish statistical "
                "independence. Preregistration timing remains the responsibility "
                "of the declaring researcher."
            ),
        }
        artifact_id = _digest(review_core)
        review = {**review_core, "artifact_id": artifact_id}
        _write_json(temporary / "review.json", review)
        retained_with_review = _file_bindings(
            temporary,
            (
                "declaration.json",
                "evidence/pairs.jsonl",
                "metrics/gate-6.json",
                "review.json",
            ),
        )
        manifest = {
            "schema_version": 1,
            "artifact_kind": REVIEW_KIND,
            "artifact_id": artifact_id,
            "status": "complete",
            "claim_status": "not_claimed",
            "review_sha256": retained_with_review["review.json"]["sha256"],
            "retained_files": retained_with_review,
        }
        _write_json(temporary / "manifest.json", manifest)
        (temporary / "SHA256SUMS").write_text(
            _checksum_lines(temporary),
            encoding="utf-8",
        )

        valid, errors = verify_gate6_cross_run_review(
            temporary,
            reverify_sources=True,
        )
        if not valid:
            raise ValueError(
                "generated Gate 6 review failed verification: "
                + "; ".join(errors)
            )
        _fsync_tree(temporary)
        if os.path.lexists(output):
            raise ValueError("review output appeared during review construction")
        os.rename(temporary, output)
        parent_descriptor = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return {
        "artifact_id": artifact_id,
        "output_dir": str(output),
        "claim_status": "not_claimed",
        "pair_count": len(pair_evidence),
        "declared_family_ids": sorted(
            pair["family_id"] for pair in pair_evidence
        ),
        "gate_6_computed_status": gate_six["computed_status"],
    }


def _verify_checksum_manifest(root: Path) -> list[str]:
    errors: list[str] = []
    checksum_path = root / "SHA256SUMS"
    if checksum_path.is_symlink() or not checksum_path.is_file():
        return ["missing or unsafe SHA256SUMS"]
    retained: set[str] = set()
    for line_number, line in enumerate(
        checksum_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            expected, relative = line.split("  ", 1)
            _require_digest(expected, "checksum")
        except ValueError as exc:
            errors.append(f"malformed checksum line {line_number}: {exc}")
            continue
        relative_path = PurePosixPath(relative)
        if (
            not relative
            or relative_path.is_absolute()
            or relative_path.as_posix() != relative
            or "\\" in relative
            or any(part in {"", ".", ".."} for part in relative_path.parts)
        ):
            errors.append(f"unsafe checksum path on line {line_number}")
            continue
        candidate = root.joinpath(*relative_path.parts)
        resolved = candidate.resolve()
        if (
            candidate.is_symlink()
            or resolved == root
            or root not in resolved.parents
        ):
            errors.append(f"unsafe review artifact path: {relative}")
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
    return errors


def _validate_pair_evidence_shape(pair: Mapping[str, Any]) -> None:
    _strict_fields(
        pair,
        {
            "schema_version",
            "pair_id",
            "family_id",
            "model_binding",
            "sensitivity",
            "experiment_a",
            "validation",
        },
        name="retained pair evidence",
    )
    if pair.get("schema_version") != 1:
        raise ValueError("unsupported retained pair schema")
    _require_text(pair.get("pair_id"), "pair_id")
    _require_text(pair.get("family_id"), "family_id")
    _validate_model_binding(pair["model_binding"], name="model_binding")
    validation = pair.get("validation")
    if not isinstance(validation, Mapping) or any(
        validation.get(field) is not expected
        for field, expected in {
            "source_runs_verified_complete": True,
            "scientific_sensitivity_config_bound": True,
            "same_actual_model_evidence_across_pair": True,
            "held_out_paraphrase_recomputed": True,
            "held_out_paraphrase_scores_bound_to_responses": True,
            "family_identity_caller_declared": True,
            "source_identity_caller_declared": True,
            "family_inferred_from_display_labels": False,
            "statistical_independence_established": False,
            "provider_calls_made_by_review": False,
        }.items()
    ):
        raise ValueError("retained pair validation semantics are incomplete")
    sensitivity = pair.get("sensitivity")
    experiment_a = pair.get("experiment_a")
    if not isinstance(sensitivity, Mapping) or not isinstance(experiment_a, Mapping):
        raise ValueError("retained pair source evidence must be objects")
    results = sensitivity.get("criterion_results")
    if not isinstance(results, Mapping) or set(results) != {
        CRITERION_IDS[0],
        CRITERION_IDS[1],
        CRITERION_IDS[2],
        CRITERION_IDS[5],
    }:
        raise ValueError("retained sensitivity criterion results differ")
    if any(value not in {True, False, None} for value in results.values()):
        raise ValueError("retained sensitivity criterion is not tri-state")
    if sensitivity.get("family_effect_survives") not in {True, False, None}:
        raise ValueError("retained family effect is not tri-state")
    if experiment_a.get("held_out_paraphrase_survives") not in {True, False}:
        raise ValueError("retained paraphrase result must be complete")


def verify_gate6_cross_run_review(
    path: str | Path,
    *,
    reverify_sources: bool = False,
) -> tuple[bool, tuple[str, ...]]:
    """Verify checksums and semantics, optionally re-reading all source runs."""

    supplied_root = Path(path)
    errors: list[str] = []
    if supplied_root.is_symlink():
        return False, ("review path must be a safe directory",)
    root = supplied_root.resolve()
    if not root.is_dir():
        return False, ("review path must be a safe directory",)
    for item in root.rglob("*"):
        if item.is_symlink():
            errors.append(f"review contains a symlink: {item.relative_to(root)}")
    errors.extend(_verify_checksum_manifest(root))
    try:
        declaration = read_gate6_cross_run_declaration(
            root / "declaration.json",
            require_source_paths=reverify_sources,
        )
        pair_rows = _safe_jsonl(root / "evidence/pairs.jsonl")
        for pair in pair_rows:
            _validate_pair_evidence_shape(pair)
        if len(pair_rows) != len(declaration["pairs"]):
            raise ValueError("declaration and retained pair counts differ")
        pairs_by_id = {pair["pair_id"]: pair for pair in pair_rows}
        if len(pairs_by_id) != len(pair_rows):
            raise ValueError("retained pair IDs are not unique")
        for declared in declaration["pairs"]:
            retained = pairs_by_id.get(declared["pair_id"])
            if retained is None:
                raise ValueError("retained evidence lacks a declared pair")
            if (
                retained["family_id"] != declared["family_id"]
                or retained["model_binding"] != declared["model_binding"]
            ):
                raise ValueError("retained pair declaration binding differs")
        scientific_hashes = {
            pair["sensitivity"].get("scientific_config_sha256")
            for pair in pair_rows
        }
        if len(scientific_hashes) != 1 or not all(
            isinstance(value, str) and _DIGEST.fullmatch(value)
            for value in scientific_hashes
        ):
            raise ValueError("retained scientific sensitivity configs differ")
        expected_gate = _aggregate_gate_six(pair_rows)
        retained_gate = _safe_json(root / "metrics/gate-6.json")
        if retained_gate != expected_gate:
            raise ValueError("retained Gate 6 differs from recomputation")
        if (
            retained_gate.get("claim_status") != "not_claimed"
            or [
                row.get("criterion_id")
                for row in retained_gate.get("criteria", ())
                if isinstance(row, Mapping)
            ]
            != list(CRITERION_IDS)
        ):
            raise ValueError("retained Gate 6 claim or criterion semantics differ")

        review = _safe_json(root / "review.json")
        manifest = _safe_json(root / "manifest.json")
        review_core = dict(review)
        artifact_id = review_core.pop("artifact_id", None)
        if artifact_id != _digest(review_core):
            raise ValueError("review artifact_id mismatch")
        if (
            review.get("schema_version") != 1
            or review.get("artifact_kind") != REVIEW_KIND
            or review.get("claim_status") != "not_claimed"
            or review.get("declaration_id") != declaration["declaration_id"]
            or review.get("declaration_sha256")
            != _file_digest(root / "declaration.json")
            or review.get("scientific_sensitivity_config_sha256")
            != next(iter(scientific_hashes))
            or review.get("pair_count") != len(pair_rows)
            or review.get("declared_family_ids")
            != sorted(pair["family_id"] for pair in pair_rows)
            or review.get("gate_6_computed_status")
            != retained_gate["computed_status"]
        ):
            raise ValueError("review metadata semantics differ")
        review_validation = review.get("validation")
        if not isinstance(review_validation, Mapping) or any(
            review_validation.get(field) is not expected
            for field, expected in {
                "source_runs_verified_complete": True,
                "source_runs_reverified_before_commit": True,
                "scientific_sensitivity_configs_matched": True,
                "model_and_provider_bindings_exact": True,
                "held_out_paraphrase_transfers_recomputed": True,
                "source_runs_mutated": False,
                "provider_calls_made": False,
                "statistical_independence_established": False,
            }.items()
        ):
            raise ValueError("review validation semantics are incomplete")
        if (
            manifest.get("schema_version") != 1
            or manifest.get("artifact_kind") != REVIEW_KIND
            or manifest.get("artifact_id") != artifact_id
            or manifest.get("status") != "complete"
            or manifest.get("claim_status") != "not_claimed"
            or manifest.get("review_sha256")
            != _file_digest(root / "review.json")
        ):
            raise ValueError("review manifest semantics differ")
        for owner_name, owner, expected_files in (
            (
                "review",
                review,
                {
                    "declaration.json",
                    "evidence/pairs.jsonl",
                    "metrics/gate-6.json",
                },
            ),
            (
                "manifest",
                manifest,
                {
                    "declaration.json",
                    "evidence/pairs.jsonl",
                    "metrics/gate-6.json",
                    "review.json",
                },
            ),
        ):
            retained_files = owner.get("retained_files")
            if (
                not isinstance(retained_files, Mapping)
                or set(retained_files) != expected_files
            ):
                raise ValueError(f"{owner_name} retained file set differs")
            for relative, binding in retained_files.items():
                if (
                    not isinstance(binding, Mapping)
                    or binding.get("sha256") != _file_digest(root / relative)
                    or binding.get("bytes") != len((root / relative).read_bytes())
                ):
                    raise ValueError(f"{owner_name} file binding differs: {relative}")

        if reverify_sources:
            recomputed: list[dict[str, Any]] = []
            scientific: list[str] = []
            for declared in declaration["pairs"]:
                evidence, normalized = _evaluate_pair(declared)
                recomputed.append(evidence)
                scientific.append(normalized)
            if len(set(scientific)) != 1:
                raise ValueError("source scientific sensitivity configs now differ")
            if canonical_json(recomputed) != canonical_json(list(pair_rows)):
                raise ValueError("source runs changed after review construction")
    except (KeyError, OSError, TypeError, ValueError) as exc:
        errors.append(f"invalid retained Gate 6 review content: {exc}")
    return not errors, tuple(errors)
