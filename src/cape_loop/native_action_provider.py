"""OpenAI-backed native terminal actions for the immutable Gate 4 workflow.

This module turns retained Experiment B native-memory states into real,
provider-produced terminal actions.  It deliberately does not project a belief
to an action locally: the exact native state and held-out suite are sent to the
declared model, and the model must return one schema-bound action per item.

Planning is credential-free.  Live collection requires an explicit opt-in,
uses the hardened OpenAI origin/key/budget configuration, writes the provider
audit before the reusable action record, and can reconcile a crash between
those two appends.  The resulting ``native-actions.jsonl`` is accepted by
``gate-review import-native`` only after that command independently revalidates
the source run, state, suite, and action bindings.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence
import json
import math
import os
import random
import tempfile
import time

from .artifacts import canonical_json, verify_run
from .file_lock import try_file_lock, unlock_file
from .gate_review import (
    NativeTerminalActionRecord,
    _eligible_trajectories,
    _native_state_id,
    _parse_terminal_suite,
    read_native_terminal_action_records,
)
from .heldout import HeldOutTerminalSuite, TerminalAction
from .openai_provider import (
    ExecutionBudget,
    HTTPResponseBodyTooLarge,
    HTTPTransport,
    LiveExecutionRequired,
    MissingAPIKey,
    OpenAIProviderConfig,
    OpenAIProviderError,
    ProviderHTTPError,
    ProviderResponseError,
    _redact_provider_value,
    _repair_trailing_jsonl,
    returned_model_is_consistent,
    urllib_transport,
)


NATIVE_ACTION_SYSTEM_ID = "cape-loop-openai-native-agent-v1"
NATIVE_ACTION_SCHEMA_NAME = "cape_loop_native_terminal_actions"
NATIVE_ACTION_INSTRUCTION = """\
Act as the deployed personalized assistant whose complete persistent memory is
provided in the request. Use only that memory and each terminal item's visible
wording/options. Return exactly one action for every item.

For a choice item, copy one displayed option_id into selected_option_id and set
declared_direction to null. For a direct_preference_probe, set
selected_option_id to null and declared_direction to -1 or +1. Copy every
item_id, item_sha256, wording_template_id, and question_type exactly. Do not
infer or request latent truth; it is intentionally absent.
"""
OPENAI_NATIVE_ACTION_REFERENCE = (
    "https://developers.openai.com/api/docs/models/gpt-5.6-sol"
)
OPENAI_STRUCTURED_OUTPUT_REFERENCE = (
    "https://developers.openai.com/api/docs/guides/structured-outputs"
)
MODEL_SELECTION_RESOLVED_ON = "2026-07-26"
NATIVE_ACTION_ATTEMPT_KIND = "native-action-transport-attempt"


class NativeActionManualReviewRequired(OpenAIProviderError):
    """A durable attempt has an unknown outcome and must not be retried."""


def _utc_timestamp(epoch_seconds: float) -> str:
    return (
        datetime.fromtimestamp(epoch_seconds, timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _digest(value: Any) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _validate_digest(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _validate_timestamp(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty timestamp")
    candidate = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{name} must include a timezone")
    return value


def _collection_config(config: OpenAIProviderConfig) -> dict[str, Any]:
    """Return the complete credential-free identity of a live collection."""

    return {
        "provider": "openai",
        "endpoint": config.endpoint,
        "base_url": config.base_url,
        "api_key_env": config.api_key_env,
        "allow_custom_base_url": config.allow_custom_base_url,
        "official_origin_locked": (
            config.base_url.rstrip("/") == "https://api.openai.com"
        ),
        "model": config.model,
        "reasoning_effort": config.reasoning_effort,
        "max_output_tokens": config.max_output_tokens,
        "timeout_seconds": config.timeout_seconds,
        "max_retries": config.max_retries,
        "initial_backoff_seconds": config.initial_backoff_seconds,
        "max_backoff_seconds": config.max_backoff_seconds,
        "jitter_fraction": config.jitter_fraction,
        "max_requests": config.max_requests,
        "max_total_tokens": config.max_total_tokens,
        "budget_accounting_unit": "actual_transport_attempt",
    }


def _collection_config_sha256(config: OpenAIProviderConfig) -> str:
    return _digest(_collection_config(config))


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
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
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(dict(decoded))
    if not rows:
        raise ValueError(f"{path}: input cannot be empty")
    return tuple(rows)


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = (canonical_json(record) + "\n").encode("utf-8")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_APPEND,
        0o600,
    )
    try:
        offset = 0
        while offset < len(line):
            offset += os.write(descriptor, line[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_bytes(
        path,
        (
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8"),
    )


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(name)
    try:
        with os.fdopen(
            descriptor,
            "w",
            encoding="utf-8",
            newline="\n",
        ) as handle:
            handle.write(content.decode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


@dataclass(frozen=True, slots=True)
class NativeActionRequest:
    """One content-addressed native state and terminal-suite request."""

    request_id: str
    trajectory_id: str
    domain_id: str
    updater_id: str
    native_state: Mapping[str, Any]
    native_state_id: str
    suite: HeldOutTerminalSuite
    system_instruction: str = NATIVE_ACTION_INSTRUCTION
    prompt_sha256: str = ""

    def __post_init__(self) -> None:
        for name in (
            "request_id",
            "trajectory_id",
            "domain_id",
            "updater_id",
            "system_instruction",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
        _validate_digest(self.native_state_id, "native_state_id")
        if self.suite.domain_id != self.domain_id:
            raise ValueError("terminal suite domain does not match request")
        if self.native_state.get("state_id") != self.native_state_id:
            raise ValueError("native state payload does not match state_id")
        expected = _digest(
            {
                "instruction": self.system_instruction,
                "payload": self.payload,
            }
        )
        if self.prompt_sha256 and self.prompt_sha256 != expected:
            raise ValueError("native action prompt digest does not match content")
        object.__setattr__(self, "prompt_sha256", expected)

    @property
    def payload(self) -> dict[str, Any]:
        return {
            "native_memory_state": dict(self.native_state),
            "terminal_suite": self.suite.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "request_id": self.request_id,
            "trajectory_id": self.trajectory_id,
            "domain_id": self.domain_id,
            "updater_id": self.updater_id,
            "native_state_id": self.native_state_id,
            "suite_id": self.suite.suite_id,
            "suite_sha256": self.suite.suite_sha256,
            "system_instruction": self.system_instruction,
            "payload": self.payload,
            "prompt_sha256": self.prompt_sha256,
        }


def build_native_action_requests(
    run_dir: str | Path,
) -> tuple[NativeActionRequest, ...]:
    """Build exact Gate 4 action requests from a verified Experiment B run."""

    run = Path(run_dir).resolve()
    valid, errors = verify_run(run)
    if not valid:
        raise ValueError(
            "source run verification failed: " + "; ".join(errors)
        )
    config = json.loads(
        (run / "config.resolved.json").read_text(encoding="utf-8")
    )
    if (
        not isinstance(config, Mapping)
        or not isinstance(config.get("experiment"), Mapping)
        or config["experiment"].get("kind") != "closed_loop"
    ):
        raise ValueError("native action collection requires Experiment B")
    if (
        not isinstance(config.get("artifacts"), Mapping)
        or config["artifacts"].get("retain_events") is not True
    ):
        raise ValueError("native action collection requires retained events")

    trajectories = _read_jsonl(
        run / "events" / "experiment-b-trajectories.jsonl"
    )
    eligible = _eligible_trajectories(trajectories)
    suite_rows = _read_jsonl(
        run / "events" / "experiment-b-held-out-terminal-suites.jsonl"
    )
    suites = tuple(_parse_terminal_suite(row) for row in suite_rows)
    if len({suite.domain_id for suite in suites}) != len(suites):
        raise ValueError("duplicate held-out terminal suite domains")
    by_domain = {suite.domain_id: suite for suite in suites}

    requests: list[NativeActionRequest] = []
    for trajectory_id, row in sorted(eligible.items()):
        updater_id = row.get("updater_id")
        domain_id = row.get("domain_id")
        state = row.get("terminal_native_state")
        if not isinstance(updater_id, str) or not updater_id:
            raise ValueError("eligible trajectory has no updater_id")
        if not isinstance(domain_id, str) or not domain_id:
            raise ValueError("eligible trajectory has no domain_id")
        if not isinstance(state, Mapping):
            raise ValueError("eligible trajectory has no terminal native state")
        state_id = _native_state_id(state, updater_id=updater_id)
        try:
            suite = by_domain[domain_id]
        except KeyError as exc:
            raise ValueError(
                f"missing held-out terminal suite for {domain_id!r}"
            ) from exc
        identity = _digest(
            {
                "trajectory_id": trajectory_id,
                "native_state_id": state_id,
                "suite_sha256": suite.suite_sha256,
                "system_id": NATIVE_ACTION_SYSTEM_ID,
            }
        )
        requests.append(
            NativeActionRequest(
                request_id=f"native-action:{identity}",
                trajectory_id=trajectory_id,
                domain_id=domain_id,
                updater_id=updater_id,
                native_state=dict(state),
                native_state_id=state_id,
                suite=suite,
            )
        )
    return tuple(requests)


def native_action_json_schema(
    request: NativeActionRequest,
) -> dict[str, Any]:
    """Return the strict output schema for one exact held-out suite."""

    item_ids = [item.item_id for item in request.suite.items]
    option_ids = sorted(
        {
            option.option_id
            for item in request.suite.items
            for option in item.options
        }
    )
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "actions": {
                "type": "array",
                "minItems": len(item_ids),
                "maxItems": len(item_ids),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "item_id": {
                            "type": "string",
                            "enum": item_ids,
                        },
                        "item_sha256": {"type": "string"},
                        "wording_template_id": {"type": "string"},
                        "question_type": {
                            "type": "string",
                            "enum": sorted(
                                {
                                    item.question_type
                                    for item in request.suite.items
                                }
                            ),
                        },
                        "selected_option_id": {
                            "type": ["string", "null"],
                            **(
                                {"enum": [*option_ids, None]}
                                if option_ids
                                else {}
                            ),
                        },
                        "declared_direction": {
                            "type": ["integer", "null"],
                            "enum": [-1, 1, None],
                        },
                    },
                    "required": [
                        "item_id",
                        "item_sha256",
                        "wording_template_id",
                        "question_type",
                        "selected_option_id",
                        "declared_direction",
                    ],
                },
            }
        },
        "required": ["actions"],
    }


@dataclass(frozen=True, slots=True)
class PreparedNativeActionRequest:
    endpoint: str
    body: Mapping[str, Any]
    body_bytes: bytes
    body_sha256: str
    headers: Mapping[str, str]
    idempotency_key: str
    client_request_id: str
    estimated_max_tokens: int


def prepare_openai_native_action_request(
    request: NativeActionRequest,
    config: OpenAIProviderConfig,
) -> PreparedNativeActionRequest:
    """Build a deterministic keyless OpenAI Responses request."""

    body: dict[str, Any] = {
        "model": config.model,
        "instructions": request.system_instruction,
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": canonical_json(request.payload),
                    }
                ],
            }
        ],
        "reasoning": {"effort": config.reasoning_effort},
        "text": {
            "format": {
                "type": "json_schema",
                "name": NATIVE_ACTION_SCHEMA_NAME,
                "description": (
                    "One exact, item-bound native-system action for every "
                    "held-out terminal item."
                ),
                "strict": True,
                "schema": native_action_json_schema(request),
            }
        },
        "max_output_tokens": config.max_output_tokens,
        "store": False,
        "metadata": {
            "cape_loop_prompt_sha256": request.prompt_sha256,
            "cape_loop_native_state_id": request.native_state_id,
            "cape_loop_suite_sha256": request.suite.suite_sha256,
            "cape_loop_collection_config_sha256": (
                _collection_config_sha256(config)
            ),
        },
    }
    body_bytes = canonical_json(body).encode("utf-8")
    body_sha256 = sha256(body_bytes).hexdigest()
    identity = sha256(
        (
            "cape-loop-native-action-v1\n"
            + request.request_id
            + "\n"
            + request.prompt_sha256
            + "\n"
            + config.model
            + "\n"
            + body_sha256
        ).encode("utf-8")
    ).hexdigest()
    key = "cape-loop-native-" + identity
    headers = MappingProxyType(
        {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Idempotency-Key": key,
            "X-Client-Request-Id": key,
            "User-Agent": "cape-loop/0.1",
        }
    )
    return PreparedNativeActionRequest(
        endpoint=config.endpoint,
        body=body,
        body_bytes=body_bytes,
        body_sha256=body_sha256,
        headers=headers,
        idempotency_key=key,
        client_request_id=key,
        estimated_max_tokens=(
            len(body_bytes) + 512 + config.max_output_tokens
        ),
    )


def _lower_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {str(key).lower(): str(value) for key, value in headers.items()}


def _retry_after_seconds(
    value: str | None,
    *,
    now_epoch: float,
) -> float | None:
    if value is None:
        return None
    try:
        seconds = float(value.strip())
    except ValueError:
        try:
            parsed = parsedate_to_datetime(value.strip())
        except (TypeError, ValueError, OverflowError):
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        seconds = parsed.timestamp() - now_epoch
    if not math.isfinite(seconds):
        return None
    return max(0.0, seconds)


def _safe_error(body: bytes, secret: str) -> str:
    message = "provider returned an error response"
    try:
        parsed = json.loads(body.decode("utf-8"))
        if isinstance(parsed, Mapping):
            error = parsed.get("error")
            if isinstance(error, Mapping) and isinstance(
                error.get("message"),
                str,
            ):
                message = error["message"]
    except (UnicodeDecodeError, json.JSONDecodeError):
        pass
    return " ".join(message.replace(secret, "[redacted]").split())[:500]


def _safe_provider_identifier(
    value: object,
    *,
    secret: str,
) -> str | None:
    """Normalize and redact an identifier controlled by the provider."""

    if not isinstance(value, str):
        return None
    safe = " ".join(value.replace(secret, "[redacted]").split())[:500]
    return safe or None


def _usage_total(usage: Mapping[str, Any]) -> int | None:
    total = usage.get("total_tokens")
    if isinstance(total, int) and not isinstance(total, bool) and total >= 0:
        return total
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    if all(
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 0
        for value in (input_tokens, output_tokens)
    ):
        return int(input_tokens) + int(output_tokens)
    return None


def _parse_json_text(value: str) -> Mapping[str, Any] | None:
    candidate = value.strip()
    if candidate.startswith("```") and candidate.endswith("```"):
        newline = candidate.find("\n")
        if newline >= 0:
            candidate = candidate[newline + 1 : -3].strip()
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, Mapping) else None


def _extract_actions_payload(raw: Mapping[str, Any]) -> Mapping[str, Any]:
    status = raw.get("status")
    if status != "completed":
        raise ProviderResponseError(
            "OpenAI native action response is not completed"
        )
    candidates: list[Mapping[str, Any]] = []
    texts: list[str] = []
    refusals: list[str] = []
    if isinstance(raw.get("output_parsed"), Mapping):
        candidates.append(raw["output_parsed"])
    if isinstance(raw.get("output_text"), str):
        texts.append(raw["output_text"])
    output = raw.get("output")
    if isinstance(output, Sequence) and not isinstance(output, (str, bytes)):
        for item in output:
            if not isinstance(item, Mapping):
                continue
            if isinstance(item.get("parsed"), Mapping):
                candidates.append(item["parsed"])
            content = item.get("content")
            if not isinstance(content, Sequence) or isinstance(
                content,
                (str, bytes),
            ):
                continue
            for part in content:
                if not isinstance(part, Mapping):
                    continue
                if (
                    part.get("type") == "refusal"
                    and isinstance(part.get("refusal"), str)
                ):
                    refusals.append(part["refusal"])
                for key in ("parsed", "json"):
                    if isinstance(part.get(key), Mapping):
                        candidates.append(part[key])
                if isinstance(part.get("text"), str):
                    texts.append(part["text"])
    if refusals:
        raise ProviderResponseError(
            "OpenAI native action response was refused: "
            + " ".join(refusals)[:500]
        )
    for candidate in candidates:
        if "actions" in candidate:
            return candidate
    for text in texts:
        parsed = _parse_json_text(text)
        if parsed is not None and "actions" in parsed:
            return parsed
    joined = _parse_json_text("".join(texts)) if len(texts) > 1 else None
    if joined is not None and "actions" in joined:
        return joined
    raise ProviderResponseError(
        "OpenAI response contains no parseable native actions"
    )


def _parse_actions(
    request: NativeActionRequest,
    payload: Mapping[str, Any],
) -> tuple[TerminalAction, ...]:
    if set(payload) != {"actions"}:
        raise ProviderResponseError(
            "native action output must contain exactly actions"
        )
    rows = payload["actions"]
    if not isinstance(rows, list):
        raise ProviderResponseError("native actions must be an array")
    allowed = {
        "item_id",
        "item_sha256",
        "wording_template_id",
        "question_type",
        "selected_option_id",
        "declared_direction",
    }
    actions: list[TerminalAction] = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != allowed:
            raise ProviderResponseError(
                "each native action must use the exact action fields"
            )
        try:
            actions.append(
                TerminalAction(
                    item_id=row["item_id"],
                    item_sha256=row["item_sha256"],
                    wording_template_id=row["wording_template_id"],
                    question_type=row["question_type"],
                    selected_option_id=row["selected_option_id"],
                    declared_direction=row["declared_direction"],
                )
            )
        except (TypeError, ValueError) as exc:
            raise ProviderResponseError(
                f"invalid native terminal action: {exc}"
            ) from exc
    by_id = {action.item_id: action for action in actions}
    if len(by_id) != len(actions):
        raise ProviderResponseError("native action item IDs are duplicated")
    expected = {item.item_id: item for item in request.suite.items}
    if set(by_id) != set(expected):
        raise ProviderResponseError(
            "native actions do not cover the terminal suite exactly"
        )
    for item_id, item in expected.items():
        action = by_id[item_id]
        if (
            action.item_sha256 != item.item_sha256
            or action.wording_template_id != item.wording_template_id
            or action.question_type != item.question_type
        ):
            raise ProviderResponseError(
                f"native action binding mismatch for {item_id}"
            )
        if (
            item.question_type != "direct_preference_probe"
            and action.selected_option_id
            not in {option.option_id for option in item.options}
        ):
            raise ProviderResponseError(
                f"native action selected an unavailable option for {item_id}"
            )
    return tuple(actions)


@dataclass(frozen=True, slots=True)
class NativeActionProviderResult:
    request: NativeActionRequest
    record: NativeTerminalActionRecord
    model_requested: str
    model_returned: str
    provider_response_id: str
    usage: Mapping[str, Any]
    started_at: str
    completed_at: str
    attempts: int
    request_body_sha256: str
    idempotency_key: str
    client_request_id: str
    server_request_id: str | None
    estimated_max_tokens: int
    raw_response_sha256: str
    raw_response: Mapping[str, Any]

    def to_audit_record(
        self,
        *,
        acceptance_status: str = "accepted",
    ) -> dict[str, Any]:
        if acceptance_status not in {
            "accepted",
            "rejected_model_mismatch",
        }:
            raise ValueError("unknown native action acceptance status")
        return {
            "schema_version": 1,
            "provider": "openai",
            "workflow": "native_terminal_actions",
            "acceptance_status": acceptance_status,
            "request_id": self.request.request_id,
            "trajectory_id": self.request.trajectory_id,
            "prompt_sha256": self.request.prompt_sha256,
            "native_state_id": self.request.native_state_id,
            "suite_sha256": self.request.suite.suite_sha256,
            "request_body_sha256": self.request_body_sha256,
            "model_requested": self.model_requested,
            "model_returned": self.model_returned,
            "provider_response_id": self.provider_response_id,
            "usage": dict(self.usage),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "attempts": self.attempts,
            "idempotency_key": self.idempotency_key,
            "client_request_id": self.client_request_id,
            "server_request_id": self.server_request_id,
            "estimated_max_tokens": self.estimated_max_tokens,
            "raw_response_sha256": self.raw_response_sha256,
            "raw_response": dict(self.raw_response),
            "action_record": self.record.to_dict(),
        }


class NativeActionModelMismatch(OpenAIProviderError):
    def __init__(self, result: NativeActionProviderResult) -> None:
        self.result = result
        super().__init__(
            "OpenAI returned a model inconsistent with the native action "
            f"request: requested={result.model_requested!r}, "
            f"returned={result.model_returned!r}"
        )


_AUDIT_FIELDS = {
    "schema_version",
    "provider",
    "workflow",
    "acceptance_status",
    "request_id",
    "trajectory_id",
    "prompt_sha256",
    "native_state_id",
    "suite_sha256",
    "request_body_sha256",
    "model_requested",
    "model_returned",
    "provider_response_id",
    "usage",
    "started_at",
    "completed_at",
    "attempts",
    "idempotency_key",
    "client_request_id",
    "server_request_id",
    "estimated_max_tokens",
    "raw_response_sha256",
    "raw_response",
    "action_record",
}


def _validate_audit_record(
    row: Mapping[str, Any],
    request: NativeActionRequest,
    prepared: PreparedNativeActionRequest,
    config: OpenAIProviderConfig,
) -> NativeTerminalActionRecord:
    """Strictly rebind one durable provider audit to its exact request."""

    if set(row) != _AUDIT_FIELDS:
        raise ValueError("native action audit has invalid fields")
    if (
        row.get("schema_version") != 1
        or row.get("provider") != "openai"
        or row.get("workflow") != "native_terminal_actions"
    ):
        raise ValueError("native action audit has an invalid record identity")
    status = row.get("acceptance_status")
    if status not in {"accepted", "rejected_model_mismatch"}:
        raise ValueError("native action audit has an invalid acceptance status")
    expected = {
        "request_id": request.request_id,
        "trajectory_id": request.trajectory_id,
        "prompt_sha256": request.prompt_sha256,
        "native_state_id": request.native_state_id,
        "suite_sha256": request.suite.suite_sha256,
        "request_body_sha256": prepared.body_sha256,
        "model_requested": config.model,
        "idempotency_key": prepared.idempotency_key,
        "client_request_id": prepared.client_request_id,
        "estimated_max_tokens": prepared.estimated_max_tokens,
    }
    if any(row.get(name) != value for name, value in expected.items()):
        raise ValueError(
            "native action audit does not match the current request/configuration"
        )
    returned_model = row.get("model_returned")
    if not isinstance(returned_model, str) or not returned_model.strip():
        raise ValueError("native action audit has an invalid returned model")
    model_matches = returned_model_is_consistent(
        config.model,
        returned_model,
    )
    if (status == "accepted") != model_matches:
        raise ValueError(
            "native action audit acceptance does not match returned model"
        )
    response_id = row.get("provider_response_id")
    if not isinstance(response_id, str) or not response_id.strip():
        raise ValueError("native action audit has an invalid response ID")
    usage = row.get("usage")
    raw_response = row.get("raw_response")
    if not isinstance(usage, Mapping) or not isinstance(raw_response, Mapping):
        raise ValueError("native action audit usage/raw response is invalid")
    attempts = row.get("attempts")
    if (
        not isinstance(attempts, int)
        or isinstance(attempts, bool)
        or not 1 <= attempts <= config.max_retries + 1
    ):
        raise ValueError("native action audit attempt count is invalid")
    _validate_timestamp(row.get("started_at"), "audit started_at")
    _validate_timestamp(row.get("completed_at"), "audit completed_at")
    raw_digest = _validate_digest(
        row.get("raw_response_sha256"),
        "audit raw_response_sha256",
    )
    server_request_id = row.get("server_request_id")
    if server_request_id is not None and not isinstance(
        server_request_id,
        str,
    ):
        raise ValueError("native action audit server request ID is invalid")
    if raw_response.get("id") != response_id:
        raise ValueError("native action audit/raw response ID mismatch")
    raw_model = raw_response.get("model")
    if not isinstance(raw_model, str) or raw_model.strip() != returned_model:
        raise ValueError("native action audit/raw response model mismatch")
    raw_usage = raw_response.get("usage")
    expected_usage = dict(raw_usage) if isinstance(raw_usage, Mapping) else {}
    if dict(usage) != expected_usage:
        raise ValueError("native action audit/raw response usage mismatch")

    raw_record = row.get("action_record")
    if not isinstance(raw_record, Mapping):
        raise ValueError("native action audit lacks its action record")
    record = NativeTerminalActionRecord.parse(raw_record)
    expected_record_id = "openai-native-action:" + _digest(
        {
            "request_id": request.request_id,
            "provider_response_id": response_id,
            "raw_response_sha256": raw_digest,
        }
    )
    if (
        record.record_id != expected_record_id
        or record.trajectory_id != request.trajectory_id
        or record.domain_id != request.domain_id
        or record.updater_id != request.updater_id
        or record.native_state_id != request.native_state_id
        or record.native_system_id != NATIVE_ACTION_SYSTEM_ID
        or record.native_system_version
        != f"openai-responses:{returned_model}"
        or record.suite_id != request.suite.suite_id
        or record.suite_sha256 != request.suite.suite_sha256
        or record.action_execution_mode != "recorded_live"
        or record.execution_trace_sha256 != raw_digest
        or record.recorded_at != row.get("completed_at")
    ):
        raise ValueError("native action audit action-record binding mismatch")
    try:
        parsed_actions = _parse_actions(
            request,
            _extract_actions_payload(raw_response),
        )
    except (TypeError, ValueError, ProviderResponseError) as exc:
        raise ValueError(
            "native action audit raw response has invalid actions"
        ) from exc
    if parsed_actions != record.actions:
        raise ValueError(
            "native action audit raw-response/action-record mismatch"
        )
    return record


_ATTEMPT_OUTCOMES = {
    "transport_error",
    "http_error",
    "invalid_response",
    "model_mismatch",
    "success",
}


class _DurableAttemptLedger:
    """Fsynced started/settled records for every physical HTTP attempt."""

    def __init__(
        self,
        path: Path,
        *,
        collection_plan_sha256: str,
        collection_config_sha256: str,
    ) -> None:
        self.path = path
        self.collection_plan_sha256 = _validate_digest(
            collection_plan_sha256,
            "collection_plan_sha256",
        )
        self.collection_config_sha256 = _validate_digest(
            collection_config_sha256,
            "collection_config_sha256",
        )
        self.starts: dict[str, dict[str, Any]] = {}
        self.settlements: dict[str, dict[str, Any]] = {}
        self._ordinals: dict[str, int] = {}
        if path.exists():
            self._read()

    def _read(self) -> None:
        with self.path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    decoded = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise NativeActionManualReviewRequired(
                        f"{self.path}:{line_number}: malformed durable attempt "
                        "journal; manual review is required"
                    ) from exc
                if not isinstance(decoded, Mapping):
                    raise ValueError(
                        f"{self.path}:{line_number}: attempt event must be an object"
                    )
                row = dict(decoded)
                if (
                    row.get("schema_version") != 1
                    or row.get("kind") != NATIVE_ACTION_ATTEMPT_KIND
                ):
                    raise ValueError(
                        f"{self.path}:{line_number}: unknown attempt schema"
                    )
                if row.get("event") == "started":
                    self._parse_start(row, line_number=line_number)
                elif row.get("event") == "settled":
                    self._parse_settlement(row, line_number=line_number)
                else:
                    raise ValueError(
                        f"{self.path}:{line_number}: invalid attempt event"
                    )

    @property
    def unresolved_attempt_ids(self) -> tuple[str, ...]:
        return tuple(
            attempt_id
            for attempt_id in self.starts
            if attempt_id not in self.settlements
        )

    def _parse_start(
        self,
        row: dict[str, Any],
        *,
        line_number: int,
    ) -> None:
        allowed = {
            "schema_version",
            "kind",
            "event",
            "attempt_id",
            "collection_plan_sha256",
            "collection_config_sha256",
            "request_id",
            "prompt_sha256",
            "native_state_id",
            "suite_sha256",
            "request_body_sha256",
            "model_requested",
            "idempotency_key",
            "client_request_id",
            "estimated_max_tokens",
            "attempt_ordinal",
            "started_at",
        }
        if set(row) != allowed:
            raise ValueError(
                f"{self.path}:{line_number}: invalid started-attempt fields"
            )
        if self.unresolved_attempt_ids:
            raise NativeActionManualReviewRequired(
                f"{self.path}:{line_number}: an unresolved attempt precedes "
                "another event; manual review is required"
            )
        attempt_id = _validate_digest(row.get("attempt_id"), "attempt_id")
        for name in (
            "collection_plan_sha256",
            "collection_config_sha256",
            "prompt_sha256",
            "native_state_id",
            "suite_sha256",
            "request_body_sha256",
        ):
            _validate_digest(row.get(name), name)
        request_id = row.get("request_id")
        ordinal = row.get("attempt_ordinal")
        estimated = row.get("estimated_max_tokens")
        if (
            not isinstance(request_id, str)
            or not request_id
            or not isinstance(ordinal, int)
            or isinstance(ordinal, bool)
            or ordinal <= 0
            or not isinstance(estimated, int)
            or isinstance(estimated, bool)
            or estimated <= 0
        ):
            raise ValueError(
                f"{self.path}:{line_number}: invalid started-attempt values"
            )
        for name in (
            "model_requested",
            "idempotency_key",
            "client_request_id",
        ):
            if not isinstance(row.get(name), str) or not row[name]:
                raise ValueError(
                    f"{self.path}:{line_number}: invalid {name}"
                )
        _validate_timestamp(row.get("started_at"), "attempt started_at")
        if attempt_id in self.starts or attempt_id in self.settlements:
            raise ValueError(
                f"{self.path}:{line_number}: duplicate attempt identity"
            )
        expected_ordinal = self._ordinals.get(request_id, 0) + 1
        if ordinal != expected_ordinal:
            raise ValueError(
                f"{self.path}:{line_number}: nonsequential attempt ordinal"
            )
        binding = {
            key: row[key]
            for key in allowed
            if key not in {"schema_version", "kind", "event", "attempt_id"}
        }
        if attempt_id != _digest(binding):
            raise ValueError(
                f"{self.path}:{line_number}: attempt digest mismatch"
            )
        self.starts[attempt_id] = row
        self._ordinals[request_id] = ordinal

    def _parse_settlement(
        self,
        row: dict[str, Any],
        *,
        line_number: int,
    ) -> None:
        allowed = {
            "schema_version",
            "kind",
            "event",
            "attempt_id",
            "settled_at",
            "outcome",
            "http_status",
            "charged_tokens",
            "server_request_id",
            "response_body_sha256",
            "response_record",
            "provider_audit",
        }
        if set(row) != allowed:
            raise ValueError(
                f"{self.path}:{line_number}: invalid settlement fields"
            )
        attempt_id = _validate_digest(row.get("attempt_id"), "attempt_id")
        if attempt_id not in self.starts:
            raise ValueError(
                f"{self.path}:{line_number}: settlement lacks its start"
            )
        if attempt_id in self.settlements:
            raise ValueError(
                f"{self.path}:{line_number}: duplicate attempt settlement"
            )
        _validate_timestamp(row.get("settled_at"), "attempt settled_at")
        outcome = row.get("outcome")
        if outcome not in _ATTEMPT_OUTCOMES:
            raise ValueError(
                f"{self.path}:{line_number}: invalid attempt outcome"
            )
        status = row.get("http_status")
        if status is not None and (
            not isinstance(status, int)
            or isinstance(status, bool)
            or not 100 <= status <= 599
        ):
            raise ValueError(
                f"{self.path}:{line_number}: invalid HTTP status"
            )
        charged = row.get("charged_tokens")
        if (
            not isinstance(charged, int)
            or isinstance(charged, bool)
            or charged < 0
        ):
            raise ValueError(
                f"{self.path}:{line_number}: invalid charged token count"
            )
        response_digest = row.get("response_body_sha256")
        if response_digest is not None:
            _validate_digest(response_digest, "response_body_sha256")
        server_request_id = row.get("server_request_id")
        if server_request_id is not None and not isinstance(
            server_request_id,
            str,
        ):
            raise ValueError(
                f"{self.path}:{line_number}: invalid server request ID"
            )
        response_record = row.get("response_record")
        if response_record is not None and not isinstance(
            response_record,
            Mapping,
        ):
            raise ValueError(
                f"{self.path}:{line_number}: invalid response record"
            )
        audit = row.get("provider_audit")
        if audit is not None and not isinstance(audit, Mapping):
            raise ValueError(
                f"{self.path}:{line_number}: invalid embedded audit"
            )
        if outcome in {"success", "model_mismatch"}:
            if not isinstance(audit, Mapping):
                raise ValueError(
                    f"{self.path}:{line_number}: final attempt lacks audit"
                )
            expected_status = (
                "accepted"
                if outcome == "success"
                else "rejected_model_mismatch"
            )
            if audit.get("acceptance_status") != expected_status:
                raise ValueError(
                    f"{self.path}:{line_number}: attempt/audit status mismatch"
                )
        elif audit is not None:
            raise ValueError(
                f"{self.path}:{line_number}: nonfinal attempt has an audit"
            )
        if charged > self.starts[attempt_id]["estimated_max_tokens"]:
            raise ValueError(
                f"{self.path}:{line_number}: charged token count exceeds "
                "the conservative reservation"
            )
        self.settlements[attempt_id] = row

    def start(
        self,
        request: NativeActionRequest,
        prepared: PreparedNativeActionRequest,
        *,
        model_requested: str,
        started_at: str,
    ) -> str:
        if self.unresolved_attempt_ids:
            raise NativeActionManualReviewRequired(
                "transport attempt journal has an unresolved started attempt; "
                "manual review is required before retry"
            )
        ordinal = self._ordinals.get(request.request_id, 0) + 1
        binding = {
            "collection_plan_sha256": self.collection_plan_sha256,
            "collection_config_sha256": self.collection_config_sha256,
            "request_id": request.request_id,
            "prompt_sha256": request.prompt_sha256,
            "native_state_id": request.native_state_id,
            "suite_sha256": request.suite.suite_sha256,
            "request_body_sha256": prepared.body_sha256,
            "model_requested": model_requested,
            "idempotency_key": prepared.idempotency_key,
            "client_request_id": prepared.client_request_id,
            "estimated_max_tokens": prepared.estimated_max_tokens,
            "attempt_ordinal": ordinal,
            "started_at": started_at,
        }
        attempt_id = _digest(binding)
        record = {
            "schema_version": 1,
            "kind": NATIVE_ACTION_ATTEMPT_KIND,
            "event": "started",
            "attempt_id": attempt_id,
            **binding,
        }
        self._parse_start(dict(record), line_number=0)
        try:
            _append_jsonl(self.path, record)
        except Exception:
            self.starts.pop(attempt_id, None)
            self._ordinals[request.request_id] = ordinal - 1
            raise
        return attempt_id

    def settle(
        self,
        attempt_id: str,
        *,
        settled_at: str,
        outcome: str,
        http_status: int | None,
        charged_tokens: int,
        server_request_id: str | None,
        response_body_sha256: str | None,
        response_record: Mapping[str, Any] | None,
        provider_audit: Mapping[str, Any] | None,
    ) -> None:
        record = {
            "schema_version": 1,
            "kind": NATIVE_ACTION_ATTEMPT_KIND,
            "event": "settled",
            "attempt_id": attempt_id,
            "settled_at": settled_at,
            "outcome": outcome,
            "http_status": http_status,
            "charged_tokens": charged_tokens,
            "server_request_id": server_request_id,
            "response_body_sha256": response_body_sha256,
            "response_record": (
                dict(response_record)
                if response_record is not None
                else None
            ),
            "provider_audit": (
                dict(provider_audit)
                if provider_audit is not None
                else None
            ),
        }
        shadow = dict(self.settlements)
        self._parse_settlement(dict(record), line_number=0)
        try:
            _append_jsonl(self.path, record)
        except Exception:
            self.settlements = shadow
            raise

    def validate_bindings(
        self,
        request_by_id: Mapping[str, NativeActionRequest],
        provider: "OpenAINativeActionProvider",
    ) -> None:
        for attempt_id, start in self.starts.items():
            request = request_by_id.get(start["request_id"])
            if request is None:
                raise ValueError(
                    "attempt journal references an unexpected request"
                )
            prepared = provider.prepare(request)
            expected = {
                "collection_plan_sha256": self.collection_plan_sha256,
                "collection_config_sha256": self.collection_config_sha256,
                "prompt_sha256": request.prompt_sha256,
                "native_state_id": request.native_state_id,
                "suite_sha256": request.suite.suite_sha256,
                "request_body_sha256": prepared.body_sha256,
                "model_requested": provider.config.model,
                "idempotency_key": prepared.idempotency_key,
                "client_request_id": prepared.client_request_id,
                "estimated_max_tokens": prepared.estimated_max_tokens,
            }
            if any(start.get(name) != value for name, value in expected.items()):
                raise ValueError(
                    "attempt journal does not match the current collection plan"
                )
            settlement = self.settlements.get(attempt_id)
            if settlement is None:
                continue
            if settlement["outcome"] in {
                "transport_error",
                "http_error",
                "invalid_response",
            } and settlement["charged_tokens"] != prepared.estimated_max_tokens:
                raise ValueError(
                    "attempt journal has a nonconservative failed charge"
                )
            audit = settlement.get("provider_audit")
            if isinstance(audit, Mapping):
                _validate_audit_record(
                    audit,
                    request,
                    prepared,
                    provider.config,
                )
                if (
                    settlement.get("response_body_sha256")
                    != audit.get("raw_response_sha256")
                ):
                    raise ValueError(
                        "attempt journal response/audit digest mismatch"
                    )
                expected_charge = _usage_total(audit["usage"])
                if expected_charge is None:
                    expected_charge = prepared.estimated_max_tokens
                if settlement["charged_tokens"] != expected_charge:
                    raise ValueError(
                        "attempt journal charge/audit usage mismatch"
                    )
                if (
                    settlement.get("server_request_id")
                    != audit.get("server_request_id")
                    or settlement.get("response_record")
                    != audit.get("raw_response")
                    or not isinstance(settlement.get("http_status"), int)
                    or not 200 <= settlement["http_status"] <= 299
                ):
                    raise ValueError(
                        "attempt journal response metadata/audit mismatch"
                    )

    def accounting(self) -> tuple[int, int]:
        return (
            len(self.settlements),
            sum(
                int(row["charged_tokens"])
                for row in self.settlements.values()
            ),
        )

    def embedded_final_audits(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for attempt_id, settlement in self.settlements.items():
            audit = settlement.get("provider_audit")
            if not isinstance(audit, Mapping):
                continue
            request_id = self.starts[attempt_id]["request_id"]
            if request_id in result:
                raise ValueError(
                    "attempt journal contains multiple final audits for one request"
                )
            result[request_id] = dict(audit)
        return result

    def requests_without_final_audit(self) -> tuple[str, ...]:
        final = set(self.embedded_final_audits())
        attempted = {
            self.starts[attempt_id]["request_id"]
            for attempt_id in self.settlements
        }
        return tuple(sorted(attempted - final))


class OpenAINativeActionProvider:
    """Synchronous, budgeted native-action client for the Responses API."""

    def __init__(
        self,
        config: OpenAIProviderConfig,
        *,
        transport: HTTPTransport = urllib_transport,
        sleep: Callable[[float], None] = time.sleep,
        random_value: Callable[[], float] = random.random,
        epoch_time: Callable[[], float] = time.time,
    ) -> None:
        self.config = config
        self._transport = transport
        self._sleep = sleep
        self._random_value = random_value
        self._epoch_time = epoch_time
        self.budget = ExecutionBudget(
            max_requests=config.max_requests,
            max_total_tokens=config.max_total_tokens,
        )

    def prepare(
        self,
        request: NativeActionRequest,
    ) -> PreparedNativeActionRequest:
        return prepare_openai_native_action_request(request, self.config)

    def restore_budget(self, *, request_count: int, total_tokens: int) -> None:
        self.budget.restore(
            request_count=request_count,
            total_tokens=total_tokens,
        )

    def _backoff(self, attempt: int, retry_after: str | None) -> float:
        base = min(
            self.config.max_backoff_seconds,
            self.config.initial_backoff_seconds * (2 ** (attempt - 1)),
        )
        unit = min(1.0, max(0.0, float(self._random_value())))
        jittered = base * (
            1 + self.config.jitter_fraction * ((2 * unit) - 1)
        )
        declared = _retry_after_seconds(
            retry_after,
            now_epoch=self._epoch_time(),
        )
        return min(
            self.config.max_backoff_seconds,
            max(
                min(self.config.max_backoff_seconds, max(0.0, jittered)),
                declared or 0.0,
            ),
        )

    def complete(
        self,
        request: NativeActionRequest,
        *,
        attempt_ledger: _DurableAttemptLedger | None = None,
    ) -> NativeActionProviderResult:
        if not self.config.live_execution:
            raise LiveExecutionRequired(
                "native action execution requires explicit live authorization"
            )
        prepared = self.prepare(request)
        secret: str | None = None
        headers: dict[str, str] | None = None
        started_at = _utc_timestamp(self._epoch_time())
        for attempt in range(1, self.config.max_retries + 2):
            self.budget.reserve(prepared.estimated_max_tokens)
            if secret is None:
                loaded = os.environ.get(self.config.api_key_env)
                if not isinstance(loaded, str) or not loaded.strip():
                    self.budget.rollback()
                    raise MissingAPIKey(
                        f"missing API key in {self.config.api_key_env}"
                    )
                loaded = loaded.strip()
                if "\r" in loaded or "\n" in loaded:
                    self.budget.rollback()
                    raise MissingAPIKey(
                        f"{self.config.api_key_env} contains an invalid newline"
                    )
                secret = loaded
                headers = dict(prepared.headers)
                headers["Authorization"] = "Bearer " + secret
            attempt_started_at = _utc_timestamp(self._epoch_time())
            try:
                attempt_id = (
                    attempt_ledger.start(
                        request,
                        prepared,
                        model_requested=self.config.model,
                        started_at=attempt_started_at,
                    )
                    if attempt_ledger is not None
                    else None
                )
            except Exception:
                self.budget.rollback()
                raise

            def settle(
                *,
                outcome: str,
                http_status: int | None = None,
                server_request_id: str | None = None,
                response_body_sha256: str | None = None,
                response_record: Mapping[str, Any] | None = None,
                charged_tokens: int | None = None,
                provider_audit: Mapping[str, Any] | None = None,
            ) -> None:
                charged = (
                    prepared.estimated_max_tokens
                    if charged_tokens is None
                    else charged_tokens
                )
                if charged > prepared.estimated_max_tokens:
                    raise ProviderResponseError(
                        "provider token usage exceeds the conservative "
                        "reservation; manual review is required"
                    )
                if attempt_ledger is not None:
                    if attempt_id is None:
                        raise RuntimeError("attempt journal identity is missing")
                    attempt_ledger.settle(
                        attempt_id,
                        settled_at=_utc_timestamp(self._epoch_time()),
                        outcome=outcome,
                        http_status=http_status,
                        charged_tokens=charged,
                        server_request_id=server_request_id,
                        response_body_sha256=response_body_sha256,
                        response_record=response_record,
                        provider_audit=provider_audit,
                    )
                self.budget.commit(charged)

            try:
                if headers is None or secret is None:
                    raise RuntimeError("live credential was not initialized")
                http = self._transport(
                    url=prepared.endpoint,
                    body=prepared.body_bytes,
                    headers=headers,
                    timeout=self.config.timeout_seconds,
                )
            except HTTPResponseBodyTooLarge as exc:
                settle(
                    outcome="invalid_response",
                    http_status=exc.status,
                )
                raise ProviderResponseError(str(exc)) from exc
            except (TimeoutError, ConnectionError, OSError) as exc:
                settle(
                    outcome="transport_error",
                    response_record={
                        "error_type": type(exc).__name__,
                    },
                )
                # No provider idempotency guarantee is available for this
                # action request. A connection failure may have happened
                # after acceptance and billing, so repeating it would risk a
                # duplicate paid action with an unknowable first result.
                raise NativeActionManualReviewRequired(
                    "native action transport outcome is ambiguous; automatic "
                    "retry is disabled and manual review is required; "
                    f"error_type={type(exc).__name__}"
                ) from exc

            lowered = _lower_headers(http.headers)
            safe_headers = _redact_provider_value(lowered, secret)
            if not isinstance(safe_headers, Mapping):
                safe_headers = {}
            server_request_id = _safe_provider_identifier(
                safe_headers.get("x-request-id"),
                secret=secret,
            )
            response_digest = sha256(http.body).hexdigest()
            if not 200 <= http.status <= 299:
                safe_message = _safe_error(http.body, secret)
                settle(
                    outcome="http_error",
                    http_status=http.status,
                    server_request_id=server_request_id,
                    response_body_sha256=response_digest,
                    response_record={"error": safe_message},
                )
                transient = (
                    http.status in {408, 409, 429}
                    or 500 <= http.status <= 599
                )
                if transient and attempt <= self.config.max_retries:
                    self._sleep(
                        self._backoff(
                            attempt,
                            lowered.get("retry-after"),
                        )
                    )
                    continue
                raise ProviderHTTPError(
                    status=http.status,
                    message=safe_message,
                    client_request_id=prepared.client_request_id,
                    server_request_id=server_request_id,
                )

            safe_raw: Mapping[str, Any] | None = None
            try:
                try:
                    raw = json.loads(http.body.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ProviderResponseError(
                        "OpenAI native action response is not valid JSON"
                    ) from exc
                if not isinstance(raw, Mapping):
                    raise ProviderResponseError(
                        "OpenAI native action response must be an object"
                    )
                safe_raw = _redact_provider_value(raw, secret)
                if not isinstance(safe_raw, Mapping):
                    raise ProviderResponseError(
                        "OpenAI native action response must be an object"
                    )
                safe_headers = _redact_provider_value(lowered, secret)
                if not isinstance(safe_headers, Mapping):
                    safe_headers = {}
                payload = _extract_actions_payload(safe_raw)
                actions = _parse_actions(request, payload)
                returned_model = safe_raw.get("model")
                if (
                    not isinstance(returned_model, str)
                    or not returned_model.strip()
                ):
                    returned_model = "<missing>"
                else:
                    returned_model = returned_model.strip()
                response_id = safe_raw.get("id")
                if not isinstance(response_id, str) or not response_id.strip():
                    raise ProviderResponseError(
                        "OpenAI native action response has no response ID"
                    )
                usage = safe_raw.get("usage")
                if not isinstance(usage, Mapping):
                    usage = {}
                completed_at = _utc_timestamp(self._epoch_time())
                raw_sha256 = sha256(http.body).hexdigest()
                record_identity = _digest(
                    {
                        "request_id": request.request_id,
                        "provider_response_id": response_id,
                        "raw_response_sha256": raw_sha256,
                    }
                )
                record = NativeTerminalActionRecord.build(
                    record_id=f"openai-native-action:{record_identity}",
                    trajectory_id=request.trajectory_id,
                    domain_id=request.domain_id,
                    updater_id=request.updater_id,
                    native_state_id=request.native_state_id,
                    native_system_id=NATIVE_ACTION_SYSTEM_ID,
                    native_system_version=(
                        f"openai-responses:{returned_model}"
                    ),
                    suite_id=request.suite.suite_id,
                    suite_sha256=request.suite.suite_sha256,
                    action_execution_mode="recorded_live",
                    execution_trace_sha256=raw_sha256,
                    recorded_at=completed_at,
                    actions=actions,
                )
                result = NativeActionProviderResult(
                    request=request,
                    record=record,
                    model_requested=self.config.model,
                    model_returned=returned_model,
                    provider_response_id=response_id,
                    usage=dict(usage),
                    started_at=started_at,
                    completed_at=completed_at,
                    attempts=attempt,
                    request_body_sha256=prepared.body_sha256,
                    idempotency_key=prepared.idempotency_key,
                    client_request_id=prepared.client_request_id,
                    server_request_id=server_request_id,
                    estimated_max_tokens=prepared.estimated_max_tokens,
                    raw_response_sha256=raw_sha256,
                    raw_response=dict(safe_raw),
                )
            except (TypeError, ValueError, ProviderResponseError):
                if safe_raw is None:
                    excerpt = " ".join(
                        http.body.decode("utf-8", errors="replace")
                        .replace(secret, "[redacted]")
                        .split()
                    )[:500]
                    response_record: Mapping[str, Any] = {
                        "body_excerpt": excerpt,
                    }
                else:
                    response_record = dict(safe_raw)
                settle(
                    outcome="invalid_response",
                    http_status=http.status,
                    server_request_id=server_request_id,
                    response_body_sha256=response_digest,
                    response_record=response_record,
                )
                raise

            mismatch = not returned_model_is_consistent(
                result.model_requested,
                result.model_returned,
            )
            audit = result.to_audit_record(
                acceptance_status=(
                    "rejected_model_mismatch" if mismatch else "accepted"
                )
            )
            settle(
                outcome="model_mismatch" if mismatch else "success",
                http_status=http.status,
                server_request_id=server_request_id,
                response_body_sha256=response_digest,
                response_record=dict(result.raw_response),
                charged_tokens=_usage_total(result.usage),
                provider_audit=audit,
            )
            if mismatch:
                raise NativeActionModelMismatch(result)
            return result

        raise OpenAIProviderError(
            "native action execution ended without a response"
        )


def _build_collection_plan(
    run_dir: str | Path,
    config: OpenAIProviderConfig,
    requests: Sequence[NativeActionRequest],
    prepared: Sequence[PreparedNativeActionRequest],
) -> dict[str, Any]:
    """Build the canonical credential-free plan used by plan and execute."""

    run = Path(run_dir).resolve()
    source_manifest = json.loads(
        (run / "manifest.json").read_text(encoding="utf-8")
    )
    if (
        not isinstance(source_manifest, Mapping)
        or not isinstance(source_manifest.get("run_id"), str)
        or not source_manifest["run_id"]
    ):
        raise ValueError("source run manifest lacks a valid run_id")
    conservative = sum(item.estimated_max_tokens for item in prepared)
    theoretical_attempts = len(requests) * (config.max_retries + 1)
    theoretical_tokens = conservative * (config.max_retries + 1)
    collection_config = _collection_config(config)
    plan: dict[str, Any] = {
        "schema_version": 1,
        "kind": "native-action-collection-plan",
        "workflow": "native_terminal_actions",
        "live_execution": False,
        "credential_read": False,
        "source_run_id": source_manifest["run_id"],
        "source_run_manifest_sha256": sha256(
            (run / "manifest.json").read_bytes()
        ).hexdigest(),
        "source_run_checksums_sha256": sha256(
            (run / "SHA256SUMS").read_bytes()
        ).hexdigest(),
        "native_system_id": NATIVE_ACTION_SYSTEM_ID,
        "model": config.model,
        "reasoning_effort": config.reasoning_effort,
        "endpoint": config.endpoint,
        "base_url": config.base_url,
        "api_key_env": config.api_key_env,
        "allow_custom_base_url": config.allow_custom_base_url,
        "official_origin_locked": collection_config[
            "official_origin_locked"
        ],
        "max_output_tokens": config.max_output_tokens,
        "timeout_seconds": config.timeout_seconds,
        "max_retries": config.max_retries,
        "initial_backoff_seconds": config.initial_backoff_seconds,
        "max_backoff_seconds": config.max_backoff_seconds,
        "jitter_fraction": config.jitter_fraction,
        "request_count": len(requests),
        "conservative_max_tokens": conservative,
        "initial_transport_attempt_count": len(requests),
        "maximum_attempts_per_request": config.max_retries + 1,
        "theoretical_max_transport_attempts": theoretical_attempts,
        "theoretical_max_tokens_with_all_retries": theoretical_tokens,
        "all_retry_attempts_within_declared_budget": (
            theoretical_attempts <= config.max_requests
            and theoretical_tokens <= config.max_total_tokens
        ),
        "budget_accounting_unit": "actual_transport_attempt",
        "max_requests": config.max_requests,
        "max_total_tokens": config.max_total_tokens,
        "within_declared_budget": (
            len(requests) <= config.max_requests
            and conservative <= config.max_total_tokens
        ),
        "collection_config": collection_config,
        "collection_config_sha256": _digest(collection_config),
        "request_bindings": [
            {
                "request_id": request.request_id,
                "trajectory_id": request.trajectory_id,
                "native_state_id": request.native_state_id,
                "suite_sha256": request.suite.suite_sha256,
                "request_body_sha256": item.body_sha256,
                "idempotency_key": item.idempotency_key,
                "client_request_id": item.client_request_id,
                "estimated_max_tokens": item.estimated_max_tokens,
            }
            for request, item in zip(requests, prepared)
        ],
        "official_references": [
            OPENAI_NATIVE_ACTION_REFERENCE,
            OPENAI_STRUCTURED_OUTPUT_REFERENCE,
        ],
        "resolved_on": MODEL_SELECTION_RESOLVED_ON,
    }
    plan["plan_sha256"] = _digest(plan)
    return plan


def plan_openai_native_actions(
    run_dir: str | Path,
    config: OpenAIProviderConfig,
) -> dict[str, Any]:
    """Return the exact keyless request/token plan for a source run."""

    if config.live_execution:
        raise ValueError("planning config must disable live_execution")
    requests = build_native_action_requests(run_dir)
    provider = OpenAINativeActionProvider(config)
    prepared = tuple(provider.prepare(request) for request in requests)
    return _build_collection_plan(run_dir, config, requests, prepared)


def _audit_rows(path: Path) -> tuple[dict[str, Any], ...]:
    if not path.exists() or path.stat().st_size == 0:
        return ()
    return _read_jsonl(path)


def execute_openai_native_actions(
    run_dir: str | Path,
    output_dir: str | Path,
    provider: OpenAINativeActionProvider,
) -> dict[str, Any]:
    """Collect native actions under an exclusive output-directory lock."""

    if not provider.config.live_execution:
        raise LiveExecutionRequired(
            "native action execution requires explicit live authorization"
        )
    run = Path(run_dir).resolve()
    output = Path(output_dir).resolve()
    if output == run or run in output.parents:
        raise ValueError(
            "native action output cannot equal or be inside the verified "
            "source run"
        )
    requests = build_native_action_requests(run)
    if not requests:
        raise ValueError(
            "native action collection has no eligible retained requests"
        )
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / ".collection.lock"
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    locked = False
    try:
        locked = try_file_lock(descriptor)
        if not locked:
            raise OpenAIProviderError(
                "another native-action collector holds the output lock"
            )
        return _execute_openai_native_actions_locked(
            run,
            output,
            provider,
            requests=requests,
        )
    finally:
        try:
            if locked:
                unlock_file(descriptor)
        finally:
            os.close(descriptor)


def _execute_openai_native_actions_locked(
    run_dir: str | Path,
    output_dir: str | Path,
    provider: OpenAINativeActionProvider,
    *,
    requests: Sequence[NativeActionRequest] | None = None,
) -> dict[str, Any]:
    """Collect or resume audit-first native actions for one verified B run."""

    if not provider.config.live_execution:
        raise LiveExecutionRequired(
            "native action execution requires explicit live authorization"
        )
    material = (
        build_native_action_requests(run_dir)
        if requests is None
        else tuple(requests)
    )
    if not material:
        raise ValueError(
            "native action collection has no eligible retained requests"
        )
    prepared_rows = tuple(provider.prepare(request) for request in material)
    prepared = {
        request.request_id: item
        for request, item in zip(material, prepared_rows)
    }
    if len(material) > provider.config.max_requests or sum(
        item.estimated_max_tokens for item in prepared.values()
    ) > provider.config.max_total_tokens:
        raise ValueError(
            "native action corpus exceeds the declared hard budget"
        )
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    request_path = output / "requests.jsonl"
    plan_path = output / "collection-plan.json"
    attempt_path = output / "transport-attempts.jsonl"
    audit_path = output / "provider-audit.jsonl"
    action_path = output / "native-actions.jsonl"
    manifest_path = output / "execution-manifest.json"
    plan = _build_collection_plan(
        run_dir,
        provider.config,
        material,
        prepared_rows,
    )
    if plan_path.exists():
        try:
            existing_plan = json.loads(plan_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                "existing native action collection plan is invalid"
            ) from exc
        if not isinstance(existing_plan, Mapping) or dict(existing_plan) != plan:
            raise ValueError(
                "existing native action collection plan has a different "
                "request, origin, credential-name, model, retry, or budget "
                "identity"
            )
    else:
        _atomic_write(plan_path, plan)

    _repair_trailing_jsonl(attempt_path)
    _repair_trailing_jsonl(audit_path)
    _repair_trailing_jsonl(action_path)

    request_bytes = "".join(
        canonical_json(request.to_dict()) + "\n" for request in material
    ).encode("utf-8")
    if request_path.exists():
        if request_path.read_bytes() != request_bytes:
            raise ValueError(
                "existing native action requests have a different identity"
            )
    else:
        _atomic_write_bytes(request_path, request_bytes)

    request_by_id = {request.request_id: request for request in material}
    attempt_ledger = _DurableAttemptLedger(
        attempt_path,
        collection_plan_sha256=plan["plan_sha256"],
        collection_config_sha256=plan["collection_config_sha256"],
    )
    attempt_ledger.validate_bindings(request_by_id, provider)
    if attempt_ledger.unresolved_attempt_ids:
        raise NativeActionManualReviewRequired(
            "transport attempt journal contains an unresolved durable started "
            "attempt; billing outcome is unknown and manual review is required"
        )
    nonfinal_attempts = attempt_ledger.requests_without_final_audit()
    if nonfinal_attempts:
        raise NativeActionManualReviewRequired(
            "transport attempts exist without an embedded accepted/rejected "
            "final audit; manual review is required before retry: "
            + repr(nonfinal_attempts)
        )

    audits_by_id: dict[str, dict[str, Any]] = {}
    for row in _audit_rows(audit_path):
        request_id = row.get("request_id")
        if not isinstance(request_id, str) or request_id in audits_by_id:
            raise ValueError("duplicate or invalid native action audit identity")
        audits_by_id[request_id] = row
    embedded_audits = attempt_ledger.embedded_final_audits()
    for request_id, embedded in embedded_audits.items():
        retained = audits_by_id.get(request_id)
        if retained is None:
            _append_jsonl(audit_path, embedded)
            audits_by_id[request_id] = embedded
        elif retained != embedded:
            raise ValueError(
                "transport-attempt/final native action audit mismatch"
            )
    audits_without_attempts = set(audits_by_id) - set(embedded_audits)
    if audits_without_attempts:
        raise ValueError(
            "native action final audits lack durable transport-attempt "
            "records: " + repr(sorted(audits_without_attempts))
        )

    accepted: dict[str, dict[str, Any]] = {}
    accepted_records: dict[str, NativeTerminalActionRecord] = {}
    for request_id, row in audits_by_id.items():
        if request_id not in request_by_id:
            raise ValueError(
                "native action audit references an unexpected request"
            )
        record = _validate_audit_record(
            row,
            request_by_id[request_id],
            prepared[request_id],
            provider.config,
        )
        if row["acceptance_status"] != "accepted":
            raise ValueError(
                "native action journal contains a rejected charged response; "
                "preserve it and start a reviewed recovery directory"
            )
        accepted[request_id] = row
        accepted_records[request_id] = record
    restored_attempts, restored_tokens = attempt_ledger.accounting()
    provider.restore_budget(
        request_count=restored_attempts,
        total_tokens=restored_tokens,
    )

    existing_records = (
        read_native_terminal_action_records(action_path)
        if action_path.exists() and action_path.stat().st_size
        else ()
    )
    records_by_trajectory = {
        record.trajectory_id: record for record in existing_records
    }
    if len(records_by_trajectory) != len(existing_records):
        raise ValueError("duplicate resumed native action trajectories")
    for request_id, record in accepted_records.items():
        existing = records_by_trajectory.get(record.trajectory_id)
        if existing is None:
            _append_jsonl(action_path, record.to_dict())
            records_by_trajectory[record.trajectory_id] = record
        elif existing.to_dict() != record.to_dict():
            raise ValueError(
                "resumed native action differs from accepted audit"
            )
    audited_trajectories = {
        request_by_id[request_id].trajectory_id for request_id in accepted
    }
    if set(records_by_trajectory) - audited_trajectories:
        raise ValueError("native action record exists without an accepted audit")

    reused = len(accepted)
    for request in material:
        if request.request_id in accepted:
            continue
        try:
            result = provider.complete(
                request,
                attempt_ledger=attempt_ledger,
            )
        except NativeActionModelMismatch as exc:
            rejected = exc.result.to_audit_record(
                acceptance_status="rejected_model_mismatch"
            )
            _append_jsonl(
                audit_path,
                rejected,
            )
            raise
        audit = result.to_audit_record()
        _append_jsonl(audit_path, audit)
        _append_jsonl(action_path, result.record.to_dict())
        accepted[request.request_id] = audit
        records_by_trajectory[result.record.trajectory_id] = result.record

    action_bytes = action_path.read_bytes()
    audit_bytes = audit_path.read_bytes()
    attempt_bytes = attempt_path.read_bytes()
    plan_bytes = plan_path.read_bytes()
    manifest = {
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
            f"openai-responses:{provider.config.model}"
        ),
        "model": provider.config.model,
        "reasoning_effort": provider.config.reasoning_effort,
        "request_count": len(material),
        "action_record_count": len(records_by_trajectory),
        "reused_request_count": reused,
        "new_request_count": len(material) - reused,
        "request_budget_used": provider.budget.request_count,
        "transport_attempt_count": provider.budget.request_count,
        "token_budget_used": provider.budget.total_tokens,
        "budget_accounting_unit": "actual_transport_attempt",
        "max_requests": provider.config.max_requests,
        "max_total_tokens": provider.config.max_total_tokens,
        "credentials_retained": False,
        "collection_plan_file": plan_path.name,
        "collection_plan_sha256": plan["plan_sha256"],
        "collection_plan_file_sha256": sha256(plan_bytes).hexdigest(),
        "collection_config": plan["collection_config"],
        "collection_config_sha256": plan["collection_config_sha256"],
        "requests_sha256": sha256(request_bytes).hexdigest(),
        "transport_attempts_sha256": sha256(attempt_bytes).hexdigest(),
        "provider_audit_sha256": sha256(audit_bytes).hexdigest(),
        "native_actions_sha256": sha256(action_bytes).hexdigest(),
        "transport_attempts_file": attempt_path.name,
        "native_actions_file": action_path.name,
        "provider_audit_file": audit_path.name,
        "official_references": [
            OPENAI_NATIVE_ACTION_REFERENCE,
            OPENAI_STRUCTURED_OUTPUT_REFERENCE,
        ],
        "resolved_on": MODEL_SELECTION_RESOLVED_ON,
    }
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(existing, Mapping):
            raise ValueError("existing native action manifest is invalid")
        comparable_existing = dict(existing)
        comparable_manifest = dict(manifest)
        for payload in (comparable_existing, comparable_manifest):
            payload.pop("reused_request_count", None)
            payload.pop("new_request_count", None)
        if comparable_existing != comparable_manifest:
            raise ValueError(
                "existing native action manifest has a different identity"
            )
    _atomic_write(manifest_path, manifest)
    return {
        "output_dir": str(output),
        "native_actions": str(action_path),
        "provider_audit": str(audit_path),
        "transport_attempts": str(attempt_path),
        "collection_plan": str(plan_path),
        "request_count": len(material),
        "transport_attempt_count": provider.budget.request_count,
        "reused_request_count": reused,
        "new_request_count": len(material) - reused,
        "claim_status": "not_claimed",
    }
