"""Dependency-free OpenAI Responses API execution for CAPE-Loop.

This module deliberately sits outside the provider-neutral replay core.  It
turns :class:`~cape_loop.llm_exchange.LLMRequest` objects into OpenAI Responses
API calls and writes ordinary ``LLMResponse`` JSONL that the existing replay
path can consume.

No credential is read at import time or while preparing a dry-run request.
Live execution must be explicitly enabled, and the API key is loaded only from
the configured environment-variable name immediately before an HTTP request.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import (
    HTTPRedirectHandler,
    Request,
    build_opener,
)
import json
import math
import os
import random
import re
import time

from .llm_exchange import (
    ATTRIBUTES,
    VALUES,
    VIEWS,
    LLMRequest,
    LLMResponse,
    read_responses,
)
from .provider_attempts import (
    DurableProviderAttemptLedger,
    ExclusiveProviderExecutionLock,
    ProviderAttemptManualReviewRequired,
    append_jsonl as append_provider_jsonl,
    default_attempt_path,
    repair_trailing_jsonl as repair_attempt_journal,
)


_ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_NON_OPENAI_API_KEY_ENVS = frozenset(
    {
        "OPENROUTER_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
    }
)
_TRANSIENT_STATUSES = frozenset({408, 409, 429})
_RESPONSE_SCHEMA_NAME = "cape_loop_preference_beliefs"
_REASONING_EFFORTS = frozenset(
    {"none", "low", "medium", "high", "xhigh", "max"}
)
# Provider output limits are request hints, not wire-level guarantees. Keep a
# generous fixed ceiling so an invalid endpoint or oversized error cannot make
# the dependency-free transport retain an unbounded response in memory.
HTTP_RESPONSE_BODY_LIMIT_BYTES = 16 * 1024 * 1024


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_nonnegative_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0
    )


def _utc_timestamp(epoch_seconds: float) -> str:
    return (
        datetime.fromtimestamp(epoch_seconds, timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


@dataclass(frozen=True, slots=True)
class OpenAIModelRole:
    """One declared role in the default paper evaluation suite."""

    role: str
    model: str
    reasoning_effort: str
    purpose: str


DEFAULT_OPENAI_MODEL_ROLES: Mapping[str, OpenAIModelRole] = MappingProxyType(
    {
        "primary": OpenAIModelRole(
            role="primary",
            model="gpt-5.6-sol",
            reasoning_effort="medium",
            purpose=(
                "Confirmatory profile-writer evaluation; use the same model "
                "for every updater information view."
            ),
        ),
        "replication": OpenAIModelRole(
            role="replication",
            model="gpt-5.6-terra",
            reasoning_effort="medium",
            purpose=(
                "Cost-balanced GPT-5.6 model-variant/tier replication; again "
                "hold the model fixed across updater information views. This "
                "is not distinct-family robustness."
            ),
        ),
        "decoder": OpenAIModelRole(
            role="decoder",
            model="gpt-5.6-luna",
            reasoning_effort="low",
            purpose=(
                "High-volume blinded decoding or pilot work; do not substitute "
                "it for the primary writer in within-model causal comparisons."
            ),
        ),
    }
)


class OpenAIProviderError(RuntimeError):
    """Base class for safe, credential-free provider errors."""


class LiveExecutionRequired(OpenAIProviderError):
    """Raised when a network call is attempted without explicit opt-in."""


class MissingAPIKey(OpenAIProviderError):
    """Raised when the configured credential environment variable is absent."""


class BudgetExceeded(OpenAIProviderError):
    """Raised before a request that would exceed a declared execution budget."""


class ProviderResponseError(OpenAIProviderError):
    """Raised for an invalid, incomplete, refused, or unparseable response."""


class HTTPResponseBodyTooLarge(OpenAIProviderError):
    """A response exceeded the fixed wire-body memory safety limit."""

    def __init__(self, *, status: int) -> None:
        self.status = status
        self.limit_bytes = HTTP_RESPONSE_BODY_LIMIT_BYTES
        super().__init__(
            "OpenAI response body exceeded the fixed "
            f"{self.limit_bytes}-byte safety limit"
        )


class ProviderResultRejected(ProviderResponseError):
    """A paid provider result that must be audited but not replayed."""

    acceptance_status = "rejected_provider_result"

    def __init__(
        self,
        result: Any,
        message: str,
        *,
        acceptance_status: str | None = None,
    ) -> None:
        self.result = result
        if acceptance_status is not None:
            self.acceptance_status = acceptance_status
        super().__init__(message)


class ProviderModelMismatch(ProviderResultRejected):
    """A completed call whose returned model is not the requested model."""

    acceptance_status = "rejected_model_mismatch"

    def __init__(self, result: Any) -> None:
        provider_label = getattr(result, "provider_label", "OpenAI")
        super().__init__(
            result,
            (
                f"{provider_label} returned a model inconsistent with the "
                "configured request: "
                f"requested={result.model_requested!r}, "
                f"returned={result.model_returned!r}; "
                f"client_request_id={result.client_request_id}"
            ),
        )


class ProviderHTTPError(OpenAIProviderError):
    """A non-retryable HTTP failure or an exhausted retry sequence."""

    def __init__(
        self,
        *,
        status: int,
        message: str,
        client_request_id: str,
        server_request_id: str | None,
    ) -> None:
        self.status = status
        self.client_request_id = client_request_id
        self.server_request_id = server_request_id
        suffix = f"; server_request_id={server_request_id}" if server_request_id else ""
        super().__init__(
            f"OpenAI request failed with HTTP {status}: {message}"
            f"; client_request_id={client_request_id}{suffix}"
        )


@dataclass(frozen=True, slots=True)
class OpenAIProviderConfig:
    """Validated configuration for one model-specific live executor."""

    model: str = DEFAULT_OPENAI_MODEL_ROLES["primary"].model
    reasoning_effort: str = DEFAULT_OPENAI_MODEL_ROLES["primary"].reasoning_effort
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str = "https://api.openai.com"
    allow_custom_base_url: bool = False
    timeout_seconds: float = 180.0
    max_retries: int = 4
    initial_backoff_seconds: float = 0.5
    max_backoff_seconds: float = 8.0
    jitter_fraction: float = 0.25
    max_output_tokens: int = 4096
    max_requests: int = 100
    max_total_tokens: int = 500_000
    live_execution: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be a non-empty string")
        if self.reasoning_effort not in _REASONING_EFFORTS:
            raise ValueError(
                "reasoning_effort must be one of "
                + ", ".join(sorted(_REASONING_EFFORTS))
            )
        if not _ENVIRONMENT_NAME.fullmatch(self.api_key_env):
            raise ValueError("api_key_env must be a valid environment-variable name")
        if self.api_key_env in _NON_OPENAI_API_KEY_ENVS:
            raise ValueError(
                "OpenAI requires an OpenAI or dedicated credential variable; "
                f"{self.api_key_env} is reserved for a different provider and "
                "must never be sent to OpenAI"
            )
        parsed = urlsplit(self.base_url)
        if (
            parsed.scheme != "https"
            or not parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("base_url must be an HTTPS origin or HTTPS path")
        if not isinstance(self.allow_custom_base_url, bool):
            raise ValueError("allow_custom_base_url must be Boolean")
        if not self.allow_custom_base_url and (
            parsed.hostname != "api.openai.com"
            or parsed.port is not None
            or parsed.path not in {"", "/"}
        ):
            raise ValueError(
                "base_url must be the official https://api.openai.com origin; "
                "set allow_custom_base_url=True only after reviewing where "
                "the credential will be sent"
            )
        if (
            self.allow_custom_base_url
            and parsed.hostname != "api.openai.com"
            and self.api_key_env == "OPENAI_API_KEY"
        ):
            raise ValueError(
                "a custom base_url requires a dedicated credential "
                "environment variable instead of OPENAI_API_KEY"
            )
        if not _is_nonnegative_number(self.timeout_seconds) or self.timeout_seconds == 0:
            raise ValueError("timeout_seconds must be positive and finite")
        if (
            not isinstance(self.max_retries, int)
            or isinstance(self.max_retries, bool)
            or self.max_retries < 0
        ):
            raise ValueError("max_retries must be a non-negative integer")
        for name, value in (
            ("initial_backoff_seconds", self.initial_backoff_seconds),
            ("max_backoff_seconds", self.max_backoff_seconds),
            ("jitter_fraction", self.jitter_fraction),
        ):
            if not _is_nonnegative_number(value):
                raise ValueError(f"{name} must be finite and non-negative")
        if self.initial_backoff_seconds > self.max_backoff_seconds:
            raise ValueError(
                "initial_backoff_seconds cannot exceed max_backoff_seconds"
            )
        if self.jitter_fraction > 1:
            raise ValueError("jitter_fraction cannot exceed 1")
        for name, value in (
            ("max_output_tokens", self.max_output_tokens),
            ("max_requests", self.max_requests),
            ("max_total_tokens", self.max_total_tokens),
        ):
            if not _is_positive_int(value):
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.live_execution, bool):
            raise ValueError("live_execution must be Boolean")

    @property
    def endpoint(self) -> str:
        return self.base_url.rstrip("/") + "/v1/responses"


def belief_json_schema() -> dict[str, Any]:
    """Return the strict Structured Outputs schema sent to the model."""

    probability_vector = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            value: {"type": "number", "minimum": 0, "maximum": 1}
            for value in VALUES
        },
        "required": list(VALUES),
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "beliefs": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    attribute: probability_vector for attribute in ATTRIBUTES
                },
                "required": list(ATTRIBUTES),
            }
        },
        "required": ["beliefs"],
    }


@dataclass(frozen=True, slots=True)
class PreparedOpenAIRequest:
    """A dry-run request with no authorization material."""

    endpoint: str
    body: Mapping[str, Any]
    body_bytes: bytes
    body_sha256: str
    headers: Mapping[str, str]
    idempotency_key: str
    client_request_id: str
    estimated_max_tokens: int


def prepare_openai_request(
    request: LLMRequest,
    config: OpenAIProviderConfig,
) -> PreparedOpenAIRequest:
    """Build a deterministic Responses API request without reading a key."""

    if request.view not in VIEWS:
        raise ValueError(f"unknown LLM view: {request.view}")
    body: dict[str, Any] = {
        "model": config.model,
        "instructions": request.system_instruction,
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": _canonical(request.payload),
                    }
                ],
            }
        ],
        "reasoning": {"effort": config.reasoning_effort},
        "text": {
            "format": {
                "type": "json_schema",
                "name": _RESPONSE_SCHEMA_NAME,
                "description": (
                    "CAPE-Loop posterior marginals over four values for each "
                    "of three latent preference attributes."
                ),
                "strict": True,
                "schema": belief_json_schema(),
            }
        },
        "max_output_tokens": config.max_output_tokens,
        "store": False,
        "metadata": {
            "cape_loop_prompt_sha256": request.prompt_sha256,
            "cape_loop_view": request.view,
        },
    }
    body_bytes = _canonical(body).encode("utf-8")
    body_digest = sha256(body_bytes).hexdigest()
    identity_digest = sha256(
        (
            "cape-loop-openai-v1\n"
            + request.request_id
            + "\n"
            + request.prompt_sha256
            + "\n"
            + config.model
            + "\n"
            + body_digest
        ).encode("utf-8")
    ).hexdigest()
    idempotency_key = "cape-loop-" + identity_digest
    client_request_id = "cape-loop-" + identity_digest
    safe_headers = MappingProxyType(
        {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Idempotency-Key": idempotency_key,
            "X-Client-Request-Id": client_request_id,
            "User-Agent": "cape-loop/0.1",
        }
    )
    # OpenAI tokenization is byte-based.  Counting each UTF-8 byte as one
    # input token, adding framing headroom, and reserving the declared maximum
    # output is intentionally conservative and requires no tokenizer package.
    estimated_max_tokens = (
        len(body_bytes) + 512 + config.max_output_tokens
    )
    return PreparedOpenAIRequest(
        endpoint=config.endpoint,
        body=body,
        body_bytes=body_bytes,
        body_sha256=body_digest,
        headers=safe_headers,
        idempotency_key=idempotency_key,
        client_request_id=client_request_id,
        estimated_max_tokens=estimated_max_tokens,
    )


@dataclass(frozen=True, slots=True)
class HTTPResult:
    """Provider transport result, including response headers."""

    status: int
    headers: Mapping[str, str]
    body: bytes


class HTTPTransport(Protocol):
    def __call__(
        self,
        *,
        url: str,
        body: bytes,
        headers: Mapping[str, str],
        timeout: float,
    ) -> HTTPResult:
        """Execute one HTTP request."""


class _RejectRedirects(HTTPRedirectHandler):
    """Keep authorization material on the already validated origin."""

    def redirect_request(
        self,
        req: Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        return None


def _read_http_body(response: Any, *, status: int) -> bytes:
    """Read at most the fixed response limit plus one detection byte."""

    body = response.read(HTTP_RESPONSE_BODY_LIMIT_BYTES + 1)
    if len(body) > HTTP_RESPONSE_BODY_LIMIT_BYTES:
        raise HTTPResponseBodyTooLarge(status=status)
    return body


def urllib_transport(
    *,
    url: str,
    body: bytes,
    headers: Mapping[str, str],
    timeout: float,
) -> HTTPResult:
    """Execute one POST using only :mod:`urllib`."""

    http_request = Request(
        url,
        data=body,
        headers=dict(headers),
        method="POST",
    )
    opener = build_opener(_RejectRedirects())
    try:
        with opener.open(http_request, timeout=timeout) as response:
            return HTTPResult(
                status=int(response.status),
                headers=dict(response.headers.items()),
                body=_read_http_body(
                    response,
                    status=int(response.status),
                ),
            )
    except HTTPError as exc:
        try:
            return HTTPResult(
                status=int(exc.code),
                headers=dict(exc.headers.items()) if exc.headers else {},
                body=_read_http_body(exc, status=int(exc.code)),
            )
        finally:
            exc.close()


class ExecutionBudget:
    """Sequential request/token ledger with preflight reservations."""

    def __init__(self, *, max_requests: int, max_total_tokens: int) -> None:
        if not _is_positive_int(max_requests):
            raise ValueError("max_requests must be a positive integer")
        if not _is_positive_int(max_total_tokens):
            raise ValueError("max_total_tokens must be a positive integer")
        self.max_requests = max_requests
        self.max_total_tokens = max_total_tokens
        self.request_count = 0
        self.total_tokens = 0
        self._reservation: int | None = None

    def restore(self, *, request_count: int, total_tokens: int) -> None:
        """Restore a fresh ledger from resumable output."""

        if self.request_count or self.total_tokens or self._reservation is not None:
            raise RuntimeError("cannot restore a budget that has already been used")
        if (
            not isinstance(request_count, int)
            or isinstance(request_count, bool)
            or request_count < 0
        ):
            raise ValueError("request_count must be a non-negative integer")
        if (
            not isinstance(total_tokens, int)
            or isinstance(total_tokens, bool)
            or total_tokens < 0
        ):
            raise ValueError("total_tokens must be a non-negative integer")
        if request_count > self.max_requests:
            raise BudgetExceeded(
                f"resumed request count {request_count} exceeds "
                f"max_requests={self.max_requests}"
            )
        if total_tokens > self.max_total_tokens:
            raise BudgetExceeded(
                f"resumed token count {total_tokens} exceeds "
                f"max_total_tokens={self.max_total_tokens}"
            )
        self.request_count = request_count
        self.total_tokens = total_tokens

    def ensure_capacity(
        self,
        *,
        request_count: int,
        total_tokens: int,
    ) -> None:
        """Check a retry-expanded corpus without changing the ledger."""

        if (
            not isinstance(request_count, int)
            or isinstance(request_count, bool)
            or request_count < 0
        ):
            raise ValueError("request_count must be a non-negative integer")
        if (
            not isinstance(total_tokens, int)
            or isinstance(total_tokens, bool)
            or total_tokens < 0
        ):
            raise ValueError("total_tokens must be a non-negative integer")
        if self.request_count + request_count > self.max_requests:
            raise BudgetExceeded(
                "remaining retry-expanded corpus would exceed "
                f"max_requests={self.max_requests}"
            )
        if self.total_tokens + total_tokens > self.max_total_tokens:
            raise BudgetExceeded(
                "remaining retry-expanded corpus's conservative token "
                f"allocation would exceed max_total_tokens={self.max_total_tokens}"
            )

    def reserve(self, estimated_max_tokens: int) -> None:
        if self._reservation is not None:
            raise RuntimeError("only one in-flight budget reservation is supported")
        if not _is_positive_int(estimated_max_tokens):
            raise ValueError("estimated_max_tokens must be a positive integer")
        if self.request_count + 1 > self.max_requests:
            raise BudgetExceeded(
                f"request would exceed max_requests={self.max_requests}"
            )
        if self.total_tokens + estimated_max_tokens > self.max_total_tokens:
            raise BudgetExceeded(
                "request's conservative token reservation would exceed "
                f"max_total_tokens={self.max_total_tokens}"
            )
        self._reservation = estimated_max_tokens

    def commit(self, actual_total_tokens: int | None) -> None:
        if self._reservation is None:
            raise RuntimeError("no in-flight budget reservation")
        charged = (
            self._reservation
            if actual_total_tokens is None
            else actual_total_tokens
        )
        if (
            not isinstance(charged, int)
            or isinstance(charged, bool)
            or charged < 0
        ):
            raise ProviderResponseError("usage.total_tokens must be non-negative")
        if charged > self._reservation:
            raise ProviderResponseError(
                "provider-reported tokens exceed the conservative preflight "
                "reservation"
            )
        self.request_count += 1
        self.total_tokens += charged
        self._reservation = None

    def rollback(self) -> None:
        self._reservation = None


@dataclass(frozen=True, slots=True)
class OpenAIProviderResult:
    """One completed response plus the reproducibility sidecar."""

    provider_label = "OpenAI"

    response: LLMResponse
    model_requested: str
    model_returned: str
    provider_response_id: str
    provider_created_at: int | float | None
    usage: Mapping[str, Any]
    started_at: str
    completed_at: str
    attempts: int
    request_body_sha256: str
    idempotency_key: str
    client_request_id: str
    server_request_id: str | None
    processing_ms: str | None
    estimated_max_tokens: int
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
            raise ValueError("unknown provider audit acceptance status")
        return {
            "schema_version": 1,
            "provider": "openai",
            "acceptance_status": acceptance_status,
            "request_id": self.response.request_id,
            "prompt_sha256": self.response.prompt_sha256,
            "request_body_sha256": self.request_body_sha256,
            "model_requested": self.model_requested,
            "model_returned": self.model_returned,
            "provider_response_id": self.provider_response_id,
            "provider_created_at": self.provider_created_at,
            "usage": self.usage,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "attempts": self.attempts,
            "idempotency_key": self.idempotency_key,
            "client_request_id": self.client_request_id,
            "server_request_id": self.server_request_id,
            "processing_ms": self.processing_ms,
            "estimated_max_tokens": self.estimated_max_tokens,
            "raw_response_sha256": self.response.raw_response_sha256,
            "raw_response": self.raw_response,
            "replay_response": self.response.to_dict(),
        }


def _lower_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {str(key).lower(): str(value) for key, value in headers.items()}


def _retry_after_seconds(
    value: str | None,
    *,
    now_epoch: float,
) -> float | None:
    if value is None:
        return None
    stripped = value.strip()
    try:
        seconds = float(stripped)
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(stripped)
        except (TypeError, ValueError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        seconds = retry_at.timestamp() - now_epoch
    if not math.isfinite(seconds):
        return None
    return max(0.0, seconds)


def _is_transient_status(status: int) -> bool:
    return status in _TRANSIENT_STATUSES or 500 <= status <= 599


def _safe_api_error(body: bytes, *, secret: str) -> str:
    message = "provider returned an error response"
    try:
        parsed = json.loads(body.decode("utf-8"))
        if isinstance(parsed, Mapping):
            error = parsed.get("error")
            if isinstance(error, Mapping) and isinstance(error.get("message"), str):
                message = error["message"]
    except (UnicodeDecodeError, json.JSONDecodeError):
        pass
    if secret:
        message = message.replace(secret, "[redacted]")
    return " ".join(message.split())[:500]


def _redact_provider_value(value: Any, secret: str) -> Any:
    """Recursively remove an echoed credential from provider-controlled data."""

    if isinstance(value, Mapping):
        redacted: dict[str, Any] = {}
        for key, nested in value.items():
            safe_key = str(key)
            if secret:
                safe_key = safe_key.replace(secret, "[redacted]")
            redacted[safe_key] = _redact_provider_value(nested, secret)
        return redacted
    if isinstance(value, list):
        return [
            _redact_provider_value(item, secret)
            for item in value
        ]
    if isinstance(value, str) and secret:
        return value.replace(secret, "[redacted]")
    return value


def _usage_total_tokens(usage: Mapping[str, Any]) -> int | None:
    total = usage.get("total_tokens")
    if isinstance(total, int) and not isinstance(total, bool) and total >= 0:
        return total
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    if all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in (input_tokens, output_tokens)
    ):
        return int(input_tokens) + int(output_tokens)
    return None


def _parse_json_text(text: str) -> Mapping[str, Any] | None:
    candidate = text.strip()
    if candidate.startswith("```") and candidate.endswith("```"):
        first_newline = candidate.find("\n")
        if first_newline >= 0:
            candidate = candidate[first_newline + 1 : -3].strip()
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, Mapping) else None


def _extract_structured_payload(raw: Mapping[str, Any]) -> Mapping[str, Any]:
    status = raw.get("status")
    if isinstance(status, str) and status != "completed":
        details = raw.get("incomplete_details") or raw.get("error")
        raise ProviderResponseError(
            f"OpenAI response status is {status!r}; details={details!r}"
        )

    mapping_candidates: list[Mapping[str, Any]] = []
    text_candidates: list[str] = []
    refusals: list[str] = []

    parsed_top = raw.get("output_parsed")
    if isinstance(parsed_top, Mapping):
        mapping_candidates.append(parsed_top)
    output_text = raw.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        text_candidates.append(output_text)

    output = raw.get("output")
    if isinstance(output, Sequence) and not isinstance(output, (str, bytes)):
        for item in output:
            if not isinstance(item, Mapping):
                continue
            if isinstance(item.get("parsed"), Mapping):
                mapping_candidates.append(item["parsed"])
            content = item.get("content")
            if not isinstance(content, Sequence) or isinstance(
                content, (str, bytes)
            ):
                content = ()
            for part in content:
                if not isinstance(part, Mapping):
                    continue
                if part.get("type") == "refusal" and isinstance(
                    part.get("refusal"), str
                ):
                    refusals.append(part["refusal"])
                for key in ("parsed", "json"):
                    if isinstance(part.get(key), Mapping):
                        mapping_candidates.append(part[key])
                if isinstance(part.get("text"), str) and part["text"].strip():
                    text_candidates.append(part["text"])

    if refusals:
        raise ProviderResponseError(
            "OpenAI response was refused: " + " ".join(refusals)[:500]
        )
    for candidate in mapping_candidates:
        if "beliefs" in candidate:
            return candidate
    for candidate in text_candidates:
        parsed = _parse_json_text(candidate)
        if parsed is not None and "beliefs" in parsed:
            return parsed
    if len(text_candidates) > 1:
        parsed = _parse_json_text("".join(text_candidates))
        if parsed is not None and "beliefs" in parsed:
            return parsed
    raise ProviderResponseError(
        "OpenAI response contains no parseable structured beliefs output"
    )


def _parse_provider_result(
    *,
    request: LLMRequest,
    prepared: PreparedOpenAIRequest,
    config: OpenAIProviderConfig,
    http_result: HTTPResult,
    attempts: int,
    started_at: str,
    completed_at: str,
    secret: str,
) -> OpenAIProviderResult:
    try:
        raw = json.loads(http_result.body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProviderResponseError("OpenAI response body is not valid JSON") from exc
    if not isinstance(raw, Mapping):
        raise ProviderResponseError("OpenAI response body must be a JSON object")
    safe_raw = _redact_provider_value(raw, secret)
    if not isinstance(safe_raw, Mapping):
        raise ProviderResponseError("OpenAI response body must be a JSON object")
    payload = _extract_structured_payload(safe_raw)
    returned_model = safe_raw.get("model")
    if not isinstance(returned_model, str) or not returned_model.strip():
        returned_model = "<missing>"
    else:
        returned_model = returned_model.strip()
    response_id = safe_raw.get("id")
    if not isinstance(response_id, str) or not response_id.strip():
        raise ProviderResponseError("OpenAI response is missing its response ID")
    usage = safe_raw.get("usage")
    if not isinstance(usage, Mapping):
        usage = {}
    raw_digest = sha256(http_result.body).hexdigest()
    response = LLMResponse.parse(
        {
            "schema_version": 1,
            "request_id": request.request_id,
            "prompt_sha256": request.prompt_sha256,
            "model_id": returned_model,
            "beliefs": payload.get("beliefs"),
            "raw_response_sha256": raw_digest,
        }
    )
    headers = _redact_provider_value(
        _lower_headers(http_result.headers),
        secret,
    )
    if not isinstance(headers, Mapping):
        headers = {}
    created_at = safe_raw.get("created_at")
    if not (
        isinstance(created_at, (int, float))
        and not isinstance(created_at, bool)
        and math.isfinite(float(created_at))
    ):
        created_at = None
    return OpenAIProviderResult(
        response=response,
        model_requested=config.model,
        model_returned=returned_model,
        provider_response_id=response_id,
        provider_created_at=created_at,
        usage=dict(usage),
        started_at=started_at,
        completed_at=completed_at,
        attempts=attempts,
        request_body_sha256=prepared.body_sha256,
        idempotency_key=prepared.idempotency_key,
        client_request_id=prepared.client_request_id,
        server_request_id=headers.get("x-request-id"),
        processing_ms=headers.get("openai-processing-ms"),
        estimated_max_tokens=prepared.estimated_max_tokens,
        raw_response=dict(safe_raw),
    )


def returned_model_is_consistent(
    requested_model: str,
    returned_model: str,
) -> bool:
    """Accept the requested alias or its dated snapshot, never another variant."""

    if returned_model == requested_model:
        return True
    if re.search(r"-\d{4}-\d{2}-\d{2}$", requested_model):
        return False
    return bool(
        re.fullmatch(
            re.escape(requested_model) + r"-\d{4}-\d{2}-\d{2}",
            returned_model,
        )
    )


class OpenAIResponsesProvider:
    """Synchronous, budgeted OpenAI Responses API client."""

    provider_name = "openai"

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

    def prepare(self, request: LLMRequest) -> PreparedOpenAIRequest:
        """Prepare a dry run.  This never reads the API-key environment."""

        return prepare_openai_request(request, self.config)

    def restore_budget(self, *, request_count: int, total_tokens: int) -> None:
        self.budget.restore(
            request_count=request_count,
            total_tokens=total_tokens,
        )

    def returned_model_is_consistent(self, returned_model: str) -> bool:
        """Validate the returned identity under the OpenAI snapshot policy."""

        return returned_model_is_consistent(
            self.config.model,
            returned_model,
        )

    def manifest_fields(self) -> dict[str, Any]:
        """Return provider-specific, credential-free manifest fields."""

        return {
            "reasoning_effort": self.config.reasoning_effort,
        }

    def complete(
        self,
        request: LLMRequest,
        *,
        attempt_ledger: DurableProviderAttemptLedger | None = None,
    ) -> OpenAIProviderResult:
        """Execute one explicitly authorized, hash-bound live completion."""

        if not self.config.live_execution:
            raise LiveExecutionRequired(
                "live OpenAI execution is disabled; set live_execution=True "
                "only after reviewing the request and budgets"
            )
        prepared = self.prepare(request)
        key: str | None = None
        headers: dict[str, str] | None = None
        started_epoch = self._epoch_time()
        started_at = _utc_timestamp(started_epoch)
        last_result: HTTPResult | None = None

        try:
            for attempt in range(1, self.config.max_retries + 2):
                self.budget.reserve(prepared.estimated_max_tokens)
                if key is None:
                    loaded = os.environ.get(self.config.api_key_env)
                    if not isinstance(loaded, str) or not loaded.strip():
                        self.budget.rollback()
                        raise MissingAPIKey(
                            "missing API key in environment variable "
                            f"{self.config.api_key_env}"
                        )
                    loaded = loaded.strip()
                    if "\r" in loaded or "\n" in loaded:
                        self.budget.rollback()
                        raise MissingAPIKey(
                            f"{self.config.api_key_env} contains an invalid "
                            "newline"
                        )
                    key = loaded
                    headers = dict(prepared.headers)
                    headers["Authorization"] = "Bearer " + key
                attempt_started_at = _utc_timestamp(self._epoch_time())
                try:
                    attempt_id = (
                        attempt_ledger.start(
                            request,
                            prepared,
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
                    automatic_retry_safe: bool = False,
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
                            raise RuntimeError(
                                "provider attempt journal identity is missing"
                            )
                        attempt_ledger.settle(
                            attempt_id,
                            settled_at=_utc_timestamp(self._epoch_time()),
                            outcome=outcome,
                            automatic_retry_safe=automatic_retry_safe,
                            http_status=http_status,
                            charged_tokens=charged,
                            server_request_id=server_request_id,
                            response_body_sha256=response_body_sha256,
                            response_record=response_record,
                            provider_audit=provider_audit,
                        )
                    self.budget.commit(charged)

                try:
                    if headers is None or key is None:
                        raise RuntimeError("live credential was not initialized")
                    result = self._transport(
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
                        automatic_retry_safe=True,
                        response_record={
                            "error_type": type(exc).__name__,
                        },
                    )
                    if attempt > self.config.max_retries:
                        raise OpenAIProviderError(
                            "OpenAI transport failed after "
                            f"{attempt} attempts; "
                            f"client_request_id={prepared.client_request_id}; "
                            f"error_type={type(exc).__name__}"
                        ) from exc
                    self._sleep(self._backoff_seconds(attempt, None))
                    continue

                last_result = result
                lowered = _lower_headers(result.headers)
                server_request_id = (
                    lowered.get("x-request-id", "").replace(
                        key,
                        "[redacted]",
                    )
                    or None
                )
                response_digest = sha256(result.body).hexdigest()
                if 200 <= result.status <= 299:
                    completed_at = _utc_timestamp(self._epoch_time())
                    safe_response: Mapping[str, Any] | None = None
                    try:
                        try:
                            decoded = json.loads(result.body.decode("utf-8"))
                        except (UnicodeDecodeError, json.JSONDecodeError):
                            decoded = None
                        if isinstance(decoded, Mapping):
                            candidate = _redact_provider_value(decoded, key)
                            if isinstance(candidate, Mapping):
                                safe_response = dict(candidate)
                        parsed = _parse_provider_result(
                            request=request,
                            prepared=prepared,
                            config=self.config,
                            http_result=result,
                            attempts=attempt,
                            started_at=started_at,
                            completed_at=completed_at,
                            secret=key,
                        )
                    except Exception:
                        settle(
                            outcome="invalid_response",
                            http_status=result.status,
                            server_request_id=server_request_id,
                            response_body_sha256=response_digest,
                            response_record=(
                                safe_response
                                if safe_response is not None
                                else {
                                    "body_excerpt": " ".join(
                                        result.body.decode(
                                            "utf-8",
                                            errors="replace",
                                        )
                                        .replace(key, "[redacted]")
                                        .split()
                                    )[:500]
                                }
                            ),
                        )
                        raise
                    mismatch = not returned_model_is_consistent(
                        parsed.model_requested,
                        parsed.model_returned,
                    )
                    audit = parsed.to_audit_record(
                        acceptance_status=(
                            "rejected_model_mismatch"
                            if mismatch
                            else "accepted"
                        )
                    )
                    settle(
                        outcome=(
                            "rejected_provider_result"
                            if mismatch
                            else "success"
                        ),
                        http_status=result.status,
                        server_request_id=server_request_id,
                        response_body_sha256=response_digest,
                        response_record=dict(parsed.raw_response),
                        charged_tokens=_usage_total_tokens(parsed.usage),
                        provider_audit=audit,
                    )
                    if mismatch:
                        raise ProviderModelMismatch(parsed)
                    return parsed

                safe_message = _safe_api_error(result.body, secret=key)
                settle(
                    outcome="http_error",
                    automatic_retry_safe=_is_transient_status(result.status),
                    http_status=result.status,
                    server_request_id=server_request_id,
                    response_body_sha256=response_digest,
                    response_record={"error": safe_message},
                )
                if _is_transient_status(result.status) and (
                    attempt <= self.config.max_retries
                ):
                    self._sleep(
                        self._backoff_seconds(
                            attempt,
                            lowered.get("retry-after"),
                        )
                    )
                    continue
                raise ProviderHTTPError(
                    status=result.status,
                    message=safe_message,
                    client_request_id=prepared.client_request_id,
                    server_request_id=server_request_id,
                )
        except Exception:
            self.budget.rollback()
            raise

        # The loop always returns or raises.  This branch makes the invariant
        # explicit for type checkers and protects against accidental edits.
        self.budget.rollback()
        raise OpenAIProviderError(
            f"OpenAI execution ended without a response: {last_result!r}"
        )

    def _backoff_seconds(
        self,
        attempt: int,
        retry_after: str | None,
    ) -> float:
        base = min(
            self.config.max_backoff_seconds,
            self.config.initial_backoff_seconds * (2 ** (attempt - 1)),
        )
        random_unit = min(1.0, max(0.0, float(self._random_value())))
        jittered = base * (
            1
            + self.config.jitter_fraction * ((2 * random_unit) - 1)
        )
        bounded = min(self.config.max_backoff_seconds, max(0.0, jittered))
        declared = _retry_after_seconds(
            retry_after,
            now_epoch=self._epoch_time(),
        )
        return max(bounded, declared or 0.0)


def parse_llm_request(raw: Mapping[str, Any]) -> LLMRequest:
    """Strictly parse and independently verify an exported replay request."""

    allowed = {
        "schema_version",
        "request_id",
        "updater_id",
        "view",
        "system_instruction",
        "payload",
        "prompt_sha256",
    }
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown request fields: {sorted(unknown)}")
    if raw.get("schema_version") != 1:
        raise ValueError("LLM request schema_version must be 1")
    for field in ("request_id", "updater_id", "system_instruction"):
        value = raw.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} is required")
    view = raw.get("view")
    if view not in VIEWS:
        raise ValueError(f"unknown LLM view: {view}")
    payload = raw.get("payload")
    if not isinstance(payload, Mapping):
        raise ValueError("payload must be an object")
    digest = raw.get("prompt_sha256")
    expected = sha256(
        (
            raw["system_instruction"]
            + "\n"
            + _canonical(payload)
        ).encode("utf-8")
    ).hexdigest()
    if digest != expected:
        raise ValueError("prompt_sha256 does not bind this request payload")
    return LLMRequest(
        request_id=raw["request_id"],
        updater_id=raw["updater_id"],
        view=view,
        payload=dict(payload),
        system_instruction=raw["system_instruction"],
        prompt_sha256=expected,
    )


def read_requests(
    path: str | Path | bytes,
) -> tuple[LLMRequest, ...]:
    """Read strict request JSONL from a path or immutable byte snapshot."""

    if isinstance(path, bytes):
        source_label = "<request-bytes>"
        try:
            lines = path.decode("utf-8").splitlines()
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"{source_label}: input must be valid UTF-8"
            ) from exc
    else:
        source = Path(path)
        source_label = str(source)
        lines = source.read_text(encoding="utf-8").splitlines()
    requests: list[LLMRequest] = []
    seen: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            if not isinstance(raw, Mapping):
                raise ValueError("request line must be a JSON object")
            request = parse_llm_request(raw)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{source_label}:{line_number}: {exc}"
            ) from exc
        if request.request_id in seen:
            raise ValueError(f"duplicate request_id: {request.request_id}")
        seen.add(request.request_id)
        requests.append(request)
    return tuple(requests)


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    """Append one canonical line with O_APPEND, flush, and fsync durability."""

    append_provider_jsonl(path, record)


def _repair_trailing_jsonl(path: Path) -> bool:
    """Repair only a crash-truncated final line; never hides a middle error."""

    return repair_attempt_journal(path)


def _read_audit_records(
    path: Path,
    *,
    provider_name: str = "openai",
) -> dict[str, Mapping[str, Any]]:
    if not path.exists():
        return {}
    records: dict[str, Mapping[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                if not isinstance(raw, Mapping):
                    raise ValueError("audit line must be a JSON object")
                if (
                    raw.get("schema_version") != 1
                    or raw.get("provider") != provider_name
                ):
                    raise ValueError("unsupported provider audit record")
                replay_raw = raw.get("replay_response")
                if not isinstance(replay_raw, Mapping):
                    raise ValueError("audit record lacks replay_response")
                replay = LLMResponse.parse(replay_raw)
                if (
                    raw.get("request_id") != replay.request_id
                    or raw.get("prompt_sha256") != replay.prompt_sha256
                ):
                    raise ValueError("audit and replay bindings differ")
                if raw.get("model_returned") != replay.model_id:
                    raise ValueError("audit and replay model labels differ")
                if (
                    raw.get("raw_response_sha256")
                    != replay.raw_response_sha256
                ):
                    raise ValueError(
                        "audit and replay raw-response digests differ"
                    )
                if raw.get("acceptance_status", "accepted") != "accepted":
                    raise ValueError(
                        "provider audit records a rejected model mismatch; "
                        "manual review is required before any retry"
                    )
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            if replay.request_id in records:
                raise ValueError(
                    f"duplicate audit request_id: {replay.request_id}"
                )
            records[replay.request_id] = dict(raw)
    return records


def _read_raw_provider_audits(
    path: Path,
    *,
    provider_name: str,
) -> dict[str, Mapping[str, Any]]:
    """Read just enough audit structure to reconcile attempt settlements."""

    if not path.exists():
        return {}
    records: dict[str, Mapping[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            if (
                not isinstance(raw, Mapping)
                or raw.get("schema_version") != 1
                or raw.get("provider") != provider_name
            ):
                raise ValueError(
                    f"{path}:{line_number}: unsupported provider audit record"
                )
            request_id = raw.get("request_id")
            if not isinstance(request_id, str) or not request_id:
                raise ValueError(
                    f"{path}:{line_number}: provider audit lacks request_id"
                )
            if request_id in records:
                raise ValueError(
                    f"duplicate audit request_id: {request_id}"
                )
            records[request_id] = dict(raw)
    return records


def _reconcile_attempt_audits(
    ledger: DurableProviderAttemptLedger,
    audit_path: Path,
    *,
    provider_name: str,
) -> None:
    """Repair a crash between final settlement and public audit append."""

    public = _read_raw_provider_audits(
        audit_path,
        provider_name=provider_name,
    )
    embedded = ledger.embedded_final_audits()
    for request_id, audit in embedded.items():
        retained = public.get(request_id)
        if retained is None:
            _append_jsonl(audit_path, audit)
            public[request_id] = audit
        elif dict(retained) != audit:
            raise ValueError(
                "provider transport-attempt/final audit mismatch for "
                f"{request_id}"
            )
    missing_attempts = sorted(set(public) - set(embedded))
    if missing_attempts:
        raise ProviderAttemptManualReviewRequired(
            "provider audits lack durable physical-attempt evidence; manual "
            "review is required before resume: "
            + ", ".join(missing_attempts)
        )


def _validate_resumed_audit(
    audit: Mapping[str, Any],
    request: LLMRequest,
    provider: Any,
) -> None:
    """Bind a resumable record to the full current request configuration."""

    provider_name = getattr(provider, "provider_name", "openai")
    prepared = provider.prepare(request)
    expected = {
        "request_id": request.request_id,
        "prompt_sha256": request.prompt_sha256,
        "request_body_sha256": prepared.body_sha256,
        "model_requested": provider.config.model,
        "client_request_id": prepared.client_request_id,
    }
    idempotency_key = getattr(prepared, "idempotency_key", None)
    if idempotency_key is not None:
        expected["idempotency_key"] = idempotency_key
    mismatches = {
        field: {
            "retained": audit.get(field),
            "expected": value,
        }
        for field, value in expected.items()
        if audit.get(field) != value
    }
    returned_model = audit.get("model_returned")
    if (
        not isinstance(returned_model, str)
        or not provider.returned_model_is_consistent(returned_model)
    ):
        mismatches["model_returned"] = {
            "retained": returned_model,
            "expected": provider.config.model,
        }
    if mismatches:
        raise ValueError(
            f"resumable {provider_name} audit does not match the current request "
            "body/model configuration: "
            + _canonical(mismatches)
        )
    if hasattr(provider, "validate_resumed_audit"):
        provider.validate_resumed_audit(
            audit,
            request=request,
            prepared=prepared,
        )


def require_retry_expanded_capacity(
    provider: Any,
    requests: Iterable[LLMRequest],
    *,
    completed_request_ids: Iterable[str] = (),
) -> dict[str, int]:
    """Admit every unfinished logical request before any provider dispatch.

    Existing journals are reconciled before this helper is called. The check
    reserves the worst-case physical-attempt and token envelope for every
    remaining request, so an empty or partially populated output path cannot
    turn an infeasible fresh corpus into a paid partial execution.
    """

    material = tuple(requests)
    request_by_id = {request.request_id: request for request in material}
    if len(request_by_id) != len(material):
        raise ValueError("requests contain duplicate request IDs")
    completed = frozenset(completed_request_ids)
    unexpected = sorted(completed - set(request_by_id))
    if unexpected:
        raise ValueError(
            "completed journals contain requests outside the current corpus: "
            + ", ".join(unexpected)
        )
    pending = tuple(
        request
        for request in material
        if request.request_id not in completed
    )
    attempts_per_request = provider.config.max_retries + 1
    conservative_tokens = sum(
        provider.prepare(request).estimated_max_tokens
        for request in pending
    )
    retry_expanded_attempts = len(pending) * attempts_per_request
    retry_expanded_tokens = conservative_tokens * attempts_per_request
    provider.budget.ensure_capacity(
        request_count=retry_expanded_attempts,
        total_tokens=retry_expanded_tokens,
    )
    return {
        "pending_request_count": len(pending),
        "maximum_attempts_per_request": attempts_per_request,
        "retry_expanded_attempt_count": retry_expanded_attempts,
        "retry_expanded_token_allocation": retry_expanded_tokens,
    }


@dataclass(frozen=True, slots=True)
class ExecutionSummary:
    request_count: int
    resumed_count: int
    executed_count: int
    total_tokens: int
    responses_path: str
    audit_path: str
    attempts_path: str
    transport_attempt_count: int
    repaired_trailing_files: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "request_count": self.request_count,
            "resumed_count": self.resumed_count,
            "executed_count": self.executed_count,
            "total_tokens": self.total_tokens,
            "responses_path": self.responses_path,
            "audit_path": self.audit_path,
            "attempts_path": self.attempts_path,
            "transport_attempt_count": self.transport_attempt_count,
            "request_budget_unit": "physical_http_attempt",
            "repaired_trailing_files": list(
                self.repaired_trailing_files
            ),
        }


def execute_requests(
    provider: Any,
    requests: Iterable[LLMRequest],
    *,
    responses_path: str | Path,
    audit_path: str | Path,
    attempts_path: str | Path | None = None,
) -> ExecutionSummary:
    """Execute/resume a static corpus as one exclusive journal transaction."""

    audit_file = Path(audit_path)
    lock_file = audit_file.with_name(
        f".{audit_file.name}.provider-execution.lock"
    )
    with ExclusiveProviderExecutionLock(lock_file):
        return _execute_requests_locked(
            provider,
            requests,
            responses_path=responses_path,
            audit_path=audit_path,
            attempts_path=attempts_path,
        )


def _execute_requests_locked(
    provider: Any,
    requests: Iterable[LLMRequest],
    *,
    responses_path: str | Path,
    audit_path: str | Path,
    attempts_path: str | Path | None = None,
) -> ExecutionSummary:
    """Execute or resume requests into replay JSONL and a provider audit JSONL.

    The audit line is durably appended before its replay response.  On resume,
    a completed audit record whose replay line was interrupted is reconciled
    without another provider request.
    """

    material = tuple(requests)
    request_by_id = {request.request_id: request for request in material}
    if len(request_by_id) != len(material):
        raise ValueError("requests contain duplicate request IDs")
    response_file = Path(responses_path)
    audit_file = Path(audit_path)
    attempt_file = (
        Path(attempts_path)
        if attempts_path is not None
        else default_attempt_path(audit_file)
    )
    resolved_paths = {
        response_file.resolve(),
        audit_file.resolve(),
        attempt_file.resolve(),
    }
    if len(resolved_paths) != 3:
        raise ValueError(
            "responses_path, audit_path, and attempts_path must differ"
        )

    repaired = tuple(
        str(path)
        for path, repair in (
            (attempt_file, repair_attempt_journal),
            (audit_file, _repair_trailing_jsonl),
            (response_file, _repair_trailing_jsonl),
        )
        if repair(path)
    )
    provider_name = getattr(provider, "provider_name", "openai")
    attempt_ledger = DurableProviderAttemptLedger(
        attempt_file,
        provider_name=provider_name,
        model_requested=provider.config.model,
    )
    attempt_ledger.validate_requests(request_by_id, provider)
    attempt_ledger.assert_safe_to_resume()
    _reconcile_attempt_audits(
        attempt_ledger,
        audit_file,
        provider_name=provider_name,
    )
    existing_responses = (
        {response.request_id: response for response in read_responses(response_file)}
        if response_file.exists()
        else {}
    )
    audit_records = _read_audit_records(
        audit_file,
        provider_name=provider_name,
    )
    responses_without_audit = sorted(
        set(existing_responses) - set(audit_records)
    )
    if responses_without_audit:
        raise ValueError(
            f"{provider_name} resume responses lack audit-first records: "
            + ", ".join(responses_without_audit)
        )

    for request_id, response in existing_responses.items():
        request = request_by_id.get(request_id)
        if request is None:
            raise ValueError(f"unexpected existing response: {request_id}")
        if response.prompt_sha256 != request.prompt_sha256:
            raise ValueError(
                f"existing response is not bound to request {request_id}"
            )
    for request_id, audit in audit_records.items():
        request = request_by_id.get(request_id)
        if request is None:
            raise ValueError(f"unexpected existing audit record: {request_id}")
        _validate_resumed_audit(audit, request, provider)
        replay = LLMResponse.parse(audit["replay_response"])
        if replay.prompt_sha256 != request.prompt_sha256:
            raise ValueError(f"audit response is not bound to request {request_id}")
        existing = existing_responses.get(request_id)
        if existing is None:
            _append_jsonl(response_file, replay.to_dict())
            existing_responses[request_id] = replay
        elif existing.to_dict() != replay.to_dict():
            raise ValueError(
                f"audit/replay response mismatch for {request_id}"
            )

    restored_request_count, restored_tokens = attempt_ledger.accounting()
    provider.restore_budget(
        request_count=restored_request_count,
        total_tokens=restored_tokens,
    )
    require_retry_expanded_capacity(
        provider,
        material,
        completed_request_ids=existing_responses,
    )

    resumed_count = len(existing_responses)
    executed_count = 0
    for request in material:
        if request.request_id in existing_responses:
            continue
        try:
            result = provider.complete(
                request,
                attempt_ledger=attempt_ledger,
            )
        except ProviderResultRejected as exc:
            _append_jsonl(
                audit_file,
                exc.result.to_audit_record(
                    acceptance_status=exc.acceptance_status,
                ),
            )
            raise
        _append_jsonl(audit_file, result.to_audit_record())
        _append_jsonl(response_file, result.response.to_dict())
        audit_records[request.request_id] = result.to_audit_record()
        existing_responses[request.request_id] = result.response
        executed_count += 1

    return ExecutionSummary(
        request_count=len(material),
        resumed_count=resumed_count,
        executed_count=executed_count,
        total_tokens=provider.budget.total_tokens,
        responses_path=str(response_file),
        audit_path=str(audit_file),
        attempts_path=str(attempt_file),
        transport_attempt_count=provider.budget.request_count,
        repaired_trailing_files=repaired,
    )


def execute_jsonl(
    provider: Any,
    requests_path: str | Path,
    *,
    responses_path: str | Path,
    audit_path: str | Path,
    attempts_path: str | Path | None = None,
) -> ExecutionSummary:
    """Read an exported request corpus and execute it resumably."""

    return execute_requests(
        provider,
        read_requests(requests_path),
        responses_path=responses_path,
        audit_path=audit_path,
        attempts_path=attempts_path,
    )


class ResumableCompletionProvider:
    """Adaptive, journaled adapter returning replay-compatible responses.

    Unlike :func:`execute_jsonl`, this adapter does not require the full request
    corpus in advance. That is necessary for closed-loop experiments whose next
    prompt depends on the previous model-written profile. Each successful audit
    record is durably appended before its replay response, and a later process
    reuses a matching request without another provider call.
    """

    def __init__(
        self,
        provider: Any,
        *,
        responses_path: str | Path,
        audit_path: str | Path,
        attempts_path: str | Path | None = None,
    ) -> None:
        self.provider = provider
        self.responses_path = Path(responses_path)
        self.audit_path = Path(audit_path)
        self.attempts_path = (
            Path(attempts_path)
            if attempts_path is not None
            else default_attempt_path(self.audit_path)
        )
        if len(
            {
                self.responses_path.resolve(),
                self.audit_path.resolve(),
                self.attempts_path.resolve(),
            }
        ) != 3:
            raise ValueError(
                "responses_path, audit_path, and attempts_path must differ"
            )
        repair_attempt_journal(self.attempts_path)
        _repair_trailing_jsonl(self.audit_path)
        _repair_trailing_jsonl(self.responses_path)
        provider_name = getattr(provider, "provider_name", "openai")
        self._attempt_ledger = DurableProviderAttemptLedger(
            self.attempts_path,
            provider_name=provider_name,
            model_requested=provider.config.model,
        )
        self._attempt_ledger.assert_safe_to_resume()
        _reconcile_attempt_audits(
            self._attempt_ledger,
            self.audit_path,
            provider_name=provider_name,
        )
        self._responses = (
            {
                response.request_id: response
                for response in read_responses(self.responses_path)
            }
            if self.responses_path.exists()
            else {}
        )
        self._audits = _read_audit_records(
            self.audit_path,
            provider_name=provider_name,
        )
        unexpected_responses = sorted(set(self._responses) - set(self._audits))
        if unexpected_responses:
            raise ValueError(
                "adaptive replay responses lack audit-first records: "
                + ", ".join(unexpected_responses)
            )
        for request_id, audit in self._audits.items():
            replay = LLMResponse.parse(audit["replay_response"])
            existing = self._responses.get(request_id)
            if existing is None:
                _append_jsonl(self.responses_path, replay.to_dict())
                self._responses[request_id] = replay
            elif existing.to_dict() != replay.to_dict():
                raise ValueError(
                    f"adaptive audit/replay mismatch for {request_id}"
                )
        restored_request_count, restored_tokens = (
            self._attempt_ledger.accounting()
        )
        provider.restore_budget(
            request_count=restored_request_count,
            total_tokens=restored_tokens,
        )
        self._used_request_ids: list[str] = []
        self.executed_count = 0
        self.resumed_count = 0

    def require_static_corpus_capacity(
        self,
        requests: Iterable[LLMRequest],
    ) -> dict[str, int]:
        """Validate one known corpus and admit all of its remaining calls."""

        material = tuple(requests)
        request_by_id = {
            request.request_id: request for request in material
        }
        if len(request_by_id) != len(material):
            raise ValueError("requests contain duplicate request IDs")
        unexpected = sorted(set(self._audits) - set(request_by_id))
        if unexpected:
            raise ValueError(
                "adaptive journals contain requests outside the current "
                "static corpus: "
                + ", ".join(unexpected)
            )
        for request_id, audit in self._audits.items():
            _validate_resumed_audit(
                audit,
                request_by_id[request_id],
                self.provider,
            )
        return require_retry_expanded_capacity(
            self.provider,
            material,
            completed_request_ids=self._responses,
        )

    def complete(self, request: LLMRequest) -> LLMResponse:
        self._attempt_ledger.validate_request(request, self.provider)
        retained_audit = self._audits.get(request.request_id)
        if (
            retained_audit is not None
            and retained_audit.get("acceptance_status", "accepted")
            != "accepted"
        ):
            raise ValueError(
                "adaptive journal contains a rejected model mismatch or "
                "provider result for "
                f"{request.request_id}; manual review is required"
            )
        existing = self._responses.get(request.request_id)
        if existing is not None:
            if existing.prompt_sha256 != request.prompt_sha256:
                raise ValueError(
                    f"journal response is not bound to {request.request_id}"
                )
            _validate_resumed_audit(
                self._audits[request.request_id],
                request,
                self.provider,
            )
            self._used_request_ids.append(request.request_id)
            self.resumed_count += 1
            return existing
        self._attempt_ledger.assert_safe_to_resume()
        try:
            result = self.provider.complete(
                request,
                attempt_ledger=self._attempt_ledger,
            )
        except ProviderResultRejected as exc:
            rejected_audit = exc.result.to_audit_record(
                acceptance_status=exc.acceptance_status,
            )
            _append_jsonl(self.audit_path, rejected_audit)
            self._audits[request.request_id] = rejected_audit
            raise
        audit = result.to_audit_record()
        _append_jsonl(self.audit_path, audit)
        _append_jsonl(self.responses_path, result.response.to_dict())
        self._audits[request.request_id] = audit
        self._responses[request.request_id] = result.response
        self._used_request_ids.append(request.request_id)
        self.executed_count += 1
        return result.response

    @property
    def used_audit_records(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(
            self._audits[request_id]
            for request_id in dict.fromkeys(self._used_request_ids)
        )

    @property
    def used_attempt_records(self) -> tuple[Mapping[str, Any], ...]:
        return self._attempt_ledger.events_for_request_ids(
            tuple(dict.fromkeys(self._used_request_ids))
        )

    def to_manifest(self) -> dict[str, Any]:
        used = tuple(dict.fromkeys(self._used_request_ids))
        provider_fields = (
            self.provider.manifest_fields()
            if hasattr(self.provider, "manifest_fields")
            else {}
        )
        return {
            "schema_version": 1,
            "provider": getattr(self.provider, "provider_name", "openai"),
            "model_requested": self.provider.config.model,
            **provider_fields,
            "requests_used": len(used),
            "requests_executed": self.executed_count,
            "requests_resumed": self.resumed_count,
            "total_tokens": self.provider.budget.total_tokens,
            "transport_attempt_count": self.provider.budget.request_count,
            "request_budget_unit": "physical_http_attempt",
            "responses_journal": str(self.responses_path),
            "audit_journal": str(self.audit_path),
            "attempts_journal": str(self.attempts_path),
        }


class ResumableOpenAICompletionProvider(ResumableCompletionProvider):
    """Backward-compatible name for the OpenAI adaptive journal adapter."""
