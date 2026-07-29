"""Audited OpenRouter Chat Completions execution for CAPE-Loop.

OpenRouter is implemented as a first-class gateway rather than as a custom
OpenAI base URL.  The distinction matters for scientific provenance:
OpenRouter chooses an upstream serving provider, exposes routing metadata, and
uses ``author/model`` slugs that are not OpenAI Responses API model IDs.

This module therefore:

* pins one explicit model slug per run (no model-fallback arrays);
* requests strict JSON Schema output and locally validates it again;
* disables OpenRouter response caching and opts into router metadata;
* records the selected upstream provider and model;
* defaults endpoint fallbacks off and supports an optional provider allowlist;
* reads credentials only for an explicitly authorized live call; and
* emits the provider-neutral ``LLMResponse`` records used by replay.

The strict first-party Gate 4 collection remains separate.  An OpenRouter
response proves that the gateway reported a route; it is not equivalent to a
direct first-party provider-origin record.
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
from urllib.request import HTTPRedirectHandler, Request, build_opener
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
)
from .openai_provider import (
    ExecutionSummary,
    ProviderResultRejected,
    ResumableCompletionProvider,
    execute_jsonl,
    execute_requests,
)
from .provider_attempts import DurableProviderAttemptLedger


OPENROUTER_OFFICIAL_BASE_URL = "https://openrouter.ai/api"
OPENROUTER_EXAMPLE_MODEL = "google/gemini-3.6-flash"
OPENROUTER_MODELS_URL = "https://openrouter.ai/models"
HTTP_RESPONSE_BODY_LIMIT_BYTES = 16 * 1024 * 1024

_ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_PROVIDER_SLUG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")
_REASONING_EFFORTS = frozenset(
    {"", "none", "minimal", "low", "medium", "high", "xhigh", "max"}
)
_TRANSIENT_STATUSES = frozenset({408, 429, 502, 503, 529})
_RESPONSE_SCHEMA_NAME = "cape_loop_preference_beliefs"


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


def _redact_provider_value(value: Any, secret: str) -> Any:
    if isinstance(value, Mapping):
        redacted: dict[str, Any] = {}
        for key, nested in value.items():
            safe_key = str(key)
            if secret:
                safe_key = safe_key.replace(secret, "[redacted]")
            redacted[safe_key] = _redact_provider_value(nested, secret)
        return redacted
    if isinstance(value, list):
        return [_redact_provider_value(item, secret) for item in value]
    if isinstance(value, str) and secret:
        return value.replace(secret, "[redacted]")
    return value


def _safe_api_error(body: bytes, *, secret: str) -> str:
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
    if secret:
        message = message.replace(secret, "[redacted]")
    return " ".join(message.split())[:500]


def _usage_total_tokens(usage: Mapping[str, Any]) -> int | None:
    total = usage.get("total_tokens")
    if isinstance(total, int) and not isinstance(total, bool) and total >= 0:
        return total
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    if all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in (prompt, completion)
    ):
        return int(prompt) + int(completion)
    return None


def _upstream_model_is_consistent(
    requested_model: str,
    selected_model: str,
) -> bool:
    """Accept the canonical model or a conservative dated snapshot label."""

    if selected_model == requested_model:
        return True
    return bool(
        re.fullmatch(
            re.escape(requested_model)
            + r"-(?:\d{8}|\d{4}-\d{2}-\d{2})",
            selected_model,
        )
    )


def _route_constraint_evidence(
    config: "OpenRouterProviderConfig",
) -> str:
    if config.upstream_provider:
        return "request_body_provider_only_and_order"
    return "request_body_provider_preferences_without_exact_route_constraint"


_SELECTED_UPSTREAM_IDENTITY_SEMANTICS = (
    "router_display_identity_not_exact_route_slug_attestation"
)


def belief_json_schema(
    *,
    include_numeric_bounds: bool = True,
) -> dict[str, Any]:
    """Return the strict profile-belief schema sent to OpenRouter.

    Some OpenRouter routes for Anthropic models use Amazon Bedrock, whose
    structured-output schema subset rejects the JSON Schema ``minimum`` and
    ``maximum`` number keywords.  The description retains the provider-facing
    range requirement when those keywords are omitted, and ``LLMResponse``
    always enforces both the range and vector normalization locally.
    """

    probability: dict[str, Any] = {"type": "number"}
    if include_numeric_bounds:
        probability.update({"minimum": 0, "maximum": 1})
    else:
        probability["description"] = (
            "Probability in the inclusive range [0, 1]."
        )
    probability_vector = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            value: probability for value in VALUES
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


class OpenRouterProviderError(RuntimeError):
    """Base class for credential-free OpenRouter failures."""


class OpenRouterLiveExecutionRequired(OpenRouterProviderError):
    """Raised when a paid request lacks explicit live authorization."""


class OpenRouterMissingAPIKey(OpenRouterProviderError):
    """Raised when the configured credential variable is absent."""


class OpenRouterBudgetExceeded(OpenRouterProviderError):
    """Raised before a request that exceeds a declared ceiling."""


class OpenRouterResponseError(OpenRouterProviderError):
    """Raised for an incomplete, invalid, or unparseable provider response."""


class OpenRouterHTTPResponseBodyTooLarge(OpenRouterProviderError):
    """A response exceeded the fixed wire-body memory safety limit."""

    def __init__(self, *, status: int) -> None:
        self.status = status
        self.limit_bytes = HTTP_RESPONSE_BODY_LIMIT_BYTES
        super().__init__(
            "OpenRouter response body exceeded the fixed "
            f"{self.limit_bytes}-byte safety limit"
        )


class OpenRouterHTTPError(OpenRouterProviderError):
    """A non-retryable HTTP failure or exhausted retry sequence."""

    def __init__(
        self,
        *,
        status: int,
        message: str,
        client_request_id: str,
        generation_id: str | None,
    ) -> None:
        self.status = status
        self.client_request_id = client_request_id
        self.generation_id = generation_id
        suffix = f"; generation_id={generation_id}" if generation_id else ""
        super().__init__(
            f"OpenRouter request failed with HTTP {status}: {message}; "
            f"client_request_id={client_request_id}{suffix}"
        )


class OpenRouterResultRejected(ProviderResultRejected):
    """A complete paid response that fails model/routing acceptance."""

    acceptance_status = "rejected_openrouter_identity"

    def __init__(
        self,
        result: "OpenRouterProviderResult",
        reasons: Sequence[str],
    ) -> None:
        self.reasons = tuple(reasons)
        super().__init__(
            result,
            (
                "OpenRouter response failed the configured model/routing "
                "identity checks: "
                + ", ".join(self.reasons)
                + f"; client_request_id={result.client_request_id}"
            ),
            acceptance_status=self.acceptance_status,
        )


@dataclass(frozen=True, slots=True)
class OpenRouterProviderConfig:
    """Validated configuration for one OpenRouter model route."""

    model: str = OPENROUTER_EXAMPLE_MODEL
    reasoning_effort: str = ""
    api_key_env: str = "OPENROUTER_API_KEY"
    base_url: str = OPENROUTER_OFFICIAL_BASE_URL
    allow_custom_base_url: bool = False
    upstream_provider: str = ""
    allow_fallbacks: bool = False
    require_parameters: bool = True
    data_collection: str = "deny"
    zdr: bool = False
    http_referer: str = ""
    app_title: str = "CAPE-Loop"
    timeout_seconds: float = 180.0
    max_retries: int = 2
    initial_backoff_seconds: float = 0.5
    max_backoff_seconds: float = 8.0
    jitter_fraction: float = 0.25
    max_output_tokens: int = 4096
    max_requests: int = 100
    max_total_tokens: int = 500_000
    live_execution: bool = False

    def __post_init__(self) -> None:
        if (
            not isinstance(self.model, str)
            or not self.model.strip()
            or self.model != self.model.strip()
            or "/" not in self.model
            or self.model.startswith(("~", "/"))
            or self.model.endswith("/")
            or any(character.isspace() for character in self.model)
            or ":" in self.model
            or self.model.lower().endswith("-latest")
        ):
            raise ValueError(
                "model must be an explicit canonical OpenRouter author/model "
                "slug, not an alias or route variant"
            )
        if self.model.lower() == "openrouter/auto":
            raise ValueError(
                "openrouter/auto is not allowed for reproducible evaluation"
            )
        if self.reasoning_effort not in _REASONING_EFFORTS:
            raise ValueError(
                "reasoning_effort must be empty or one of "
                + ", ".join(sorted(_REASONING_EFFORTS - {""}))
            )
        if not _ENVIRONMENT_NAME.fullmatch(self.api_key_env):
            raise ValueError(
                "api_key_env must be a valid environment-variable name"
            )
        if self.api_key_env in {
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GEMINI_API_KEY",
        }:
            raise ValueError(
                "OpenRouter requires a dedicated credential variable; "
                "a first-party provider key must never be sent to the "
                "OpenRouter gateway"
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
            parsed.hostname != "openrouter.ai"
            or parsed.port is not None
            or parsed.path.rstrip("/") != "/api"
        ):
            raise ValueError(
                "base_url must be the official https://openrouter.ai/api "
                "path; set allow_custom_base_url=True only after reviewing "
                "where the credential will be sent"
            )
        if (
            self.allow_custom_base_url
            and parsed.hostname != "openrouter.ai"
            and self.api_key_env == "OPENROUTER_API_KEY"
        ):
            raise ValueError(
                "a custom base_url requires a dedicated credential "
                "environment variable instead of OPENROUTER_API_KEY"
            )
        if not isinstance(self.upstream_provider, str):
            raise ValueError("upstream_provider must be a string")
        if self.upstream_provider and (
            not _PROVIDER_SLUG.fullmatch(self.upstream_provider)
            or "//" in self.upstream_provider
            or ".." in self.upstream_provider
        ):
            raise ValueError(
                "upstream_provider must be one exact OpenRouter provider slug"
            )
        for name in ("allow_fallbacks", "require_parameters", "zdr"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be Boolean")
        if self.data_collection not in {"allow", "deny"}:
            raise ValueError("data_collection must be 'allow' or 'deny'")
        self._validate_attribution()
        if (
            not _is_nonnegative_number(self.timeout_seconds)
            or self.timeout_seconds == 0
        ):
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

    def _validate_attribution(self) -> None:
        for name, value in (
            ("http_referer", self.http_referer),
            ("app_title", self.app_title),
        ):
            if not isinstance(value, str):
                raise ValueError(f"{name} must be a string")
            if "\r" in value or "\n" in value:
                raise ValueError(f"{name} must not contain newlines")
        if self.http_referer:
            parsed = urlsplit(self.http_referer)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.netloc
                or parsed.username is not None
                or parsed.password is not None
            ):
                raise ValueError(
                    "http_referer must be an absolute HTTP(S) URL"
                )
        if len(self.app_title) > 200:
            raise ValueError("app_title must contain at most 200 characters")

    @property
    def endpoint(self) -> str:
        return self.base_url.rstrip("/") + "/v1/chat/completions"

    def provider_preferences(self) -> dict[str, Any]:
        preferences: dict[str, Any] = {
            "allow_fallbacks": self.allow_fallbacks,
            "require_parameters": self.require_parameters,
            "data_collection": self.data_collection,
        }
        if self.zdr:
            preferences["zdr"] = True
        if self.upstream_provider:
            preferences["order"] = [self.upstream_provider]
            preferences["only"] = [self.upstream_provider]
        return preferences


@dataclass(frozen=True, slots=True)
class PreparedOpenRouterRequest:
    """A deterministic dry-run request containing no credential."""

    endpoint: str
    body: Mapping[str, Any]
    body_bytes: bytes
    body_sha256: str
    headers: Mapping[str, str]
    client_request_id: str
    estimated_max_tokens: int


def prepare_openrouter_request(
    request: LLMRequest,
    config: OpenRouterProviderConfig,
) -> PreparedOpenRouterRequest:
    """Build one Chat Completions request without reading an API key."""

    if request.view not in VIEWS:
        raise ValueError(f"unknown LLM view: {request.view}")
    body: dict[str, Any] = {
        "model": config.model,
        "messages": [
            {
                "role": "system",
                "content": request.system_instruction,
            },
            {
                "role": "user",
                "content": _canonical(request.payload),
            },
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": _RESPONSE_SCHEMA_NAME,
                "strict": True,
                "schema": belief_json_schema(
                    include_numeric_bounds=not config.model.startswith(
                        "anthropic/"
                    )
                ),
            },
        },
        "max_tokens": config.max_output_tokens,
        "stream": False,
        "provider": config.provider_preferences(),
        "metadata": {
            "cape_loop_request_id": request.request_id[:512],
            "cape_loop_prompt_sha256": request.prompt_sha256,
            "cape_loop_view": request.view,
        },
    }
    if config.reasoning_effort:
        body["reasoning"] = {"effort": config.reasoning_effort}
    body_bytes = _canonical(body).encode("utf-8")
    body_digest = sha256(body_bytes).hexdigest()
    identity_digest = sha256(
        (
            "cape-loop-openrouter-v1\n"
            + request.request_id
            + "\n"
            + request.prompt_sha256
            + "\n"
            + config.model
            + "\n"
            + body_digest
        ).encode("utf-8")
    ).hexdigest()
    client_request_id = "cape-loop-" + identity_digest
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "X-OpenRouter-Metadata": "enabled",
        "X-OpenRouter-Cache": "false",
        "User-Agent": "cape-loop/0.1",
    }
    if config.http_referer:
        headers["HTTP-Referer"] = config.http_referer
    if config.app_title:
        headers["X-OpenRouter-Title"] = config.app_title
    estimated_max_tokens = (
        len(body_bytes) + 512 + config.max_output_tokens
    )
    return PreparedOpenRouterRequest(
        endpoint=config.endpoint,
        body=MappingProxyType(body),
        body_bytes=body_bytes,
        body_sha256=body_digest,
        headers=MappingProxyType(headers),
        client_request_id=client_request_id,
        estimated_max_tokens=estimated_max_tokens,
    )


@dataclass(frozen=True, slots=True)
class HTTPResult:
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
        """Execute one HTTP POST."""


class _RejectRedirects(HTTPRedirectHandler):
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
    body = response.read(HTTP_RESPONSE_BODY_LIMIT_BYTES + 1)
    if len(body) > HTTP_RESPONSE_BODY_LIMIT_BYTES:
        raise OpenRouterHTTPResponseBodyTooLarge(status=status)
    return body


def urllib_transport(
    *,
    url: str,
    body: bytes,
    headers: Mapping[str, str],
    timeout: float,
) -> HTTPResult:
    """Execute one redirect-blocked POST using only :mod:`urllib`."""

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
    """Sequential physical-attempt/token ledger with hard reservations."""

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
            raise OpenRouterBudgetExceeded(
                f"resumed request count {request_count} exceeds "
                f"max_requests={self.max_requests}"
            )
        if total_tokens > self.max_total_tokens:
            raise OpenRouterBudgetExceeded(
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
            raise OpenRouterBudgetExceeded(
                "remaining retry-expanded corpus would exceed "
                f"max_requests={self.max_requests}"
            )
        if self.total_tokens + total_tokens > self.max_total_tokens:
            raise OpenRouterBudgetExceeded(
                "remaining retry-expanded corpus's conservative token "
                f"allocation would exceed max_total_tokens={self.max_total_tokens}"
            )

    def reserve(self, estimated_max_tokens: int) -> None:
        if self._reservation is not None:
            raise RuntimeError("only one in-flight reservation is supported")
        if not _is_positive_int(estimated_max_tokens):
            raise ValueError("estimated_max_tokens must be a positive integer")
        if self.request_count + 1 > self.max_requests:
            raise OpenRouterBudgetExceeded(
                f"request would exceed max_requests={self.max_requests}"
            )
        if self.total_tokens + estimated_max_tokens > self.max_total_tokens:
            raise OpenRouterBudgetExceeded(
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
            raise OpenRouterResponseError(
                "usage.total_tokens must be non-negative"
            )
        if charged > self._reservation:
            raise OpenRouterResponseError(
                "provider-reported tokens exceed the conservative "
                "preflight reservation"
            )
        self.request_count += 1
        self.total_tokens += charged
        self._reservation = None

    def rollback(self) -> None:
        self._reservation = None


def _provider_identifier(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or len(value) > 500
        or "\r" in value
        or "\n" in value
    ):
        raise OpenRouterResponseError(
            f"OpenRouter response contains an invalid {field}"
        )
    return value


def _routing_identity(
    raw: Mapping[str, Any],
) -> tuple[Mapping[str, Any], str, str, str, str, int]:
    metadata = raw.get("openrouter_metadata")
    if not isinstance(metadata, Mapping):
        raise OpenRouterResponseError(
            "OpenRouter response lacks opted-in router metadata"
        )
    requested = _provider_identifier(
        metadata.get("requested"),
        field="requested-model identity",
    )
    strategy = _provider_identifier(
        metadata.get("strategy"),
        field="routing strategy",
    )
    attempt = metadata.get("attempt")
    if (
        not isinstance(attempt, int)
        or isinstance(attempt, bool)
        or attempt < 1
    ):
        raise OpenRouterResponseError(
            "OpenRouter response contains an invalid routing attempt"
        )
    endpoints = metadata.get("endpoints")
    if not isinstance(endpoints, Mapping):
        raise OpenRouterResponseError(
            "OpenRouter response lacks endpoint routing metadata"
        )
    available = endpoints.get("available")
    if not isinstance(available, Sequence) or isinstance(
        available,
        (str, bytes),
    ):
        raise OpenRouterResponseError(
            "OpenRouter response lacks available endpoint metadata"
        )
    selected = [
        endpoint
        for endpoint in available
        if isinstance(endpoint, Mapping)
        and endpoint.get("selected") is True
    ]
    if len(selected) != 1:
        raise OpenRouterResponseError(
            "OpenRouter response must identify exactly one selected endpoint"
        )
    selected_provider = _provider_identifier(
        selected[0].get("provider"),
        field="selected upstream provider",
    )
    selected_model = _provider_identifier(
        selected[0].get("model"),
        field="selected upstream model",
    )
    return (
        dict(metadata),
        requested,
        strategy,
        selected_provider,
        selected_model,
        attempt,
    )


@dataclass(frozen=True, slots=True)
class OpenRouterProviderResult:
    """One completion plus gateway and upstream provenance."""

    provider_label = "OpenRouter"

    response: LLMResponse
    model_requested: str
    model_returned: str
    upstream_provider: str
    upstream_model: str
    routing_strategy: str
    routing_attempt: int
    routing_metadata: Mapping[str, Any]
    provider_response_id: str
    provider_created_at: int | float | None
    usage: Mapping[str, Any]
    started_at: str
    completed_at: str
    transport_attempts: int
    request_body_sha256: str
    client_request_id: str
    generation_id: str | None
    cache_status: str | None
    estimated_max_tokens: int
    upstream_provider_constraint: str | None
    provider_preferences: Mapping[str, Any]
    route_constraint_evidence: str
    selected_upstream_identity_semantics: str
    raw_response: Mapping[str, Any]

    def to_audit_record(
        self,
        *,
        acceptance_status: str = "accepted",
    ) -> dict[str, Any]:
        if acceptance_status not in {
            "accepted",
            "rejected_openrouter_identity",
        }:
            raise ValueError("unknown OpenRouter audit acceptance status")
        return {
            "schema_version": 1,
            "provider": "openrouter",
            "gateway": "openrouter",
            "acceptance_status": acceptance_status,
            "request_id": self.response.request_id,
            "prompt_sha256": self.response.prompt_sha256,
            "request_body_sha256": self.request_body_sha256,
            "model_requested": self.model_requested,
            "model_returned": self.model_returned,
            "upstream_provider": self.upstream_provider,
            "upstream_model": self.upstream_model,
            "routing_strategy": self.routing_strategy,
            "routing_attempt": self.routing_attempt,
            "routing_metadata": self.routing_metadata,
            "provider_response_id": self.provider_response_id,
            "provider_created_at": self.provider_created_at,
            "usage": self.usage,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "transport_attempts": self.transport_attempts,
            "attempts": self.transport_attempts,
            "client_request_id": self.client_request_id,
            "generation_id": self.generation_id,
            "cache_status": self.cache_status,
            "estimated_max_tokens": self.estimated_max_tokens,
            "upstream_provider_constraint": self.upstream_provider_constraint,
            "provider_preferences": self.provider_preferences,
            "route_constraint_evidence": self.route_constraint_evidence,
            "selected_upstream_identity_semantics": (
                self.selected_upstream_identity_semantics
            ),
            "raw_response_sha256": self.response.raw_response_sha256,
            "raw_response": self.raw_response,
            "replay_response": self.response.to_dict(),
            "first_party_origin_claimed": False,
        }


def _parse_provider_result(
    *,
    request: LLMRequest,
    prepared: PreparedOpenRouterRequest,
    config: OpenRouterProviderConfig,
    http_result: HTTPResult,
    attempts: int,
    started_at: str,
    completed_at: str,
    secret: str,
) -> OpenRouterProviderResult:
    try:
        raw = json.loads(http_result.body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OpenRouterResponseError(
            "OpenRouter response body is not valid JSON"
        ) from exc
    if not isinstance(raw, Mapping):
        raise OpenRouterResponseError(
            "OpenRouter response body must be a JSON object"
        )
    safe_raw = _redact_provider_value(raw, secret)
    if not isinstance(safe_raw, Mapping):
        raise OpenRouterResponseError(
            "OpenRouter response body must be a JSON object"
        )
    if safe_raw.get("object") != "chat.completion":
        raise OpenRouterResponseError(
            "OpenRouter response is not a chat.completion"
        )
    response_id = _provider_identifier(
        safe_raw.get("id"),
        field="generation ID",
    )
    returned_model = _provider_identifier(
        safe_raw.get("model"),
        field="returned model identity",
    )
    choices = safe_raw.get("choices")
    if not isinstance(choices, Sequence) or isinstance(
        choices,
        (str, bytes),
    ) or len(choices) != 1:
        raise OpenRouterResponseError(
            "OpenRouter response must contain exactly one choice"
        )
    choice = choices[0]
    if not isinstance(choice, Mapping):
        raise OpenRouterResponseError(
            "OpenRouter response choice must be an object"
        )
    if choice.get("index") not in {None, 0}:
        raise OpenRouterResponseError(
            "OpenRouter response choice has an unexpected index"
        )
    if choice.get("finish_reason") != "stop":
        raise OpenRouterResponseError(
            "OpenRouter response did not finish with stop"
        )
    if choice.get("error") is not None:
        raise OpenRouterResponseError(
            "OpenRouter response choice contains a provider error"
        )
    message = choice.get("message")
    if not isinstance(message, Mapping) or message.get("role") != "assistant":
        raise OpenRouterResponseError(
            "OpenRouter response lacks the assistant message identity"
        )
    refusal = message.get("refusal")
    if isinstance(refusal, str) and refusal.strip():
        raise OpenRouterResponseError("OpenRouter response was refused")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise OpenRouterResponseError(
            "OpenRouter response lacks structured message content"
        )
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise OpenRouterResponseError(
            "OpenRouter structured output is not valid JSON"
        ) from exc
    if not isinstance(payload, Mapping) or set(payload) != {"beliefs"}:
        raise OpenRouterResponseError(
            "OpenRouter structured output must contain exactly beliefs"
        )
    (
        routing_metadata,
        _requested_model,
        routing_strategy,
        upstream_provider,
        upstream_model,
        routing_attempt,
    ) = _routing_identity(safe_raw)
    usage = safe_raw.get("usage")
    if not isinstance(usage, Mapping):
        usage = {}
    headers = _redact_provider_value(
        _lower_headers(http_result.headers),
        secret,
    )
    if not isinstance(headers, Mapping):
        headers = {}
    cache_status = headers.get("x-openrouter-cache-status")
    if cache_status is not None:
        cache_status = str(cache_status).upper()
    generation_id = headers.get("x-generation-id")
    if generation_id is not None:
        generation_id = _provider_identifier(
            generation_id,
            field="X-Generation-Id",
        )
    created = safe_raw.get("created")
    if not (
        isinstance(created, (int, float))
        and not isinstance(created, bool)
        and math.isfinite(float(created))
    ):
        created = None
    response = LLMResponse.parse(
        {
            "schema_version": 1,
            "request_id": request.request_id,
            "prompt_sha256": request.prompt_sha256,
            "model_id": returned_model,
            "beliefs": payload.get("beliefs"),
            "raw_response_sha256": sha256(http_result.body).hexdigest(),
        }
    )
    return OpenRouterProviderResult(
        response=response,
        model_requested=config.model,
        model_returned=returned_model,
        upstream_provider=upstream_provider,
        upstream_model=upstream_model,
        routing_strategy=routing_strategy,
        routing_attempt=routing_attempt,
        routing_metadata=routing_metadata,
        provider_response_id=response_id,
        provider_created_at=created,
        usage=dict(usage),
        started_at=started_at,
        completed_at=completed_at,
        transport_attempts=attempts,
        request_body_sha256=prepared.body_sha256,
        client_request_id=prepared.client_request_id,
        generation_id=generation_id,
        cache_status=cache_status,
        estimated_max_tokens=prepared.estimated_max_tokens,
        upstream_provider_constraint=(
            config.upstream_provider or None
        ),
        provider_preferences=config.provider_preferences(),
        route_constraint_evidence=_route_constraint_evidence(config),
        selected_upstream_identity_semantics=(
            _SELECTED_UPSTREAM_IDENTITY_SEMANTICS
        ),
        raw_response=dict(safe_raw),
    )


class OpenRouterChatProvider:
    """Synchronous, budgeted OpenRouter Chat Completions client."""

    provider_name = "openrouter"

    def __init__(
        self,
        config: OpenRouterProviderConfig,
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

    def prepare(self, request: LLMRequest) -> PreparedOpenRouterRequest:
        return prepare_openrouter_request(request, self.config)

    def restore_budget(self, *, request_count: int, total_tokens: int) -> None:
        self.budget.restore(
            request_count=request_count,
            total_tokens=total_tokens,
        )

    def restored_request_count(
        self,
        audits: Sequence[Mapping[str, Any]],
    ) -> int:
        total = 0
        for audit in audits:
            attempts = audit.get("transport_attempts", audit.get("attempts"))
            if (
                not isinstance(attempts, int)
                or isinstance(attempts, bool)
                or attempts < 1
            ):
                raise ValueError(
                    "OpenRouter audit lacks a positive transport-attempt count"
                )
            total += attempts
        return total

    def returned_model_is_consistent(self, returned_model: str) -> bool:
        return returned_model == self.config.model

    def manifest_fields(self) -> dict[str, Any]:
        return {
            "reasoning_effort": self.config.reasoning_effort or None,
            "upstream_provider_constraint": (
                self.config.upstream_provider or None
            ),
            "allow_fallbacks": self.config.allow_fallbacks,
            "require_parameters": self.config.require_parameters,
            "data_collection": self.config.data_collection,
            "zdr": self.config.zdr,
            "response_cache_enabled": False,
            "router_metadata_requested": True,
            "first_party_origin_claimed": False,
            "request_budget_unit": "physical_http_attempt",
            "route_constraint_evidence": _route_constraint_evidence(
                self.config
            ),
            "selected_upstream_identity_semantics": (
                _SELECTED_UPSTREAM_IDENTITY_SEMANTICS
            ),
        }

    def validate_resumed_audit(
        self,
        audit: Mapping[str, Any],
        *,
        request: LLMRequest,
        prepared: PreparedOpenRouterRequest,
    ) -> None:
        """Revalidate retained routing provenance before replaying it."""

        routing = audit.get("routing_metadata")
        if not isinstance(routing, Mapping):
            raise ValueError(
                "resumable OpenRouter audit lacks routing metadata"
            )
        raw_response = audit.get("raw_response")
        if not isinstance(raw_response, Mapping):
            raise ValueError(
                "resumable OpenRouter audit lacks the redacted raw response"
            )
        raw_routing = raw_response.get("openrouter_metadata")
        raw_usage = raw_response.get("usage")
        if not isinstance(raw_usage, Mapping):
            raw_usage = {}
        raw_created = raw_response.get("created")
        if not (
            isinstance(raw_created, (int, float))
            and not isinstance(raw_created, bool)
            and math.isfinite(float(raw_created))
        ):
            raw_created = None
        expected = {
            "model_returned": raw_response.get("model"),
            "routing_metadata": raw_routing,
            "routing_strategy": routing.get("strategy"),
            "routing_attempt": routing.get("attempt"),
            "provider_response_id": raw_response.get("id"),
            "provider_created_at": raw_created,
            "usage": raw_usage,
            "gateway": "openrouter",
            "first_party_origin_claimed": False,
            "upstream_provider_constraint": (
                self.config.upstream_provider or None
            ),
            "provider_preferences": self.config.provider_preferences(),
            "route_constraint_evidence": _route_constraint_evidence(
                self.config
            ),
            "selected_upstream_identity_semantics": (
                _SELECTED_UPSTREAM_IDENTITY_SEMANTICS
            ),
        }
        if prepared.body.get("provider") != expected["provider_preferences"]:
            raise ValueError(
                "current OpenRouter request body does not contain the "
                "configured provider preferences"
            )
        try:
            (
                _metadata,
                requested_model,
                strategy,
                selected_provider,
                selected_model,
                routing_attempt,
            ) = _routing_identity(raw_response)
        except OpenRouterProviderError as exc:
            raise ValueError(
                "resumable OpenRouter audit contains invalid routing metadata"
            ) from exc
        expected.update(
            {
                "model_requested": requested_model,
                "upstream_provider": selected_provider,
                "upstream_model": selected_model,
                "routing_strategy": strategy,
                "routing_attempt": routing_attempt,
            }
        )
        mismatches = {
            field: {
                "retained": audit.get(field),
                "expected": value,
            }
            for field, value in expected.items()
            if audit.get(field) != value
        }
        replay = audit.get("replay_response")
        if not isinstance(replay, Mapping) or (
            replay.get("model_id") != audit.get("model_returned")
        ):
            mismatches["replay_response.model_id"] = {
                "retained": (
                    replay.get("model_id")
                    if isinstance(replay, Mapping)
                    else None
                ),
                "expected": audit.get("model_returned"),
            }
        if isinstance(replay, Mapping):
            raw_choices = raw_response.get("choices")
            raw_content: Any = None
            if (
                isinstance(raw_choices, Sequence)
                and not isinstance(raw_choices, (str, bytes))
                and len(raw_choices) == 1
                and isinstance(raw_choices[0], Mapping)
                and isinstance(raw_choices[0].get("message"), Mapping)
            ):
                raw_content = raw_choices[0]["message"].get("content")
            try:
                raw_payload = (
                    json.loads(raw_content)
                    if isinstance(raw_content, str)
                    else None
                )
            except json.JSONDecodeError:
                raw_payload = None
            canonical_raw_beliefs: Mapping[str, Any] | None = None
            if isinstance(raw_payload, Mapping):
                try:
                    canonical_raw_beliefs = LLMResponse.parse(
                        {
                            "schema_version": 1,
                            "request_id": request.request_id,
                            "prompt_sha256": request.prompt_sha256,
                            "model_id": audit.get("model_returned"),
                            "beliefs": raw_payload.get("beliefs"),
                        }
                    ).beliefs
                except (TypeError, ValueError):
                    canonical_raw_beliefs = None
            if (
                canonical_raw_beliefs is None
                or canonical_raw_beliefs != replay.get("beliefs")
            ):
                mismatches["replay_response.beliefs"] = {
                    "retained": replay.get("beliefs"),
                    "expected": (
                        canonical_raw_beliefs
                    ),
                }
            if replay.get("request_id") != request.request_id:
                mismatches["replay_response.request_id"] = {
                    "retained": replay.get("request_id"),
                    "expected": request.request_id,
                }
            if replay.get("prompt_sha256") != request.prompt_sha256:
                mismatches["replay_response.prompt_sha256"] = {
                    "retained": replay.get("prompt_sha256"),
                    "expected": request.prompt_sha256,
                }
        attempts = audit.get("transport_attempts")
        if (
            not isinstance(attempts, int)
            or isinstance(attempts, bool)
            or attempts < 1
            or audit.get("attempts") != attempts
        ):
            mismatches["transport_attempts"] = {
                "retained": attempts,
                "expected": "a positive integer equal to attempts",
            }
        if self._audit_rejection_reasons(audit, routing):
            mismatches["routing_acceptance"] = {
                "retained": "invalid",
                "expected": "direct, uncached, untransformed route",
            }
        if mismatches:
            raise ValueError(
                "resumable OpenRouter routing audit does not match its raw "
                "response or configured acceptance policy: "
                + _canonical(mismatches)
            )

    def complete(
        self,
        request: LLMRequest,
        *,
        attempt_ledger: DurableProviderAttemptLedger | None = None,
    ) -> OpenRouterProviderResult:
        if not self.config.live_execution:
            raise OpenRouterLiveExecutionRequired(
                "live OpenRouter execution is disabled; set "
                "live_execution=True only after reviewing the request, "
                "route, and budgets"
            )
        prepared = self.prepare(request)
        key: str | None = None
        headers: dict[str, str] | None = None
        started_at = _utc_timestamp(self._epoch_time())
        last_result: HTTPResult | None = None
        try:
            for attempt in range(1, self.config.max_retries + 2):
                self.budget.reserve(prepared.estimated_max_tokens)
                if key is None:
                    loaded = os.environ.get(self.config.api_key_env)
                    if not isinstance(loaded, str) or not loaded.strip():
                        self.budget.rollback()
                        raise OpenRouterMissingAPIKey(
                            "missing API key in environment variable "
                            f"{self.config.api_key_env}"
                        )
                    loaded = loaded.strip()
                    if "\r" in loaded or "\n" in loaded:
                        self.budget.rollback()
                        raise OpenRouterMissingAPIKey(
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
                        raise OpenRouterResponseError(
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
                except OpenRouterHTTPResponseBodyTooLarge as exc:
                    settle(
                        outcome="invalid_response",
                        http_status=exc.status,
                    )
                    raise OpenRouterResponseError(str(exc)) from exc
                except (TimeoutError, ConnectionError, OSError) as exc:
                    settle(
                        outcome="transport_error",
                        response_record={
                            "error_type": type(exc).__name__,
                        },
                    )
                    raise OpenRouterProviderError(
                        "OpenRouter transport outcome is ambiguous after "
                        f"attempt {attempt}; automatic retry is disabled "
                        "because Chat Completions has no documented general "
                        "idempotency guarantee; review before retrying; "
                        f"client_request_id={prepared.client_request_id}; "
                        f"error_type={type(exc).__name__}"
                    ) from exc
                last_result = result
                lowered = _lower_headers(result.headers)
                generation_id = (
                    lowered.get("x-generation-id", "").replace(
                        key,
                        "[redacted]",
                    )
                    or None
                )
                response_digest = sha256(result.body).hexdigest()
                if 200 <= result.status <= 299:
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
                            completed_at=_utc_timestamp(self._epoch_time()),
                            secret=key,
                        )
                    except Exception:
                        settle(
                            outcome="invalid_response",
                            http_status=result.status,
                            server_request_id=generation_id,
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
                    rejection_reasons = self._rejection_reasons(parsed)
                    audit = parsed.to_audit_record(
                        acceptance_status=(
                            "rejected_openrouter_identity"
                            if rejection_reasons
                            else "accepted"
                        )
                    )
                    settle(
                        outcome=(
                            "rejected_provider_result"
                            if rejection_reasons
                            else "success"
                        ),
                        http_status=result.status,
                        server_request_id=generation_id,
                        response_body_sha256=response_digest,
                        response_record=dict(parsed.raw_response),
                        charged_tokens=_usage_total_tokens(parsed.usage),
                        provider_audit=audit,
                    )
                    if rejection_reasons:
                        raise OpenRouterResultRejected(
                            parsed,
                            rejection_reasons,
                        )
                    return parsed
                safe_message = _safe_api_error(result.body, secret=key)
                transient = result.status in _TRANSIENT_STATUSES
                settle(
                    outcome="http_error",
                    automatic_retry_safe=transient,
                    http_status=result.status,
                    server_request_id=generation_id,
                    response_body_sha256=response_digest,
                    response_record={"error": safe_message},
                )
                if (
                    transient
                    and attempt <= self.config.max_retries
                ):
                    self._sleep(
                        self._backoff_seconds(
                            attempt,
                            lowered.get("retry-after"),
                        )
                    )
                    continue
                raise OpenRouterHTTPError(
                    status=result.status,
                    message=safe_message,
                    client_request_id=prepared.client_request_id,
                    generation_id=generation_id,
                )
        except Exception:
            self.budget.rollback()
            raise
        self.budget.rollback()
        raise OpenRouterProviderError(
            "OpenRouter execution ended without a response: "
            f"{last_result!r}"
        )

    def _rejection_reasons(
        self,
        result: OpenRouterProviderResult,
    ) -> tuple[str, ...]:
        reasons = []
        if result.model_returned != self.config.model:
            reasons.append("returned model differs from requested model")
        requested = result.routing_metadata.get("requested")
        if requested != self.config.model:
            reasons.append("router metadata requested model differs")
        if result.routing_strategy != "direct":
            reasons.append("routing strategy is not direct")
        if not _upstream_model_is_consistent(
            self.config.model,
            result.upstream_model,
        ):
            reasons.append("selected upstream model is outside requested family")
        if (
            not self.config.allow_fallbacks
            and result.routing_attempt != 1
        ):
            reasons.append("router fallback occurred while fallbacks were disabled")
        if result.cache_status == "HIT":
            reasons.append("response was served from gateway cache")
        if (
            result.generation_id is not None
            and result.generation_id != result.provider_response_id
        ):
            reasons.append("header and body generation IDs differ")
        route_attempts = result.routing_metadata.get("attempts")
        if (
            not self.config.allow_fallbacks
            and isinstance(route_attempts, Sequence)
            and not isinstance(route_attempts, (str, bytes))
            and len(route_attempts) > 1
        ):
            reasons.append("router metadata records fallback attempts")
        pipeline = result.routing_metadata.get("pipeline")
        if (
            isinstance(pipeline, Sequence)
            and not isinstance(pipeline, (str, bytes))
            and pipeline
        ):
            reasons.append("router pipeline materially transformed the request")
        return tuple(reasons)

    def _audit_rejection_reasons(
        self,
        audit: Mapping[str, Any],
        routing: Mapping[str, Any],
    ) -> tuple[str, ...]:
        reasons = []
        if audit.get("model_returned") != self.config.model:
            reasons.append("returned model differs")
        if routing.get("requested") != self.config.model:
            reasons.append("requested model differs")
        if routing.get("strategy") != "direct":
            reasons.append("strategy is not direct")
        upstream_model = audit.get("upstream_model")
        upstream_provider = audit.get("upstream_provider")
        if (
            not isinstance(upstream_model, str)
            or not _upstream_model_is_consistent(
                self.config.model,
                upstream_model,
            )
        ):
            reasons.append("selected model is outside requested family")
        if (
            not isinstance(upstream_provider, str)
            or not upstream_provider.strip()
        ):
            reasons.append("selected upstream provider is missing")
        if (
            not self.config.allow_fallbacks
            and routing.get("attempt") != 1
        ):
            reasons.append("fallback occurred")
        if str(audit.get("cache_status", "")).upper() == "HIT":
            reasons.append("cache hit")
        generation_id = audit.get("generation_id")
        if (
            generation_id is not None
            and generation_id != audit.get("provider_response_id")
        ):
            reasons.append("generation IDs differ")
        route_attempts = routing.get("attempts")
        if (
            not self.config.allow_fallbacks
            and isinstance(route_attempts, Sequence)
            and not isinstance(route_attempts, (str, bytes))
            and len(route_attempts) > 1
        ):
            reasons.append("router metadata records fallback attempts")
        pipeline = routing.get("pipeline")
        if (
            isinstance(pipeline, Sequence)
            and not isinstance(pipeline, (str, bytes))
            and pipeline
        ):
            reasons.append("router pipeline transformed the request")
        return tuple(reasons)

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
        bounded = min(
            self.config.max_backoff_seconds,
            max(0.0, jittered),
        )
        declared = _retry_after_seconds(
            retry_after,
            now_epoch=self._epoch_time(),
        )
        return max(bounded, declared or 0.0)


class ResumableOpenRouterCompletionProvider(ResumableCompletionProvider):
    """Adaptive OpenRouter journal returning replay-compatible responses."""

    def to_manifest(self) -> dict[str, Any]:
        manifest = super().to_manifest()
        used = tuple(dict.fromkeys(self._used_request_ids))
        audits = tuple(self._audits[request_id] for request_id in used)
        manifest.update(
            {
                "upstream_providers_returned": sorted(
                    {
                        str(audit["upstream_provider"])
                        for audit in audits
                    }
                ),
                "upstream_models_returned": sorted(
                    {
                        str(audit["upstream_model"])
                        for audit in audits
                    }
                ),
                "routing_strategies_returned": sorted(
                    {
                        str(audit["routing_strategy"])
                        for audit in audits
                    }
                ),
            }
        )
        return manifest


def execute_openrouter_requests(
    provider: OpenRouterChatProvider,
    requests: Iterable[LLMRequest],
    *,
    responses_path: str | Path,
    audit_path: str | Path,
    attempts_path: str | Path | None = None,
) -> ExecutionSummary:
    """Execute or resume a static request corpus through OpenRouter."""

    return execute_requests(
        provider,
        requests,
        responses_path=responses_path,
        audit_path=audit_path,
        attempts_path=attempts_path,
    )


def execute_openrouter_jsonl(
    provider: OpenRouterChatProvider,
    requests_path: str | Path,
    *,
    responses_path: str | Path,
    audit_path: str | Path,
    attempts_path: str | Path | None = None,
) -> ExecutionSummary:
    """Execute or resume exported request JSONL through OpenRouter."""

    return execute_jsonl(
        provider,
        requests_path,
        responses_path=responses_path,
        audit_path=audit_path,
        attempts_path=attempts_path,
    )
