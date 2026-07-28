"""First-party Anthropic and Gemini execution for external decoder judgments.

The external-decoder study needs genuinely different provider/model families,
not another projection of the profile writer.  This module implements a small,
dependency-free client for two first-party APIs while preserving the blinded,
content-addressed contracts in :mod:`cape_loop.decoder_study`.

Preparing or planning requests never reads credentials.  A credential is read
from the configured environment variable only after live execution has been
explicitly enabled and immediately before the first HTTP attempt.
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
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
import json
import math
import os
import random
import re
import time

from .decoder_study import (
    ExternalDecoderJudgment,
    ExternalDecoderRequest,
    external_decoder_judgment_from_response,
    external_decoder_llm_request,
    read_external_decoder_judgments,
)
from .file_lock import try_file_lock, unlock_file
from .llm_exchange import ATTRIBUTES, VALUES, LLMResponse


OFFICIAL_SOURCE_RESOLVED_DATE = "2026-07-26"

ANTHROPIC_OFFICIAL_ORIGIN = "https://api.anthropic.com"
ANTHROPIC_DEFAULT_MODEL = "claude-sonnet-5"
ANTHROPIC_MODEL_DOC_URL = (
    "https://platform.claude.com/docs/en/about-claude/models/overview"
)
ANTHROPIC_API_DOC_URL = (
    "https://platform.claude.com/docs/en/api/messages/create"
)
ANTHROPIC_STRUCTURED_OUTPUT_DOC_URL = (
    "https://platform.claude.com/docs/en/build-with-claude/"
    "structured-outputs"
)
ANTHROPIC_EFFORT_DOC_URL = (
    "https://platform.claude.com/docs/en/build-with-claude/effort"
)
ANTHROPIC_SONNET_5_DOC_URL = (
    "https://platform.claude.com/docs/en/about-claude/models/"
    "whats-new-sonnet-5"
)

GEMINI_OFFICIAL_ORIGIN = "https://generativelanguage.googleapis.com"
GEMINI_DEFAULT_MODEL = "gemini-3.6-flash"
GEMINI_MODEL_DOC_URL = "https://ai.google.dev/gemini-api/docs/models"
GEMINI_API_DOC_URL = "https://ai.google.dev/api/generate-content"
GEMINI_STRUCTURED_OUTPUT_DOC_URL = (
    "https://ai.google.dev/gemini-api/docs/generate-content/"
    "structured-output"
)
GEMINI_THINKING_DOC_URL = (
    "https://ai.google.dev/gemini-api/docs/generate-content/thinking"
)

_ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_PROVIDERS = ("anthropic", "google_gemini")
_TRANSIENT_STATUSES = frozenset({408, 409, 425, 429})
_RESERVED_PROVIDER_KEY_ENVS = frozenset(
    {
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
    }
)
_SAFE_PROVIDER_IDENTIFIER = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._:/@_-]{0,255}$"
)
_SENSITIVE_FIELD_NAMES = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "proxy-authorization",
        "x-api-key",
        "x-goog-api-key",
    }
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
        allow_nan=False,
    )


def _digest(value: Any) -> str:
    return sha256(_canonical(value).encode("utf-8")).hexdigest()


def _utc_timestamp(epoch_seconds: float) -> str:
    return (
        datetime.fromtimestamp(epoch_seconds, timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _positive_integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_number(value: object, name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise ValueError(f"{name} must be finite and non-negative")
    return float(value)


@dataclass(frozen=True, slots=True)
class ExternalDecoderSource:
    """A declared first-party provider/model-family source."""

    provider: str
    model_family: str
    default_model: str
    official_origin: str
    default_api_key_env: str
    decoder_family_id: str
    decoder_instance_prefix: str
    official_sources: tuple[str, ...]


EXTERNAL_DECODER_SOURCES: Mapping[str, ExternalDecoderSource] = (
    MappingProxyType(
        {
            "anthropic": ExternalDecoderSource(
                provider="anthropic",
                model_family="Claude",
                default_model=ANTHROPIC_DEFAULT_MODEL,
                official_origin=ANTHROPIC_OFFICIAL_ORIGIN,
                default_api_key_env="ANTHROPIC_API_KEY",
                decoder_family_id="anthropic-claude",
                decoder_instance_prefix="anthropic-claude-api",
                official_sources=(
                    ANTHROPIC_MODEL_DOC_URL,
                    ANTHROPIC_API_DOC_URL,
                    ANTHROPIC_STRUCTURED_OUTPUT_DOC_URL,
                    ANTHROPIC_EFFORT_DOC_URL,
                    ANTHROPIC_SONNET_5_DOC_URL,
                ),
            ),
            "google_gemini": ExternalDecoderSource(
                provider="google_gemini",
                model_family="Gemini",
                default_model=GEMINI_DEFAULT_MODEL,
                official_origin=GEMINI_OFFICIAL_ORIGIN,
                default_api_key_env="GEMINI_API_KEY",
                decoder_family_id="google-gemini",
                decoder_instance_prefix="google-gemini-api",
                official_sources=(
                    GEMINI_MODEL_DOC_URL,
                    GEMINI_API_DOC_URL,
                    GEMINI_STRUCTURED_OUTPUT_DOC_URL,
                    GEMINI_THINKING_DOC_URL,
                ),
            ),
        }
    )
)


class ExternalDecoderProviderError(RuntimeError):
    """Base class for provider errors that never expose credentials."""


class LiveExternalDecoderExecutionRequired(ExternalDecoderProviderError):
    """Raised when a caller has not explicitly authorized live execution."""


class MissingExternalDecoderAPIKey(ExternalDecoderProviderError):
    """Raised when an explicitly enabled provider lacks its environment key."""


class ExternalDecoderBudgetExceeded(ExternalDecoderProviderError):
    """Raised before an execution would exceed a declared hard ceiling."""


class ExternalDecoderResponseError(ExternalDecoderProviderError):
    """Raised for a malformed, blocked, truncated, or otherwise invalid body."""


class HTTPResponseBodyTooLarge(ExternalDecoderProviderError):
    """A response exceeded the fixed wire-body memory safety limit."""

    def __init__(self, *, status: int) -> None:
        self.status = status
        self.limit_bytes = HTTP_RESPONSE_BODY_LIMIT_BYTES
        super().__init__(
            "external-decoder response body exceeded the fixed "
            f"{self.limit_bytes}-byte safety limit"
        )


class ExternalDecoderHTTPError(ExternalDecoderProviderError):
    """A non-retryable HTTP error or an exhausted retry sequence."""

    def __init__(
        self,
        *,
        provider: str,
        status: int,
        message: str,
        client_request_id: str,
        server_request_id: str | None,
    ) -> None:
        self.provider = provider
        self.status = status
        self.client_request_id = client_request_id
        self.server_request_id = server_request_id
        request_suffix = (
            f"; server_request_id={server_request_id}"
            if server_request_id
            else ""
        )
        super().__init__(
            f"{provider} external-decoder request failed with HTTP {status}: "
            f"{message}; client_request_id={client_request_id}"
            f"{request_suffix}"
        )


class ExternalDecoderIdentityMismatch(ExternalDecoderResponseError):
    """A paid response whose returned model is not the configured model."""

    def __init__(self, result: "ExternalDecoderProviderResult") -> None:
        self.result = result
        super().__init__(
            "external decoder returned an inconsistent provider/model "
            f"identity: provider={result.provider!r}, "
            f"requested={result.model_requested!r}, "
            f"returned={result.model_returned!r}; "
            f"client_request_id={result.client_request_id}"
        )


class ExternalDecoderExecutionLocked(ExternalDecoderProviderError):
    """Raised when another writer holds the collection output lock."""


@dataclass(frozen=True, slots=True)
class ExternalDecoderProviderConfig:
    """Validated configuration for one first-party decoder source."""

    provider: str
    model: str | None = None
    api_key_env: str | None = None
    base_url: str | None = None
    allow_custom_base_url: bool = False
    timeout_seconds: float = 180.0
    max_retries: int = 4
    initial_backoff_seconds: float = 0.5
    max_backoff_seconds: float = 8.0
    jitter_fraction: float = 0.25
    max_output_tokens: int = 1_024
    max_requests: int = 900
    max_total_tokens: int = 6_000_000
    live_execution: bool = False

    def __post_init__(self) -> None:
        if self.provider not in EXTERNAL_DECODER_SOURCES:
            raise ValueError(
                "provider must be one of " + ", ".join(_PROVIDERS)
            )
        source = EXTERNAL_DECODER_SOURCES[self.provider]
        model = source.default_model if self.model is None else self.model
        key_env = (
            source.default_api_key_env
            if self.api_key_env is None
            else self.api_key_env
        )
        base_url = (
            source.official_origin
            if self.base_url is None
            else self.base_url
        )
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a nonempty string")
        if not isinstance(key_env, str) or not _ENVIRONMENT_NAME.fullmatch(
            key_env
        ):
            raise ValueError(
                "api_key_env must be a valid environment-variable name"
            )
        if not isinstance(base_url, str):
            raise ValueError("base_url must be an HTTPS origin")
        parsed = urlsplit(base_url)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.port is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("base_url must be an HTTPS origin without userinfo")
        official = urlsplit(source.official_origin)
        is_official = (
            parsed.hostname == official.hostname
            and parsed.path in {"", "/"}
        )
        if not self.allow_custom_base_url and not is_official:
            raise ValueError(
                f"{self.provider} base_url must use the official "
                f"{source.official_origin} origin; custom routing requires "
                "allow_custom_base_url=True"
            )
        if (
            is_official
            and key_env in _RESERVED_PROVIDER_KEY_ENVS
            and key_env != source.default_api_key_env
        ):
            raise ValueError(
                f"{key_env} is reserved for a different provider and cannot "
                f"be used with {self.provider}"
            )
        if (
            self.allow_custom_base_url
            and not is_official
            and key_env in _RESERVED_PROVIDER_KEY_ENVS
        ):
            raise ValueError(
                "a custom base_url requires a dedicated credential "
                "environment variable instead of a reserved provider "
                "default key variable"
            )
        if not isinstance(self.allow_custom_base_url, bool):
            raise ValueError("allow_custom_base_url must be Boolean")
        _nonnegative_number(self.timeout_seconds, "timeout_seconds")
        if self.timeout_seconds == 0:
            raise ValueError("timeout_seconds must be positive")
        if (
            not isinstance(self.max_retries, int)
            or isinstance(self.max_retries, bool)
            or self.max_retries < 0
        ):
            raise ValueError("max_retries must be a non-negative integer")
        initial = _nonnegative_number(
            self.initial_backoff_seconds,
            "initial_backoff_seconds",
        )
        maximum = _nonnegative_number(
            self.max_backoff_seconds,
            "max_backoff_seconds",
        )
        jitter = _nonnegative_number(
            self.jitter_fraction,
            "jitter_fraction",
        )
        if initial > maximum:
            raise ValueError(
                "initial_backoff_seconds cannot exceed max_backoff_seconds"
            )
        if jitter > 1:
            raise ValueError("jitter_fraction cannot exceed 1")
        _positive_integer(self.max_output_tokens, "max_output_tokens")
        _positive_integer(self.max_requests, "max_requests")
        _positive_integer(self.max_total_tokens, "max_total_tokens")
        if not isinstance(self.live_execution, bool):
            raise ValueError("live_execution must be Boolean")
        object.__setattr__(self, "model", model.strip())
        object.__setattr__(self, "api_key_env", key_env)
        object.__setattr__(self, "base_url", base_url.rstrip("/"))

    @property
    def source(self) -> ExternalDecoderSource:
        return EXTERNAL_DECODER_SOURCES[self.provider]

    @property
    def decoder_instance_id(self) -> str:
        return f"{self.source.decoder_instance_prefix}:{self.model}"

    @property
    def endpoint(self) -> str:
        if self.provider == "anthropic":
            return f"{self.base_url}/v1/messages"
        encoded_model = quote(str(self.model), safe="-._")
        return (
            f"{self.base_url}/v1beta/models/"
            f"{encoded_model}:generateContent"
        )

    @property
    def source_descriptor(self) -> str:
        routing = (
            "official-first-party-origin"
            if self.base_url == self.source.official_origin
            else "explicit-custom-routing"
        )
        return (
            f"provider={self.provider};family={self.source.model_family};"
            f"model={self.model};routing={routing};"
            f"source_resolved={OFFICIAL_SOURCE_RESOLVED_DATE}"
        )


def default_external_decoder_configs(
    *,
    live_execution: bool = False,
    max_retries: int = 0,
    max_requests: int = 900,
    max_total_tokens: int = 6_000_000,
) -> tuple[ExternalDecoderProviderConfig, ...]:
    """Return the paper's distinct-family Anthropic and Gemini decoder pair."""

    return (
        ExternalDecoderProviderConfig(
            provider="anthropic",
            live_execution=live_execution,
            max_retries=max_retries,
            max_requests=max_requests,
            max_total_tokens=max_total_tokens,
        ),
        ExternalDecoderProviderConfig(
            provider="google_gemini",
            live_execution=live_execution,
            max_retries=max_retries,
            max_requests=max_requests,
            max_total_tokens=max_total_tokens,
        ),
    )


def _validate_distinct_config_set(
    configs: Sequence[ExternalDecoderProviderConfig],
    *,
    require_pair: bool = True,
) -> None:
    if require_pair and len(configs) < 2:
        raise ValueError("external decoder collection requires two sources")
    if not configs:
        raise ValueError("at least one external decoder source is required")
    families = {item.source.decoder_family_id for item in configs}
    providers = {item.provider for item in configs}
    instances = {item.decoder_instance_id for item in configs}
    key_environments = {item.api_key_env for item in configs}
    if (
        len(families) != len(configs)
        or len(providers) != len(configs)
        or len(instances) != len(configs)
    ):
        raise ValueError(
            "external decoder sources must use distinct provider/model families"
        )
    if len(key_environments) != len(configs):
        raise ValueError(
            "external decoder providers must not share one credential "
            "environment variable"
        )


def decoder_probability_schema(
    *,
    include_numeric_bounds: bool = True,
) -> dict[str, Any]:
    """Return the provider-facing strict probability schema."""

    probability: dict[str, Any] = {
        "type": "number",
        "description": "Probability in the inclusive range [0, 1].",
    }
    if include_numeric_bounds:
        probability.update({"minimum": 0, "maximum": 1})
    vector = {
        "type": "object",
        "additionalProperties": False,
        "properties": {value: probability for value in VALUES},
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
                    attribute: vector for attribute in ATTRIBUTES
                },
                "required": list(ATTRIBUTES),
            }
        },
        "required": ["beliefs"],
    }


@dataclass(frozen=True, slots=True)
class PreparedExternalDecoderRequest:
    """A deterministic provider request containing no authorization material."""

    provider: str
    decoder_instance_id: str
    decoder_family_id: str
    external_request_id: str
    external_request_sha256: str
    prompt_sha256: str
    model: str
    endpoint: str
    body: Mapping[str, Any]
    body_bytes: bytes
    body_sha256: str
    headers: Mapping[str, str]
    client_request_id: str
    estimated_max_tokens: int

    def to_plan_record(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "decoder_instance_id": self.decoder_instance_id,
            "decoder_family_id": self.decoder_family_id,
            "request_id": self.external_request_id,
            "request_sha256": self.external_request_sha256,
            "prompt_sha256": self.prompt_sha256,
            "model": self.model,
            "endpoint": self.endpoint,
            "request_body_sha256": self.body_sha256,
            "client_request_id": self.client_request_id,
            "estimated_max_tokens": self.estimated_max_tokens,
            "credential_read": False,
        }


def prepare_external_decoder_request(
    request: ExternalDecoderRequest,
    config: ExternalDecoderProviderConfig,
) -> PreparedExternalDecoderRequest:
    """Prepare one provider request without consulting the environment."""

    llm_request = external_decoder_llm_request(
        request,
        decoder_instance_id=config.decoder_instance_id,
    )
    payload_text = _canonical(llm_request.payload)
    if config.provider == "anthropic":
        body: dict[str, Any] = {
            "model": config.model,
            "max_tokens": config.max_output_tokens,
            # Sonnet 5 rejects non-default sampling parameters and enables
            # adaptive thinking by default. This bounded classification task
            # needs neither, so omit temperature and disable thinking.
            "thinking": {"type": "disabled"},
            "system": llm_request.system_instruction,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": payload_text}],
                }
            ],
            "output_config": {
                "effort": "low",
                "format": {
                    "type": "json_schema",
                    # Anthropic's raw-JSON-schema path supports strict object
                    # shape. Numeric ranges are enforced again locally.
                    "schema": decoder_probability_schema(
                        include_numeric_bounds=False
                    ),
                },
            },
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "anthropic-version": "2023-06-01",
            "User-Agent": "cape-loop/0.1",
        }
    else:
        body = {
            "systemInstruction": {
                "parts": [{"text": llm_request.system_instruction}]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": payload_text}],
                }
            ],
            "generationConfig": {
                "maxOutputTokens": config.max_output_tokens,
                "thinkingConfig": {"thinkingLevel": "low"},
                "responseFormat": {
                    "text": {
                        "mimeType": "application/json",
                        "schema": decoder_probability_schema(),
                    }
                },
            },
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "cape-loop/0.1",
        }
    body_bytes = _canonical(body).encode("utf-8")
    body_sha256 = sha256(body_bytes).hexdigest()
    identity = sha256(
        (
            "cape-loop-external-decoder-v1\n"
            + config.provider
            + "\n"
            + config.decoder_instance_id
            + "\n"
            + request.request_sha256
            + "\n"
            + llm_request.prompt_sha256
            + "\n"
            + str(config.model)
            + "\n"
            + body_sha256
        ).encode("utf-8")
    ).hexdigest()
    # One UTF-8 byte per input token, framing headroom, and the complete
    # declared output allowance is a deliberately conservative tokenizer-free
    # reservation.
    estimated_max_tokens = (
        len(body_bytes) + 512 + config.max_output_tokens
    )
    headers["X-Client-Request-Id"] = "cape-loop-decoder-" + identity
    return PreparedExternalDecoderRequest(
        provider=config.provider,
        decoder_instance_id=config.decoder_instance_id,
        decoder_family_id=config.source.decoder_family_id,
        external_request_id=request.request_id,
        external_request_sha256=request.request_sha256,
        prompt_sha256=llm_request.prompt_sha256,
        model=str(config.model),
        endpoint=config.endpoint,
        body=body,
        body_bytes=body_bytes,
        body_sha256=body_sha256,
        headers=MappingProxyType(headers),
        client_request_id="cape-loop-decoder-" + identity,
        estimated_max_tokens=estimated_max_tokens,
    )


def _require_retry_expanded_collection_budget(
    requests: Sequence[ExternalDecoderRequest],
    configs: Sequence[ExternalDecoderProviderConfig],
) -> dict[str, tuple[PreparedExternalDecoderRequest, ...]]:
    """Prepare a corpus and reject any source's all-retries upper bound."""

    prepared_by_provider: dict[
        str,
        tuple[PreparedExternalDecoderRequest, ...],
    ] = {}
    for config in configs:
        prepared = tuple(
            prepare_external_decoder_request(request, config)
            for request in requests
        )
        estimated_tokens = sum(
            item.estimated_max_tokens for item in prepared
        )
        theoretical_attempts = len(prepared) * (config.max_retries + 1)
        theoretical_tokens = estimated_tokens * (config.max_retries + 1)
        if theoretical_attempts > config.max_requests:
            raise ExternalDecoderBudgetExceeded(
                f"{config.provider} plan can require "
                f"{theoretical_attempts} physical transport attempts after "
                f"retry expansion, above max_requests={config.max_requests}"
            )
        if theoretical_tokens > config.max_total_tokens:
            raise ExternalDecoderBudgetExceeded(
                f"{config.provider} plan can reserve "
                f"{theoretical_tokens} tokens after retry expansion, above "
                f"max_total_tokens={config.max_total_tokens}"
            )
        prepared_by_provider[config.provider] = prepared
    return prepared_by_provider


def plan_external_decoder_collection(
    requests: Iterable[ExternalDecoderRequest],
    configs: Sequence[ExternalDecoderProviderConfig] | None = None,
) -> dict[str, Any]:
    """Create a deterministic, credential-free two-source execution plan."""

    request_rows = tuple(sorted(requests, key=lambda row: row.request_id))
    if not request_rows:
        raise ValueError("at least one external decoder request is required")
    if len({row.request_id for row in request_rows}) != len(request_rows):
        raise ValueError("external decoder requests contain duplicate IDs")
    configured = (
        default_external_decoder_configs()
        if configs is None
        else tuple(configs)
    )
    _validate_distinct_config_set(configured)
    prepared_by_provider = _require_retry_expanded_collection_budget(
        request_rows,
        configured,
    )
    source_records: list[dict[str, Any]] = []
    for config in sorted(configured, key=lambda item: item.provider):
        prepared = prepared_by_provider[config.provider]
        estimated_tokens = sum(
            item.estimated_max_tokens for item in prepared
        )
        theoretical_attempts = len(prepared) * (config.max_retries + 1)
        theoretical_tokens = estimated_tokens * (config.max_retries + 1)
        source_records.append(
            {
                "provider": config.provider,
                "model_family": config.source.model_family,
                "model": config.model,
                "decoder_family_id": config.source.decoder_family_id,
                "decoder_instance_id": config.decoder_instance_id,
                "source_descriptor": config.source_descriptor,
                "official_origin_locked": (
                    config.base_url == config.source.official_origin
                ),
                "endpoint": config.endpoint,
                "api_key_env": config.api_key_env,
                "credential_read": False,
                "live_execution_configured": config.live_execution,
                "max_requests": config.max_requests,
                "max_total_tokens": config.max_total_tokens,
                "max_output_tokens": config.max_output_tokens,
                "timeout_seconds": config.timeout_seconds,
                "max_retries": config.max_retries,
                "initial_backoff_seconds": (
                    config.initial_backoff_seconds
                ),
                "max_backoff_seconds": config.max_backoff_seconds,
                "jitter_fraction": config.jitter_fraction,
                "request_count": len(prepared),
                "estimated_max_tokens": estimated_tokens,
                "initial_workload_within_declared_budget": (
                    len(prepared) <= config.max_requests
                    and estimated_tokens <= config.max_total_tokens
                ),
                "initial_transport_attempt_count": len(prepared),
                "maximum_attempts_per_request": config.max_retries + 1,
                "theoretical_max_transport_attempts": theoretical_attempts,
                "theoretical_max_tokens_with_all_retries": (
                    theoretical_tokens
                ),
                "all_retry_attempts_within_declared_budget": (
                    theoretical_attempts <= config.max_requests
                    and theoretical_tokens <= config.max_total_tokens
                ),
                "within_declared_budget": (
                    theoretical_attempts <= config.max_requests
                    and theoretical_tokens <= config.max_total_tokens
                ),
                "budget_accounting_unit": "actual_transport_attempt",
                "official_sources": list(config.source.official_sources),
                "requests": [
                    item.to_plan_record() for item in prepared
                ],
            }
        )
    plan: dict[str, Any] = {
        "schema_version": 1,
        "kind": "external-decoder-collection-plan",
        "official_source_resolved_date": OFFICIAL_SOURCE_RESOLVED_DATE,
        "request_count": len(request_rows),
        "source_count": len(source_records),
        "distinct_provider_model_families": True,
        "credential_read": False,
        "sources": source_records,
    }
    plan["plan_sha256"] = _digest(plan)
    return plan


@dataclass(frozen=True, slots=True)
class HTTPResult:
    """Small dependency-free transport result."""

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


class _NoRedirectHandler(HTTPRedirectHandler):
    """Return redirect responses to the caller instead of following them."""

    def redirect_request(
        self,
        req: Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Mapping[str, str],
        newurl: str,
    ) -> None:
        del req, fp, code, msg, headers, newurl
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
    """Execute one HTTP POST without following same- or cross-origin redirects."""

    request = Request(
        url,
        data=body,
        headers=dict(headers),
        method="POST",
    )
    opener = build_opener(_NoRedirectHandler())
    try:
        with opener.open(request, timeout=timeout) as response:
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


class DecoderExecutionBudget:
    """Sequential transport-attempt and token ceiling."""

    def __init__(self, *, max_requests: int, max_total_tokens: int) -> None:
        self.max_requests = _positive_integer(max_requests, "max_requests")
        self.max_total_tokens = _positive_integer(
            max_total_tokens,
            "max_total_tokens",
        )
        self.request_count = 0
        self.total_tokens = 0
        self._reservation: int | None = None

    def restore(self, *, request_count: int, total_tokens: int) -> None:
        if self.request_count or self.total_tokens or self._reservation:
            raise RuntimeError("cannot restore a budget after use")
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
            raise ExternalDecoderBudgetExceeded(
                "resumed request count exceeds max_requests"
            )
        if total_tokens > self.max_total_tokens:
            raise ExternalDecoderBudgetExceeded(
                "resumed token count exceeds max_total_tokens"
            )
        self.request_count = request_count
        self.total_tokens = total_tokens

    def reserve(self, estimated_max_tokens: int) -> None:
        if self._reservation is not None:
            raise RuntimeError("only one in-flight reservation is supported")
        _positive_integer(estimated_max_tokens, "estimated_max_tokens")
        if self.request_count + 1 > self.max_requests:
            raise ExternalDecoderBudgetExceeded(
                f"request would exceed max_requests={self.max_requests}"
            )
        if self.total_tokens + estimated_max_tokens > self.max_total_tokens:
            raise ExternalDecoderBudgetExceeded(
                "request's conservative token reservation would exceed "
                f"max_total_tokens={self.max_total_tokens}"
            )
        self._reservation = estimated_max_tokens

    def commit(self, actual_total_tokens: int | None) -> None:
        if self._reservation is None:
            raise RuntimeError("no in-flight reservation")
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
            raise ExternalDecoderResponseError(
                "provider token usage must be a non-negative integer"
            )
        if charged > self._reservation:
            raise ExternalDecoderResponseError(
                "provider token usage exceeds the conservative reservation"
            )
        self.request_count += 1
        self.total_tokens += charged
        self._reservation = None

    def rollback(self) -> None:
        self._reservation = None


def _lower_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {str(key).lower(): str(value) for key, value in headers.items()}


def _redact_json(value: Any, secrets: Sequence[str]) -> Any:
    if isinstance(value, Mapping):
        redacted: dict[str, Any] = {}
        for key, nested in value.items():
            original_key = str(key)
            safe_key = original_key
            for secret in secrets:
                if secret:
                    safe_key = safe_key.replace(secret, "[REDACTED]")
            redacted[safe_key] = (
                "[REDACTED]"
                if original_key.lower() in _SENSITIVE_FIELD_NAMES
                else _redact_json(nested, secrets)
            )
        return redacted
    if isinstance(value, list):
        return [_redact_json(item, secrets) for item in value]
    if isinstance(value, str):
        redacted = value
        for secret in secrets:
            if secret:
                redacted = redacted.replace(secret, "[REDACTED]")
        return redacted
    return value


def _safe_api_error(body: bytes, *, secret: str) -> str:
    text = body.decode("utf-8", errors="replace")[:2_000]
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        decoded = text
    redacted = _redact_json(decoded, (secret,))
    if isinstance(redacted, Mapping):
        error = redacted.get("error")
        if isinstance(error, Mapping):
            selected = {
                key: error[key]
                for key in ("type", "status", "code", "message")
                if key in error
            }
            return _canonical(selected)[:800]
    return str(redacted)[:800]


def _retry_after_seconds(value: str | None, *, now_epoch: float) -> float:
    if not value:
        return 0.0
    try:
        seconds = float(value)
    except ValueError:
        try:
            date = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return 0.0
        if date.tzinfo is None:
            date = date.replace(tzinfo=timezone.utc)
        return max(0.0, date.timestamp() - now_epoch)
    if not math.isfinite(seconds):
        return 0.0
    return max(0.0, seconds)


def _actual_tokens(provider: str, usage: Mapping[str, Any]) -> int | None:
    if provider == "google_gemini":
        total = usage.get("totalTokenCount")
        if isinstance(total, int) and not isinstance(total, bool) and total >= 0:
            return total
        return None
    names = (
        "input_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
        "output_tokens",
    )
    present = False
    total = 0
    for name in names:
        value = usage.get(name)
        if value is None:
            continue
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            return None
        total += value
        present = True
    return total if present else None


def _provider_identifier(
    value: object,
    *,
    name: str,
    secret: str,
) -> str:
    """Validate an opaque provider identifier without reflecting bad input."""

    if (
        not isinstance(value, str)
        or not value.strip()
        or (bool(secret) and secret in value)
        or not _SAFE_PROVIDER_IDENTIFIER.fullmatch(value.strip())
    ):
        raise ExternalDecoderResponseError(
            f"provider response contains an invalid {name}"
        )
    return value.strip()


def _optional_provider_identifier(
    value: object,
    *,
    name: str,
    secret: str,
) -> str | None:
    if value is None:
        return None
    return _provider_identifier(value, name=name, secret=secret)


def _sanitized_usage(
    provider: str,
    usage: object,
    *,
    secret: str,
) -> dict[str, int]:
    """Retain only documented non-negative integer token counters."""

    if not isinstance(usage, Mapping):
        return {}
    allowed = (
        {
            "input_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
            "output_tokens",
        }
        if provider == "anthropic"
        else {
            "promptTokenCount",
            "cachedContentTokenCount",
            "candidatesTokenCount",
            "toolUsePromptTokenCount",
            "thoughtsTokenCount",
            "totalTokenCount",
        }
    )
    retained: dict[str, int] = {}
    for name in sorted(allowed):
        value = usage.get(name)
        if value is None:
            continue
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
        ):
            # Never reflect a provider-supplied value, especially one that
            # could contain an echoed credential.
            raise ExternalDecoderResponseError(
                f"provider response contains invalid usage field {name}"
            )
        retained[name] = value
    if secret and secret in _canonical(retained):
        raise ExternalDecoderResponseError(
            "provider usage metadata failed credential redaction"
        )
    return retained


def _strict_payload(text: str, provider: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ExternalDecoderResponseError(
            f"{provider} structured output is not valid JSON"
        ) from exc
    if not isinstance(payload, Mapping) or set(payload) != {"beliefs"}:
        raise ExternalDecoderResponseError(
            f"{provider} structured output must contain exactly beliefs"
        )
    return payload


def _parse_anthropic_response(
    raw: Mapping[str, Any],
    *,
    secret: str,
) -> tuple[str, str, Mapping[str, Any], Mapping[str, Any]]:
    if raw.get("type") != "message" or raw.get("role") != "assistant":
        raise ExternalDecoderResponseError(
            "Anthropic response lacks the Messages API assistant identity"
        )
    response_id = _provider_identifier(
        raw.get("id"),
        name="Anthropic message ID",
        secret=secret,
    )
    model = _provider_identifier(
        raw.get("model"),
        name="Anthropic model identity",
        secret=secret,
    )
    if raw.get("stop_reason") != "end_turn":
        raise ExternalDecoderResponseError(
            "Anthropic response did not complete with end_turn"
        )
    content = raw.get("content")
    if not isinstance(content, list):
        raise ExternalDecoderResponseError(
            "Anthropic response content must be an array"
        )
    text_blocks = [
        item.get("text")
        for item in content
        if isinstance(item, Mapping) and item.get("type") == "text"
    ]
    if len(text_blocks) != 1 or not isinstance(text_blocks[0], str):
        raise ExternalDecoderResponseError(
            "Anthropic response must contain exactly one text block"
        )
    usage = _sanitized_usage(
        "anthropic",
        raw.get("usage"),
        secret=secret,
    )
    return (
        model,
        response_id,
        _strict_payload(text_blocks[0], "anthropic"),
        usage,
    )


def _parse_gemini_response(
    raw: Mapping[str, Any],
    *,
    secret: str,
) -> tuple[str, str, Mapping[str, Any], Mapping[str, Any]]:
    feedback = raw.get("promptFeedback")
    if isinstance(feedback, Mapping) and feedback.get("blockReason"):
        raise ExternalDecoderResponseError(
            "Gemini blocked the external-decoder prompt"
        )
    model = _provider_identifier(
        raw.get("modelVersion"),
        name="Gemini modelVersion",
        secret=secret,
    )
    response_id = _provider_identifier(
        raw.get("responseId"),
        name="Gemini responseId",
        secret=secret,
    )
    candidates = raw.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 1:
        raise ExternalDecoderResponseError(
            "Gemini response must contain exactly one candidate"
        )
    candidate = candidates[0]
    if not isinstance(candidate, Mapping):
        raise ExternalDecoderResponseError(
            "Gemini candidate must be an object"
        )
    if candidate.get("finishReason") != "STOP":
        raise ExternalDecoderResponseError(
            "Gemini candidate did not finish with STOP"
        )
    content = candidate.get("content")
    if not isinstance(content, Mapping) or content.get("role") != "model":
        raise ExternalDecoderResponseError(
            "Gemini response lacks the model content identity"
        )
    parts = content.get("parts")
    if not isinstance(parts, list):
        raise ExternalDecoderResponseError(
            "Gemini response parts must be an array"
        )
    text_parts = [
        item.get("text")
        for item in parts
        if (
            isinstance(item, Mapping)
            and item.get("thought") is not True
            and "text" in item
        )
    ]
    if len(text_parts) != 1 or not isinstance(text_parts[0], str):
        raise ExternalDecoderResponseError(
            "Gemini response must contain exactly one non-thought text part"
        )
    status = raw.get("modelStatus")
    if isinstance(status, Mapping):
        stage = status.get("modelStage")
        if stage not in {None, "STABLE"}:
            raise ExternalDecoderResponseError(
                "Gemini returned a non-stable or invalid model stage"
            )
    usage = _sanitized_usage(
        "google_gemini",
        raw.get("usageMetadata"),
        secret=secret,
    )
    return (
        model,
        response_id,
        _strict_payload(text_parts[0], "google_gemini"),
        usage,
    )


def returned_model_is_consistent(
    provider: str,
    requested_model: str,
    returned_model: str,
) -> bool:
    """Require a pinned model identity, allowing Gemini's ``models/`` prefix."""

    if provider == "anthropic":
        return returned_model == requested_model
    return returned_model in {
        requested_model,
        f"models/{requested_model}",
    }


@dataclass(frozen=True, slots=True)
class ExternalDecoderProviderResult:
    """A validated judgment plus safe provider-side reproducibility data."""

    judgment: ExternalDecoderJudgment
    llm_response: LLMResponse
    provider: str
    model_requested: str
    model_returned: str
    provider_response_id: str
    usage: Mapping[str, Any]
    started_at: str
    completed_at: str
    attempts: int
    request_body_sha256: str
    client_request_id: str
    server_request_id: str | None
    estimated_max_tokens: int
    raw_response: Mapping[str, Any]

    def to_audit_record(
        self,
        *,
        acceptance_status: str = "accepted",
    ) -> dict[str, Any]:
        if acceptance_status not in {
            "accepted",
            "rejected_identity_mismatch",
        }:
            raise ValueError("unknown external-decoder acceptance status")
        return {
            "schema_version": 1,
            "kind": "external-decoder-provider-audit",
            "acceptance_status": acceptance_status,
            "provider": self.provider,
            "request_id": self.judgment.request_id,
            "request_sha256": self.judgment.request_sha256,
            "prompt_sha256": self.llm_response.prompt_sha256,
            "decoder_instance_id": self.judgment.decoder_instance_id,
            "decoder_family_id": self.judgment.decoder_family_id,
            "source_descriptor": self.judgment.source_descriptor,
            "request_body_sha256": self.request_body_sha256,
            "model_requested": self.model_requested,
            "model_returned": self.model_returned,
            "provider_response_id": self.provider_response_id,
            "usage": dict(self.usage),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "attempts": self.attempts,
            "client_request_id": self.client_request_id,
            "server_request_id": self.server_request_id,
            "estimated_max_tokens": self.estimated_max_tokens,
            "judgment": self.judgment.to_dict(),
            "llm_response": self.llm_response.to_dict(),
            "raw_response": self.raw_response,
        }


class ExternalDecoderProvider:
    """Synchronous, budgeted Anthropic or Gemini decoder client."""

    def __init__(
        self,
        config: ExternalDecoderProviderConfig,
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
        self.budget = DecoderExecutionBudget(
            max_requests=config.max_requests,
            max_total_tokens=config.max_total_tokens,
        )

    def prepare(
        self,
        request: ExternalDecoderRequest,
    ) -> PreparedExternalDecoderRequest:
        return prepare_external_decoder_request(request, self.config)

    def restore_budget(self, *, request_count: int, total_tokens: int) -> None:
        self.budget.restore(
            request_count=request_count,
            total_tokens=total_tokens,
        )

    def _load_live_key(self) -> str:
        if not self.config.live_execution:
            raise LiveExternalDecoderExecutionRequired(
                "live external-decoder execution is disabled; set "
                "live_execution=True only after reviewing the keyless plan "
                "and provider budgets"
            )
        key = os.environ.get(str(self.config.api_key_env))
        if not isinstance(key, str) or not key.strip():
            raise MissingExternalDecoderAPIKey(
                "missing API key in environment variable "
                f"{self.config.api_key_env}"
            )
        key = key.strip()
        if "\r" in key or "\n" in key:
            raise MissingExternalDecoderAPIKey(
                f"{self.config.api_key_env} contains an invalid newline"
            )
        return key

    def preflight_live_availability(self) -> None:
        """Validate live authorization and key presence without retaining it."""

        key = self._load_live_key()
        # Do not return, cache, hash, log, or otherwise retain the secret.
        del key

    def complete(
        self,
        request: ExternalDecoderRequest,
        *,
        attempt_ledger: _DurableAttemptLedger | None = None,
    ) -> ExternalDecoderProviderResult:
        """Execute one live, blinded decoder request after explicit opt-in."""

        prepared = self.prepare(request)
        key: str | None = None
        headers: dict[str, str] | None = None
        started_at = _utc_timestamp(self._epoch_time())
        for attempt in range(1, self.config.max_retries + 2):
            self.budget.reserve(prepared.estimated_max_tokens)
            if key is None:
                try:
                    key = self._load_live_key()
                except Exception:
                    self.budget.rollback()
                    raise
                headers = dict(prepared.headers)
                if self.config.provider == "anthropic":
                    headers["x-api-key"] = key
                else:
                    headers["x-goog-api-key"] = key
            assert headers is not None
            attempt_started_at = _utc_timestamp(self._epoch_time())
            try:
                attempt_id = (
                    attempt_ledger.start(
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
                http_status: int | None = None,
                server_request_id: str | None = None,
                response_body_sha256: str | None = None,
                charged_tokens: int | None = None,
                provider_audit: Mapping[str, Any] | None = None,
            ) -> None:
                charged = (
                    prepared.estimated_max_tokens
                    if charged_tokens is None
                    else charged_tokens
                )
                if charged > prepared.estimated_max_tokens:
                    raise ExternalDecoderResponseError(
                        "provider token usage exceeds the conservative "
                        "reservation; manual review is required"
                    )
                if attempt_ledger is not None:
                    if attempt_id is None:
                        raise RuntimeError("attempt ledger identity is missing")
                    attempt_ledger.settle(
                        attempt_id,
                        settled_at=_utc_timestamp(self._epoch_time()),
                        outcome=outcome,
                        http_status=http_status,
                        charged_tokens=charged,
                        server_request_id=server_request_id,
                        response_body_sha256=response_body_sha256,
                        provider_audit=provider_audit,
                    )
                self.budget.commit(charged)

            try:
                http_result = self._transport(
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
                raise ExternalDecoderResponseError(str(exc)) from exc
            except (TimeoutError, ConnectionError, OSError) as exc:
                settle(outcome="transport_error")
                # A timeout/connection failure can occur after the provider
                # accepted and billed the request. Without provider-supported
                # idempotency there is no safe automatic retry, even inside
                # this process. The durable nonfinal settlement makes the
                # same ambiguity block a later resume as well.
                raise ExternalDecoderProviderError(
                    f"{self.config.provider} transport outcome is ambiguous; "
                    "automatic retry is disabled and manual review is "
                    "required; "
                    f"client_request_id={prepared.client_request_id}; "
                    f"error_type={type(exc).__name__}"
                ) from exc

            response_digest = sha256(http_result.body).hexdigest()
            lowered_headers = _lower_headers(http_result.headers)
            try:
                server_request_id = _optional_provider_identifier(
                    lowered_headers.get("request-id")
                    or lowered_headers.get("x-request-id"),
                    name="server request ID",
                    secret=key,
                )
            except ExternalDecoderResponseError:
                settle(
                    outcome="invalid_provider_metadata",
                    http_status=http_result.status,
                    response_body_sha256=response_digest,
                )
                raise

            if not 200 <= http_result.status <= 299:
                settle(
                    outcome="http_error",
                    http_status=http_result.status,
                    server_request_id=server_request_id,
                    response_body_sha256=response_digest,
                )
                if (
                    http_result.status in _TRANSIENT_STATUSES
                    or http_result.status >= 500
                ) and attempt <= self.config.max_retries:
                    self._sleep(
                        self._backoff_seconds(
                            attempt,
                            lowered_headers.get("retry-after"),
                        )
                    )
                    continue
                raise ExternalDecoderHTTPError(
                    provider=self.config.provider,
                    status=http_result.status,
                    message=_safe_api_error(
                        http_result.body,
                        secret=key,
                    ),
                    client_request_id=prepared.client_request_id,
                    server_request_id=server_request_id,
                )

            try:
                try:
                    raw = json.loads(http_result.body)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ExternalDecoderResponseError(
                        f"{self.config.provider} response is not valid JSON"
                    ) from exc
                if not isinstance(raw, Mapping):
                    raise ExternalDecoderResponseError(
                        f"{self.config.provider} response must be an object"
                    )
                if self.config.provider == "anthropic":
                    (
                        returned_model,
                        response_id,
                        payload,
                        usage,
                    ) = _parse_anthropic_response(raw, secret=key)
                else:
                    (
                        returned_model,
                        response_id,
                        payload,
                        usage,
                    ) = _parse_gemini_response(raw, secret=key)
                llm_request = external_decoder_llm_request(
                    request,
                    decoder_instance_id=self.config.decoder_instance_id,
                )
                llm_response = LLMResponse.parse(
                    {
                        "schema_version": 1,
                        "request_id": llm_request.request_id,
                        "prompt_sha256": llm_request.prompt_sha256,
                        "model_id": returned_model,
                        "beliefs": payload["beliefs"],
                        "raw_response_sha256": response_digest,
                    }
                )
                judgment = external_decoder_judgment_from_response(
                    request,
                    llm_response,
                    decoder_instance_id=self.config.decoder_instance_id,
                    decoder_family_id=(
                        self.config.source.decoder_family_id
                    ),
                    source_descriptor=self.config.source_descriptor,
                )
                completed_at = _utc_timestamp(self._epoch_time())
                result = ExternalDecoderProviderResult(
                    judgment=judgment,
                    llm_response=llm_response,
                    provider=self.config.provider,
                    model_requested=str(self.config.model),
                    model_returned=returned_model,
                    provider_response_id=response_id,
                    usage=usage,
                    started_at=started_at,
                    completed_at=completed_at,
                    attempts=attempt,
                    request_body_sha256=prepared.body_sha256,
                    client_request_id=prepared.client_request_id,
                    server_request_id=server_request_id,
                    estimated_max_tokens=prepared.estimated_max_tokens,
                    raw_response=_redact_json(raw, (key,)),
                )
            except (TypeError, ValueError, ExternalDecoderResponseError):
                settle(
                    outcome="invalid_response",
                    http_status=http_result.status,
                    server_request_id=server_request_id,
                    response_body_sha256=response_digest,
                )
                raise

            mismatch = not returned_model_is_consistent(
                self.config.provider,
                str(self.config.model),
                returned_model,
            )
            provider_audit = result.to_audit_record(
                acceptance_status=(
                    "rejected_identity_mismatch"
                    if mismatch
                    else "accepted"
                )
            )
            settle(
                outcome=(
                    "identity_mismatch" if mismatch else "success"
                ),
                http_status=http_result.status,
                server_request_id=server_request_id,
                response_body_sha256=response_digest,
                charged_tokens=_actual_tokens(
                    self.config.provider,
                    usage,
                ),
                provider_audit=provider_audit,
            )
            if mismatch:
                raise ExternalDecoderIdentityMismatch(result)
            return result

        raise ExternalDecoderProviderError(
            f"{self.config.provider} execution ended without a response"
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
        bounded = min(
            self.config.max_backoff_seconds,
            max(0.0, jittered),
        )
        declared = _retry_after_seconds(
            retry_after,
            now_epoch=self._epoch_time(),
        )
        return min(
            self.config.max_backoff_seconds,
            max(bounded, declared),
        )


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    """Append one canonical line durably using a single O_APPEND writer."""

    path.parent.mkdir(parents=True, exist_ok=True)
    line = (_canonical(value) + "\n").encode("utf-8")
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


def _repair_trailing_jsonl(path: Path) -> bool:
    """Repair only one crash-truncated final JSONL tail."""

    if not path.exists():
        return False
    material = path.read_bytes()
    if not material or material.endswith(b"\n"):
        return False
    tail_start = material.rfind(b"\n") + 1
    tail = material[tail_start:]
    try:
        decoded = json.loads(tail.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        with path.open("r+b") as handle:
            handle.truncate(tail_start)
            handle.flush()
            os.fsync(handle.fileno())
    else:
        if not isinstance(decoded, Mapping):
            raise ValueError(
                f"{path}: complete final JSONL value must be an object"
            )
        descriptor = os.open(path, os.O_WRONLY | os.O_APPEND)
        try:
            os.write(descriptor, b"\n")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    return True


class _ExclusiveCollectionLock:
    """Process-scoped advisory lock covering reconciliation and execution."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._descriptor: int | None = None

    def __enter__(self) -> "_ExclusiveCollectionLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            locked = try_file_lock(descriptor)
        except Exception:
            os.close(descriptor)
            raise
        if not locked:
            os.close(descriptor)
            raise ExternalDecoderExecutionLocked(
                "another external-decoder collector holds the output lock; "
                "wait for it to finish or review the owning process"
            )
        # The lock file is deliberately opaque. ``try_file_lock`` preserves a
        # permanent byte-zero marker for Windows byte-range locking; rewriting
        # or truncating that byte while it is locked is not portable.
        self._descriptor = descriptor
        return self

    def __exit__(self, *_: object) -> None:
        descriptor = self._descriptor
        self._descriptor = None
        if descriptor is not None:
            try:
                unlock_file(descriptor)
            finally:
                os.close(descriptor)


_ATTEMPT_OUTCOMES = frozenset(
    {
        "transport_error",
        "http_error",
        "invalid_provider_metadata",
        "invalid_response",
        "identity_mismatch",
        "success",
    }
)


class _DurableAttemptLedger:
    """Durable started/settled records for every actual HTTP attempt."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.starts: dict[str, dict[str, Any]] = {}
        self.settlements: dict[str, dict[str, Any]] = {}
        self._ordinals: dict[tuple[str, str], int] = {}
        if path.exists():
            self._read()

    def _read(self) -> None:
        with self.path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{self.path}:{line_number}: {exc}"
                    ) from exc
                if not isinstance(raw, Mapping):
                    raise ValueError(
                        f"{self.path}:{line_number}: attempt event "
                        "must be an object"
                    )
                if (
                    raw.get("schema_version") != 1
                    or raw.get("kind")
                    != "external-decoder-transport-attempt"
                ):
                    raise ValueError(
                        f"{self.path}:{line_number}: unknown attempt schema"
                    )
                event = raw.get("event")
                attempt_id = raw.get("attempt_id")
                if (
                    event not in {"started", "settled"}
                    or not isinstance(attempt_id, str)
                    or len(attempt_id) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in attempt_id
                    )
                ):
                    raise ValueError(
                        f"{self.path}:{line_number}: invalid attempt identity"
                    )
                if event == "started":
                    self._parse_start(dict(raw), line_number)
                else:
                    self._parse_settlement(dict(raw), line_number)

    def _parse_start(
        self,
        raw: dict[str, Any],
        line_number: int,
    ) -> None:
        allowed = {
            "schema_version",
            "kind",
            "event",
            "attempt_id",
            "provider",
            "request_id",
            "request_sha256",
            "prompt_sha256",
            "decoder_instance_id",
            "request_body_sha256",
            "model_requested",
            "client_request_id",
            "estimated_max_tokens",
            "attempt_ordinal",
            "started_at",
        }
        if set(raw) != allowed:
            raise ValueError(
                f"{self.path}:{line_number}: invalid started-attempt fields"
            )
        if self.unresolved_attempt_ids:
            raise ValueError(
                f"{self.path}:{line_number}: an unresolved attempt occurs "
                "before a later event; manual review is required"
            )
        attempt_id = raw["attempt_id"]
        if attempt_id in self.starts or attempt_id in self.settlements:
            raise ValueError(
                f"{self.path}:{line_number}: duplicate attempt identity"
            )
        provider = raw["provider"]
        request_id = raw["request_id"]
        ordinal = raw["attempt_ordinal"]
        estimate = raw["estimated_max_tokens"]
        if (
            not isinstance(provider, str)
            or not isinstance(request_id, str)
            or not isinstance(ordinal, int)
            or isinstance(ordinal, bool)
            or ordinal <= 0
            or not isinstance(estimate, int)
            or isinstance(estimate, bool)
            or estimate <= 0
        ):
            raise ValueError(
                f"{self.path}:{line_number}: invalid started-attempt values"
            )
        key = (provider, request_id)
        expected_ordinal = self._ordinals.get(key, 0) + 1
        if ordinal != expected_ordinal:
            raise ValueError(
                f"{self.path}:{line_number}: nonsequential attempt ordinal"
            )
        binding = {
            key: raw[key]
            for key in allowed
            if key not in {"schema_version", "kind", "event", "attempt_id"}
        }
        if attempt_id != _digest(binding):
            raise ValueError(
                f"{self.path}:{line_number}: attempt digest mismatch"
            )
        self.starts[attempt_id] = raw
        self._ordinals[key] = ordinal

    def _parse_settlement(
        self,
        raw: dict[str, Any],
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
            "provider_audit",
        }
        if set(raw) != allowed:
            raise ValueError(
                f"{self.path}:{line_number}: invalid settlement fields"
            )
        attempt_id = raw["attempt_id"]
        if attempt_id not in self.starts:
            raise ValueError(
                f"{self.path}:{line_number}: settlement lacks its start"
            )
        if attempt_id in self.settlements:
            raise ValueError(
                f"{self.path}:{line_number}: duplicate attempt settlement"
            )
        outcome = raw["outcome"]
        charged = raw["charged_tokens"]
        status = raw["http_status"]
        digest = raw["response_body_sha256"]
        provider_audit = raw["provider_audit"]
        start = self.starts[attempt_id]
        conservative_charge = start["estimated_max_tokens"]
        if outcome not in _ATTEMPT_OUTCOMES:
            raise ValueError(
                f"{self.path}:{line_number}: invalid attempt outcome"
            )
        if (
            not isinstance(charged, int)
            or isinstance(charged, bool)
            or charged < 0
        ):
            raise ValueError(
                f"{self.path}:{line_number}: invalid charged token count"
            )
        if status is not None and (
            not isinstance(status, int)
            or isinstance(status, bool)
            or status < 100
            or status > 599
        ):
            raise ValueError(
                f"{self.path}:{line_number}: invalid HTTP status"
            )
        if digest is not None and (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in digest
            )
        ):
            raise ValueError(
                f"{self.path}:{line_number}: invalid response digest"
            )
        if provider_audit is not None and not isinstance(
            provider_audit,
            Mapping,
        ):
            raise ValueError(
                f"{self.path}:{line_number}: invalid embedded audit"
            )
        if outcome in {"success", "identity_mismatch"}:
            if not isinstance(provider_audit, Mapping):
                raise ValueError(
                    f"{self.path}:{line_number}: final attempt lacks audit"
                )
            expected_status = (
                "accepted"
                if outcome == "success"
                else "rejected_identity_mismatch"
            )
            if provider_audit.get("acceptance_status") != expected_status:
                raise ValueError(
                    f"{self.path}:{line_number}: attempt/audit status mismatch"
                )
            if (
                provider_audit.get("estimated_max_tokens")
                != conservative_charge
            ):
                raise ValueError(
                    f"{self.path}:{line_number}: attempt/audit token "
                    "reservation mismatch"
                )
            usage = provider_audit.get("usage")
            if not isinstance(usage, Mapping):
                raise ValueError(
                    f"{self.path}:{line_number}: final audit lacks usage"
                )
            actual_charge = _actual_tokens(start["provider"], usage)
            expected_charge = (
                conservative_charge
                if actual_charge is None
                else actual_charge
            )
            llm_response = provider_audit.get("llm_response")
            if not isinstance(llm_response, Mapping):
                raise ValueError(
                    f"{self.path}:{line_number}: final audit lacks its "
                    "provider-neutral response"
                )
            if (
                not isinstance(status, int)
                or not 200 <= status <= 299
                or not isinstance(digest, str)
                or digest != llm_response.get("raw_response_sha256")
                or raw["server_request_id"]
                != provider_audit.get("server_request_id")
            ):
                raise ValueError(
                    f"{self.path}:{line_number}: attempt response "
                    "metadata does not match the embedded audit"
                )
        elif provider_audit is not None:
            raise ValueError(
                f"{self.path}:{line_number}: nonfinal attempt has an audit"
            )
        else:
            expected_charge = conservative_charge
        if charged != expected_charge:
            raise ValueError(
                f"{self.path}:{line_number}: charged token count does not "
                "match provider usage or the conservative reservation"
            )
        if charged > conservative_charge:
            raise ValueError(
                f"{self.path}:{line_number}: charged token count exceeds "
                "the conservative reservation"
            )
        self.settlements[attempt_id] = raw

    @property
    def unresolved_attempt_ids(self) -> tuple[str, ...]:
        return tuple(
            attempt_id
            for attempt_id in self.starts
            if attempt_id not in self.settlements
        )

    def start(
        self,
        prepared: PreparedExternalDecoderRequest,
        *,
        started_at: str,
    ) -> str:
        if self.unresolved_attempt_ids:
            raise ValueError(
                "transport attempt ledger has an unresolved started attempt; "
                "manual review is required before any retry"
            )
        key = (prepared.provider, prepared.external_request_id)
        ordinal = self._ordinals.get(key, 0) + 1
        binding = {
            "provider": prepared.provider,
            "request_id": prepared.external_request_id,
            "request_sha256": prepared.external_request_sha256,
            "prompt_sha256": prepared.prompt_sha256,
            "decoder_instance_id": prepared.decoder_instance_id,
            "request_body_sha256": prepared.body_sha256,
            "model_requested": prepared.model,
            "client_request_id": prepared.client_request_id,
            "estimated_max_tokens": prepared.estimated_max_tokens,
            "attempt_ordinal": ordinal,
            "started_at": started_at,
        }
        attempt_id = _digest(binding)
        record = {
            "schema_version": 1,
            "kind": "external-decoder-transport-attempt",
            "event": "started",
            "attempt_id": attempt_id,
            **binding,
        }
        _append_jsonl(self.path, record)
        self.starts[attempt_id] = record
        self._ordinals[key] = ordinal
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
        provider_audit: Mapping[str, Any] | None,
    ) -> None:
        if attempt_id not in self.starts:
            raise ValueError("cannot settle an unknown transport attempt")
        if attempt_id in self.settlements:
            raise ValueError("transport attempt is already settled")
        record = {
            "schema_version": 1,
            "kind": "external-decoder-transport-attempt",
            "event": "settled",
            "attempt_id": attempt_id,
            "settled_at": settled_at,
            "outcome": outcome,
            "http_status": http_status,
            "charged_tokens": charged_tokens,
            "server_request_id": server_request_id,
            "response_body_sha256": response_body_sha256,
            "provider_audit": (
                dict(provider_audit)
                if provider_audit is not None
                else None
            ),
        }
        # Parse before writing so malformed local values never enter the
        # durable ledger.
        shadow = dict(self.settlements)
        self._parse_settlement(record, line_number=0)
        try:
            _append_jsonl(self.path, record)
        except Exception:
            self.settlements = shadow
            raise

    def validate_bindings(
        self,
        providers: Mapping[str, ExternalDecoderProvider],
        requests: Mapping[str, ExternalDecoderRequest],
    ) -> None:
        for attempt_id, start in self.starts.items():
            provider_name = start["provider"]
            request_id = start["request_id"]
            provider = providers.get(provider_name)
            request = requests.get(request_id)
            if provider is None or request is None:
                raise ValueError(
                    "attempt ledger references an unexpected provider/request"
                )
            prepared = provider.prepare(request)
            expected = {
                "request_sha256": prepared.external_request_sha256,
                "prompt_sha256": prepared.prompt_sha256,
                "decoder_instance_id": prepared.decoder_instance_id,
                "request_body_sha256": prepared.body_sha256,
                "model_requested": prepared.model,
                "client_request_id": prepared.client_request_id,
                "estimated_max_tokens": prepared.estimated_max_tokens,
            }
            if any(start.get(name) != value for name, value in expected.items()):
                raise ValueError(
                    "attempt ledger does not match the current "
                    "request/provider configuration"
                )
            settlement = self.settlements.get(attempt_id)
            if settlement is None:
                continue
            audit = settlement.get("provider_audit")
            if isinstance(audit, Mapping):
                if (
                    audit.get("provider") != provider_name
                    or audit.get("request_id") != request_id
                    or audit.get("request_body_sha256")
                    != prepared.body_sha256
                    or audit.get("model_requested") != prepared.model
                    or audit.get("client_request_id")
                    != prepared.client_request_id
                    or audit.get("estimated_max_tokens")
                    != prepared.estimated_max_tokens
                ):
                    raise ValueError(
                        "attempt ledger embedded audit binding mismatch"
                    )

    def accounting_for(self, provider: str) -> tuple[int, int]:
        attempt_ids = [
            attempt_id
            for attempt_id, start in self.starts.items()
            if start["provider"] == provider
        ]
        settlements = [
            self.settlements[attempt_id]
            for attempt_id in attempt_ids
            if attempt_id in self.settlements
        ]
        return (
            len(settlements),
            sum(int(row["charged_tokens"]) for row in settlements),
        )

    def embedded_final_audits(
        self,
    ) -> dict[tuple[str, str], dict[str, Any]]:
        records: dict[tuple[str, str], dict[str, Any]] = {}
        for attempt_id, settlement in self.settlements.items():
            audit = settlement.get("provider_audit")
            if not isinstance(audit, Mapping):
                continue
            start = self.starts[attempt_id]
            key = (start["provider"], start["request_id"])
            if key in records:
                raise ValueError(
                    "attempt ledger contains multiple final audits for one "
                    "provider/request"
                )
            records[key] = dict(audit)
        return records


def _read_audits(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    records: dict[tuple[str, str], dict[str, Any]] = {}
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            if not isinstance(raw, Mapping):
                raise ValueError(
                    f"{path}:{line_number}: audit must be an object"
                )
            if (
                raw.get("schema_version") != 1
                or raw.get("kind")
                != "external-decoder-provider-audit"
            ):
                raise ValueError(
                    f"{path}:{line_number}: unknown audit schema"
                )
            provider = raw.get("provider")
            request_id = raw.get("request_id")
            if not isinstance(provider, str) or not isinstance(
                request_id,
                str,
            ):
                raise ValueError(
                    f"{path}:{line_number}: audit identity is missing"
                )
            key = (provider, request_id)
            if key in records:
                raise ValueError(
                    f"{path}:{line_number}: duplicate audit identity {key}"
                )
            records[key] = dict(raw)
    return records


def _validate_resumed_audit(
    audit: Mapping[str, Any],
    request: ExternalDecoderRequest,
    provider: ExternalDecoderProvider,
) -> ExternalDecoderJudgment:
    prepared = provider.prepare(request)
    expected = {
        "provider": provider.config.provider,
        "request_id": request.request_id,
        "request_sha256": request.request_sha256,
        "prompt_sha256": prepared.prompt_sha256,
        "decoder_instance_id": provider.config.decoder_instance_id,
        "decoder_family_id": provider.config.source.decoder_family_id,
        "source_descriptor": provider.config.source_descriptor,
        "request_body_sha256": prepared.body_sha256,
        "model_requested": provider.config.model,
        "client_request_id": prepared.client_request_id,
    }
    mismatches = {
        field: {"retained": audit.get(field), "expected": value}
        for field, value in expected.items()
        if audit.get(field) != value
    }
    returned = audit.get("model_returned")
    if (
        not isinstance(returned, str)
        or not returned_model_is_consistent(
            provider.config.provider,
            str(provider.config.model),
            returned,
        )
    ):
        mismatches["model_returned"] = {
            "retained": returned,
            "expected": provider.config.model,
        }
    if mismatches:
        raise ValueError(
            "resumable external-decoder audit does not match the current "
            "request/provider configuration; mismatched fields: "
            + ", ".join(sorted(mismatches))
        )
    judgment_raw = audit.get("judgment")
    if not isinstance(judgment_raw, Mapping):
        raise ValueError("external-decoder audit lacks a judgment")
    judgment = ExternalDecoderJudgment.parse(judgment_raw)
    if (
        judgment.request_id != request.request_id
        or judgment.request_sha256 != request.request_sha256
        or judgment.decoder_instance_id
        != provider.config.decoder_instance_id
    ):
        raise ValueError(
            "external-decoder audit judgment is not request-bound"
        )
    return judgment


@dataclass(frozen=True, slots=True)
class ExternalDecoderExecutionSummary:
    request_count: int
    source_count: int
    judgment_count: int
    resumed_count: int
    executed_count: int
    transport_attempts_by_provider: Mapping[str, int]
    total_tokens_by_provider: Mapping[str, int]
    judgments_path: str
    audit_path: str
    attempt_path: str
    repaired_trailing_files: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "request_count": self.request_count,
            "source_count": self.source_count,
            "judgment_count": self.judgment_count,
            "resumed_count": self.resumed_count,
            "executed_count": self.executed_count,
            "transport_attempts_by_provider": dict(
                self.transport_attempts_by_provider
            ),
            "total_tokens_by_provider": dict(
                self.total_tokens_by_provider
            ),
            "judgments_path": self.judgments_path,
            "audit_path": self.audit_path,
            "attempt_path": self.attempt_path,
            "repaired_trailing_files": list(
                self.repaired_trailing_files
            ),
        }


def _execute_external_decoder_collection_locked(
    providers: Sequence[ExternalDecoderProvider],
    requests: Iterable[ExternalDecoderRequest],
    *,
    judgment_file: Path,
    audit_file: Path,
    attempt_file: Path,
) -> ExternalDecoderExecutionSummary:
    """Execute or resume a decoder corpus with durable audit-before-judgment.

    Each accepted provider audit embeds the validated judgment and is fsynced
    before the import-compatible judgment JSONL line.  If the second append is
    interrupted, a later call reconstructs it without another provider call.
    """

    configured = tuple(providers)
    _validate_distinct_config_set(
        tuple(provider.config for provider in configured),
        require_pair=False,
    )
    provider_by_name = {
        provider.config.provider: provider for provider in configured
    }
    if len(provider_by_name) != len(configured):
        raise ValueError("external decoder providers must be unique")
    requests_tuple = tuple(sorted(requests, key=lambda row: row.request_id))
    request_by_id = {row.request_id: row for row in requests_tuple}
    if not requests_tuple:
        raise ValueError("at least one external decoder request is required")
    if len(request_by_id) != len(requests_tuple):
        raise ValueError("external decoder requests contain duplicate IDs")
    repaired = tuple(
        str(path)
        for path in (attempt_file, audit_file, judgment_file)
        if _repair_trailing_jsonl(path)
    )
    attempt_ledger = _DurableAttemptLedger(attempt_file)
    attempt_ledger.validate_bindings(provider_by_name, request_by_id)
    if attempt_ledger.unresolved_attempt_ids:
        raise ValueError(
            "transport attempt ledger contains an unresolved durable "
            "started attempt; billing outcome is unknown and manual review "
            "is required before retry"
        )
    existing_audits = _read_audits(audit_file)
    embedded_audits = attempt_ledger.embedded_final_audits()
    for key, embedded in embedded_audits.items():
        retained = existing_audits.get(key)
        if retained is None:
            _append_jsonl(audit_file, embedded)
            existing_audits[key] = embedded
        elif retained != embedded:
            raise ValueError(
                "transport-attempt/final-audit mismatch for "
                f"{key}"
            )
    settled_without_final_audit = sorted(
        {
            (
                attempt_ledger.starts[attempt_id]["provider"],
                attempt_ledger.starts[attempt_id]["request_id"],
            )
            for attempt_id in attempt_ledger.settlements
        }
        - set(embedded_audits)
    )
    if settled_without_final_audit:
        raise ValueError(
            "transport attempt ledger contains a paid or otherwise attempted "
            "request without an accepted/rejected final provider audit; "
            "manual review is required before retry: "
            + repr(settled_without_final_audit)
        )
    audits_without_attempts = sorted(
        set(existing_audits) - set(embedded_audits)
    )
    if audits_without_attempts:
        raise ValueError(
            "external-decoder final audits lack durable transport-attempt "
            "records: " + repr(audits_without_attempts)
        )
    existing_judgments = (
        read_external_decoder_judgments(judgment_file)
        if judgment_file.exists()
        else ()
    )
    judgment_by_key: dict[tuple[str, str], ExternalDecoderJudgment] = {}
    for judgment in existing_judgments:
        key = (judgment.decoder_instance_id, judgment.request_id)
        if key in judgment_by_key:
            raise ValueError(
                "duplicate external decoder judgment identity "
                f"{key}"
            )
        judgment_by_key[key] = judgment
        matching_audit_key = next(
            (
                audit_key
                for audit_key, audit in existing_audits.items()
                if (
                    audit.get("decoder_instance_id")
                    == judgment.decoder_instance_id
                    and audit.get("request_id") == judgment.request_id
                )
            ),
            None,
        )
        if matching_audit_key is None:
            raise ValueError(
                "external decoder judgment lacks an audit-first record: "
                f"{key}"
            )

    expected_audit_keys = {
        (provider.config.provider, request.request_id)
        for provider in configured
        for request in requests_tuple
    }
    unexpected = sorted(set(existing_audits) - expected_audit_keys)
    if unexpected:
        raise ValueError(
            "unexpected external-decoder audits: " + repr(unexpected)
        )

    resumed_count = 0
    for provider_name, provider in provider_by_name.items():
        restored_attempts, restored_tokens = (
            attempt_ledger.accounting_for(provider_name)
        )
        provider.restore_budget(
            request_count=restored_attempts,
            total_tokens=restored_tokens,
        )

    for provider in sorted(
        configured,
        key=lambda item: item.config.provider,
    ):
        for request in requests_tuple:
            audit_key = (provider.config.provider, request.request_id)
            retained = existing_audits.get(audit_key)
            if retained is None:
                continue
            if retained.get("acceptance_status") != "accepted":
                raise ValueError(
                    "external-decoder journal contains a rejected identity "
                    f"for {audit_key}; manual review is required"
                )
            judgment = _validate_resumed_audit(
                retained,
                request,
                provider,
            )
            judgment_key = (
                judgment.decoder_instance_id,
                judgment.request_id,
            )
            existing = judgment_by_key.get(judgment_key)
            if existing is None:
                _append_jsonl(judgment_file, judgment.to_dict())
                judgment_by_key[judgment_key] = judgment
            elif existing.to_dict() != judgment.to_dict():
                raise ValueError(
                    "external-decoder audit/judgment mismatch for "
                    f"{audit_key}"
                )
            resumed_count += 1

    pending = [
        (provider, request)
        for provider in configured
        for request in requests_tuple
        if (provider.config.provider, request.request_id)
        not in existing_audits
    ]
    if pending:
        # Validate both providers and both keys before the first provider is
        # contacted. Keys are discarded immediately and never retained.
        for provider in configured:
            provider.preflight_live_availability()

    executed_count = 0
    for provider in sorted(
        configured,
        key=lambda item: item.config.provider,
    ):
        for request in requests_tuple:
            audit_key = (provider.config.provider, request.request_id)
            if audit_key in existing_audits:
                continue
            try:
                result = provider.complete(
                    request,
                    attempt_ledger=attempt_ledger,
                )
            except ExternalDecoderIdentityMismatch as exc:
                rejected = exc.result.to_audit_record(
                    acceptance_status="rejected_identity_mismatch"
                )
                _append_jsonl(audit_file, rejected)
                existing_audits[audit_key] = rejected
                raise
            audit = result.to_audit_record()
            _append_jsonl(audit_file, audit)
            _append_jsonl(judgment_file, result.judgment.to_dict())
            existing_audits[audit_key] = audit
            judgment_by_key[
                (
                    result.judgment.decoder_instance_id,
                    result.judgment.request_id,
                )
            ] = result.judgment
            executed_count += 1

    return ExternalDecoderExecutionSummary(
        request_count=len(requests_tuple),
        source_count=len(configured),
        judgment_count=len(judgment_by_key),
        resumed_count=resumed_count,
        executed_count=executed_count,
        transport_attempts_by_provider={
            provider.config.provider: provider.budget.request_count
            for provider in configured
        },
        total_tokens_by_provider={
            provider.config.provider: provider.budget.total_tokens
            for provider in configured
        },
        judgments_path=str(judgment_file),
        audit_path=str(audit_file),
        attempt_path=str(attempt_file),
        repaired_trailing_files=repaired,
    )


def execute_external_decoder_collection(
    providers: Sequence[ExternalDecoderProvider],
    requests: Iterable[ExternalDecoderRequest],
    *,
    judgments_path: str | Path,
    audit_path: str | Path,
    attempt_path: str | Path | None = None,
) -> ExternalDecoderExecutionSummary:
    """Execute/resume under an exclusive lock and durable attempt ledger.

    ``transport-attempts.jsonl`` is written and fsynced before every actual
    HTTP attempt. A complete ``started`` record without its ``settled`` pair
    has an unknown billing outcome and blocks automatic resume for manual
    review. Only one truncated final JSONL tail is repaired; corruption in an
    earlier line is never hidden.
    """

    configured = tuple(providers)
    _validate_distinct_config_set(
        tuple(provider.config for provider in configured),
        require_pair=False,
    )
    requests_tuple = tuple(sorted(requests, key=lambda row: row.request_id))
    if not requests_tuple:
        raise ValueError("at least one external decoder request is required")
    if len({row.request_id for row in requests_tuple}) != len(requests_tuple):
        raise ValueError("external decoder requests contain duplicate IDs")
    _require_retry_expanded_collection_budget(
        requests_tuple,
        tuple(provider.config for provider in configured),
    )

    judgment_file = Path(judgments_path)
    audit_file = Path(audit_path)
    attempt_file = (
        Path(attempt_path)
        if attempt_path is not None
        else audit_file.with_name("transport-attempts.jsonl")
    )
    resolved = {
        judgment_file.resolve(),
        audit_file.resolve(),
        attempt_file.resolve(),
    }
    if len(resolved) != 3:
        raise ValueError(
            "judgments_path, audit_path, and attempt_path must differ"
        )
    lock_file = audit_file.with_name(
        ".external-decoder-collection.lock"
    )
    with _ExclusiveCollectionLock(lock_file):
        return _execute_external_decoder_collection_locked(
            configured,
            requests_tuple,
            judgment_file=judgment_file,
            audit_file=audit_file,
            attempt_file=attempt_file,
        )
