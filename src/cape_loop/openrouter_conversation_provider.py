"""OpenRouter authoring client for frozen, human-readable conversations.

This module has one deliberately narrow responsibility: ask one pinned
OpenRouter model to author reusable language templates for one
``ScenarioSpec``.  It does not choose an option and it never receives latent
user state, numeric feature vectors, target indices, profile state, or split
metadata.

The generated record is intentionally a small, readable intermediate format.
The experiment runner can validate and render it later after the mathematical
simulator has selected an option.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from string import Formatter
from types import MappingProxyType
from typing import Any, Mapping, Sequence
import json
import math
import os
import re

from .openrouter_provider import (
    HTTPResult,
    HTTPTransport,
    OPENROUTER_OFFICIAL_BASE_URL,
    urllib_transport,
)
from .conversation_surfaces import (
    ConversationTemplateBank,
    ScenarioConversationTemplate,
)
from .scenarios import ScenarioCatalog, ScenarioSpec


OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
OPENROUTER_CONVERSATION_ENDPOINT = (
    OPENROUTER_OFFICIAL_BASE_URL + "/v1/chat/completions"
)
TEMPLATE_SCHEMA_VERSION = 1

PRESENTATION_KINDS = (
    "balanced",
    "restricted",
    "default",
    "suggested",
    "ranking",
)

PAIR_PLACEHOLDERS = (
    "prompt",
    "option_1_name",
    "option_1_description",
    "option_2_name",
    "option_2_description",
)
BASE_TEMPLATE_PLACEHOLDERS = PAIR_PLACEHOLDERS
PRESENTATION_PLACEHOLDERS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "balanced": PAIR_PLACEHOLDERS,
        "restricted": PAIR_PLACEHOLDERS,
        "default": (*PAIR_PLACEHOLDERS, "default_name"),
        "suggested": (*PAIR_PLACEHOLDERS, "suggested_name"),
        "ranking": PAIR_PLACEHOLDERS,
    }
)
CHOICE_PLACEHOLDERS = ("selected_name",)
FIXED_CHOICE_TEMPLATE = "I choose {selected_name}."
DEFAULT_TREATMENT_SENTENCE = (
    "{default_name} is preselected as the default."
)
SUGGESTED_TREATMENT_SENTENCE = "I suggest {suggested_name}."

_MODEL_SLUG = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")
_OPTION_NAME = re.compile(
    r"^(?P<stem>[A-Z][A-Za-z]*(?: [A-Za-z]+){0,3}) (?P<letter>[A-D])$"
)
_NEUTRAL_TREATMENT_WORDS = re.compile(
    r"\b(default|preselect(?:ed)?|recommend(?:ed|ation)?|suggest(?:ed|ion)?)\b",
    re.IGNORECASE,
)
_RESTRICTED_CUES = re.compile(
    r"\b(restrict(?:ed|ion)?|only available|limited choice)\b",
    re.IGNORECASE,
)
_RANKING_CUES = re.compile(
    r"\b(best|better|top[- ]?ranked|ranked first|preferred)\b",
    re.IGNORECASE,
)
_GENERIC_NAME_STEMS = frozenset(
    {"option", "choice", "alternative", "selection", "item"}
)
_MATHEMATICAL_SURFACE = re.compile(
    r"\b(attribute|coefficient|feature vector|latent|numeric|probability|"
    r"statistic(?:al)?|target index|utility score)\b",
    re.IGNORECASE,
)
class ConversationProviderError(RuntimeError):
    """Base class for conversation-template authoring failures."""


class ConversationLiveExecutionRequired(ConversationProviderError):
    """A paid request was attempted without explicit authorization."""


class ConversationMissingAPIKey(ConversationProviderError):
    """The OpenRouter credential was absent at the live-call boundary."""


class ConversationBudgetExceeded(ConversationProviderError):
    """A request would exceed a configured local budget."""


class ConversationHTTPError(ConversationProviderError):
    """OpenRouter returned a non-success HTTP response."""


class ConversationResponseError(ConversationProviderError):
    """OpenRouter returned an incomplete or invalid template response."""


class ConversationModelMismatch(ConversationResponseError):
    """The returned model did not exactly match the pinned requested model."""


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _positive_integer(
    value: Any,
    name: str,
    *,
    maximum: int,
) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 1
        or value > maximum
    ):
        raise ValueError(f"{name} must be an integer in [1, {maximum}]")
    return value


@dataclass(frozen=True, slots=True)
class OpenRouterConversationConfig:
    """Small, pinned budget for authoring a finite scenario-template bank."""

    model: str
    timeout_seconds: float = 60.0
    max_output_tokens: int = 1200
    max_requests: int = 32
    max_total_tokens: int = 500_000
    upstream_provider: str = ""
    live_execution: bool = False

    def __post_init__(self) -> None:
        if (
            not isinstance(self.model, str)
            or self.model != self.model.strip()
            or not _MODEL_SLUG.fullmatch(self.model)
            or self.model == "openrouter/auto"
            or self.model.lower().endswith("-latest")
            or ":" in self.model
        ):
            raise ValueError(
                "model must be one pinned author/model slug; aliases, "
                "route variants, and openrouter/auto are not accepted"
            )
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(float(self.timeout_seconds))
            or float(self.timeout_seconds) <= 0.0
            or float(self.timeout_seconds) > 300.0
        ):
            raise ValueError(
                "timeout_seconds must be finite and lie in (0, 300]"
            )
        _positive_integer(
            self.max_output_tokens,
            "max_output_tokens",
            maximum=2048,
        )
        _positive_integer(
            self.max_requests,
            "max_requests",
            maximum=128,
        )
        _positive_integer(
            self.max_total_tokens,
            "max_total_tokens",
            maximum=2_000_000,
        )
        if self.max_total_tokens < self.max_output_tokens:
            raise ValueError(
                "max_total_tokens cannot be smaller than max_output_tokens"
            )
        if not isinstance(self.upstream_provider, str):
            raise TypeError("upstream_provider must be a string")
        if self.upstream_provider and (
            self.upstream_provider != self.upstream_provider.strip()
            or not re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9._/-]*",
                self.upstream_provider,
            )
            or ".." in self.upstream_provider
            or "//" in self.upstream_provider
        ):
            raise ValueError(
                "upstream_provider must be one exact OpenRouter provider slug"
            )
        if not isinstance(self.live_execution, bool):
            raise TypeError("live_execution must be Boolean")


def _scenario_input(scenario: ScenarioSpec) -> dict[str, Any]:
    """Project a scenario onto the complete and only model-visible fields."""

    if not isinstance(scenario, ScenarioSpec):
        raise TypeError("scenario must be a ScenarioSpec")
    return {
        "prompt": scenario.prompt,
        "domain": scenario.domain,
        "task_family": scenario.task_family,
        "options": [
            {
                "option_id": option.option_id,
                "label": option.label,
            }
            for option in scenario.options
        ],
    }


def _option_name_schema(
    scenario: ScenarioSpec,
) -> dict[str, Any]:
    letters = "ABCD"
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            option.option_id: {
                "type": "string",
                "minLength": 3,
                "maxLength": 40,
                "pattern": (
                    r"^[A-Z][A-Za-z]*(?: [A-Za-z]+){0,3} "
                    + letters[index]
                    + "$"
                ),
                "description": (
                    "Neutral display name ending in "
                    f"{letters[index]}, with the same noun stem as the "
                    "other three names."
                ),
            }
            for index, option in enumerate(scenario.options)
        },
        "required": [option.option_id for option in scenario.options],
    }


def conversation_template_json_schema(
    scenario: ScenarioSpec,
) -> dict[str, Any]:
    """Return the strict, scenario-bound schema sent to OpenRouter."""

    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "display_names": _option_name_schema(scenario),
            "base_template": {
                "type": "string",
                "minLength": 70,
                "maxLength": 800,
                "description": (
                    "One neutral assistant utterance using exactly these "
                    "placeholders once each: "
                    + ", ".join(
                        "{" + name + "}"
                        for name in BASE_TEMPLATE_PLACEHOLDERS
                    )
                ),
            },
        },
        "required": [
            "display_names",
            "base_template",
        ],
    }


_SYSTEM_INSTRUCTION = """\
Return JSON only and follow the supplied strict schema.

You author reusable, natural-language conversation templates. You never choose
an option. Assign the four options neutral display names that share one short,
scenario-appropriate concrete noun stem and end in A, B, C, and D in the
supplied option order, for example "Hotel A" through "Hotel D", "Draft A"
through "Draft D", or "Route A" through "Route D". Never use generic names
such as "Option A", "Choice A", or "Alternative A". Names must not encode
price, quality, preference direction, or any other option property.

Author exactly one neutral base_template. It must be a single natural
assistant utterance ending in a question and must contain {prompt},
{option_1_name}, {option_1_description}, {option_2_name},
and {option_2_description} exactly once each. Descriptions are inserted later
from the source labels, so do not repeat, paraphrase, embellish, or invent
option facts.

The base_template must be a meaningful, natural assistant utterance that
presents the pair in the scenario's ordinary language. It must work unchanged
for balanced, restricted, default, suggested, and ranking conditions. It must
not mention restrictions, ranking, defaults, preselection, suggestions,
recommendations, preference, mathematical features, probabilities,
statistics, or which option will be chosen. Do not author treatment wording
or a user reply.
"""


@dataclass(frozen=True, slots=True)
class PreparedConversationRequest:
    """Credential-free request material suitable for inspection and testing."""

    scenario_id: str
    endpoint: str
    model_input: Mapping[str, Any]
    body: Mapping[str, Any]
    body_bytes: bytes
    headers: Mapping[str, str]
    token_ceiling: int

    def to_log(self) -> dict[str, Any]:
        return {
            "event": "conversation_template_request_prepared",
            "scenario_id": self.scenario_id,
            "endpoint": self.endpoint,
            "model": self.body["model"],
            "model_input": deepcopy(dict(self.model_input)),
            "authored_fields": ["display_names", "base_template"],
            "base_template_placeholders": list(
                BASE_TEMPLATE_PLACEHOLDERS
            ),
            "local_treatment_expansion": {
                "neutral_presentations": [
                    "balanced",
                    "restricted",
                    "ranking",
                ],
                "insertion_point": "immediately after {prompt}",
                "default_sentence": DEFAULT_TREATMENT_SENTENCE,
                "suggested_sentence": SUGGESTED_TREATMENT_SENTENCE,
                "choice_template": FIXED_CHOICE_TEMPLATE,
            },
            "max_output_tokens": self.body["max_tokens"],
            "token_ceiling": self.token_ceiling,
            "allow_fallbacks": False,
        }


def prepare_conversation_request(
    scenario: ScenarioSpec,
    config: OpenRouterConversationConfig,
) -> PreparedConversationRequest:
    """Build a strict OpenRouter request without reading any credential."""

    model_input = _scenario_input(scenario)
    provider_preferences: dict[str, Any] = {
        "allow_fallbacks": False,
        "require_parameters": True,
        "data_collection": "deny",
    }
    if config.upstream_provider:
        provider_preferences["order"] = [config.upstream_provider]
        provider_preferences["only"] = [config.upstream_provider]
    body: dict[str, Any] = {
        "model": config.model,
        "messages": [
            {
                "role": "system",
                "content": _SYSTEM_INSTRUCTION,
            },
            {
                "role": "user",
                "content": _canonical(model_input),
            },
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "cape_loop_conversation_templates",
                "strict": True,
                "schema": conversation_template_json_schema(scenario),
            },
        },
        "max_tokens": config.max_output_tokens,
        "stream": False,
        "provider": provider_preferences,
    }
    body_bytes = _canonical(body).encode("utf-8")
    # UTF-8 byte count is a deliberately conservative input-token ceiling for
    # the small text-only payload; output uses the explicit provider ceiling.
    token_ceiling = len(body_bytes) + config.max_output_tokens
    return PreparedConversationRequest(
        scenario_id=scenario.scenario_id,
        endpoint=OPENROUTER_CONVERSATION_ENDPOINT,
        model_input=MappingProxyType(model_input),
        body=MappingProxyType(body),
        body_bytes=body_bytes,
        headers=MappingProxyType(
            {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "X-OpenRouter-Metadata": "enabled",
                "X-OpenRouter-Cache": "false",
                "User-Agent": "cape-loop/0.1 conversation-author",
            }
        ),
        token_ceiling=token_ceiling,
    )


def _template_fields(text: str, *, label: str) -> tuple[str, ...]:
    if (
        not isinstance(text, str)
        or text != text.strip()
        or "\x00" in text
        or "\r" in text
        or "\n" in text
    ):
        raise ConversationResponseError(
            f"{label} must be a non-empty, single-line trimmed string"
        )
    fields: list[str] = []
    try:
        parsed = Formatter().parse(text)
        for _, field_name, format_spec, conversion in parsed:
            if field_name is None:
                continue
            if not field_name or format_spec or conversion:
                raise ConversationResponseError(
                    f"{label} uses an unsupported formatted placeholder"
                )
            fields.append(field_name)
    except ValueError as exc:
        raise ConversationResponseError(
            f"{label} contains malformed braces"
        ) from exc
    return tuple(fields)


def _validate_option_names(
    raw: Any,
    scenario: ScenarioSpec,
) -> dict[str, str]:
    if not isinstance(raw, Mapping):
        raise ConversationResponseError("display_names must be an object")
    expected_ids = tuple(option.option_id for option in scenario.options)
    if set(raw) != set(expected_ids):
        raise ConversationResponseError(
            "display_names must contain exactly all four scenario option IDs"
        )
    result: dict[str, str] = {}
    stems: set[str] = set()
    for index, option_id in enumerate(expected_ids):
        value = raw[option_id]
        if (
            not isinstance(value, str)
            or value != value.strip()
            or "\r" in value
            or "\n" in value
            or len(value) > 40
        ):
            raise ConversationResponseError(
                f"display_names[{option_id!r}] is not a valid display name"
            )
        match = _OPTION_NAME.fullmatch(value)
        if match is None or match.group("letter") != "ABCD"[index]:
            raise ConversationResponseError(
                "option names must share a neutral stem and end in A-D in "
                "scenario option order"
            )
        stems.add(match.group("stem"))
        result[option_id] = value
    if len(stems) != 1:
        raise ConversationResponseError(
            "all four option names must share the same neutral noun stem"
        )
    stem = next(iter(stems))
    if stem.casefold() in _GENERIC_NAME_STEMS:
        raise ConversationResponseError(
            "display names must use a scenario-appropriate concrete noun "
            "stem, not a generic option label"
        )
    return result


def _validate_base_template(
    raw: Any,
    scenario: ScenarioSpec,
) -> str:
    if not isinstance(raw, str) or not 70 <= len(raw) <= 800:
        raise ConversationResponseError(
            "base_template must contain 70-800 characters"
        )
    fields = _template_fields(raw, label="base_template")
    if Counter(fields) != Counter(BASE_TEMPLATE_PLACEHOLDERS):
        expected = ", ".join(
            "{" + field + "}" for field in BASE_TEMPLATE_PLACEHOLDERS
        )
        raise ConversationResponseError(
            f"base_template must use exactly {expected}"
        )
    if "?" not in raw:
        raise ConversationResponseError(
            "base_template must ask a question"
        )
    forbidden_surfaces = tuple(
        text.casefold()
        for option in scenario.options
        for text in (option.option_id, option.label)
    )
    folded = raw.casefold()
    if any(surface in folded for surface in forbidden_surfaces):
        raise ConversationResponseError(
            "base_template repeats source option text instead of using "
            "placeholders"
        )
    if _NEUTRAL_TREATMENT_WORDS.search(raw):
        raise ConversationResponseError(
            "base_template introduces a default or suggestion treatment cue"
        )
    if _RESTRICTED_CUES.search(raw):
        raise ConversationResponseError(
            "base_template must not announce a restriction"
        )
    if _RANKING_CUES.search(raw):
        raise ConversationResponseError(
            "base_template must not characterize rank as quality"
        )
    if _MATHEMATICAL_SURFACE.search(raw):
        raise ConversationResponseError(
            "base_template must be natural scenario language, not a "
            "mathematical or statistical surface"
        )
    return raw


def _expand_base_template(base_template: str) -> dict[str, str]:
    def with_treatment(treatment_sentence: str) -> str:
        return base_template.replace(
            "{prompt}",
            "{prompt} " + treatment_sentence,
            1,
        )

    neutral = base_template
    result = {
        "balanced": neutral,
        "restricted": neutral,
        "default": with_treatment(DEFAULT_TREATMENT_SENTENCE),
        "suggested": with_treatment(SUGGESTED_TREATMENT_SENTENCE),
        "ranking": neutral,
    }
    for kind in PRESENTATION_KINDS:
        fields = _template_fields(
            result[kind],
            label=f"expanded presentation_templates.{kind}",
        )
        if Counter(fields) != Counter(PRESENTATION_PLACEHOLDERS[kind]):
            raise AssertionError(
                "local treatment expansion produced an invalid placeholder "
                f"contract for {kind}"
            )
    if not (
        result["balanced"]
        == result["restricted"]
        == result["ranking"]
    ):
        raise AssertionError(
            "neutral local presentation expansion must be identical"
        )
    return result


def _validated_content(
    raw: Any,
    scenario: ScenarioSpec,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ConversationResponseError(
            "OpenRouter template content must be a JSON object"
        )
    expected = {
        "display_names",
        "base_template",
    }
    if set(raw) != expected:
        raise ConversationResponseError(
            "OpenRouter template content contains missing or unknown fields"
        )
    base_template = _validate_base_template(
        raw["base_template"],
        scenario,
    )
    return {
        "display_names": _validate_option_names(
            raw["display_names"],
            scenario,
        ),
        "base_template": base_template,
        "presentation_templates": _expand_base_template(base_template),
        "choice_template": FIXED_CHOICE_TEMPLATE,
    }


def _usage(raw: Any) -> dict[str, int]:
    if not isinstance(raw, Mapping):
        raise ConversationResponseError("OpenRouter response lacks usage")
    result: dict[str, int] = {}
    for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = raw.get(name)
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
        ):
            raise ConversationResponseError(
                f"OpenRouter usage.{name} must be a non-negative integer"
            )
        result[name] = value
    if result["total_tokens"] < (
        result["prompt_tokens"] + result["completion_tokens"]
    ):
        raise ConversationResponseError(
            "OpenRouter total_tokens is smaller than its components"
        )
    return result


def _selected_upstream_provider(raw: Mapping[str, Any]) -> str | None:
    metadata = raw.get("openrouter_metadata")
    if not isinstance(metadata, Mapping):
        return None
    endpoints = metadata.get("endpoints")
    if not isinstance(endpoints, Mapping):
        return None
    available = endpoints.get("available")
    if not isinstance(available, Sequence) or isinstance(
        available,
        (str, bytes),
    ):
        return None
    selected = [
        item
        for item in available
        if isinstance(item, Mapping) and item.get("selected") is True
    ]
    if len(selected) != 1:
        return None
    provider = selected[0].get("provider")
    return provider if isinstance(provider, str) and provider else None


def _safe_http_message(body: bytes, *, secret: str) -> str:
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


def _parse_result(
    *,
    scenario: ScenarioSpec,
    config: OpenRouterConversationConfig,
    http_result: HTTPResult,
    secret: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if http_result.status != 200:
        raise ConversationHTTPError(
            "OpenRouter conversation authoring failed with HTTP "
            f"{http_result.status}: "
            + _safe_http_message(http_result.body, secret=secret)
        )
    try:
        raw = json.loads(http_result.body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConversationResponseError(
            "OpenRouter response body is not valid UTF-8 JSON"
        ) from exc
    if not isinstance(raw, Mapping):
        raise ConversationResponseError(
            "OpenRouter response body must be an object"
        )
    response_id = raw.get("id")
    returned_model = raw.get("model")
    if not isinstance(response_id, str) or not response_id.strip():
        raise ConversationResponseError(
            "OpenRouter response lacks a provider response ID"
        )
    if not isinstance(returned_model, str) or not returned_model:
        raise ConversationResponseError(
            "OpenRouter response lacks a returned model"
        )
    if returned_model != config.model:
        raise ConversationModelMismatch(
            "OpenRouter returned a model different from the pinned model: "
            f"requested={config.model!r}, returned={returned_model!r}"
        )
    choices = raw.get("choices")
    if (
        not isinstance(choices, list)
        or len(choices) != 1
        or not isinstance(choices[0], Mapping)
    ):
        raise ConversationResponseError(
            "OpenRouter response must contain exactly one choice"
        )
    choice = choices[0]
    if choice.get("finish_reason") != "stop":
        raise ConversationResponseError(
            "OpenRouter conversation response did not finish with stop"
        )
    message = choice.get("message")
    if not isinstance(message, Mapping):
        raise ConversationResponseError(
            "OpenRouter response choice lacks a message"
        )
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ConversationResponseError(
            "OpenRouter response message lacks JSON content"
        )
    try:
        generated = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ConversationResponseError(
            "OpenRouter message content is not valid JSON"
        ) from exc
    validated = _validated_content(generated, scenario)
    usage = _usage(raw.get("usage"))
    upstream = _selected_upstream_provider(raw)
    generator = {
        "mode": "llm",
        "provider": "openrouter",
        "model_requested": config.model,
        "model_returned": returned_model,
        "provider_response_id": response_id,
        "upstream_provider": upstream,
        "usage": usage,
        "allow_fallbacks": False,
        "validation_status": "passed",
        "authored_fields": ["display_names", "base_template"],
        "treatment_expansion": "local-fixed-v1",
        "choice_source": "local-fixed-v1",
    }
    record = {
        "schema_version": TEMPLATE_SCHEMA_VERSION,
        "scenario_id": scenario.scenario_id,
        "display_names": validated["display_names"],
        "presentation_templates": validated["presentation_templates"],
        "choice_template": validated["choice_template"],
        "generator": generator,
    }
    result_log = {
        "event": "conversation_template_request_completed",
        "scenario_id": scenario.scenario_id,
        "provider": "openrouter",
        "model_requested": config.model,
        "model_returned": returned_model,
        "provider_response_id": response_id,
        "upstream_provider": upstream,
        "usage": usage,
        "validation_status": "passed",
        "allow_fallbacks": False,
        "authored_display_names": deepcopy(
            validated["display_names"]
        ),
        "authored_base_template": validated["base_template"],
        "local_treatment_expansion": {
            "balanced": "",
            "restricted": "",
            "default": DEFAULT_TREATMENT_SENTENCE,
            "suggested": SUGGESTED_TREATMENT_SENTENCE,
            "ranking": "",
            "choice_template": FIXED_CHOICE_TEMPLATE,
        },
    }
    return record, result_log


class OpenRouterConversationProvider:
    """Synchronous one-call-per-scenario OpenRouter authoring client."""

    def __init__(
        self,
        config: OpenRouterConversationConfig,
        *,
        transport: HTTPTransport = urllib_transport,
    ) -> None:
        self.config = config
        self._transport = transport
        self._request_count = 0
        self._reserved_tokens = 0
        self._prepared_inputs: dict[str, dict[str, Any]] = {}
        self._records: dict[str, dict[str, Any]] = {}
        self._request_logs: list[dict[str, Any]] = []
        self._result_logs: list[dict[str, Any]] = []

    @property
    def request_count(self) -> int:
        return self._request_count

    @property
    def reserved_tokens(self) -> int:
        return self._reserved_tokens

    @property
    def request_logs(self) -> tuple[dict[str, Any], ...]:
        return tuple(deepcopy(self._request_logs))

    @property
    def result_logs(self) -> tuple[dict[str, Any], ...]:
        return tuple(deepcopy(self._result_logs))

    def prepare(
        self,
        scenario: ScenarioSpec,
    ) -> PreparedConversationRequest:
        prepared = prepare_conversation_request(scenario, self.config)
        visible = deepcopy(dict(prepared.model_input))
        prior = self._prepared_inputs.get(scenario.scenario_id)
        if prior is not None and prior != visible:
            raise ValueError(
                "one scenario_id was prepared with different visible content"
            )
        self._prepared_inputs[scenario.scenario_id] = visible
        self._request_logs.append(prepared.to_log())
        return prepared

    def _reserve(self, prepared: PreparedConversationRequest) -> None:
        if self._request_count + 1 > self.config.max_requests:
            raise ConversationBudgetExceeded(
                f"conversation request would exceed max_requests="
                f"{self.config.max_requests}"
            )
        if (
            self._reserved_tokens + prepared.token_ceiling
            > self.config.max_total_tokens
        ):
            raise ConversationBudgetExceeded(
                "conversation request would exceed max_total_tokens="
                f"{self.config.max_total_tokens}"
            )
        self._request_count += 1
        self._reserved_tokens += prepared.token_ceiling

    def complete(self, scenario: ScenarioSpec) -> dict[str, Any]:
        """Author one scenario record after explicit live authorization.

        Repeated calls for an already completed scenario return the validated
        in-memory record and never issue a second provider request.
        """

        existing = self._records.get(scenario.scenario_id)
        if existing is not None:
            if self._prepared_inputs.get(scenario.scenario_id) != _scenario_input(
                scenario
            ):
                raise ValueError(
                    "one scenario_id was completed with different visible "
                    "content"
                )
            return deepcopy(existing)

        prepared = self.prepare(scenario)
        if not self.config.live_execution:
            raise ConversationLiveExecutionRequired(
                "conversation authoring requires live_execution=True"
            )

        # This is the first and only credential read in the complete path.
        secret = os.environ.get(OPENROUTER_API_KEY_ENV, "")
        if not secret:
            raise ConversationMissingAPIKey(
                f"{OPENROUTER_API_KEY_ENV} is not set"
            )
        self._reserve(prepared)
        headers = dict(prepared.headers)
        headers["Authorization"] = f"Bearer {secret}"
        try:
            http_result = self._transport(
                url=prepared.endpoint,
                body=prepared.body_bytes,
                headers=headers,
                timeout=float(self.config.timeout_seconds),
            )
        except Exception as exc:
            message = str(exc)
            if secret:
                message = message.replace(secret, "[redacted]")
            raise ConversationProviderError(
                "OpenRouter conversation transport failed: "
                + " ".join(message.split())[:500]
            ) from exc
        record, result_log = _parse_result(
            scenario=scenario,
            config=self.config,
            http_result=http_result,
            secret=secret,
        )
        self._records[scenario.scenario_id] = deepcopy(record)
        self._result_logs.append(result_log)
        return deepcopy(record)


def generate_conversation_bank(
    catalog: ScenarioCatalog,
    provider: OpenRouterConversationProvider,
    *,
    bank_id: str = "cape-loop-conversation-templates-v1",
) -> ConversationTemplateBank:
    """Author one template per catalog scenario and return a validated bank."""

    if not isinstance(catalog, ScenarioCatalog):
        raise TypeError("catalog must be a ScenarioCatalog")
    if not isinstance(provider, OpenRouterConversationProvider):
        raise TypeError(
            "provider must be an OpenRouterConversationProvider"
        )
    source = (
        "openrouter:"
        f"{provider.config.model}:llm-authored-unreviewed"
    )
    templates = []
    for scenario in sorted(
        catalog.scenarios,
        key=lambda item: item.scenario_id,
    ):
        record = provider.complete(scenario)
        templates.append(
            ScenarioConversationTemplate(
                scenario_id=record["scenario_id"],
                display_names=record["display_names"],
                presentation_templates=record[
                    "presentation_templates"
                ],
                # The mathematical simulator already fixed the action. Keep
                # its language identical across mechanisms and model runs;
                # the LLM authors the meaningful assistant presentation.
                choice_template=FIXED_CHOICE_TEMPLATE,
                source=source,
            )
        )
    bank = ConversationTemplateBank(
        bank_id=bank_id,
        templates=tuple(templates),
        source=source,
    )
    bank.validate_catalog(catalog)
    return bank


__all__ = [
    "BASE_TEMPLATE_PLACEHOLDERS",
    "CHOICE_PLACEHOLDERS",
    "DEFAULT_TREATMENT_SENTENCE",
    "FIXED_CHOICE_TEMPLATE",
    "OPENROUTER_API_KEY_ENV",
    "PRESENTATION_KINDS",
    "PRESENTATION_PLACEHOLDERS",
    "SUGGESTED_TREATMENT_SENTENCE",
    "ConversationBudgetExceeded",
    "ConversationHTTPError",
    "ConversationLiveExecutionRequired",
    "ConversationMissingAPIKey",
    "ConversationModelMismatch",
    "ConversationProviderError",
    "ConversationResponseError",
    "OpenRouterConversationConfig",
    "OpenRouterConversationProvider",
    "PreparedConversationRequest",
    "conversation_template_json_schema",
    "generate_conversation_bank",
    "prepare_conversation_request",
]
