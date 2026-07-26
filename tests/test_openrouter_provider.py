from __future__ import annotations

from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Mapping
from unittest.mock import patch
from urllib.error import HTTPError
import json
import unittest

import cape_loop.openrouter_provider as openrouter_provider
from cape_loop.config import ConfigError, LLMSection
from cape_loop.llm_exchange import (
    LLMRequest,
    ReplayProvider,
    read_responses,
)
from cape_loop.openrouter_provider import (
    HTTPResult,
    OpenRouterBudgetExceeded,
    OpenRouterChatProvider,
    OpenRouterHTTPError,
    OpenRouterLiveExecutionRequired,
    OpenRouterMissingAPIKey,
    OpenRouterProviderConfig,
    OpenRouterProviderError,
    OpenRouterResponseError,
    OpenRouterResultRejected,
    ResumableOpenRouterCompletionProvider,
    execute_openrouter_requests,
    prepare_openrouter_request,
    urllib_transport,
)


MODEL = "google/gemini-3.6-flash"
UPSTREAM_SLUG = "google-ai-studio"
UPSTREAM_NAME = "Google AI Studio"
TEST_KEY_ENV = "CAPE_LOOP_TEST_OPENROUTER_KEY"

BELIEFS = {
    f"attribute_{attribute}": {
        "-2": 0.1,
        "-1": 0.2,
        "+1": 0.3,
        "+2": 0.4,
    }
    for attribute in range(1, 4)
}


def build_request(label: str = "one") -> LLMRequest:
    return LLMRequest.build(
        request_id=f"llm_full_context:{label}",
        updater_id="llm_full_context",
        view="full_context",
        prior={
            f"attribute_{attribute}": {
                "-2": 0.25,
                "-1": 0.25,
                "+1": 0.25,
                "+2": 0.25,
            }
            for attribute in range(1, 4)
        },
        observation={"selected_option_id": label},
        context={
            "domain": "travel",
            "displayed_options": [label, "alternative"],
            "ranking": [label, "alternative"],
            "target_attribute": 0,
        },
    )


def response_body(
    response_id: str = "gen_test",
    *,
    model: str = MODEL,
    requested_model: str | None = None,
    routing_strategy: str = "direct",
    routing_attempt: int = 1,
    upstream_provider: str = UPSTREAM_NAME,
    upstream_model: str | None = None,
    pipeline: list[dict[str, Any]] | None = None,
    extra_raw: Mapping[str, Any] | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> bytes:
    metadata: dict[str, Any] = {
        "requested": requested_model or model,
        "strategy": routing_strategy,
        "region": "fra",
        "summary": "available=1, selected=Google AI Studio",
        "attempt": routing_attempt,
        "is_byok": False,
        "endpoints": {
            "total": 1,
            "available": [
                {
                    "provider": upstream_provider,
                    "model": upstream_model or model,
                    "selected": True,
                }
            ],
        },
        "attempts": [
            {
                "provider": upstream_provider,
                "model": upstream_model or model,
                "status": 200,
            }
        ],
        "pipeline": [] if pipeline is None else pipeline,
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    raw: dict[str, Any] = {
        "id": response_id,
        "object": "chat.completion",
        "created": 1_700_000_001,
        "model": model,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": json.dumps(
                        {"beliefs": BELIEFS},
                        separators=(",", ":"),
                    ),
                },
            }
        ],
        "usage": {
            "prompt_tokens": 40,
            "completion_tokens": 35,
            "total_tokens": 75,
        },
        "openrouter_metadata": metadata,
    }
    if extra_raw:
        raw.update(extra_raw)
    return json.dumps(raw).encode("utf-8")


def live_config(**overrides: Any) -> OpenRouterProviderConfig:
    values: dict[str, Any] = {
        "model": MODEL,
        "api_key_env": TEST_KEY_ENV,
        "upstream_provider": UPSTREAM_SLUG,
        "live_execution": True,
        "max_total_tokens": 100_000,
    }
    values.update(overrides)
    return OpenRouterProviderConfig(**values)


class _TrackingBody(BytesIO):
    def __init__(self, payload: bytes, *, status: int = 200) -> None:
        super().__init__(payload)
        self.read_sizes: list[int] = []
        self.status = status
        self.headers: dict[str, str] = {}

    def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        return super().read(size)


class _NoCredentialReads(dict[str, str]):
    def get(self, key: str, default: Any = None) -> Any:
        raise AssertionError(f"credential environment was read: {key}")


class OpenRouterConfigAndRequestTests(unittest.TestCase):
    def test_config_locks_origin_credentials_and_canonical_models(self) -> None:
        with self.assertRaisesRegex(ValueError, "official"):
            OpenRouterProviderConfig(
                model=MODEL,
                base_url="https://router-proxy.example.test/api",
            )
        with self.assertRaisesRegex(ValueError, "dedicated credential"):
            OpenRouterProviderConfig(
                model=MODEL,
                base_url="https://router-proxy.example.test/api",
                allow_custom_base_url=True,
            )
        custom = OpenRouterProviderConfig(
            model=MODEL,
            base_url="https://router-proxy.example.test/api",
            allow_custom_base_url=True,
            api_key_env="CAPE_LOOP_OPENROUTER_PROXY_KEY",
        )
        self.assertEqual(
            custom.endpoint,
            "https://router-proxy.example.test/api/v1/chat/completions",
        )
        for first_party_key in (
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GEMINI_API_KEY",
        ):
            with self.subTest(first_party_key=first_party_key):
                with self.assertRaisesRegex(
                    ValueError,
                    "first-party provider key",
                ):
                    OpenRouterProviderConfig(
                        model=MODEL,
                        api_key_env=first_party_key,
                    )
                with self.assertRaisesRegex(
                    ConfigError,
                    "first-party provider key",
                ):
                    LLMSection.parse(
                        {
                            "mode": "openrouter",
                            "model": MODEL,
                            "api_key_env": first_party_key,
                        }
                    )
        for noncanonical in (
            "openrouter/auto",
            "~google/gemini-flash-latest",
            "google/gemini-flash-latest",
            f"{MODEL}:free",
            f"{MODEL}:online",
        ):
            with self.subTest(noncanonical=noncanonical):
                with self.assertRaisesRegex(
                    ValueError,
                    "reproducible|canonical",
                ):
                    OpenRouterProviderConfig(model=noncanonical)

    def test_llm_section_supplies_mode_sensitive_safe_defaults(self) -> None:
        section = LLMSection.parse(
            {
                "mode": "openrouter",
                "model": MODEL,
                "openrouter_upstream_provider": UPSTREAM_SLUG,
            }
        )
        self.assertEqual(section.api_key_env, "OPENROUTER_API_KEY")
        self.assertEqual(section.base_url, "https://openrouter.ai/api")
        self.assertEqual(section.max_retries, 2)
        self.assertFalse(section.openrouter_allow_fallbacks)
        self.assertTrue(section.openrouter_require_parameters)
        self.assertEqual(section.openrouter_data_collection, "deny")

    def test_dry_run_is_keyless_deterministic_and_schema_bound(self) -> None:
        request = build_request()
        config = OpenRouterProviderConfig(
            model=MODEL,
            reasoning_effort="low",
            upstream_provider=UPSTREAM_SLUG,
            http_referer="https://example.test/cape-loop",
            app_title="CAPE-Loop tests",
        )
        with patch.object(
            openrouter_provider.os,
            "environ",
            _NoCredentialReads(),
        ):
            first = prepare_openrouter_request(request, config)
            second = OpenRouterChatProvider(config).prepare(request)

        self.assertEqual(
            first.endpoint,
            "https://openrouter.ai/api/v1/chat/completions",
        )
        self.assertEqual(first.body_bytes, second.body_bytes)
        self.assertEqual(first.body_sha256, second.body_sha256)
        self.assertEqual(first.client_request_id, second.client_request_id)
        self.assertEqual(first.body["model"], MODEL)
        self.assertFalse(first.body["stream"])
        self.assertNotIn("models", first.body)
        self.assertEqual(
            first.body["provider"],
            {
                "allow_fallbacks": False,
                "require_parameters": True,
                "data_collection": "deny",
                "order": [UPSTREAM_SLUG],
                "only": [UPSTREAM_SLUG],
            },
        )
        self.assertEqual(first.body["reasoning"], {"effort": "low"})
        response_format = first.body["response_format"]
        self.assertEqual(response_format["type"], "json_schema")
        json_schema = response_format["json_schema"]
        self.assertTrue(json_schema["strict"])
        self.assertEqual(json_schema["schema"]["required"], ["beliefs"])
        self.assertFalse(json_schema["schema"]["additionalProperties"])
        beliefs = json_schema["schema"]["properties"]["beliefs"]
        self.assertEqual(beliefs["required"], list(BELIEFS))
        for attribute in BELIEFS:
            vector = beliefs["properties"][attribute]
            self.assertFalse(vector["additionalProperties"])
            self.assertEqual(vector["required"], list(BELIEFS[attribute]))
        self.assertNotIn("Authorization", first.headers)
        self.assertEqual(first.headers["X-OpenRouter-Metadata"], "enabled")
        self.assertEqual(first.headers["X-OpenRouter-Cache"], "false")
        self.assertEqual(
            first.headers["HTTP-Referer"],
            "https://example.test/cape-loop",
        )
        self.assertEqual(
            first.headers["X-OpenRouter-Title"],
            "CAPE-Loop tests",
        )

    def test_live_execution_and_runtime_key_are_both_explicit(self) -> None:
        request = build_request()
        disabled = OpenRouterChatProvider(
            OpenRouterProviderConfig(model=MODEL)
        )
        with self.assertRaises(OpenRouterLiveExecutionRequired):
            disabled.complete(request)

        enabled = OpenRouterChatProvider(live_config())
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(OpenRouterMissingAPIKey):
                enabled.complete(request)
        newline_key = OpenRouterChatProvider(live_config())
        with patch.dict(
            "os.environ",
            {TEST_KEY_ENV: "secret\r\ninjection"},
            clear=True,
        ):
            with self.assertRaisesRegex(
                OpenRouterMissingAPIKey,
                "newline",
            ):
                newline_key.complete(request)


class OpenRouterTransportAndParsingTests(unittest.TestCase):
    def test_valid_response_records_gateway_and_upstream_provenance(self) -> None:
        seen: list[dict[str, Any]] = []

        def transport(**kwargs: Any) -> HTTPResult:
            seen.append(dict(kwargs))
            return HTTPResult(
                status=200,
                headers={
                    "X-Generation-Id": "gen_test",
                    "X-OpenRouter-Cache-Status": "MISS",
                },
                body=response_body(
                    extra_metadata={"future_additive_field": {"value": 1}}
                ),
            )

        provider = OpenRouterChatProvider(
            live_config(),
            transport=transport,
            epoch_time=lambda: 1_700_000_000.0,
        )
        secret = "sk-or-test-never-retain"
        with patch.dict(
            "os.environ",
            {TEST_KEY_ENV: secret},
            clear=True,
        ):
            result = provider.complete(build_request())

        self.assertEqual(result.response.beliefs, BELIEFS)
        self.assertEqual(result.response.model_id, MODEL)
        self.assertEqual(result.provider_response_id, "gen_test")
        self.assertEqual(result.generation_id, "gen_test")
        self.assertEqual(result.cache_status, "MISS")
        self.assertEqual(result.upstream_provider, UPSTREAM_NAME)
        self.assertEqual(result.upstream_model, MODEL)
        self.assertEqual(result.routing_strategy, "direct")
        self.assertEqual(result.routing_attempt, 1)
        self.assertEqual(result.transport_attempts, 1)
        self.assertEqual(
            result.routing_metadata["future_additive_field"],
            {"value": 1},
        )
        self.assertEqual(provider.budget.request_count, 1)
        self.assertEqual(provider.budget.total_tokens, 75)
        self.assertEqual(
            seen[0]["headers"]["Authorization"],
            f"Bearer {secret}",
        )
        audit = result.to_audit_record()
        self.assertEqual(audit["provider"], "openrouter")
        self.assertEqual(audit["gateway"], "openrouter")
        self.assertFalse(audit["first_party_origin_claimed"])
        self.assertEqual(audit["upstream_provider"], UPSTREAM_NAME)
        self.assertNotIn(secret, json.dumps(audit))

    def test_secret_is_redacted_from_success_and_http_error(self) -> None:
        secret = "sk-or-secret-echo"
        echoed = response_body(
            response_id=f"gen-{secret}",
            extra_metadata={
                "summary": f"debug credential {secret}",
                f"key-{secret}": secret,
            },
            extra_raw={
                "debug": {
                    "authorization": secret,
                },
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                    "debug": secret,
                },
            },
        )
        accepted = OpenRouterChatProvider(
            live_config(upstream_provider=""),
            transport=lambda **_: HTTPResult(
                200,
                {
                    "X-Generation-Id": f"gen-{secret}",
                    "X-OpenRouter-Cache-Status": "MISS",
                },
                echoed,
            ),
        )
        with patch.dict(
            "os.environ",
            {TEST_KEY_ENV: secret},
            clear=True,
        ):
            result = accepted.complete(build_request())
        retained = json.dumps(result.to_audit_record())
        self.assertNotIn(secret, retained)
        self.assertIn("[redacted]", retained)

        rejected = OpenRouterChatProvider(
            live_config(),
            transport=lambda **_: HTTPResult(
                400,
                {"X-Generation-Id": f"header-{secret}"},
                json.dumps(
                    {
                        "error": {
                            "message": f"invalid credential {secret}",
                        }
                    }
                ).encode("utf-8"),
            ),
        )
        with patch.dict(
            "os.environ",
            {TEST_KEY_ENV: secret},
            clear=True,
        ):
            with self.assertRaises(OpenRouterHTTPError) as caught:
                rejected.complete(build_request())
        self.assertNotIn(secret, str(caught.exception))
        self.assertIn("[redacted]", str(caught.exception))

    def test_default_transport_refuses_redirects(self) -> None:
        redirect = HTTPError(
            "https://openrouter.ai/api/v1/chat/completions",
            302,
            "Found",
            {"Location": "https://attacker.example.test/collect"},
            BytesIO(b'{"error":{"message":"redirect refused"}}'),
        )
        with patch.object(
            openrouter_provider,
            "build_opener",
        ) as build:
            build.return_value.open.side_effect = redirect
            result = urllib_transport(
                url="https://openrouter.ai/api/v1/chat/completions",
                body=b"{}",
                headers={"Authorization": "Bearer secret"},
                timeout=1,
            )
        self.assertEqual(result.status, 302)
        redirect_handler = build.call_args.args[0]
        self.assertIsNone(
            redirect_handler.redirect_request(
                object(),
                None,
                302,
                "Found",
                {},
                "https://attacker.example.test/collect",
            )
        )

    def test_default_transport_caps_success_and_error_bodies(self) -> None:
        for status in (200, 413):
            with self.subTest(status=status):
                body = _TrackingBody(b"sensitive" + (b"x" * 64), status=status)
                with (
                    patch.object(
                        openrouter_provider,
                        "HTTP_RESPONSE_BODY_LIMIT_BYTES",
                        32,
                    ),
                    patch.object(
                        openrouter_provider,
                        "build_opener",
                    ) as build,
                ):
                    if status == 200:
                        build.return_value.open.return_value = body
                    else:
                        build.return_value.open.side_effect = HTTPError(
                            "https://openrouter.ai/api/v1/chat/completions",
                            status,
                            "Content Too Large",
                            {},
                            body,
                        )
                    with self.assertRaises(
                        openrouter_provider.OpenRouterHTTPResponseBodyTooLarge
                    ) as caught:
                        urllib_transport(
                            url=(
                                "https://openrouter.ai/api/v1/"
                                "chat/completions"
                            ),
                            body=b"{}",
                            headers={"Authorization": "Bearer request-secret"},
                            timeout=1,
                        )
                self.assertEqual(body.read_sizes, [33])
                self.assertEqual(caught.exception.status, status)
                self.assertNotIn("sensitive", str(caught.exception))
                self.assertTrue(body.closed)


class OpenRouterRejectionAndBudgetTests(unittest.TestCase):
    def test_paid_identity_failures_are_rejected_and_audit_journaled(
        self,
    ) -> None:
        cases = {
            "cache": (
                response_body(),
                {"X-OpenRouter-Cache-Status": "HIT"},
                "gateway cache",
            ),
            "fallback": (
                response_body(routing_attempt=2),
                {"X-OpenRouter-Cache-Status": "MISS"},
                "fallback occurred",
            ),
            "alias": (
                response_body(
                    requested_model="~google/gemini-flash-latest",
                ),
                {"X-OpenRouter-Cache-Status": "MISS"},
                "requested model differs",
            ),
            "pipeline": (
                response_body(
                    pipeline=[
                        {
                            "type": "response_healing",
                            "name": "response-healing",
                            "data": {"improved": True},
                        }
                    ]
                ),
                {"X-OpenRouter-Cache-Status": "MISS"},
                "pipeline materially transformed",
            ),
            "model": (
                response_body(
                    model="google/gemini-3.5-flash",
                    requested_model=MODEL,
                    upstream_model="google/gemini-3.5-flash",
                ),
                {"X-OpenRouter-Cache-Status": "MISS"},
                "returned model differs",
            ),
            "upstream": (
                response_body(upstream_provider="Google Vertex"),
                {"X-OpenRouter-Cache-Status": "MISS"},
                "configured route",
            ),
        }
        request = build_request()
        for label, (body, headers, reason) in cases.items():
            with self.subTest(label=label), TemporaryDirectory() as directory:
                root = Path(directory)
                calls: list[int] = []

                def transport(
                    *,
                    _body: bytes = body,
                    _headers: Mapping[str, str] = headers,
                    **_: Any,
                ) -> HTTPResult:
                    calls.append(1)
                    return HTTPResult(200, _headers, _body)

                provider = OpenRouterChatProvider(
                    live_config(),
                    transport=transport,
                )
                with patch.dict(
                    "os.environ",
                    {TEST_KEY_ENV: "test"},
                    clear=True,
                ):
                    with self.assertRaisesRegex(
                        OpenRouterResultRejected,
                        reason,
                    ):
                        execute_openrouter_requests(
                            provider,
                            (request,),
                            responses_path=root / "responses.jsonl",
                            audit_path=root / "audit.jsonl",
                        )
                self.assertEqual(calls, [1])
                self.assertFalse((root / "responses.jsonl").exists())
                audit = json.loads(
                    (root / "audit.jsonl").read_text(encoding="utf-8")
                )
                self.assertEqual(
                    audit["acceptance_status"],
                    "rejected_openrouter_identity",
                )
                self.assertEqual(audit["provider"], "openrouter")
                self.assertEqual(audit["request_id"], request.request_id)
                self.assertEqual(provider.budget.request_count, 1)
                adaptive_audit = root / "adaptive-audit.jsonl"
                adaptive_audit.write_bytes(
                    (root / "audit.jsonl").read_bytes()
                )
                with self.assertRaisesRegex(
                    ValueError,
                    "rejected",
                ):
                    ResumableOpenRouterCompletionProvider(
                        OpenRouterChatProvider(
                            live_config(live_execution=False)
                        ),
                        responses_path=root / "adaptive-responses.jsonl",
                        audit_path=adaptive_audit,
                    )

    def test_missing_router_metadata_is_invalid_and_charged(self) -> None:
        raw = json.loads(response_body())
        raw.pop("openrouter_metadata")
        provider = OpenRouterChatProvider(
            live_config(),
            transport=lambda **_: HTTPResult(
                200,
                {"X-OpenRouter-Cache-Status": "MISS"},
                json.dumps(raw).encode("utf-8"),
            ),
        )
        reservation = provider.prepare(build_request()).estimated_max_tokens
        with patch.dict(
            "os.environ",
            {TEST_KEY_ENV: "test"},
            clear=True,
        ):
            with self.assertRaisesRegex(
                OpenRouterResponseError,
                "router metadata",
            ):
                provider.complete(build_request())
        self.assertEqual(provider.budget.request_count, 1)
        self.assertEqual(provider.budget.total_tokens, reservation)

    def test_retry_after_and_hard_budgets(self) -> None:
        outcomes = [
            HTTPResult(
                429,
                {"Retry-After": "3"},
                b'{"error":{"message":"slow down"}}',
            ),
            HTTPResult(
                503,
                {},
                b'{"error":{"message":"temporarily unavailable"}}',
            ),
            HTTPResult(
                200,
                {"X-OpenRouter-Cache-Status": "MISS"},
                response_body(),
            ),
        ]
        seen_bodies: list[bytes] = []
        sleeps: list[float] = []

        def transport(**kwargs: Any) -> HTTPResult:
            seen_bodies.append(kwargs["body"])
            return outcomes.pop(0)

        provider = OpenRouterChatProvider(
            live_config(
                max_retries=2,
                initial_backoff_seconds=1,
                max_backoff_seconds=4,
                jitter_fraction=0,
            ),
            transport=transport,
            sleep=sleeps.append,
            random_value=lambda: 0.5,
            epoch_time=lambda: 1_700_000_000.0,
        )
        with patch.dict(
            "os.environ",
            {TEST_KEY_ENV: "test"},
            clear=True,
        ):
            result = provider.complete(build_request())
        self.assertEqual(result.transport_attempts, 3)
        self.assertEqual(sleeps, [3.0, 2.0])
        self.assertEqual(len(set(seen_bodies)), 1)
        self.assertEqual(provider.budget.request_count, 3)
        self.assertEqual(provider.budget.total_tokens, 75)

        calls: list[int] = []
        capped = OpenRouterChatProvider(
            live_config(max_requests=1),
            transport=lambda **_: (
                calls.append(1)
                or HTTPResult(
                    200,
                    {"X-OpenRouter-Cache-Status": "MISS"},
                    response_body(),
                )
            ),
        )
        with patch.dict(
            "os.environ",
            {TEST_KEY_ENV: "test"},
            clear=True,
        ):
            capped.complete(build_request("first"))
            with self.assertRaisesRegex(
                OpenRouterBudgetExceeded,
                "max_requests",
            ):
                capped.complete(build_request("second"))
        self.assertEqual(calls, [1])

        token_capped = OpenRouterChatProvider(
            live_config(
                max_output_tokens=100,
                max_total_tokens=100,
            ),
            transport=lambda **_: calls.append(1),  # type: ignore[arg-type]
        )
        with patch.dict(
            "os.environ",
            {TEST_KEY_ENV: "test"},
            clear=True,
        ):
            with self.assertRaisesRegex(
                OpenRouterBudgetExceeded,
                "token reservation",
            ):
                token_capped.complete(build_request())
        self.assertEqual(calls, [1])

    def test_ambiguous_transport_failure_is_not_automatically_retried(
        self,
    ) -> None:
        calls: list[int] = []
        sleeps: list[float] = []

        def transport(**_: Any) -> HTTPResult:
            calls.append(1)
            raise TimeoutError("socket outcome unknown")

        provider = OpenRouterChatProvider(
            live_config(max_retries=4),
            transport=transport,
            sleep=sleeps.append,
        )
        with patch.dict(
            "os.environ",
            {TEST_KEY_ENV: "sk-or-do-not-reflect"},
            clear=True,
        ):
            with self.assertRaisesRegex(
                OpenRouterProviderError,
                "ambiguous.*automatic retry is disabled",
            ) as caught:
                provider.complete(build_request())
        self.assertEqual(calls, [1])
        self.assertEqual(sleeps, [])
        self.assertEqual(provider.budget.request_count, 1)
        self.assertEqual(provider.budget.total_tokens, 0)
        self.assertNotIn("sk-or-do-not-reflect", str(caught.exception))


class OpenRouterResumeTests(unittest.TestCase):
    def test_static_executor_resumes_without_repeating_accepted_calls(
        self,
    ) -> None:
        requests = (build_request("first"), build_request("second"))
        with TemporaryDirectory() as directory:
            root = Path(directory)
            responses_path = root / "responses.jsonl"
            audit_path = root / "audit.jsonl"
            first_calls: list[int] = []

            def first_transport(**_: Any) -> HTTPResult:
                first_calls.append(1)
                return HTTPResult(
                    200,
                    {"X-OpenRouter-Cache-Status": "MISS"},
                    response_body("gen_first"),
                )

            first = OpenRouterChatProvider(
                live_config(max_requests=1),
                transport=first_transport,
            )
            with patch.dict(
                "os.environ",
                {TEST_KEY_ENV: "test"},
                clear=True,
            ):
                with self.assertRaises(OpenRouterBudgetExceeded):
                    execute_openrouter_requests(
                        first,
                        requests,
                        responses_path=responses_path,
                        audit_path=audit_path,
                    )
            self.assertEqual(first_calls, [1])
            self.assertEqual(len(read_responses(responses_path)), 1)

            second_calls: list[int] = []

            def second_transport(**_: Any) -> HTTPResult:
                second_calls.append(1)
                return HTTPResult(
                    200,
                    {"X-OpenRouter-Cache-Status": "MISS"},
                    response_body("gen_second"),
                )

            second = OpenRouterChatProvider(
                live_config(max_requests=2),
                transport=second_transport,
            )
            with patch.dict(
                "os.environ",
                {TEST_KEY_ENV: "test"},
                clear=True,
            ):
                summary = execute_openrouter_requests(
                    second,
                    requests,
                    responses_path=responses_path,
                    audit_path=audit_path,
                )
            self.assertEqual(second_calls, [1])
            self.assertEqual(summary.resumed_count, 1)
            self.assertEqual(summary.executed_count, 1)
            responses = read_responses(responses_path)
            ReplayProvider(responses).validate_coverage(requests)

    def test_static_resume_rejects_route_configuration_drift(self) -> None:
        request = build_request()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            responses_path = root / "responses.jsonl"
            audit_path = root / "audit.jsonl"
            first = OpenRouterChatProvider(
                live_config(),
                transport=lambda **_: HTTPResult(
                    200,
                    {"X-OpenRouter-Cache-Status": "MISS"},
                    response_body(),
                ),
            )
            with patch.dict(
                "os.environ",
                {TEST_KEY_ENV: "test"},
                clear=True,
            ):
                execute_openrouter_requests(
                    first,
                    (request,),
                    responses_path=responses_path,
                    audit_path=audit_path,
                )
            calls: list[int] = []
            changed_route = OpenRouterChatProvider(
                live_config(
                    upstream_provider="google-vertex",
                    live_execution=False,
                ),
                transport=lambda **_: calls.append(1),  # type: ignore[arg-type]
            )
            with self.assertRaisesRegex(
                ValueError,
                "does not match the current request",
            ):
                execute_openrouter_requests(
                    changed_route,
                    (request,),
                    responses_path=responses_path,
                    audit_path=audit_path,
                )
            self.assertEqual(calls, [])

    def test_adaptive_resume_revalidates_tampered_upstream_provenance(
        self,
    ) -> None:
        request = build_request()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            responses_path = root / "responses.jsonl"
            audit_path = root / "audit.jsonl"
            calls: list[int] = []

            def transport(**_: Any) -> HTTPResult:
                calls.append(1)
                return HTTPResult(
                    200,
                    {"X-OpenRouter-Cache-Status": "MISS"},
                    response_body(),
                )

            first = ResumableOpenRouterCompletionProvider(
                OpenRouterChatProvider(
                    live_config(),
                    transport=transport,
                ),
                responses_path=responses_path,
                audit_path=audit_path,
            )
            with patch.dict(
                "os.environ",
                {TEST_KEY_ENV: "test"},
                clear=True,
            ):
                accepted = first.complete(request)
            self.assertEqual(calls, [1])

            no_calls: list[int] = []
            resumed = ResumableOpenRouterCompletionProvider(
                OpenRouterChatProvider(
                    live_config(),
                    transport=lambda **_: no_calls.append(1),  # type: ignore[arg-type]
                ),
                responses_path=responses_path,
                audit_path=audit_path,
            )
            self.assertEqual(resumed.complete(request), accepted)
            self.assertEqual(no_calls, [])
            manifest = resumed.to_manifest()
            self.assertEqual(manifest["provider"], "openrouter")
            self.assertEqual(
                manifest["upstream_providers_returned"],
                [UPSTREAM_NAME],
            )

            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            audit["upstream_provider"] = "Google Vertex"
            audit["routing_metadata"]["endpoints"]["available"][0][
                "provider"
            ] = "Google Vertex"
            audit["routing_metadata"]["attempts"][0][
                "provider"
            ] = "Google Vertex"
            audit["raw_response"]["openrouter_metadata"] = audit[
                "routing_metadata"
            ]
            audit_path.write_text(
                json.dumps(
                    audit,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n",
                encoding="utf-8",
            )
            tampered = ResumableOpenRouterCompletionProvider(
                OpenRouterChatProvider(
                    live_config(live_execution=False),
                    transport=lambda **_: no_calls.append(1),  # type: ignore[arg-type]
                ),
                responses_path=responses_path,
                audit_path=audit_path,
            )
            with self.assertRaisesRegex(
                ValueError,
                "routing audit",
            ):
                tampered.complete(request)
            self.assertEqual(no_calls, [])


if __name__ == "__main__":
    unittest.main()
