from __future__ import annotations

from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.error import HTTPError
from unittest.mock import patch
import json
import unittest

import cape_loop.external_decoder_providers as external_providers
from cape_loop.decoder_study import (
    ExternalDecoderJudgment,
    ExternalDecoderRequest,
    read_external_decoder_judgments,
)
from cape_loop.external_decoder_providers import (
    ANTHROPIC_API_DOC_URL,
    ANTHROPIC_DEFAULT_MODEL,
    ANTHROPIC_MODEL_DOC_URL,
    GEMINI_API_DOC_URL,
    GEMINI_DEFAULT_MODEL,
    GEMINI_MODEL_DOC_URL,
    OFFICIAL_SOURCE_RESOLVED_DATE,
    ExternalDecoderBudgetExceeded,
    ExternalDecoderExecutionLocked,
    ExternalDecoderHTTPError,
    ExternalDecoderIdentityMismatch,
    ExternalDecoderProvider,
    ExternalDecoderProviderConfig,
    ExternalDecoderProviderError,
    ExternalDecoderResponseError,
    HTTPResult,
    LiveExternalDecoderExecutionRequired,
    MissingExternalDecoderAPIKey,
    default_external_decoder_configs,
    execute_external_decoder_collection,
    plan_external_decoder_collection,
    prepare_external_decoder_request,
    urllib_transport,
)
from cape_loop.schema_export import SCHEMAS


class _TrackingBody(BytesIO):
    def __init__(self, payload: bytes) -> None:
        super().__init__(payload)
        self.read_sizes: list[int] = []
        self.status = 200
        self.headers: dict[str, str] = {}

    def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        return super().read(size)


BELIEFS = {
    f"attribute_{attribute}": {
        "-2": 0.1,
        "-1": 0.2,
        "+1": 0.3,
        "+2": 0.4,
    }
    for attribute in range(1, 4)
}


def decoder_request(label: str = "one") -> ExternalDecoderRequest:
    return ExternalDecoderRequest.build(
        request_id=f"decoder-{label}",
        pseudonymous_state_id=f"state-{label}",
        representation_id="blinded-native-content-v1",
        evaluation_split="development",
        payload={
            "representation_version": "blinded-native-content-v1",
            "episodes": [],
            "semantic_claims": [],
            "persona_text": "",
        },
    )


def anthropic_response(
    *,
    model: str = ANTHROPIC_DEFAULT_MODEL,
    secret_echo: str | None = None,
) -> bytes:
    raw = {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [
            {
                "type": "text",
                "text": json.dumps({"beliefs": BELIEFS}),
            }
        ],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 80, "output_tokens": 40},
    }
    if secret_echo is not None:
        raw["debug"] = {
            "authorization": secret_echo,
            "message": f"echo={secret_echo}",
        }
        raw[f"provider-{secret_echo}-field"] = "also unsafe as a key"
    return json.dumps(raw).encode("utf-8")


def gemini_response(
    *,
    model: str = GEMINI_DEFAULT_MODEL,
) -> bytes:
    return json.dumps(
        {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {
                                "text": json.dumps(
                                    {"beliefs": BELIEFS}
                                )
                            }
                        ],
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 70,
                "candidatesTokenCount": 30,
                "thoughtsTokenCount": 10,
                "totalTokenCount": 110,
            },
            "modelVersion": model,
            "responseId": "gemini-response-test",
            "modelStatus": {"modelStage": "STABLE"},
        }
    ).encode("utf-8")


class ExternalDecoderPlanningTests(unittest.TestCase):
    def test_defaults_are_current_stable_distinct_families(self) -> None:
        configs = default_external_decoder_configs()
        self.assertEqual(
            [(item.provider, item.model) for item in configs],
            [
                ("anthropic", "claude-sonnet-5"),
                ("google_gemini", "gemini-3.6-flash"),
            ],
        )
        self.assertEqual(OFFICIAL_SOURCE_RESOLVED_DATE, "2026-07-26")
        for url in (
            ANTHROPIC_MODEL_DOC_URL,
            ANTHROPIC_API_DOC_URL,
            GEMINI_MODEL_DOC_URL,
            GEMINI_API_DOC_URL,
        ):
            self.assertTrue(url.startswith("https://"))

    def test_keyless_plan_is_deterministic_and_does_not_read_keys(self) -> None:
        requests = (decoder_request("b"), decoder_request("a"))
        with patch.dict(
            "os.environ",
            {
                "ANTHROPIC_API_KEY": "must-not-be-read",
                "GEMINI_API_KEY": "must-not-be-read",
            },
            clear=True,
        ):
            first = plan_external_decoder_collection(requests)
            second = plan_external_decoder_collection(reversed(requests))
        self.assertEqual(first, second)
        self.assertFalse(first["credential_read"])
        self.assertTrue(first["distinct_provider_model_families"])
        self.assertEqual(first["source_count"], 2)
        retained = json.dumps(first)
        self.assertNotIn("must-not-be-read", retained)
        self.assertTrue(
            all(
                not source["credential_read"]
                for source in first["sources"]
            )
        )
        for source in first["sources"]:
            self.assertEqual(
                source["budget_accounting_unit"],
                "actual_transport_attempt",
            )
            self.assertEqual(
                source["maximum_attempts_per_request"],
                1,
            )
            self.assertEqual(
                source["theoretical_max_transport_attempts"],
                source["request_count"],
            )

    def test_plan_fails_before_execution_when_ceiling_is_too_low(self) -> None:
        configs = default_external_decoder_configs(max_requests=1)
        with self.assertRaises(ExternalDecoderBudgetExceeded):
            plan_external_decoder_collection(
                (decoder_request("one"), decoder_request("two")),
                configs,
            )

    def test_plan_requires_capacity_for_all_retry_attempts(self) -> None:
        configs = tuple(
            ExternalDecoderProviderConfig(
                provider=provider,
                max_retries=1,
                max_requests=2,
                max_total_tokens=6_000_000,
            )
            for provider in ("anthropic", "google_gemini")
        )
        with self.assertRaisesRegex(
            ExternalDecoderBudgetExceeded,
            "4 physical transport attempts after retry expansion",
        ):
            plan_external_decoder_collection(
                (decoder_request("one"), decoder_request("two")),
                configs,
            )

    def test_plan_rejects_retry_expanded_tokens_when_initial_fits(
        self,
    ) -> None:
        request = decoder_request()
        baseline = ExternalDecoderProviderConfig(
            provider="anthropic",
            max_retries=1,
            max_requests=10,
            max_total_tokens=6_000_000,
        )
        initial_tokens = prepare_external_decoder_request(
            request,
            baseline,
        ).estimated_max_tokens
        configs = (
            ExternalDecoderProviderConfig(
                provider="anthropic",
                max_retries=1,
                max_requests=10,
                max_total_tokens=initial_tokens,
            ),
            ExternalDecoderProviderConfig(
                provider="google_gemini",
                max_retries=0,
                max_requests=10,
                max_total_tokens=6_000_000,
            ),
        )
        with self.assertRaisesRegex(
            ExternalDecoderBudgetExceeded,
            "tokens after retry expansion",
        ):
            plan_external_decoder_collection((request,), configs)

    def test_custom_origins_require_two_explicit_safety_controls(self) -> None:
        with self.assertRaisesRegex(ValueError, "official"):
            ExternalDecoderProviderConfig(
                provider="anthropic",
                base_url="https://proxy.example.test",
            )
        with self.assertRaisesRegex(ValueError, "dedicated credential"):
            ExternalDecoderProviderConfig(
                provider="anthropic",
                base_url="https://proxy.example.test",
                allow_custom_base_url=True,
            )
        configured = ExternalDecoderProviderConfig(
            provider="anthropic",
            base_url="https://proxy.example.test",
            allow_custom_base_url=True,
            api_key_env="CAPE_LOOP_ANTHROPIC_PROXY_KEY",
        )
        self.assertEqual(
            configured.endpoint,
            "https://proxy.example.test/v1/messages",
        )
        with self.assertRaisesRegex(ValueError, "userinfo"):
            ExternalDecoderProviderConfig(
                provider="google_gemini",
                base_url="https://user@generativelanguage.googleapis.com",
            )
        for provider, default_env in (
            ("anthropic", "GEMINI_API_KEY"),
            ("google_gemini", "ANTHROPIC_API_KEY"),
            ("anthropic", "OPENAI_API_KEY"),
            ("google_gemini", "OPENROUTER_API_KEY"),
        ):
            with self.subTest(provider=provider, default_env=default_env):
                with self.assertRaisesRegex(
                    ValueError,
                    "reserved provider default",
                ):
                    ExternalDecoderProviderConfig(
                        provider=provider,
                        base_url="https://proxy.example.test",
                        allow_custom_base_url=True,
                        api_key_env=default_env,
                    )

    def test_official_origins_reject_other_provider_key_variables(
        self,
    ) -> None:
        for provider, reserved_envs in (
            (
                "anthropic",
                (
                    "OPENAI_API_KEY",
                    "OPENROUTER_API_KEY",
                    "GEMINI_API_KEY",
                ),
            ),
            (
                "google_gemini",
                (
                    "OPENAI_API_KEY",
                    "OPENROUTER_API_KEY",
                    "ANTHROPIC_API_KEY",
                ),
            ),
        ):
            for reserved_env in reserved_envs:
                with self.subTest(
                    provider=provider,
                    reserved_env=reserved_env,
                ):
                    with self.assertRaisesRegex(
                        ValueError,
                        "reserved for a different provider",
                    ):
                        ExternalDecoderProviderConfig(
                            provider=provider,
                            api_key_env=reserved_env,
                        )

    def test_two_sources_cannot_share_a_credential_environment(self) -> None:
        configs = (
            ExternalDecoderProviderConfig(
                provider="anthropic",
                api_key_env="CAPE_LOOP_SHARED_DECODER_KEY",
            ),
            ExternalDecoderProviderConfig(
                provider="google_gemini",
                api_key_env="CAPE_LOOP_SHARED_DECODER_KEY",
            ),
        )
        with self.assertRaisesRegex(ValueError, "must not share"):
            plan_external_decoder_collection((decoder_request(),), configs)


class ExternalDecoderPreparationTests(unittest.TestCase):
    def test_anthropic_uses_messages_structured_outputs_without_key(self) -> None:
        config = ExternalDecoderProviderConfig(provider="anthropic")
        with patch.dict("os.environ", {}, clear=True):
            prepared = prepare_external_decoder_request(
                decoder_request(),
                config,
            )
        self.assertEqual(
            prepared.endpoint,
            "https://api.anthropic.com/v1/messages",
        )
        self.assertEqual(prepared.body["model"], "claude-sonnet-5")
        self.assertEqual(
            prepared.body["output_config"]["effort"],  # type: ignore[index]
            "low",
        )
        self.assertEqual(
            prepared.body["thinking"],  # type: ignore[index]
            {"type": "disabled"},
        )
        self.assertNotIn("temperature", prepared.body)
        output_format = prepared.body["output_config"]["format"]  # type: ignore[index]
        self.assertEqual(output_format["type"], "json_schema")
        self.assertFalse(output_format["schema"]["additionalProperties"])
        self.assertNotIn("x-api-key", prepared.headers)
        self.assertEqual(
            prepared.headers["X-Client-Request-Id"],
            prepared.client_request_id,
        )
        self.assertEqual(
            prepared.body_sha256,
            prepare_external_decoder_request(
                decoder_request(),
                config,
            ).body_sha256,
        )

    def test_gemini_uses_generate_content_json_schema_without_key(self) -> None:
        config = ExternalDecoderProviderConfig(provider="google_gemini")
        prepared = prepare_external_decoder_request(
            decoder_request(),
            config,
        )
        self.assertEqual(
            prepared.endpoint,
            "https://generativelanguage.googleapis.com/v1beta/models/"
            "gemini-3.6-flash:generateContent",
        )
        generation = prepared.body["generationConfig"]
        self.assertNotIn("candidateCount", generation)
        self.assertNotIn("temperature", generation)
        response_text = generation["responseFormat"]["text"]  # type: ignore[index]
        self.assertEqual(
            response_text["mimeType"],
            "application/json",
        )
        self.assertEqual(
            generation["thinkingConfig"]["thinkingLevel"],  # type: ignore[index]
            "low",
        )
        self.assertEqual(
            response_text["schema"]["required"],
            ["beliefs"],
        )
        self.assertNotIn("x-goog-api-key", prepared.headers)
        self.assertEqual(
            prepared.headers["X-Client-Request-Id"],
            prepared.client_request_id,
        )

    def test_live_execution_and_environment_key_are_both_required(self) -> None:
        request = decoder_request()
        provider = ExternalDecoderProvider(
            ExternalDecoderProviderConfig(provider="anthropic")
        )
        with self.assertRaises(LiveExternalDecoderExecutionRequired):
            provider.complete(request)
        enabled = ExternalDecoderProvider(
            ExternalDecoderProviderConfig(
                provider="anthropic",
                live_execution=True,
                api_key_env="ABSENT_CAPE_LOOP_ANTHROPIC_KEY",
            )
        )
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(MissingExternalDecoderAPIKey):
                enabled.complete(request)


class ExternalDecoderTransportTests(unittest.TestCase):
    def test_anthropic_result_is_import_compatible_and_redacted(self) -> None:
        seen: list[dict[str, object]] = []
        secret = "sk-ant-test-never-retain"

        def transport(**kwargs: object) -> HTTPResult:
            seen.append(dict(kwargs))
            return HTTPResult(
                status=200,
                headers={"request-id": "anthropic-server-request"},
                body=anthropic_response(secret_echo=secret),
            )

        provider = ExternalDecoderProvider(
            ExternalDecoderProviderConfig(
                provider="anthropic",
                live_execution=True,
                api_key_env="CAPE_LOOP_TEST_ANTHROPIC_KEY",
                max_total_tokens=100_000,
            ),
            transport=transport,
            epoch_time=lambda: 1_700_000_000.0,
        )
        with patch.dict(
            "os.environ",
            {"CAPE_LOOP_TEST_ANTHROPIC_KEY": secret},
            clear=True,
        ):
            result = provider.complete(decoder_request())
        self.assertEqual(
            seen[0]["headers"]["x-api-key"],  # type: ignore[index]
            secret,
        )
        reparsed = ExternalDecoderJudgment.parse(
            result.judgment.to_dict()
        )
        self.assertEqual(reparsed, result.judgment)
        self.assertEqual(
            result.judgment.decoder_family_id,
            "anthropic-claude",
        )
        self.assertEqual(provider.budget.total_tokens, 120)
        audit_record = result.to_audit_record()
        audit = json.dumps(audit_record)
        self.assertNotIn(secret, audit)
        self.assertIn("[REDACTED]", audit)
        schema = SCHEMAS["external-decoder-provider-audit"]
        self.assertEqual(
            set(audit_record),
            set(schema["properties"]),
        )
        self.assertTrue(
            set(schema["required"]).issubset(audit_record)
        )

    def test_gemini_result_checks_identity_and_usage(self) -> None:
        provider = ExternalDecoderProvider(
            ExternalDecoderProviderConfig(
                provider="google_gemini",
                live_execution=True,
                api_key_env="CAPE_LOOP_TEST_GEMINI_KEY",
                max_total_tokens=100_000,
            ),
            transport=lambda **_: HTTPResult(
                status=200,
                headers={"x-request-id": "google-server-request"},
                body=gemini_response(),
            ),
        )
        with patch.dict(
            "os.environ",
            {"CAPE_LOOP_TEST_GEMINI_KEY": "google-test-secret"},
            clear=True,
        ):
            result = provider.complete(decoder_request())
        self.assertEqual(result.model_returned, "gemini-3.6-flash")
        self.assertEqual(
            result.judgment.decoder_family_id,
            "google-gemini",
        )
        self.assertEqual(provider.budget.total_tokens, 110)
        self.assertEqual(
            result.server_request_id,
            "google-server-request",
        )

    def test_model_mismatch_is_rejected_after_conservative_accounting(
        self,
    ) -> None:
        provider = ExternalDecoderProvider(
            ExternalDecoderProviderConfig(
                provider="anthropic",
                live_execution=True,
                api_key_env="CAPE_LOOP_TEST_ANTHROPIC_KEY",
                max_total_tokens=100_000,
            ),
            transport=lambda **_: HTTPResult(
                200,
                {},
                anthropic_response(model="claude-opus-5"),
            ),
        )
        with patch.dict(
            "os.environ",
            {"CAPE_LOOP_TEST_ANTHROPIC_KEY": "anthropic-test-api-key"},
            clear=True,
        ):
            with self.assertRaises(ExternalDecoderIdentityMismatch):
                provider.complete(decoder_request())
        self.assertEqual(provider.budget.request_count, 1)

    def test_retry_and_http_errors_never_retain_the_key(self) -> None:
        calls: list[int] = []

        def retrying(**_: object) -> HTTPResult:
            calls.append(1)
            if len(calls) == 1:
                return HTTPResult(
                    429,
                    {"Retry-After": "0"},
                    b'{"error":{"message":"retry"}}',
                )
            return HTTPResult(200, {}, gemini_response())

        provider = ExternalDecoderProvider(
            ExternalDecoderProviderConfig(
                provider="google_gemini",
                live_execution=True,
                api_key_env="CAPE_LOOP_TEST_GEMINI_KEY",
                max_retries=1,
                initial_backoff_seconds=0,
                max_backoff_seconds=0,
                max_total_tokens=100_000,
            ),
            transport=retrying,
            sleep=lambda _: None,
        )
        with patch.dict(
            "os.environ",
            {"CAPE_LOOP_TEST_GEMINI_KEY": "google-test-secret"},
            clear=True,
        ):
            provider.complete(decoder_request())
        self.assertEqual(calls, [1, 1])
        self.assertEqual(provider.budget.request_count, 2)
        self.assertGreater(provider.budget.total_tokens, 110)

        secret = "google-error-secret"
        failed = ExternalDecoderProvider(
            ExternalDecoderProviderConfig(
                provider="google_gemini",
                live_execution=True,
                api_key_env="CAPE_LOOP_TEST_GEMINI_KEY",
                max_retries=0,
                max_total_tokens=100_000,
            ),
            transport=lambda **_: HTTPResult(
                400,
                {},
                json.dumps(
                    {"error": {"message": f"bad key {secret}"}}
                ).encode(),
            ),
        )
        with patch.dict(
            "os.environ",
            {"CAPE_LOOP_TEST_GEMINI_KEY": secret},
            clear=True,
        ):
            with self.assertRaises(ExternalDecoderHTTPError) as raised:
                failed.complete(decoder_request())
        self.assertNotIn(secret, str(raised.exception))
        self.assertEqual(failed.budget.request_count, 1)

    def test_ambiguous_transport_failure_is_never_retried(self) -> None:
        calls: list[int] = []
        provider = ExternalDecoderProvider(
            ExternalDecoderProviderConfig(
                provider="google_gemini",
                live_execution=True,
                api_key_env="CAPE_LOOP_TEST_GEMINI_KEY",
                max_retries=9,
                initial_backoff_seconds=0,
                max_backoff_seconds=0,
                max_total_tokens=100_000,
            ),
            transport=lambda **_: (
                calls.append(1)
                or (_ for _ in ()).throw(TimeoutError("outcome unknown"))
            ),
            sleep=lambda _: self.fail(
                "ambiguous transport outcomes must not back off and retry"
            ),
        )
        reservation = provider.prepare(
            decoder_request()
        ).estimated_max_tokens
        with patch.dict(
            "os.environ",
            {"CAPE_LOOP_TEST_GEMINI_KEY": "google-test-secret"},
            clear=True,
        ):
            with self.assertRaisesRegex(
                ExternalDecoderProviderError,
                "ambiguous.*manual review",
            ):
                provider.complete(decoder_request())
        self.assertEqual(calls, [1])
        self.assertEqual(provider.budget.request_count, 1)
        self.assertEqual(provider.budget.total_tokens, reservation)

    def test_oversized_response_is_charged_without_body_reflection(self) -> None:
        secret = "oversized-provider-secret"

        def oversized_response(**_: object) -> HTTPResult:
            raise external_providers.HTTPResponseBodyTooLarge(status=413)

        provider = ExternalDecoderProvider(
            ExternalDecoderProviderConfig(
                provider="anthropic",
                live_execution=True,
                api_key_env="CAPE_LOOP_TEST_ANTHROPIC_KEY",
                max_total_tokens=100_000,
            ),
            transport=oversized_response,
        )
        expected_charge = provider.prepare(
            decoder_request()
        ).estimated_max_tokens
        with patch.dict(
            "os.environ",
            {"CAPE_LOOP_TEST_ANTHROPIC_KEY": secret},
            clear=True,
        ):
            with self.assertRaises(ExternalDecoderResponseError) as caught:
                provider.complete(decoder_request())
        self.assertNotIn(secret, str(caught.exception))
        self.assertEqual(provider.budget.request_count, 1)
        self.assertEqual(provider.budget.total_tokens, expected_charge)

    def test_provider_metadata_cannot_echo_the_api_key(self) -> None:
        secret = "sk-ant-metadata-secret"

        def mutated_response(field: str) -> bytes:
            raw = json.loads(anthropic_response())
            if field == "usage":
                raw["usage"]["input_tokens"] = secret
            else:
                raw[field] = secret
            return json.dumps(raw).encode("utf-8")

        cases = (
            ("id", {}, mutated_response("id")),
            ("model", {}, mutated_response("model")),
            ("usage", {}, mutated_response("usage")),
            (
                "server_request_id",
                {"request-id": secret},
                anthropic_response(),
            ),
        )
        for name, headers, body in cases:
            with self.subTest(name=name):
                provider = ExternalDecoderProvider(
                    ExternalDecoderProviderConfig(
                        provider="anthropic",
                        live_execution=True,
                        api_key_env="CAPE_LOOP_TEST_ANTHROPIC_KEY",
                        max_total_tokens=100_000,
                    ),
                    transport=lambda h=headers, b=body, **_: HTTPResult(
                        200,
                        h,
                        b,
                    ),
                )
                with patch.dict(
                    "os.environ",
                    {"CAPE_LOOP_TEST_ANTHROPIC_KEY": secret},
                    clear=True,
                ):
                    with self.assertRaises(
                        ExternalDecoderResponseError
                    ) as raised:
                        provider.complete(decoder_request())
                self.assertNotIn(secret, str(raised.exception))
                self.assertEqual(provider.budget.request_count, 1)

    def test_urllib_transport_disables_redirects(self) -> None:
        handlers: list[object] = []

        class RedirectingOpener:
            def open(self, request: object, *, timeout: float) -> object:
                del request, timeout
                raise HTTPError(
                    "https://api.example.test",
                    302,
                    "Found",
                    {"Location": "https://attacker.example.test"},
                    BytesIO(b'{"error":{"message":"redirect"}}'),
                )

        def fake_build_opener(*installed: object) -> RedirectingOpener:
            handlers.extend(installed)
            return RedirectingOpener()

        with patch.object(
            external_providers,
            "build_opener",
            side_effect=fake_build_opener,
        ):
            response = urllib_transport(
                url="https://api.example.test",
                body=b"{}",
                headers={"Authorization": "not-a-real-key"},
                timeout=1,
            )
        self.assertEqual(response.status, 302)
        self.assertEqual(len(handlers), 1)
        self.assertIsInstance(
            handlers[0],
            external_providers._NoRedirectHandler,
        )
        self.assertIsNone(
            handlers[0].redirect_request(  # type: ignore[union-attr]
                None,
                None,
                302,
                "Found",
                {},
                "https://attacker.example.test",
            )
        )

    def test_urllib_transport_caps_success_response_body(self) -> None:
        secret = "oversized-success-secret"
        body = _TrackingBody(secret.encode("utf-8") + (b"x" * 64))
        with (
            patch.object(
                external_providers,
                "HTTP_RESPONSE_BODY_LIMIT_BYTES",
                32,
            ),
            patch.object(external_providers, "build_opener") as build,
        ):
            build.return_value.open.return_value = body
            with self.assertRaises(
                external_providers.HTTPResponseBodyTooLarge
            ) as caught:
                urllib_transport(
                    url="https://api.anthropic.com/v1/messages",
                    body=b"{}",
                    headers={"x-api-key": "request-secret"},
                    timeout=1,
                )
        self.assertEqual(body.read_sizes, [33])
        self.assertEqual(caught.exception.status, 200)
        self.assertNotIn(secret, str(caught.exception))
        self.assertTrue(body.closed)

    def test_urllib_transport_caps_http_error_response_body(self) -> None:
        secret = "oversized-error-secret"
        body = _TrackingBody(secret.encode("utf-8") + (b"x" * 64))
        oversized = HTTPError(
            "https://api.anthropic.com/v1/messages",
            413,
            "Content Too Large",
            {},
            body,
        )
        with (
            patch.object(
                external_providers,
                "HTTP_RESPONSE_BODY_LIMIT_BYTES",
                32,
            ),
            patch.object(external_providers, "build_opener") as build,
        ):
            build.return_value.open.side_effect = oversized
            with self.assertRaises(
                external_providers.HTTPResponseBodyTooLarge
            ) as caught:
                urllib_transport(
                    url="https://api.anthropic.com/v1/messages",
                    body=b"{}",
                    headers={"x-api-key": "request-secret"},
                    timeout=1,
                )
        self.assertEqual(body.read_sizes, [33])
        self.assertEqual(caught.exception.status, 413)
        self.assertNotIn(secret, str(caught.exception))
        self.assertTrue(body.closed)

    def test_redirect_is_an_ordinary_nonretryable_http_error(self) -> None:
        calls: list[int] = []

        def transport(**_: object) -> HTTPResult:
            calls.append(1)
            return HTTPResult(
                307,
                {"location": "https://attacker.example.test"},
                b'{"error":{"message":"redirect refused"}}',
            )

        provider = ExternalDecoderProvider(
            ExternalDecoderProviderConfig(
                provider="anthropic",
                live_execution=True,
                api_key_env="CAPE_LOOP_TEST_ANTHROPIC_KEY",
                max_retries=3,
                max_total_tokens=100_000,
            ),
            transport=transport,
        )
        with patch.dict(
            "os.environ",
            {"CAPE_LOOP_TEST_ANTHROPIC_KEY": "anthropic-test-api-key"},
            clear=True,
        ):
            with self.assertRaises(ExternalDecoderHTTPError) as raised:
                provider.complete(decoder_request())
        self.assertEqual(raised.exception.status, 307)
        self.assertEqual(calls, [1])
        self.assertEqual(provider.budget.request_count, 1)

    def test_retry_after_cannot_exceed_the_declared_backoff_cap(self) -> None:
        calls: list[int] = []
        sleeps: list[float] = []

        def transport(**_: object) -> HTTPResult:
            calls.append(1)
            if len(calls) == 1:
                return HTTPResult(
                    429,
                    {"Retry-After": "3600"},
                    b'{"error":{"message":"retry later"}}',
                )
            return HTTPResult(200, {}, gemini_response())

        provider = ExternalDecoderProvider(
            ExternalDecoderProviderConfig(
                provider="google_gemini",
                live_execution=True,
                api_key_env="CAPE_LOOP_TEST_GEMINI_KEY",
                max_retries=1,
                initial_backoff_seconds=0,
                max_backoff_seconds=2,
                jitter_fraction=0,
                max_total_tokens=100_000,
            ),
            transport=transport,
            sleep=sleeps.append,
        )
        with patch.dict(
            "os.environ",
            {"CAPE_LOOP_TEST_GEMINI_KEY": "google-test-api-key"},
            clear=True,
        ):
            provider.complete(decoder_request())
        self.assertEqual(calls, [1, 1])
        self.assertEqual(sleeps, [2])

    def test_request_budget_blocks_before_a_second_transport(self) -> None:
        calls: list[int] = []

        def transport(**_: object) -> HTTPResult:
            calls.append(1)
            return HTTPResult(200, {}, anthropic_response())

        provider = ExternalDecoderProvider(
            ExternalDecoderProviderConfig(
                provider="anthropic",
                live_execution=True,
                api_key_env="CAPE_LOOP_TEST_ANTHROPIC_KEY",
                max_requests=1,
                max_total_tokens=100_000,
            ),
            transport=transport,
        )
        with patch.dict(
            "os.environ",
            {"CAPE_LOOP_TEST_ANTHROPIC_KEY": "anthropic-test-api-key"},
            clear=True,
        ):
            provider.complete(decoder_request("first"))
            with self.assertRaises(ExternalDecoderBudgetExceeded):
                provider.complete(decoder_request("second"))
        self.assertEqual(calls, [1])

    def test_exhausted_budget_is_checked_before_credential_access(
        self,
    ) -> None:
        provider = ExternalDecoderProvider(
            ExternalDecoderProviderConfig(
                provider="anthropic",
                live_execution=True,
                api_key_env="ABSENT_CAPE_LOOP_DECODER_KEY",
                max_requests=1,
                max_total_tokens=100_000,
            ),
            transport=lambda **_: self.fail(
                "budget exhaustion must precede transport"
            ),
        )
        provider.restore_budget(request_count=1, total_tokens=0)
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ExternalDecoderBudgetExceeded):
                provider.complete(decoder_request())


class ExternalDecoderResumeTests(unittest.TestCase):
    def test_corpus_preflight_rejects_before_output_or_key_access(
        self,
    ) -> None:
        request = decoder_request()
        provider = ExternalDecoderProvider(
            ExternalDecoderProviderConfig(
                provider="anthropic",
                live_execution=True,
                api_key_env="ABSENT_CAPE_LOOP_DECODER_KEY",
                max_retries=1,
                max_requests=1,
                max_total_tokens=6_000_000,
            ),
            transport=lambda **_: self.fail(
                "corpus preflight must precede transport"
            ),
        )
        with TemporaryDirectory() as directory:
            output = Path(directory) / "must-not-exist"
            with patch.dict("os.environ", {}, clear=True):
                with self.assertRaisesRegex(
                    ExternalDecoderBudgetExceeded,
                    "retry expansion",
                ):
                    execute_external_decoder_collection(
                        (provider,),
                        (request,),
                        judgments_path=output / "judgments.jsonl",
                        audit_path=output / "audit.jsonl",
                    )
            self.assertFalse(output.exists())

    def test_audit_precedes_judgment_and_repairs_interrupted_append(
        self,
    ) -> None:
        request = decoder_request()
        calls: list[int] = []

        def transport(**_: object) -> HTTPResult:
            calls.append(1)
            return HTTPResult(200, {}, anthropic_response())

        with TemporaryDirectory() as directory:
            root = Path(directory)
            judgments_path = root / "judgments.jsonl"
            audit_path = root / "audit.jsonl"
            first = ExternalDecoderProvider(
                ExternalDecoderProviderConfig(
                    provider="anthropic",
                    live_execution=True,
                    api_key_env="CAPE_LOOP_TEST_ANTHROPIC_KEY",
                    max_total_tokens=100_000,
                ),
                transport=transport,
            )
            with patch.dict(
                "os.environ",
                {
                    "CAPE_LOOP_TEST_ANTHROPIC_KEY":
                    "anthropic-test-api-key"
                },
                clear=True,
            ):
                summary = execute_external_decoder_collection(
                    (first,),
                    (request,),
                    judgments_path=judgments_path,
                    audit_path=audit_path,
                )
            self.assertEqual(summary.executed_count, 1)
            self.assertTrue(audit_path.exists())
            self.assertEqual(
                len(read_external_decoder_judgments(judgments_path)),
                1,
            )

            # Simulate interruption after the durable audit append but before
            # the import-compatible judgment append.
            judgments_path.unlink()
            second = ExternalDecoderProvider(
                ExternalDecoderProviderConfig(
                    provider="anthropic",
                    api_key_env="CAPE_LOOP_TEST_ANTHROPIC_KEY",
                    max_total_tokens=100_000,
                ),
                transport=lambda **_: self.fail(
                    "resume must not call the provider"
                ),
            )
            with patch.dict("os.environ", {}, clear=True):
                resumed = execute_external_decoder_collection(
                    (second,),
                    (request,),
                    judgments_path=judgments_path,
                    audit_path=audit_path,
                )
            self.assertEqual(calls, [1])
            self.assertEqual(resumed.resumed_count, 1)
            self.assertEqual(resumed.executed_count, 0)
            repaired = read_external_decoder_judgments(judgments_path)
            self.assertEqual(len(repaired), 1)
            self.assertEqual(repaired[0].request_id, request.request_id)

    def test_retry_attempts_are_durable_and_restore_exact_accounting(
        self,
    ) -> None:
        request = decoder_request()
        calls: list[int] = []

        def transport(**_: object) -> HTTPResult:
            calls.append(1)
            if len(calls) == 1:
                return HTTPResult(
                    429,
                    {"Retry-After": "0"},
                    b'{"error":{"message":"retry"}}',
                )
            return HTTPResult(200, {}, gemini_response())

        with TemporaryDirectory() as directory:
            root = Path(directory)
            judgments_path = root / "judgments.jsonl"
            audit_path = root / "audit.jsonl"
            attempt_path = root / "attempts.jsonl"
            first = ExternalDecoderProvider(
                ExternalDecoderProviderConfig(
                    provider="google_gemini",
                    live_execution=True,
                    api_key_env="CAPE_LOOP_TEST_GEMINI_KEY",
                    max_retries=1,
                    initial_backoff_seconds=0,
                    max_backoff_seconds=0,
                    max_total_tokens=100_000,
                ),
                transport=transport,
                sleep=lambda _: None,
            )
            estimate = first.prepare(request).estimated_max_tokens
            with patch.dict(
                "os.environ",
                {"CAPE_LOOP_TEST_GEMINI_KEY": "gemini-test-api-key"},
                clear=True,
            ):
                summary = execute_external_decoder_collection(
                    (first,),
                    (request,),
                    judgments_path=judgments_path,
                    audit_path=audit_path,
                    attempt_path=attempt_path,
                )
            self.assertEqual(calls, [1, 1])
            self.assertEqual(
                summary.transport_attempts_by_provider,
                {"google_gemini": 2},
            )
            self.assertEqual(
                summary.total_tokens_by_provider,
                {"google_gemini": estimate + 110},
            )
            events = [
                json.loads(line)
                for line in attempt_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            attempt_schema = SCHEMAS[
                "external-decoder-transport-attempt"
            ]
            for event in events:
                branch = attempt_schema["oneOf"][
                    0 if event["event"] == "started" else 1
                ]
                self.assertEqual(set(event), set(branch["properties"]))
                self.assertTrue(set(branch["required"]).issubset(event))
            self.assertEqual(
                [(row["event"], row.get("outcome")) for row in events],
                [
                    ("started", None),
                    ("settled", "http_error"),
                    ("started", None),
                    ("settled", "success"),
                ],
            )
            self.assertEqual(events[1]["charged_tokens"], estimate)
            self.assertEqual(events[3]["charged_tokens"], 110)
            final_audit = json.loads(
                audit_path.read_text(encoding="utf-8").strip()
            )
            self.assertEqual(events[3]["provider_audit"], final_audit)

            restored = ExternalDecoderProvider(
                ExternalDecoderProviderConfig(
                    provider="google_gemini",
                    api_key_env="CAPE_LOOP_TEST_GEMINI_KEY",
                    max_retries=1,
                    max_total_tokens=100_000,
                ),
                transport=lambda **_: self.fail(
                    "a completed attempt ledger must resume without transport"
                ),
            )
            with patch.dict("os.environ", {}, clear=True):
                resumed = execute_external_decoder_collection(
                    (restored,),
                    (request,),
                    judgments_path=judgments_path,
                    audit_path=audit_path,
                    attempt_path=attempt_path,
                )
            self.assertEqual(resumed.executed_count, 0)
            self.assertEqual(restored.budget.request_count, 2)
            self.assertEqual(restored.budget.total_tokens, estimate + 110)

    def test_resume_rejects_tampered_attempt_token_accounting(self) -> None:
        request = decoder_request()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            judgments_path = root / "judgments.jsonl"
            audit_path = root / "audit.jsonl"
            attempt_path = root / "attempts.jsonl"
            provider = ExternalDecoderProvider(
                ExternalDecoderProviderConfig(
                    provider="google_gemini",
                    live_execution=True,
                    api_key_env="CAPE_LOOP_TEST_GEMINI_KEY",
                    max_total_tokens=100_000,
                ),
                transport=lambda **_: HTTPResult(
                    200,
                    {},
                    gemini_response(),
                ),
            )
            with patch.dict(
                "os.environ",
                {"CAPE_LOOP_TEST_GEMINI_KEY": "gemini-test-api-key"},
                clear=True,
            ):
                execute_external_decoder_collection(
                    (provider,),
                    (request,),
                    judgments_path=judgments_path,
                    audit_path=audit_path,
                    attempt_path=attempt_path,
                )

            events = [
                json.loads(line)
                for line in attempt_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            events[-1]["charged_tokens"] = 0
            attempt_path.write_text(
                "".join(
                    json.dumps(row, sort_keys=True) + "\n"
                    for row in events
                ),
                encoding="utf-8",
            )
            resumed = ExternalDecoderProvider(
                ExternalDecoderProviderConfig(
                    provider="google_gemini",
                    api_key_env="CAPE_LOOP_TEST_GEMINI_KEY",
                    max_total_tokens=100_000,
                ),
                transport=lambda **_: self.fail(
                    "tampered accounting must fail before transport"
                ),
            )
            with patch.dict("os.environ", {}, clear=True):
                with self.assertRaisesRegex(
                    ValueError,
                    "charged token count",
                ):
                    execute_external_decoder_collection(
                        (resumed,),
                        (request,),
                        judgments_path=judgments_path,
                        audit_path=audit_path,
                        attempt_path=attempt_path,
                    )

    def test_resume_rejects_tampered_final_response_metadata(self) -> None:
        request = decoder_request()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            judgments_path = root / "judgments.jsonl"
            audit_path = root / "audit.jsonl"
            attempt_path = root / "attempts.jsonl"
            provider = ExternalDecoderProvider(
                ExternalDecoderProviderConfig(
                    provider="google_gemini",
                    live_execution=True,
                    api_key_env="CAPE_LOOP_TEST_GEMINI_KEY",
                    max_total_tokens=100_000,
                ),
                transport=lambda **_: HTTPResult(
                    200,
                    {"X-Request-Id": "google-server-request"},
                    gemini_response(),
                ),
            )
            with patch.dict(
                "os.environ",
                {"CAPE_LOOP_TEST_GEMINI_KEY": "gemini-test-api-key"},
                clear=True,
            ):
                execute_external_decoder_collection(
                    (provider,),
                    (request,),
                    judgments_path=judgments_path,
                    audit_path=audit_path,
                    attempt_path=attempt_path,
                )
            original = [
                json.loads(line)
                for line in attempt_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            mutations = (
                ("http_status", 500),
                ("response_body_sha256", "f" * 64),
                ("response_body_sha256", None),
                ("server_request_id", "different-server-request"),
            )
            for field, value in mutations:
                with self.subTest(field=field):
                    events = [dict(row) for row in original]
                    events[-1][field] = value
                    attempt_path.write_text(
                        "".join(
                            json.dumps(row, sort_keys=True) + "\n"
                            for row in events
                        ),
                        encoding="utf-8",
                    )
                    resumed = ExternalDecoderProvider(
                        ExternalDecoderProviderConfig(
                            provider="google_gemini",
                            api_key_env="CAPE_LOOP_TEST_GEMINI_KEY",
                            max_total_tokens=100_000,
                        ),
                        transport=lambda **_: self.fail(
                            "tampered metadata must fail before transport"
                        ),
                    )
                    with patch.dict("os.environ", {}, clear=True):
                        with self.assertRaisesRegex(
                            ValueError,
                            "response metadata",
                        ):
                            execute_external_decoder_collection(
                                (resumed,),
                                (request,),
                                judgments_path=judgments_path,
                                audit_path=audit_path,
                                attempt_path=attempt_path,
                            )

    def test_unresolved_started_attempt_blocks_automatic_resume(
        self,
    ) -> None:
        request = decoder_request()
        second_calls: list[int] = []

        with TemporaryDirectory() as directory:
            root = Path(directory)
            judgments_path = root / "judgments.jsonl"
            audit_path = root / "audit.jsonl"
            attempt_path = root / "attempts.jsonl"
            crashing = ExternalDecoderProvider(
                ExternalDecoderProviderConfig(
                    provider="anthropic",
                    live_execution=True,
                    api_key_env="CAPE_LOOP_TEST_ANTHROPIC_KEY",
                    max_total_tokens=100_000,
                ),
                transport=lambda **_: (_ for _ in ()).throw(
                    RuntimeError("simulated process failure")
                ),
            )
            with patch.dict(
                "os.environ",
                {
                    "CAPE_LOOP_TEST_ANTHROPIC_KEY":
                    "anthropic-test-api-key"
                },
                clear=True,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "simulated process failure",
                ):
                    execute_external_decoder_collection(
                        (crashing,),
                        (request,),
                        judgments_path=judgments_path,
                        audit_path=audit_path,
                        attempt_path=attempt_path,
                    )
            events = attempt_path.read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(len(events), 1)
            self.assertEqual(json.loads(events[0])["event"], "started")

            retry = ExternalDecoderProvider(
                ExternalDecoderProviderConfig(
                    provider="anthropic",
                    live_execution=True,
                    api_key_env="CAPE_LOOP_TEST_ANTHROPIC_KEY",
                    max_total_tokens=100_000,
                ),
                transport=lambda **_: (
                    second_calls.append(1)
                    or HTTPResult(200, {}, anthropic_response())
                ),
            )
            with patch.dict("os.environ", {}, clear=True):
                with self.assertRaisesRegex(
                    ValueError,
                    "manual review",
                ):
                    execute_external_decoder_collection(
                        (retry,),
                        (request,),
                        judgments_path=judgments_path,
                        audit_path=audit_path,
                        attempt_path=attempt_path,
                    )
            self.assertEqual(second_calls, [])

    def test_usage_over_reservation_stops_with_unresolved_attempt(
        self,
    ) -> None:
        request = decoder_request()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            judgments_path = root / "judgments.jsonl"
            audit_path = root / "audit.jsonl"
            attempt_path = root / "attempts.jsonl"
            config = ExternalDecoderProviderConfig(
                provider="google_gemini",
                live_execution=True,
                api_key_env="CAPE_LOOP_TEST_GEMINI_KEY",
                max_total_tokens=100_000,
            )
            provider = ExternalDecoderProvider(
                config,
                transport=lambda **_: HTTPResult(
                    200,
                    {},
                    json.dumps(
                        {
                            **json.loads(gemini_response()),
                            "usageMetadata": {
                                "totalTokenCount": (
                                    provider.prepare(
                                        request
                                    ).estimated_max_tokens
                                    + 1
                                )
                            },
                        }
                    ).encode("utf-8"),
                ),
            )
            with patch.dict(
                "os.environ",
                {"CAPE_LOOP_TEST_GEMINI_KEY": "gemini-test-api-key"},
                clear=True,
            ):
                with self.assertRaisesRegex(
                    ExternalDecoderResponseError,
                    "exceeds the conservative reservation",
                ):
                    execute_external_decoder_collection(
                        (provider,),
                        (request,),
                        judgments_path=judgments_path,
                        audit_path=audit_path,
                        attempt_path=attempt_path,
                    )
            events = [
                json.loads(line)
                for line in attempt_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertEqual(
                [event["event"] for event in events],
                ["started"],
            )

            resumed = ExternalDecoderProvider(
                config,
                transport=lambda **_: self.fail(
                    "an over-reservation response must not be retried"
                ),
            )
            with patch.dict("os.environ", {}, clear=True):
                with self.assertRaisesRegex(ValueError, "manual review"):
                    execute_external_decoder_collection(
                        (resumed,),
                        (request,),
                        judgments_path=judgments_path,
                        audit_path=audit_path,
                        attempt_path=attempt_path,
                    )

    def test_settled_failed_attempt_blocks_automatic_resume(self) -> None:
        request = decoder_request()
        retry_calls: list[int] = []

        with TemporaryDirectory() as directory:
            root = Path(directory)
            judgments_path = root / "judgments.jsonl"
            audit_path = root / "audit.jsonl"
            attempt_path = root / "attempts.jsonl"
            failed = ExternalDecoderProvider(
                ExternalDecoderProviderConfig(
                    provider="anthropic",
                    live_execution=True,
                    api_key_env="CAPE_LOOP_TEST_ANTHROPIC_KEY",
                    max_retries=0,
                    max_total_tokens=100_000,
                ),
                transport=lambda **_: HTTPResult(
                    400,
                    {},
                    b'{"error":{"message":"invalid request"}}',
                ),
            )
            with patch.dict(
                "os.environ",
                {
                    "CAPE_LOOP_TEST_ANTHROPIC_KEY":
                    "anthropic-test-api-key"
                },
                clear=True,
            ):
                with self.assertRaises(ExternalDecoderHTTPError):
                    execute_external_decoder_collection(
                        (failed,),
                        (request,),
                        judgments_path=judgments_path,
                        audit_path=audit_path,
                        attempt_path=attempt_path,
                    )

            retry = ExternalDecoderProvider(
                ExternalDecoderProviderConfig(
                    provider="anthropic",
                    live_execution=True,
                    api_key_env="CAPE_LOOP_TEST_ANTHROPIC_KEY",
                    max_total_tokens=100_000,
                ),
                transport=lambda **_: (
                    retry_calls.append(1)
                    or HTTPResult(200, {}, anthropic_response())
                ),
            )
            with patch.dict("os.environ", {}, clear=True):
                with self.assertRaisesRegex(
                    ValueError,
                    "manual review",
                ):
                    execute_external_decoder_collection(
                        (retry,),
                        (request,),
                        judgments_path=judgments_path,
                        audit_path=audit_path,
                        attempt_path=attempt_path,
                    )
            self.assertEqual(retry_calls, [])

    def test_all_provider_credentials_are_preflighted_before_transport(
        self,
    ) -> None:
        request = decoder_request()
        calls: list[str] = []
        providers = (
            ExternalDecoderProvider(
                ExternalDecoderProviderConfig(
                    provider="anthropic",
                    live_execution=True,
                    api_key_env="CAPE_LOOP_TEST_ANTHROPIC_KEY",
                    max_total_tokens=100_000,
                ),
                transport=lambda **_: (
                    calls.append("anthropic")
                    or HTTPResult(200, {}, anthropic_response())
                ),
            ),
            ExternalDecoderProvider(
                ExternalDecoderProviderConfig(
                    provider="google_gemini",
                    live_execution=True,
                    api_key_env="CAPE_LOOP_TEST_GEMINI_KEY",
                    max_total_tokens=100_000,
                ),
                transport=lambda **_: (
                    calls.append("google_gemini")
                    or HTTPResult(200, {}, gemini_response())
                ),
            ),
        )
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(
                "os.environ",
                {
                    "CAPE_LOOP_TEST_ANTHROPIC_KEY":
                    "anthropic-test-api-key"
                },
                clear=True,
            ):
                with self.assertRaises(MissingExternalDecoderAPIKey):
                    execute_external_decoder_collection(
                        providers,
                        (request,),
                        judgments_path=root / "judgments.jsonl",
                        audit_path=root / "audit.jsonl",
                    )
        self.assertEqual(calls, [])

    def test_rejected_identity_audit_is_embedded_and_parseable(self) -> None:
        request = decoder_request()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            judgments_path = root / "judgments.jsonl"
            audit_path = root / "audit.jsonl"
            attempt_path = root / "attempts.jsonl"
            provider = ExternalDecoderProvider(
                ExternalDecoderProviderConfig(
                    provider="anthropic",
                    live_execution=True,
                    api_key_env="CAPE_LOOP_TEST_ANTHROPIC_KEY",
                    max_total_tokens=100_000,
                ),
                transport=lambda **_: HTTPResult(
                    200,
                    {},
                    anthropic_response(model="claude-opus-5"),
                ),
            )
            with patch.dict(
                "os.environ",
                {
                    "CAPE_LOOP_TEST_ANTHROPIC_KEY":
                    "anthropic-test-api-key"
                },
                clear=True,
            ):
                with self.assertRaises(ExternalDecoderIdentityMismatch):
                    execute_external_decoder_collection(
                        (provider,),
                        (request,),
                        judgments_path=judgments_path,
                        audit_path=audit_path,
                        attempt_path=attempt_path,
                    )
            audit = json.loads(
                audit_path.read_text(encoding="utf-8").strip()
            )
            events = [
                json.loads(line)
                for line in attempt_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            settlement = events[-1]
            self.assertEqual(settlement["outcome"], "identity_mismatch")
            self.assertEqual(settlement["provider_audit"], audit)
            self.assertEqual(
                audit["acceptance_status"],
                "rejected_identity_mismatch",
            )
            parsed = ExternalDecoderJudgment.parse(audit["judgment"])
            self.assertEqual(parsed.request_id, request.request_id)
            self.assertFalse(judgments_path.exists())

    def test_only_a_truncated_final_jsonl_tail_is_repaired(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            path.write_bytes(b'{"complete":true}\n{"partial"')
            self.assertTrue(
                external_providers._repair_trailing_jsonl(path)
            )
            self.assertEqual(
                path.read_bytes(),
                b'{"complete":true}\n',
            )

            path.write_bytes(b'{"complete":true}\n{broken}\n{"partial"')
            self.assertTrue(
                external_providers._repair_trailing_jsonl(path)
            )
            self.assertEqual(
                path.read_bytes(),
                b'{"complete":true}\n{broken}\n',
            )
            with self.assertRaises(json.JSONDecodeError):
                [
                    json.loads(line)
                    for line in path.read_text(
                        encoding="utf-8"
                    ).splitlines()
                ]

    def test_collection_lock_fails_fast_and_is_released(self) -> None:
        request = decoder_request()
        calls: list[int] = []
        with TemporaryDirectory() as directory:
            root = Path(directory)
            judgments_path = root / "judgments.jsonl"
            audit_path = root / "audit.jsonl"
            lock_path = root / ".external-decoder-collection.lock"
            provider = ExternalDecoderProvider(
                ExternalDecoderProviderConfig(
                    provider="anthropic",
                    live_execution=True,
                    api_key_env="CAPE_LOOP_TEST_ANTHROPIC_KEY",
                    max_total_tokens=100_000,
                ),
                transport=lambda **_: (
                    calls.append(1)
                    or HTTPResult(200, {}, anthropic_response())
                ),
            )
            with external_providers._ExclusiveCollectionLock(lock_path):
                with self.assertRaises(ExternalDecoderExecutionLocked):
                    execute_external_decoder_collection(
                        (provider,),
                        (request,),
                        judgments_path=judgments_path,
                        audit_path=audit_path,
                    )
            with patch.dict(
                "os.environ",
                {
                    "CAPE_LOOP_TEST_ANTHROPIC_KEY":
                    "anthropic-test-api-key"
                },
                clear=True,
            ):
                summary = execute_external_decoder_collection(
                    (provider,),
                    (request,),
                    judgments_path=judgments_path,
                    audit_path=audit_path,
                )
            self.assertEqual(summary.executed_count, 1)
            self.assertEqual(calls, [1])


if __name__ == "__main__":
    unittest.main()
