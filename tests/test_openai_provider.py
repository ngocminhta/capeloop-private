from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Mapping
from unittest.mock import patch
import json
import unittest

from cape_loop.llm_exchange import (
    LLMRequest,
    ReplayProvider,
    read_responses,
    write_requests,
)
from cape_loop.openai_provider import (
    DEFAULT_OPENAI_MODEL_ROLES,
    BudgetExceeded,
    HTTPResult,
    LiveExecutionRequired,
    MissingAPIKey,
    OpenAIProviderConfig,
    OpenAIResponsesProvider,
    ProviderHTTPError,
    ProviderModelMismatch,
    ProviderResponseError,
    ResumableOpenAICompletionProvider,
    execute_jsonl,
    parse_llm_request,
    prepare_openai_request,
    returned_model_is_consistent,
)


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
    response_id: str = "resp_test",
    *,
    model: str = "gpt-5.6-sol-2026-07-01",
    direct_output_text: bool = False,
) -> bytes:
    structured = json.dumps({"beliefs": BELIEFS}, separators=(",", ":"))
    raw = {
        "id": response_id,
        "object": "response",
        "created_at": 1_700_000_001,
        "status": "completed",
        "model": model,
        "usage": {
            "input_tokens": 40,
            "output_tokens": 35,
            "total_tokens": 75,
        },
    }
    if direct_output_text:
        raw["output_text"] = structured
    else:
        raw["output"] = [
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": structured,
                    }
                ],
            }
        ]
    return json.dumps(raw).encode("utf-8")


class OpenAIRequestTests(unittest.TestCase):
    def test_custom_base_url_requires_a_separate_explicit_opt_in(self) -> None:
        with self.assertRaisesRegex(ValueError, "HTTPS"):
            OpenAIProviderConfig(
                base_url="https://user@api.openai.com"
            )
        with self.assertRaisesRegex(ValueError, "official"):
            OpenAIProviderConfig(base_url="https://proxy.example.test")
        with self.assertRaisesRegex(ValueError, "dedicated credential"):
            OpenAIProviderConfig(
                base_url="https://proxy.example.test",
                allow_custom_base_url=True,
            )
        configured = OpenAIProviderConfig(
            base_url="https://proxy.example.test",
            allow_custom_base_url=True,
            api_key_env="CAPE_LOOP_PROXY_KEY",
        )
        self.assertEqual(
            configured.endpoint,
            "https://proxy.example.test/v1/responses",
        )

    def test_default_model_roles_are_deliberate_and_data_backed(self) -> None:
        self.assertEqual(
            DEFAULT_OPENAI_MODEL_ROLES["primary"].model,
            "gpt-5.6-sol",
        )
        self.assertEqual(
            DEFAULT_OPENAI_MODEL_ROLES["replication"].model,
            "gpt-5.6-terra",
        )
        self.assertEqual(
            DEFAULT_OPENAI_MODEL_ROLES["decoder"].model,
            "gpt-5.6-luna",
        )
        suite_path = (
            Path(__file__).resolve().parents[1]
            / "data"
            / "model-suites"
            / "openai-gpt-5.6.json"
        )
        suite = json.loads(suite_path.read_text(encoding="utf-8"))
        self.assertEqual(
            {record["role"]: record["model"] for record in suite["roles"]},
            {
                key: role.model
                for key, role in DEFAULT_OPENAI_MODEL_ROLES.items()
            },
        )

    def test_dry_run_uses_responses_structured_outputs_without_key(self) -> None:
        request = build_request()
        config = OpenAIProviderConfig(api_key_env="ABSENT_CAPE_LOOP_TEST_KEY")
        with patch.dict("os.environ", {}, clear=True):
            prepared = prepare_openai_request(request, config)
        self.assertEqual(
            prepared.endpoint,
            "https://api.openai.com/v1/responses",
        )
        self.assertEqual(prepared.body["model"], "gpt-5.6-sol")
        self.assertFalse(prepared.body["store"])
        output_format = prepared.body["text"]["format"]
        self.assertEqual(output_format["type"], "json_schema")
        self.assertTrue(output_format["strict"])
        self.assertFalse(output_format["schema"]["additionalProperties"])
        self.assertEqual(
            output_format["schema"]["required"],
            ["beliefs"],
        )
        self.assertNotIn("Authorization", prepared.headers)
        self.assertIn("Idempotency-Key", prepared.headers)
        self.assertIn("X-Client-Request-Id", prepared.headers)
        self.assertEqual(
            prepared.idempotency_key,
            prepare_openai_request(request, config).idempotency_key,
        )

    def test_model_consistency_allows_only_the_requested_alias_snapshot(self) -> None:
        self.assertTrue(
            returned_model_is_consistent(
                "gpt-5.6-sol",
                "gpt-5.6-sol-2026-07-01",
            )
        )
        self.assertFalse(
            returned_model_is_consistent(
                "gpt-5.6-sol",
                "gpt-5.6-terra",
            )
        )
        self.assertFalse(
            returned_model_is_consistent(
                "gpt-5.6-sol-2026-07-01",
                "gpt-5.6-sol-2026-07-01-2026-08-01",
            )
        )

    def test_request_parser_recomputes_prompt_binding(self) -> None:
        request = build_request()
        self.assertEqual(
            parse_llm_request(request.to_dict()),
            request,
        )
        changed = json.loads(json.dumps(request.to_dict()))
        changed["payload"]["observation"]["selected_option_id"] = "changed"
        with self.assertRaisesRegex(ValueError, "does not bind"):
            parse_llm_request(changed)

    def test_live_execution_and_key_are_both_explicit(self) -> None:
        request = build_request()
        disabled = OpenAIResponsesProvider(OpenAIProviderConfig())
        with self.assertRaises(LiveExecutionRequired):
            disabled.complete(request)

        enabled = OpenAIResponsesProvider(
            OpenAIProviderConfig(
                live_execution=True,
                api_key_env="ABSENT_CAPE_LOOP_TEST_KEY",
            )
        )
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(MissingAPIKey):
                enabled.complete(request)


class OpenAITransportTests(unittest.TestCase):
    def test_success_parses_nested_output_and_captures_safe_metadata(self) -> None:
        seen: list[dict[str, object]] = []

        def transport(**kwargs: object) -> HTTPResult:
            seen.append(dict(kwargs))
            return HTTPResult(
                status=200,
                headers={
                    "X-Request-Id": "req_server_123",
                    "OpenAI-Processing-Ms": "42",
                },
                body=response_body(),
            )

        config = OpenAIProviderConfig(
            live_execution=True,
            api_key_env="CAPE_LOOP_TEST_OPENAI_KEY",
            max_total_tokens=100_000,
        )
        provider = OpenAIResponsesProvider(
            config,
            transport=transport,
            epoch_time=lambda: 1_700_000_000.0,
        )
        with patch.dict(
            "os.environ",
            {"CAPE_LOOP_TEST_OPENAI_KEY": "sk-test-never-retain"},
            clear=True,
        ):
            result = provider.complete(build_request())

        self.assertEqual(result.response.beliefs, BELIEFS)
        self.assertEqual(result.response.model_id, "gpt-5.6-sol-2026-07-01")
        self.assertEqual(result.provider_response_id, "resp_test")
        self.assertEqual(result.server_request_id, "req_server_123")
        self.assertEqual(result.processing_ms, "42")
        self.assertEqual(result.attempts, 1)
        self.assertEqual(provider.budget.request_count, 1)
        self.assertEqual(provider.budget.total_tokens, 75)
        self.assertEqual(
            seen[0]["headers"]["Authorization"],  # type: ignore[index]
            "Bearer sk-test-never-retain",
        )
        retained = json.dumps(result.to_audit_record())
        self.assertNotIn("sk-test-never-retain", retained)
        self.assertIn('"raw_response"', retained)

    def test_direct_output_text_is_supported(self) -> None:
        provider = OpenAIResponsesProvider(
            OpenAIProviderConfig(
                live_execution=True,
                api_key_env="CAPE_LOOP_TEST_OPENAI_KEY",
                max_total_tokens=100_000,
            ),
            transport=lambda **_: HTTPResult(
                status=200,
                headers={},
                body=response_body(direct_output_text=True),
            ),
        )
        with patch.dict(
            "os.environ",
            {"CAPE_LOOP_TEST_OPENAI_KEY": "test"},
            clear=True,
        ):
            result = provider.complete(build_request())
        self.assertEqual(result.response.beliefs, BELIEFS)

    def test_retry_after_backoff_and_idempotency_are_stable(self) -> None:
        outcomes = [
            HTTPResult(
                429,
                {"Retry-After": "3", "X-Request-Id": "rate-limited"},
                b'{"error":{"message":"slow down"}}',
            ),
            HTTPResult(
                503,
                {},
                b'{"error":{"message":"temporarily unavailable"}}',
            ),
            HTTPResult(200, {"X-Request-Id": "success"}, response_body()),
        ]
        seen_headers: list[Mapping[str, str]] = []
        sleeps: list[float] = []

        def transport(**kwargs: object) -> HTTPResult:
            seen_headers.append(dict(kwargs["headers"]))  # type: ignore[arg-type]
            return outcomes.pop(0)

        provider = OpenAIResponsesProvider(
            OpenAIProviderConfig(
                live_execution=True,
                api_key_env="CAPE_LOOP_TEST_OPENAI_KEY",
                max_retries=2,
                initial_backoff_seconds=1,
                max_backoff_seconds=4,
                jitter_fraction=0,
                max_total_tokens=100_000,
            ),
            transport=transport,
            sleep=sleeps.append,
            random_value=lambda: 0.5,
            epoch_time=lambda: 1_700_000_000.0,
        )
        with patch.dict(
            "os.environ",
            {"CAPE_LOOP_TEST_OPENAI_KEY": "test"},
            clear=True,
        ):
            result = provider.complete(build_request())
        self.assertEqual(result.attempts, 3)
        self.assertEqual(sleeps, [3.0, 2.0])
        self.assertEqual(
            {headers["Idempotency-Key"] for headers in seen_headers},
            {seen_headers[0]["Idempotency-Key"]},
        )
        self.assertEqual(
            {headers["X-Client-Request-Id"] for headers in seen_headers},
            {seen_headers[0]["X-Client-Request-Id"]},
        )

    def test_nontransient_error_redacts_a_key_even_if_echoed(self) -> None:
        secret = "sk-extremely-secret"
        provider = OpenAIResponsesProvider(
            OpenAIProviderConfig(
                live_execution=True,
                api_key_env="CAPE_LOOP_TEST_OPENAI_KEY",
                max_total_tokens=100_000,
            ),
            transport=lambda **_: HTTPResult(
                status=400,
                headers={"X-Request-Id": "bad-request"},
                body=json.dumps(
                    {"error": {"message": f"invalid credential {secret}"}}
                ).encode(),
            ),
        )
        with patch.dict(
            "os.environ",
            {"CAPE_LOOP_TEST_OPENAI_KEY": secret},
            clear=True,
        ):
            with self.assertRaises(ProviderHTTPError) as raised:
                provider.complete(build_request())
        self.assertNotIn(secret, str(raised.exception))
        self.assertIn("[redacted]", str(raised.exception))
        self.assertEqual(provider.budget.request_count, 0)

    def test_refusal_is_not_mistaken_for_structured_output(self) -> None:
        refusal = {
            "id": "resp_refusal",
            "status": "completed",
            "model": "gpt-5.6-sol",
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "refusal",
                            "refusal": "I cannot provide this.",
                        }
                    ],
                }
            ],
            "usage": {"total_tokens": 10},
        }
        provider = OpenAIResponsesProvider(
            OpenAIProviderConfig(
                live_execution=True,
                api_key_env="CAPE_LOOP_TEST_OPENAI_KEY",
                max_total_tokens=100_000,
            ),
            transport=lambda **_: HTTPResult(
                200,
                {},
                json.dumps(refusal).encode(),
            ),
        )
        with patch.dict(
            "os.environ",
            {"CAPE_LOOP_TEST_OPENAI_KEY": "test"},
            clear=True,
        ):
            with self.assertRaisesRegex(ProviderResponseError, "refused"):
                provider.complete(build_request())


class OpenAIBudgetAndResumeTests(unittest.TestCase):
    def test_static_model_mismatch_is_rejected_and_audited(self) -> None:
        request = build_request()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            requests_path = root / "requests.jsonl"
            responses_path = root / "responses.jsonl"
            audit_path = root / "audit.jsonl"
            write_requests(requests_path, (request,))
            provider = OpenAIResponsesProvider(
                OpenAIProviderConfig(
                    live_execution=True,
                    api_key_env="CAPE_LOOP_TEST_OPENAI_KEY",
                    model="gpt-5.6-sol",
                    max_total_tokens=100_000,
                ),
                transport=lambda **_: HTTPResult(
                    200,
                    {},
                    response_body(model="gpt-5.6-terra"),
                ),
            )
            with patch.dict(
                "os.environ",
                {"CAPE_LOOP_TEST_OPENAI_KEY": "test"},
                clear=True,
            ):
                with self.assertRaises(ProviderModelMismatch):
                    execute_jsonl(
                        provider,
                        requests_path,
                        responses_path=responses_path,
                        audit_path=audit_path,
                    )
            self.assertFalse(responses_path.exists())
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            self.assertEqual(
                audit["acceptance_status"],
                "rejected_model_mismatch",
            )
            self.assertEqual(audit["model_requested"], "gpt-5.6-sol")
            self.assertEqual(audit["model_returned"], "gpt-5.6-terra")
            self.assertEqual(provider.budget.request_count, 1)

    def test_adaptive_model_mismatch_is_rejected_and_audited(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            calls: list[int] = []

            def mismatch_transport(**_: object) -> HTTPResult:
                calls.append(1)
                return HTTPResult(
                    200,
                    {},
                    response_body(model="gpt-5.6-luna"),
                )

            provider = OpenAIResponsesProvider(
                OpenAIProviderConfig(
                    live_execution=True,
                    api_key_env="CAPE_LOOP_TEST_OPENAI_KEY",
                    model="gpt-5.6-sol",
                    max_total_tokens=100_000,
                ),
                transport=mismatch_transport,
            )
            adapter = ResumableOpenAICompletionProvider(
                provider,
                responses_path=root / "responses.jsonl",
                audit_path=root / "audit.jsonl",
            )
            with patch.dict(
                "os.environ",
                {"CAPE_LOOP_TEST_OPENAI_KEY": "test"},
                clear=True,
            ):
                with self.assertRaises(ProviderModelMismatch):
                    adapter.complete(build_request())
                with self.assertRaisesRegex(
                    ValueError,
                    "manual review",
                ):
                    adapter.complete(build_request())
            self.assertEqual(calls, [1])
            self.assertFalse((root / "responses.jsonl").exists())
            audit = json.loads(
                (root / "audit.jsonl").read_text(encoding="utf-8")
            )
            self.assertEqual(
                audit["acceptance_status"],
                "rejected_model_mismatch",
            )
            with self.assertRaisesRegex(ValueError, "rejected model mismatch"):
                ResumableOpenAICompletionProvider(
                    OpenAIResponsesProvider(
                        OpenAIProviderConfig(
                            model="gpt-5.6-sol",
                            max_total_tokens=100_000,
                        )
                    ),
                    responses_path=root / "responses.jsonl",
                    audit_path=root / "audit.jsonl",
                )

    def test_request_and_token_limits_fail_before_transport(self) -> None:
        calls: list[int] = []

        def transport(**_: object) -> HTTPResult:
            calls.append(1)
            return HTTPResult(200, {}, response_body())

        provider = OpenAIResponsesProvider(
            OpenAIProviderConfig(
                live_execution=True,
                api_key_env="CAPE_LOOP_TEST_OPENAI_KEY",
                max_requests=1,
                max_total_tokens=100_000,
            ),
            transport=transport,
        )
        with patch.dict(
            "os.environ",
            {"CAPE_LOOP_TEST_OPENAI_KEY": "test"},
            clear=True,
        ):
            provider.complete(build_request("first"))
            with self.assertRaisesRegex(BudgetExceeded, "max_requests"):
                provider.complete(build_request("second"))
        self.assertEqual(len(calls), 1)

        token_limited = OpenAIResponsesProvider(
            OpenAIProviderConfig(
                live_execution=True,
                api_key_env="CAPE_LOOP_TEST_OPENAI_KEY",
                max_output_tokens=100,
                max_total_tokens=100,
            ),
            transport=transport,
        )
        with patch.dict(
            "os.environ",
            {"CAPE_LOOP_TEST_OPENAI_KEY": "test"},
            clear=True,
        ):
            with self.assertRaisesRegex(BudgetExceeded, "token reservation"):
                token_limited.complete(build_request())
        self.assertEqual(len(calls), 1)

    def test_jsonl_executor_resumes_and_outputs_replay_compatible_rows(self) -> None:
        requests = (build_request("first"), build_request("second"))
        with TemporaryDirectory() as directory:
            root = Path(directory)
            requests_path = root / "requests.jsonl"
            responses_path = root / "responses.jsonl"
            audit_path = root / "openai-audit.jsonl"
            write_requests(requests_path, requests)

            first_calls: list[int] = []

            def first_transport(**_: object) -> HTTPResult:
                first_calls.append(1)
                return HTTPResult(
                    200,
                    {"X-Request-Id": "first-live"},
                    response_body("resp_first"),
                )

            first_provider = OpenAIResponsesProvider(
                OpenAIProviderConfig(
                    live_execution=True,
                    api_key_env="CAPE_LOOP_TEST_OPENAI_KEY",
                    max_requests=1,
                    max_total_tokens=100_000,
                ),
                transport=first_transport,
            )
            with patch.dict(
                "os.environ",
                {"CAPE_LOOP_TEST_OPENAI_KEY": "sk-not-retained"},
                clear=True,
            ):
                with self.assertRaises(BudgetExceeded):
                    execute_jsonl(
                        first_provider,
                        requests_path,
                        responses_path=responses_path,
                        audit_path=audit_path,
                    )
            self.assertEqual(first_calls, [1])
            self.assertEqual(len(read_responses(responses_path)), 1)

            second_calls: list[int] = []

            def second_transport(**_: object) -> HTTPResult:
                second_calls.append(1)
                return HTTPResult(
                    200,
                    {"X-Request-Id": "second-live"},
                    response_body("resp_second"),
                )

            second_provider = OpenAIResponsesProvider(
                OpenAIProviderConfig(
                    live_execution=True,
                    api_key_env="CAPE_LOOP_TEST_OPENAI_KEY",
                    max_requests=2,
                    max_total_tokens=100_000,
                ),
                transport=second_transport,
            )
            with patch.dict(
                "os.environ",
                {"CAPE_LOOP_TEST_OPENAI_KEY": "sk-not-retained"},
                clear=True,
            ):
                summary = execute_jsonl(
                    second_provider,
                    requests_path,
                    responses_path=responses_path,
                    audit_path=audit_path,
                )
            self.assertEqual(second_calls, [1])
            self.assertEqual(summary.resumed_count, 1)
            self.assertEqual(summary.executed_count, 1)
            responses = read_responses(responses_path)
            self.assertEqual(len(responses), 2)
            ReplayProvider(responses).validate_coverage(requests)
            self.assertNotIn(
                "sk-not-retained",
                audit_path.read_text(encoding="utf-8"),
            )
            self.assertEqual(
                len(audit_path.read_text(encoding="utf-8").splitlines()),
                2,
            )

    def test_audit_journal_reconciles_a_missing_replay_append(self) -> None:
        request = build_request()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            request_path = root / "request.jsonl"
            responses_path = root / "responses.jsonl"
            audit_path = root / "audit.jsonl"
            write_requests(request_path, (request,))
            calls: list[int] = []

            def transport(**_: object) -> HTTPResult:
                calls.append(1)
                return HTTPResult(200, {}, response_body())

            first = OpenAIResponsesProvider(
                OpenAIProviderConfig(
                    live_execution=True,
                    api_key_env="CAPE_LOOP_TEST_OPENAI_KEY",
                    max_total_tokens=100_000,
                ),
                transport=transport,
            )
            with patch.dict(
                "os.environ",
                {"CAPE_LOOP_TEST_OPENAI_KEY": "test"},
                clear=True,
            ):
                execute_jsonl(
                    first,
                    request_path,
                    responses_path=responses_path,
                    audit_path=audit_path,
                )
            responses_path.unlink()

            second = OpenAIResponsesProvider(
                OpenAIProviderConfig(
                    live_execution=True,
                    api_key_env="CAPE_LOOP_TEST_OPENAI_KEY",
                    max_total_tokens=100_000,
                ),
                transport=transport,
            )
            with patch.dict(
                "os.environ",
                {"CAPE_LOOP_TEST_OPENAI_KEY": "test"},
                clear=True,
            ):
                summary = execute_jsonl(
                    second,
                    request_path,
                    responses_path=responses_path,
                    audit_path=audit_path,
                )
            self.assertEqual(calls, [1])
            self.assertEqual(summary.resumed_count, 1)
            ReplayProvider(read_responses(responses_path)).validate_coverage(
                (request,)
            )

    def test_resume_rejects_a_different_model_configuration(self) -> None:
        request = build_request()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            requests_path = root / "requests.jsonl"
            responses_path = root / "responses.jsonl"
            audit_path = root / "audit.jsonl"
            write_requests(requests_path, (request,))
            first = OpenAIResponsesProvider(
                OpenAIProviderConfig(
                    live_execution=True,
                    api_key_env="CAPE_LOOP_TEST_OPENAI_KEY",
                    model="gpt-5.6-sol",
                    max_total_tokens=100_000,
                ),
                transport=lambda **_: HTTPResult(
                    200, {}, response_body(model="gpt-5.6-sol")
                ),
            )
            with patch.dict(
                "os.environ",
                {"CAPE_LOOP_TEST_OPENAI_KEY": "test"},
                clear=True,
            ):
                execute_jsonl(
                    first,
                    requests_path,
                    responses_path=responses_path,
                    audit_path=audit_path,
                )
            second_calls: list[int] = []
            second = OpenAIResponsesProvider(
                OpenAIProviderConfig(
                    live_execution=True,
                    api_key_env="CAPE_LOOP_TEST_OPENAI_KEY",
                    model="gpt-5.6-terra",
                    max_total_tokens=100_000,
                ),
                transport=lambda **_: second_calls.append(1),  # type: ignore[arg-type]
            )
            with self.assertRaisesRegex(
                ValueError, "does not match the current request"
            ):
                execute_jsonl(
                    second,
                    requests_path,
                    responses_path=responses_path,
                    audit_path=audit_path,
                )
            self.assertEqual(second_calls, [])


if __name__ == "__main__":
    unittest.main()
