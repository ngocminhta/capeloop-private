from __future__ import annotations

from pathlib import Path
from shutil import copytree
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json
import os
import unittest

from cape_loop import native_action_provider as native_action_module
from cape_loop.artifacts import verify_run
from cape_loop.config import (
    AppConfig,
    ExperimentSection,
    InferenceSection,
    RunSection,
)
from cape_loop.file_lock import try_file_lock, unlock_file
from cape_loop.gate_review import read_native_terminal_action_records
from cape_loop.native_action_provider import (
    NATIVE_ACTION_SYSTEM_ID,
    NativeActionManualReviewRequired,
    OpenAINativeActionProvider,
    build_native_action_requests,
    execute_openai_native_actions,
    plan_openai_native_actions,
    prepare_openai_native_action_request,
)
from cape_loop.openai_provider import (
    BudgetExceeded,
    HTTPResponseBodyTooLarge,
    HTTPResult,
    OpenAIProviderConfig,
    ProviderHTTPError,
    ProviderResponseError,
)
from cape_loop.runner import run_experiment
from cape_loop.schema_export import SCHEMAS


def _closed_loop_run(root: Path) -> Path:
    config = AppConfig(
        run=RunSection(name="native-action-provider-test", seed=151),
        experiment=ExperimentSection(
            kind="closed_loop",
            domains=("travel",),
            mechanisms=("ranking", "default", "suggestion"),
            response_modes=("naturally_sampled",),
            policies=("balanced", "soft_profile_conditioned"),
            updaters=("semantic_memory", "provenance_linked_memory"),
            users=1,
            trajectories_per_cell=1,
            turns=3,
            bootstrap_replicates=0,
        ),
        inference=InferenceSection(
            training_interactions=24,
            fit_steps=10,
            learning_rate=0.04,
            l2=0.001,
        ),
    )
    result = run_experiment(config, output_root=root / "runs")
    return Path(result["run_dir"])


def _native_response(**kwargs: object) -> HTTPResult:
    body = json.loads(bytes(kwargs["body"]).decode("utf-8"))
    visible = json.loads(body["input"][0]["content"][0]["text"])
    actions = []
    for item in visible["terminal_suite"]["items"]:
        direct = item["question_type"] == "direct_preference_probe"
        actions.append(
            {
                "item_id": item["item_id"],
                "item_sha256": item["item_sha256"],
                "wording_template_id": item["wording_template_id"],
                "question_type": item["question_type"],
                "selected_option_id": (
                    None if direct else item["options"][0]["option_id"]
                ),
                "declared_direction": 1 if direct else None,
            }
        )
    raw = {
        "id": (
            "resp_native_"
            + body["metadata"]["cape_loop_native_state_id"][:12]
        ),
        "status": "completed",
        "model": "gpt-5.6-sol",
        "usage": {
            "input_tokens": 60,
            "output_tokens": 40,
            "total_tokens": 100,
            "debug": "test-only",
        },
        "debug": {"credential_echo": "test-only"},
        "output_text": json.dumps({"actions": actions}),
    }
    return HTTPResult(
        status=200,
        headers={"X-Request-Id": "server-test-only"},
        body=json.dumps(raw).encode("utf-8"),
    )


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(
            json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


class NativeActionProviderTests(unittest.TestCase):
    def test_response_requires_explicit_completed_status(self) -> None:
        with self.assertRaisesRegex(
            ProviderResponseError,
            "not completed",
        ):
            native_action_module._extract_actions_payload(
                {"output_text": json.dumps({"actions": []})}
            )

    def test_plan_is_keyless_and_binds_retained_state_and_suite(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = _closed_loop_run(root)
            config = OpenAIProviderConfig(
                api_key_env="ABSENT_NATIVE_ACTION_TEST_KEY",
                max_retries=0,
                max_requests=900,
                max_total_tokens=6_000_000,
            )
            with patch.dict("os.environ", {}, clear=True):
                plan = plan_openai_native_actions(run_dir, config)
                requests = build_native_action_requests(run_dir)
                prepared = prepare_openai_native_action_request(
                    requests[0],
                    config,
                )

            self.assertFalse(plan["live_execution"])
            self.assertFalse(plan["credential_read"])
            self.assertTrue(plan["within_declared_budget"])
            self.assertTrue(
                plan["initial_workload_within_declared_budget"]
            )
            self.assertTrue(
                plan["all_retry_attempts_within_declared_budget"]
            )
            self.assertEqual(plan["request_count"], len(requests))
            self.assertEqual(plan["native_system_id"], NATIVE_ACTION_SYSTEM_ID)
            self.assertEqual(
                plan["endpoint"],
                "https://api.openai.com/v1/responses",
            )
            self.assertEqual(
                plan["api_key_env"],
                "ABSENT_NATIVE_ACTION_TEST_KEY",
            )
            self.assertFalse(plan["allow_custom_base_url"])
            self.assertTrue(plan["official_origin_locked"])
            self.assertEqual(
                plan["budget_accounting_unit"],
                "actual_transport_attempt",
            )
            self.assertNotIn("source_run", plan)
            self.assertEqual(
                plan["source_run_id"],
                json.loads(
                    (run_dir / "manifest.json").read_text(encoding="utf-8")
                )["run_id"],
            )
            self.assertEqual(
                plan["collection_config"]["max_retries"],
                config.max_retries,
            )
            self.assertEqual(len(plan["plan_sha256"]), 64)
            self.assertNotIn("Authorization", prepared.headers)
            self.assertFalse(prepared.body["store"])
            self.assertEqual(
                prepared.body["metadata"][
                    "cape_loop_collection_config_sha256"
                ],
                plan["collection_config_sha256"],
            )
            self.assertEqual(
                prepared.body["metadata"]["cape_loop_native_state_id"],
                requests[0].native_state_id,
            )
            visible = json.loads(
                prepared.body["input"][0]["content"][0]["text"]
            )
            self.assertEqual(
                visible["native_memory_state"],
                dict(requests[0].native_state),
            )
            self.assertEqual(
                visible["terminal_suite"],
                requests[0].suite.to_dict(),
            )
            self.assertNotIn("latent_truth", visible)
            self.assertNotIn("user_id", visible)

            moved = root / "mirror" / run_dir.name
            moved.parent.mkdir()
            copytree(run_dir, moved)
            with patch.dict("os.environ", {}, clear=True):
                moved_plan = plan_openai_native_actions(moved, config)
            self.assertEqual(plan, moved_plan)

    def test_live_collection_rejects_retry_expansion_before_output_or_key(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = _closed_loop_run(root)
            requests = build_native_action_requests(run_dir)
            output = root / "must-not-exist"
            provider = OpenAINativeActionProvider(
                OpenAIProviderConfig(
                    live_execution=True,
                    api_key_env="ABSENT_NATIVE_ACTION_TEST_KEY",
                    max_retries=1,
                    max_requests=len(requests),
                    max_total_tokens=6_000_000,
                ),
                transport=lambda **_: self.fail(
                    "budget preflight must run before transport"
                ),
            )
            with patch.dict("os.environ", {}, clear=True):
                with self.assertRaisesRegex(
                    ValueError,
                    "hard budget after retry expansion",
                ):
                    execute_openai_native_actions(
                        run_dir,
                        output,
                        provider,
                    )
            self.assertFalse(output.exists())

    def test_ambiguous_transport_failure_is_never_retried(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = _closed_loop_run(root)
            request = build_native_action_requests(run_dir)[0]
            config = OpenAIProviderConfig(
                live_execution=True,
                api_key_env="CAPE_LOOP_NATIVE_ACTION_TEST_KEY",
                max_retries=9,
                max_requests=900,
                max_total_tokens=6_000_000,
            )
            calls: list[int] = []
            provider = OpenAINativeActionProvider(
                config,
                transport=lambda **_: (
                    calls.append(1)
                    or (_ for _ in ()).throw(
                        ConnectionError("outcome unknown")
                    )
                ),
                sleep=lambda _: self.fail(
                    "ambiguous transport outcomes must not back off and retry"
                ),
            )
            reservation = provider.prepare(request).estimated_max_tokens
            with patch.dict(
                "os.environ",
                {"CAPE_LOOP_NATIVE_ACTION_TEST_KEY": "test-only"},
                clear=True,
            ):
                with self.assertRaisesRegex(
                    NativeActionManualReviewRequired,
                    "ambiguous.*manual review",
                ):
                    provider.complete(request)
            self.assertEqual(calls, [1])
            self.assertEqual(provider.budget.request_count, 1)
            self.assertEqual(provider.budget.total_tokens, reservation)

    def test_live_collection_is_audit_first_valid_and_resumable(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = _closed_loop_run(root)
            output = root / "native-actions"
            config = OpenAIProviderConfig(
                live_execution=True,
                api_key_env="CAPE_LOOP_NATIVE_ACTION_TEST_KEY",
                max_requests=900,
                max_total_tokens=6_000_000,
            )
            provider = OpenAINativeActionProvider(
                config,
                transport=_native_response,
                epoch_time=lambda: 1_800_000_000.0,
            )
            with patch.dict(
                "os.environ",
                {"CAPE_LOOP_NATIVE_ACTION_TEST_KEY": "test-only"},
                clear=True,
            ):
                result = execute_openai_native_actions(
                    run_dir,
                    output,
                    provider,
                )

            records = read_native_terminal_action_records(
                output / "native-actions.jsonl"
            )
            self.assertEqual(len(records), result["request_count"])
            self.assertTrue(
                all(
                    record.native_system_id == NATIVE_ACTION_SYSTEM_ID
                    for record in records
                )
            )
            self.assertTrue(
                all(
                    record.adapter_kind == "native_end_to_end_recorded"
                    for record in records
                )
            )
            audit_text = (output / "provider-audit.jsonl").read_text(
                encoding="utf-8"
            )
            self.assertNotIn("test-only", audit_text)
            first_audit = json.loads(audit_text.splitlines()[0])
            audit_schema = SCHEMAS["native-action-provider-audit"]
            self.assertEqual(
                set(first_audit),
                set(audit_schema["properties"]),
            )
            self.assertTrue(
                set(audit_schema["required"]).issubset(first_audit)
            )
            manifest = json.loads(
                (output / "execution-manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(manifest["claim_status"], "not_claimed")
            self.assertFalse(manifest["credentials_retained"])
            plan = json.loads(
                (output / "collection-plan.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["collection_plan_sha256"],
                plan["plan_sha256"],
            )
            self.assertEqual(
                manifest["collection_config_sha256"],
                plan["collection_config_sha256"],
            )
            self.assertNotIn("source_run", manifest)
            self.assertEqual(
                manifest["source_run_id"],
                plan["source_run_id"],
            )
            attempt_rows = _jsonl(output / "transport-attempts.jsonl")
            self.assertEqual(len(attempt_rows), 2 * result["request_count"])
            attempt_schema = SCHEMAS["native-action-transport-attempt"]
            for row in attempt_rows:
                branch = attempt_schema["oneOf"][
                    0 if row["event"] == "started" else 1
                ]
                self.assertEqual(set(row), set(branch["properties"]))
                self.assertTrue(set(branch["required"]).issubset(row))
            self.assertEqual(
                manifest["transport_attempt_count"],
                result["request_count"],
            )
            self.assertTrue(
                all(
                    row["collection_plan_sha256"] == plan["plan_sha256"]
                    for row in attempt_rows
                    if row["event"] == "started"
                )
            )

            def unexpected_transport(**_: object) -> HTTPResult:
                raise AssertionError("a complete resume must not call a provider")

            resumed = OpenAINativeActionProvider(
                config,
                transport=unexpected_transport,
            )
            with patch.dict(
                "os.environ",
                {"CAPE_LOOP_NATIVE_ACTION_TEST_KEY": "test-only"},
                clear=True,
            ):
                second = execute_openai_native_actions(
                    run_dir,
                    output,
                    resumed,
                )
            self.assertEqual(second["new_request_count"], 0)
            self.assertEqual(
                second["reused_request_count"],
                result["request_count"],
            )

            lock_descriptor = os.open(
                output / ".collection.lock",
                os.O_RDWR | os.O_CREAT,
                0o600,
            )
            lock_acquired = False
            try:
                lock_acquired = try_file_lock(lock_descriptor)
                self.assertTrue(lock_acquired)
                with self.assertRaisesRegex(
                    RuntimeError,
                    "holds the output lock",
                ):
                    execute_openai_native_actions(
                        run_dir,
                        output,
                        resumed,
                    )
            finally:
                if lock_acquired:
                    unlock_file(lock_descriptor)
                os.close(lock_descriptor)

    def test_output_cannot_equal_or_mutate_the_verified_source_run(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = _closed_loop_run(root)
            config = OpenAIProviderConfig(
                live_execution=True,
                api_key_env="ABSENT_NATIVE_ACTION_TEST_KEY",
                max_requests=900,
                max_total_tokens=6_000_000,
            )

            def unexpected_transport(**_: object) -> HTTPResult:
                raise AssertionError("invalid output must fail before transport")

            for output in (run_dir, run_dir / "native-actions"):
                with self.subTest(output=output):
                    provider = OpenAINativeActionProvider(
                        config,
                        transport=unexpected_transport,
                    )
                    with self.assertRaisesRegex(
                        ValueError,
                        "cannot equal or be inside",
                    ):
                        execute_openai_native_actions(
                            run_dir,
                            output,
                            provider,
                        )
            self.assertFalse((run_dir / ".collection.lock").exists())
            self.assertFalse((run_dir / "native-actions").exists())
            valid, errors = verify_run(run_dir)
            self.assertTrue(valid, errors)

    def test_empty_request_corpus_fails_before_output_creation(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "native-actions"
            provider = OpenAINativeActionProvider(
                OpenAIProviderConfig(live_execution=True),
                transport=lambda **_: self.fail("transport must not run"),
            )
            with patch.object(
                native_action_module,
                "build_native_action_requests",
                return_value=(),
            ):
                with self.assertRaisesRegex(ValueError, "no eligible"):
                    execute_openai_native_actions(
                        root / "source-run",
                        output,
                        provider,
                    )
            self.assertFalse(output.exists())

    def test_resume_rejects_origin_retry_and_budget_plan_drift(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = _closed_loop_run(root)
            output = root / "native-actions"
            base = OpenAIProviderConfig(
                live_execution=True,
                api_key_env="CAPE_LOOP_NATIVE_ACTION_TEST_KEY",
                max_requests=900,
                max_total_tokens=6_000_000,
            )
            with patch.dict(
                "os.environ",
                {"CAPE_LOOP_NATIVE_ACTION_TEST_KEY": "test-only"},
                clear=True,
            ):
                execute_openai_native_actions(
                    run_dir,
                    output,
                    OpenAINativeActionProvider(
                        base,
                        transport=_native_response,
                    ),
                )
            original_plan = (output / "collection-plan.json").read_bytes()
            drifted = (
                OpenAIProviderConfig(
                    live_execution=True,
                    api_key_env="CAPE_LOOP_PROXY_KEY",
                    base_url="https://proxy.example.test",
                    allow_custom_base_url=True,
                    max_requests=900,
                    max_total_tokens=6_000_000,
                ),
                OpenAIProviderConfig(
                    live_execution=True,
                    api_key_env="CAPE_LOOP_NATIVE_ACTION_TEST_KEY",
                    max_retries=3,
                    max_requests=900,
                    max_total_tokens=6_000_000,
                ),
                OpenAIProviderConfig(
                    live_execution=True,
                    api_key_env="CAPE_LOOP_NATIVE_ACTION_TEST_KEY",
                    max_requests=901,
                    max_total_tokens=6_000_000,
                ),
            )

            def unexpected_transport(**_: object) -> HTTPResult:
                raise AssertionError("plan drift must fail before transport")

            for config in drifted:
                with self.subTest(config=config):
                    with self.assertRaisesRegex(
                        ValueError,
                        "collection plan has a different",
                    ):
                        execute_openai_native_actions(
                            run_dir,
                            output,
                            OpenAINativeActionProvider(
                                config,
                                transport=unexpected_transport,
                            ),
                        )
            self.assertEqual(
                (output / "collection-plan.json").read_bytes(),
                original_plan,
            )

    def test_unresolved_attempt_blocks_resume_without_another_call(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = _closed_loop_run(root)
            output = root / "native-actions"
            config = OpenAIProviderConfig(
                live_execution=True,
                api_key_env="CAPE_LOOP_NATIVE_ACTION_TEST_KEY",
                max_requests=900,
                max_total_tokens=6_000_000,
            )
            append = native_action_module._append_jsonl

            def crash_before_settlement(
                path: Path,
                record: dict[str, object],
            ) -> None:
                if (
                    path.name == "transport-attempts.jsonl"
                    and record.get("event") == "settled"
                ):
                    raise OSError("simulated settlement crash")
                append(path, record)

            with patch.dict(
                "os.environ",
                {"CAPE_LOOP_NATIVE_ACTION_TEST_KEY": "test-only"},
                clear=True,
            ), patch.object(
                native_action_module,
                "_append_jsonl",
                side_effect=crash_before_settlement,
            ):
                with self.assertRaisesRegex(
                    OSError,
                    "simulated settlement crash",
                ):
                    execute_openai_native_actions(
                        run_dir,
                        output,
                        OpenAINativeActionProvider(
                            config,
                            transport=_native_response,
                        ),
                    )
            attempts = _jsonl(output / "transport-attempts.jsonl")
            self.assertEqual([row["event"] for row in attempts], ["started"])

            calls = 0

            def unexpected_transport(**_: object) -> HTTPResult:
                nonlocal calls
                calls += 1
                raise AssertionError("unresolved attempt must not be retried")

            with self.assertRaisesRegex(
                NativeActionManualReviewRequired,
                "manual review",
            ):
                execute_openai_native_actions(
                    run_dir,
                    output,
                    OpenAINativeActionProvider(
                        config,
                        transport=unexpected_transport,
                    ),
                )
            self.assertEqual(calls, 0)

    def test_usage_over_reservation_stops_with_unresolved_attempt(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = _closed_loop_run(root)
            output = root / "native-actions"
            config = OpenAIProviderConfig(
                live_execution=True,
                api_key_env="CAPE_LOOP_NATIVE_ACTION_TEST_KEY",
                max_requests=900,
                max_total_tokens=6_000_000,
            )

            def over_reservation(**kwargs: object) -> HTTPResult:
                response = _native_response(**kwargs)
                raw = json.loads(response.body)
                request = build_native_action_requests(run_dir)[0]
                estimate = prepare_openai_native_action_request(
                    request,
                    config,
                ).estimated_max_tokens
                raw["usage"] = {"total_tokens": estimate + 1}
                return HTTPResult(
                    status=response.status,
                    headers=response.headers,
                    body=json.dumps(raw).encode("utf-8"),
                )

            with patch.dict(
                "os.environ",
                {"CAPE_LOOP_NATIVE_ACTION_TEST_KEY": "test-only"},
                clear=True,
            ):
                with self.assertRaisesRegex(
                    ProviderResponseError,
                    "exceeds the conservative reservation",
                ):
                    execute_openai_native_actions(
                        run_dir,
                        output,
                        OpenAINativeActionProvider(
                            config,
                            transport=over_reservation,
                        ),
                    )
            attempts = _jsonl(output / "transport-attempts.jsonl")
            self.assertEqual(
                [row["event"] for row in attempts],
                ["started"],
            )

            with self.assertRaisesRegex(
                NativeActionManualReviewRequired,
                "manual review",
            ):
                execute_openai_native_actions(
                    run_dir,
                    output,
                    OpenAINativeActionProvider(
                        config,
                        transport=lambda **_: self.fail(
                            "an over-reservation response must not be retried"
                        ),
                    ),
                )

    def test_invalid_paid_response_is_charged_and_blocks_resume(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = _closed_loop_run(root)
            output = root / "native-actions"
            config = OpenAIProviderConfig(
                live_execution=True,
                api_key_env="CAPE_LOOP_NATIVE_ACTION_TEST_KEY",
                max_requests=900,
                max_total_tokens=6_000_000,
            )

            def invalid_response(**_: object) -> HTTPResult:
                return HTTPResult(
                    status=200,
                    headers={"X-Request-Id": "server-test-only"},
                    body=b"not-json test-only",
                )

            with patch.dict(
                "os.environ",
                {"CAPE_LOOP_NATIVE_ACTION_TEST_KEY": "test-only"},
                clear=True,
            ):
                with self.assertRaises(ProviderResponseError):
                    execute_openai_native_actions(
                        run_dir,
                        output,
                        OpenAINativeActionProvider(
                            config,
                            transport=invalid_response,
                        ),
                    )
            attempts = _jsonl(output / "transport-attempts.jsonl")
            self.assertEqual(
                [row["event"] for row in attempts],
                ["started", "settled"],
            )
            self.assertEqual(attempts[1]["outcome"], "invalid_response")
            self.assertEqual(
                attempts[1]["charged_tokens"],
                attempts[0]["estimated_max_tokens"],
            )
            self.assertNotIn(
                "test-only",
                (output / "transport-attempts.jsonl").read_text(
                    encoding="utf-8"
                ),
            )

            calls = 0

            def unexpected_transport(**_: object) -> HTTPResult:
                nonlocal calls
                calls += 1
                raise AssertionError("settled invalid response must not retry")

            with self.assertRaisesRegex(
                NativeActionManualReviewRequired,
                "without an embedded",
            ):
                execute_openai_native_actions(
                    run_dir,
                    output,
                    OpenAINativeActionProvider(
                        config,
                        transport=unexpected_transport,
                    ),
                )
            self.assertEqual(calls, 0)

    def test_oversized_response_is_settled_without_body_retention(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = _closed_loop_run(root)
            output = root / "native-actions"
            config = OpenAIProviderConfig(
                live_execution=True,
                api_key_env="CAPE_LOOP_NATIVE_ACTION_TEST_KEY",
                max_requests=900,
                max_total_tokens=6_000_000,
            )

            def oversized_response(**_: object) -> HTTPResult:
                raise HTTPResponseBodyTooLarge(status=413)

            with patch.dict(
                "os.environ",
                {"CAPE_LOOP_NATIVE_ACTION_TEST_KEY": "test-only"},
                clear=True,
            ):
                with self.assertRaises(ProviderResponseError) as caught:
                    execute_openai_native_actions(
                        run_dir,
                        output,
                        OpenAINativeActionProvider(
                            config,
                            transport=oversized_response,
                        ),
                    )
            self.assertNotIn("test-only", str(caught.exception))
            attempts = _jsonl(output / "transport-attempts.jsonl")
            self.assertEqual(
                [row["event"] for row in attempts],
                ["started", "settled"],
            )
            settlement = attempts[1]
            self.assertEqual(settlement["outcome"], "invalid_response")
            self.assertEqual(settlement["http_status"], 413)
            self.assertIsNone(settlement["response_body_sha256"])
            self.assertIsNone(settlement["response_record"])
            self.assertIsNone(settlement["provider_audit"])
            self.assertEqual(
                settlement["charged_tokens"],
                attempts[0]["estimated_max_tokens"],
            )
            self.assertNotIn(
                "test-only",
                (output / "transport-attempts.jsonl").read_text(
                    encoding="utf-8"
                ),
            )

    def test_physical_attempt_budget_caps_retries(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = _closed_loop_run(root)
            request = build_native_action_requests(run_dir)[0]
            config = OpenAIProviderConfig(
                live_execution=True,
                api_key_env="CAPE_LOOP_NATIVE_ACTION_TEST_KEY",
                max_retries=100,
                initial_backoff_seconds=0,
                max_backoff_seconds=0,
                jitter_fraction=0,
                max_requests=1,
                max_total_tokens=100_000_000,
            )
            calls = 0

            def rate_limited(**_: object) -> HTTPResult:
                nonlocal calls
                calls += 1
                return HTTPResult(
                    status=429,
                    headers={"Retry-After": "999999"},
                    body=b'{"error":{"message":"rate limited"}}',
                )

            with patch.dict(
                "os.environ",
                {"CAPE_LOOP_NATIVE_ACTION_TEST_KEY": "test-only"},
                clear=True,
            ):
                with self.assertRaises(BudgetExceeded):
                    provider = OpenAINativeActionProvider(
                        config,
                        transport=rate_limited,
                        sleep=lambda _: None,
                    )
                    provider.complete(request)
            self.assertEqual(calls, 1)
            self.assertEqual(provider.budget.request_count, 1)

            bounded = OpenAINativeActionProvider(
                OpenAIProviderConfig(
                    max_backoff_seconds=2,
                    initial_backoff_seconds=1,
                ),
                random_value=lambda: 0.5,
                epoch_time=lambda: 0,
            )
            self.assertEqual(bounded._backoff(1, "999999"), 2)

    def test_provider_error_header_identifier_is_redacted(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = _closed_loop_run(root)
            output = root / "native-actions"
            config = OpenAIProviderConfig(
                live_execution=True,
                api_key_env="CAPE_LOOP_NATIVE_ACTION_TEST_KEY",
                max_requests=900,
                max_total_tokens=6_000_000,
            )

            def rejected(**_: object) -> HTTPResult:
                return HTTPResult(
                    status=400,
                    headers={"X-Request-Id": "server-test-only"},
                    body=b'{"error":{"message":"bad test-only"}}',
                )

            with patch.dict(
                "os.environ",
                {"CAPE_LOOP_NATIVE_ACTION_TEST_KEY": "test-only"},
                clear=True,
            ):
                with self.assertRaises(ProviderHTTPError) as caught:
                    execute_openai_native_actions(
                        run_dir,
                        output,
                        OpenAINativeActionProvider(
                            config,
                            transport=rejected,
                        ),
                    )
            self.assertNotIn("test-only", str(caught.exception))
            self.assertNotIn(
                "test-only",
                caught.exception.server_request_id or "",
            )
            self.assertNotIn(
                "test-only",
                (output / "transport-attempts.jsonl").read_text(
                    encoding="utf-8"
                ),
            )

    def test_attempt_audit_recovers_crash_before_public_audit_append(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = _closed_loop_run(root)
            output = root / "native-actions"
            requests = build_native_action_requests(run_dir)
            config = OpenAIProviderConfig(
                live_execution=True,
                api_key_env="CAPE_LOOP_NATIVE_ACTION_TEST_KEY",
                max_requests=900,
                max_total_tokens=6_000_000,
            )
            append = native_action_module._append_jsonl

            def crash_on_audit(
                path: Path,
                record: dict[str, object],
            ) -> None:
                if path.name == "provider-audit.jsonl":
                    raise OSError("simulated audit append crash")
                append(path, record)

            with patch.dict(
                "os.environ",
                {"CAPE_LOOP_NATIVE_ACTION_TEST_KEY": "test-only"},
                clear=True,
            ), patch.object(
                native_action_module,
                "_append_jsonl",
                side_effect=crash_on_audit,
            ):
                with self.assertRaisesRegex(
                    OSError,
                    "simulated audit append crash",
                ):
                    execute_openai_native_actions(
                        run_dir,
                        output,
                        OpenAINativeActionProvider(
                            config,
                            transport=_native_response,
                        ),
                    )
            first_state_id = _jsonl(
                output / "transport-attempts.jsonl"
            )[0]["native_state_id"]
            resumed_state_ids: list[str] = []

            def resumed_response(**kwargs: object) -> HTTPResult:
                body = json.loads(bytes(kwargs["body"]).decode("utf-8"))
                resumed_state_ids.append(
                    body["metadata"]["cape_loop_native_state_id"]
                )
                return _native_response(**kwargs)

            with patch.dict(
                "os.environ",
                {"CAPE_LOOP_NATIVE_ACTION_TEST_KEY": "test-only"},
                clear=True,
            ):
                result = execute_openai_native_actions(
                    run_dir,
                    output,
                    OpenAINativeActionProvider(
                        config,
                        transport=resumed_response,
                    ),
                )
            self.assertEqual(result["reused_request_count"], 1)
            self.assertEqual(
                len(resumed_state_ids),
                len(requests) - 1,
            )
            self.assertNotIn(first_state_id, resumed_state_ids)

    def test_resume_strictly_rejects_tampered_returned_model(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = _closed_loop_run(root)
            output = root / "native-actions"
            config = OpenAIProviderConfig(
                live_execution=True,
                api_key_env="CAPE_LOOP_NATIVE_ACTION_TEST_KEY",
                max_requests=900,
                max_total_tokens=6_000_000,
            )
            with patch.dict(
                "os.environ",
                {"CAPE_LOOP_NATIVE_ACTION_TEST_KEY": "test-only"},
                clear=True,
            ):
                execute_openai_native_actions(
                    run_dir,
                    output,
                    OpenAINativeActionProvider(
                        config,
                        transport=_native_response,
                    ),
                )
            audits = _jsonl(output / "provider-audit.jsonl")
            audits[0]["model_returned"] = "gpt-5.6-terra"
            attempts = _jsonl(output / "transport-attempts.jsonl")
            for row in attempts:
                embedded = row.get("provider_audit")
                if (
                    isinstance(embedded, dict)
                    and embedded.get("request_id") == audits[0]["request_id"]
                ):
                    embedded["model_returned"] = "gpt-5.6-terra"
            _write_jsonl(output / "provider-audit.jsonl", audits)
            _write_jsonl(output / "transport-attempts.jsonl", attempts)

            with self.assertRaisesRegex(
                ValueError,
                "acceptance does not match returned model",
            ):
                execute_openai_native_actions(
                    run_dir,
                    output,
                    OpenAINativeActionProvider(
                        config,
                        transport=lambda **_: self.fail(
                            "tampered resume must not call transport"
                        ),
                    ),
                )


if __name__ == "__main__":
    unittest.main()
