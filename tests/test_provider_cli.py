from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json
import threading
import unittest

from cape_loop.cli import build_parser, main as cli_main
from cape_loop.decoder_study import ExternalDecoderRequest
from cape_loop.external_decoder_providers import (
    ANTHROPIC_DEFAULT_MODEL,
    ANTHROPIC_OFFICIAL_ORIGIN,
    ExternalDecoderExecutionLocked,
    GEMINI_DEFAULT_MODEL,
    GEMINI_OFFICIAL_ORIGIN,
    _ExclusiveCollectionLock,
)
from cape_loop.llm_exchange import (
    ATTRIBUTES,
    VALUES,
    LLMRequest,
    LLMResponse,
    write_requests,
)
from cape_loop.native_action_provider import NATIVE_ACTION_SYSTEM_ID
from cape_loop.openai_provider import DEFAULT_OPENAI_MODEL_ROLES


class ProviderCLIContractTests(unittest.TestCase):
    @staticmethod
    def _request() -> ExternalDecoderRequest:
        return ExternalDecoderRequest.build(
            request_id="decoder-cli-test",
            pseudonymous_state_id="state-cli-test",
            representation_id="blinded-native-content-v1",
            evaluation_split="development",
            payload={
                "representation_version": "blinded-native-content-v1",
                "episodes": [],
                "semantic_claims": [],
                "persona_text": "",
            },
        )

    def _write_request(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self._request().to_dict(), sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _write_llm_request(path: Path) -> None:
        request = LLMRequest.build(
            request_id="provider-cli-profile-test",
            updater_id="llm_response_only",
            view="response_only",
            prior={},
            observation={},
        )
        write_requests(path, (request,))

    def test_all_openai_cli_paths_reject_other_provider_credentials(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            llm_requests = root / "llm-requests.jsonl"
            decoder_requests = root / "decoder-requests.jsonl"
            self._write_llm_request(llm_requests)
            self._write_request(decoder_requests)
            cases = (
                (
                    [
                        "llm",
                        "plan",
                        str(llm_requests),
                        "--api-key-env",
                        "OPENROUTER_API_KEY",
                    ],
                    "OPENROUTER_API_KEY",
                ),
                (
                    [
                        "decoder-study",
                        "plan-openai",
                        str(decoder_requests),
                        "--api-key-env",
                        "ANTHROPIC_API_KEY",
                    ],
                    "ANTHROPIC_API_KEY",
                ),
                (
                    [
                        "native-action",
                        "plan-openai",
                        str(root / "absent-run"),
                        "--api-key-env",
                        "GEMINI_API_KEY",
                    ],
                    "GEMINI_API_KEY",
                ),
            )
            for arguments, reserved_key in cases:
                with self.subTest(arguments=arguments):
                    stderr = StringIO()
                    with redirect_stderr(stderr):
                        with self.assertRaises(SystemExit) as raised:
                            cli_main(arguments)
                    self.assertEqual(raised.exception.code, 2)
                    self.assertIn(
                        "reserved for a different provider",
                        stderr.getvalue(),
                    )
                    self.assertIn(reserved_key, stderr.getvalue())

    def test_selected_provider_defaults_are_exposed_by_the_parser(self) -> None:
        parser = build_parser()
        distinct = parser.parse_args(
            ["decoder-study", "plan-distinct", "requests.jsonl"]
        )
        self.assertEqual(distinct.anthropic_model, ANTHROPIC_DEFAULT_MODEL)
        self.assertEqual(distinct.gemini_model, GEMINI_DEFAULT_MODEL)
        self.assertEqual(distinct.anthropic_api_key_env, "ANTHROPIC_API_KEY")
        self.assertEqual(distinct.gemini_api_key_env, "GEMINI_API_KEY")
        self.assertEqual(distinct.max_requests_per_source, 900)
        self.assertEqual(distinct.max_total_tokens_per_source, 6_000_000)

        native = parser.parse_args(
            ["native-action", "plan-openai", "run"]
        )
        self.assertEqual(
            native.model,
            DEFAULT_OPENAI_MODEL_ROLES["primary"].model,
        )
        self.assertEqual(
            native.reasoning_effort,
            DEFAULT_OPENAI_MODEL_ROLES["primary"].reasoning_effort,
        )
        self.assertEqual(native.max_requests, 900)
        self.assertEqual(native.max_total_tokens, 6_000_000)

    def test_gate4_model_manifest_matches_executable_defaults(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        manifest = json.loads(
            (
                repository
                / "data/model-suites/gate4-native-and-distinct-decoders.json"
            ).read_text(encoding="utf-8")
        )
        native = manifest["native_action_system"]
        self.assertEqual(native["system_id"], NATIVE_ACTION_SYSTEM_ID)
        self.assertEqual(
            native["model"],
            DEFAULT_OPENAI_MODEL_ROLES["primary"].model,
        )
        self.assertEqual(
            native["reasoning_effort"],
            DEFAULT_OPENAI_MODEL_ROLES["primary"].reasoning_effort,
        )
        self.assertEqual(native["request_budget_unit"], "physical_http_attempt")

        sources = {
            source["provider"]: source
            for source in manifest["external_decoders"]
        }
        self.assertEqual(
            sources["anthropic"]["model"],
            ANTHROPIC_DEFAULT_MODEL,
        )
        self.assertEqual(
            sources["anthropic"]["official_origin"],
            ANTHROPIC_OFFICIAL_ORIGIN,
        )
        self.assertEqual(
            sources["google_gemini"]["model"],
            GEMINI_DEFAULT_MODEL,
        )
        self.assertEqual(
            sources["google_gemini"]["official_origin"],
            GEMINI_OFFICIAL_ORIGIN,
        )
        self.assertTrue(
            all(
                source["request_budget_unit"] == "physical_http_attempt"
                and source["max_requests"] == 900
                and source["max_total_tokens"] == 6_000_000
                for source in sources.values()
            )
        )

    def test_distinct_plan_is_keyless_and_machine_readable(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "requests.jsonl"
            self._write_request(path)
            stdout = StringIO()
            with (
                patch.dict(
                    "os.environ",
                    {
                        "ANTHROPIC_API_KEY": "must-not-be-read",
                        "GEMINI_API_KEY": "must-not-be-read",
                    },
                    clear=True,
                ),
                redirect_stdout(stdout),
            ):
                status = cli_main(
                    ["decoder-study", "plan-distinct", str(path)]
                )
        self.assertEqual(status, 0)
        plan = json.loads(stdout.getvalue())
        self.assertFalse(plan["credential_read"])
        self.assertEqual(plan["request_count"], 1)
        self.assertEqual(plan["source_count"], 2)
        self.assertNotIn("must-not-be-read", stdout.getvalue())

    def test_plan_cannot_mutate_a_source_run(self) -> None:
        with TemporaryDirectory() as directory:
            run = Path(directory) / "source-run"
            request_path = run / "decoder" / "external-requests.jsonl"
            self._write_request(request_path)
            for marker in (
                "manifest.json",
                "config.resolved.json",
                "SHA256SUMS",
            ):
                (run / marker).write_text("{}\n", encoding="utf-8")
            output = run / "gate4-plan.json"
            stderr = StringIO()
            with redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as raised:
                    cli_main(
                        [
                            "decoder-study",
                            "plan-distinct",
                            str(request_path),
                            "--output",
                            str(output),
                        ]
                    )
            self.assertEqual(raised.exception.code, 2)
            self.assertIn("outside the immutable source run", stderr.getvalue())
            self.assertFalse(output.exists())

    def test_distinct_plan_cannot_overwrite_its_request_corpus(self) -> None:
        with TemporaryDirectory() as directory:
            request_path = Path(directory) / "requests.jsonl"
            self._write_request(request_path)
            original = request_path.read_bytes()
            stderr = StringIO()
            with redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as raised:
                    cli_main(
                        [
                            "decoder-study",
                            "plan-distinct",
                            str(request_path),
                            "--output",
                            str(request_path),
                        ]
                    )
            self.assertEqual(raised.exception.code, 2)
            self.assertIn("cannot overwrite its input", stderr.getvalue())
            self.assertEqual(request_path.read_bytes(), original)

    def test_native_plan_cannot_mutate_a_source_run(self) -> None:
        with TemporaryDirectory() as directory:
            run = Path(directory) / "source-run"
            run.mkdir()
            for marker in (
                "manifest.json",
                "config.resolved.json",
                "SHA256SUMS",
            ):
                (run / marker).write_text("{}\n", encoding="utf-8")
            output = run / "native-plan.json"
            stderr = StringIO()
            with redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as raised:
                    cli_main(
                        [
                            "native-action",
                            "plan-openai",
                            str(run),
                            "--output",
                            str(output),
                        ]
                    )
            self.assertEqual(raised.exception.code, 2)
            self.assertIn("outside the immutable source run", stderr.getvalue())
            self.assertFalse(output.exists())

    def test_execute_lock_covers_plan_through_manifest(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            request_path = root / "requests.jsonl"
            output = root / "collection"
            self._write_request(request_path)
            lock_path = output / ".external-decoder-command.lock"
            stderr = StringIO()
            with _ExclusiveCollectionLock(lock_path):
                with redirect_stderr(stderr):
                    with self.assertRaises(SystemExit) as raised:
                        cli_main(
                            [
                                "decoder-study",
                                "execute-distinct",
                                str(request_path),
                                str(output),
                                "--execute-live",
                            ]
                        )
            self.assertEqual(raised.exception.code, 2)
            self.assertIn("holds the output lock", stderr.getvalue())
            self.assertFalse((output / "collection-plan.json").exists())

    def test_generic_decoder_commands_share_one_dispatch_wide_lock(
        self,
    ) -> None:
        cases = (
            (
                "cape_loop.cli.ResumableOpenAICompletionProvider",
                "cape_loop.cli.ResumableOpenRouterCompletionProvider",
                "openai",
                "openrouter",
            ),
            (
                "cape_loop.cli.ResumableOpenRouterCompletionProvider",
                "cape_loop.cli.ResumableOpenAICompletionProvider",
                "openrouter",
                "openai",
            ),
        )
        for first_adapter, second_adapter, first, second in cases:
            with self.subTest(first=first, second=second):
                self._assert_generic_decoder_lock_transaction(
                    first_adapter=first_adapter,
                    second_adapter=second_adapter,
                    first=first,
                    second=second,
                )

    def _assert_generic_decoder_lock_transaction(
        self,
        *,
        first_adapter: str,
        second_adapter: str,
        first: str,
        second: str,
    ) -> None:
        entered_dispatch = threading.Event()
        release_dispatch = threading.Event()
        dispatched_request_ids: list[str] = []
        thread_errors: list[BaseException] = []

        class BlockingAdapter:
            def __init__(self, provider: object, **_: object) -> None:
                self.provider = provider

            def complete(self, request: LLMRequest) -> LLMResponse:
                dispatched_request_ids.append(request.request_id)
                entered_dispatch.set()
                if not release_dispatch.wait(timeout=5):
                    raise TimeoutError("test did not release provider dispatch")
                return LLMResponse(
                    request_id=request.request_id,
                    prompt_sha256=request.prompt_sha256,
                    model_id=self.provider.config.model,
                    beliefs={
                        attribute: {value: 0.25 for value in VALUES}
                        for attribute in ATTRIBUTES
                    },
                )

            def to_manifest(self) -> dict[str, object]:
                return {
                    "schema_version": 1,
                    "model": self.provider.config.model,
                }

        class FailIfReachedAdapter:
            def __init__(self, *_: object, **__: object) -> None:
                raise AssertionError(
                    "concurrent decoder reached provider reconciliation"
                )

        with TemporaryDirectory() as directory:
            root = Path(directory)
            request_path = root / "requests.jsonl"
            output = root / "generic-collection"
            self._write_request(request_path)

            def command(provider: str) -> list[str]:
                arguments = [
                    "decoder-study",
                    f"execute-{provider}",
                    str(request_path),
                    str(output),
                ]
                if provider == "openai":
                    arguments.extend(["--roles", "decoder"])
                arguments.append("--execute-live")
                return arguments

            parser = build_parser()
            first_args = parser.parse_args(command(first))
            second_args = parser.parse_args(command(second))

            def run_first() -> None:
                try:
                    with redirect_stdout(StringIO()):
                        first_args.handler(first_args)
                except BaseException as exc:
                    thread_errors.append(exc)

            with (
                patch(first_adapter, BlockingAdapter),
                patch(second_adapter, FailIfReachedAdapter),
            ):
                thread = threading.Thread(target=run_first, daemon=True)
                thread.start()
                try:
                    self.assertTrue(
                        entered_dispatch.wait(timeout=5),
                        "first decoder never entered provider dispatch",
                    )
                    with self.assertRaises(
                        ExternalDecoderExecutionLocked
                    ):
                        second_args.handler(second_args)
                    self.assertEqual(len(dispatched_request_ids), 1)
                finally:
                    release_dispatch.set()
                    thread.join(timeout=5)

            self.assertFalse(thread.is_alive())
            self.assertEqual(thread_errors, [])
            self.assertEqual(len(dispatched_request_ids), 1)

    def test_live_commands_fail_before_reading_missing_inputs(self) -> None:
        cases = (
            (
                [
                    "decoder-study",
                    "execute-distinct",
                    "absent-requests.jsonl",
                    "output",
                ],
                "explicit --execute-live flag",
            ),
            (
                [
                    "native-action",
                    "execute-openai",
                    "absent-run",
                    "output",
                ],
                "explicit --execute-live flag",
            ),
        )
        for arguments, expected in cases:
            with self.subTest(arguments=arguments):
                stderr = StringIO()
                with redirect_stderr(stderr):
                    with self.assertRaises(SystemExit) as raised:
                        cli_main(arguments)
                self.assertEqual(raised.exception.code, 2)
                self.assertIn(expected, stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
