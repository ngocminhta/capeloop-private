from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json
import unittest

from cape_loop.cli import build_parser, main as cli_main
from cape_loop.decoder_study import ExternalDecoderRequest
from cape_loop.external_decoder_providers import (
    ANTHROPIC_DEFAULT_MODEL,
    ANTHROPIC_OFFICIAL_ORIGIN,
    GEMINI_DEFAULT_MODEL,
    GEMINI_OFFICIAL_ORIGIN,
    _ExclusiveCollectionLock,
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
