from __future__ import annotations

from contextlib import redirect_stdout
from hashlib import sha256
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
import json
import unittest
from unittest.mock import patch

from cape_loop.conversation_reporting import render_markdown
from cape_loop.conversation_surfaces import load_conversation_bank
from cape_loop.cli import (
    _BoundedRequestJournal,
    _experiment_b_diagnostic_live_provider,
    build_parser,
)
from cape_loop.domains import TRAVEL
from cape_loop.experiment_b_diagnostic import (
    ALLOWED_DIAGNOSTIC_TURNS,
    DIAGNOSTIC_POLICY_IDS,
    DIAGNOSTIC_TURNS,
    EXPECTED_PROVIDER_CALLS,
    expected_provider_calls,
    run_experiment_b_diagnostic,
)
from cape_loop.llm_exchange import (
    ATTRIBUTES,
    VALUES,
    LLMRequest,
    LLMResponse,
)
from cape_loop.openai_provider import read_requests
from cape_loop.scenarios import load_scenario_catalog
from cape_loop.schemas import LatentUser, Susceptibility


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "data/scenarios/scenario-catalog-v1.json"
BANK_PATH = ROOT / "data/scenarios/conversation-templates-v1.json"


class _FixedProvider:
    def __init__(
        self,
        *,
        mismatched_field: str | None = None,
    ) -> None:
        self.calls: list[LLMRequest] = []
        self.mismatched_field = mismatched_field

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        return LLMResponse.parse(
            {
                "schema_version": 1,
                "request_id": (
                    "mismatched-request"
                    if self.mismatched_field == "request_id"
                    else request.request_id
                ),
                "prompt_sha256": (
                    "0" * 64
                    if self.mismatched_field == "prompt_sha256"
                    else request.prompt_sha256
                ),
                "model_id": "fixture/full-context-model",
                "beliefs": {
                    attribute: {
                        value: probability
                        for value, probability in zip(
                            VALUES,
                            (0.4, 0.3, 0.2, 0.1),
                        )
                    }
                    for attribute in ATTRIBUTES
                },
            }
        )


class _DeduplicatingDiagnosticAdapter:
    """Small adaptive-adapter fixture with content-addressed reuse."""

    def __init__(self) -> None:
        self._provider = _FixedProvider()
        self._responses: dict[str, LLMResponse] = {}
        self._audits: dict[str, dict[str, Any]] = {}
        self.logical_calls = 0
        self.resumed_count = 0

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.logical_calls += 1
        existing = self._responses.get(request.request_id)
        if existing is not None:
            self.resumed_count += 1
            return existing
        response = self._provider.complete(request)
        self._responses[request.request_id] = response
        self._audits[request.request_id] = {
            "usage": {
                "total_tokens": 10,
                "cost": 0.001,
            },
        }
        return response

    @property
    def used_audit_records(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._audits.values())

    def to_manifest(self) -> dict[str, Any]:
        unique = len(self._responses)
        return {
            "schema_version": 1,
            "provider": "fixture",
            "model_requested": "fixture/full-context-model",
            "requests_used": unique,
            "requests_executed": unique,
            "requests_resumed": self.resumed_count,
            "total_tokens": unique * 10,
            "transport_attempt_count": unique,
            "request_budget_unit": "physical_http_attempt",
        }


class ExperimentBDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = load_scenario_catalog(
            CATALOG_PATH,
            expected_sha256=sha256(
                CATALOG_PATH.read_bytes()
            ).hexdigest(),
        ).catalog
        cls.bank = load_conversation_bank(BANK_PATH)
        cls.user = LatentUser(
            "experiment-b-diagnostic-user",
            (-2, 1, 2),
            Susceptibility(
                ranking=0.35,
                default=0.80,
                suggestion=0.65,
            ),
        )

    def _run(
        self,
        provider: Any | None = None,
        *,
        turns: int = DIAGNOSTIC_TURNS,
    ):
        active = provider or _FixedProvider()
        return active, run_experiment_b_diagnostic(
            user=self.user,
            domain=TRAVEL,
            provider=active,
            scenario_catalog=self.catalog,
            conversation_bank=self.bank,
            seed=20260729,
            turns=turns,
        )

    def test_runs_exact_six_call_matched_diagnostic(self) -> None:
        provider, diagnostic = self._run()

        self.assertEqual(EXPECTED_PROVIDER_CALLS, 6)
        self.assertEqual(len(provider.calls), EXPECTED_PROVIDER_CALLS)
        self.assertEqual(
            diagnostic.provider_call_count,
            EXPECTED_PROVIDER_CALLS,
        )
        self.assertEqual(diagnostic.turns, DIAGNOSTIC_TURNS)
        self.assertEqual(len(diagnostic.requests), EXPECTED_PROVIDER_CALLS)
        self.assertEqual(len(diagnostic.responses), EXPECTED_PROVIDER_CALLS)

        trajectories = diagnostic.experiment_result.trajectories
        self.assertEqual(
            {trajectory.policy_id for trajectory in trajectories},
            set(DIAGNOSTIC_POLICY_IDS),
        )
        self.assertEqual(len(trajectories), 2)
        for trajectory in trajectories:
            self.assertEqual(
                trajectory.initial_profile_condition,
                "incorrect",
            )
            self.assertEqual(trajectory.updater_id, "llm_full_context")
            self.assertEqual(len(trajectory.turns), DIAGNOSTIC_TURNS)
            self.assertEqual(
                len(trajectory.audit_record.interactions),
                DIAGNOSTIC_TURNS,
            )
            self.assertTrue(trajectory.same_history_shadow)

        for request in diagnostic.requests:
            self.assertEqual(request.view, "full_context")
            conversation = request.payload["context"]["conversation"]
            self.assertEqual(
                [message["role"] for message in conversation],
                ["assistant", "user"],
            )
            self.assertTrue(conversation[0]["content"])
            self.assertTrue(conversation[1]["content"])

    def test_supports_complete_attribute_cycles_with_dynamic_call_bound(
        self,
    ) -> None:
        self.assertEqual(ALLOWED_DIAGNOSTIC_TURNS, (3, 6, 9, 12))
        for turns in ALLOWED_DIAGNOSTIC_TURNS:
            with self.subTest(turns=turns):
                provider, diagnostic = self._run(turns=turns)
                expected = 2 * turns
                self.assertEqual(expected_provider_calls(turns), expected)
                self.assertEqual(len(provider.calls), expected)
                self.assertEqual(
                    diagnostic.provider_call_count,
                    expected,
                )
                self.assertEqual(len(diagnostic.requests), expected)
                self.assertEqual(len(diagnostic.responses), expected)
                self.assertEqual(diagnostic.turns, turns)
                self.assertTrue(
                    all(
                        len(trajectory.turns) == turns
                        for trajectory in (
                            diagnostic.experiment_result.trajectories
                        )
                    )
                )
                self.assertTrue(
                    all(
                        len(record["dialogue"]) == turns
                        for record in diagnostic.conversation_records
                    )
                )

    def test_rejects_partial_or_unsupported_cycles_before_provider_call(
        self,
    ) -> None:
        for turns in (0, 1, 2, 4, 15, 3.0, True):
            with self.subTest(turns=turns):
                provider = _FixedProvider()
                with self.assertRaisesRegex(
                    ValueError,
                    "complete attribute cycle",
                ):
                    self._run(provider, turns=turns)  # type: ignore[arg-type]
                self.assertEqual(provider.calls, [])

    def test_returns_records_for_existing_conversation_renderer(self) -> None:
        _, diagnostic = self._run()

        self.assertEqual(len(diagnostic.conversation_records), 2)
        for record in diagnostic.conversation_records:
            self.assertEqual(record["record_kind"], "conversation_trace")
            self.assertEqual(record["experiment"], "B")
            self.assertEqual(record["conversation_kind"], "closed_loop")
            self.assertEqual(len(record["dialogue"]), DIAGNOSTIC_TURNS)
            self.assertTrue(record["conditions"]["diagnostic_only"])
            self.assertFalse(record["conditions"]["claim_eligible"])
            self.assertEqual(
                record["outcomes"][0]["model_ids"],
                ["fixture/full-context-model"],
            )
            for turn in record["dialogue"]:
                self.assertTrue(turn["assistant"])
                self.assertTrue(turn["user"])

        markdown = render_markdown(
            diagnostic.conversation_records,
            experiment="B diagnostic",
            complete_jsonl_path="conversation.jsonl",
        )
        self.assertIn("6 dialogue turns", markdown)
        self.assertIn("Scenario presenter (assistant)", markdown)
        self.assertIn("fixture/full-context-model", markdown)

    def test_rejects_unbound_provider_response(self) -> None:
        for field in ("request_id", "prompt_sha256"):
            with self.subTest(field=field):
                with self.assertRaisesRegex(
                    ValueError,
                    field,
                ):
                    self._run(_FixedProvider(mismatched_field=field))

    def test_cli_exposes_bounded_live_diagnostic_defaults(self) -> None:
        args = build_parser().parse_args(
            [
                "demo",
                "experiment-b-case",
                "artifacts/example-b-case",
                "--execute-live",
            ]
        )

        self.assertEqual(args.provider, "openrouter")
        self.assertEqual(args.model, "")
        self.assertEqual(args.reasoning_effort, "")
        self.assertEqual(args.upstream_provider, "")
        self.assertEqual(args.timeout_seconds, 40.0)
        self.assertTrue(args.execute_live)

        with TemporaryDirectory() as directory:
            adapter, provider, model, effort = (
                _experiment_b_diagnostic_live_provider(
                    args,
                    llm_dir=Path(directory),
                )
            )
        self.assertEqual(provider, "openrouter")
        self.assertEqual(model, "google/gemini-3.6-flash")
        self.assertEqual(effort, "minimal")
        self.assertEqual(
            adapter.provider.config.upstream_provider,
            "google-vertex/global",
        )
        self.assertEqual(adapter.provider.config.max_requests, 6)
        self.assertEqual(adapter.provider.config.max_retries, 0)

    def test_explicit_default_openrouter_model_keeps_pinned_route(self) -> None:
        args = build_parser().parse_args(
            [
                "demo",
                "experiment-b-case",
                "artifacts/example-b-case",
                "--model",
                "google/gemini-3.6-flash",
                "--turns",
                "12",
                "--execute-live",
            ]
        )

        with TemporaryDirectory() as directory:
            adapter, _, _, _ = _experiment_b_diagnostic_live_provider(
                args,
                llm_dir=Path(directory),
            )
        self.assertEqual(
            adapter.provider.config.upstream_provider,
            "google-vertex/global",
        )
        self.assertEqual(adapter.provider.config.max_requests, 24)

    def test_cli_resolves_direct_openai_diagnostic_defaults(self) -> None:
        args = build_parser().parse_args(
            [
                "demo",
                "experiment-b-case",
                "artifacts/example-b-openai-case",
                "--provider",
                "openai",
                "--turns",
                "6",
                "--execute-live",
            ]
        )

        with TemporaryDirectory() as directory:
            adapter, provider, model, effort = (
                _experiment_b_diagnostic_live_provider(
                    args,
                    llm_dir=Path(directory),
                )
            )
        self.assertEqual(provider, "openai")
        self.assertEqual(model, "gpt-5.6-sol")
        self.assertEqual(effort, "medium")
        self.assertEqual(
            adapter.provider.config.api_key_env,
            "OPENAI_API_KEY",
        )
        self.assertEqual(adapter.provider.config.max_requests, 12)
        self.assertEqual(adapter.provider.config.max_retries, 0)

    def test_request_journal_survives_a_provider_failure(self) -> None:
        class _FailOnSecondCall(_FixedProvider):
            def complete(self, request: LLMRequest) -> LLMResponse:
                if len(self.calls) == 1:
                    self.calls.append(request)
                    raise RuntimeError("fixture provider interruption")
                return super().complete(request)

        provider = _FailOnSecondCall()
        with TemporaryDirectory() as directory:
            request_path = Path(directory) / "requests.jsonl"
            journal = _BoundedRequestJournal(
                provider,
                request_path=request_path,
                logical_event_path=(
                    Path(directory) / "logical-request-events.jsonl"
                ),
                max_calls=6,
            )
            first_request = LLMRequest.build(
                request_id="first",
                updater_id="llm_full_context",
                view="full_context",
                prior={"fixture": True},
                observation={"fixture": "first"},
                context={"fixture": "first"},
            )
            second_request = LLMRequest.build(
                request_id="second",
                updater_id="llm_full_context",
                view="full_context",
                prior={"fixture": True},
                observation={"fixture": "second"},
                context={"fixture": "second"},
            )

            journal.complete(first_request)
            with self.assertRaisesRegex(
                RuntimeError,
                "fixture provider interruption",
            ):
                journal.complete(second_request)

            retained = [
                json.loads(line)
                for line in request_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            logical_events = [
                json.loads(line)
                for line in journal.logical_event_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
        self.assertEqual(
            [row["request_id"] for row in retained],
            ["first", "second"],
        )
        self.assertEqual(
            [row["logical_call"] for row in logical_events],
            [1, 2],
        )
        self.assertEqual(journal.call_count, 2)

    def test_cli_retains_logical_requests_and_accounts_for_dedup(self) -> None:
        adapter = _DeduplicatingDiagnosticAdapter()
        with TemporaryDirectory() as directory:
            output = Path(directory) / "diagnostic"
            args = build_parser().parse_args(
                [
                    "demo",
                    "experiment-b-case",
                    str(output),
                    "--execute-live",
                ]
            )
            stdout = StringIO()
            with (
                patch(
                    "cape_loop.cli."
                    "_experiment_b_diagnostic_live_provider",
                    return_value=(
                        adapter,
                        "fixture",
                        "fixture/full-context-model",
                        "none",
                    ),
                ),
                redirect_stdout(stdout),
            ):
                self.assertEqual(args.handler(args), 0)

            receipt = json.loads(stdout.getvalue())
            request_rows = [
                json.loads(line)
                for line in (
                    output / "llm/requests.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]
            logical_events = [
                json.loads(line)
                for line in (
                    output / "llm/logical-request-events.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]
            result = json.loads(
                (output / "result.json").read_text(encoding="utf-8")
            )
            manifest = json.loads(
                (
                    output / "llm/provider-manifest.json"
                ).read_text(encoding="utf-8")
            )
            replayable_request_count = len(
                read_requests(output / "llm/requests.jsonl")
            )

        self.assertEqual(len(logical_events), 6)
        self.assertEqual(receipt["logical_profile_updates"], 6)
        self.assertLess(receipt["physical_provider_calls"], 6)
        self.assertEqual(
            len(request_rows),
            receipt["unique_profile_update_requests"],
        )
        self.assertEqual(replayable_request_count, len(request_rows))
        self.assertEqual(
            sum(
                bool(row["reused_content_addressed_request"])
                for row in logical_events
            ),
            6 - len(request_rows),
        )
        self.assertEqual(
            receipt["unique_profile_update_requests"],
            receipt["physical_provider_calls"],
        )
        self.assertEqual(
            result["design"]["logical_profile_updates"],
            6,
        )
        self.assertEqual(
            manifest["requests_resumed"],
            6 - manifest["requests_used"],
        )


if __name__ == "__main__":
    unittest.main()
