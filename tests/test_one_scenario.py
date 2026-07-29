from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from hashlib import sha256
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch
import json
import unittest

from cape_loop.cli import build_parser, main as cli_main
from cape_loop.conversation_surfaces import load_conversation_bank
from cape_loop.llm_exchange import LLMRequest, LLMResponse
from cape_loop.one_scenario import (
    render_one_scenario_markdown,
    run_one_scenario,
)
from cape_loop.openrouter_provider import (
    HTTPResult,
    OPENROUTER_EXAMPLE_MODEL,
    OpenRouterChatProvider,
)
from cape_loop.scenarios import load_scenario_catalog
from cape_loop.schemas import LatentUser, Susceptibility


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "data/scenarios/scenario-catalog-v1.json"
BANK_PATH = ROOT / "data/scenarios/conversation-templates-v1.json"
SCENARIO_ID = "travel-scenario-atlas-lodging-price-01"


def _openrouter_response(model: str) -> bytes:
    beliefs = {
        f"attribute_{attribute}": {
            "-2": 0.4,
            "-1": 0.3,
            "+1": 0.2,
            "+2": 0.1,
        }
        for attribute in range(1, 4)
    }
    return json.dumps(
        {
            "id": "gen_one_scenario_test",
            "object": "chat.completion",
            "created": 1_700_000_001,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({"beliefs": beliefs}),
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 400,
                "completion_tokens": 100,
                "total_tokens": 500,
            },
            "openrouter_metadata": {
                "requested": model,
                "strategy": "direct",
                "region": "fra",
                "summary": "available=1, selected=Google AI Studio",
                "attempt": 1,
                "is_byok": False,
                "endpoints": {
                    "total": 1,
                    "available": [
                        {
                            "provider": "Google AI Studio",
                            "model": model,
                            "selected": True,
                        }
                    ],
                },
                "attempts": [
                    {
                        "provider": "Google AI Studio",
                        "model": model,
                        "status": 200,
                    }
                ],
                "pipeline": [],
            },
        }
    ).encode("utf-8")


class _FixedProvider:
    def __init__(
        self,
        *,
        model_id: str = "test/full-context-model",
        mismatched_request: bool = False,
    ) -> None:
        self.model_id = model_id
        self.mismatched_request = mismatched_request
        self.calls: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        return LLMResponse.parse(
            {
                "schema_version": 1,
                "request_id": (
                    "wrong-request"
                    if self.mismatched_request
                    else request.request_id
                ),
                "prompt_sha256": request.prompt_sha256,
                "model_id": self.model_id,
                "beliefs": {
                    "attribute_1": {
                        "-2": 0.45,
                        "-1": 0.30,
                        "+1": 0.15,
                        "+2": 0.10,
                    },
                    "attribute_2": {
                        "-2": 0.20,
                        "-1": 0.25,
                        "+1": 0.30,
                        "+2": 0.25,
                    },
                    "attribute_3": {
                        "-2": 0.10,
                        "-1": 0.20,
                        "+1": 0.30,
                        "+2": 0.40,
                    },
                },
            }
        )


def _nested_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value).union(
            *(_nested_keys(item) for item in value.values()),
            set(),
        )
    if isinstance(value, (list, tuple)):
        return set().union(*(_nested_keys(item) for item in value), set())
    return set()


class OneScenarioWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = load_scenario_catalog(
            CATALOG_PATH,
            expected_sha256=sha256(CATALOG_PATH.read_bytes()).hexdigest(),
        ).catalog
        cls.bank = load_conversation_bank(BANK_PATH)
        cls.user = LatentUser(
            "walkthrough-user",
            (-2, 1, 2),
            Susceptibility(
                ranking=0.35,
                default=0.80,
                suggestion=0.65,
            ),
        )

    def _run(self, provider: _FixedProvider | None = None):
        active = provider or _FixedProvider()
        return active, run_one_scenario(
            catalog=self.catalog,
            conversation_bank=self.bank,
            scenario_id=SCENARIO_ID,
            user=self.user,
            provider=active,
            mechanism="balanced",
            anchor_direction=-1,
            seed=20260729,
        )

    def test_runs_one_natural_choice_and_exactly_one_model_call(self) -> None:
        provider, result = self._run()

        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(result.provider_call_count, 1)
        self.assertEqual(result.updater_view, "full_context")
        self.assertEqual(result.model_id, "test/full-context-model")
        self.assertIn("Lodging A", result.assistant_message)
        self.assertIn("Lodging B", result.assistant_message)
        self.assertRegex(
            result.user_message,
            r"^I choose Lodging [AB]\.$",
        )
        self.assertEqual(
            result.selected_option_label,
            next(
                option.label
                for option in self.catalog.scenario(SCENARIO_ID).options
                if option.option_id == result.selected_option_id
            ),
        )
        self.assertGreater(result.choice_probability, 0.0)
        self.assertLess(result.choice_probability, 1.0)

        request = provider.calls[0]
        self.assertEqual(request.view, "full_context")
        self.assertEqual(
            request.payload["context"]["conversation"],
            [
                {
                    "role": "assistant",
                    "content": result.assistant_message,
                },
                {
                    "role": "user",
                    "content": result.user_message,
                },
            ],
        )
        keys = _nested_keys(request.payload)
        self.assertNotIn("features", keys)
        self.assertNotIn("target_attribute", keys)

    def test_choice_is_deterministic_and_language_does_not_choose(self) -> None:
        _, first = self._run(_FixedProvider())
        _, second = self._run(_FixedProvider())

        self.assertEqual(
            first.selected_option_id,
            second.selected_option_id,
        )
        self.assertEqual(first.user_message, second.user_message)
        self.assertEqual(first.surface_id, second.surface_id)
        payload = first.to_dict()
        self.assertEqual(
            payload["choice"]["choice_source"],
            "mathematical_user_simulator",
        )
        self.assertEqual(
            payload["choice"]["surface_source"],
            first.surface_source,
        )
        self.assertTrue(first.surface_source)

    def test_machine_payloads_are_compact_complete_and_non_claiming(self) -> None:
        _, result = self._run()
        payload = result.to_dict()
        records = result.jsonl_records()

        self.assertEqual(len(records), 1)
        trace = records[0]
        self.assertEqual(trace["record_kind"], "conversation_trace")
        self.assertEqual(trace["experiment"], "demo")
        self.assertEqual(trace["conversation_kind"], "single_turn")
        self.assertEqual(trace["conditions"]["split"], "train")
        self.assertEqual(trace["conditions"]["mechanism"], "balanced")
        self.assertTrue(trace["conditions"]["demonstration_only"])
        self.assertFalse(trace["conditions"]["claim_eligible"])
        self.assertEqual(len(trace["dialogue"]), 1)
        self.assertEqual(
            trace["dialogue"][0]["turn_metrics"],
            {"choice_probability": result.choice_probability},
        )
        self.assertEqual(
            trace["dialogue"][0]["surface_source"],
            result.surface_source,
        )
        self.assertEqual(len(trace["outcomes"]), 1)
        self.assertEqual(
            trace["outcomes"][0]["model_ids"],
            ["test/full-context-model"],
        )
        self.assertEqual(trace["assessments"], [])
        self.assertEqual(trace["comparisons"], [])
        trace_keys = _nested_keys(trace)
        self.assertNotIn("semantic_profile", trace_keys)
        self.assertNotIn("model_input", trace_keys)
        self.assertNotIn("profile_outputs", trace_keys)
        self.assertFalse(payload["claim_status"]["paper_eligible"])
        self.assertFalse(payload["claim_status"]["claim_eligible"])
        self.assertEqual(
            payload["claim_status"]["status"],
            "demonstration_only",
        )
        self.assertEqual(
            trace["outcomes"][0]["metrics"],
            payload["metrics"],
        )
        self.assertEqual(
            set(payload["metrics"]),
            {
                "action_conditioned_update_error",
                "marginal_kl_from_exact_reference",
                "evaluated_model_brier",
                "exact_reference_brier",
                "excess_brier_vs_exact_reference",
                "evaluated_update_magnitude",
            },
        )
        self.assertIn(
            "exact_action_aware_reference",
            payload["profile_outputs"],
        )
        json.dumps(payload, allow_nan=False)
        for record in records:
            json.dumps(record, allow_nan=False)

    def test_markdown_leads_with_conversation_and_explains_metrics(self) -> None:
        _, result = self._run()
        markdown = render_one_scenario_markdown(result)

        self.assertIn("# One-scenario hybrid experiment walkthrough", markdown)
        self.assertIn("## Natural conversation", markdown)
        self.assertIn(result.assistant_message, markdown)
        self.assertIn(result.user_message, markdown)
        self.assertIn("Provider calls: **1**", markdown)
        self.assertIn(result.surface_source, markdown)
        self.assertIn("Update error vs exact action-aware reference ↓", markdown)
        self.assertIn("Paper eligible: **no**", markdown)
        self.assertIn("Claim eligible: **no**", markdown)
        self.assertLess(
            markdown.index("## Natural conversation"),
            markdown.index("## Metrics"),
        )
        self.assertEqual(result.render_markdown(), markdown)

    def test_rejects_a_provider_response_bound_to_another_request(self) -> None:
        provider = _FixedProvider(mismatched_request=True)
        with self.assertRaisesRegex(
            ValueError,
            "request_id does not match",
        ):
            self._run(provider)
        self.assertEqual(len(provider.calls), 1)


class OneScenarioCLITests(unittest.TestCase):
    def test_parser_has_one_scenario_defaults(self) -> None:
        args = build_parser().parse_args(
            ["demo", "one-scenario", "artifacts/example", "--execute-live"]
        )

        self.assertEqual(args.model, OPENROUTER_EXAMPLE_MODEL)
        self.assertEqual(args.scenario_id, SCENARIO_ID)
        self.assertEqual(args.mechanism, "balanced")
        self.assertEqual(args.seed, 1729)
        self.assertEqual(args.api_key_env, "OPENROUTER_API_KEY")

    def test_missing_live_flag_reads_no_key_and_creates_no_output(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "not-created"
            stderr = StringIO()
            with (
                patch.dict("os.environ", {}, clear=True),
                redirect_stderr(stderr),
                self.assertRaises(SystemExit),
            ):
                cli_main(
                    [
                        "demo",
                        "one-scenario",
                        str(output),
                        "--scenario-catalog",
                        str(CATALOG_PATH),
                        "--conversation-bank",
                        str(BANK_PATH),
                    ]
                )

            self.assertFalse(output.exists())
            self.assertIn("--execute-live", stderr.getvalue())

    def test_live_cli_makes_one_attempt_and_writes_readable_log(self) -> None:
        physical_calls: list[dict[str, Any]] = []
        provider_configs = []

        def transport(**kwargs: Any) -> HTTPResult:
            physical_calls.append(kwargs)
            return HTTPResult(
                status=200,
                headers={"content-type": "application/json"},
                body=_openrouter_response(OPENROUTER_EXAMPLE_MODEL),
            )

        def provider_factory(config):
            provider_configs.append(config)
            return OpenRouterChatProvider(config, transport=transport)

        with TemporaryDirectory() as directory:
            output = Path(directory) / "one-scenario"
            stdout = StringIO()
            with (
                patch.dict(
                    "os.environ",
                    {"OPENROUTER_API_KEY": "test-openrouter-key"},
                    clear=False,
                ),
                patch(
                    "cape_loop.cli.OpenRouterChatProvider",
                    side_effect=provider_factory,
                ),
                redirect_stdout(stdout),
            ):
                status = cli_main(
                    [
                        "demo",
                        "one-scenario",
                        str(output),
                        "--scenario-catalog",
                        str(CATALOG_PATH),
                        "--conversation-bank",
                        str(BANK_PATH),
                        "--execute-live",
                    ]
                )

            self.assertEqual(status, 0)
            self.assertEqual(len(physical_calls), 1)
            self.assertEqual(len(provider_configs), 1)
            config = provider_configs[0]
            self.assertEqual(config.max_requests, 1)
            self.assertEqual(config.max_retries, 0)
            self.assertEqual(config.max_output_tokens, 2048)
            self.assertEqual(config.max_total_tokens, 10_000)
            self.assertFalse(config.allow_fallbacks)
            self.assertEqual(config.data_collection, "deny")

            receipt = json.loads(stdout.getvalue())
            self.assertEqual(receipt["physical_openrouter_calls"], 1)
            self.assertFalse(receipt["paper_eligible"])
            self.assertFalse(receipt["claim_eligible"])
            self.assertEqual(
                receipt["model_returned"],
                OPENROUTER_EXAMPLE_MODEL,
            )

            expected_files = (
                "conversation.md",
                "conversation.jsonl",
                "result.json",
                "llm/requests.jsonl",
                "llm/responses.jsonl",
                "llm/provider-audit.jsonl",
                "llm/provider-attempts.jsonl",
                "llm/provider-manifest.json",
            )
            for relative in expected_files:
                self.assertTrue((output / relative).is_file(), relative)
            markdown = (output / "conversation.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("## Natural conversation", markdown)
            self.assertIn("Provider calls: **1**", markdown)
            trace = json.loads(
                (output / "conversation.jsonl").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(trace["experiment"], "demo")
            self.assertEqual(len(trace["dialogue"]), 1)
            self.assertEqual(len(trace["outcomes"]), 1)
            manifest = json.loads(
                (output / "llm/provider-manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(manifest["requests_executed"], 1)
            self.assertEqual(manifest["requests_resumed"], 0)
            self.assertEqual(manifest["transport_attempt_count"], 1)


if __name__ == "__main__":
    unittest.main()
