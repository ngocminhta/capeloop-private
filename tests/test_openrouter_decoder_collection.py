from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json
import unittest

from cape_loop.cli import build_parser, main as cli_main
from cape_loop.decoder_study import (
    ExternalDecoderRequest,
    external_decoder_llm_request,
    read_external_decoder_requests,
)
from cape_loop.gate_review import (
    OPENROUTER_COLLECTION_PROVENANCE,
)
from cape_loop.openrouter_decoder_collection import (
    OPENROUTER_CLAUDE_DECODER_MODEL,
    OPENROUTER_DECODER_MAX_OUTPUT_TOKENS,
    OPENROUTER_DECODER_MAX_REQUESTS,
    OPENROUTER_DECODER_MAX_TOTAL_TOKENS,
    OPENROUTER_GEMINI_DECODER_MODEL,
    validate_openrouter_decoder_collection,
)
from cape_loop.openrouter_provider import (
    HTTPResult,
    OpenRouterChatProvider,
    OpenRouterProviderConfig,
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


def _response(model: str) -> bytes:
    upstream = "Anthropic" if model.startswith("anthropic/") else "Google"
    raw = {
        "id": "generation-test",
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
        "openrouter_metadata": {
            "requested": model,
            "strategy": "direct",
            "attempt": 1,
            "endpoints": {
                "available": [
                    {
                        "provider": upstream,
                        "model": model,
                        "selected": True,
                    }
                ]
            },
            "attempts": [
                {
                    "provider": upstream,
                    "model": model,
                    "status": 200,
                }
            ],
            "pipeline": [],
        },
    }
    return json.dumps(raw).encode("utf-8")


class OpenRouterDecoderCollectionTests(unittest.TestCase):
    @staticmethod
    def _write_request(path: Path) -> None:
        request = ExternalDecoderRequest.build(
            request_id="selected-openrouter-decoder",
            pseudonymous_state_id="selected-openrouter-state",
            representation_id="blinded-native-content-v1",
            evaluation_split="development",
            payload={
                "representation_version": "blinded-native-content-v1",
                "episodes": [],
                "semantic_claims": [],
                "persona_text": "",
            },
        )
        path.write_text(
            json.dumps(request.to_dict(), sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def test_decoder_cli_defaults_to_bounded_pair_and_per_model_effort(
        self,
    ) -> None:
        args = build_parser().parse_args(
            ["decoder-study", "plan-openrouter", "requests.jsonl"]
        )
        self.assertIsNone(args.model)
        self.assertEqual(args.additional_model, [])
        self.assertEqual(args.max_retries, 0)
        self.assertEqual(
            args.max_output_tokens,
            OPENROUTER_DECODER_MAX_OUTPUT_TOKENS,
        )
        self.assertEqual(args.max_requests, OPENROUTER_DECODER_MAX_REQUESTS)
        self.assertEqual(
            args.max_total_tokens,
            OPENROUTER_DECODER_MAX_TOTAL_TOKENS,
        )
        gate = build_parser().parse_args(
            [
                "gate-review",
                "import-native",
                "run",
                "requests",
                "judgments",
                "truth",
                "native",
                "source-review",
                "output",
                "--openrouter-collection-dir",
                "openrouter-collection",
            ]
        )
        self.assertEqual(
            gate.external_collection_dir,
            Path("openrouter-collection"),
        )
        self.assertEqual(
            gate.external_collection_provenance_mode,
            OPENROUTER_COLLECTION_PROVENANCE,
        )
        experiment_c = build_parser().parse_args(
            [
                "experiment-c-decoder",
                "import",
                "run",
                "judgments",
                "output",
                "--openrouter-collection-dir",
                "openrouter-collection",
            ]
        )
        self.assertEqual(
            experiment_c.external_collection_dir,
            Path("openrouter-collection"),
        )
        self.assertEqual(
            experiment_c.external_collection_provenance_mode,
            OPENROUTER_COLLECTION_PROVENANCE,
        )

    def test_selected_gate4_claude_request_uses_bedrock_safe_schema(
        self,
    ) -> None:
        request = ExternalDecoderRequest.build(
            request_id="selected-openrouter-decoder",
            pseudonymous_state_id="selected-openrouter-state",
            representation_id="blinded-native-content-v1",
            evaluation_split="development",
            payload={
                "representation_version": "blinded-native-content-v1",
                "episodes": [],
                "semantic_claims": [],
                "persona_text": "",
            },
        )
        provider_request = external_decoder_llm_request(
            request,
            decoder_instance_id="selected-openrouter-claude",
        )
        prepared = OpenRouterChatProvider(
            OpenRouterProviderConfig(
                model=OPENROUTER_CLAUDE_DECODER_MODEL,
                reasoning_effort="low",
            )
        ).prepare(provider_request)
        schema = prepared.body["response_format"]["json_schema"]["schema"]
        encoded_schema = json.dumps(schema, sort_keys=True)
        self.assertNotIn('"minimum"', encoded_schema)
        self.assertNotIn('"maximum"', encoded_schema)
        self.assertIn(
            "Probability in the inclusive range [0, 1].",
            encoded_schema,
        )

    def test_complete_collection_revalidates_and_rejects_tampering(
        self,
    ) -> None:
        def provider_factory(config: object) -> OpenRouterChatProvider:
            model = str(getattr(config, "model"))

            def transport(**_: object) -> HTTPResult:
                return HTTPResult(
                    status=200,
                    headers={
                        "X-OpenRouter-Cache-Status": "MISS",
                        "X-Generation-Id": "generation-test",
                    },
                    body=_response(model),
                )

            return OpenRouterChatProvider(config, transport=transport)

        with TemporaryDirectory() as directory:
            root = Path(directory)
            requests_path = root / "requests.jsonl"
            output = root / "collection"
            self._write_request(requests_path)
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
            ):
                status = cli_main(
                    [
                        "decoder-study",
                        "execute-openrouter",
                        str(requests_path),
                        str(output),
                        "--execute-live",
                    ]
                )
            self.assertEqual(status, 0)
            plan = json.loads(
                (output / "collection-plan.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                plan["models"],
                [
                    OPENROUTER_CLAUDE_DECODER_MODEL,
                    OPENROUTER_GEMINI_DECODER_MODEL,
                ],
            )
            self.assertEqual(
                {
                    source["model"]: source["reasoning_effort"]
                    for source in plan["sources"]
                },
                {
                    OPENROUTER_CLAUDE_DECODER_MODEL: "low",
                    OPENROUTER_GEMINI_DECODER_MODEL: "minimal",
                },
            )
            self.assertFalse(
                plan["strict_first_party_gate4_eligible"]
            )
            self.assertTrue(
                plan["eligible_for_reviewed_shared_gateway_admission"]
            )
            manifest = json.loads(
                (output / "execution-manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(manifest["shared_gateway"])
            self.assertFalse(manifest["first_party_origin_claimed"])
            self.assertFalse(
                manifest["strict_first_party_gate4_eligible"]
            )
            self.assertTrue(
                manifest[
                    "eligible_for_reviewed_shared_gateway_admission"
                ]
            )
            self.assertFalse(manifest["statistical_independence_claimed"])
            requests = read_external_decoder_requests(requests_path)
            judgments, _, summary = validate_openrouter_decoder_collection(
                output,
                requests=requests,
                judgments_path=output / "judgments.jsonl",
            )
            self.assertEqual(len(judgments), 2)
            self.assertEqual(
                summary["provenance_mode"],
                "selected_openrouter_gateway_collection",
            )
            audit_path = output / "provider-audit.jsonl"
            audit_path.write_text(
                audit_path.read_text(encoding="utf-8") + "{}\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError,
                "aggregate OpenRouter audits",
            ):
                validate_openrouter_decoder_collection(
                    output,
                    requests=requests,
                    judgments_path=output / "judgments.jsonl",
                )

    def test_live_execution_rejects_symlinked_journal_before_provider_setup(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            requests_path = root / "requests.jsonl"
            output = root / "collection"
            self._write_request(requests_path)
            journals = output / "journals"
            journals.mkdir(parents=True)
            model_digest = __import__("hashlib").sha256(
                OPENROUTER_CLAUDE_DECODER_MODEL.encode("utf-8")
            ).hexdigest()[:12]
            journal = journals / model_digest
            journal.mkdir()
            (journal / "provider-audit.jsonl").symlink_to(
                root / "outside.jsonl"
            )
            with patch(
                "cape_loop.cli.OpenRouterChatProvider",
                side_effect=AssertionError("provider was constructed"),
            ):
                with self.assertRaises(SystemExit) as caught:
                    cli_main(
                        [
                            "decoder-study",
                            "execute-openrouter",
                            str(requests_path),
                            str(output),
                            "--execute-live",
                        ]
                    )
            self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
