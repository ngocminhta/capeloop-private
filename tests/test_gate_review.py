from __future__ import annotations

from contextlib import redirect_stdout
from hashlib import sha256
from io import StringIO
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json
import os
import shutil
import unittest

import cape_loop.gate_review as gate_review_module
from cape_loop.artifacts import canonical_json, verify_run
from cape_loop.cli import main as cli_main
from cape_loop.config import (
    AppConfig,
    ExperimentSection,
    InferenceSection,
    RunSection,
)
from cape_loop.decoder_study import (
    ExternalDecoderJudgment,
    read_external_decoder_judgments,
    read_external_decoder_requests,
)
from cape_loop.external_decoder_providers import (
    ANTHROPIC_DEFAULT_MODEL,
    GEMINI_DEFAULT_MODEL,
    ExternalDecoderProvider,
    HTTPResult as ExternalHTTPResult,
)
from cape_loop.file_lock import try_file_lock, unlock_file
from cape_loop.gate_review import (
    DecoderSourceAssessment,
    DecoderSourcePairAssessment,
    DecoderSourceReview,
    NativeTerminalActionRecord,
    OPENROUTER_COLLECTION_PROVENANCE,
    import_native_gate_review,
    validate_official_external_decoder_collection,
    verify_gate_review,
)
from cape_loop.heldout import TerminalAction
from cape_loop.native_action_provider import (
    OpenAINativeActionProvider,
    _build_collection_plan,
    build_native_action_requests,
    execute_openai_native_actions,
)
from cape_loop.openai_provider import (
    HTTPResult as OpenAIHTTPResult,
    OpenAIProviderConfig,
)
from cape_loop.openrouter_provider import (
    HTTPResult as OpenRouterHTTPResult,
    OpenRouterChatProvider,
)
from cape_loop.runner import run_experiment
from cape_loop.schema_export import SCHEMAS


_ROWS = (
    (0.25, 0.25, 0.25, 0.25),
    (0.25, 0.25, 0.25, 0.25),
    (0.25, 0.25, 0.25, 0.25),
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(canonical_json(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _native_response(**kwargs: object) -> OpenAIHTTPResult:
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
            "resp_gate_review_"
            + body["metadata"]["cape_loop_native_state_id"][:12]
        ),
        "status": "completed",
        "model": "gpt-5.6-sol",
        "usage": {
            "input_tokens": 60,
            "output_tokens": 40,
            "total_tokens": 100,
        },
        "output_text": json.dumps({"actions": actions}),
    }
    return OpenAIHTTPResult(
        status=200,
        headers={"X-Request-Id": "gate-review-test-request"},
        body=json.dumps(raw).encode("utf-8"),
    )


_EXTERNAL_BELIEFS = {
    f"attribute_{attribute}": {
        "-2": 0.1,
        "-1": 0.2,
        "+1": 0.3,
        "+2": 0.4,
    }
    for attribute in range(1, 4)
}


def _external_provider(config: object) -> ExternalDecoderProvider:
    def transport(**_: object) -> ExternalHTTPResult:
        if config.provider == "anthropic":
            raw = {
                "id": "msg_gate_review_test",
                "type": "message",
                "role": "assistant",
                "model": ANTHROPIC_DEFAULT_MODEL,
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {"beliefs": _EXTERNAL_BELIEFS}
                        ),
                    }
                ],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 80, "output_tokens": 40},
            }
        else:
            raw = {
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [
                                {
                                    "text": json.dumps(
                                        {"beliefs": _EXTERNAL_BELIEFS}
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
                "modelVersion": GEMINI_DEFAULT_MODEL,
                "responseId": "gemini-gate-review-test",
                "modelStatus": {"modelStage": "STABLE"},
            }
        return ExternalHTTPResult(
            status=200,
            headers={"X-Request-Id": f"{config.provider}-gate-review"},
            body=json.dumps(raw).encode("utf-8"),
        )

    return ExternalDecoderProvider(
        config,
        transport=transport,
        epoch_time=lambda: 1_800_000_000.0,
    )


def _openrouter_external_provider(config: object) -> OpenRouterChatProvider:
    model = str(getattr(config, "model"))
    upstream = "Anthropic" if model.startswith("anthropic/") else "Google"

    def transport(**_: object) -> OpenRouterHTTPResult:
        raw = {
            "id": f"generation-gate-review-{upstream.lower()}",
            "object": "chat.completion",
            "created": 1_800_000_000,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(
                            {"beliefs": _EXTERNAL_BELIEFS}
                        ),
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 80,
                "completion_tokens": 40,
                "total_tokens": 120,
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
        return OpenRouterHTTPResult(
            status=200,
            headers={
                "X-OpenRouter-Cache-Status": "MISS",
                "X-Generation-Id": raw["id"],
            },
            body=json.dumps(raw).encode("utf-8"),
        )

    return OpenRouterChatProvider(
        config,
        transport=transport,
        epoch_time=lambda: 1_800_000_000.0,
    )


class GateReviewRecordTests(unittest.TestCase):
    def test_action_record_rejects_wrong_split_and_reference_adapter(self) -> None:
        action = TerminalAction(
            item_id="item-1",
            item_sha256="a" * 64,
            wording_template_id="wording-1",
            question_type="direct_preference_probe",
            declared_direction=1,
        )
        record = NativeTerminalActionRecord.build(
            record_id="record-1",
            trajectory_id="trajectory-1",
            domain_id="travel",
            updater_id="semantic_memory",
            native_state_id="b" * 64,
            native_system_id="native-system",
            native_system_version="v1",
            suite_id="suite-1",
            suite_sha256="c" * 64,
            action_execution_mode="recorded_replay",
            execution_trace_sha256="d" * 64,
            recorded_at="2026-07-26T12:00:00+00:00",
            actions=(action,),
        )
        wrong_split = record.to_dict()
        wrong_split["evaluation_split"] = "development"
        with self.assertRaisesRegex(ValueError, "must be test"):
            NativeTerminalActionRecord.parse(wrong_split)
        reference = record.to_dict()
        reference["adapter_kind"] = "native_persona_action_reference"
        with self.assertRaisesRegex(ValueError, "reference"):
            NativeTerminalActionRecord.parse(reference)

    def test_public_gate_review_records_have_strict_schemas(self) -> None:
        for name in (
            "native-terminal-action-record",
            "decoder-source-review",
            "gate4-review-artifact",
        ):
            with self.subTest(schema=name):
                schema = SCHEMAS[name]
                self.assertFalse(schema["additionalProperties"])
                self.assertEqual(
                    schema["$id"],
                    f"urn:cape-loop:schema:{name}:v1",
                )
        review_inputs = SCHEMAS["gate4-review-artifact"]["properties"][
            "inputs"
        ]
        self.assertTrue(
            {
                "native_collection_plan",
                "native_action_requests",
                "native_transport_attempts",
                "native_provider_audit",
                "native_terminal_actions",
                "native_execution_manifest",
            }
            <= set(review_inputs["required"])
        )
        self.assertTrue(
            {
                "decoder_collection_plan",
                "decoder_transport_attempts",
                "decoder_provider_audit",
                "decoder_execution_manifest",
            }
            <= set(review_inputs["properties"])
        )
        self.assertEqual(len(review_inputs["oneOf"]), 2)


class GateReviewIntegrationTests(unittest.TestCase):
    def _assert_no_partial_output(self, output: Path) -> None:
        self.assertFalse(output.exists())
        self.assertFalse(output.is_symlink())
        self.assertFalse(
            (
                output.parent
                / f".{output.name}.gate4-review.lock"
            ).exists()
        )
        self.assertEqual(
            list(output.parent.glob(f".{output.name}.*.staging")),
            [],
        )

    def test_import_is_complete_checksum_bound_and_does_not_mutate_run(
        self,
    ) -> None:
        config = AppConfig(
            run=RunSection(name="gate-review-test", seed=71),
            experiment=ExperimentSection(
                kind="closed_loop",
                domains=("travel",),
                mechanisms=("ranking", "default", "suggestion"),
                response_modes=("naturally_sampled",),
                policies=("balanced", "soft_profile_conditioned"),
                updaters=(
                    "semantic_memory",
                    "provenance_linked_memory",
                ),
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
        with TemporaryDirectory() as directory:
            root = Path(directory)
            run_result = run_experiment(config, output_root=root / "runs")
            run_dir = Path(run_result["run_dir"])
            before = sha256(
                (run_dir / "SHA256SUMS").read_bytes()
            ).hexdigest()

            requests_path = (
                run_dir / "decoder" / "external-requests.jsonl"
            )
            truth_path = (
                run_dir
                / "decoder"
                / "truth-labels.researcher-only.jsonl"
            )
            requests = read_external_decoder_requests(requests_path)
            external_collection = root / "external-decoder-collection"
            with (
                patch(
                    "cape_loop.cli.ExternalDecoderProvider",
                    side_effect=_external_provider,
                ),
                patch.dict(
                    "os.environ",
                    {
                        "ANTHROPIC_API_KEY": "test-anthropic-key",
                        "GEMINI_API_KEY": "test-gemini-key",
                    },
                    clear=True,
                ),
                redirect_stdout(StringIO()),
            ):
                self.assertEqual(
                    cli_main(
                        [
                            "decoder-study",
                            "execute-distinct",
                            str(requests_path),
                            str(external_collection),
                            "--execute-live",
                        ]
                    ),
                    0,
                )
            portable_summary = json.loads(
                (
                    external_collection / "execution-manifest.json"
                ).read_text(encoding="utf-8")
            )["execution_summary"]
            self.assertEqual(
                portable_summary["judgments_path"],
                "judgments.jsonl",
            )
            self.assertEqual(
                portable_summary["audit_path"],
                "provider-audit.jsonl",
            )
            self.assertEqual(
                portable_summary["attempt_path"],
                "transport-attempts.jsonl",
            )
            judgments_path = external_collection / "judgments.jsonl"
            judgments = read_external_decoder_judgments(judgments_path)
            (
                wrapped_judgments,
                wrapped_inputs,
                wrapped_summary,
            ) = validate_official_external_decoder_collection(
                external_collection,
                run_dir=run_dir,
                requests=requests,
                judgments_path=judgments_path,
            )
            self.assertEqual(wrapped_judgments, judgments)
            self.assertEqual(
                set(wrapped_inputs),
                {
                    "decoder_collection_plan",
                    "decoder_transport_attempts",
                    "decoder_provider_audit",
                    "decoder_judgments",
                    "decoder_execution_manifest",
                },
            )
            self.assertEqual(
                wrapped_summary["provenance_mode"],
                "validated_direct_first_party_collection",
            )
            source_metadata = {
                judgment.decoder_instance_id: (
                    judgment.decoder_family_id,
                    judgment.judgment_origin,
                    judgment.source_descriptor,
                )
                for judgment in judgments
            }
            source_ids = sorted(source_metadata)

            source_review = DecoderSourceReview.build(
                review_id="gate4-source-review-1",
                responsible_researcher_id="researcher-1",
                reviewed_at="2026-07-26T12:00:00+00:00",
                requests_sha256=sha256(
                    requests_path.read_bytes()
                ).hexdigest(),
                judgments_sha256=sha256(
                    judgments_path.read_bytes()
                ).hexdigest(),
                decision="eligible_distinct_sources",
                source_assessments=tuple(
                    DecoderSourceAssessment(
                        source_id,
                        source_metadata[source_id][0],
                        source_metadata[source_id][1],
                        source_metadata[source_id][2],
                        True,
                        "Reviewed provider, training, prompt, and adjudication.",
                    )
                    for source_id in source_ids
                ),
                pair_assessments=tuple(
                    DecoderSourcePairAssessment(
                        left,
                        right,
                        True,
                        "Responsible researcher found no disqualifying shared "
                        "generation or adjudication dependency.",
                    )
                    for left, right in combinations(source_ids, 2)
                ),
            )
            source_review_path = root / "source-review.json"
            source_review_path.write_text(
                canonical_json(source_review.to_dict()) + "\n",
                encoding="utf-8",
            )

            native_collection = root / "native-action-collection"
            provider = OpenAINativeActionProvider(
                OpenAIProviderConfig(
                    live_execution=True,
                    api_key_env="GATE_REVIEW_NATIVE_TEST_KEY",
                    max_requests=900,
                    max_total_tokens=6_000_000,
                ),
                transport=_native_response,
                epoch_time=lambda: 1_800_000_000.0,
            )
            with patch.dict(
                "os.environ",
                {"GATE_REVIEW_NATIVE_TEST_KEY": "test-only"},
                clear=True,
            ):
                execute_openai_native_actions(
                    run_dir,
                    native_collection,
                    provider,
                )

            output = root / "gate-review"
            result = import_native_gate_review(
                run_dir=run_dir,
                requests_path=requests_path,
                judgments_path=judgments_path,
                truth_labels_path=truth_path,
                native_collection_dir=native_collection,
                source_review_path=source_review_path,
                output_dir=output,
                external_collection_dir=external_collection,
            )
            self.assertEqual(result["claim_status"], "not_claimed")
            valid_review, review_errors = verify_gate_review(
                output,
                source_run_dir=run_dir,
            )
            self.assertTrue(valid_review, review_errors)
            review = json.loads(
                (output / "gate-review.json").read_text(encoding="utf-8")
            )
            self.assertEqual(review["claim_status"], "not_claimed")
            criteria = {
                row["criterion_id"]: row
                for row in review["gate_4"]["criteria"]
            }
            self.assertTrue(
                criteria["independent-blinded-decoder-judgments"]["passed"]
            )
            self.assertTrue(
                criteria["native-end-to-end-terminal-actions"]["passed"]
            )
            self.assertFalse(
                review["validation_summary"][
                    "native_terminal_action_evidence"
                ]["reference_or_projection_actions_accepted"]
            )
            provenance = review["validation_summary"][
                "native_terminal_action_evidence"
            ]["collection_provenance"]
            self.assertEqual(provenance["model"], "gpt-5.6-sol")
            self.assertEqual(provenance["reasoning_effort"], "medium")
            self.assertTrue(provenance["all_collection_files_digest_bound"])
            decoder_provenance = review["validation_summary"][
                "external_decoder_evidence"
            ]["collection_provenance"]
            self.assertEqual(
                decoder_provenance["provenance_mode"],
                "validated_direct_first_party_collection",
            )
            self.assertTrue(
                decoder_provenance["all_collection_files_digest_bound"]
            )
            external_inputs = {
                "decoder_collection_plan": "collection-plan.json",
                "decoder_transport_attempts": "transport-attempts.jsonl",
                "decoder_provider_audit": "provider-audit.jsonl",
                "decoder_judgments": "judgments.jsonl",
                "decoder_execution_manifest": "execution-manifest.json",
            }
            for input_name, filename in external_inputs.items():
                with self.subTest(input_name=input_name):
                    entry = review["inputs"][input_name]
                    self.assertEqual(entry["filename"], filename)
                    self.assertEqual(
                        entry["sha256"],
                        sha256(
                            (external_collection / filename).read_bytes()
                        ).hexdigest(),
                    )
            native_inputs = {
                "native_collection_plan": "collection-plan.json",
                "native_action_requests": "requests.jsonl",
                "native_transport_attempts": "transport-attempts.jsonl",
                "native_provider_audit": "provider-audit.jsonl",
                "native_terminal_actions": "native-actions.jsonl",
                "native_execution_manifest": "execution-manifest.json",
            }
            for input_name, filename in native_inputs.items():
                with self.subTest(input_name=input_name):
                    entry = review["inputs"][input_name]
                    self.assertEqual(entry["filename"], filename)
                    self.assertEqual(
                        entry["sha256"],
                        sha256(
                            (native_collection / filename).read_bytes()
                        ).hexdigest(),
                    )

            openrouter_collection = root / "openrouter-decoder-collection"
            with (
                patch(
                    "cape_loop.cli.OpenRouterChatProvider",
                    side_effect=_openrouter_external_provider,
                ),
                patch.dict(
                    "os.environ",
                    {"OPENROUTER_API_KEY": "test-openrouter-key"},
                    clear=True,
                ),
                redirect_stdout(StringIO()),
            ):
                self.assertEqual(
                    cli_main(
                        [
                            "decoder-study",
                            "execute-openrouter",
                            str(requests_path),
                            str(openrouter_collection),
                            "--execute-live",
                        ]
                    ),
                    0,
                )
            openrouter_judgments_path = (
                openrouter_collection / "judgments.jsonl"
            )
            openrouter_judgments = read_external_decoder_judgments(
                openrouter_judgments_path
            )
            openrouter_source_metadata = {
                judgment.decoder_instance_id: (
                    judgment.decoder_family_id,
                    judgment.judgment_origin,
                    judgment.source_descriptor,
                )
                for judgment in openrouter_judgments
            }
            openrouter_source_ids = sorted(openrouter_source_metadata)
            openrouter_review = DecoderSourceReview.build(
                review_id="gate4-openrouter-source-review",
                responsible_researcher_id="researcher-1",
                reviewed_at="2026-07-26T12:00:00+00:00",
                requests_sha256=sha256(
                    requests_path.read_bytes()
                ).hexdigest(),
                judgments_sha256=sha256(
                    openrouter_judgments_path.read_bytes()
                ).hexdigest(),
                decision="eligible_distinct_sources",
                source_assessments=tuple(
                    DecoderSourceAssessment(
                        source_id,
                        openrouter_source_metadata[source_id][0],
                        openrouter_source_metadata[source_id][1],
                        openrouter_source_metadata[source_id][2],
                        True,
                        "Reviewed family and shared-gateway dependencies.",
                    )
                    for source_id in openrouter_source_ids
                ),
                pair_assessments=tuple(
                    DecoderSourcePairAssessment(
                        left,
                        right,
                        True,
                        "Families are admitted for this reviewed scope; the "
                        "artifact does not claim statistical independence.",
                    )
                    for left, right in combinations(
                        openrouter_source_ids, 2
                    )
                ),
            )
            openrouter_review_path = root / "openrouter-source-review.json"
            openrouter_review_path.write_text(
                canonical_json(openrouter_review.to_dict()) + "\n",
                encoding="utf-8",
            )
            openrouter_output = root / "gate-review-openrouter"
            openrouter_result = import_native_gate_review(
                run_dir=run_dir,
                requests_path=requests_path,
                judgments_path=openrouter_judgments_path,
                truth_labels_path=truth_path,
                native_collection_dir=native_collection,
                source_review_path=openrouter_review_path,
                output_dir=openrouter_output,
                external_collection_dir=openrouter_collection,
                external_collection_provenance_mode=(
                    OPENROUTER_COLLECTION_PROVENANCE
                ),
            )
            self.assertEqual(
                openrouter_result["claim_status"],
                "not_claimed",
            )
            openrouter_valid, openrouter_errors = verify_gate_review(
                openrouter_output,
                source_run_dir=run_dir,
            )
            self.assertTrue(openrouter_valid, openrouter_errors)
            openrouter_artifact = json.loads(
                (openrouter_output / "gate-review.json").read_text(
                    encoding="utf-8"
                )
            )
            openrouter_provenance = openrouter_artifact[
                "validation_summary"
            ]["external_decoder_evidence"]["collection_provenance"]
            self.assertEqual(
                openrouter_provenance["provenance_mode"],
                "selected_openrouter_gateway_collection",
            )
            self.assertTrue(openrouter_provenance["shared_gateway"])
            self.assertFalse(
                openrouter_provenance["first_party_origin_claimed"]
            )
            self.assertFalse(
                openrouter_provenance[
                    "strict_first_party_gate4_eligible"
                ]
            )
            self.assertFalse(
                openrouter_provenance[
                    "statistical_independence_claimed"
                ]
            )

            rejected_openrouter_review = DecoderSourceReview.build(
                review_id="gate4-openrouter-source-review-rejected",
                responsible_researcher_id="researcher-1",
                reviewed_at="2026-07-26T12:00:00+00:00",
                requests_sha256=sha256(
                    requests_path.read_bytes()
                ).hexdigest(),
                judgments_sha256=sha256(
                    openrouter_judgments_path.read_bytes()
                ).hexdigest(),
                decision="eligible_distinct_sources",
                source_assessments=openrouter_review.source_assessments,
                pair_assessments=tuple(
                    DecoderSourcePairAssessment(
                        left,
                        right,
                        False,
                        "Shared-gateway dependency rejected for this scope.",
                    )
                    for left, right in combinations(
                        openrouter_source_ids, 2
                    )
                ),
            )
            rejected_openrouter_review_path = (
                root / "openrouter-source-review-rejected.json"
            )
            rejected_openrouter_review_path.write_text(
                canonical_json(rejected_openrouter_review.to_dict()) + "\n",
                encoding="utf-8",
            )
            rejected_openrouter_output = (
                root / "gate-review-openrouter-rejected"
            )
            with self.assertRaisesRegex(
                ValueError,
                "not reviewed as genuinely distinct",
            ):
                import_native_gate_review(
                    run_dir=run_dir,
                    requests_path=requests_path,
                    judgments_path=openrouter_judgments_path,
                    truth_labels_path=truth_path,
                    native_collection_dir=native_collection,
                    source_review_path=rejected_openrouter_review_path,
                    output_dir=rejected_openrouter_output,
                    external_collection_dir=openrouter_collection,
                    external_collection_provenance_mode=(
                        OPENROUTER_COLLECTION_PROVENANCE
                    ),
                )
            self._assert_no_partial_output(rejected_openrouter_output)

            self.assertEqual(
                before,
                sha256((run_dir / "SHA256SUMS").read_bytes()).hexdigest(),
            )
            run_valid, run_errors = verify_run(run_dir)
            self.assertTrue(run_valid, run_errors)
            with self.assertRaises(FileExistsError):
                import_native_gate_review(
                    run_dir=run_dir,
                    requests_path=requests_path,
                    judgments_path=judgments_path,
                    truth_labels_path=truth_path,
                    native_collection_dir=native_collection,
                    source_review_path=source_review_path,
                    output_dir=output,
                    external_collection_dir=external_collection,
                )

            legacy_actions = native_collection / "native-actions.jsonl"
            with self.assertRaisesRegex(
                ValueError,
                "complete native action collection directory",
            ):
                import_native_gate_review(
                    run_dir=run_dir,
                    requests_path=requests_path,
                    judgments_path=judgments_path,
                    truth_labels_path=truth_path,
                    native_collection_dir=legacy_actions,
                    source_review_path=source_review_path,
                    output_dir=root / "legacy-review",
                    external_collection_dir=external_collection,
                )
            self.assertFalse((root / "legacy-review").exists())

            incomplete_collection = root / "incomplete-native-collection"
            shutil.copytree(native_collection, incomplete_collection)
            (incomplete_collection / "provider-audit.jsonl").unlink()
            with self.assertRaisesRegex(ValueError, "missing or unexpected"):
                import_native_gate_review(
                    run_dir=run_dir,
                    requests_path=requests_path,
                    judgments_path=judgments_path,
                    truth_labels_path=truth_path,
                    native_collection_dir=incomplete_collection,
                    source_review_path=source_review_path,
                    output_dir=root / "incomplete-review",
                    external_collection_dir=external_collection,
                )
            self.assertFalse((root / "incomplete-review").exists())

            wrong_model_collection = root / "wrong-model-collection"
            shutil.copytree(native_collection, wrong_model_collection)
            wrong_plan_path = wrong_model_collection / "collection-plan.json"
            wrong_plan = json.loads(
                wrong_plan_path.read_text(encoding="utf-8")
            )
            wrong_plan["model"] = "gpt-5.6-terra"
            wrong_plan["collection_config"]["model"] = "gpt-5.6-terra"
            wrong_plan_path.write_text(
                canonical_json(wrong_plan) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError,
                "gpt-5.6-sol/medium",
            ):
                import_native_gate_review(
                    run_dir=run_dir,
                    requests_path=requests_path,
                    judgments_path=judgments_path,
                    truth_labels_path=truth_path,
                    native_collection_dir=wrong_model_collection,
                    source_review_path=source_review_path,
                    output_dir=root / "wrong-model-review",
                    external_collection_dir=external_collection,
                )
            self.assertFalse((root / "wrong-model-review").exists())

            high_native_budget = root / "high-native-budget"
            shutil.copytree(native_collection, high_native_budget)
            high_native_plan_path = (
                high_native_budget / "collection-plan.json"
            )
            high_native_plan = json.loads(
                high_native_plan_path.read_text(encoding="utf-8")
            )
            high_native_plan["collection_config"]["max_requests"] = 901
            high_native_plan_path.write_text(
                canonical_json(high_native_plan) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "approved Gate 4 ceilings"):
                import_native_gate_review(
                    run_dir=run_dir,
                    requests_path=requests_path,
                    judgments_path=judgments_path,
                    truth_labels_path=truth_path,
                    native_collection_dir=high_native_budget,
                    source_review_path=source_review_path,
                    output_dir=root / "high-native-budget-review",
                    external_collection_dir=external_collection,
                )

            overcommitted_native = root / "overcommitted-native"
            shutil.copytree(native_collection, overcommitted_native)
            overcommitted_plan_path = (
                overcommitted_native / "collection-plan.json"
            )
            overcommitted_plan = json.loads(
                overcommitted_plan_path.read_text(encoding="utf-8")
            )
            raw_collection_config = dict(
                overcommitted_plan["collection_config"]
            )
            raw_collection_config["max_total_tokens"] = 1
            overcommitted_config = OpenAIProviderConfig(
                model=raw_collection_config["model"],
                reasoning_effort=raw_collection_config[
                    "reasoning_effort"
                ],
                api_key_env=raw_collection_config["api_key_env"],
                base_url=raw_collection_config["base_url"],
                allow_custom_base_url=raw_collection_config[
                    "allow_custom_base_url"
                ],
                timeout_seconds=raw_collection_config[
                    "timeout_seconds"
                ],
                max_retries=raw_collection_config["max_retries"],
                initial_backoff_seconds=raw_collection_config[
                    "initial_backoff_seconds"
                ],
                max_backoff_seconds=raw_collection_config[
                    "max_backoff_seconds"
                ],
                jitter_fraction=raw_collection_config[
                    "jitter_fraction"
                ],
                max_output_tokens=raw_collection_config[
                    "max_output_tokens"
                ],
                max_requests=raw_collection_config["max_requests"],
                max_total_tokens=raw_collection_config[
                    "max_total_tokens"
                ],
                live_execution=False,
            )
            overcommitted_requests = build_native_action_requests(run_dir)
            overcommitted_provider = OpenAINativeActionProvider(
                overcommitted_config
            )
            overcommitted_prepared = tuple(
                overcommitted_provider.prepare(request)
                for request in overcommitted_requests
            )
            overcommitted_plan = _build_collection_plan(
                run_dir,
                overcommitted_config,
                overcommitted_requests,
                overcommitted_prepared,
            )
            self.assertFalse(
                overcommitted_plan["within_declared_budget"]
            )
            overcommitted_plan_path.write_text(
                canonical_json(overcommitted_plan) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError,
                "retry-expanded plan exceeds",
            ):
                import_native_gate_review(
                    run_dir=run_dir,
                    requests_path=requests_path,
                    judgments_path=judgments_path,
                    truth_labels_path=truth_path,
                    native_collection_dir=overcommitted_native,
                    source_review_path=source_review_path,
                    output_dir=root / "overcommitted-native-review",
                    external_collection_dir=external_collection,
                )

            wrong_native_ordinal = root / "wrong-native-ordinal"
            shutil.copytree(native_collection, wrong_native_ordinal)
            wrong_native_audit_path = (
                wrong_native_ordinal / "provider-audit.jsonl"
            )
            wrong_native_audits = [
                json.loads(line)
                for line in wrong_native_audit_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            tampered_request_id = wrong_native_audits[0]["request_id"]
            wrong_native_audits[0]["attempts"] = 2
            _write_jsonl(wrong_native_audit_path, wrong_native_audits)
            wrong_native_attempt_path = (
                wrong_native_ordinal / "transport-attempts.jsonl"
            )
            wrong_native_attempts = [
                json.loads(line)
                for line in wrong_native_attempt_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            for attempt in wrong_native_attempts:
                embedded = attempt.get("provider_audit")
                if (
                    isinstance(embedded, dict)
                    and embedded.get("request_id") == tampered_request_id
                ):
                    embedded["attempts"] = 2
            _write_jsonl(wrong_native_attempt_path, wrong_native_attempts)
            wrong_native_manifest_path = (
                wrong_native_ordinal / "execution-manifest.json"
            )
            wrong_native_manifest = json.loads(
                wrong_native_manifest_path.read_text(encoding="utf-8")
            )
            wrong_native_manifest["provider_audit_sha256"] = sha256(
                wrong_native_audit_path.read_bytes()
            ).hexdigest()
            wrong_native_manifest["transport_attempts_sha256"] = sha256(
                wrong_native_attempt_path.read_bytes()
            ).hexdigest()
            wrong_native_manifest_path.write_text(
                canonical_json(wrong_native_manifest) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError,
                "attempt count does not match",
            ):
                import_native_gate_review(
                    run_dir=run_dir,
                    requests_path=requests_path,
                    judgments_path=judgments_path,
                    truth_labels_path=truth_path,
                    native_collection_dir=wrong_native_ordinal,
                    source_review_path=source_review_path,
                    output_dir=root / "wrong-native-ordinal-review",
                    external_collection_dir=external_collection,
                )

            wrong_native_raw_actions = root / "wrong-native-raw-actions"
            shutil.copytree(native_collection, wrong_native_raw_actions)
            wrong_raw_audit_path = (
                wrong_native_raw_actions / "provider-audit.jsonl"
            )
            wrong_raw_audits = [
                json.loads(line)
                for line in wrong_raw_audit_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            wrong_raw_audit = wrong_raw_audits[0]
            wrong_raw_request_id = wrong_raw_audit["request_id"]
            wrong_raw_payload = json.loads(
                wrong_raw_audit["raw_response"]["output_text"]
            )
            direct_action = next(
                action
                for action in wrong_raw_payload["actions"]
                if action["question_type"] == "direct_preference_probe"
            )
            direct_action["declared_direction"] *= -1
            wrong_raw_audit["raw_response"]["output_text"] = json.dumps(
                wrong_raw_payload
            )
            _write_jsonl(wrong_raw_audit_path, wrong_raw_audits)
            wrong_raw_attempt_path = (
                wrong_native_raw_actions / "transport-attempts.jsonl"
            )
            wrong_raw_attempts = [
                json.loads(line)
                for line in wrong_raw_attempt_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            for attempt in wrong_raw_attempts:
                embedded = attempt.get("provider_audit")
                if (
                    isinstance(embedded, dict)
                    and embedded.get("request_id") == wrong_raw_request_id
                ):
                    attempt["provider_audit"] = wrong_raw_audit
                    attempt["response_record"] = wrong_raw_audit[
                        "raw_response"
                    ]
            _write_jsonl(wrong_raw_attempt_path, wrong_raw_attempts)
            wrong_raw_manifest_path = (
                wrong_native_raw_actions / "execution-manifest.json"
            )
            wrong_raw_manifest = json.loads(
                wrong_raw_manifest_path.read_text(encoding="utf-8")
            )
            wrong_raw_manifest["provider_audit_sha256"] = sha256(
                wrong_raw_audit_path.read_bytes()
            ).hexdigest()
            wrong_raw_manifest["transport_attempts_sha256"] = sha256(
                wrong_raw_attempt_path.read_bytes()
            ).hexdigest()
            wrong_raw_manifest_path.write_text(
                canonical_json(wrong_raw_manifest) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError,
                "raw-response/action-record mismatch",
            ):
                import_native_gate_review(
                    run_dir=run_dir,
                    requests_path=requests_path,
                    judgments_path=judgments_path,
                    truth_labels_path=truth_path,
                    native_collection_dir=wrong_native_raw_actions,
                    source_review_path=source_review_path,
                    output_dir=root / "wrong-native-raw-actions-review",
                    external_collection_dir=external_collection,
                )

            incomplete_external = root / "incomplete-external-collection"
            shutil.copytree(external_collection, incomplete_external)
            (incomplete_external / "provider-audit.jsonl").unlink()
            with self.assertRaisesRegex(ValueError, "missing or unexpected"):
                import_native_gate_review(
                    run_dir=run_dir,
                    requests_path=requests_path,
                    judgments_path=judgments_path,
                    truth_labels_path=truth_path,
                    native_collection_dir=native_collection,
                    source_review_path=source_review_path,
                    output_dir=root / "incomplete-external-review",
                    external_collection_dir=incomplete_external,
                )

            high_external_budget = root / "high-external-budget"
            shutil.copytree(external_collection, high_external_budget)
            high_external_plan_path = (
                high_external_budget / "collection-plan.json"
            )
            high_external_plan = json.loads(
                high_external_plan_path.read_text(encoding="utf-8")
            )
            high_external_plan["sources"][0]["max_output_tokens"] = 1_025
            high_external_plan_path.write_text(
                canonical_json(high_external_plan) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "approved per-source"):
                import_native_gate_review(
                    run_dir=run_dir,
                    requests_path=requests_path,
                    judgments_path=(
                        high_external_budget / "judgments.jsonl"
                    ),
                    truth_labels_path=truth_path,
                    native_collection_dir=native_collection,
                    source_review_path=source_review_path,
                    output_dir=root / "high-external-budget-review",
                    external_collection_dir=high_external_budget,
                )

            mismatched_judgments = root / "mismatched-judgments.jsonl"
            mismatched_judgments.write_bytes(
                judgments_path.read_bytes() + b"\n"
            )
            with self.assertRaisesRegex(ValueError, "byte-identical"):
                import_native_gate_review(
                    run_dir=run_dir,
                    requests_path=requests_path,
                    judgments_path=mismatched_judgments,
                    truth_labels_path=truth_path,
                    native_collection_dir=native_collection,
                    source_review_path=source_review_path,
                    output_dir=root / "mismatched-judgments-review",
                    external_collection_dir=external_collection,
                )

            missing_external_digest = root / "missing-external-digest"
            shutil.copytree(external_collection, missing_external_digest)
            missing_digest_audit_path = (
                missing_external_digest / "provider-audit.jsonl"
            )
            missing_digest_audits = [
                json.loads(line)
                for line in missing_digest_audit_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            missing_digest_request_id = missing_digest_audits[0][
                "request_id"
            ]
            missing_digest_audits[0]["llm_response"][
                "raw_response_sha256"
            ] = None
            _write_jsonl(missing_digest_audit_path, missing_digest_audits)
            missing_digest_attempt_path = (
                missing_external_digest / "transport-attempts.jsonl"
            )
            missing_digest_attempts = [
                json.loads(line)
                for line in missing_digest_attempt_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            for attempt in missing_digest_attempts:
                embedded = attempt.get("provider_audit")
                if (
                    isinstance(embedded, dict)
                    and embedded.get("request_id")
                    == missing_digest_request_id
                ):
                    embedded["llm_response"][
                        "raw_response_sha256"
                    ] = None
                    attempt["response_body_sha256"] = None
            _write_jsonl(
                missing_digest_attempt_path,
                missing_digest_attempts,
            )
            missing_digest_manifest_path = (
                missing_external_digest / "execution-manifest.json"
            )
            missing_digest_manifest = json.loads(
                missing_digest_manifest_path.read_text(encoding="utf-8")
            )
            missing_digest_manifest["provider_audit_sha256"] = sha256(
                missing_digest_audit_path.read_bytes()
            ).hexdigest()
            missing_digest_manifest[
                "transport_attempts_sha256"
            ] = sha256(
                missing_digest_attempt_path.read_bytes()
            ).hexdigest()
            missing_digest_manifest_path.write_text(
                canonical_json(missing_digest_manifest) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError,
                "raw_response_sha256|response metadata",
            ):
                import_native_gate_review(
                    run_dir=run_dir,
                    requests_path=requests_path,
                    judgments_path=(
                        missing_external_digest / "judgments.jsonl"
                    ),
                    truth_labels_path=truth_path,
                    native_collection_dir=native_collection,
                    source_review_path=source_review_path,
                    output_dir=root / "missing-external-digest-review",
                    external_collection_dir=missing_external_digest,
                )

            with self.assertRaisesRegex(
                ValueError,
                "complete distinct-decoder collection directory",
            ):
                import_native_gate_review(
                    run_dir=run_dir,
                    requests_path=requests_path,
                    judgments_path=judgments_path,
                    truth_labels_path=truth_path,
                    native_collection_dir=native_collection,
                    source_review_path=source_review_path,
                    output_dir=root / "standalone-decoder-review",
                    external_collection_dir=judgments_path,
                )

            with self.assertRaisesRegex(
                ValueError,
                "choose exactly one external decoder provenance mode",
            ):
                import_native_gate_review(
                    run_dir=run_dir,
                    requests_path=requests_path,
                    judgments_path=judgments_path,
                    truth_labels_path=truth_path,
                    native_collection_dir=native_collection,
                    source_review_path=source_review_path,
                    output_dir=root / "implicit-generic-review",
                )

            generic_judgments = [
                ExternalDecoderJudgment(
                    request_id=request.request_id,
                    request_sha256=request.request_sha256,
                    decoder_instance_id=instance,
                    decoder_family_id=family,
                    judgment_origin="human_annotator",
                    source_descriptor=descriptor,
                    blind_to_system_identity=True,
                    blind_to_latent_truth=True,
                    probabilities=_ROWS,
                )
                for request in requests
                for instance, family, descriptor in (
                    (
                        "human-decoder-a",
                        "human-panel-a",
                        "blinded human panel A",
                    ),
                    (
                        "human-decoder-b",
                        "human-panel-b",
                        "blinded human panel B",
                    ),
                )
            ]
            generic_judgments_path = root / "generic-judgments.jsonl"
            _write_jsonl(
                generic_judgments_path,
                [row.to_dict() for row in generic_judgments],
            )
            generic_review = DecoderSourceReview.build(
                review_id="gate4-generic-source-review",
                responsible_researcher_id="researcher-1",
                reviewed_at="2026-07-26T12:00:00+00:00",
                requests_sha256=sha256(
                    requests_path.read_bytes()
                ).hexdigest(),
                judgments_sha256=sha256(
                    generic_judgments_path.read_bytes()
                ).hexdigest(),
                decision="eligible_distinct_sources",
                source_assessments=(
                    DecoderSourceAssessment(
                        "human-decoder-a",
                        "human-panel-a",
                        "human_annotator",
                        "blinded human panel A",
                        True,
                        "Reviewed panel recruitment and adjudication.",
                    ),
                    DecoderSourceAssessment(
                        "human-decoder-b",
                        "human-panel-b",
                        "human_annotator",
                        "blinded human panel B",
                        True,
                        "Reviewed panel recruitment and adjudication.",
                    ),
                ),
                pair_assessments=(
                    DecoderSourcePairAssessment(
                        "human-decoder-a",
                        "human-decoder-b",
                        True,
                        "Panels were reviewed as distinct for this scope.",
                    ),
                ),
            )
            generic_review_path = root / "generic-source-review.json"
            generic_review_path.write_text(
                canonical_json(generic_review.to_dict()) + "\n",
                encoding="utf-8",
            )
            generic_output = root / "generic-gate-review"
            import_native_gate_review(
                run_dir=run_dir,
                requests_path=requests_path,
                judgments_path=generic_judgments_path,
                truth_labels_path=truth_path,
                native_collection_dir=native_collection,
                source_review_path=generic_review_path,
                output_dir=generic_output,
                allow_reviewed_generic_decoders=True,
            )
            generic_valid, generic_errors = verify_gate_review(
                generic_output
            )
            self.assertTrue(generic_valid, generic_errors)
            generic_artifact = json.loads(
                (generic_output / "gate-review.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertNotIn(
                "decoder_collection_plan",
                generic_artifact["inputs"],
            )
            self.assertEqual(
                generic_artifact["validation_summary"][
                    "external_decoder_evidence"
                ]["collection_provenance"]["provenance_mode"],
                "reviewed_generic_import",
            )

            with self.assertRaisesRegex(
                ValueError,
                "cannot equal or be inside the native action collection",
            ):
                import_native_gate_review(
                    run_dir=run_dir,
                    requests_path=requests_path,
                    judgments_path=judgments_path,
                    truth_labels_path=truth_path,
                    native_collection_dir=native_collection,
                    source_review_path=source_review_path,
                    output_dir=native_collection / "review",
                    external_collection_dir=external_collection,
                )
            self.assertFalse((native_collection / "review").exists())

            with self.assertRaisesRegex(
                ValueError,
                "cannot equal or be inside the external decoder collection",
            ):
                import_native_gate_review(
                    run_dir=run_dir,
                    requests_path=requests_path,
                    judgments_path=judgments_path,
                    truth_labels_path=truth_path,
                    native_collection_dir=native_collection,
                    source_review_path=source_review_path,
                    output_dir=external_collection / "review",
                    external_collection_dir=external_collection,
                )
            self.assertFalse((external_collection / "review").exists())

            external_lock_descriptor = os.open(
                external_collection / ".external-decoder-command.lock",
                os.O_RDWR,
            )
            external_lock_acquired = False
            try:
                external_lock_acquired = try_file_lock(
                    external_lock_descriptor
                )
                self.assertTrue(external_lock_acquired)
                with self.assertRaisesRegex(
                    ValueError,
                    "currently locked by a collector",
                ):
                    import_native_gate_review(
                        run_dir=run_dir,
                        requests_path=requests_path,
                        judgments_path=judgments_path,
                        truth_labels_path=truth_path,
                        native_collection_dir=native_collection,
                        source_review_path=source_review_path,
                        output_dir=root / "external-locked-review",
                        external_collection_dir=external_collection,
                    )
            finally:
                if external_lock_acquired:
                    unlock_file(external_lock_descriptor)
                os.close(external_lock_descriptor)
            self.assertFalse((root / "external-locked-review").exists())

            lock_descriptor = os.open(
                native_collection / ".collection.lock",
                os.O_RDWR,
            )
            native_lock_acquired = False
            try:
                native_lock_acquired = try_file_lock(lock_descriptor)
                self.assertTrue(native_lock_acquired)
                with self.assertRaisesRegex(
                    ValueError,
                    "currently locked by a collector",
                ):
                    import_native_gate_review(
                        run_dir=run_dir,
                        requests_path=requests_path,
                        judgments_path=judgments_path,
                        truth_labels_path=truth_path,
                        native_collection_dir=native_collection,
                        source_review_path=source_review_path,
                        output_dir=root / "locked-review",
                        external_collection_dir=external_collection,
                    )
            finally:
                if native_lock_acquired:
                    unlock_file(lock_descriptor)
                os.close(lock_descriptor)
            self.assertFalse((root / "locked-review").exists())

            atomic_output = root / "atomic-visibility-review"
            original_write_json = gate_review_module._write_json_durable
            output_seen_during_write: list[bool] = []

            def observe_staged_write(path: Path, value: object) -> None:
                output_seen_during_write.append(atomic_output.exists())
                original_write_json(path, value)

            with patch.object(
                gate_review_module,
                "_write_json_durable",
                side_effect=observe_staged_write,
            ):
                import_native_gate_review(
                    run_dir=run_dir,
                    requests_path=requests_path,
                    judgments_path=judgments_path,
                    truth_labels_path=truth_path,
                    native_collection_dir=native_collection,
                    source_review_path=source_review_path,
                    output_dir=atomic_output,
                    external_collection_dir=external_collection,
                )
            self.assertTrue(output_seen_during_write)
            self.assertFalse(any(output_seen_during_write))
            self.assertTrue(atomic_output.is_dir())

            failed_output = root / "injected-write-failure-review"
            write_count = 0

            def fail_staged_write(path: Path, value: object) -> None:
                nonlocal write_count
                original_write_json(path, value)
                write_count += 1
                if write_count == 2:
                    raise OSError("injected Gate 4 staged write failure")

            with patch.object(
                gate_review_module,
                "_write_json_durable",
                side_effect=fail_staged_write,
            ):
                with self.assertRaisesRegex(
                    OSError,
                    "injected Gate 4 staged write failure",
                ):
                    import_native_gate_review(
                        run_dir=run_dir,
                        requests_path=requests_path,
                        judgments_path=judgments_path,
                        truth_labels_path=truth_path,
                        native_collection_dir=native_collection,
                        source_review_path=source_review_path,
                        output_dir=failed_output,
                        external_collection_dir=external_collection,
                    )
            self._assert_no_partial_output(failed_output)

            raced_output = root / "destination-race-review"
            destination_created = False

            def create_destination(path: Path, value: object) -> None:
                nonlocal destination_created
                if not destination_created:
                    raced_output.mkdir()
                    (raced_output / "owner-marker.txt").write_text(
                        "unrelated owner\n",
                        encoding="utf-8",
                    )
                    destination_created = True
                original_write_json(path, value)

            with patch.object(
                gate_review_module,
                "_write_json_durable",
                side_effect=create_destination,
            ):
                with self.assertRaisesRegex(
                    FileExistsError,
                    "already exists",
                ):
                    import_native_gate_review(
                        run_dir=run_dir,
                        requests_path=requests_path,
                        judgments_path=judgments_path,
                        truth_labels_path=truth_path,
                        native_collection_dir=native_collection,
                        source_review_path=source_review_path,
                        output_dir=raced_output,
                        external_collection_dir=external_collection,
                    )
            self.assertEqual(
                (raced_output / "owner-marker.txt").read_text(
                    encoding="utf-8"
                ),
                "unrelated owner\n",
            )
            self.assertFalse(
                (
                    raced_output.parent
                    / f".{raced_output.name}.gate4-review.lock"
                ).exists()
            )
            self.assertEqual(
                list(
                    raced_output.parent.glob(
                        f".{raced_output.name}.*.staging"
                    )
                ),
                [],
            )

            review_race_input = root / "source-review-race.json"
            review_race_input.write_bytes(source_review_path.read_bytes())
            review_race_output = root / "source-review-race-output"
            review_input_mutated = False

            def mutate_review_input(path: Path, value: object) -> None:
                nonlocal review_input_mutated
                if not review_input_mutated:
                    review_race_input.write_bytes(
                        review_race_input.read_bytes() + b"\n"
                    )
                    review_input_mutated = True
                original_write_json(path, value)

            with patch.object(
                gate_review_module,
                "_write_json_durable",
                side_effect=mutate_review_input,
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "decoder source review changed",
                ):
                    import_native_gate_review(
                        run_dir=run_dir,
                        requests_path=requests_path,
                        judgments_path=judgments_path,
                        truth_labels_path=truth_path,
                        native_collection_dir=native_collection,
                        source_review_path=review_race_input,
                        output_dir=review_race_output,
                        external_collection_dir=external_collection,
                    )
            self._assert_no_partial_output(review_race_output)

            native_race_collection = root / "native-collection-race"
            shutil.copytree(native_collection, native_race_collection)
            native_race_output = root / "native-collection-race-output"
            native_collection_mutated = False

            def mutate_native_collection(path: Path, value: object) -> None:
                nonlocal native_collection_mutated
                if not native_collection_mutated:
                    plan_path = (
                        native_race_collection / "collection-plan.json"
                    )
                    plan_path.write_bytes(plan_path.read_bytes() + b"\n")
                    native_collection_mutated = True
                original_write_json(path, value)

            with patch.object(
                gate_review_module,
                "_write_json_durable",
                side_effect=mutate_native_collection,
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "native action collection changed",
                ):
                    import_native_gate_review(
                        run_dir=run_dir,
                        requests_path=requests_path,
                        judgments_path=judgments_path,
                        truth_labels_path=truth_path,
                        native_collection_dir=native_race_collection,
                        source_review_path=source_review_path,
                        output_dir=native_race_output,
                        external_collection_dir=external_collection,
                    )
            self._assert_no_partial_output(native_race_output)

            copied_run_parent = root / "source-run-race-copy"
            copied_run_parent.mkdir()
            copied_run = copied_run_parent / run_dir.name
            shutil.copytree(run_dir, copied_run)
            source_race_output = root / "source-run-race-output"
            source_run_mutated = False

            def mutate_source_run(path: Path, value: object) -> None:
                nonlocal source_run_mutated
                if not source_run_mutated:
                    copied_request_path = (
                        copied_run
                        / "decoder"
                        / "external-requests.jsonl"
                    )
                    copied_request_path.write_bytes(
                        copied_request_path.read_bytes() + b"\n"
                    )
                    source_run_mutated = True
                original_write_json(path, value)

            with patch.object(
                gate_review_module,
                "_write_json_durable",
                side_effect=mutate_source_run,
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "source run changed",
                ):
                    import_native_gate_review(
                        run_dir=copied_run,
                        requests_path=(
                            copied_run
                            / "decoder"
                            / "external-requests.jsonl"
                        ),
                        judgments_path=judgments_path,
                        truth_labels_path=(
                            copied_run
                            / "decoder"
                            / "truth-labels.researcher-only.jsonl"
                        ),
                        native_collection_dir=native_collection,
                        source_review_path=source_review_path,
                        output_dir=source_race_output,
                        external_collection_dir=external_collection,
                    )
            self._assert_no_partial_output(source_race_output)

            staged_failure_output = root / "staged-verifier-failure-review"
            with patch.object(
                gate_review_module,
                "verify_gate_review",
                return_value=(
                    False,
                    ("injected staged verification failure",),
                ),
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "staged Gate 4 review failed verification",
                ):
                    import_native_gate_review(
                        run_dir=run_dir,
                        requests_path=requests_path,
                        judgments_path=judgments_path,
                        truth_labels_path=truth_path,
                        native_collection_dir=native_collection,
                        source_review_path=source_review_path,
                        output_dir=staged_failure_output,
                        external_collection_dir=external_collection,
                    )
            self._assert_no_partial_output(staged_failure_output)

            output_locked = root / "output-locked-review"
            output_lock = (
                output_locked.parent
                / f".{output_locked.name}.gate4-review.lock"
            )
            output_lock.write_text("held\n", encoding="utf-8")
            try:
                with self.assertRaisesRegex(FileExistsError, "is locked"):
                    import_native_gate_review(
                        run_dir=run_dir,
                        requests_path=requests_path,
                        judgments_path=judgments_path,
                        truth_labels_path=truth_path,
                        native_collection_dir=native_collection,
                        source_review_path=source_review_path,
                        output_dir=output_locked,
                        external_collection_dir=external_collection,
                    )
                self.assertFalse(output_locked.exists())
            finally:
                output_lock.unlink()

            run_link = root / "source-run-link"
            run_link.symlink_to(run_dir, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "safe directory"):
                import_native_gate_review(
                    run_dir=run_link,
                    requests_path=requests_path,
                    judgments_path=judgments_path,
                    truth_labels_path=truth_path,
                    native_collection_dir=native_collection,
                    source_review_path=source_review_path,
                    output_dir=root / "source-run-link-review",
                    external_collection_dir=external_collection,
                )
            review_link = root / "source-review-link.json"
            review_link.symlink_to(source_review_path)
            with self.assertRaisesRegex(ValueError, "safe regular file"):
                import_native_gate_review(
                    run_dir=run_dir,
                    requests_path=requests_path,
                    judgments_path=judgments_path,
                    truth_labels_path=truth_path,
                    native_collection_dir=native_collection,
                    source_review_path=review_link,
                    output_dir=root / "source-review-link-output",
                    external_collection_dir=external_collection,
                )
            output_link = root / "gate-review-output-link"
            output_link.symlink_to(
                root / "uncreated-gate-review-target",
                target_is_directory=True,
            )
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                import_native_gate_review(
                    run_dir=run_dir,
                    requests_path=requests_path,
                    judgments_path=judgments_path,
                    truth_labels_path=truth_path,
                    native_collection_dir=native_collection,
                    source_review_path=source_review_path,
                    output_dir=output_link,
                    external_collection_dir=external_collection,
                )
            inside_run_output = run_dir / "forbidden-gate-review"
            with self.assertRaisesRegex(
                ValueError,
                "inside the completed source run",
            ):
                import_native_gate_review(
                    run_dir=run_dir,
                    requests_path=requests_path,
                    judgments_path=judgments_path,
                    truth_labels_path=truth_path,
                    native_collection_dir=native_collection,
                    source_review_path=source_review_path,
                    output_dir=inside_run_output,
                    external_collection_dir=external_collection,
                )
            self.assertFalse(inside_run_output.exists())
            self.assertFalse(
                (
                    run_dir
                    / ".forbidden-gate-review.gate4-review.lock"
                ).exists()
            )

            symlinked_review = root / "symlinked-review-artifact"
            shutil.copytree(atomic_output, symlinked_review)
            outside_payload = root / "outside-gate-review.json"
            (symlinked_review / "gate-review.json").replace(
                outside_payload
            )
            (symlinked_review / "gate-review.json").symlink_to(
                outside_payload
            )
            symlink_valid, symlink_errors = verify_gate_review(
                symlinked_review
            )
            self.assertFalse(symlink_valid)
            self.assertTrue(
                any("symlink" in error for error in symlink_errors),
                symlink_errors,
            )


if __name__ == "__main__":
    unittest.main()
