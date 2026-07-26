from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

from cape_loop.artifacts import canonical_json, verify_run
from cape_loop.config import (
    AppConfig,
    ExperimentSection,
    InferenceSection,
    RunSection,
)
from cape_loop.decoder_study import (
    ExternalDecoderJudgment,
    read_external_decoder_requests,
)
from cape_loop.gate_review import (
    DecoderSourceAssessment,
    DecoderSourcePairAssessment,
    DecoderSourceReview,
    NativeTerminalActionRecord,
    import_native_gate_review,
    verify_gate_review,
)
from cape_loop.heldout import TerminalAction
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


class GateReviewIntegrationTests(unittest.TestCase):
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
            judgments: list[ExternalDecoderJudgment] = []
            for request in requests:
                for instance, family, descriptor in (
                    ("decoder-a", "family-a", "independent source A"),
                    ("decoder-b", "family-b", "independent source B"),
                ):
                    judgments.append(
                        ExternalDecoderJudgment(
                            request_id=request.request_id,
                            request_sha256=request.request_sha256,
                            decoder_instance_id=instance,
                            decoder_family_id=family,
                            judgment_origin="external_model",
                            source_descriptor=descriptor,
                            blind_to_system_identity=True,
                            blind_to_latent_truth=True,
                            probabilities=_ROWS,
                        )
                    )
            judgments_path = root / "judgments.jsonl"
            _write_jsonl(
                judgments_path,
                [row.to_dict() for row in judgments],
            )

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
                source_assessments=(
                    DecoderSourceAssessment(
                        "decoder-a",
                        "family-a",
                        "external_model",
                        "independent source A",
                        True,
                        "Reviewed provider, training, prompt, and adjudication.",
                    ),
                    DecoderSourceAssessment(
                        "decoder-b",
                        "family-b",
                        "external_model",
                        "independent source B",
                        True,
                        "Reviewed provider, training, prompt, and adjudication.",
                    ),
                ),
                pair_assessments=(
                    DecoderSourcePairAssessment(
                        "decoder-a",
                        "decoder-b",
                        True,
                        "Responsible researcher found no disqualifying shared "
                        "generation or adjudication dependency.",
                    ),
                ),
            )
            source_review_path = root / "source-review.json"
            source_review_path.write_text(
                canonical_json(source_review.to_dict()) + "\n",
                encoding="utf-8",
            )

            suites = {
                row["domain_id"]: row
                for row in (
                    json.loads(line)
                    for line in (
                        run_dir
                        / "events"
                        / "experiment-b-held-out-terminal-suites.jsonl"
                    )
                    .read_text(encoding="utf-8")
                    .splitlines()
                )
            }
            trajectories = [
                json.loads(line)
                for line in (
                    run_dir / "events" / "experiment-b-trajectories.jsonl"
                )
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            eligible = [
                row
                for row in trajectories
                if row["updater_id"]
                in {"semantic_memory", "provenance_linked_memory"}
                and row["policy_id"] == "soft_profile_conditioned"
                and row["initial_profile_condition"] == "incorrect"
            ]
            action_records = []
            for trajectory in eligible:
                suite = suites[trajectory["domain_id"]]
                actions = []
                for item in suite["items"]:
                    if item["question_type"] == "direct_preference_probe":
                        actions.append(
                            TerminalAction(
                                item_id=item["item_id"],
                                item_sha256=item["item_sha256"],
                                wording_template_id=(
                                    item["wording_template_id"]
                                ),
                                question_type=item["question_type"],
                                declared_direction=1,
                            )
                        )
                    else:
                        actions.append(
                            TerminalAction(
                                item_id=item["item_id"],
                                item_sha256=item["item_sha256"],
                                wording_template_id=(
                                    item["wording_template_id"]
                                ),
                                question_type=item["question_type"],
                                selected_option_id=(
                                    item["options"][0]["option_id"]
                                ),
                            )
                        )
                action_records.append(
                    NativeTerminalActionRecord.build(
                        record_id=(
                            "recorded-native:" + trajectory["trajectory_id"]
                        ),
                        trajectory_id=trajectory["trajectory_id"],
                        domain_id=trajectory["domain_id"],
                        updater_id=trajectory["updater_id"],
                        native_state_id=trajectory[
                            "terminal_native_state"
                        ]["state_id"],
                        native_system_id="studied-native-system",
                        native_system_version="v1-frozen",
                        suite_id=suite["suite_id"],
                        suite_sha256=suite["suite_sha256"],
                        action_execution_mode="recorded_replay",
                        execution_trace_sha256=sha256(
                            trajectory["trajectory_id"].encode("utf-8")
                        ).hexdigest(),
                        recorded_at="2026-07-26T12:00:00+00:00",
                        actions=actions,
                    )
                )
            actions_path = root / "native-actions.jsonl"
            _write_jsonl(
                actions_path,
                [row.to_dict() for row in action_records],
            )

            output = root / "gate-review"
            result = import_native_gate_review(
                run_dir=run_dir,
                requests_path=requests_path,
                judgments_path=judgments_path,
                truth_labels_path=truth_path,
                actions_path=actions_path,
                source_review_path=source_review_path,
                output_dir=output,
            )
            self.assertEqual(result["claim_status"], "not_claimed")
            valid_review, review_errors = verify_gate_review(output)
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
                    actions_path=actions_path,
                    source_review_path=source_review_path,
                    output_dir=output,
                )

            incomplete_actions = root / "incomplete-actions.jsonl"
            _write_jsonl(
                incomplete_actions,
                [action_records[0].to_dict()],
            )
            with self.assertRaisesRegex(ValueError, "cover eligible"):
                import_native_gate_review(
                    run_dir=run_dir,
                    requests_path=requests_path,
                    judgments_path=judgments_path,
                    truth_labels_path=truth_path,
                    actions_path=incomplete_actions,
                    source_review_path=source_review_path,
                    output_dir=root / "incomplete-review",
                )
            self.assertFalse((root / "incomplete-review").exists())


if __name__ == "__main__":
    unittest.main()
