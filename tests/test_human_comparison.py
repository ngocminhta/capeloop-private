from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from unittest.mock import patch
import json
import unittest

import cape_loop.cli as cli_module
from cape_loop.cli import main
from cape_loop.decoder_study import HumanCollectionRecord
from cape_loop.human_comparison import (
    EVIDENCE_METRIC,
    ModelEvidenceStrength,
    analyze_h8_human_model_comparison,
    convert_experiment_a_metrics_to_model_evidence,
)
from cape_loop.schema_export import SCHEMAS


_DIGEST = "a" * 64


class HumanModelComparisonTests(unittest.TestCase):
    def _fixture(self):
        codebook = {"assignment-1": {}}
        human_rows = []
        human_ratings = {
            "balanced": 7,
            "restricted": 2,
            "default": 3,
            "suggested": 3,
        }
        for condition_index, (condition, rating) in enumerate(
            human_ratings.items(),
            start=1,
        ):
            display_id = f"item-{condition_index:04d}"
            codebook["assignment-1"][display_id] = {
                "item_id": f"item:{condition}",
                "scenario_id": "scenario-1",
                "condition": condition,
            }
            for participant in range(8):
                human_rows.append(
                    HumanCollectionRecord(
                        participant_code=f"participant-{participant}",
                        assignment_id="assignment-1",
                        assignment_protocol_id="assignment-v1",
                        display_id=display_id,
                        rating=rating,
                        response_time_ms=1000,
                        consent_version="consent-v1",
                        consented=True,
                        blinding_version="blind-v1",
                        comprehension_check_id="check-v1",
                        comprehension_passed=True,
                    )
                )

        model_rows = []
        strengths = {
            "aware": {
                "balanced": 0.80,
                "restricted": 0.20,
                "default": 0.32,
                "suggested": 0.40,
            },
            # The ordinary writer discounts policy-conditioned evidence much
            # less than the human participants.
            "llm-primary": {
                "balanced": 0.80,
                "restricted": 0.68,
                "default": 0.64,
                "suggested": 0.60,
            },
        }
        roles = {
            "aware": "fitted_action_aware",
            "llm-primary": "ordinary_llm",
        }
        for source_id, by_condition in strengths.items():
            for cluster in range(8):
                for condition, strength in by_condition.items():
                    model_rows.append(
                        ModelEvidenceStrength(
                            source_run_id="experiment-a-test-run",
                            source_artifact_sha256=_DIGEST,
                            source_record_id=(
                                f"{source_id}:case-{cluster}:{condition}"
                            ),
                            source_id=source_id,
                            source_role=roles[source_id],
                            cluster_id=f"case-{cluster}",
                            scenario_id="scenario-1",
                            condition=condition,
                            evidence_strength=strength,
                            evidence_metric=EVIDENCE_METRIC,
                            zero_means_no_evidence=True,
                        )
                    )
        return tuple(human_rows), codebook, tuple(model_rows)

    def test_h8_uses_dimensionless_complete_cluster_contrasts(self) -> None:
        human, codebook, model = self._fixture()
        analysis = analyze_h8_human_model_comparison(
            human,
            model,
            assignment_codebooks=codebook,
            expected_assignment_protocol_id="assignment-v1",
            expected_consent_version="consent-v1",
            expected_blinding_version="blind-v1",
            primary_llm_source_id="llm-primary",
            bootstrap_replicates=200,
        )
        self.assertEqual(analysis.computed_status, "computed")
        self.assertTrue(analysis.criterion_met)
        self.assertEqual(
            set(analysis.qualifying_primary_llm_mechanisms),
            {"restricted", "default", "suggested"},
        )
        primary = [
            row
            for row in analysis.contrasts
            if row.source_id == "llm-primary"
        ]
        self.assertEqual(len(primary), 3)
        self.assertTrue(all(row.bootstrap_lower > 0 for row in primary))
        self.assertTrue(all(row.human_cluster_count == 8 for row in primary))
        self.assertEqual(analysis.to_dict()["claim_status"], "not_claimed")

    def test_inadequate_or_missing_primary_evidence_is_incomplete(self) -> None:
        human, codebook, model = self._fixture()
        incomplete = analyze_h8_human_model_comparison(
            human,
            tuple(row for row in model if row.source_id == "aware"),
            assignment_codebooks=codebook,
            expected_assignment_protocol_id="assignment-v1",
            expected_consent_version="consent-v1",
            expected_blinding_version="blind-v1",
            primary_llm_source_id="llm-primary",
            bootstrap_replicates=50,
        )
        self.assertEqual(incomplete.computed_status, "incomplete")
        self.assertIsNone(incomplete.criterion_met)
        self.assertEqual(
            set(incomplete.missing_primary_llm_mechanisms),
            {"restricted", "default", "suggested"},
        )

    def test_source_semantics_and_duplicate_pairs_fail_closed(self) -> None:
        human, codebook, model = self._fixture()
        invalid = ModelEvidenceStrength(
            source_run_id="experiment-a-test-run",
            source_artifact_sha256=_DIGEST,
            source_record_id="llm-primary:new-case:balanced",
            source_id="llm-primary",
            source_role="ordinary_llm",
            cluster_id="new-case",
            scenario_id="scenario-1",
            condition="balanced",
            evidence_strength=0.4,
            evidence_metric=EVIDENCE_METRIC,
            zero_means_no_evidence=True,
        )
        invalid = replace(invalid, source_artifact_sha256="b" * 64)
        with self.assertRaisesRegex(ValueError, "changes role, metric, run"):
            analyze_h8_human_model_comparison(
                human,
                model + (invalid,),
                assignment_codebooks=codebook,
                expected_assignment_protocol_id="assignment-v1",
                expected_consent_version="consent-v1",
                expected_blinding_version="blind-v1",
                primary_llm_source_id="llm-primary",
                bootstrap_replicates=20,
            )

    def test_experiment_a_converter_uses_support_direction_and_no_volunteered(
        self,
    ) -> None:
        rows = []
        for updater_id in ("fitted_action_aware", "llm_full_context"):
            for mechanism, update in (
                ("balanced", -0.8),
                ("restricted", -0.2),
                ("default", 0.3),
                ("suggested", -0.4),
            ):
                rows.append(
                    {
                        "trial_id": f"trial:{mechanism}",
                        "user_id": "test-user-1",
                        "domain": "travel",
                        "target_attribute": 0,
                        "anchor_direction": -1,
                        "prior_stratum": "neutral",
                        "prior_strength": 0.0,
                        "mechanism": mechanism,
                        "response_mode": "controlled_anchor",
                        "updater_id": updater_id,
                        "log_odds_update": update,
                    }
                )
        evidence = convert_experiment_a_metrics_to_model_evidence(
            rows,
            source_run_id="run-a",
            source_artifact_sha256=_DIGEST,
            sources={
                "aware": "fitted_action_aware",
                "primary": "llm_full_context",
            },
            test_user_domain_pairs={("test-user-1", "travel")},
        )
        self.assertEqual(len(evidence), 8)
        self.assertNotIn("volunteered", {row.condition for row in evidence})
        default_rows = [row for row in evidence if row.condition == "default"]
        self.assertTrue(all(row.evidence_strength == 0.0 for row in default_rows))
        with self.assertRaisesRegex(ValueError, "accepts only"):
            convert_experiment_a_metrics_to_model_evidence(
                rows,
                source_run_id="run-a",
                source_artifact_sha256=_DIGEST,
                sources={"proxy": "full_context_blind"},
            )

    def test_public_schema_and_compare_cli_are_strict_and_atomic(self) -> None:
        human, codebook, model = self._fixture()
        schema = SCHEMAS["human-model-evidence"]
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(
            schema["properties"]["evidence_metric"]["const"],
            EVIDENCE_METRIC,
        )
        checked_in = (
            Path(__file__).resolve().parents[1]
            / "schemas"
            / "human-model-evidence.schema.json"
        )
        self.assertEqual(
            json.loads(checked_in.read_text(encoding="utf-8")),
            schema,
        )
        with TemporaryDirectory() as directory:
            root = Path(directory)
            responses = root / "responses.jsonl"
            responses.write_text(
                "".join(
                    json.dumps(row.to_dict(), sort_keys=True) + "\n"
                    for row in human
                ),
                encoding="utf-8",
            )
            codebook_path = root / "codebook.json"
            codebook_path.write_text(
                json.dumps({"assignments": codebook}, sort_keys=True),
                encoding="utf-8",
            )
            evidence_path = root / "evidence.jsonl"
            evidence_path.write_text(
                "".join(
                    json.dumps(row.to_dict(), sort_keys=True) + "\n"
                    for row in model
                ),
                encoding="utf-8",
            )
            output = root / "h8.json"
            command = [
                "human-study",
                "compare",
                str(responses),
                str(codebook_path),
                str(evidence_path),
                str(output),
                "--primary-llm-source-id",
                "llm-primary",
                "--assignment-protocol-id",
                "assignment-v1",
                "--consent-version",
                "consent-v1",
                "--blinding-version",
                "blind-v1",
                "--bootstrap-replicates",
                "20",
            ]
            with redirect_stdout(StringIO()):
                status = main(command)
            self.assertEqual(status, 0)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["claim_status"], "not_claimed")
            self.assertEqual(
                payload["input_artifacts"]["model_evidence_sha256"],
                sha256(evidence_path.read_bytes()).hexdigest(),
            )
            retained = output.read_bytes()
            error_output = StringIO()
            with self.assertRaises(SystemExit) as already_exists:
                with (
                    redirect_stdout(StringIO()),
                    redirect_stderr(error_output),
                ):
                    main(command)
            self.assertEqual(already_exists.exception.code, 2)
            self.assertIn("already exists", error_output.getvalue())
            self.assertEqual(output.read_bytes(), retained)

            original_responses = responses.read_bytes()
            changed_output = root / "h8-changed-input.json"
            changed_command = [*command]
            changed_command[5] = str(changed_output)
            original_analyze = (
                cli_module.analyze_h8_human_model_comparison
            )

            def analyze_then_change_input(*args, **kwargs):
                result = original_analyze(*args, **kwargs)
                responses.write_bytes(original_responses + b"\n")
                return result

            with patch(
                "cape_loop.cli.analyze_h8_human_model_comparison",
                side_effect=analyze_then_change_input,
            ):
                changed_error = StringIO()
                with self.assertRaises(SystemExit) as changed:
                    with (
                        redirect_stdout(StringIO()),
                        redirect_stderr(changed_error),
                    ):
                        main(changed_command)
                self.assertEqual(changed.exception.code, 2)
                self.assertIn(
                    "changed while the analysis was running",
                    changed_error.getvalue(),
                )
            self.assertFalse(changed_output.exists())
            responses.write_bytes(original_responses)
        with self.assertRaisesRegex(ValueError, "duplicate model evidence"):
            analyze_h8_human_model_comparison(
                human,
                model + (model[0],),
                assignment_codebooks=codebook,
                expected_assignment_protocol_id="assignment-v1",
                expected_consent_version="consent-v1",
                expected_blinding_version="blind-v1",
                primary_llm_source_id="llm-primary",
                bootstrap_replicates=20,
            )


if __name__ == "__main__":
    unittest.main()
