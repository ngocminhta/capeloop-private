from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

from cape_loop.artifacts import verify_run
from cape_loop.config import (
    AppConfig,
    ExperimentSection,
    InferenceSection,
    ResponseModelSection,
    RunSection,
)
from cape_loop.runner import run_experiment


class EndToEndRunnerTests(unittest.TestCase):
    def test_small_experiment_a_artifact_is_complete_and_reusable(self) -> None:
        config = AppConfig(
            run=RunSection(name="runner-test", seed=31),
            experiment=ExperimentSection(
                kind="provenance_audit",
                domains=("travel",),
                mechanisms=("balanced", "default"),
                response_modes=("controlled_anchor", "naturally_sampled"),
                policies=("balanced",),
                updaters=(
                    "exact_action_aware",
                    "fitted_action_aware",
                    "fitted_action_unaware",
                ),
                users=1,
                trajectories_per_cell=1,
                turns=1,
                bootstrap_replicates=0,
            ),
            response_model=ResponseModelSection(
                minimum_matched_probability=0.01
            ),
            inference=InferenceSection(
                training_interactions=48,
                fit_steps=40,
                learning_rate=0.04,
                l2=0.001,
            ),
        )
        with TemporaryDirectory() as directory:
            first = run_experiment(config, output_root=directory)
            run_dir = Path(first["run_dir"])
            valid, errors = verify_run(run_dir)
            self.assertTrue(valid, errors)
            summary = json.loads(
                (run_dir / "metrics" / "summary.json").read_text()
            )
            self.assertEqual(summary["scientific_claim_status"], "not_claimed")
            self.assertGreater(summary["row_count"], 0)
            gate_report = json.loads(
                (run_dir / "metrics" / "gate-report.json").read_text()
            )
            self.assertEqual(gate_report["claim_status"], "not_claimed")
            calibration = json.loads(
                (run_dir / "models" / "calibration.json").read_text()
            )
            self.assertEqual(calibration["kind"], "temperature")
            self.assertTrue(
                (run_dir / "models" / "raw-fitted-likelihoods.json").is_file()
            )
            split_audit = json.loads(
                (
                    run_dir / "metrics" / "split-leakage-audit.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(split_audit["status"], "passed")
            self.assertTrue(
                (
                    run_dir / "events" / "fitted-model-development.jsonl"
                ).is_file()
            )
            self.assertFalse(
                any(split_audit["overlaps"].values())
            )
            paraphrase = json.loads(
                (
                    run_dir
                    / "metrics"
                    / "experiment-a-held-out-paraphrase-transfer.json"
                ).read_text(encoding="utf-8")
            )
            self.assertIsNone(paraphrase["verified"])
            self.assertTrue(paraphrase["missing_pairs"])
            event_row = json.loads(
                (run_dir / "events" / "experiment-a.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            reference_rows = {
                row["exact_reference_id"]: row
                for row in (
                    json.loads(line)
                    for line in (
                        run_dir
                        / "events"
                        / "experiment-a-exact-references.jsonl"
                    )
                    .read_text(encoding="utf-8")
                    .splitlines()
                )
            }
            self.assertNotIn("exact_theta_psi", event_row)
            self.assertNotIn("exact_posterior", event_row)
            self.assertIn(event_row["exact_reference_id"], reference_rows)
            self.assertIsNotNone(
                reference_rows[event_row["exact_reference_id"]][
                    "exact_theta_psi"
                ]
            )
            second = run_experiment(
                config,
                output_root=directory,
                allow_existing=True,
            )
            self.assertTrue(second["reused"])
            self.assertEqual(first["summary"], second["summary"])

    def test_small_experiment_b_artifact_and_gate_adapter(self) -> None:
        config = AppConfig(
            run=RunSection(name="runner-b-test", seed=37),
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
        with TemporaryDirectory() as directory:
            result = run_experiment(config, output_root=directory)
            run_dir = Path(result["run_dir"])
            valid, errors = verify_run(run_dir)
            self.assertTrue(valid, errors)
            terminal = (
                run_dir / "metrics" / "experiment-b-terminal.jsonl"
            ).read_text(encoding="utf-8")
            self.assertTrue(terminal.strip())
            self.assertTrue(
                (
                    run_dir
                    / "metrics"
                    / "experiment-b-held-out-actions.jsonl"
                )
                .read_text(encoding="utf-8")
                .strip()
            )
            calibration = json.loads(
                (
                    run_dir
                    / "metrics"
                    / "experiment-b-terminal-calibration.json"
                ).read_text(encoding="utf-8")
            )
            self.assertTrue(calibration["groups"])
            self.assertTrue(
                all(
                    group["profile_calibration_sample_unit"]
                    == "preference_attribute_forecast"
                    for group in calibration["groups"]
                )
            )
            decoder_manifest = json.loads(
                (
                    run_dir / "decoder" / "design-manifest.json"
                ).read_text(encoding="utf-8")
            )
            self.assertGreater(
                decoder_manifest["development_request_count"], 0
            )
            self.assertGreater(decoder_manifest["test_request_count"], 0)
            gate_report = json.loads(
                (run_dir / "metrics" / "gate-report.json").read_text()
            )
            self.assertEqual(gate_report["claim_status"], "not_claimed")
            gate_4 = next(
                gate
                for gate in gate_report["gates"]
                if gate["gate_id"] == "gate-4"
            )
            self.assertEqual(gate_4["computed_status"], "incomplete")
            criteria = {
                item["criterion_id"]: item
                for item in gate_4["criteria"]
            }
            self.assertIsNone(
                criteria["independent-blinded-decoder-judgments"]["passed"]
            )
            self.assertFalse(
                criteria["independent-blinded-decoder-judgments"][
                    "observed"
                ]["deterministic_projections_count_as_external"]
            )
            self.assertIsNone(
                criteria["native-end-to-end-terminal-actions"]["passed"]
            )
            self.assertFalse(
                criteria["native-end-to-end-terminal-actions"]["observed"][
                    "persona_or_structured_references_count_as_native_actions"
                ]
            )
            inference = json.loads(
                (
                    run_dir / "metrics" / "experiment-b-inference.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(inference["analysis_status"], "not_computed")

    def test_small_experiment_c_writes_paired_and_calibration_artifacts(
        self,
    ) -> None:
        config = AppConfig(
            run=RunSection(name="runner-c-test", seed=43),
            experiment=ExperimentSection(
                kind="evaluation_validity",
                domains=("travel",),
                mechanisms=("ranking", "default", "suggestion"),
                response_modes=("naturally_sampled",),
                policies=(
                    "balanced",
                    "fixed_bias",
                    "soft_profile_conditioned",
                ),
                updaters=("response_only", "provenance_aware"),
                users=2,
                trajectories_per_cell=1,
                turns=2,
                bootstrap_replicates=20,
            ),
            inference=InferenceSection(
                training_interactions=24,
                fit_steps=10,
                learning_rate=0.04,
                l2=0.001,
            ),
        )
        with TemporaryDirectory() as directory:
            result = run_experiment(config, output_root=directory)
            run_dir = Path(result["run_dir"])
            valid, errors = verify_run(run_dir)
            self.assertTrue(valid, errors)
            calibration = json.loads(
                (
                    run_dir
                    / "metrics"
                    / "experiment-c-terminal-calibration.json"
                ).read_text(encoding="utf-8")
            )
            self.assertTrue(calibration["groups"])
            self.assertTrue(
                all(
                    group["profile_calibration_sample_unit"]
                    == "preference_attribute_forecast"
                    for group in calibration["groups"]
                )
            )
            rankings = json.loads(
                (
                    run_dir / "metrics" / "experiment-c-rankings.json"
                ).read_text(encoding="utf-8")
            )
            self.assertTrue(
                rankings["pairwise_open_closed_shift_intervals"]
            )
            self.assertEqual(
                rankings["evaluation_selection_regret"][
                    "selection_basis"
                ],
                "paired development error-difference confidence-set top tiers",
            )
            self.assertIn("open_partial_order", rankings)
            self.assertIn("closed_partial_order", rankings)


if __name__ == "__main__":
    unittest.main()
