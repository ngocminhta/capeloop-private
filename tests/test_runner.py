from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json
import unittest

from cape_loop.artifacts import verify_run
from cape_loop.config import (
    AppConfig,
    ExperimentSection,
    InferenceSection,
    ResponseModelSection,
    RunSection,
    ScenarioSection,
)
from cape_loop.runner import run_experiment


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = (
    REPOSITORY_ROOT / "data" / "scenarios" / "scenario-catalog-v1.json"
)
CONVERSATION_PATH = (
    REPOSITORY_ROOT
    / "data"
    / "scenarios"
    / "conversation-templates-v1.json"
)
CATALOG_SHA256 = (
    "7b7144b3b3f75ac7284ab6153d1b6ce62cf293aec94004ee2cb3111bcc1f6cf1"
)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


class EndToEndRunnerTests(unittest.TestCase):
    def test_source_config_must_match_before_provider_or_artifact(self) -> None:
        config = AppConfig(
            run=RunSection(name="config-a", seed=31),
            experiment=ExperimentSection(
                kind="provenance_audit",
                domains=("travel",),
                mechanisms=("balanced",),
                response_modes=("controlled_anchor",),
                policies=("balanced",),
                updaters=("exact_action_aware",),
                users=1,
                trajectories_per_cell=1,
                turns=1,
                bootstrap_replicates=0,
            ),
        )
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "mismatched.toml"
            source.write_text(
                "\n".join(
                    (
                        "schema_version = 1",
                        "",
                        "[run]",
                        'name = "config-b"',
                        "seed = 31",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            output = root / "runs"
            with patch(
                "cape_loop.runner._live_completion_provider",
                side_effect=AssertionError(
                    "provider construction preceded source validation"
                ),
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "does not resolve to the supplied AppConfig",
                ):
                    run_experiment(
                        config,
                        output_root=output,
                        source_config=source,
                        execute_live=True,
                    )
            self.assertFalse(output.exists())

    def test_source_config_size_limit_matches_run_verifier(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "oversized.toml"
            source.write_text(
                "schema_version = 1\n#" + ("x" * 128) + "\n",
                encoding="utf-8",
            )
            output = root / "runs"
            with patch(
                "cape_loop.artifacts._MAX_CONTROL_FILE_BYTES",
                64,
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "source_config exceeds 64 bytes",
                ):
                    run_experiment(
                        AppConfig(),
                        output_root=output,
                        source_config=source,
                    )
            self.assertFalse(output.exists())

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
            self.assertEqual(
                summary["control_battery_status"],
                "executable_separate_control_study",
            )
            self.assertEqual(
                summary["control_reference_criterion_pass_count"],
                6,
            )
            self.assertEqual(
                summary["control_baseline_criterion_pass_count"],
                3,
            )
            self.assertEqual(summary["control_live_evidence_count"], 0)
            self.assertEqual(summary["control_provider_request_count"], 6)
            control_plan = json.loads(
                (
                    run_dir
                    / "models"
                    / "experiment-a-control-plan.json"
                ).read_text()
            )
            control_reference = json.loads(
                (
                    run_dir
                    / "metrics"
                    / "experiment-a-control-reference.json"
                ).read_text()
            )
            control_baseline = json.loads(
                (
                    run_dir
                    / "metrics"
                    / "experiment-a-control-baseline.json"
                ).read_text()
            )
            control_exchange = json.loads(
                (
                    run_dir
                    / "llm"
                    / "experiment-a-control-exchange.json"
                ).read_text()
            )
            self.assertEqual(
                control_plan["plan_sha256"],
                summary["control_plan_sha256"],
            )
            self.assertTrue(control_reference["coverage"]["complete"])
            self.assertTrue(control_baseline["coverage"]["complete"])
            self.assertEqual(
                control_reference["evidence_class"],
                "diagnostic_reference",
            )
            self.assertEqual(
                control_baseline["evidence_class"],
                "diagnostic_baseline",
            )
            self.assertTrue(
                control_exchange["coverage"]["complete_for_all_six_controls"]
            )
            self.assertEqual(
                len(
                    (
                        run_dir
                        / "llm"
                        / "experiment-a-control-requests.jsonl"
                    )
                    .read_text()
                    .splitlines()
                ),
                6,
            )
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
            analysis_rows = _read_jsonl(
                run_dir / "analysis" / "experiment-a-rows.jsonl"
            )
            conversation_rows = _read_jsonl(
                run_dir / "conversations" / "experiment-a.jsonl"
            )
            self.assertEqual(
                summary["conversation_record_count"],
                len({row["trial_id"] for row in analysis_rows}),
            )
            self.assertEqual(
                len(conversation_rows),
                summary["conversation_record_count"],
            )
            self.assertEqual(
                summary["conversation_outcome_count"],
                summary["row_count"],
            )
            self.assertEqual(
                sum(len(row["outcomes"]) for row in conversation_rows),
                summary["row_count"],
            )
            self.assertEqual(
                summary["analysis_artifact"],
                "analysis/experiment-a-rows.jsonl",
            )
            self.assertEqual(
                summary["analysis_row_count"],
                summary["row_count"],
            )
            self.assertEqual(len(analysis_rows), summary["analysis_row_count"])
            self.assertEqual(
                summary["analysis_exclusion_artifact"],
                "analysis/experiment-a-exclusions.jsonl",
            )
            self.assertEqual(
                _read_jsonl(
                    run_dir
                    / "analysis"
                    / "experiment-a-exclusions.jsonl"
                ),
                _read_jsonl(
                    run_dir
                    / "events"
                    / "experiment-a-exclusions.jsonl"
                ),
            )
            self.assertEqual(
                [row["source_record_index"] for row in analysis_rows],
                list(range(1, len(analysis_rows) + 1)),
            )
            self.assertEqual(
                set(analysis_rows[0]),
                {
                    "schema_version",
                    "source_record_index",
                    "trial_id",
                    "user_id",
                    "domain_id",
                    "scenario_id",
                    "updater_id",
                    "mechanism",
                    "prior_strength",
                    "response_mode",
                    "update_error",
                },
            )
            self.assertEqual(
                analysis_rows[0]["scenario_id"],
                event_row["context"]["scenario_id"],
            )
            self.assertEqual(
                analysis_rows[0]["update_error"],
                event_row["metrics"]["acue"],
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

    def test_catalog_backed_run_retains_and_binds_scenario_input(self) -> None:
        config = AppConfig(
            run=RunSection(name="catalog-runner-test", seed=31),
            experiment=ExperimentSection(
                kind="provenance_audit",
                domains=("travel",),
                mechanisms=("balanced",),
                response_modes=("controlled_anchor",),
                policies=("balanced",),
                updaters=("exact_action_aware",),
                users=1,
                trajectories_per_cell=1,
                turns=1,
                bootstrap_replicates=0,
            ),
            response_model=ResponseModelSection(
                minimum_matched_probability=0.01
            ),
            inference=InferenceSection(
                training_interactions=24,
                fit_steps=10,
                learning_rate=0.04,
                l2=0.001,
            ),
            scenarios=ScenarioSection(
                catalog_file=str(CATALOG_PATH),
                catalog_sha256=CATALOG_SHA256,
                conversation_file=str(CONVERSATION_PATH),
                selection_policy="deterministic-stratified-v1",
            ),
        )
        with TemporaryDirectory() as directory:
            first = run_experiment(config, output_root=directory)
            run_dir = Path(first["run_dir"])
            valid, errors = verify_run(run_dir)
            self.assertTrue(valid, errors)

            retained_catalog = run_dir / "inputs" / "scenario-catalog.json"
            self.assertEqual(retained_catalog.read_bytes(), CATALOG_PATH.read_bytes())
            input_manifest = json.loads(
                (
                    run_dir / "inputs" / "scenario-catalog-manifest.json"
                ).read_text(encoding="utf-8")
            )
            run_manifest = json.loads(
                (run_dir / "manifest.json").read_text(encoding="utf-8")
            )
            coverage = json.loads(
                (
                    run_dir / "metrics" / "scenario-coverage.json"
                ).read_text(encoding="utf-8")
            )
            summary = json.loads(
                (run_dir / "metrics" / "summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(input_manifest["source_sha256"], CATALOG_SHA256)
            self.assertEqual(
                run_manifest["inputs"]["scenario_catalog"],
                input_manifest,
            )
            retained_conversations = (
                run_dir / "inputs" / "conversation-templates.json"
            )
            self.assertEqual(
                retained_conversations.read_bytes(),
                CONVERSATION_PATH.read_bytes(),
            )
            conversation_manifest = json.loads(
                (
                    run_dir
                    / "inputs"
                    / "conversation-templates-manifest.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(
                run_manifest["inputs"]["conversation_templates"],
                conversation_manifest,
            )
            self.assertEqual(
                conversation_manifest["scenario_count"],
                48,
            )
            self.assertEqual(summary["scenario_catalog"], input_manifest)
            self.assertEqual(
                summary["conversation_templates"]["runtime_mode"],
                "mathematical_choice_with_frozen_llm_dialogue",
            )
            self.assertEqual(coverage["scenario_count"], 48)
            self.assertEqual(coverage["family_count"], 48)
            self.assertFalse(coverage["paper_eligible"])
            self.assertEqual(
                coverage["coverage_kind"],
                "catalog_availability",
            )

            catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
            test_scenarios = {
                scenario["scenario_id"]: scenario
                for scenario in catalog["scenarios"]
                if scenario["split"] == "test"
            }
            event = _read_jsonl(
                run_dir / "events" / "experiment-a.jsonl"
            )[0]
            context = event["context"]
            observation = event["observation"]
            self.assertIn(context["scenario_id"], test_scenarios)
            self.assertTrue(observation["assistant_message"])
            self.assertTrue(observation["surface_id"])
            self.assertRegex(
                observation["surface_response"],
                r"^I choose .+ [A-D]\.$",
            )
            self.assertIn(
                context["prompt"],
                observation["assistant_message"],
            )
            scenario = test_scenarios[context["scenario_id"]]
            self.assertEqual(context["prompt"], scenario["prompt"])
            catalog_options = {
                option["option_id"]: option["label"]
                for option in (
                    scenario["negative_option"],
                    scenario["positive_option"],
                    scenario["negative_same_direction_option"],
                    scenario["positive_same_direction_option"],
                )
            }
            self.assertEqual(
                {
                    option["option_id"]: option["label"]
                    for option in context["options"]
                },
                {
                    option["option_id"]: catalog_options[option["option_id"]]
                    for option in context["options"]
                },
            )
            split_audit = json.loads(
                (
                    run_dir / "metrics" / "split-leakage-audit.json"
                ).read_text(encoding="utf-8")
            )
            for split in ("train", "development"):
                coverage_row = split_audit[
                    "realized_fitted_data_scenario_coverage"
                ][split][0]
                self.assertTrue(coverage_row["complete"])
                self.assertEqual(
                    coverage_row[
                        "mechanism_target_direction_cell_count"
                    ],
                    24,
                )
                self.assertFalse(coverage_row["missing_scenario_ids"])
            consumption = json.loads(
                (
                    run_dir / "metrics" / "scenario-consumption.json"
                ).read_text(encoding="utf-8")
            )
            self.assertIn(
                context["scenario_id"],
                consumption["experiment"]["test"][
                    "observed_scenario_ids"
                ],
            )
            self.assertEqual(
                consumption["catalog_availability_artifact"],
                "metrics/scenario-coverage.json",
            )
            conversation_rows = _read_jsonl(
                run_dir / "conversations" / "experiment-a.jsonl"
            )
            self.assertEqual(
                summary["conversation_log_artifact"],
                "conversations/experiment-a.jsonl",
            )
            self.assertEqual(
                summary["conversation_log_markdown_artifact"],
                "conversations/experiment-a.md",
            )
            self.assertEqual(
                summary["conversation_record_count"],
                len(conversation_rows),
            )
            self.assertEqual(
                summary["conversation_turn_count"],
                len(conversation_rows),
            )
            first_trace = conversation_rows[0]
            first_turn = first_trace["dialogue"][0]
            self.assertEqual(
                first_turn["assistant"],
                observation["assistant_message"],
            )
            self.assertEqual(
                first_turn["user"],
                observation["surface_response"],
            )
            self.assertTrue(first_turn["surface_available"])
            self.assertEqual(
                first_turn["choice_source"],
                "mathematical_user_simulator",
            )
            self.assertTrue(first_turn["selected_option_label"])
            self.assertNotIn("belief", first_trace)
            self.assertNotIn("theta", first_trace)
            readable_log = (
                run_dir / "conversations" / "experiment-a.md"
            ).read_text(encoding="utf-8")
            self.assertIn("Scenario presenter (assistant)", readable_log)
            self.assertIn("Simulated user", readable_log)
            self.assertIn("Evaluated profile updater", readable_log)
            self.assertIn(first_turn["assistant"], readable_log)
            self.assertIn(first_turn["user"], readable_log)

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
            scenarios=ScenarioSection(
                catalog_file=str(CATALOG_PATH),
                catalog_sha256=CATALOG_SHA256,
            ),
        )
        with TemporaryDirectory() as directory:
            result = run_experiment(config, output_root=directory)
            run_dir = Path(result["run_dir"])
            valid, errors = verify_run(run_dir)
            self.assertTrue(valid, errors)
            summary = result["summary"]
            analysis_rows = _read_jsonl(
                run_dir / "analysis" / "experiment-b-turns.jsonl"
            )
            conversation_rows = _read_jsonl(
                run_dir / "conversations" / "experiment-b.jsonl"
            )
            self.assertEqual(
                summary["conversation_record_count"],
                summary["trajectories"],
            )
            self.assertEqual(
                len(conversation_rows),
                summary["trajectories"],
            )
            self.assertEqual(
                summary["conversation_turn_count"],
                summary["analysis_row_count"],
            )
            self.assertEqual(
                summary["conversation_outcome_count"],
                summary["trajectories"],
            )
            self.assertTrue(
                all(
                    len(row["dialogue"]) == config.experiment.turns
                    for row in conversation_rows
                )
            )
            first_turn_metrics = conversation_rows[0]["dialogue"][0][
                "turn_metrics"
            ]
            self.assertIn(
                "profile_error_after_turn",
                first_turn_metrics,
            )
            self.assertNotIn("terminal_error", first_turn_metrics)
            self.assertIn(
                "terminal_error",
                conversation_rows[0]["outcomes"][0]["metrics"],
            )
            self.assertEqual(
                summary["analysis_artifact"],
                "analysis/experiment-b-turns.jsonl",
            )
            self.assertEqual(
                summary["analysis_row_count"],
                summary["trajectories"] * config.experiment.turns,
            )
            self.assertEqual(len(analysis_rows), summary["analysis_row_count"])
            self.assertEqual(
                set(analysis_rows[0]),
                {
                    "schema_version",
                    "source_record_index",
                    "source_turn_index",
                    "trajectory_id",
                    "user_id",
                    "domain_id",
                    "scenario_id",
                    "crn_key",
                    "updater_id",
                    "policy_id",
                    "initial_profile_condition",
                    "turn",
                    "terminal_error",
                    "retained_terminal_error",
                    "same_history_shadow",
                },
            )
            grouped_analysis: dict[int, list[dict[str, object]]] = {}
            for row in analysis_rows:
                grouped_analysis.setdefault(
                    int(row["source_record_index"]),
                    [],
                ).append(row)
            self.assertEqual(
                sorted(grouped_analysis),
                list(range(1, summary["trajectories"] + 1)),
            )
            for rows in grouped_analysis.values():
                self.assertEqual(
                    [row["source_turn_index"] for row in rows],
                    list(range(config.experiment.turns)),
                )
                self.assertEqual(
                    [row["turn"] for row in rows],
                    list(range(1, config.experiment.turns + 1)),
                )
                self.assertTrue(
                    all(row["same_history_shadow"] is True for row in rows)
                )
                self.assertEqual(
                    rows[-1]["terminal_error"],
                    rows[-1]["retained_terminal_error"],
                )
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
            power = json.loads(
                (
                    run_dir / "metrics" / "experiment-b-power.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(power["status"], "not_estimable")
            self.assertEqual(power["artifact_role"], "pilot_design_evidence")
            power_summary = (
                run_dir / "tables" / "experiment-b-power.md"
            ).read_text(encoding="utf-8")
            self.assertIn("Scientific claim status: `not_claimed`", power_summary)
            self.assertIn("not empirical evidence", power_summary)
            scenario_consumption = json.loads(
                (
                    run_dir / "metrics" / "scenario-consumption.json"
                ).read_text(encoding="utf-8")
            )
            self.assertGreater(
                scenario_consumption["experiment"]["test"][
                    "observed_scenario_count"
                ],
                0,
            )

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
            scenarios=ScenarioSection(
                catalog_file=str(CATALOG_PATH),
                catalog_sha256=CATALOG_SHA256,
            ),
        )
        with TemporaryDirectory() as directory:
            result = run_experiment(config, output_root=directory)
            run_dir = Path(result["run_dir"])
            valid, errors = verify_run(run_dir)
            self.assertTrue(valid, errors)
            summary = result["summary"]
            analysis_rows = _read_jsonl(
                run_dir / "analysis" / "experiment-c-rows.jsonl"
            )
            conversation_rows = _read_jsonl(
                run_dir / "conversations" / "experiment-c.jsonl"
            )
            self.assertEqual(
                summary["conversation_record_count"],
                summary["fixed_histories"]
                + summary["endogenous_trajectories"],
            )
            self.assertEqual(
                len(conversation_rows),
                summary["conversation_record_count"],
            )
            self.assertEqual(
                summary["conversation_outcome_count"],
                summary["evaluation_rows"],
            )
            self.assertEqual(
                sum(len(row["outcomes"]) for row in conversation_rows),
                summary["evaluation_rows"],
            )
            self.assertEqual(
                summary["conversation_turn_count"],
                summary["conversation_record_count"]
                * config.experiment.turns,
            )
            fixed_traces = [
                row
                for row in conversation_rows
                if row["conversation_kind"] == "fixed_history"
            ]
            self.assertTrue(fixed_traces)
            self.assertTrue(
                all(
                    len(row["outcomes"]) == len(config.experiment.updaters)
                    for row in fixed_traces
                )
            )
            self.assertEqual(
                summary["analysis_artifact"],
                "analysis/experiment-c-rows.jsonl",
            )
            self.assertEqual(
                summary["analysis_row_count"],
                summary["evaluation_rows"],
            )
            self.assertEqual(len(analysis_rows), summary["analysis_row_count"])
            self.assertEqual(
                [row["source_record_index"] for row in analysis_rows],
                list(range(1, len(analysis_rows) + 1)),
            )
            self.assertEqual(
                set(analysis_rows[0]),
                {
                    "schema_version",
                    "source_record_index",
                    "split",
                    "regime",
                    "replicate",
                    "user_id",
                    "domain_id",
                    "updater_id",
                    "profile_error",
                    "behavioral_accuracy",
                    "cross_context_accuracy",
                    "intrinsic_regret",
                    "score_basis",
                    "history_digest",
                    "battery_id",
                    "battery_digest",
                },
            )
            full_metric_row = _read_jsonl(
                run_dir / "metrics" / "experiment-c.jsonl"
            )[0]
            for field in (
                "split",
                "regime",
                "replicate",
                "user_id",
                "domain_id",
                "updater_id",
                "profile_error",
                "behavioral_accuracy",
                "cross_context_accuracy",
                "intrinsic_regret",
                "score_basis",
                "history_digest",
                "battery_id",
                "battery_digest",
            ):
                self.assertEqual(
                    analysis_rows[0][field],
                    full_metric_row[field],
                )
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
            scenario_consumption = json.loads(
                (
                    run_dir / "metrics" / "scenario-consumption.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(
                set(scenario_consumption["experiment"]),
                {"development", "test"},
            )


if __name__ == "__main__":
    unittest.main()
