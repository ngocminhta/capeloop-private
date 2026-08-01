from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import json
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis" / "confirmatory-mixed-effects"


class MixedEffectsAnalysisContractTests(unittest.TestCase):
    def load_json(self, name: str) -> dict[str, object]:
        return json.loads((ANALYSIS / name).read_text(encoding="utf-8"))

    def read_r_source(self, relative: str) -> str:
        return (ANALYSIS / relative).read_text(encoding="utf-8")

    def test_static_validator_runs_without_r(self) -> None:
        module_spec = spec_from_file_location(
            "cape_loop_mixed_effects_contract",
            ANALYSIS / "validate_contract.py",
        )
        self.assertIsNotNone(module_spec)
        self.assertIsNotNone(module_spec.loader)
        module = module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        module.validate()

    def test_protocol_cannot_be_mistaken_for_results(self) -> None:
        declaration = self.load_json("analysis-spec.json")
        self.assertEqual(declaration["schema_version"], 3)
        self.assertEqual(
            declaration["analysis_id"],
            "cape-loop-confirmatory-mixed-effects-v3",
        )
        self.assertEqual(declaration["status"], "analysis_protocol_not_results")
        self.assertEqual(declaration["family"], "gaussian")
        self.assertEqual(declaration["link"], "identity")
        self.assertEqual(
            declaration["fallback_policy"]["rank_deficient"],
            (
                "Record not_estimable; do not drop proposal fixed effects "
                "automatically."
            ),
        )
        checked_json = {
            path.name
            for path in ANALYSIS.glob("*.json")
        }
        self.assertEqual(
            checked_json,
            {
                "analysis-spec.json",
                "analysis-result.schema.json",
            },
        )

    def test_exact_formulas_factor_coding_and_estimation_are_frozen(
        self,
    ) -> None:
        declaration = self.load_json("analysis-spec.json")
        experiments = declaration["experiments"]
        self.assertEqual(
            experiments["A"]["formula"],
            (
                "calibration_residual ~ mechanism + domain + "
                "prior_strength + (1 + mechanism | user) + (1 | scenario)"
            ),
        )
        self.assertEqual(
            experiments["A"]["pooled_formula"],
            (
                "calibration_residual ~ mechanism + domain + "
                "prior_strength + (1 + mechanism | user) + "
                "(1 | scenario) + (1 | replicate)"
            ),
        )
        self.assertEqual(
            experiments["A"]["proposal_formula"],
            (
                "CalibrationResidual ~ Mechanism + Domain + "
                "PriorStrength + (1 + Mechanism | User) + (1 | Scenario)"
            ),
        )
        self.assertEqual(
            experiments["B"]["formula"],
            (
                "terminal_error ~ updater * policy * initial_profile + "
                "domain + turn + (1 + policy | user) + (1 | scenario) + "
                "(1 | crn_set)"
            ),
        )
        self.assertEqual(
            experiments["B"]["pooled_formula"],
            (
                "terminal_error ~ updater * policy * initial_profile + "
                "domain + turn + (1 + policy | user) + (1 | scenario) + "
                "(1 | crn_set) + (1 | replicate)"
            ),
        )
        self.assertEqual(
            experiments["B"]["proposal_formula"],
            (
                "TerminalError ~ Updater * Policy * InitialProfile + "
                "Domain + Turn + (1 + Policy | User) + (1 | Scenario) + "
                "(1 | CRNSet)"
            ),
        )
        self.assertEqual(
            declaration["factor_coding"],
            {
                "contrasts": ["contr.treatment", "contr.poly"],
                "experiment_a_oracle_reference": "exact_action_aware",
                "experiment_b_updater_reference": "fitted_action_aware",
                "experiment_a_mechanism_reference": "balanced",
                "experiment_b_policy_reference": "balanced",
                "experiment_b_initial_profile_reference": "incorrect",
                "domain_reference_preference": "travel",
            },
        )
        self.assertEqual(
            declaration["estimation"],
            {
                "method": "maximum_likelihood",
                "reml": False,
                "optimizer": "bobyqa",
                "maxfun": 200000,
                "degrees_of_freedom": "Satterthwaite",
                "confidence_level": 0.95,
                "singularity_tolerance": 0.0001,
                "gradient_tolerance": 0.002,
                "gradient_scaling": "inverse_cholesky_hessian",
                "minimum_user_clusters": 8,
                "minimum_scenario_clusters": 8,
            },
        )

    def test_exact_planned_contrasts_are_frozen(self) -> None:
        experiments = self.load_json("analysis-spec.json")["experiments"]
        self.assertEqual(
            experiments["A"]["primary_contrast"],
            {
                "id": (
                    "target-calibration-residual-mechanism-minus-balanced"
                ),
                "mechanisms": [
                    "restricted",
                    "ranking",
                    "default",
                    "suggested",
                ],
                "expression": (
                    "calibration_residual[target_updater, mechanism] - "
                    "calibration_residual[target_updater, balanced]"
                ),
                "multiplicity": (
                    "Holm within the four-mechanism primary family"
                ),
            },
        )
        self.assertEqual(
            experiments["A"]["secondary_contrasts"],
            [],
        )
        self.assertEqual(
            experiments["A"]["secondary_estimands"],
            [
                {
                    "id": "target-exact-update-error-by-mechanism",
                    "outcome": "exact_update_error",
                    "aggregation": (
                        "descriptive distribution and mean by mechanism"
                    ),
                    "inferential_status": (
                        "not fitted by the primary signed-residual model"
                    ),
                }
            ],
        )
        self.assertEqual(
            experiments["B"]["primary_contrast"],
            {
                "id": "incorrect-profile-updater-by-policy",
                "initial_profile": "incorrect",
                "expression": (
                    "(target_updater[soft_profile_conditioned] - "
                    "target_updater[balanced]) - "
                    "(fitted_action_aware[soft_profile_conditioned] - "
                    "fitted_action_aware[balanced])"
                ),
                "multiplicity": "single predeclared contrast",
            },
        )
        self.assertEqual(
            experiments["B"]["secondary_contrasts"],
            [
                {
                    "id": "incorrect-profile-target-policy-effect",
                    "expression": (
                        "target_updater[soft_profile_conditioned] - "
                        "target_updater[balanced]"
                    ),
                },
                {
                    "id": (
                        "incorrect-profile-target-minus-aware-by-policy"
                    ),
                    "expression": (
                        "target_updater[policy] - "
                        "fitted_action_aware[policy]"
                    ),
                    "policies": [
                        "balanced",
                        "soft_profile_conditioned",
                    ],
                },
                {
                    "id": "incorrect-versus-correct-three-way",
                    "expression": (
                        "updater_by_policy[incorrect] - "
                        "updater_by_policy[correct]"
                    ),
                    "requires_initial_profile": "correct",
                },
            ],
        )

    def test_experiment_b_is_a_retained_turn_analysis(self) -> None:
        experiment_b = self.load_json("analysis-spec.json")["experiments"]["B"]
        self.assertEqual(
            experiment_b["input_file"],
            "analysis/experiment-b-turns.jsonl",
        )
        self.assertEqual(
            experiment_b["legacy_input_file"],
            "events/experiment-b-trajectories.jsonl",
        )
        self.assertEqual(
            experiment_b["bundle_analysis_unit"],
            "retained_trajectory_turn",
        )
        self.assertEqual(experiment_b["analysis_unit"], "retained_turn")
        self.assertEqual(
            experiment_b["outcome_source"],
            (
                "runner-native compact terminal_error derived from "
                "turns[].belief_after marginal Brier against top-level theta"
            ),
        )
        self.assertEqual(experiment_b["input_row_schema_version"], 1)
        self.assertEqual(
            experiment_b["turn_source"],
            "compact turn equals source_turn_index + 1",
        )
        self.assertEqual(
            experiment_b["terminal_consistency_check"],
            "final compact turn error equals retained_terminal_error",
        )
        self.assertEqual(experiment_b["outcome_name"], "terminal_error")
        self.assertEqual(
            experiment_b["scenario_mapping"],
            "turn.scenario_id",
        )
        self.assertEqual(
            experiment_b["crn_set_mapping"],
            "run_id + crn_key",
        )
        self.assertEqual(experiment_b["user_mapping"], "user_id")
        self.assertEqual(experiment_b["replicate_mapping"], "run_id")
        self.assertIn(
            "Different seeds are analyzed separately",
            experiment_b["pooling_policy"],
        )

    def test_r_input_contract_binds_source_and_validates_compact_rows(
        self,
    ) -> None:
        experiment_a = self.load_json("analysis-spec.json")["experiments"]["A"]
        self.assertEqual(
            experiment_a["input_file"],
            "analysis/experiment-a-rows.jsonl",
        )
        self.assertEqual(
            experiment_a["exclusion_file"],
            "analysis/experiment-a-exclusions.jsonl",
        )
        self.assertEqual(
            experiment_a["legacy_input_file"],
            "events/experiment-a.jsonl",
        )
        self.assertEqual(
            experiment_a["legacy_exclusion_file"],
            "events/experiment-a-exclusions.jsonl",
        )
        self.assertEqual(experiment_a["input_row_schema_version"], 2)
        self.assertEqual(experiment_a["response_mode"], "controlled_anchor")
        self.assertEqual(
            experiment_a["analysis_track"],
            "same_response_provenance",
        )
        self.assertEqual(
            experiment_a["reference_basis"],
            "exact_action_aware",
        )
        self.assertEqual(
            experiment_a["outcome_name"],
            "calibration_residual",
        )
        self.assertEqual(
            experiment_a["retained_calibration_fields"],
            [
                "system_log_odds_update",
                "exact_log_odds_update",
                "fitted_log_odds_update",
                "calibration_residual",
            ],
        )
        self.assertEqual(
            experiment_a["secondary_outcomes"],
            [
                {
                    "name": "exact_update_error",
                    "source": (
                        "runner-native compact exact_update_error derived "
                        "from metrics.exact_acue"
                    ),
                    "required": True,
                    "role": "descriptive_secondary_absolute_magnitude",
                },
                {
                    "name": "fitted_update_error",
                    "source": (
                        "runner-native compact fitted_update_error derived "
                        "from metrics.acue"
                    ),
                    "required": True,
                    "role": "secondary_reference_diagnostic",
                }
            ],
        )
        self.assertEqual(experiment_a["user_mapping"], "user_id")
        self.assertEqual(experiment_a["scenario_mapping"], "scenario_id")
        self.assertEqual(experiment_a["replicate_mapping"], "run_id")
        self.assertIn(
            "Different seeds are analyzed separately",
            experiment_a["pooling_policy"],
        )
        source = self.read_r_source("R/io.R")
        for anchor in (
            "canonical_config_sha256 <- function(path)",
            "python_ascii_fragment <- function(code_point)",
            "if (code_point <= 127L)",
            'sprintf("\\\\u%04x", code_point)',
            "high <- 55296L",
            "low <- 56320L",
            'Encoding(payload_text) <- "UTF-8"',
            "utf8ToInt(payload_text)",
            "charToRaw(python_canonical_payload)",
            "observed_config_sha256 <- canonical_config_sha256(config_path)",
            "source$manifest$source_sha256",
            "length(unique(source_digests)) != 1L",
            "pooled source runs must have identical config.run.seed",
            "analyze different seeds separately as robustness replicates",
            "population = source$config$population %||% list()",
            "replicate = source$run_id",
            "population_seed = source$population_seed",
            "all_levels_present <- function(",
            "assert_balanced_level_crossing <- function(",
            "rows <- rows[rows$updater == target_updater",
            "Experiment A target updater cannot be the exact oracle reference",
            "every Experiment A target user must contain every mechanism",
            "the same Experiment B horizon",
            "if (!identical(reference, candidate))",
            "assert_exact_fields <- function(value, expected, label)",
            "source$summary$analysis_row_count",
            "source$summary$controlled_row_count",
            "row$retained_terminal_error",
            "row$same_history_shadow",
            '"analysis_track"',
            '"reference_basis"',
            '"exact_update_error"',
            '"fitted_update_error"',
            '"system_log_odds_update"',
            '"exact_log_odds_update"',
            '"fitted_log_odds_update"',
            '"calibration_residual"',
            "same_response_provenance",
            "natural_response_secondary",
            "calibration_residual differs from",
            "compact Experiment A source_record_index values must cover",
            "compact Experiment B source_record_index values must cover",
            "source summary compact exclusion declaration differs",
            "compact Experiment B rows are not in canonical source/turn order",
            "compact turn indexes are not contiguous from zero",
            "compact turn must equal source_turn_index + 1",
            "final compact turn error differs from retained_terminal_error",
            "verify_compact_bundle <- function(",
            "compact bundle inventory must be exactly",
            "compact bundle source manifest digest mismatch",
            "compact bundle source SHA256SUMS digest mismatch",
            "compact bundle source config file digest mismatch",
            "compact bundle source summary file digest mismatch",
            "compact bundle source input binding mismatch",
            "compact bundle source exclusion binding mismatch",
            "compact bundle analysis row digest mismatch",
            "source$analysis_input_path",
            "source$analysis_input_sha256",
        ):
            self.assertIn(anchor, source)
        self.assertNotIn(
            "source run did not declare artifacts.retain_events=true",
            source,
        )
        self.assertNotIn(
            "read_jsonl_objects(\n"
            "    source$input_path,\n"
            '    paste0(source$run_id, " Experiment B trajectories")',
            source,
        )
        self.assertNotRegex(
            source,
            r"experiment_config\s*\$turns\s*<-\s*NULL",
        )

    def test_output_containment_is_checked_before_parent_creation(
        self,
    ) -> None:
        runner = self.read_r_source("run_analysis.R")
        start = runner.index(
            "safe_output_path <- function(output, source_paths)"
        )
        end = runner.index("\nargs <- parse_arguments", start)
        function = runner[start:end]
        source_check = function.index("for (source in source_paths)")
        rejection = function.index(
            "analysis output cannot be inside a source run or compact bundle"
        )
        parent_creation = function.index("dir.create(parent")
        self.assertLess(source_check, rejection)
        self.assertLess(rejection, parent_creation)

    def test_historical_bundle_cli_is_strictly_paired(self) -> None:
        runner = self.read_r_source("run_analysis.R")
        for anchor in (
            '"--compact-bundle"',
            "length(result$compact_bundles) != length(result$runs)",
            "exactly one per --run in the same order",
            "use_legacy_input = if (uses_historical_bundles) NULL else FALSE",
            "verify_compact_bundle(",
            "source_runs[[index]]$analysis_input_path <- bundle$rows_path",
            "source_runs[[index]]$analysis_input_sha256 <- bundle$rows_sha256",
            "source_lineage_with_external_compact_analysis_rows",
        ):
            self.assertIn(anchor, runner)

    def test_model_defenses_scaled_gradient_and_pointwise_intervals(
        self,
    ) -> None:
        model = self.read_r_source("R/model.R")
        design_start = model.index(
            "fixed_design_diagnostics <- function(formula, rows)"
        )
        design_end = model.index(
            "\noptimizer_diagnostics <- function",
            design_start,
        )
        design = model[design_start:design_end]
        for anchor in (
            "tryCatch(",
            "stats::model.frame(",
            "stats::model.matrix(",
            "error = function(error)",
            "error = conditionMessage(error)",
        ):
            self.assertIn(anchor, design)

        fit = model[model.index("fit_confirmatory_model <- function("):]
        for anchor in (
            "if (!is.null(design$error))",
            "fixed-effects design could not be constructed:",
            "length(unique(rows$prior_strength)) < 2L",
            (
                "Experiment A requires at least two distinct "
                "prior_strength values"
            ),
            "length(unique(rows$turn)) < 2L",
            "Experiment B requires at least two retained turns",
            "not_estimable_result(",
            "solve(chol(hessian), raw_gradient)",
            "raw_gradient = as.list(raw_gradient)",
            "curvature_scaled_gradient",
            "max_absolute_raw_gradient",
            "max_absolute_scaled_gradient",
            "convergence$max_absolute_scaled_gradient <=",
            "estimation_spec$gradient_tolerance",
            "pointwise_unadjusted_confidence_lower",
            "pointwise_unadjusted_confidence_upper",
            "standardized_pointwise_unadjusted_confidence_lower",
            "standardized_pointwise_unadjusted_confidence_upper",
            "specs = ~ mechanism",
            "A:calibration-residual:",
            "A_primary_signed_calibration_mechanism_vs_balanced",
        ):
            self.assertIn(anchor, model if anchor not in fit else fit)
        self.assertNotIn("max_absolute_gradient =", model)
        self.assertIsNone(
            re.search(
                (
                    r"(?m)^\s+(?:standardized_)?"
                    r"(?:confidence|pointwise_confidence)_"
                    r"(?:lower|upper)\s*="
                ),
                model,
            )
        )

    def test_output_inventory_and_result_schema_agree(self) -> None:
        declaration = self.load_json("analysis-spec.json")
        result_schema = self.load_json("analysis-result.schema.json")
        outputs = set(declaration["outputs"])
        self.assertEqual(
            outputs,
            {
                "analysis-result.json",
                "diagnostics.json",
                "input-manifest.json",
                "analysis-rows.csv",
                "fixed-effects.csv",
                "omnibus-tests.csv",
                "random-effects.csv",
                "contrasts.csv",
                "session-info.txt",
                "SHA256SUMS",
            },
        )
        self.assertEqual(
            result_schema["properties"]["claim_status"]["const"],
            "not_claimed",
        )
        self.assertEqual(
            result_schema["properties"]["schema_version"]["const"],
            3,
        )
        self.assertEqual(
            result_schema["properties"]["analysis_id"]["const"],
            "cape-loop-confirmatory-mixed-effects-v3",
        )
        self.assertEqual(
            result_schema["properties"]["reference_updater"]["enum"],
            ["exact_action_aware", "fitted_action_aware"],
        )
        self.assertEqual(
            result_schema["properties"]["outcome"]["enum"],
            ["calibration_residual", "terminal_error"],
        )
        self.assertEqual(
            result_schema["allOf"],
            [
                {
                    "if": {
                        "properties": {"experiment": {"const": "A"}},
                        "required": ["experiment"],
                    },
                    "then": {
                        "properties": {
                            "reference_updater": {
                                "const": "exact_action_aware"
                            },
                            "outcome": {
                                "const": "calibration_residual"
                            },
                        }
                    },
                },
                {
                    "if": {
                        "properties": {"experiment": {"const": "B"}},
                        "required": ["experiment"],
                    },
                    "then": {
                        "properties": {
                            "reference_updater": {
                                "const": "fitted_action_aware"
                            },
                            "outcome": {"const": "terminal_error"},
                        }
                    },
                },
            ],
        )
        self.assertEqual(
            set(result_schema["properties"]["status"]["enum"]),
            {"complete", "not_confirmatory", "not_estimable"},
        )
        estimation_schema = result_schema["properties"]["estimation"]
        self.assertIn("gradient_scaling", estimation_schema["required"])
        self.assertEqual(
            estimation_schema["properties"]["gradient_scaling"]["const"],
            "inverse_cholesky_hessian",
        )

    def test_interpretation_limits_are_explicit(self) -> None:
        readme = (ANALYSIS / "README.md").read_text(encoding="utf-8")
        guide = (ROOT / "docs" / "metrics.md").read_text(encoding="utf-8")
        for text in (readme, guide):
            normalized = " ".join(text.split())
            self.assertIn(
                "does not alone establish all five self-confirmation clauses",
                normalized,
            )
            self.assertIn(
                "pointwise_unadjusted_confidence_lower",
                normalized,
            )
            self.assertIn(
                "Holm adjustment applies to p-values",
                normalized,
            )
        normalized_readme = " ".join(readme.split())
        self.assertIn(
            "Positive contrasts mean the treatment causes more over-updating",
            normalized_readme,
        )
        self.assertIn(
            "descriptive secondary magnitude estimand",
            normalized_readme,
        )

    def test_every_locked_package_has_a_basic_immutable_record(self) -> None:
        lock = self.load_json("renv.lock")
        self.assertEqual(lock["R"]["Version"], "4.6.1")
        self.assertTrue(lock["Packages"])
        for package, record in lock["Packages"].items():
            with self.subTest(package=package):
                self.assertEqual(record["Package"], package)
                self.assertIsInstance(record["Version"], str)
                self.assertTrue(record["Version"])
                self.assertEqual(record["Source"], "Repository")
                self.assertEqual(record["Repository"], "CRAN")

    def test_runtime_version_check_uses_r_package_version_semantics(
        self,
    ) -> None:
        runner = self.read_r_source("run_analysis.R")
        self.assertIn("base::package_version(expected)", runner)
        self.assertIn(
            "if (!isTRUE(observed_version == expected_version))",
            runner,
        )
        self.assertNotIn(
            "if (!identical(observed, expected))",
            runner,
        )


if __name__ == "__main__":
    unittest.main()
