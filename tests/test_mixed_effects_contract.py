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
                "update_error ~ updater * mechanism + domain + "
                "prior_strength + (1 + mechanism | user) + (1 | scenario)"
            ),
        )
        self.assertEqual(
            experiments["A"]["proposal_formula"],
            (
                "UpdateError ~ Updater * Mechanism + Domain + "
                "PriorStrength + (1 + Mechanism | User) + (1 | Scenario)"
            ),
        )
        self.assertEqual(
            experiments["B"]["formula"],
            (
                "terminal_error ~ updater * policy * initial_profile + "
                "domain + turn + (1 + policy | user) + (1 | scenario)"
            ),
        )
        self.assertEqual(
            experiments["B"]["proposal_formula"],
            (
                "TerminalError ~ Updater * Policy * InitialProfile + "
                "Domain + Turn + (1 + Policy | User) + (1 | Scenario)"
            ),
        )
        self.assertEqual(
            declaration["factor_coding"],
            {
                "contrasts": ["contr.treatment", "contr.poly"],
                "updater_reference": "fitted_action_aware",
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
                    "target-minus-aware-within-policy-conditioned-mechanism"
                ),
                "mechanisms": ["restricted", "default", "suggested"],
                "expression": (
                    "target_updater[mechanism] - "
                    "fitted_action_aware[mechanism]"
                ),
                "multiplicity": (
                    "Holm within the three-mechanism primary family"
                ),
            },
        )
        self.assertEqual(
            experiments["A"]["secondary_contrasts"],
            [
                {
                    "id": "balanced-target-minus-aware",
                    "expression": (
                        "target_updater[balanced] - "
                        "fitted_action_aware[balanced]"
                    ),
                },
                {
                    "id": (
                        "updater-by-mechanism-difference-in-differences"
                    ),
                    "expression": (
                        "(target_updater[mechanism] - "
                        "fitted_action_aware[mechanism]) - "
                        "(target_updater[balanced] - "
                        "fitted_action_aware[balanced])"
                    ),
                    "mechanisms": [
                        "restricted",
                        "default",
                        "suggested",
                    ],
                },
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
        self.assertEqual(experiment_b["analysis_unit"], "retained_turn")
        self.assertEqual(
            experiment_b["outcome_source"],
            "turns[].belief_after marginal Brier against top-level theta",
        )
        self.assertEqual(experiment_b["turn_source"], "turns[].turn + 1")
        self.assertEqual(
            experiment_b["terminal_consistency_check"],
            "final reconstructed turn error equals retained terminal_error",
        )
        self.assertEqual(experiment_b["outcome_name"], "terminal_error")

    def test_r_input_contract_binds_source_and_reconstructs_turn_rows(
        self,
    ) -> None:
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
            "the same Experiment B horizon",
            "if (!identical(reference, candidate))",
            "marginal_brier_from_retained_belief <- function(",
            "turn$belief_after",
            "theta <- validate_theta_values(row$theta",
            "sum((numeric_probabilities - expected)^2)",
            "sum(attribute_scores) / 3",
            "source_turn_index = observed",
            "turn = observed + 1L",
            "source_turn_index = turn_values$source_turn_index",
            "terminal_error = turn_values$terminal_error",
            "reconstructed_terminal_error <- tail(",
            "abs(reconstructed_terminal_error - retained_terminal_error)",
            "final turn Brier error differs from retained terminal_error",
        ):
            self.assertIn(anchor, source)
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
            'abort_analysis("analysis output cannot be inside a source run")'
        )
        parent_creation = function.index("dir.create(parent")
        self.assertLess(source_check, rejection)
        self.assertLess(rejection, parent_creation)

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
        guide = (ROOT / "docs" / "mixed-effects-analysis.md").read_text(
            encoding="utf-8"
        )
        for text in (readme, guide):
            normalized = " ".join(text.split())
            self.assertIn(
                "do not test the direction or magnitude of the target's "
                "belief update",
                normalized,
            )
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


if __name__ == "__main__":
    unittest.main()
