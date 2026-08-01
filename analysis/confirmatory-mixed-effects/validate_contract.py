#!/usr/bin/env python3
"""Static validation for the optional R analysis contract.

This deliberately uses only the Python standard library so core CI can verify
the scientific contract even when R is not installed. It validates declarations
and source anchors; it does not pretend to execute or validate a model fit.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import json
import re
import sys


ROOT = Path(__file__).resolve().parent

FORMULAS = {
    "A": (
        "calibration_residual ~ mechanism + domain + prior_strength + "
        "(1 + mechanism | user) + (1 | scenario)"
    ),
    "B": (
        "terminal_error ~ updater * policy * initial_profile + domain + turn + "
        "(1 + policy | user) + (1 | scenario) + (1 | crn_set)"
    ),
}

POOLED_FORMULAS = {
    experiment: f"{formula} + (1 | replicate)"
    for experiment, formula in FORMULAS.items()
}

PROPOSAL_FORMULAS = {
    "A": (
        "CalibrationResidual ~ Mechanism + Domain + PriorStrength + "
        "(1 + Mechanism | User) + (1 | Scenario)"
    ),
    "B": (
        "TerminalError ~ Updater * Policy * InitialProfile + Domain + Turn + "
        "(1 + Policy | User) + (1 | Scenario) + (1 | CRNSet)"
    ),
}

POOLED_PROPOSAL_FORMULAS = {
    experiment: f"{formula} + (1 | Replicate)"
    for experiment, formula in PROPOSAL_FORMULAS.items()
}

ESTIMATION = {
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
}

FACTOR_CODING = {
    "contrasts": ["contr.treatment", "contr.poly"],
    "experiment_a_oracle_reference": "exact_action_aware",
    "experiment_b_updater_reference": "fitted_action_aware",
    "experiment_a_mechanism_reference": "balanced",
    "experiment_b_policy_reference": "balanced",
    "experiment_b_initial_profile_reference": "incorrect",
    "domain_reference_preference": "travel",
}

EXPERIMENT_SEMANTICS = {
    "A": {
        "run_kind": "provenance_audit",
        "input_file": "analysis/experiment-a-rows.jsonl",
        "exclusion_file": "analysis/experiment-a-exclusions.jsonl",
        "legacy_input_file": "events/experiment-a.jsonl",
        "legacy_exclusion_file": "events/experiment-a-exclusions.jsonl",
        "bundle_analysis_unit": "updater_trial",
        "input_row_schema_version": 2,
        "bundle_outcome_derivation": (
            "retained anchor-directional system minus exact log-odds update "
            "as calibration_residual; exact and fitted update errors retained "
            "as required secondary magnitude diagnostics"
        ),
        "response_mode": "controlled_anchor",
        "analysis_track": "same_response_provenance",
        "reference_basis": "exact_action_aware",
        "outcome_source": (
            "runner-native compact calibration_residual equal to "
            "anchor-directional system_log_odds_update minus "
            "exact_log_odds_update"
        ),
        "outcome_name": "calibration_residual",
        "retained_calibration_fields": [
            "system_log_odds_update",
            "exact_log_odds_update",
            "fitted_log_odds_update",
            "calibration_residual",
        ],
        "secondary_outcomes": [
            {
                "name": "exact_update_error",
                "source": (
                    "runner-native compact exact_update_error derived from "
                    "metrics.exact_acue"
                ),
                "required": True,
                "role": "descriptive_secondary_absolute_magnitude",
            },
            {
                "name": "fitted_update_error",
                "source": (
                    "runner-native compact fitted_update_error derived from "
                    "metrics.acue"
                ),
                "required": True,
                "role": "secondary_reference_diagnostic",
            }
        ],
        "secondary_estimands": [
            {
                "id": "target-exact-update-error-by-mechanism",
                "outcome": "exact_update_error",
                "aggregation": "descriptive distribution and mean by mechanism",
                "inferential_status": (
                    "not fitted by the primary signed-residual model"
                ),
            }
        ],
        "required_mechanisms": [
            "balanced",
            "restricted",
            "ranking",
            "default",
            "suggested",
        ],
        "scenario_mapping": "scenario_id",
        "user_mapping": "user_id",
        "replicate_mapping": "run_id",
        "pooling_policy": (
            "Pooled fits require one identical run.seed and add a crossed "
            "run-level random intercept. Raw user_id and scenario_id remain "
            "shared clusters across same-seed reruns. Different seeds are "
            "analyzed separately as robustness replicates and never increase "
            "the independent-user count."
        ),
    },
    "B": {
        "run_kind": "closed_loop",
        "input_file": "analysis/experiment-b-turns.jsonl",
        "legacy_input_file": "events/experiment-b-trajectories.jsonl",
        "input_row_schema_version": 1,
        "bundle_analysis_unit": "retained_trajectory_turn",
        "bundle_outcome_derivation": (
            "per-turn marginal Brier from retained belief_after "
            "against immutable trajectory theta"
        ),
        "analysis_unit": "retained_turn",
        "outcome_source": (
            "runner-native compact terminal_error derived from "
            "turns[].belief_after marginal Brier against top-level theta"
        ),
        "outcome_name": "terminal_error",
        "turn_source": "compact turn equals source_turn_index + 1",
        "terminal_consistency_check": (
            "final compact turn error equals retained_terminal_error"
        ),
        "required_policies": ["balanced", "soft_profile_conditioned"],
        "required_initial_profiles": ["incorrect"],
        "scenario_mapping": "turn.scenario_id",
        "crn_set_mapping": "run_id + crn_key",
        "user_mapping": "user_id",
        "replicate_mapping": "run_id",
        "pooling_policy": (
            "Pooled fits require one identical run.seed and add a crossed "
            "run-level random intercept. Raw user_id and scenario_id remain "
            "shared clusters across same-seed reruns. Different seeds are "
            "analyzed separately as robustness replicates and never increase "
            "the independent-user count."
        ),
    },
}

PRIMARY_CONTRASTS = {
    "A": {
        "id": "target-calibration-residual-mechanism-minus-balanced",
        "mechanisms": ["restricted", "ranking", "default", "suggested"],
        "expression": (
            "calibration_residual[target_updater, mechanism] - "
            "calibration_residual[target_updater, balanced]"
        ),
        "multiplicity": "Holm within the four-mechanism primary family",
    },
    "B": {
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
}

SECONDARY_CONTRASTS = {
    "A": [],
    "B": [
        {
            "id": "incorrect-profile-target-policy-effect",
            "expression": (
                "target_updater[soft_profile_conditioned] - "
                "target_updater[balanced]"
            ),
        },
        {
            "id": "incorrect-profile-target-minus-aware-by-policy",
            "expression": (
                "target_updater[policy] - fitted_action_aware[policy]"
            ),
            "policies": ["balanced", "soft_profile_conditioned"],
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
}

OUTPUTS = [
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
]

DIRECT_PACKAGES = {
    "digest": "0.6.39",
    "emmeans": "2.0.4",
    "jsonlite": "2.0.0",
    "lme4": "2.0-6",
    "lmerTest": "3.2-1",
    "renv": "1.2.3",
}

REQUIRED_R_SOURCES = (
    "run_analysis.R",
    "restore.R",
    "R/io.R",
    "R/model.R",
)


class ContractError(ValueError):
    """Raised when a checked-in analysis declaration is inconsistent."""


def _unique_object(pairs: Iterable[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _load_json(relative: str) -> dict[str, object]:
    path = ROOT / relative
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read {relative}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{relative} must contain one JSON object")
    return value


def _read_text(relative: str) -> str:
    path = ROOT / relative
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ContractError(f"cannot read {relative}: {exc}") from exc


def _require_anchors(
    relative: str,
    text: str,
    anchors: Iterable[str],
) -> None:
    for anchor in anchors:
        if anchor not in text:
            raise ContractError(f"{relative} is missing contract anchor {anchor}")


def _source_block(
    relative: str,
    text: str,
    start: str,
    end: str,
) -> str:
    start_index = text.find(start)
    if start_index < 0:
        raise ContractError(f"{relative} is missing source block {start}")
    end_index = text.find(end, start_index + len(start))
    if end_index < 0:
        raise ContractError(
            f"{relative} source block {start} has no terminator {end}"
        )
    return text[start_index:end_index]


def _description_imports() -> dict[str, str]:
    text = _read_text("DESCRIPTION")
    match = re.search(r"(?ms)^Imports:\n(?P<body>(?:    .+\n?)+)", text)
    if match is None:
        raise ContractError("DESCRIPTION is missing Imports")
    result: dict[str, str] = {}
    for item in match.group("body").split(","):
        normalized = " ".join(item.split())
        package_match = re.fullmatch(
            r"([A-Za-z][A-Za-z0-9.]*) \(== ([^)]+)\)",
            normalized,
        )
        if package_match is None:
            raise ContractError(
                f"DESCRIPTION import is not exactly pinned: {normalized!r}"
            )
        result[package_match.group(1)] = package_match.group(2)
    return result


def validate() -> None:
    spec = _load_json("analysis-spec.json")
    if spec.get("schema_version") != 3:
        raise ContractError("analysis spec schema_version must be 3")
    if spec.get("analysis_id") != "cape-loop-confirmatory-mixed-effects-v3":
        raise ContractError("analysis spec must use the v3 analysis ID")
    if spec.get("status") != "analysis_protocol_not_results":
        raise ContractError("analysis spec must declare that it is not results")
    if spec.get("family") != "gaussian" or spec.get("link") != "identity":
        raise ContractError("analysis family/link must remain Gaussian/identity")
    if spec.get("estimation") != ESTIMATION:
        raise ContractError("analysis estimation declaration changed")
    if spec.get("factor_coding") != FACTOR_CODING:
        raise ContractError("analysis factor coding declaration changed")
    if spec.get("outputs") != OUTPUTS:
        raise ContractError("analysis output inventory changed")
    experiments = spec.get("experiments")
    if not isinstance(experiments, dict) or set(experiments) != {"A", "B"}:
        raise ContractError("analysis spec must define exactly Experiments A and B")
    for experiment, expected_formula in FORMULAS.items():
        declaration = experiments[experiment]
        if not isinstance(declaration, dict):
            raise ContractError(f"Experiment {experiment} must be an object")
        if declaration.get("formula") != expected_formula:
            raise ContractError(
                f"Experiment {experiment} formula differs from the proposal"
            )
        if declaration.get("pooled_formula") != POOLED_FORMULAS[experiment]:
            raise ContractError(
                f"Experiment {experiment} pooled_formula changed"
            )
        if declaration.get("proposal_formula") != PROPOSAL_FORMULAS[experiment]:
            raise ContractError(
                f"Experiment {experiment} proposal_formula changed"
            )
        if (
            declaration.get("pooled_proposal_formula")
            != POOLED_PROPOSAL_FORMULAS[experiment]
        ):
            raise ContractError(
                f"Experiment {experiment} pooled_proposal_formula changed"
            )
        for key, expected in EXPERIMENT_SEMANTICS[experiment].items():
            if declaration.get(key) != expected:
                raise ContractError(
                    f"Experiment {experiment} {key} declaration changed"
                )
        if declaration.get("primary_contrast") != PRIMARY_CONTRASTS[experiment]:
            raise ContractError(
                f"Experiment {experiment} primary contrast changed"
            )
        if (
            declaration.get("secondary_contrasts")
            != SECONDARY_CONTRASTS[experiment]
        ):
            raise ContractError(
                f"Experiment {experiment} secondary contrasts changed"
            )

    lock = _load_json("renv.lock")
    r_record = lock.get("R")
    if not isinstance(r_record, dict) or r_record.get("Version") != "4.6.1":
        raise ContractError("renv.lock must pin R 4.6.1")
    packages = lock.get("Packages")
    if not isinstance(packages, dict) or not packages:
        raise ContractError("renv.lock is missing Packages")
    for package, record in packages.items():
        if not isinstance(record, dict):
            raise ContractError(f"renv.lock package {package} is not an object")
        if record.get("Package") != package:
            raise ContractError(
                f"renv.lock package record name differs for {package}"
            )
        version = record.get("Version")
        if not isinstance(version, str) or not version:
            raise ContractError(
                f"renv.lock package {package} has no pinned Version"
            )
        if (
            record.get("Source") != "Repository"
            or record.get("Repository") != "CRAN"
        ):
            raise ContractError(
                f"renv.lock package {package} is not pinned to CRAN"
            )
    for package, version in DIRECT_PACKAGES.items():
        record = packages.get(package)
        if not isinstance(record, dict) or record.get("Version") != version:
            raise ContractError(
                f"renv.lock must pin {package} exactly to {version}"
            )

    imports = _description_imports()
    expected_imports = {
        package: version
        for package, version in DIRECT_PACKAGES.items()
        if package != "renv"
    }
    if imports != expected_imports:
        raise ContractError(
            "DESCRIPTION imports differ from direct locked dependencies"
        )

    result_schema = _load_json("analysis-result.schema.json")
    if (
        result_schema.get("$id")
        != "urn:cape-loop:schema:confirmatory-mixed-effects-result:v3"
    ):
        raise ContractError("result schema must use the v3 identifier")
    required = result_schema.get("required")
    for field in (
        "status",
        "claim_status",
        "formula",
        "fit",
        "contrast_families",
        "artifacts",
    ):
        if not isinstance(required, list) or field not in required:
            raise ContractError(f"result schema does not require {field}")
    schema_properties = result_schema.get("properties")
    if not isinstance(schema_properties, dict):
        raise ContractError("result schema has no properties object")
    if schema_properties.get("schema_version") != {"const": 3}:
        raise ContractError("result schema must require schema_version 3")
    if schema_properties.get("analysis_id") != {
        "const": "cape-loop-confirmatory-mixed-effects-v3"
    }:
        raise ContractError("result schema must require the v3 analysis ID")
    if schema_properties.get("reference_updater") != {
        "enum": ["exact_action_aware", "fitted_action_aware"]
    }:
        raise ContractError(
            "result schema must distinguish exact A and fitted B references"
        )
    if schema_properties.get("outcome") != {
        "enum": ["calibration_residual", "terminal_error"]
    }:
        raise ContractError(
            "result schema must distinguish signed A and terminal B outcomes"
        )
    conditional_boundaries = result_schema.get("allOf")
    expected_boundaries = [
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
                    "outcome": {"const": "calibration_residual"},
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
    ]
    if conditional_boundaries != expected_boundaries:
        raise ContractError(
            "result schema must contain A/B reference-outcome boundaries"
        )
    estimation_schema = schema_properties.get("estimation")
    if not isinstance(estimation_schema, dict):
        raise ContractError("result schema has no estimation object")
    estimation_required = estimation_schema.get("required")
    if (
        not isinstance(estimation_required, list)
        or estimation_required != list(ESTIMATION)
    ):
        raise ContractError(
            "result schema estimation requirements differ from the spec"
        )
    estimation_properties = estimation_schema.get("properties")
    if not isinstance(estimation_properties, dict):
        raise ContractError("result schema estimation properties are missing")
    gradient_schema = estimation_properties.get("gradient_scaling")
    if (
        not isinstance(gradient_schema, dict)
        or gradient_schema.get("const")
        != ESTIMATION["gradient_scaling"]
    ):
        raise ContractError(
            "result schema must require inverse-Cholesky Hessian "
            "gradient scaling"
        )

    for relative in REQUIRED_R_SOURCES:
        path = ROOT / relative
        if not path.is_file() or path.stat().st_size == 0:
            raise ContractError(f"missing R source {relative}")

    runner = _read_text("run_analysis.R")
    _require_anchors("run_analysis.R", runner, (
        "analysis-spec.json",
        "base::package_version(expected)",
        "if (!isTRUE(observed_version == expected_version))",
        "verify_source_run",
        "verify_compact_bundle",
        "--compact-bundle",
        "exactly one per --run in the same order",
        "validate_analysis_rows",
        "fit_confirmatory_model",
        "write_output_checksums",
        "not_claimed",
        "experiment_a_oracle_reference",
        "experiment_b_updater_reference",
        "target_updater_controlled_anchor_event",
        "checksum_bound_anchor_directional_exact_oracle_calibration_residual",
        "pooled_analysis <- length(source_runs) > 1L",
        "experiment_spec$pooled_formula",
        "pooled_run_random_intercept = pooled_analysis",
        'options(\n  contrasts = unlist(spec$factor_coding$contrasts',
    ))
    safe_output = _source_block(
        "run_analysis.R",
        runner,
        "safe_output_path <- function(output, source_paths)",
        "\nargs <- parse_arguments",
    )
    source_check_index = safe_output.find("for (source in source_paths)")
    rejection_index = safe_output.find(
        "analysis output cannot be inside a source run or compact bundle"
    )
    parent_creation_index = safe_output.find("dir.create(parent")
    if not (
        0 <= source_check_index < rejection_index < parent_creation_index
    ):
        raise ContractError(
            "safe_output_path must reject source-contained output before "
            "creating its parent"
        )

    io_source = _read_text("R/io.R")
    _require_anchors("R/io.R", io_source, (
        "canonical_config_sha256 <- function(path)",
        "config.resolved.json must be canonical one-line JSON",
        "python_ascii_fragment <- function(code_point)",
        "if (code_point <= 127L)",
        'sprintf("\\\\u%04x", code_point)',
        "high <- 55296L",
        "low <- 56320L",
        'Encoding(payload_text) <- "UTF-8"',
        "utf8ToInt(payload_text)",
        "charToRaw(python_canonical_payload)",
        "observed_config_sha256 <- canonical_config_sha256(config_path)",
        "manifest$config_sha256",
        "source$manifest$source_sha256",
        "length(unique(source_digests)) != 1L",
        "pooled source runs must have identical manifest.source_sha256 values",
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
        "assert_exact_fields <- function(value, expected, label)",
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
        "source$summary$controlled_row_count",
        "compact Experiment A source_record_index values must cover",
        "compact Experiment B source_record_index values must cover",
        "source summary compact exclusion declaration differs",
        "source$summary$analysis_row_count",
        "row$retained_terminal_error",
        "row$same_history_shadow",
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
    ))
    if re.search(
        r"experiment_config\s*\$turns\s*<-\s*NULL",
        io_source,
    ):
        raise ContractError(
            "Experiment B horizon-removal pooling workaround must remain absent"
        )
    same_design = _source_block(
        "R/io.R",
        io_source,
        "assert_same_design <- function(source_runs, experiment)",
        "\nnormalize_experiment_a <- function",
    )
    if "experiment = experiment_config" not in same_design:
        raise ContractError(
            "pooled design signature must retain the full experiment config"
        )
    if "if (!identical(reference, candidate))" not in same_design:
        raise ContractError("pooled source designs must be exactly identical")
    if "population = source$config$population" not in same_design:
        raise ContractError(
            "pooled design signature must retain population policy"
        )

    model = _read_text("R/model.R")
    _require_anchors("R/model.R", model, (
        "lmerTest::lmer",
        "lme4::isSingular",
        "emmeans::emmeans",
        'p.adjust(raw$p.value, method = "holm")',
        "rank_deficient",
        "pointwise_unadjusted_confidence_lower",
        "pointwise_unadjusted_confidence_upper",
        "standardized_pointwise_unadjusted_confidence_lower",
        "standardized_pointwise_unadjusted_confidence_upper",
        "raw_gradient = as.list(raw_gradient)",
        "curvature_scaled_gradient",
        "max_absolute_raw_gradient",
        "max_absolute_scaled_gradient",
        "solve(chol(hessian), raw_gradient)",
        "A:calibration-residual:",
        "A_primary_signed_calibration_mechanism_vs_balanced",
    ))
    if re.search(
        (
            r"(?m)^\s+(?:standardized_)?"
            r"(?:confidence|pointwise_confidence)_(?:lower|upper)\s*="
        ),
        model,
    ):
        raise ContractError(
            "confidence interval output names must explicitly say "
            "pointwise_unadjusted"
        )
    if "max_absolute_gradient =" in model:
        raise ContractError(
            "ambiguous unscaled max_absolute_gradient output must remain absent"
        )

    design_diagnostics = _source_block(
        "R/model.R",
        model,
        "fixed_design_diagnostics <- function(formula, rows)",
        "\noptimizer_diagnostics <- function",
    )
    _require_anchors("R/model.R fixed_design_diagnostics", design_diagnostics, (
        "tryCatch(",
        "stats::model.frame(",
        "stats::model.matrix(",
        "error = function(error)",
        "error = conditionMessage(error)",
    ))

    fit_start = model.find("fit_confirmatory_model <- function(")
    if fit_start < 0:
        raise ContractError(
            "R/model.R is missing source block fit_confirmatory_model"
        )
    fit_model = model[fit_start:]
    _require_anchors("R/model.R fit_confirmatory_model", fit_model, (
        "not_estimable_result(",
        "if (!is.null(design$error))",
        "fixed-effects design could not be constructed:",
        "length(unique(rows$prior_strength)) < 2L",
        "Experiment A requires at least two distinct prior_strength values",
        "length(unique(rows$turn)) < 2L",
        "Experiment B requires at least two retained turns",
        "convergence$max_absolute_scaled_gradient <=",
        "estimation_spec$gradient_tolerance",
    ))


def main() -> int:
    try:
        validate()
    except ContractError as exc:
        print(f"mixed-effects contract invalid: {exc}", file=sys.stderr)
        return 1
    print("mixed-effects contract valid (static; no model was fitted)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
