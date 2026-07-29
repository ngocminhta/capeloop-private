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
        "update_error ~ updater * mechanism + domain + prior_strength + "
        "(1 + mechanism | user) + (1 | scenario)"
    ),
    "B": (
        "terminal_error ~ updater * policy * initial_profile + domain + turn + "
        "(1 + policy | user) + (1 | scenario)"
    ),
}

PROPOSAL_FORMULAS = {
    "A": (
        "UpdateError ~ Updater * Mechanism + Domain + PriorStrength + "
        "(1 + Mechanism | User) + (1 | Scenario)"
    ),
    "B": (
        "TerminalError ~ Updater * Policy * InitialProfile + Domain + Turn + "
        "(1 + Policy | User) + (1 | Scenario)"
    ),
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
    "updater_reference": "fitted_action_aware",
    "experiment_a_mechanism_reference": "balanced",
    "experiment_b_policy_reference": "balanced",
    "experiment_b_initial_profile_reference": "incorrect",
    "domain_reference_preference": "travel",
}

EXPERIMENT_SEMANTICS = {
    "A": {
        "run_kind": "provenance_audit",
        "input_file": "events/experiment-a.jsonl",
        "exclusion_file": "events/experiment-a-exclusions.jsonl",
        "response_mode": "naturally_sampled",
        "outcome_source": "metrics.acue",
        "outcome_name": "update_error",
        "required_mechanisms": [
            "balanced",
            "restricted",
            "default",
            "suggested",
        ],
        "scenario_mapping": "run_id + context.scenario_id",
        "user_mapping": "run_id + user_id",
    },
    "B": {
        "run_kind": "closed_loop",
        "input_file": "events/experiment-b-trajectories.jsonl",
        "analysis_unit": "retained_turn",
        "outcome_source": (
            "turns[].belief_after marginal Brier against top-level theta"
        ),
        "outcome_name": "terminal_error",
        "turn_source": "turns[].turn + 1",
        "terminal_consistency_check": (
            "final reconstructed turn error equals retained terminal_error"
        ),
        "required_policies": ["balanced", "soft_profile_conditioned"],
        "required_initial_profiles": ["incorrect"],
        "scenario_mapping": "run_id + crn_key",
        "user_mapping": "run_id + user_id",
    },
}

PRIMARY_CONTRASTS = {
    "A": {
        "id": "target-minus-aware-within-policy-conditioned-mechanism",
        "mechanisms": ["restricted", "default", "suggested"],
        "expression": (
            "target_updater[mechanism] - "
            "fitted_action_aware[mechanism]"
        ),
        "multiplicity": "Holm within the three-mechanism primary family",
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
    "A": [
        {
            "id": "balanced-target-minus-aware",
            "expression": (
                "target_updater[balanced] - "
                "fitted_action_aware[balanced]"
            ),
        },
        {
            "id": "updater-by-mechanism-difference-in-differences",
            "expression": (
                "(target_updater[mechanism] - "
                "fitted_action_aware[mechanism]) - "
                "(target_updater[balanced] - "
                "fitted_action_aware[balanced])"
            ),
            "mechanisms": ["restricted", "default", "suggested"],
        },
    ],
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
        if declaration.get("proposal_formula") != PROPOSAL_FORMULAS[experiment]:
            raise ContractError(
                f"Experiment {experiment} proposal_formula changed"
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
        "validate_analysis_rows",
        "fit_confirmatory_model",
        "write_output_checksums",
        "not_claimed",
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
        'abort_analysis("analysis output cannot be inside a source run")'
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
        "the same Experiment B horizon",
        "marginal_brier_from_retained_belief <- function(",
        "turn$belief_after",
        "theta <- validate_theta_values(row$theta",
        "sum((numeric_probabilities - expected)^2)",
        "sum(attribute_scores) / 3",
        "source_turn_index = observed",
        "turn = observed + 1L",
        "source_turn_index = turn_values$source_turn_index",
        "terminal_error = turn_values$terminal_error",
        "retained turn count differs from config.experiment.turns",
        "reconstructed_terminal_error <- tail(",
        "abs(reconstructed_terminal_error - retained_terminal_error)",
        "final turn Brier error differs from retained terminal_error",
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
