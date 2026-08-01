#!/usr/bin/env Rscript

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (length(script_arg) != 1L) {
  stop("run_analysis.R must be executed with Rscript", call. = FALSE)
}
project_dir <- normalizePath(
  dirname(sub("^--file=", "", script_arg[[1L]])),
  mustWork = TRUE
)
if (!requireNamespace("renv", quietly = TRUE)) {
  stop(
    "renv is unavailable; run Rscript restore.R first",
    call. = FALSE
  )
}
renv::load(project = project_dir, quiet = TRUE)

lock_path <- file.path(project_dir, "renv.lock")
if (!file.exists(lock_path)) {
  stop("renv.lock is missing", call. = FALSE)
}
lock <- jsonlite::fromJSON(lock_path, simplifyVector = FALSE)
if (as.character(getRversion()) != lock$R$Version) {
  stop(
    "R version differs from renv.lock: expected ",
    lock$R$Version,
    ", observed ",
    as.character(getRversion()),
    call. = FALSE
  )
}
for (package in names(lock$Packages)) {
  expected <- lock$Packages[[package]]$Version
  if (!requireNamespace(package, quietly = TRUE)) {
    stop("locked package is not installed: ", package, call. = FALSE)
  }
  expected_version <- tryCatch(
    base::package_version(expected),
    error = function(error) {
      stop(
        "invalid locked package version for ",
        package,
        ": ",
        expected,
        call. = FALSE
      )
    }
  )
  observed_version <- utils::packageVersion(package)
  if (!isTRUE(observed_version == expected_version)) {
    stop(
      "package version mismatch for ",
      package,
      ": expected ",
      expected,
      ", observed ",
      as.character(observed_version),
      call. = FALSE
    )
  }
}

source(file.path(project_dir, "R", "io.R"), local = FALSE)
source(file.path(project_dir, "R", "model.R"), local = FALSE)

usage <- function() {
  cat(
    paste(
      "Usage:",
      "  Rscript run_analysis.R --experiment A|B --run RUN_DIR",
      "    [--run RUN_DIR ...] --output OUTPUT_DIR",
      "    [--compact-bundle BUNDLE_DIR ...]",
      "    [--target-updater llm_full_context]",
      "",
      paste(
        "Supply no compact bundles for runner-native inputs, or exactly one",
        "bundle per --run in the same order."
      ),
      paste(
        "The output directory must not exist and must be outside every source",
        "run and compact bundle."
      ),
      sep = "\n"
    )
  )
}

parse_arguments <- function(arguments) {
  result <- list(
    experiment = NULL,
    runs = character(),
    compact_bundles = character(),
    output = NULL,
    target_updater = "llm_full_context"
  )
  index <- 1L
  while (index <= length(arguments)) {
    flag <- arguments[[index]]
    if (flag %in% c("-h", "--help")) {
      usage()
      quit(save = "no", status = 0L)
    }
    if (
      !(flag %in% c(
        "--experiment",
        "--run",
        "--compact-bundle",
        "--output",
        "--target-updater"
      ))
    ) {
      abort_analysis("unknown argument: ", flag)
    }
    if (index == length(arguments)) {
      abort_analysis("missing value after ", flag)
    }
    value <- arguments[[index + 1L]]
    if (!nzchar(value) || startsWith(value, "--")) {
      abort_analysis("invalid value after ", flag)
    }
    if (identical(flag, "--experiment")) {
      if (!is.null(result$experiment)) {
        abort_analysis("--experiment may be supplied only once")
      }
      result$experiment <- value
    } else if (identical(flag, "--run")) {
      result$runs <- c(result$runs, value)
    } else if (identical(flag, "--compact-bundle")) {
      result$compact_bundles <- c(result$compact_bundles, value)
    } else if (identical(flag, "--output")) {
      if (!is.null(result$output)) {
        abort_analysis("--output may be supplied only once")
      }
      result$output <- value
    } else {
      result$target_updater <- value
    }
    index <- index + 2L
  }
  if (!(result$experiment %in% c("A", "B"))) {
    abort_analysis("--experiment must be exactly A or B")
  }
  if (!length(result$runs)) {
    abort_analysis("at least one --run is required")
  }
  if (anyDuplicated(result$runs)) {
    abort_analysis("duplicate --run arguments are not allowed")
  }
  if (anyDuplicated(result$compact_bundles)) {
    abort_analysis("duplicate --compact-bundle arguments are not allowed")
  }
  if (
    length(result$compact_bundles) &&
    length(result$compact_bundles) != length(result$runs)
  ) {
    abort_analysis(
      "supply either zero compact bundles or exactly one per --run in the same order"
    )
  }
  if (is.null(result$output)) {
    abort_analysis("--output is required")
  }
  require_scalar_character(result$target_updater, "--target-updater")
  result
}

safe_output_path <- function(output, source_paths) {
  if (file.exists(output) || dir.exists(output)) {
    abort_analysis("output path already exists: ", output)
  }
  if (basename(output) %in% c(".", "..")) {
    abort_analysis("output path must name a new directory")
  }
  unresolved_parent <- path.expand(dirname(output))
  missing_components <- character()
  while (!dir.exists(unresolved_parent)) {
    if (file.exists(unresolved_parent)) {
      abort_analysis("output parent is not a directory: ", unresolved_parent)
    }
    parent_component <- basename(unresolved_parent)
    if (parent_component %in% c("", ".", "..")) {
      abort_analysis("could not resolve output parent safely: ", output)
    }
    missing_components <- c(parent_component, missing_components)
    next_parent <- dirname(unresolved_parent)
    if (identical(next_parent, unresolved_parent)) {
      abort_analysis("could not find an existing output ancestor: ", output)
    }
    unresolved_parent <- next_parent
  }
  resolved_parent <- normalizePath(unresolved_parent, mustWork = TRUE)
  for (component in missing_components) {
    resolved_parent <- file.path(resolved_parent, component)
  }
  destination <- file.path(resolved_parent, basename(output))
  for (source in source_paths) {
    source <- normalizePath(source, mustWork = TRUE)
    if (
      identical(destination, source) ||
      startsWith(destination, paste0(source, .Platform$file.sep))
    ) {
      abort_analysis(
        "analysis output cannot be inside a source run or compact bundle"
      )
    }
  }
  parent <- dirname(destination)
  if (!dir.exists(parent)) {
    dir.create(parent, recursive = TRUE, showWarnings = FALSE)
  }
  if (!dir.exists(parent)) {
    abort_analysis("could not create output parent: ", parent)
  }
  destination
}

args <- parse_arguments(commandArgs(trailingOnly = TRUE))
spec_path <- file.path(project_dir, "analysis-spec.json")
spec <- read_json_object(spec_path, "analysis specification")
if (
  !identical(spec$schema_version, 3L) &&
  !identical(spec$schema_version, 3)
) {
  abort_analysis("unsupported analysis specification version")
}
experiment_spec <- spec$experiments[[args$experiment]]
if (is.null(experiment_spec)) {
  abort_analysis("analysis specification has no requested experiment")
}
reference_updater <- if (identical(args$experiment, "A")) {
  spec$factor_coding$experiment_a_oracle_reference
} else {
  spec$factor_coding$experiment_b_updater_reference
}
reference_updater <- require_scalar_character(
  reference_updater,
  "experiment-specific updater reference"
)

uses_historical_bundles <- length(args$compact_bundles) > 0L
source_runs <- lapply(args$runs, function(path) {
  verify_source_run(
    path,
    experiment_spec,
    use_legacy_input = if (uses_historical_bundles) NULL else FALSE
  )
})
if (anyDuplicated(vapply(source_runs, `[[`, character(1), "run_id"))) {
  abort_analysis("source run IDs must be unique")
}
assert_same_design(source_runs, args$experiment)
pooled_analysis <- length(source_runs) > 1L
experiment_spec$formula <- require_scalar_character(
  if (pooled_analysis) {
    experiment_spec$pooled_formula
  } else {
    experiment_spec$formula
  },
  "selected experiment formula"
)
experiment_spec$proposal_formula <- require_scalar_character(
  if (pooled_analysis) {
    experiment_spec$pooled_proposal_formula
  } else {
    experiment_spec$proposal_formula
  },
  "selected proposal formula"
)
if (uses_historical_bundles) {
  for (index in seq_along(source_runs)) {
    bundle <- verify_compact_bundle(
      args$compact_bundles[[index]],
      source_runs[[index]],
      args$experiment,
      experiment_spec
    )
    source_runs[[index]]$compact_bundle <- bundle
    source_runs[[index]]$analysis_input_path <- bundle$rows_path
    source_runs[[index]]$analysis_input_sha256 <- bundle$rows_sha256
  }
}
output_dir <- safe_output_path(
  args$output,
  c(
    vapply(source_runs, `[[`, character(1), "path"),
    if (uses_historical_bundles) {
      vapply(
        source_runs,
        function(source) source$compact_bundle$path,
        character(1)
      )
    } else {
      character()
    }
  )
)

rows <- if (identical(args$experiment, "A")) {
  do.call(rbind, lapply(source_runs, normalize_experiment_a))
} else {
  do.call(rbind, lapply(source_runs, normalize_experiment_b))
}
rows <- validate_analysis_rows(
  rows,
  args$experiment,
  experiment_spec,
  args$target_updater,
  spec$factor_coding,
  as.integer(spec$estimation$minimum_user_clusters),
  as.integer(spec$estimation$minimum_scenario_clusters)
)
sort_fields <- if (identical(args$experiment, "A")) {
  c(
    "source_run_id",
    "user",
    "scenario",
    "trial_id",
    "updater",
    "mechanism"
  )
} else {
  c(
    "source_run_id",
    "user",
    "scenario",
    "trajectory_id",
    "turn",
    "updater",
    "policy"
  )
}
ordering <- do.call(order, c(unname(rows[sort_fields]), list(method = "radix")))
rows <- rows[ordering, , drop = FALSE]
row.names(rows) <- NULL

options(
  contrasts = unlist(spec$factor_coding$contrasts, use.names = FALSE)
)
model <- fit_confirmatory_model(
  rows,
  args$experiment,
  experiment_spec,
  spec$estimation,
  args$target_updater,
  reference_updater
)

dir.create(output_dir, recursive = FALSE, showWarnings = FALSE)
if (!dir.exists(output_dir)) {
  abort_analysis("could not create output directory: ", output_dir)
}
analysis_rows_path <- file.path(output_dir, "analysis-rows.csv")
write_csv(analysis_rows_path, rows)
write_csv(
  file.path(output_dir, "fixed-effects.csv"),
  model$fixed_effects
)
write_csv(
  file.path(output_dir, "omnibus-tests.csv"),
  model$omnibus_tests
)
write_csv(
  file.path(output_dir, "random-effects.csv"),
  model$random_effects
)
write_csv(
  file.path(output_dir, "contrasts.csv"),
  model$contrasts
)

source_records <- lapply(source_runs, function(source) {
  list(
    run_id = source$run_id,
    population_seed = source$population_seed,
    config_sha256 = source$manifest$config_sha256,
    config_digest_verification = (
      "recomputed_from_retained_python_canonical_payload"
    ),
    source_sha256 = source$manifest$source_sha256,
    resolved_config_file_sha256 = unname(
      source$checksums[["config.resolved.json"]]
    ),
    model_role = source$config$llm$model_role,
    model = source$config$llm$model,
    reasoning_effort = source$config$llm$reasoning_effort,
    configured_turns = source$config$experiment$turns,
    checksum_manifest_sha256 = source$checksum_manifest_sha256,
    input_file = source$input_relative,
    input_sha256 = source$input_sha256,
    input_role = if (is.null(source$compact_bundle)) {
      "runner_native_compact_analysis_rows"
    } else {
      "source_lineage_with_external_compact_analysis_rows"
    },
    analysis_rows_sha256 = source$analysis_input_sha256,
    compact_bundle_manifest_sha256 = if (is.null(source$compact_bundle)) {
      NULL
    } else {
      source$compact_bundle$manifest_sha256
    },
    compact_bundle_checksum_manifest_sha256 = if (
      is.null(source$compact_bundle)
    ) {
      NULL
    } else {
      source$compact_bundle$checksum_manifest_sha256
    },
    exclusion_file = source$exclusion_relative %||% NULL,
    exclusion_sha256 = source$exclusion_sha256,
    exclusion_count = source$exclusion_count
  )
})
source_file_digests <- list(
  analysis_spec_sha256 = sha256_file(spec_path),
  result_schema_sha256 = sha256_file(
    file.path(project_dir, "analysis-result.schema.json")
  ),
  runner_sha256 = sha256_file(file.path(project_dir, "run_analysis.R")),
  io_sha256 = sha256_file(file.path(project_dir, "R", "io.R")),
  model_sha256 = sha256_file(file.path(project_dir, "R", "model.R")),
  lock_sha256 = sha256_file(lock_path)
)
input_manifest <- list(
  schema_version = 3,
  analysis_id = spec$analysis_id,
  experiment = args$experiment,
  source_runs = source_records,
  source_run_count = length(source_runs),
  population_seed = source_runs[[1L]]$population_seed,
  pooled_run_random_intercept = pooled_analysis,
  user_mapping = experiment_spec$user_mapping,
  scenario_mapping = experiment_spec$scenario_mapping,
  replicate_mapping = experiment_spec$replicate_mapping,
  pooling_policy = experiment_spec$pooling_policy,
  pooled_source_sha256 = source_runs[[1L]]$manifest$source_sha256,
  compact_input_row_count = sum(vapply(source_runs, function(source) {
    length(readLines(source$analysis_input_path, warn = FALSE))
  }, integer(1))),
  retained_analysis_row_count = nrow(rows),
  excluded_matched_set_count = sum(vapply(
    source_runs,
    `[[`,
    integer(1),
    "exclusion_count"
  )),
  analysis_unit = if (identical(args$experiment, "B")) {
    experiment_spec$analysis_unit
  } else {
    "target_updater_controlled_anchor_event"
  },
  outcome_source = experiment_spec$outcome_source,
  primary_reference_updater = reference_updater,
  primary_reference_role = if (identical(args$experiment, "A")) {
    "oracle_embedded_in_outcome_not_stochastic_model_group"
  } else {
    "updater_factor_reference"
  },
  retained_secondary_outcomes = if (identical(args$experiment, "A")) {
    list("exact_update_error", "fitted_update_error")
  } else {
    list()
  },
  retained_calibration_fields = if (identical(args$experiment, "A")) {
    as.list(unlist(
      experiment_spec$retained_calibration_fields,
      use.names = FALSE
    ))
  } else {
    list()
  },
  turn_source = experiment_spec$turn_source %||% NULL,
  terminal_consistency_check = (
    experiment_spec$terminal_consistency_check %||% NULL
  ),
  normalized_rows_file = "analysis-rows.csv",
  normalized_rows_sha256 = sha256_file(analysis_rows_path),
  user_cluster_count = length(unique(rows$user)),
  scenario_cluster_count = length(unique(rows$scenario)),
  factor_levels = lapply(
    rows[vapply(rows, is.factor, logical(1))],
    function(values) as.list(levels(values))
  ),
  turn_values = if ("turn" %in% names(rows)) {
    as.list(sort(unique(rows$turn)))
  } else {
    NULL
  },
  score_layer = if (identical(args$experiment, "A")) {
    "checksum_bound_anchor_directional_exact_oracle_calibration_residual"
  } else {
    "checksum_bound_compact_active_state_calibrated_when_configured"
  },
  source_file_digests = source_file_digests
)
write_json(file.path(output_dir, "input-manifest.json"), input_manifest)

diagnostics <- c(
  list(
    schema_version = 3,
    analysis_id = spec$analysis_id,
    experiment = args$experiment,
    status = model$status,
    reason = model$reason,
    observation_count = nrow(rows),
    user_cluster_count = length(unique(rows$user)),
    scenario_cluster_count = length(unique(rows$scenario)),
    factor_levels = input_manifest$factor_levels
  ),
  model$diagnostics
)
write_json(file.path(output_dir, "diagnostics.json"), diagnostics)

fit_summary <- if (is.null(model$fit)) {
  NULL
} else {
  list(
    model_class = as.list(class(model$fit)),
    observation_count = stats::nobs(model$fit),
    log_likelihood = as.numeric(stats::logLik(model$fit)),
    degrees_of_freedom = attr(stats::logLik(model$fit), "df"),
    aic = stats::AIC(model$fit),
    bic = stats::BIC(model$fit),
    residual_sigma = stats::sigma(model$fit)
  )
}
contrast_families <- lapply(
  unique(model$contrasts$contrast_family),
  function(family) {
    family_rows <- model$contrasts[
      model$contrasts$contrast_family == family,
      ,
      drop = FALSE
    ]
    list(
      family = family,
      confirmatory_role = unique(family_rows$confirmatory_role)[[1L]],
      contrast_count = nrow(family_rows),
      multiplicity = "Holm within family",
      contrast_ids = as.list(family_rows$contrast_id)
    )
  }
)
result <- list(
  schema_version = 3,
  analysis_id = spec$analysis_id,
  experiment = args$experiment,
  status = model$status,
  claim_status = "not_claimed",
  source_run_ids = as.list(vapply(
    source_runs,
    `[[`,
    character(1),
    "run_id"
  )),
  target_updater = args$target_updater,
  reference_updater = reference_updater,
  outcome = experiment_spec$outcome_name,
  formula = experiment_spec$formula,
  family = spec$family,
  link = spec$link,
  estimation = spec$estimation,
  input = list(
    manifest = "input-manifest.json",
    normalized_rows_sha256 = input_manifest$normalized_rows_sha256,
    observation_count = nrow(rows),
    user_cluster_count = input_manifest$user_cluster_count,
    scenario_cluster_count = input_manifest$scenario_cluster_count
  ),
  fit = fit_summary,
  contrast_families = contrast_families,
  artifacts = list(
    diagnostics = "diagnostics.json",
    analysis_rows = "analysis-rows.csv",
    fixed_effects = "fixed-effects.csv",
    omnibus_tests = "omnibus-tests.csv",
    random_effects = "random-effects.csv",
    contrasts = "contrasts.csv",
    session_info = "session-info.txt",
    checksums = "SHA256SUMS"
  ),
  notes = list(
    "This artifact reports a software analysis result and does not itself assert a paper claim.",
    "The primary outcome uses a checksum-bound compact projection of the active state, which is calibrated when configured.",
    if (identical(args$experiment, "B")) {
      paste(
        "Experiment B has one compact row per retained after-turn belief",
        "at turn 1..T; turn indexes are checked and the final compact",
        "marginal Brier score must equal retained_terminal_error."
      )
    } else {
      paste(
        "Experiment A fits only the predeclared target updater's controlled",
        "identical-response rows. Its primary contrasts compare signed",
        "anchor-directional system-minus-exact calibration residuals for",
        "each treatment mechanism against balanced presentation."
      )
    },
    if (identical(args$experiment, "A")) {
      paste(
        "Absolute exact_update_error is retained as a descriptive secondary",
        "magnitude estimand, and fitted_update_error remains a learned-reference",
        "diagnostic; neither is substituted for the signed primary outcome."
      )
    } else {
      paste(
        "Experiment B keeps its fitted-aware updater comparison and exact",
        "same-history-shadow decomposition in the source run unchanged."
      )
    },
    if (pooled_analysis) {
      paste(
        "Pooled inputs share one population seed, preserve raw user and",
        "scenario clusters across reruns, and include a run-level random",
        "intercept. Different seeds must be analyzed separately."
      )
    } else {
      paste(
        "This is a single-run fit; no run-level random intercept is included.",
        "Different seeds remain separate robustness analyses."
      )
    },
    "Raw/calibrated diagnostic scores remain separate; no recursively raw closed-loop trajectory is imputed.",
    "A singular, nonconverged, or rank-deficient maximal model is not silently simplified or replaced by CR1."
  )
)
result_schema <- read_json_object(
  file.path(project_dir, "analysis-result.schema.json"),
  "analysis result schema"
)
validate_result_contract(result, result_schema)
write_json(file.path(output_dir, "analysis-result.json"), result)

session_text <- c(
  paste0("analysis_id: ", spec$analysis_id),
  paste0("analysis_spec_sha256: ", source_file_digests$analysis_spec_sha256),
  paste0("renv_lock_sha256: ", source_file_digests$lock_sha256),
  "",
  capture.output(utils::sessionInfo())
)
writeLines(
  session_text,
  file.path(output_dir, "session-info.txt"),
  useBytes = TRUE
)
write_output_checksums(output_dir)

cat(
  "Mixed-effects analysis status: ",
  model$status,
  "\nOutput: ",
  output_dir,
  "\nNo paper claim was asserted.\n",
  sep = ""
)
