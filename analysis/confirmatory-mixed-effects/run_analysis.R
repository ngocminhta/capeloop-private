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
  observed <- as.character(utils::packageVersion(package))
  if (!identical(observed, expected)) {
    stop(
      "package version mismatch for ",
      package,
      ": expected ",
      expected,
      ", observed ",
      observed,
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
      "    [--target-updater llm_full_context]",
      "",
      "The output directory must not exist and must be outside every source run.",
      sep = "\n"
    )
  )
}

parse_arguments <- function(arguments) {
  result <- list(
    experiment = NULL,
    runs = character(),
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
      abort_analysis("analysis output cannot be inside a source run")
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
  !identical(spec$schema_version, 1L) &&
  !identical(spec$schema_version, 1)
) {
  abort_analysis("unsupported analysis specification version")
}
experiment_spec <- spec$experiments[[args$experiment]]
if (is.null(experiment_spec)) {
  abort_analysis("analysis specification has no requested experiment")
}

source_runs <- lapply(args$runs, function(path) {
  verify_source_run(path, experiment_spec)
})
if (anyDuplicated(vapply(source_runs, `[[`, character(1), "run_id"))) {
  abort_analysis("source run IDs must be unique")
}
assert_same_design(source_runs, args$experiment)
output_dir <- safe_output_path(
  args$output,
  vapply(source_runs, `[[`, character(1), "path")
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
  spec$factor_coding$updater_reference
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
    input_file = experiment_spec$input_file,
    input_sha256 = source$input_sha256,
    exclusion_file = experiment_spec$exclusion_file %||% NULL,
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
  schema_version = 1,
  analysis_id = spec$analysis_id,
  experiment = args$experiment,
  source_runs = source_records,
  source_run_count = length(source_runs),
  pooled_source_sha256 = source_runs[[1L]]$manifest$source_sha256,
  original_input_row_count = sum(vapply(source_runs, function(source) {
    length(readLines(source$input_path, warn = FALSE))
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
    "naturally_sampled_event"
  },
  outcome_source = experiment_spec$outcome_source,
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
  score_layer = "active_event_state_calibrated_when_configured",
  source_file_digests = source_file_digests
)
write_json(file.path(output_dir, "input-manifest.json"), input_manifest)

diagnostics <- c(
  list(
    schema_version = 1,
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
  schema_version = 1,
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
  reference_updater = spec$factor_coding$updater_reference,
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
    "The primary outcome uses the active retained event state, which is calibrated when the source run configured calibration.",
    if (identical(args$experiment, "B")) {
      paste(
        "Experiment B has one row per retained after-turn belief at turn",
        "1..T; the final reconstructed marginal Brier score was checked",
        "against the retained terminal_error."
      )
    } else {
      paste(
        "Experiment A uses naturally sampled ACUE rows; its contrast",
        "tests error relative to the fitted-aware reference, not the",
        "direction of the target updater's belief change."
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
