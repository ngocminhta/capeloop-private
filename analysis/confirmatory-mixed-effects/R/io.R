`%||%` <- function(value, fallback) {
  if (is.null(value)) fallback else value
}

abort_analysis <- function(...) {
  stop(paste0(...), call. = FALSE)
}

is_scalar_character <- function(value) {
  is.character(value) && length(value) == 1L && !is.na(value) &&
    nzchar(value)
}

require_scalar_character <- function(value, label) {
  if (!is_scalar_character(value)) {
    abort_analysis(label, " must be one non-empty string")
  }
  value
}

require_scalar_number <- function(value, label) {
  if (
    !is.numeric(value) || length(value) != 1L || is.na(value) ||
    !is.finite(value)
  ) {
    abort_analysis(label, " must be one finite number")
  }
  as.numeric(value)
}

require_scalar_integer <- function(value, label, minimum = NULL) {
  numeric_value <- require_scalar_number(value, label)
  if (numeric_value != floor(numeric_value)) {
    abort_analysis(label, " must be an integer")
  }
  if (!is.null(minimum) && numeric_value < minimum) {
    abort_analysis(label, " must be at least ", minimum)
  }
  as.integer(numeric_value)
}

sha256_file <- function(path) {
  digest::digest(file = path, algo = "sha256", serialize = FALSE)
}

canonical_config_sha256 <- function(path) {
  size <- file.info(path)$size
  if (
    length(size) != 1L || is.na(size) || !is.finite(size) ||
    size < 2
  ) {
    abort_analysis("resolved config is empty or unreadable: ", path)
  }
  bytes <- readBin(path, what = "raw", n = size)
  if (
    length(bytes) < 2L ||
    !identical(bytes[[length(bytes)]], as.raw(10L))
  ) {
    abort_analysis(
      "config.resolved.json must be canonical one-line JSON with one trailing LF"
    )
  }
  payload <- bytes[seq_len(length(bytes) - 1L)]
  if (any(payload %in% as.raw(c(10L, 13L)))) {
    abort_analysis(
      "config.resolved.json is not the canonical one-line JSON payload"
    )
  }
  payload_text <- rawToChar(payload)
  Encoding(payload_text) <- "UTF-8"
  if (!isTRUE(validUTF8(payload_text))) {
    abort_analysis("config.resolved.json is not valid UTF-8")
  }
  python_ascii_fragment <- function(code_point) {
    if (code_point <= 127L) {
      return(intToUtf8(code_point))
    }
    if (code_point <= 65535L) {
      return(sprintf("\\u%04x", code_point))
    }
    offset <- code_point - 65536L
    high <- 55296L + offset %/% 1024L
    low <- 56320L + offset %% 1024L
    paste0(sprintf("\\u%04x", high), sprintf("\\u%04x", low))
  }
  python_canonical_payload <- paste0(
    vapply(
      utf8ToInt(payload_text),
      python_ascii_fragment,
      character(1)
    ),
    collapse = ""
  )
  digest::digest(
    charToRaw(python_canonical_payload),
    algo = "sha256",
    serialize = FALSE
  )
}

read_json_object <- function(path, label = basename(path)) {
  value <- jsonlite::fromJSON(path, simplifyVector = FALSE)
  if (!is.list(value) || is.null(names(value))) {
    abort_analysis(label, " must contain one JSON object")
  }
  value
}

read_jsonl_objects <- function(path, label = basename(path)) {
  lines <- readLines(path, warn = FALSE, encoding = "UTF-8")
  if (!length(lines)) {
    abort_analysis(label, " must contain at least one row")
  }
  if (any(!nzchar(lines))) {
    abort_analysis(label, " contains a blank line")
  }
  lapply(seq_along(lines), function(index) {
    value <- jsonlite::fromJSON(lines[[index]], simplifyVector = FALSE)
    if (!is.list(value) || is.null(names(value))) {
      abort_analysis(label, " row ", index, " is not a JSON object")
    }
    value
  })
}

parse_checksum_manifest <- function(path) {
  lines <- readLines(path, warn = FALSE, encoding = "UTF-8")
  if (!length(lines)) {
    abort_analysis("empty checksum manifest: ", path)
  }
  result <- character()
  for (index in seq_along(lines)) {
    match <- regexec("^([0-9a-f]{64})  (.+)$", lines[[index]])
    captures <- regmatches(lines[[index]], match)[[1L]]
    if (length(captures) != 3L) {
      abort_analysis("invalid SHA256SUMS line ", index, " in ", path)
    }
    relative <- captures[[3L]]
    components <- strsplit(relative, "/", fixed = TRUE)[[1L]]
    if (
      startsWith(relative, "/") || grepl("\\\\", relative) ||
      any(components %in% c("", ".", ".."))
    ) {
      abort_analysis("unsafe checksum path ", relative)
    }
    if (relative %in% names(result)) {
      abort_analysis("duplicate checksum path ", relative)
    }
    result[[relative]] <- captures[[2L]]
  }
  result
}

listed_run_files <- function(run_dir) {
  files <- list.files(
    run_dir,
    recursive = TRUE,
    all.files = TRUE,
    full.names = FALSE,
    include.dirs = FALSE,
    no.. = TRUE
  )
  sort(setdiff(gsub("\\\\", "/", files), "SHA256SUMS"))
}

verify_source_run <- function(run_dir, experiment_spec) {
  run_dir <- normalizePath(run_dir, mustWork = TRUE)
  if (!dir.exists(run_dir)) {
    abort_analysis("source run is not a directory: ", run_dir)
  }
  checksum_path <- file.path(run_dir, "SHA256SUMS")
  if (!file.exists(checksum_path)) {
    abort_analysis("source run has no SHA256SUMS: ", run_dir)
  }
  checksums <- parse_checksum_manifest(checksum_path)
  actual_files <- listed_run_files(run_dir)
  if (!identical(sort(names(checksums)), actual_files)) {
    missing <- setdiff(names(checksums), actual_files)
    extra <- setdiff(actual_files, names(checksums))
    abort_analysis(
      "source run file inventory differs from SHA256SUMS; missing=[",
      paste(missing, collapse = ", "),
      "], extra=[",
      paste(extra, collapse = ", "),
      "]"
    )
  }
  root_prefix <- paste0(run_dir, .Platform$file.sep)
  for (relative in names(checksums)) {
    components <- strsplit(relative, "/", fixed = TRUE)[[1L]]
    unresolved <- run_dir
    for (component in components) {
      unresolved <- file.path(unresolved, component)
      link <- Sys.readlink(unresolved)
      if (length(link) != 1L || is.na(link)) {
        abort_analysis("could not inspect source path: ", relative)
      }
      if (nzchar(link)) {
        abort_analysis("source run contains a symbolic link: ", relative)
      }
    }
    path <- normalizePath(unresolved, mustWork = TRUE)
    if (!startsWith(path, root_prefix)) {
      abort_analysis("checksum path escapes source run: ", relative)
    }
    observed <- sha256_file(path)
    if (!identical(observed, unname(checksums[[relative]]))) {
      abort_analysis("checksum mismatch for ", relative)
    }
  }

  required_common <- c(
    "manifest.json",
    "config.resolved.json",
    "metrics/summary.json",
    experiment_spec$input_file
  )
  if (!is.null(experiment_spec$exclusion_file)) {
    required_common <- c(required_common, experiment_spec$exclusion_file)
  }
  absent <- setdiff(required_common, names(checksums))
  if (length(absent)) {
    abort_analysis(
      "verified source run is missing required files: ",
      paste(absent, collapse = ", ")
    )
  }
  manifest <- read_json_object(file.path(run_dir, "manifest.json"))
  config_path <- file.path(run_dir, "config.resolved.json")
  config <- read_json_object(config_path)
  summary <- read_json_object(file.path(run_dir, "metrics/summary.json"))
  run_id <- require_scalar_character(manifest$run_id, "manifest.run_id")
  if (!identical(manifest$status, "complete")) {
    abort_analysis("source run is not complete: ", run_id)
  }
  if (!identical(basename(run_dir), run_id)) {
    abort_analysis("source run directory name does not equal manifest.run_id")
  }
  for (field in c("config_sha256", "source_sha256")) {
    value <- manifest[[field]]
    if (
      !is_scalar_character(value) ||
      !grepl("^[0-9a-f]{64}$", value)
    ) {
      abort_analysis("manifest.", field, " is not a lowercase SHA-256")
    }
  }
  observed_config_sha256 <- canonical_config_sha256(config_path)
  if (!identical(observed_config_sha256, manifest$config_sha256)) {
    abort_analysis(
      "manifest.config_sha256 does not match canonical config.resolved.json"
    )
  }
  if (!identical(config$experiment$kind, experiment_spec$run_kind)) {
    abort_analysis(
      "run ",
      run_id,
      " has kind ",
      config$experiment$kind %||% "<missing>",
      "; expected ",
      experiment_spec$run_kind
    )
  }
  expected_label <- if (identical(experiment_spec$run_kind, "provenance_audit")) {
    "A"
  } else {
    "B"
  }
  if (!identical(summary$experiment, expected_label)) {
    abort_analysis("summary experiment label differs for ", run_id)
  }
  if (!identical(summary$scientific_claim_status, "not_claimed")) {
    abort_analysis("source summary claim status is not not_claimed for ", run_id)
  }
  if (!isTRUE(config$artifacts$retain_events)) {
    abort_analysis("source run did not declare artifacts.retain_events=true")
  }

  input_path <- file.path(run_dir, experiment_spec$input_file)
  exclusion_path <- NULL
  exclusion_digest <- NULL
  exclusion_count <- 0L
  if (!is.null(experiment_spec$exclusion_file)) {
    relative <- experiment_spec$exclusion_file
    exclusion_path <- file.path(run_dir, relative)
    exclusion_digest <- unname(checksums[[relative]])
    exclusion_lines <- readLines(
      exclusion_path,
      warn = FALSE,
      encoding = "UTF-8"
    )
    if (length(exclusion_lines)) {
      if (any(!nzchar(exclusion_lines))) {
        abort_analysis("Experiment A exclusions contains a blank line")
      }
      exclusion_records <- lapply(seq_along(exclusion_lines), function(index) {
        value <- jsonlite::fromJSON(
          exclusion_lines[[index]],
          simplifyVector = FALSE
        )
        if (!is.list(value) || is.null(names(value))) {
          abort_analysis(
            "Experiment A exclusion row ",
            index,
            " is not a JSON object"
          )
        }
        expected_fields <- c(
          "schema_version",
          "user_id",
          "domain_id",
          "target_attribute",
          "anchor_direction",
          "minimum_probability",
          "choice_probabilities"
        )
        if (!setequal(names(value), expected_fields)) {
          abort_analysis(
            "Experiment A exclusion row ",
            index,
            " has unexpected fields"
          )
        }
        if (require_scalar_integer(
          value$schema_version,
          "exclusion.schema_version"
        ) != 1L) {
          abort_analysis("unsupported Experiment A exclusion schema version")
        }
        require_scalar_character(value$user_id, "exclusion.user_id")
        require_scalar_character(value$domain_id, "exclusion.domain_id")
        target_attribute <- require_scalar_integer(
          value$target_attribute,
          "exclusion.target_attribute"
        )
        if (!(target_attribute %in% 0:2)) {
          abort_analysis("exclusion.target_attribute must lie in 0:2")
        }
        anchor_direction <- require_scalar_integer(
          value$anchor_direction,
          "exclusion.anchor_direction"
        )
        if (!(anchor_direction %in% c(-1L, 1L))) {
          abort_analysis("exclusion.anchor_direction must be -1 or 1")
        }
        minimum_probability <- require_scalar_number(
          value$minimum_probability,
          "exclusion.minimum_probability"
        )
        if (minimum_probability <= 0 || minimum_probability >= 1) {
          abort_analysis("exclusion.minimum_probability must lie in (0, 1)")
        }
        probabilities <- value$choice_probabilities
        if (
          !is.list(probabilities) ||
          is.null(names(probabilities)) ||
          !setequal(
            names(probabilities),
            unlist(experiment_spec$required_mechanisms, use.names = FALSE)
          )
        ) {
          abort_analysis(
            "exclusion.choice_probabilities must cover every mechanism"
          )
        }
        numeric_probabilities <- vapply(
          names(probabilities),
          function(mechanism) require_scalar_number(
            probabilities[[mechanism]],
            paste0("exclusion.choice_probabilities.", mechanism)
          ),
          numeric(1)
        )
        if (any(numeric_probabilities < 0 | numeric_probabilities > 1)) {
          abort_analysis(
            "exclusion.choice_probabilities values must lie in [0, 1]"
          )
        }
        if (all(numeric_probabilities > minimum_probability)) {
          abort_analysis(
            "exclusion row does not violate its minimum-probability rule"
          )
        }
        value
      })
      exclusion_keys <- vapply(exclusion_records, function(value) {
        paste(
          value$user_id,
          value$domain_id,
          value$target_attribute,
          value$anchor_direction,
          sep = "\r"
        )
      }, character(1))
      if (anyDuplicated(exclusion_keys)) {
        abort_analysis("Experiment A exclusions contain duplicate matched sets")
      }
    }
    exclusion_count <- length(exclusion_lines)
    expected_exclusions <- require_scalar_integer(
      summary$excluded_matched_sets,
      "summary.excluded_matched_sets",
      minimum = 0L
    )
    if (exclusion_count != expected_exclusions) {
      abort_analysis(
        "Experiment A exclusion count differs from metrics/summary.json"
      )
    }
  }
  list(
    path = run_dir,
    run_id = run_id,
    manifest = manifest,
    config = config,
    summary = summary,
    checksums = checksums,
    checksum_manifest_sha256 = sha256_file(checksum_path),
    input_path = input_path,
    input_sha256 = unname(checksums[[experiment_spec$input_file]]),
    exclusion_path = exclusion_path,
    exclusion_sha256 = exclusion_digest,
    exclusion_count = exclusion_count
  )
}

assert_same_design <- function(source_runs, experiment) {
  source_digests <- vapply(source_runs, function(source) {
    source$manifest$source_sha256
  }, character(1))
  if (length(unique(source_digests)) != 1L) {
    abort_analysis(
      "pooled source runs must have identical manifest.source_sha256 values"
    )
  }
  design_signature <- function(source) {
    experiment_config <- source$config$experiment
    llm <- source$config$llm
    llm_semantics <- llm[c(
      "calibration",
      "calibration_users",
      "model_role",
      "model",
      "reasoning_effort",
      "base_url"
    )]
    list(
      schema_version = source$config$schema_version,
      experiment = experiment_config,
      response_model = source$config$response_model,
      inference = source$config$inference,
      thresholds = source$config$thresholds,
      llm = llm_semantics
    )
  }
  reference <- design_signature(source_runs[[1L]])
  for (source in source_runs[-1L]) {
    candidate <- design_signature(source)
    if (!identical(reference, candidate)) {
      abort_analysis(
        "source runs differ on scientific design or model semantics; ",
        "combine only independent repeats of the same declaration, including ",
        "the same Experiment B horizon"
      )
    }
  }
  invisible(TRUE)
}

normalize_experiment_a <- function(source) {
  records <- read_jsonl_objects(
    source$input_path,
    paste0(source$run_id, " Experiment A events")
  )
  invisible(lapply(seq_along(records), function(index) {
    row <- records[[index]]
    if (require_scalar_integer(
      row$schema_version,
      paste0("Experiment A row ", index, " schema_version")
    ) != 1L) {
      abort_analysis("unsupported Experiment A event schema version")
    }
    mode <- require_scalar_character(
      row$response_mode,
      paste0("Experiment A row ", index, " response_mode")
    )
    if (!(mode %in% c("controlled_anchor", "naturally_sampled"))) {
      abort_analysis("unknown Experiment A response_mode: ", mode)
    }
    NULL
  }))
  selected_indexes <- which(vapply(records, function(row) {
    identical(row$response_mode, "naturally_sampled")
  }, logical(1)))
  selected <- records[selected_indexes]
  if (!length(selected)) {
    abort_analysis(source$run_id, " has no naturally sampled Experiment A rows")
  }
  if (
    require_scalar_integer(
      source$summary$row_count,
      "summary.row_count",
      minimum = 1L
    ) != length(records) ||
    require_scalar_integer(
      source$summary$natural_row_count,
      "summary.natural_row_count",
      minimum = 1L
    ) != length(selected)
  ) {
    abort_analysis(
      source$run_id,
      " Experiment A event counts differ from metrics/summary.json"
    )
  }
  data.frame(
    source_run_id = rep(source$run_id, length(selected)),
    source_record_index = selected_indexes,
    trial_id = vapply(
      selected,
      function(row) require_scalar_character(row$trial_id, "trial_id"),
      character(1)
    ),
    user = vapply(selected, function(row) {
      paste(
        source$run_id,
        require_scalar_character(row$user_id, "user_id"),
        sep = "::"
      )
    }, character(1)),
    domain = vapply(
      selected,
      function(row) require_scalar_character(row$domain_id, "domain_id"),
      character(1)
    ),
    scenario = vapply(selected, function(row) {
      paste(
        source$run_id,
        require_scalar_character(
          row$context$scenario_id,
          "context.scenario_id"
        ),
        sep = "::"
      )
    }, character(1)),
    updater = vapply(
      selected,
      function(row) require_scalar_character(row$updater_id, "updater_id"),
      character(1)
    ),
    mechanism = vapply(
      selected,
      function(row) require_scalar_character(row$mechanism, "mechanism"),
      character(1)
    ),
    prior_strength = vapply(
      selected,
      function(row) require_scalar_number(
        row$prior_strength,
        "prior_strength"
      ),
      numeric(1)
    ),
    update_error = vapply(
      selected,
      function(row) require_scalar_number(row$metrics$acue, "metrics.acue"),
      numeric(1)
    ),
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
}

validate_theta_values <- function(theta, label) {
  if (!is.list(theta) || length(theta) != 3L) {
    abort_analysis(label, " must contain three latent preference values")
  }
  values <- vapply(seq_along(theta), function(index) {
    require_scalar_integer(theta[[index]], paste0(label, "[", index, "]"))
  }, integer(1))
  if (any(!(values %in% c(-2L, -1L, 1L, 2L)))) {
    abort_analysis(label, " values must lie in {-2, -1, 1, 2}")
  }
  values
}

marginal_brier_from_retained_belief <- function(belief, theta, label) {
  if (!is.list(belief) || is.null(names(belief))) {
    abort_analysis(label, " must be a retained belief object")
  }
  marginals <- belief$marginals
  if (!is.list(marginals) || length(marginals) != 3L) {
    abort_analysis(label, ".marginals must contain three rows")
  }
  theta_values <- c(-2L, -1L, 1L, 2L)
  attribute_scores <- vapply(seq_along(marginals), function(attribute) {
    probabilities <- marginals[[attribute]]
    if (!is.list(probabilities) || length(probabilities) != 4L) {
      abort_analysis(
        label,
        ".marginals[",
        attribute,
        "] must contain four probabilities"
      )
    }
    numeric_probabilities <- vapply(
      seq_along(probabilities),
      function(index) require_scalar_number(
        probabilities[[index]],
        paste0(
          label,
          ".marginals[",
          attribute,
          "][",
          index,
          "]"
        )
      ),
      numeric(1)
    )
    if (
      any(numeric_probabilities < 0 | numeric_probabilities > 1) ||
      abs(sum(numeric_probabilities) - 1) > 1e-9
    ) {
      abort_analysis(
        label,
        ".marginals[",
        attribute,
        "] is not a probability vector"
      )
    }
    expected <- as.numeric(theta_values == theta[[attribute]])
    sum((numeric_probabilities - expected)^2)
  }, numeric(1))
  sum(attribute_scores) / 3
}

normalize_turn_errors <- function(turns, theta, trajectory_id) {
  if (!is.list(turns) || !length(turns)) {
    abort_analysis(trajectory_id, " has no turns")
  }
  observed <- vapply(seq_along(turns), function(index) {
    value <- turns[[index]]$turn
    numeric_value <- require_scalar_number(
      value,
      paste0(trajectory_id, ".turns[", index, "].turn")
    )
    if (numeric_value != floor(numeric_value)) {
      abort_analysis(trajectory_id, " has a non-integer turn index")
    }
    as.integer(numeric_value)
  }, integer(1))
  expected <- seq.int(0L, length(turns) - 1L)
  if (!identical(observed, expected)) {
    abort_analysis(trajectory_id, " turn indexes are not contiguous from zero")
  }
  errors <- vapply(seq_along(turns), function(index) {
    turn <- turns[[index]]
    if (!is.null(turn$theta_snapshot)) {
      snapshot <- validate_theta_values(
        turn$theta_snapshot,
        paste0(trajectory_id, ".turns[", index, "].theta_snapshot")
      )
      if (!identical(snapshot, theta)) {
        abort_analysis(trajectory_id, " changes latent theta within a trajectory")
      }
    }
    marginal_brier_from_retained_belief(
      turn$belief_after,
      theta,
      paste0(trajectory_id, ".turns[", index, "].belief_after")
    )
  }, numeric(1))
  list(
    source_turn_index = observed,
    turn = observed + 1L,
    terminal_error = errors
  )
}

normalize_experiment_b <- function(source) {
  records <- read_jsonl_objects(
    source$input_path,
    paste0(source$run_id, " Experiment B trajectories")
  )
  if (
    require_scalar_integer(
      source$summary$trajectories,
      "summary.trajectories",
      minimum = 1L
    ) != length(records)
  ) {
    abort_analysis(
      source$run_id,
      " Experiment B trajectory count differs from metrics/summary.json"
    )
  }
  configured_turns <- require_scalar_integer(
    source$config$experiment$turns,
    "config.experiment.turns",
    minimum = 1L
  )
  rows <- lapply(seq_along(records), function(index) {
    row <- records[[index]]
    if (require_scalar_integer(
      row$schema_version,
      paste0("Experiment B row ", index, " schema_version")
    ) != 1L) {
      abort_analysis("unsupported Experiment B trajectory schema version")
    }
    trajectory_id <- require_scalar_character(
      row$trajectory_id,
      "trajectory_id"
    )
    if (!isTRUE(row$same_history_shadow)) {
      abort_analysis(trajectory_id, " does not retain a same-history shadow")
    }
    theta <- validate_theta_values(row$theta, paste0(trajectory_id, ".theta"))
    turn_values <- normalize_turn_errors(row$turns, theta, trajectory_id)
    if (length(turn_values$turn) != configured_turns) {
      abort_analysis(
        trajectory_id,
        " retained turn count differs from config.experiment.turns"
      )
    }
    retained_terminal_error <- require_scalar_number(
      row$terminal_error,
      paste0(trajectory_id, ".terminal_error")
    )
    reconstructed_terminal_error <- tail(
      turn_values$terminal_error,
      1L
    )
    terminal_tolerance <- 1e-12 + 1e-9 * abs(retained_terminal_error)
    if (
      abs(reconstructed_terminal_error - retained_terminal_error) >
        terminal_tolerance
    ) {
      abort_analysis(
        trajectory_id,
        " final turn Brier error differs from retained terminal_error"
      )
    }
    turn_count <- length(turn_values$turn)
    data.frame(
      source_run_id = rep(source$run_id, turn_count),
      source_record_index = rep(index, turn_count),
      source_turn_index = turn_values$source_turn_index,
      trajectory_id = rep(trajectory_id, turn_count),
      user = rep(
        paste(
          source$run_id,
          require_scalar_character(row$user_id, "user_id"),
          sep = "::"
        ),
        turn_count
      ),
      domain = rep(
        require_scalar_character(row$domain_id, "domain_id"),
        turn_count
      ),
      scenario = rep(
        paste(
          source$run_id,
          require_scalar_character(row$crn_key, "crn_key"),
          sep = "::"
        ),
        turn_count
      ),
      updater = rep(
        require_scalar_character(row$updater_id, "updater_id"),
        turn_count
      ),
      policy = rep(
        require_scalar_character(row$policy_id, "policy_id"),
        turn_count
      ),
      initial_profile = rep(
        require_scalar_character(
          row$initial_profile_condition,
          "initial_profile_condition"
        ),
        turn_count
      ),
      turn = turn_values$turn,
      terminal_error = turn_values$terminal_error,
      stringsAsFactors = FALSE,
      check.names = FALSE
    )
  })
  do.call(rbind, rows)
}

all_cells_positive <- function(data, group, first, second) {
  first_levels <- if (is.factor(data[[first]])) {
    levels(data[[first]])
  } else {
    sort(unique(data[[first]]))
  }
  second_levels <- if (is.factor(data[[second]])) {
    levels(data[[second]])
  } else {
    sort(unique(data[[second]]))
  }
  split_rows <- split(data, data[[group]], drop = TRUE)
  all(vapply(split_rows, function(rows) {
    observed <- table(
      factor(rows[[first]], levels = first_levels),
      factor(rows[[second]], levels = second_levels)
    )
    all(observed > 0L)
  }, logical(1)))
}

assert_exact_crossing <- function(
  data,
  grouping_columns,
  first,
  second,
  label
) {
  first_levels <- sort(unique(as.character(data[[first]])))
  second_levels <- sort(unique(as.character(data[[second]])))
  group_key <- do.call(
    interaction,
    c(
      lapply(data[grouping_columns], as.character),
      list(drop = TRUE, lex.order = TRUE, sep = "\r")
    )
  )
  split_rows <- split(data, group_key, drop = TRUE)
  valid <- all(vapply(split_rows, function(rows) {
    observed <- table(
      factor(as.character(rows[[first]]), levels = first_levels),
      factor(as.character(rows[[second]]), levels = second_levels)
    )
    all(observed == 1L)
  }, logical(1)))
  if (!valid) {
    abort_analysis(
      label,
      " must contain exactly one row per ",
      first,
      "-by-",
      second,
      " cell"
    )
  }
  invisible(TRUE)
}

set_reference <- function(values, reference, label) {
  factor_values <- factor(values, levels = sort(unique(values)))
  if (!(reference %in% levels(factor_values))) {
    abort_analysis(label, " reference level is absent: ", reference)
  }
  stats::relevel(factor_values, ref = reference)
}

validate_analysis_rows <- function(
  rows,
  experiment,
  experiment_spec,
  target_updater,
  factor_coding,
  minimum_users,
  minimum_scenarios
) {
  if (!nrow(rows)) {
    abort_analysis("analysis has no retained rows")
  }
  source_key <- c("source_run_id", "source_record_index")
  if (identical(experiment, "B")) {
    source_key <- c(source_key, "source_turn_index")
  }
  if (anyDuplicated(rows[source_key])) {
    abort_analysis("normalized analysis contains duplicate source records")
  }
  business_key <- if (identical(experiment, "A")) {
    c("source_run_id", "trial_id", "updater")
  } else {
    c("source_run_id", "trajectory_id", "turn")
  }
  if (anyDuplicated(rows[business_key])) {
    abort_analysis("normalized analysis contains duplicate scientific row IDs")
  }
  outcome <- if (identical(experiment, "A")) {
    rows$update_error
  } else {
    rows$terminal_error
  }
  if (any(!is.finite(outcome)) || any(outcome < 0)) {
    abort_analysis("analysis outcome must be finite and non-negative")
  }
  if (!(target_updater %in% rows$updater)) {
    abort_analysis("target updater is absent: ", target_updater)
  }
  aware <- factor_coding$updater_reference
  if (!(aware %in% rows$updater)) {
    abort_analysis("reference updater is absent: ", aware)
  }
  if (length(unique(rows$user)) < minimum_users) {
    abort_analysis(
      "analysis requires at least ",
      minimum_users,
      " complete user clusters"
    )
  }
  if (length(unique(rows$scenario)) < minimum_scenarios) {
    abort_analysis(
      "analysis requires at least ",
      minimum_scenarios,
      " scenario clusters"
    )
  }

  rows$updater <- set_reference(rows$updater, aware, "updater")
  domain_reference <- factor_coding$domain_reference_preference
  if (!(domain_reference %in% rows$domain)) {
    domain_reference <- sort(unique(rows$domain))[[1L]]
  }
  rows$domain <- set_reference(rows$domain, domain_reference, "domain")
  rows$user <- factor(rows$user)
  rows$scenario <- factor(rows$scenario)

  if (identical(experiment, "A")) {
    required <- unlist(experiment_spec$required_mechanisms, use.names = FALSE)
    absent <- setdiff(required, unique(rows$mechanism))
    if (length(absent)) {
      abort_analysis(
        "Experiment A is missing required mechanisms: ",
        paste(absent, collapse = ", ")
      )
    }
    if (any(rows$prior_strength < 0 | rows$prior_strength >= 1)) {
      abort_analysis("Experiment A prior_strength must lie in [0, 1)")
    }
    if (!all_cells_positive(rows, "user", "updater", "mechanism")) {
      abort_analysis(
        "every Experiment A user must contain every updater-by-mechanism cell"
      )
    }
    assert_exact_crossing(
      rows,
      c("source_run_id", "user", "scenario", "prior_strength"),
      "updater",
      "mechanism",
      "each Experiment A matched stratum"
    )
    rows$mechanism <- set_reference(
      rows$mechanism,
      factor_coding$experiment_a_mechanism_reference,
      "mechanism"
    )
  } else {
    required_policies <- unlist(
      experiment_spec$required_policies,
      use.names = FALSE
    )
    absent_policies <- setdiff(required_policies, unique(rows$policy))
    if (length(absent_policies)) {
      abort_analysis(
        "Experiment B is missing required policies: ",
        paste(absent_policies, collapse = ", ")
      )
    }
    required_profiles <- unlist(
      experiment_spec$required_initial_profiles,
      use.names = FALSE
    )
    absent_profiles <- setdiff(
      required_profiles,
      unique(rows$initial_profile)
    )
    if (length(absent_profiles)) {
      abort_analysis(
        "Experiment B is missing required initial profiles: ",
        paste(absent_profiles, collapse = ", ")
      )
    }
    if (!all_cells_positive(rows, "user", "updater", "policy")) {
      abort_analysis(
        "every Experiment B user must contain every updater-by-policy cell"
      )
    }
    assert_exact_crossing(
      rows,
      c("source_run_id", "scenario", "turn"),
      "updater",
      "policy",
      "each Experiment B common-random-number scenario turn"
    )
    rows$policy <- set_reference(
      rows$policy,
      factor_coding$experiment_b_policy_reference,
      "policy"
    )
    rows$initial_profile <- set_reference(
      rows$initial_profile,
      factor_coding$experiment_b_initial_profile_reference,
      "initial_profile"
    )
  }
  rows
}

write_json <- function(path, value) {
  jsonlite::write_json(
    value,
    path,
    auto_unbox = TRUE,
    pretty = TRUE,
    digits = 17,
    na = "null",
    null = "null"
  )
  cat("\n", file = path, append = TRUE)
}

validate_result_contract <- function(result, schema) {
  required <- unlist(schema$required, use.names = FALSE)
  properties <- names(schema$properties)
  missing <- setdiff(required, names(result))
  extra <- setdiff(names(result), properties)
  if (length(missing) || length(extra)) {
    abort_analysis(
      "analysis result differs from its schema; missing=[",
      paste(missing, collapse = ", "),
      "], extra=[",
      paste(extra, collapse = ", "),
      "]"
    )
  }
  if (!(result$status %in% unlist(
    schema$properties$status$enum,
    use.names = FALSE
  ))) {
    abort_analysis("analysis result has an invalid status")
  }
  expected_version <- schema$properties$schema_version$const
  if (
    !is.numeric(result$schema_version) ||
    !is.numeric(expected_version) ||
    result$schema_version != expected_version
  ) {
    abort_analysis(
      "analysis result violates schema const for schema_version"
    )
  }
  for (field in c("analysis_id", "claim_status")) {
    expected <- schema$properties[[field]]$const
    if (!identical(result[[field]], expected)) {
      abort_analysis("analysis result violates schema const for ", field)
    }
  }
  if (!(result$experiment %in% c("A", "B"))) {
    abort_analysis("analysis result has an invalid experiment")
  }
  invisible(TRUE)
}

write_csv <- function(path, value) {
  utils::write.table(
    value,
    file = path,
    sep = ",",
    row.names = FALSE,
    col.names = TRUE,
    quote = TRUE,
    na = "",
    qmethod = "double",
    fileEncoding = "UTF-8",
    eol = "\n"
  )
}

write_output_checksums <- function(output_dir) {
  relative_files <- sort(
    setdiff(
      list.files(
        output_dir,
        recursive = TRUE,
        all.files = TRUE,
        full.names = FALSE,
        include.dirs = FALSE,
        no.. = TRUE
      ),
      "SHA256SUMS"
    )
  )
  relative_files <- gsub("\\\\", "/", relative_files)
  lines <- vapply(relative_files, function(relative) {
    paste0(sha256_file(file.path(output_dir, relative)), "  ", relative)
  }, character(1))
  writeLines(lines, file.path(output_dir, "SHA256SUMS"), useBytes = TRUE)
}
