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

require_scalar_logical <- function(value, label) {
  if (!is.logical(value) || length(value) != 1L || is.na(value)) {
    abort_analysis(label, " must be one boolean")
  }
  value
}

assert_exact_fields <- function(value, expected, label) {
  if (!is.list(value) || is.null(names(value))) {
    abort_analysis(label, " must be one JSON object")
  }
  missing <- setdiff(expected, names(value))
  extra <- setdiff(names(value), expected)
  if (length(missing) || length(extra) || anyDuplicated(names(value))) {
    abort_analysis(
      label,
      " has an invalid field set; missing=[",
      paste(missing, collapse = ", "),
      "], extra=[",
      paste(extra, collapse = ", "),
      "]"
    )
  }
  invisible(TRUE)
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

verify_source_run <- function(
  run_dir,
  experiment_spec,
  use_legacy_input = FALSE
) {
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

  resolved_legacy_input <- if (is.null(use_legacy_input)) {
    !(experiment_spec$input_file %in% names(checksums))
  } else {
    isTRUE(use_legacy_input)
  }
  input_relative <- if (resolved_legacy_input) {
    require_scalar_character(
      experiment_spec$legacy_input_file,
      "analysis specification legacy_input_file"
    )
  } else {
    experiment_spec$input_file
  }
  exclusion_relative <- if (resolved_legacy_input) {
    experiment_spec$legacy_exclusion_file %||% NULL
  } else {
    experiment_spec$exclusion_file %||% NULL
  }
  required_common <- c(
    "manifest.json",
    "config.resolved.json",
    "metrics/summary.json",
    input_relative
  )
  if (!is.null(exclusion_relative)) {
    required_common <- c(required_common, exclusion_relative)
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
  if (
    !resolved_legacy_input &&
    !is.null(exclusion_relative) &&
    !identical(
      summary$analysis_exclusion_artifact,
      exclusion_relative
    )
  ) {
    abort_analysis(
      "source summary compact exclusion declaration differs for ",
      run_id
    )
  }
  input_path <- file.path(run_dir, input_relative)
  exclusion_path <- NULL
  exclusion_digest <- NULL
  exclusion_count <- 0L
  if (!is.null(exclusion_relative)) {
    relative <- exclusion_relative
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
        assert_exact_fields(
          value,
          expected_fields,
          paste0("Experiment A exclusion row ", index)
        )
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
    input_relative = input_relative,
    input_path = input_path,
    input_sha256 = unname(checksums[[input_relative]]),
    input_is_legacy = resolved_legacy_input,
    analysis_input_path = input_path,
    analysis_input_sha256 = unname(checksums[[input_relative]]),
    compact_bundle = NULL,
    exclusion_relative = exclusion_relative,
    exclusion_path = exclusion_path,
    exclusion_sha256 = exclusion_digest,
    exclusion_count = exclusion_count
  )
}

verify_compact_bundle <- function(
  bundle_dir,
  source,
  experiment,
  experiment_spec
) {
  supplied <- path.expand(bundle_dir)
  supplied_link <- Sys.readlink(supplied)
  if (
    length(supplied_link) != 1L || is.na(supplied_link) ||
    nzchar(supplied_link)
  ) {
    abort_analysis("compact bundle must be a non-symlink directory")
  }
  bundle_dir <- normalizePath(supplied, mustWork = TRUE)
  if (!dir.exists(bundle_dir)) {
    abort_analysis("compact bundle is not a directory: ", bundle_dir)
  }
  source_prefix <- paste0(source$path, .Platform$file.sep)
  if (
    identical(bundle_dir, source$path) ||
    startsWith(bundle_dir, source_prefix)
  ) {
    abort_analysis("compact bundle cannot be inside its immutable source run")
  }
  bundle_entries <- list.files(
    bundle_dir,
    recursive = TRUE,
    all.files = TRUE,
    full.names = TRUE,
    include.dirs = TRUE,
    no.. = TRUE
  )
  if (length(bundle_entries)) {
    entry_info <- file.info(bundle_entries)
    if (any(is.na(entry_info$isdir)) || any(entry_info$isdir)) {
      abort_analysis("compact bundle cannot contain subdirectories")
    }
  }

  expected_files <- sort(c(
    "SHA256SUMS",
    "analysis-rows.jsonl",
    "manifest.json"
  ))
  actual_files <- sort(c(
    listed_run_files(bundle_dir),
    if (file.exists(file.path(bundle_dir, "SHA256SUMS"))) {
      "SHA256SUMS"
    } else {
      character()
    }
  ))
  if (!identical(actual_files, expected_files)) {
    abort_analysis(
      "compact bundle inventory must be exactly ",
      paste(expected_files, collapse = ", ")
    )
  }
  for (relative in expected_files) {
    path <- file.path(bundle_dir, relative)
    link <- Sys.readlink(path)
    if (
      length(link) != 1L || is.na(link) || nzchar(link) ||
      !file.exists(path) || dir.exists(path)
    ) {
      abort_analysis(
        "compact bundle entry must be one regular non-symlink file: ",
        relative
      )
    }
  }

  checksum_path <- file.path(bundle_dir, "SHA256SUMS")
  checksums <- parse_checksum_manifest(checksum_path)
  expected_checksum_names <- sort(c(
    "analysis-rows.jsonl",
    "manifest.json"
  ))
  if (!identical(sort(names(checksums)), expected_checksum_names)) {
    abort_analysis(
      "compact bundle SHA256SUMS must list exactly analysis-rows.jsonl and manifest.json"
    )
  }
  for (relative in names(checksums)) {
    observed <- sha256_file(file.path(bundle_dir, relative))
    if (!identical(observed, unname(checksums[[relative]]))) {
      abort_analysis("compact bundle checksum mismatch for ", relative)
    }
  }

  manifest_path <- file.path(bundle_dir, "manifest.json")
  rows_path <- file.path(bundle_dir, "analysis-rows.jsonl")
  manifest <- read_json_object(manifest_path, "compact manifest.json")
  base_fields <- c(
    "schema_version",
    "artifact_kind",
    "status",
    "claim_status",
    "experiment",
    "analysis_unit",
    "row_schema_version",
    "row_count",
    "source_record_count",
    "configured_turns",
    "analysis_rows_file",
    "analysis_rows_sha256",
    "source_run_id",
    "source_manifest_sha256",
    "source_checksums_sha256",
    "source_config_file_sha256",
    "source_summary_file_sha256",
    "source_config_sha256",
    "source_tree_sha256",
    "source_input_file",
    "source_input_sha256",
    "source_input_is_runner_compact",
    "exporter_version",
    "exporter_source_sha256",
    "outcome_derivation"
  )
  expected_manifest_fields <- if (identical(experiment, "A")) {
    c(
      base_fields,
      "source_exclusion_file",
      "source_exclusion_sha256"
    )
  } else {
    base_fields
  }
  assert_exact_fields(
    manifest,
    expected_manifest_fields,
    "compact manifest.json"
  )
  if (require_scalar_integer(
    manifest$schema_version,
    "compact manifest.schema_version"
  ) != 1L) {
    abort_analysis("unsupported compact manifest schema_version")
  }
  if (!identical(
    manifest$artifact_kind,
    "cape-loop-compact-analysis-bundle"
  )) {
    abort_analysis("compact manifest has an unexpected artifact_kind")
  }
  if (!identical(manifest$status, "complete")) {
    abort_analysis("compact manifest status is not complete")
  }
  if (!identical(manifest$claim_status, "not_claimed")) {
    abort_analysis("compact manifest claim_status is not not_claimed")
  }
  if (!identical(manifest$experiment, experiment)) {
    abort_analysis("compact manifest experiment differs from requested analysis")
  }
  if (!identical(
    manifest$analysis_unit,
    experiment_spec$bundle_analysis_unit
  )) {
    abort_analysis("compact manifest analysis_unit differs from the protocol")
  }
  if (require_scalar_integer(
    manifest$row_schema_version,
    "compact manifest.row_schema_version"
  ) != 1L) {
    abort_analysis("unsupported compact analysis row schema_version")
  }
  if (!identical(manifest$analysis_rows_file, "analysis-rows.jsonl")) {
    abort_analysis("compact manifest analysis_rows_file is not canonical")
  }
  if (!identical(
    manifest$outcome_derivation,
    experiment_spec$bundle_outcome_derivation
  )) {
    abort_analysis(
      "compact manifest outcome_derivation differs from the protocol"
    )
  }
  require_scalar_character(
    manifest$exporter_version,
    "compact manifest.exporter_version"
  )

  digest_fields <- c(
    "analysis_rows_sha256",
    "source_manifest_sha256",
    "source_checksums_sha256",
    "source_config_file_sha256",
    "source_summary_file_sha256",
    "source_config_sha256",
    "source_tree_sha256",
    "source_input_sha256",
    "exporter_source_sha256"
  )
  if (identical(experiment, "A")) {
    digest_fields <- c(digest_fields, "source_exclusion_sha256")
  }
  for (field in digest_fields) {
    value <- manifest[[field]]
    if (
      !is_scalar_character(value) ||
      !grepl("^[0-9a-f]{64}$", value)
    ) {
      abort_analysis(
        "compact manifest.",
        field,
        " is not a lowercase SHA-256"
      )
    }
  }

  if (!identical(manifest$source_run_id, source$run_id)) {
    abort_analysis("compact bundle source_run_id differs from paired --run")
  }
  if (!identical(
    manifest$source_manifest_sha256,
    sha256_file(file.path(source$path, "manifest.json"))
  )) {
    abort_analysis("compact bundle source manifest digest mismatch")
  }
  if (!identical(
    manifest$source_checksums_sha256,
    sha256_file(file.path(source$path, "SHA256SUMS"))
  )) {
    abort_analysis("compact bundle source SHA256SUMS digest mismatch")
  }
  if (!identical(
    manifest$source_config_file_sha256,
    sha256_file(file.path(source$path, "config.resolved.json"))
  )) {
    abort_analysis("compact bundle source config file digest mismatch")
  }
  if (!identical(
    manifest$source_summary_file_sha256,
    sha256_file(file.path(source$path, "metrics", "summary.json"))
  )) {
    abort_analysis("compact bundle source summary file digest mismatch")
  }
  if (!identical(
    manifest$source_config_sha256,
    source$manifest$config_sha256
  )) {
    abort_analysis("compact bundle source config digest mismatch")
  }
  if (!identical(
    manifest$source_tree_sha256,
    source$manifest$source_sha256
  )) {
    abort_analysis("compact bundle source tree digest mismatch")
  }
  source_input_is_runner_compact <- require_scalar_logical(
    manifest$source_input_is_runner_compact,
    "compact manifest.source_input_is_runner_compact"
  )
  if (
    !identical(manifest$source_input_file, source$input_relative) ||
    !identical(manifest$source_input_sha256, source$input_sha256) ||
    !identical(source_input_is_runner_compact, !source$input_is_legacy)
  ) {
    abort_analysis("compact bundle source input binding mismatch")
  }

  if (identical(experiment, "A")) {
    if (
      !identical(
        manifest$source_exclusion_file,
        source$exclusion_relative
      ) ||
      !identical(
        manifest$source_exclusion_sha256,
        sha256_file(source$exclusion_path)
      )
    ) {
      abort_analysis("compact bundle source exclusion binding mismatch")
    }
  }

  rows_sha256 <- sha256_file(rows_path)
  if (
    !identical(
      rows_sha256,
      unname(checksums[["analysis-rows.jsonl"]])
    ) ||
    !identical(rows_sha256, manifest$analysis_rows_sha256)
  ) {
    abort_analysis("compact bundle analysis row digest mismatch")
  }
  row_lines <- readLines(rows_path, warn = FALSE, encoding = "UTF-8")
  if (!length(row_lines) || any(!nzchar(row_lines))) {
    abort_analysis("compact bundle analysis rows are empty or contain blanks")
  }
  row_count <- require_scalar_integer(
    manifest$row_count,
    "compact manifest.row_count",
    minimum = 1L
  )
  if (row_count != length(row_lines)) {
    abort_analysis("compact bundle row count differs from analysis-rows.jsonl")
  }
  source_record_count <- require_scalar_integer(
    manifest$source_record_count,
    "compact manifest.source_record_count",
    minimum = 1L
  )
  expected_source_count <- if (identical(experiment, "A")) {
    require_scalar_integer(
      source$summary$row_count,
      "summary.row_count",
      minimum = 1L
    )
  } else {
    require_scalar_integer(
      source$summary$trajectories,
      "summary.trajectories",
      minimum = 1L
    )
  }
  if (source_record_count != expected_source_count) {
    abort_analysis(
      "compact bundle source_record_count differs from source summary"
    )
  }
  if (identical(experiment, "A")) {
    if (!is.null(manifest$configured_turns)) {
      abort_analysis("compact Experiment A configured_turns must be null")
    }
    if (row_count != source_record_count) {
      abort_analysis("compact Experiment A row_count differs from source rows")
    }
  } else {
    configured_turns <- require_scalar_integer(
      manifest$configured_turns,
      "compact manifest.configured_turns",
      minimum = 1L
    )
    source_turns <- require_scalar_integer(
      source$config$experiment$turns,
      "config.experiment.turns",
      minimum = 1L
    )
    if (
      configured_turns != source_turns ||
      row_count != source_record_count * source_turns
    ) {
      abort_analysis(
        "compact Experiment B row or turn count differs from source design"
      )
    }
  }

  list(
    path = bundle_dir,
    manifest = manifest,
    manifest_sha256 = sha256_file(manifest_path),
    checksum_manifest_sha256 = sha256_file(checksum_path),
    rows_path = rows_path,
    rows_sha256 = rows_sha256,
    row_count = row_count,
    source_record_count = source_record_count
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
    source$analysis_input_path,
    paste0(source$run_id, " compact Experiment A rows")
  )
  expected_fields <- c(
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
    "update_error"
  )
  rows <- lapply(seq_along(records), function(index) {
    row <- records[[index]]
    label <- paste0("Experiment A compact row ", index)
    assert_exact_fields(row, expected_fields, label)
    if (require_scalar_integer(
      row$schema_version,
      paste0(label, ".schema_version")
    ) != 1L) {
      abort_analysis("unsupported compact Experiment A row schema version")
    }
    source_record_index <- require_scalar_integer(
      row$source_record_index,
      paste0(label, ".source_record_index"),
      minimum = 1L
    )
    mode <- require_scalar_character(
      row$response_mode,
      paste0(label, ".response_mode")
    )
    if (!(mode %in% c("controlled_anchor", "naturally_sampled"))) {
      abort_analysis("unknown compact Experiment A response_mode: ", mode)
    }
    data.frame(
      source_run_id = source$run_id,
      source_record_index = source_record_index,
      trial_id = require_scalar_character(row$trial_id, paste0(label, ".trial_id")),
      user = paste(
        source$run_id,
        require_scalar_character(row$user_id, paste0(label, ".user_id")),
        sep = "::"
      ),
      domain = require_scalar_character(
        row$domain_id,
        paste0(label, ".domain_id")
      ),
      scenario = paste(
        source$run_id,
        require_scalar_character(
          row$scenario_id,
          paste0(label, ".scenario_id")
        ),
        sep = "::"
      ),
      updater = require_scalar_character(
        row$updater_id,
        paste0(label, ".updater_id")
      ),
      mechanism = require_scalar_character(
        row$mechanism,
        paste0(label, ".mechanism")
      ),
      prior_strength = require_scalar_number(
        row$prior_strength,
        paste0(label, ".prior_strength")
      ),
      update_error = require_scalar_number(
        row$update_error,
        paste0(label, ".update_error")
      ),
      response_mode = mode,
      stringsAsFactors = FALSE,
      check.names = FALSE
    )
  })
  all_rows <- do.call(rbind, rows)
  if (!identical(
    all_rows$source_record_index,
    seq_len(nrow(all_rows))
  )) {
    abort_analysis(
      source$run_id,
      " compact Experiment A source_record_index values must cover 1..row_count in order"
    )
  }
  native_declaration_valid <- (
    !is.null(source$compact_bundle) ||
    (
      require_scalar_integer(
        source$summary$analysis_row_count,
        "summary.analysis_row_count",
        minimum = 1L
      ) == nrow(all_rows) &&
      identical(
        source$summary$analysis_artifact,
        "analysis/experiment-a-rows.jsonl"
      )
    )
  )
  if (
    require_scalar_integer(
      source$summary$row_count,
      "summary.row_count",
      minimum = 1L
    ) != nrow(all_rows) ||
    !native_declaration_valid
  ) {
    abort_analysis(
      source$run_id,
      " compact Experiment A source declaration differs from metrics/summary.json"
    )
  }
  normalized <- all_rows[
    all_rows$response_mode == "naturally_sampled",
    ,
    drop = FALSE
  ]
  normalized$response_mode <- NULL
  if (
    require_scalar_integer(
      source$summary$natural_row_count,
      "summary.natural_row_count",
      minimum = 1L
    ) != nrow(normalized)
  ) {
    abort_analysis(
      source$run_id,
      " compact Experiment A row count differs from metrics/summary.json"
    )
  }
  if (!identical(
    sha256_file(source$analysis_input_path),
    source$analysis_input_sha256
  )) {
    abort_analysis(
      source$run_id,
      " compact Experiment A rows changed while being normalized"
    )
  }
  normalized
}

normalize_experiment_b <- function(source) {
  records <- read_jsonl_objects(
    source$analysis_input_path,
    paste0(source$run_id, " compact Experiment B turns")
  )
  expected_fields <- c(
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
    "same_history_shadow"
  )
  trajectory_count <- require_scalar_integer(
    source$summary$trajectories,
    "summary.trajectories",
    minimum = 1L
  )
  configured_turns <- require_scalar_integer(
    source$config$experiment$turns,
    "config.experiment.turns",
    minimum = 1L
  )
  expected_row_count <- trajectory_count * configured_turns
  native_declaration_valid <- (
    !is.null(source$compact_bundle) ||
    (
      require_scalar_integer(
        source$summary$analysis_row_count,
        "summary.analysis_row_count",
        minimum = 1L
      ) == expected_row_count &&
      identical(
        source$summary$analysis_artifact,
        "analysis/experiment-b-turns.jsonl"
      )
    )
  )
  if (
    length(records) != expected_row_count ||
    !native_declaration_valid
  ) {
    abort_analysis(
      source$run_id,
      " compact Experiment B declaration or turn count differs from metrics/summary.json"
    )
  }
  rows <- lapply(seq_along(records), function(index) {
    row <- records[[index]]
    label <- paste0("Experiment B compact row ", index)
    assert_exact_fields(row, expected_fields, label)
    if (require_scalar_integer(
      row$schema_version,
      paste0(label, ".schema_version")
    ) != 1L) {
      abort_analysis("unsupported compact Experiment B row schema version")
    }
    trajectory_id <- require_scalar_character(
      row$trajectory_id,
      paste0(label, ".trajectory_id")
    )
    if (!isTRUE(require_scalar_logical(
      row$same_history_shadow,
      paste0(label, ".same_history_shadow")
    ))) {
      abort_analysis(trajectory_id, " does not retain a same-history shadow")
    }
    data.frame(
      source_run_id = source$run_id,
      source_record_index = require_scalar_integer(
        row$source_record_index,
        paste0(label, ".source_record_index"),
        minimum = 1L
      ),
      source_turn_index = require_scalar_integer(
        row$source_turn_index,
        paste0(label, ".source_turn_index"),
        minimum = 0L
      ),
      trajectory_id = trajectory_id,
      user = paste(
        source$run_id,
        require_scalar_character(row$user_id, paste0(label, ".user_id")),
        sep = "::"
      ),
      domain = require_scalar_character(
        row$domain_id,
        paste0(label, ".domain_id")
      ),
      scenario = paste(
        source$run_id,
        require_scalar_character(
          row$scenario_id,
          paste0(label, ".scenario_id")
        ),
        sep = "::"
      ),
      crn_set = paste(
        source$run_id,
        require_scalar_character(row$crn_key, paste0(label, ".crn_key")),
        sep = "::"
      ),
      updater = require_scalar_character(
        row$updater_id,
        paste0(label, ".updater_id")
      ),
      policy = require_scalar_character(
        row$policy_id,
        paste0(label, ".policy_id")
      ),
      initial_profile = require_scalar_character(
        row$initial_profile_condition,
        paste0(label, ".initial_profile_condition")
      ),
      turn = require_scalar_integer(
        row$turn,
        paste0(label, ".turn"),
        minimum = 1L
      ),
      terminal_error = require_scalar_number(
        row$terminal_error,
        paste0(label, ".terminal_error")
      ),
      retained_terminal_error = require_scalar_number(
        row$retained_terminal_error,
        paste0(label, ".retained_terminal_error")
      ),
      same_history_shadow = TRUE,
      stringsAsFactors = FALSE,
      check.names = FALSE
    )
  })
  normalized <- do.call(rbind, rows)
  expected_source_indexes <- seq_len(trajectory_count)
  if (!identical(
    sort(unique(normalized$source_record_index)),
    expected_source_indexes
  )) {
    abort_analysis(
      source$run_id,
      " compact Experiment B source_record_index values must cover 1..trajectories"
    )
  }
  if (
    !identical(
      normalized$source_record_index,
      rep(expected_source_indexes, each = configured_turns)
    ) ||
    !identical(
      normalized$source_turn_index,
      rep(
        seq.int(0L, configured_turns - 1L),
        times = trajectory_count
      )
    )
  ) {
    abort_analysis(
      source$run_id,
      " compact Experiment B rows are not in canonical source/turn order"
    )
  }
  groups <- split(
    normalized,
    normalized$source_record_index,
    drop = TRUE
  )
  invariant_fields <- c(
    "trajectory_id",
    "user",
    "domain",
    "crn_set",
    "updater",
    "policy",
    "initial_profile",
    "retained_terminal_error",
    "same_history_shadow"
  )
  for (group in groups) {
    trajectory_id <- group$trajectory_id[[1L]]
    for (field in invariant_fields) {
      if (length(unique(group[[field]])) != 1L) {
        abort_analysis(
          trajectory_id,
          " changes compact trajectory field ",
          field
        )
      }
    }
    ordering <- order(group$source_turn_index, method = "radix")
    group <- group[ordering, , drop = FALSE]
    expected_source_turns <- seq.int(0L, configured_turns - 1L)
    if (!identical(group$source_turn_index, expected_source_turns)) {
      abort_analysis(
        trajectory_id,
        " compact turn indexes are not contiguous from zero"
      )
    }
    if (!identical(group$turn, expected_source_turns + 1L)) {
      abort_analysis(
        trajectory_id,
        " compact turn must equal source_turn_index + 1"
      )
    }
    retained_terminal_error <- group$retained_terminal_error[[1L]]
    terminal_tolerance <- 1e-12 + 1e-9 * abs(retained_terminal_error)
    if (
      abs(tail(group$terminal_error, 1L) - retained_terminal_error) >
        terminal_tolerance
    ) {
      abort_analysis(
        trajectory_id,
        " final compact turn error differs from retained_terminal_error"
      )
    }
  }
  if (anyDuplicated(normalized$trajectory_id) != 0L) {
    trajectory_sources <- tapply(
      normalized$source_record_index,
      normalized$trajectory_id,
      function(values) length(unique(values))
    )
    if (any(trajectory_sources != 1L)) {
      abort_analysis(
        source$run_id,
        " compact trajectory_id maps to multiple source records"
      )
    }
  }
  normalized$retained_terminal_error <- NULL
  normalized$same_history_shadow <- NULL
  if (!identical(
    sha256_file(source$analysis_input_path),
    source$analysis_input_sha256
  )) {
    abort_analysis(
      source$run_id,
      " compact Experiment B rows changed while being normalized"
    )
  }
  normalized
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
  if (identical(experiment, "B")) {
    rows$crn_set <- factor(rows$crn_set)
  }

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
