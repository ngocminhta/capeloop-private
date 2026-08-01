empty_fixed_effects <- function() {
  data.frame(
    term = character(),
    estimate = numeric(),
    standard_error = numeric(),
    degrees_of_freedom = numeric(),
    statistic = numeric(),
    p_value = numeric(),
    p_value_holm_secondary_family = numeric(),
    pointwise_unadjusted_confidence_lower = numeric(),
    pointwise_unadjusted_confidence_upper = numeric(),
    stringsAsFactors = FALSE
  )
}

empty_omnibus_tests <- function() {
  data.frame(
    term = character(),
    numerator_df = numeric(),
    denominator_df = numeric(),
    f_statistic = numeric(),
    p_value = numeric(),
    p_value_holm_secondary_family = numeric(),
    stringsAsFactors = FALSE
  )
}

empty_random_effects <- function() {
  data.frame(
    group = character(),
    term_1 = character(),
    term_2 = character(),
    variance_or_covariance = numeric(),
    standard_deviation_or_correlation = numeric(),
    stringsAsFactors = FALSE
  )
}

empty_contrasts <- function() {
  data.frame(
    contrast_id = character(),
    contrast_family = character(),
    confirmatory_role = character(),
    expression = character(),
    estimate = numeric(),
    standard_error = numeric(),
    degrees_of_freedom = numeric(),
    statistic = numeric(),
    p_value = numeric(),
    p_value_holm = numeric(),
    pointwise_unadjusted_confidence_lower = numeric(),
    pointwise_unadjusted_confidence_upper = numeric(),
    standardized_estimate = numeric(),
    standardized_pointwise_unadjusted_confidence_lower = numeric(),
    standardized_pointwise_unadjusted_confidence_upper = numeric(),
    stringsAsFactors = FALSE
  )
}

fixed_design_diagnostics <- function(formula, rows) {
  tryCatch(
    {
      fixed_formula <- lme4::nobars(formula)
      frame <- stats::model.frame(
        fixed_formula,
        data = rows,
        na.action = stats::na.fail,
        drop.unused.levels = TRUE
      )
      matrix <- stats::model.matrix(fixed_formula, data = frame)
      decomposition <- qr(matrix)
      rank <- decomposition$rank
      columns <- ncol(matrix)
      dropped <- if (rank < columns) {
        column_names <- colnames(matrix)
        column_names[decomposition$pivot[seq.int(rank + 1L, columns)]]
      } else {
        character()
      }
      list(
        row_count = nrow(matrix),
        column_count = columns,
        rank = rank,
        full_rank = identical(rank, columns),
        rank_deficient_columns = unname(dropped),
        error = NULL
      )
    },
    error = function(error) {
      list(
        row_count = nrow(rows),
        column_count = NULL,
        rank = NULL,
        full_rank = FALSE,
        rank_deficient_columns = character(),
        error = conditionMessage(error)
      )
    }
  )
}

optimizer_diagnostics <- function(fit, warnings, singularity_tolerance) {
  info <- fit@optinfo
  optimizer_code <- info$conv$opt %||% 0L
  if (length(optimizer_code) > 1L) {
    optimizer_code <- max(abs(as.numeric(optimizer_code)))
  }
  messages <- unlist(info$conv$lme4$messages %||% character(), use.names = FALSE)
  raw_gradient <- as.numeric(info$derivs$gradient %||% numeric())
  max_raw_gradient <- if (length(raw_gradient)) {
    max(abs(raw_gradient))
  } else {
    NULL
  }
  hessian <- info$derivs$Hessian %||% NULL
  scaled_gradient_error <- NULL
  scaled_gradient <- if (!length(raw_gradient) || is.null(hessian)) {
    NULL
  } else {
    tryCatch(
      as.numeric(solve(chol(hessian), raw_gradient)),
      error = function(error) {
        scaled_gradient_error <<- conditionMessage(error)
        NULL
      }
    )
  }
  max_scaled_gradient <- if (
    is.null(scaled_gradient) ||
    !length(scaled_gradient) ||
    any(!is.finite(scaled_gradient))
  ) {
    NULL
  } else {
    max(abs(scaled_gradient))
  }
  hessian_eigenvalues <- if (is.null(hessian)) {
    NULL
  } else {
    tryCatch(
      unname(eigen(hessian, symmetric = TRUE, only.values = TRUE)$values),
      error = function(error) NULL
    )
  }
  hessian_positive_definite <- if (
    is.null(hessian_eigenvalues) ||
    !length(hessian_eigenvalues) ||
    any(!is.finite(hessian_eigenvalues))
  ) {
    NULL
  } else {
    all(hessian_eigenvalues > 0)
  }
  list(
    optimizer = as.character(info$optimizer %||% "unknown"),
    optimizer_code = as.numeric(optimizer_code),
    convergence_messages = as.list(as.character(messages)),
    captured_warnings = as.list(unique(as.character(warnings))),
    gradient_scaling = "inverse_cholesky_hessian",
    raw_gradient = as.list(raw_gradient),
    curvature_scaled_gradient = if (is.null(scaled_gradient)) {
      NULL
    } else {
      as.list(scaled_gradient)
    },
    max_absolute_raw_gradient = max_raw_gradient,
    max_absolute_scaled_gradient = max_scaled_gradient,
    scaled_gradient_error = scaled_gradient_error,
    hessian_eigenvalues = if (is.null(hessian_eigenvalues)) {
      NULL
    } else {
      as.list(hessian_eigenvalues)
    },
    hessian_positive_definite = hessian_positive_definite,
    singular = lme4::isSingular(fit, tol = singularity_tolerance)
  )
}

residual_diagnostics <- function(fit) {
  residuals <- stats::residuals(fit)
  fitted <- stats::fitted(fit)
  quantiles <- stats::quantile(
    residuals,
    probs = c(0, 0.25, 0.5, 0.75, 1),
    names = FALSE,
    type = 8
  )
  list(
    residual_mean = mean(residuals),
    residual_standard_deviation = stats::sd(residuals),
    residual_quantiles = list(
      minimum = quantiles[[1L]],
      first_quartile = quantiles[[2L]],
      median = quantiles[[3L]],
      third_quartile = quantiles[[4L]],
      maximum = quantiles[[5L]]
    ),
    fitted_minimum = min(fitted),
    fitted_maximum = max(fitted),
    residual_sigma = stats::sigma(fit)
  )
}

fixed_effect_table <- function(fit, confidence_level) {
  raw <- as.data.frame(summary(fit)$coefficients)
  degrees <- raw[["df"]]
  critical <- stats::qt(
    1 - (1 - confidence_level) / 2,
    df = degrees
  )
  p_values <- raw[["Pr(>|t|)"]]
  adjusted <- rep(NA_real_, length(p_values))
  secondary <- rownames(raw) != "(Intercept)"
  adjusted[secondary] <- stats::p.adjust(
    p_values[secondary],
    method = "holm"
  )
  data.frame(
    term = rownames(raw),
    estimate = raw[["Estimate"]],
    standard_error = raw[["Std. Error"]],
    degrees_of_freedom = degrees,
    statistic = raw[["t value"]],
    p_value = p_values,
    p_value_holm_secondary_family = adjusted,
    pointwise_unadjusted_confidence_lower = (
      raw[["Estimate"]] - critical * raw[["Std. Error"]]
    ),
    pointwise_unadjusted_confidence_upper = (
      raw[["Estimate"]] + critical * raw[["Std. Error"]]
    ),
    row.names = NULL,
    stringsAsFactors = FALSE
  )
}

omnibus_test_table <- function(fit) {
  raw <- as.data.frame(
    stats::anova(fit, type = 3, ddf = "Satterthwaite")
  )
  p_values <- raw[["Pr(>F)"]]
  data.frame(
    term = rownames(raw),
    numerator_df = raw[["NumDF"]],
    denominator_df = raw[["DenDF"]],
    f_statistic = raw[["F value"]],
    p_value = p_values,
    p_value_holm_secondary_family = stats::p.adjust(
      p_values,
      method = "holm"
    ),
    row.names = NULL,
    stringsAsFactors = FALSE
  )
}

random_effect_table <- function(fit) {
  raw <- as.data.frame(lme4::VarCorr(fit))
  data.frame(
    group = raw$grp,
    term_1 = raw$var1,
    term_2 = ifelse(is.na(raw$var2), "", raw$var2),
    variance_or_covariance = raw$vcov,
    standard_deviation_or_correlation = raw$sdcor,
    stringsAsFactors = FALSE
  )
}

cell_vector <- function(grid, conditions, coefficient = 1) {
  selected <- rep(TRUE, nrow(grid))
  for (field in names(conditions)) {
    selected <- selected & as.character(grid[[field]]) == conditions[[field]]
  }
  indexes <- which(selected)
  if (length(indexes) != 1L) {
    abort_analysis(
      "estimated-marginal-mean cell is not unique: ",
      paste(
        paste(names(conditions), unlist(conditions), sep = "="),
        collapse = ","
      )
    )
  }
  vector <- numeric(nrow(grid))
  vector[[indexes]] <- coefficient
  vector
}

evaluate_contrast_family <- function(
  emmeans_grid,
  methods,
  expressions,
  family,
  role,
  confidence_level,
  residual_sigma
) {
  if (!length(methods)) {
    return(empty_contrasts())
  }
  evaluated <- emmeans::contrast(
    emmeans_grid,
    method = methods,
    adjust = "none"
  )
  raw <- as.data.frame(
    summary(
      evaluated,
      infer = c(TRUE, TRUE),
      level = confidence_level,
      adjust = "none"
    )
  )
  if (!all(names(methods) %in% raw$contrast)) {
    abort_analysis("emmeans contrast identifiers were not preserved")
  }
  raw <- raw[match(names(methods), raw$contrast), , drop = FALSE]
  lower_column <- intersect(c("lower.CL", "asymp.LCL"), names(raw))
  upper_column <- intersect(c("upper.CL", "asymp.UCL"), names(raw))
  statistic_column <- intersect(c("t.ratio", "z.ratio"), names(raw))
  if (
    length(lower_column) != 1L || length(upper_column) != 1L ||
    length(statistic_column) != 1L
  ) {
    abort_analysis("unexpected emmeans summary columns")
  }
  p_holm <- stats::p.adjust(raw$p.value, method = "holm")
  standardization_scale <- if (
    is.numeric(residual_sigma) &&
    length(residual_sigma) == 1L &&
    is.finite(residual_sigma) &&
    residual_sigma > 0
  ) {
    residual_sigma
  } else {
    NA_real_
  }
  data.frame(
    contrast_id = raw$contrast,
    contrast_family = family,
    confirmatory_role = role,
    expression = unname(expressions[raw$contrast]),
    estimate = raw$estimate,
    standard_error = raw$SE,
    degrees_of_freedom = raw$df,
    statistic = raw[[statistic_column]],
    p_value = raw$p.value,
    p_value_holm = p_holm,
    pointwise_unadjusted_confidence_lower = raw[[lower_column]],
    pointwise_unadjusted_confidence_upper = raw[[upper_column]],
    standardized_estimate = raw$estimate / standardization_scale,
    standardized_pointwise_unadjusted_confidence_lower = (
      raw[[lower_column]] / standardization_scale
    ),
    standardized_pointwise_unadjusted_confidence_upper = (
      raw[[upper_column]] / standardization_scale
    ),
    row.names = NULL,
    stringsAsFactors = FALSE
  )
}

experiment_a_contrasts <- function(
  fit,
  target_updater,
  experiment_spec,
  confidence_level
) {
  emmeans_grid <- emmeans::emmeans(
    fit,
    specs = ~ mechanism,
    lmer.df = "satterthwaite"
  )
  grid <- emmeans_grid@grid
  mechanism_gap <- function(mechanism) {
    cell_vector(
      grid,
      list(mechanism = mechanism),
      1
    ) +
      cell_vector(
        grid,
        list(mechanism = "balanced"),
        -1
      )
  }
  primary_mechanisms <- unlist(
    experiment_spec$primary_contrast$mechanisms,
    use.names = FALSE
  )
  primary_methods <- setNames(
    lapply(primary_mechanisms, mechanism_gap),
    paste0(
      "A:calibration-residual:",
      primary_mechanisms,
      "-vs-balanced"
    )
  )
  primary_expressions <- setNames(
    paste0(
      "calibration_residual[",
      target_updater,
      ", ",
      primary_mechanisms,
      "] - calibration_residual[",
      target_updater,
      ", balanced]"
    ),
    names(primary_methods)
  )
  evaluate_contrast_family(
    emmeans_grid,
    primary_methods,
    primary_expressions,
    "A_primary_signed_calibration_mechanism_vs_balanced",
    "primary",
    confidence_level,
    stats::sigma(fit)
  )
}

experiment_b_contrasts <- function(
  fit,
  target_updater,
  reference_updater,
  experiment_spec,
  confidence_level
) {
  emmeans_grid <- emmeans::emmeans(
    fit,
    specs = ~ updater * policy * initial_profile,
    lmer.df = "satterthwaite"
  )
  grid <- emmeans_grid@grid
  updater_policy_gap <- function(initial_profile) {
    target_policy <- cell_vector(
      grid,
      list(
        updater = target_updater,
        policy = "soft_profile_conditioned",
        initial_profile = initial_profile
      ),
      1
    ) +
      cell_vector(
        grid,
        list(
          updater = target_updater,
          policy = "balanced",
          initial_profile = initial_profile
        ),
        -1
      )
    aware_policy <- cell_vector(
      grid,
      list(
        updater = reference_updater,
        policy = "soft_profile_conditioned",
        initial_profile = initial_profile
      ),
      1
    ) +
      cell_vector(
        grid,
        list(
          updater = reference_updater,
          policy = "balanced",
          initial_profile = initial_profile
        ),
        -1
      )
    target_policy - aware_policy
  }
  primary_methods <- list(
    "B:incorrect-profile-updater-by-policy" = updater_policy_gap("incorrect")
  )
  primary_expressions <- c(
    "B:incorrect-profile-updater-by-policy" = paste0(
      "(",
      target_updater,
      "[soft_profile_conditioned] - ",
      target_updater,
      "[balanced]) - (",
      reference_updater,
      "[soft_profile_conditioned] - ",
      reference_updater,
      "[balanced]) | initial_profile=incorrect"
    )
  )
  primary <- evaluate_contrast_family(
    emmeans_grid,
    primary_methods,
    primary_expressions,
    "B_primary_incorrect_profile_interaction",
    "primary",
    confidence_level,
    stats::sigma(fit)
  )

  target_policy_effect <- cell_vector(
    grid,
    list(
      updater = target_updater,
      policy = "soft_profile_conditioned",
      initial_profile = "incorrect"
    ),
    1
  ) +
    cell_vector(
      grid,
      list(
        updater = target_updater,
        policy = "balanced",
        initial_profile = "incorrect"
      ),
      -1
    )
  secondary_methods <- list(
    "B:incorrect-target-policy-effect" = target_policy_effect
  )
  secondary_expressions <- c(
    "B:incorrect-target-policy-effect" = paste0(
      target_updater,
      "[soft_profile_conditioned] - ",
      target_updater,
      "[balanced] | initial_profile=incorrect"
    )
  )
  for (policy in c("balanced", "soft_profile_conditioned")) {
    identifier <- paste0("B:incorrect-target-minus-aware:", policy)
    secondary_methods[[identifier]] <- cell_vector(
      grid,
      list(
        updater = target_updater,
        policy = policy,
        initial_profile = "incorrect"
      ),
      1
    ) +
      cell_vector(
        grid,
        list(
          updater = reference_updater,
          policy = policy,
          initial_profile = "incorrect"
        ),
        -1
      )
    secondary_expressions[[identifier]] <- paste0(
      target_updater,
      "[",
      policy,
      "] - ",
      reference_updater,
      "[",
      policy,
      "] | initial_profile=incorrect"
    )
  }
  if ("correct" %in% as.character(grid$initial_profile)) {
    identifier <- "B:three-way:incorrect-minus-correct"
    secondary_methods[[identifier]] <- updater_policy_gap("incorrect") -
      updater_policy_gap("correct")
    secondary_expressions[[identifier]] <- paste0(
      "updater-by-policy(",
      target_updater,
      " vs ",
      reference_updater,
      ")[incorrect - correct]"
    )
  }
  secondary <- evaluate_contrast_family(
    emmeans_grid,
    secondary_methods,
    secondary_expressions,
    "B_secondary_planned",
    "secondary",
    confidence_level,
    stats::sigma(fit)
  )
  rbind(primary, secondary)
}

not_estimable_result <- function(design, reason) {
  list(
    status = "not_estimable",
    reason = reason,
    fit = NULL,
    diagnostics = list(
      fixed_design = design,
      convergence = NULL,
      residuals = NULL
    ),
    fixed_effects = empty_fixed_effects(),
    omnibus_tests = empty_omnibus_tests(),
    random_effects = empty_random_effects(),
    contrasts = empty_contrasts()
  )
}

fit_confirmatory_model <- function(
  rows,
  experiment,
  experiment_spec,
  estimation_spec,
  target_updater,
  reference_updater
) {
  formula <- stats::as.formula(experiment_spec$formula)
  design <- fixed_design_diagnostics(formula, rows)
  if (!is.null(design$error)) {
    return(not_estimable_result(
      design,
      paste0(
        "fixed-effects design could not be constructed: ",
        design$error
      )
    ))
  }
  if (
    identical(experiment, "A") &&
    length(unique(rows$prior_strength)) < 2L
  ) {
    return(not_estimable_result(
      design,
      paste0(
        "Experiment A requires at least two distinct prior_strength values; ",
        "a one-level pilot design cannot estimate the preregistered slope"
      )
    ))
  }
  if (
    identical(experiment, "B") &&
    length(unique(rows$turn)) < 2L
  ) {
    return(not_estimable_result(
      design,
      paste0(
        "Experiment B requires at least two retained turns; ",
        "a one-turn design cannot estimate the preregistered turn slope"
      )
    ))
  }
  if (!isTRUE(design$full_rank)) {
    return(not_estimable_result(
      design,
      paste0(
        "rank_deficient fixed-effects design; columns: ",
        paste(design$rank_deficient_columns, collapse = ", ")
      )
    ))
  }

  captured_warnings <- character()
  fit_error <- NULL
  fit <- tryCatch(
    withCallingHandlers(
      lmerTest::lmer(
        formula,
        data = rows,
        REML = FALSE,
        control = lme4::lmerControl(
          optimizer = estimation_spec$optimizer,
          optCtrl = list(maxfun = estimation_spec$maxfun),
          calc.derivs = TRUE
        )
      ),
      warning = function(warning) {
        captured_warnings <<- c(
          captured_warnings,
          conditionMessage(warning)
        )
        invokeRestart("muffleWarning")
      }
    ),
    error = function(error) {
      fit_error <<- conditionMessage(error)
      NULL
    }
  )
  if (is.null(fit)) {
    return(list(
      status = "not_confirmatory",
      reason = paste0("maximal mixed-effects fit failed: ", fit_error),
      fit = NULL,
      diagnostics = list(
        fixed_design = design,
        convergence = list(fit_error = fit_error),
        residuals = NULL
      ),
      fixed_effects = empty_fixed_effects(),
      omnibus_tests = empty_omnibus_tests(),
      random_effects = empty_random_effects(),
      contrasts = empty_contrasts()
    ))
  }

  convergence <- optimizer_diagnostics(
    fit,
    captured_warnings,
    estimation_spec$singularity_tolerance
  )
  gradient_ok <- is.numeric(convergence$max_absolute_scaled_gradient) &&
    length(convergence$max_absolute_scaled_gradient) == 1L &&
    is.finite(convergence$max_absolute_scaled_gradient) &&
    convergence$max_absolute_scaled_gradient <=
      estimation_spec$gradient_tolerance
  hessian_ok <- identical(convergence$hessian_positive_definite, TRUE)
  residual_sigma <- stats::sigma(fit)
  residual_scale_ok <- is.numeric(residual_sigma) &&
    length(residual_sigma) == 1L &&
    is.finite(residual_sigma) &&
    residual_sigma > 0
  converged <- isTRUE(convergence$optimizer_code == 0) &&
    !length(convergence$convergence_messages) &&
    !length(convergence$captured_warnings) &&
    gradient_ok &&
    hessian_ok &&
    residual_scale_ok
  confirmatory <- converged && !isTRUE(convergence$singular)
  status <- if (confirmatory) "complete" else "not_confirmatory"
  reason <- if (confirmatory) {
    NULL
  } else {
    paste(
      c(
        if (!converged) "optimizer/convergence/residual diagnostics failed",
        if (isTRUE(convergence$singular)) "maximal random-effects fit is singular"
      ),
      collapse = "; "
    )
  }

  post_fit_error <- NULL
  post_fit_warnings <- character()
  post_fit <- tryCatch(
    withCallingHandlers(
      {
        contrasts <- if (identical(experiment, "A")) {
          experiment_a_contrasts(
            fit,
            target_updater,
            experiment_spec,
            estimation_spec$confidence_level
          )
        } else {
          experiment_b_contrasts(
            fit,
            target_updater,
            reference_updater,
            experiment_spec,
            estimation_spec$confidence_level
          )
        }
        list(
          fixed_effects = fixed_effect_table(
            fit,
            estimation_spec$confidence_level
          ),
          omnibus_tests = omnibus_test_table(fit),
          random_effects = random_effect_table(fit),
          contrasts = contrasts,
          residuals = residual_diagnostics(fit)
        )
      },
      warning = function(warning) {
        post_fit_warnings <<- c(
          post_fit_warnings,
          conditionMessage(warning)
        )
        invokeRestart("muffleWarning")
      }
    ),
    error = function(error) {
      post_fit_error <<- conditionMessage(error)
      NULL
    }
  )
  if (is.null(post_fit) || length(post_fit_warnings)) {
    status <- "not_confirmatory"
    post_fit_reason <- if (is.null(post_fit)) {
      paste0("post-fit inference failed: ", post_fit_error)
    } else {
      "post-fit inference emitted warnings"
    }
    reason <- paste(
      c(
        if (!is.null(reason) && nzchar(reason)) reason,
        post_fit_reason
      ),
      collapse = "; "
    )
  }
  if (is.null(post_fit)) {
    post_fit <- list(
      fixed_effects = empty_fixed_effects(),
      omnibus_tests = empty_omnibus_tests(),
      random_effects = empty_random_effects(),
      contrasts = empty_contrasts(),
      residuals = NULL
    )
  }
  final_confirmatory <- confirmatory &&
    is.null(post_fit_error) &&
    !length(post_fit_warnings)
  list(
    status = status,
    reason = reason,
    fit = fit,
    diagnostics = list(
      fixed_design = design,
      convergence = c(
        convergence,
        list(
          gradient_within_tolerance = gradient_ok,
          hessian_acceptable = hessian_ok,
          residual_scale_acceptable = residual_scale_ok,
          converged = converged,
          confirmatory_diagnostics_passed = final_confirmatory
        )
      ),
      post_fit = list(
        completed = is.null(post_fit_error),
        acceptable_for_confirmatory = (
          is.null(post_fit_error) && !length(post_fit_warnings)
        ),
        warnings = as.list(unique(post_fit_warnings)),
        error = post_fit_error
      ),
      residuals = post_fit$residuals
    ),
    fixed_effects = post_fit$fixed_effects,
    omnibus_tests = post_fit$omnibus_tests,
    random_effects = post_fit$random_effects,
    contrasts = post_fit$contrasts
  )
}
