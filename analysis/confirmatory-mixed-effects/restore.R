#!/usr/bin/env Rscript

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (length(script_arg) != 1L) {
  stop("restore.R must be executed with Rscript", call. = FALSE)
}
project_dir <- normalizePath(
  dirname(sub("^--file=", "", script_arg[[1L]])),
  mustWork = TRUE
)
required_renv <- "1.2.3"

installed_renv <- tryCatch(
  as.character(utils::packageVersion("renv")),
  error = function(error) NULL
)
if (!identical(installed_renv, required_renv)) {
  message("Installing the locked renv bootstrap version ", required_renv)
  bootstrap_urls <- c(
    paste0(
      "https://cloud.r-project.org/src/contrib/renv_",
      required_renv,
      ".tar.gz"
    ),
    paste0(
      "https://cloud.r-project.org/src/contrib/Archive/renv/renv_",
      required_renv,
      ".tar.gz"
    )
  )
  bootstrap_errors <- character()
  for (bootstrap_url in bootstrap_urls) {
    attempt <- tryCatch(
      {
        utils::install.packages(
          bootstrap_url,
          repos = NULL,
          type = "source"
        )
        NULL
      },
      error = function(error) conditionMessage(error)
    )
    installed_renv <- tryCatch(
      as.character(utils::packageVersion("renv")),
      error = function(error) NULL
    )
    if (identical(installed_renv, required_renv)) {
      break
    }
    bootstrap_errors <- c(
      bootstrap_errors,
      paste0(
        bootstrap_url,
        ": ",
        if (is.null(attempt)) "version did not match" else attempt
      )
    )
  }
}
if (!identical(installed_renv, required_renv)) {
  stop(
    "renv bootstrap version mismatch: expected ",
    required_renv,
    "; attempts: ",
    paste(bootstrap_errors, collapse = " | "),
    call. = FALSE
  )
}

renv::restore(
  project = project_dir,
  lockfile = file.path(project_dir, "renv.lock"),
  prompt = FALSE,
  clean = TRUE
)
message("Restored locked CAPE-Loop analysis environment at ", project_dir)
