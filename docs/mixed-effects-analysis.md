# Confirmatory mixed-effects analysis

CAPE-Loop includes an optional, version-pinned R harness for the two
mixed-effects formulas in the proposal. The harness is executable analysis
software; the repository still contains no fitted paper result.

Its complete operator and statistical reference is
[`analysis/confirmatory-mixed-effects/README.md`](../analysis/confirmatory-mixed-effects/README.md).
The frozen machine-readable protocol is
[`analysis-spec.json`](../analysis/confirmatory-mixed-effects/analysis-spec.json).

## Boundary with the Python analysis

The standard-library Python runner emits user-clustered CR1 marginal OLS and
paired bootstrap results. Those are transparent robustness analyses. They are
not substitutes for the proposal's user-random-slope/scenario-random-intercept
models.

The optional R harness uses:

- Gaussian identity-link mixed models fit by
  [`lme4`](https://CRAN.R-project.org/package=lme4);
- Satterthwaite degrees of freedom from
  [`lmerTest`](https://CRAN.R-project.org/package=lmerTest); and
- coding-invariant planned contrasts from
  [`emmeans`](https://CRAN.R-project.org/package=emmeans).

The environment pins R 4.6.1 and every non-base runtime dependency in
`renv.lock`.

## Exact models

Experiment A consumes naturally sampled event rows and maps `UpdateError` to
the retained Action-Conditioned Update Error:

```text
update_error ~ updater * mechanism + domain + prior_strength
             + (1 + mechanism | user) + (1 | scenario)
```

Experiment B consumes complete trajectories but normalizes them to one row per
retained turn:

```text
terminal_error ~ updater * policy * initial_profile + domain + turn
               + (1 + policy | user) + (1 | scenario)
```

Run IDs prefix user and scenario keys when matched runs are combined. For A,
scenario is the retained `context.scenario_id`. For B, scenario is the
common-random-number `crn_key` shared by policy/updater twins.

For B, each row's outcome retains the frozen formula name `terminal_error` but
is reconstructed as marginal Brier error from `turns[].belief_after` against
the trajectory's top-level `theta`. Retained zero-based indices are normalized
to `turn = 1, ..., T`, and the final reconstructed error must equal the
top-level retained `terminal_error`. Thus an ordinary multi-turn run supplies
within-trajectory `turn` variation. A one-turn design is `not_estimable`;
`turn` is never silently removed.

The A formula likewise requires at least two retained `prior_strength` values.
The current one-level pilot configurations are correctly `not_estimable` for
this exact model.

## Run it

First verify every source run:

```bash
PYTHONPATH=src python -m cape_loop verify runs/<run-id>
```

Restore the R environment:

```bash
cd analysis/confirmatory-mixed-effects
Rscript restore.R
```

Then fit one model from the repository root:

```bash
Rscript analysis/confirmatory-mixed-effects/run_analysis.R \
  --experiment A \
  --run runs/<run-id> \
  --output analyses/<analysis-id>
```

Repeat `--run` to combine independent repeats of the same scientific and model
declaration. Primary and different-model replication roles are fitted
separately, not pooled under one updater label. The default target is
`llm_full_context`; a preregistered alternative may be supplied with
`--target-updater`. Pooled repeats must have identical factorial designs and B
horizons and the same source-code `source_sha256`; cross-horizon pooling is not
a workaround for missing variation.

The runner independently verifies the full source inventories and checksums,
requires the retained resolved configuration to be canonical one-line JSON,
recomputes its manifest configuration digest from that canonical payload,
validates factorial cells and cluster counts, refuses an existing output
directory, and never writes into a source run. For B it also reconstructs every
after-turn outcome and verifies the final-outcome consistency check before
fitting.

## Inference and diagnostics

Experiment A's primary family compares the target with
`fitted_action_aware` within restricted, default, and suggested mechanisms,
with Holm correction. The balanced contrast and updater-by-mechanism
difference-in-differences form a separate secondary family.

These ACUE contrasts test whether the target's error contrast is nonzero
relative to the aware reference. They do not test the direction or magnitude
of the target's belief update and cannot alone establish the proposal's
directional H1 updating claim.

Experiment B's primary contrast is the target-versus-aware updater by
soft-versus-balanced policy interaction for incorrect initial profiles.
Predeclared secondary contrasts are Holm corrected. This terminal-error
interaction does not alone establish all five self-confirmation clauses.

Reported confidence limits use the explicit
`pointwise_unadjusted_confidence_lower` and
`pointwise_unadjusted_confidence_upper` fields (and standardized equivalents).
They are pointwise 95% intervals; Holm adjustment applies to p-values, not
confidence limits.

Every output records fixed-effect rank, optimizer status, warnings, the raw
gradient, the curvature-scaled gradient used for convergence, Hessian,
singularity, variance components, residual summaries, package versions,
formulas, coding, contrasts, and exact input/source digests. A fit is
`complete` only if the maximal model is full rank, converged, nonsingular,
passes its numerical checks, has a positive finite residual scale, and
completes the predeclared post-fit inference without warnings. A fixed design
that cannot be constructed, lacks required covariate variation, or is rank
deficient is `not_estimable`; other failed numerical checks are
`not_confirmatory`. The random-effects structure is never simplified
automatically.

## Outputs and interpretation

The separate analysis directory contains JSON summaries, normalized input
rows (one per retained turn for B), fixed/omnibus/random-effect CSVs, planned
contrasts, `sessionInfo()`, and `SHA256SUMS`. See
[Outputs](outputs.md#optional-r-mixed-effects-analysis).

Every result fixes `claim_status = "not_claimed"`. A technically successful fit
still requires preregistration, author review, multiplicity interpretation,
robustness checks, and paper-artifact freezing before it can support a claim.

The harness uses the active event state, calibrated when configured. It does
not transform same-history raw terminal diagnostics into a fictional
recursively raw closed-loop trajectory. A raw end-to-end mixed model requires
a separately declared raw run.
