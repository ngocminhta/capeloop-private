# Confirmatory mixed-effects analysis

This optional R project implements the two mixed-effects formulas declared in
the CAPE-Loop proposal. It consumes completed, checksum-verified CAPE-Loop
runs and writes a separate checksum-bound analysis directory. It contains no
study observations or results.

The core Python implementation remains dependency-free. This directory is the
explicit boundary for paper-grade mixed-effects software:

- [`lme4`](https://CRAN.R-project.org/package=lme4) fits the maximal Gaussian
  mixed models;
- [`lmerTest`](https://CRAN.R-project.org/package=lmerTest) supplies
  Satterthwaite degrees of freedom; and
- [`emmeans`](https://CRAN.R-project.org/package=emmeans) evaluates the
  predeclared, coding-invariant contrasts.

The machine-readable source of truth is
[`analysis-spec.json`](analysis-spec.json). The JSON file is a protocol, not a
result.

## Models

Experiment A uses naturally sampled rows and operationalizes proposal
`UpdateError` as Action-Conditioned Update Error (`metrics.acue`):

```text
update_error ~ updater * mechanism + domain + prior_strength
             + (1 + mechanism | user) + (1 | scenario)
```

`user` is `run_id + user_id`; `scenario` is
`run_id + context.scenario_id`. Run prefixes prevent accidental cluster
collisions when matched replications are combined.

Experiment B contributes one analysis row for every retained turn:

```text
terminal_error ~ updater * policy * initial_profile + domain + turn
               + (1 + policy | user) + (1 | scenario)
```

For each row, the outcome named `terminal_error` by the frozen formula is
reconstructed as marginal Brier error from that turn's
`turns[].belief_after` and the trajectory's top-level latent `theta`.
Zero-based retained turn indices are normalized to `1, ..., T`. The final
reconstructed value must equal the trajectory's retained top-level
`terminal_error`; a mismatch aborts analysis. Here `user` is
`run_id + user_id`, and `scenario` is `run_id + crn_key`. The latter is the
complete common-random-number twin set shared by counterfactual policy/updater
branches.

Both models use a Gaussian identity-link likelihood, maximum likelihood
(`REML = FALSE`), treatment coding, `bobyqa`, and a 200,000-evaluation ceiling.
The outcomes are continuous errors; this is why a linear mixed model is used
rather than a binomial or count GLMM. No outcome transformation or automatic
random-effect simplification is applied.

## Installation

Use R 4.6.1 with a working C/C++/Fortran toolchain. On Linux, installation of
`nloptr` may also require CMake. Restore the project-local library:

```bash
cd analysis/confirmatory-mixed-effects
Rscript restore.R
```

`renv.lock` pins R, the five direct runtime packages, and their non-base
runtime dependency graph. `restore.R` also requires the bootstrap `renv`
version to equal 1.2.3. The runner refuses every package-version mismatch.
Each output retains `sessionInfo()` and a digest of the lock file.

The lock was resolved against CRAN package versions available on 2026-07-26.
Refreshing it is a protocol change: update the lock, rerun static validation,
and create a new analysis ID rather than silently reusing v1.

## Continuous integration and static validation

The repository CI has a dedicated R 4.6.1 job. It restores `renv.lock`,
generates a checksum-verified synthetic Experiment A run from
`configs/confirmatory_ci.toml`, executes this harness, verifies every output
checksum, and checks that the result remains explicitly `not_claimed`. The
synthetic fit may validly finish as `complete`, `not_confirmatory`, or
`not_estimable`; CI tests the runtime and failure semantics, not a scientific
finding. Every third-party GitHub Action in that workflow is pinned to an
immutable commit SHA; updating one is a reviewed supply-chain change.

When R is unavailable, the checked-in formulas, contrasts, dependency pins,
result schema, and required source anchors can still be validated statically:

```bash
python3 analysis/confirmatory-mixed-effects/validate_contract.py
```

Success says only that the static analysis contract is internally consistent.
It does not fit a model or validate empirical findings.

## Inputs

Every source must be a completed CAPE-Loop run with `SHA256SUMS` and retained
events:

- Experiment A:
  `events/experiment-a.jsonl`, with `response_mode = naturally_sampled`;
- Experiment B:
  `events/experiment-b-trajectories.jsonl`.

The runner checks the complete run inventory and every source checksum,
manifest status/run ID, resolved experiment kind, summary label, retained-event
declaration, required fields, finite outcomes, duplicate records, factor
coverage, and complete user-level updater-by-treatment cells. It also requires
the retained `config.resolved.json` to be the canonical one-line JSON payload
and recomputes the manifest configuration digest from that payload. It requires
at least eight user and eight scenario clusters. Experiment A requires and
records the excluded-matched-set file and digest even when the file contains
zero rows. Experiment B reconstructs every after-turn marginal Brier score and
checks its final value against the retained terminal score.

Source runs combined in one analysis must be independent repeats of the same
scientific and model declaration, use an identical B horizon/design when
applicable, and have the same `source_sha256`; run IDs are added to cluster
keys. Different source builds or horizons require separate analyses.
Primary and different-model replication roles must be fitted separately rather
than pooled under one updater label.

Before analysis, also use the core verifier:

```bash
PYTHONPATH=src python -m cape_loop verify runs/<run-id>
```

## Execution

The default target is the ordinary full-context writer:

```bash
Rscript analysis/confirmatory-mixed-effects/run_analysis.R \
  --experiment A \
  --run runs/<experiment-a-run> \
  --output analyses/<experiment-a-analysis>
```

Use `--target-updater ID` only when the target was fixed before examining test
outcomes. The reference updater is fixed to `fitted_action_aware` and is not a
command-line choice.

Multiple independent repeats of the same target/model can be combined:

```bash
Rscript analysis/confirmatory-mixed-effects/run_analysis.R \
  --experiment A \
  --run runs/<repeat-1-a-run> \
  --run runs/<repeat-2-a-run> \
  --output analyses/<combined-repeat-a-analysis>
```

The exact Experiment A formula requires variation in `prior_strength`. A run
or pooled set with fewer than two retained prior-strength values is reported as
`not_estimable`; the current one-level pilot configurations therefore cannot
produce this confirmatory fit.

Experiment B uses the same interface:

```bash
Rscript analysis/confirmatory-mixed-effects/run_analysis.R \
  --experiment B \
  --run runs/<closed-loop-run> \
  --output analyses/<closed-loop-analysis>
```

Because B is normalized to one row per retained turn, an ordinary multi-turn
run supplies within-trajectory `turn` variation. A one-turn design is reported
as `not_estimable`; the runner does not remove `turn`. Pooled repeats must use
the same preregistered horizon and factorial design. Combining different
horizons to manufacture fixed-effect variation is not permitted.

## Planned contrasts

For Experiment A, the primary family compares the target updater with
`fitted_action_aware` separately after restricted, defaulted, and suggested
choices. Holm correction is applied across those three tests. The balanced
contrast and each mechanism-versus-balanced updater difference-in-differences
form a separate Holm-corrected secondary family.

These Action-Conditioned Update Error contrasts test whether the target's error
contrast is nonzero relative to the aware reference. They do not test the
direction or magnitude of the target's belief update and therefore do not, by
themselves, establish the proposal's directional H1 updating claim.

For Experiment B, the single primary contrast is the target-versus-aware
updater by soft-versus-balanced policy interaction among incorrect initial
profiles. Secondary contrasts cover the target's policy effect, target versus
aware within both policies, and—when available—the incorrect-versus-correct
three-way contrast. Holm correction is applied within the secondary family.
The terminal-error interaction tests only that model-based error interaction;
it does not alone establish all five self-confirmation clauses or the complete
paper claim.

All contrasts report raw error units and estimates standardized by the
model's conditional residual standard deviation. Their
`pointwise_unadjusted_confidence_lower` and
`pointwise_unadjusted_confidence_upper` fields, including the
`standardized_pointwise_unadjusted_confidence_*` counterparts, are pointwise
95% intervals; Holm adjustment applies to p-values only. Fixed-coefficient and
Type III omnibus tables are secondary, coding-dependent diagnostics; their
non-intercept families are also Holm corrected.

## Diagnostics and failure semantics

`diagnostics.json` retains:

- fixed-effect matrix rank and aliased columns;
- optimizer code/messages and captured warnings;
- maximum absolute raw gradient and the curvature-scaled gradient used for the
  convergence threshold;
- Hessian eigenvalues and positive-definiteness;
- maximal-model singularity;
- post-fit table/contrast warnings or errors;
- user/scenario cluster counts and factor levels; and
- residual/fitted summaries.

The result status is:

- `complete` only when the exact model is full rank, converged, has an
  acceptable Hessian/gradient and residual scale, is nonsingular, and its
  predeclared post-fit inference completes without warnings;
- `not_confirmatory` when the maximal fit fails convergence or singularity
  diagnostics; or
- `not_estimable` when the fixed design cannot be constructed, lacks required
  covariate variation, or is rank deficient.

No status asserts a paper claim: `claim_status` is always `not_claimed`.
Singular or failed fits are not silently simplified. The existing Python CR1
analysis remains a separately labelled robustness result and is never promoted
to the confirmatory result.

## Output contract

The requested output directory must not already exist and cannot be inside a
source run. It contains:

```text
analysis-result.json
diagnostics.json
input-manifest.json
analysis-rows.csv
fixed-effects.csv
omnibus-tests.csv
random-effects.csv
contrasts.csv
session-info.txt
SHA256SUMS
```

`input-manifest.json` binds source run/config/source digests, the canonical
resolved-config payload digest, complete source checksum-manifest digests,
exact event/exclusion digests, normalized row digest, cluster counts, factor
levels, analysis source digests, and the dependency lock. For Experiment B,
`analysis-rows.csv` contains one reconstructed row per retained turn.
`analysis-result.json` follows
[`analysis-result.schema.json`](analysis-result.schema.json).

The output intentionally excludes the fitted binary R object. CSV/JSON
estimates plus the exact inputs, formula, software lock, and session record are
the portable analysis artifact; refitting must reproduce them before release.

## Calibration scope

Event rows contain the active updater state, calibrated when calibration was
configured. Raw and calibrated forecast diagnostics remain in their existing
run artifacts. Experiment B's raw terminal record is a same-realized-history
diagnostic, not a recursively raw trajectory, so this pipeline does not invent
a raw closed-loop mixed model. Any fully raw confirmatory analysis requires a
separate, preregistered, end-to-end raw run.
