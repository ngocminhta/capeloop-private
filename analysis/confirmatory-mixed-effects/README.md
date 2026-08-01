# Confirmatory mixed-effects analysis

This optional R project implements the version-3 mixed-effects contract for
CAPE-Loop. It consumes completed, checksum-verified CAPE-Loop runs and writes
a separate checksum-bound analysis directory. It contains no study observations
or results.

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

## Relationship to the core analyses

The standard-library Python runner emits complete-user paired inference and a
user-clustered CR1 marginal OLS analysis. Experiment B's paper-primary
directional decisions use the one-sided complete-user sign-flip procedure in
`experiment-b-clustered-randomization-v5`; its user-cluster bootstrap intervals
are sensitivity evidence. Other bootstrap and CR1 outputs remain transparent
primary or robustness summaries for their declared artifacts, but none is a
substitute for the proposal's user-random-slope and scenario-random-intercept
models.

This R project is the sole canonical mixed-effects harness. It verifies
completed source runs and normalizes either their runner-native compact rows or
a separately verified compact sidecar for a historical run. It fits the exact
maximal models below, evaluates coding-invariant planned contrasts, retains
complete numerical diagnostics, and publishes a separate checksum-bound
analysis directory. It never edits a source run or promotes a technically
successful fit to a scientific claim.

## Models

Experiment A uses only the predeclared target updater's `controlled_anchor`
same-response rows. Its primary outcome is the signed, anchor-directional
calibration residual:

```text
calibration_residual =
  system_log_odds_update - exact_log_odds_update
```

Both updates are expressed toward the matched anchor direction. Positive
values mean the target updated more strongly toward the observed anchor than
the exact declared-model posterior warrants; negative values mean it updated
less strongly. The single-run model is:

```text
calibration_residual ~ mechanism + domain + prior_strength
                     + (1 + mechanism | user) + (1 | scenario)
```

The exact action-aware posterior is embedded in the outcome calculation, not
included as a deterministic zero-error stochastic group. This oracle is exact
only inside the declared synthetic response model. `exact_update_error`
remains a required descriptive secondary measure of absolute full-belief
update magnitude, and `fitted_update_error` remains a learned-reference
diagnostic. Neither is substituted for the signed primary outcome or tested
using the primary residual fit.

`user` is raw `user_id`, and `scenario` is raw `scenario_id`. A same-seed
rerun therefore cannot manufacture new user or scenario clusters by changing
its run ID. The source design pairs one scenario across anchor directions and
mechanisms while reversing physical anchor position, so `(1 | scenario)`
models the shared stimulus rather than a display-order proxy.

Experiment B contributes one analysis row for every retained turn:

```text
terminal_error ~ updater * policy * initial_profile + domain + turn
               + (1 + policy | user) + (1 | scenario) + (1 | crn_set)
```

For each row, the runner derives the outcome named `terminal_error` by the
frozen formula as marginal Brier error from that turn's `belief_after` and the
trajectory's top-level latent `theta`. The compact file retains both the
zero-based source index and its normalized `1, ..., T` value. R checks their
relationship, complete turn coverage, invariant trajectory metadata, and that
the final compact value equals the trajectory's retained top-level
`terminal_error`; a mismatch aborts analysis. Here `user` is raw `user_id`,
`scenario` is raw `turn.scenario_id`, and `crn_set` is `run_id + crn_key`.
`scenario` therefore changes with the actual stimulus
displayed on each retained turn. `crn_set` instead identifies the complete
common-random-number twin set shared by counterfactual policy/updater branches.
They are deliberately separate random effects: endogenous branches may remain
in one CRN set after their target sequences diverge and cause different
scenarios to be displayed.

Version 3 does not otherwise change Experiment B. Its mixed-model reference remains the
evaluated `fitted_action_aware` updater, while the source run's exact
same-history shadows continue to define the separate selection/attribution
decomposition. This harness does not replace, relabel, or approximate those
shadow quantities.

Both models use a Gaussian identity-link likelihood, maximum likelihood
(`REML = FALSE`), treatment coding, `bobyqa`, and a 200,000-evaluation ceiling.
The outcomes are continuous quantities; this is why a linear mixed model is
used rather than a binomial or count GLMM. No outcome transformation or
automatic random-effect simplification is applied.

When more than one same-seed rerun is pooled, both formulas add
`(1 | replicate)`, where `replicate` is the source `run_id`. Different
`run.seed` values are rejected from a pooled fit and must be analyzed
separately as robustness replicates; seeds never increase the independent-user
count.

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
and create a new analysis ID rather than silently reusing v3.

## Continuous integration and static validation

The repository CI has a dedicated R 4.6.1 job. It restores `renv.lock`,
generates a checksum-verified synthetic Experiment A run from
`analysis/confirmatory-mixed-effects/fixtures/confirmatory_ci.toml`, executes this harness, verifies every output
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

Every source must be a completed CAPE-Loop run with `SHA256SUMS`. Compact
analysis projections are written unconditionally, independently of optional
raw event retention:

- Experiment A:
  `analysis/experiment-a-rows.jsonl`, filtered by R to
  `response_mode = controlled_anchor`, `analysis_track =
  same_response_provenance`, and `reference_basis = exact_action_aware`, plus
  `analysis/experiment-a-exclusions.jsonl`;
- Experiment B:
  `analysis/experiment-b-turns.jsonl`.

The runner checks the complete run inventory and every source checksum,
manifest status/run ID, resolved experiment kind, summary label, compact-file
declaration, registered required/optional field sets, finite outcomes,
duplicate records, factor
coverage, and complete user-level target-mechanism or updater-policy cells. It
also requires
the retained `config.resolved.json` to be the canonical one-line JSON payload
and recomputes the manifest configuration digest from that payload. It requires
at least eight user and eight scenario clusters. Experiment A requires compact
row schema version 2, validates the response-mode/track mapping, exact reference
basis, calibration-residual identity, and target-updater mechanism crossing.
It also records the excluded-matched-set file and digest even
when the file contains zero rows. Experiment B keeps compact row schema version
1, checks every source/turn key, requires contiguous
turns and invariant trajectory metadata, and compares each final compact
marginal Brier score with the retained terminal score.

New runner-native B rows also carry same-history, expected-information,
action-characterization, choice-denominator, and DIR fields. The R harness
accepts that registered optional set but deliberately fits only the core
terminal-error columns. Historical compact sidecars contain the core columns
only. Accordingly, this supporting R fit cannot reproduce or replace the
paper-primary clustered inference in
`metrics/experiment-b-inference.json`.

Historical runs created before these compact files existed remain analyzable
through a three-file bundle produced by `cape_loop artifact compact`. Such a
bundle contains exactly `manifest.json`, `analysis-rows.jsonl`, and
`SHA256SUMS`. R still verifies the complete immutable source run, including the
legacy A event/exclusion files or B trajectory file. It then verifies the
bundle inventory and checksums and binds its run ID, experiment, source
manifest, source checksum manifest, configuration, summary, legacy input,
exclusion, row count, and row digest to that paired source. The large legacy
event file is lineage evidence; R reads the compact sidecar for modeling.

Source runs combined in one analysis must be same-seed reruns of the same
scientific, population-policy, and model declaration, use an identical B
horizon/design when applicable, and have the same `source_sha256`. Raw user and
scenario IDs remain shared clusters, while run ID enters as a crossed replicate
random intercept. Different seeds, source builds, or horizons require separate
analyses.
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
outcomes. References are experiment-specific and are not command-line choices:
Experiment A's outcome uses the `exact_action_aware` oracle; Experiment B retains
`fitted_action_aware`. Experiment A rejects `exact_action_aware` as the target
because its signed residual is deterministically zero by construction.

Multiple same-seed reruns of the same target/model can be combined:

```bash
Rscript analysis/confirmatory-mixed-effects/run_analysis.R \
  --experiment A \
  --run runs/<repeat-1-a-run> \
  --run runs/<repeat-2-a-run> \
  --output analyses/<combined-repeat-a-analysis>
```

The command aborts if the run seeds differ. Run each seed separately and compare
the estimates as robustness replications; do not pool seeds to inflate the user
count.

The exact Experiment A formula requires variation in `prior_strength`. A run
or pooled set with fewer than two retained prior-strength values is reported as
`not_estimable`. The current live pilot configurations use two retained levels
and satisfy this condition, but their four users remain below the required
minimum of eight user and scenario clusters, so they cannot produce a
confirmatory fit.

Experiment B uses the same interface:

```bash
Rscript analysis/confirmatory-mixed-effects/run_analysis.R \
  --experiment B \
  --run runs/<closed-loop-run> \
  --output analyses/<closed-loop-analysis>
```

For a historical run, first create a compact sidecar, then pair it explicitly:

```bash
PYTHONPATH=src python -m cape_loop artifact compact \
  runs/<historical-a-run> artifacts/<historical-a-compact>

Rscript analysis/confirmatory-mixed-effects/run_analysis.R \
  --experiment A \
  --run runs/<historical-a-run> \
  --compact-bundle artifacts/<historical-a-compact> \
  --output analyses/<historical-a-analysis>
```

Supply either no `--compact-bundle` arguments or exactly one per `--run`, in
the same order. Swapped, missing, duplicated, or source-mismatched bundles
abort before fitting. The output must be outside every source run and bundle.

Because B is normalized to one row per retained turn, an ordinary multi-turn
run supplies within-trajectory `turn` variation. A one-turn design is reported
as `not_estimable`; the runner does not remove `turn`. Pooled repeats must use
the same preregistered horizon and factorial design. Combining different
horizons to manufacture fixed-effect variation is not permitted.

## Planned contrasts

For Experiment A, the four primary contrasts compare the target updater's
signed calibration residual after restricted, ranking, defaulted, and suggested
presentation with its residual under balanced presentation:

```text
[system - exact]mechanism - [system - exact]balanced
```

Holm correction is applied across these four tests. Positive contrasts mean the
treatment causes more over-updating toward the observed anchor, or less
under-updating, than balanced presentation; negative contrasts mean the
opposite. The exact oracle remains conditional on the declared simulator.
Absolute `exact_update_error` is reported only as a descriptive secondary
magnitude estimand; the primary residual model does not manufacture an
inferential magnitude contrast from a different outcome.

For Experiment B, this supporting terminal-error model has one predeclared
focal contrast: the target-versus-aware updater by soft-versus-balanced policy
interaction among incorrect initial profiles. The `primary` role in its CSV
output means primary *within this supporting R model*; it is not the
paper-level primary Experiment B estimand. The paper-primary Gate 3 conjunction
of the soft-policy exact same-history gap, its soft-minus-balanced contrast,
and paired exact-shadow SelectionCost is analyzed by
`experiment-b-clustered-randomization-v5`. Secondary R contrasts cover
the target's policy effect, target versus aware within both policies, and—when
available—the incorrect-versus-correct three-way contrast. Holm correction is
applied within the secondary family. The terminal-error interaction does not
alone establish all five self-confirmation clauses. It also does not establish
same-history attribution, evidential noninferiority, or the complete paper
claim by itself.
The adaptive soft-minus-exploratory contrast remains a supporting whole-policy
comparison rather than a primary turn-matched claim.

All contrasts report native outcome units and estimates standardized by the
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
- user/scenario cluster counts and factor levels, including pooled replicate
  and retained Experiment B CRN-set levels; and
- residual/fitted summaries.

The result status is:

- `complete` only when the declared model is full rank, converged, has an
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
source run or compact sidecar bundle. It contains:

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
exact compact-input/exclusion digests, normalized row digest, cluster counts, factor
levels, sidecar manifest/checksum digests when applicable, analysis source
digests, the oracle-versus-factor reference role, pooling rule, population seed,
and the dependency lock. For Experiment B,
`analysis-rows.csv` contains one validated compact row per retained turn.
`analysis-result.json` follows
[`analysis-result.schema.json`](analysis-result.schema.json).

The output intentionally excludes the fitted binary R object. CSV/JSON
estimates plus the exact inputs, formula, software lock, and session record are
the portable analysis artifact; refitting must reproduce them before release.

## Calibration scope

Experiment A's primary outcome subtracts the exact declared-model log-odds
update and is therefore unaffected by fitted-response-model misspecification.
Its sign directly distinguishes over- from under-updating toward the anchor.
The required absolute and fitted-reference errors remain secondary diagnostics.
Evaluated LLM rows still contain the active updater state, calibrated when
calibration was configured. Raw and calibrated forecast diagnostics remain in
their existing run artifacts. Experiment B's raw terminal record is a
same-realized-history diagnostic, not a recursively raw trajectory, so this
pipeline does not invent a raw closed-loop mixed model. Any fully raw
confirmatory analysis requires a separate, preregistered, end-to-end raw run.
