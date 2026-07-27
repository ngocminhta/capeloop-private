# Run outputs

## Run identity

The default output path is:

```text
<run.output_root>/<run.name>-<config-digest-prefix>
```

`config-digest-prefix` is the first 12 hexadecimal characters of SHA-256 over
the fully resolved configuration's canonical JSON.

```bash
PYTHONPATH=src python -m cape_loop run configs/smoke.toml
```

prints JSON with the exact `run_dir`. `--output-root DIR` changes the parent but
not the digest. An existing directory is rejected unless `--allow-existing`
finds a checksum-valid result with a summary and matching current source digest.
For LLM replay, the retained input-corpus manifest must also exactly match the
currently configured response file.

## Files created at run start

The artifact layer creates:

```text
runs/<run-id>/
├── config.resolved.json
├── config.source.toml
├── environment.json
├── manifest.json
├── events/
├── figures/
├── llm/
├── metrics/
├── models/
└── tables/
```

`config.source.toml` is written by the CLI because it passes the input path to
the runner. A direct Python API caller may omit `source_config`, in which case
that file is absent and the manifest records a hash-bound `programmatic`
configuration origin instead.

### `config.resolved.json`

Contains every default after strict validation. This exact canonical
configuration determines the config digest and run ID.

### `config.source.toml`

Byte-for-byte text read from the source configuration passed to the CLI. It
preserves comments and author formatting but does not determine identity
independently of the resolved JSON. Verification reparses this TOML and requires
its resolved config digest to equal `config.resolved.json` and the manifest.

### `environment.json`

Contains:

- full Python version;
- implementation name;
- platform string; and
- interpreter path.

It is not a complete operating-system image or external-harness environment.

### `manifest.json`

Starts with:

```json
{
  "schema_version": 1,
  "run_id": "...",
  "config_origin": {
    "kind": "toml_file",
    "retained_file": "config.source.toml",
    "source_filename": "smoke.toml",
    "source_sha256": "...",
    "config_sha256": "..."
  },
  "config_sha256": "...",
  "git_revision": null,
  "source_sha256": "...",
  "deterministic": true,
  "status": "created"
}
```

`git_revision` is `null` outside a Git worktree. `source_sha256` hashes every
Python file under `src/cape_loop`, `py.typed`, and `pyproject.toml` in stable
path order.

On success, status becomes `complete`. On a caught runner failure, status
becomes `failed` and `failure.json` records the exception class and message.

## Shared preparation artifacts

Experiments A–C add:

```text
splits.json
population/users.jsonl
models/raw-fitted-likelihoods.json
models/fitted-likelihoods.json
models/calibration.json
models/held-out-response-diagnostics.json
events/fitted-model-training.jsonl  # only when retain_events = true
events/fitted-model-development.jsonl # only when retain_events = true
metrics/split-leakage-audit.json
```

`population/users.jsonl` contains evaluator-side latent records for train,
development, and test groups in every selected domain. It is not updater input.
The split audit proves that the actual option, dialogue, scenario, and
paraphrase families consumed by the runner are disjoint; see
[Data splits](data-splits.md).

### Raw versus active fitted models

`raw-fitted-likelihoods.json` is the direct training fit.
`fitted-likelihoods.json` is the active bundle supplied to fitted updaters. With
temperature calibration, aware and unaware coefficients in the active bundle
are divided by their separately fitted development temperatures.

`calibration.json` records those transformations.
`held-out-response-diagnostics.json` contains active diagnostics plus raw
diagnostics. Aware option NLL and unaware semantic NLL use different outcome
spaces and must not be compared as if they were the same score.

Sensitivity does not write the shared preparation files above. Instead, every
grid row in `models/sensitivity-fits.jsonl` embeds its raw bundle, active bundle,
calibration, and development diagnostics.

## Experiment-specific files

### Experiment A

```text
events/experiment-a.jsonl                  # conditional
events/experiment-a-exact-references.jsonl # conditional
events/experiment-a-exclusions.jsonl       # conditional
events/experiment-a-held-out-paraphrases.jsonl # conditional
models/experiment-a-control-battery.json
models/experiment-a-control-plan.json
models/held-out-paraphrase-suite.json
llm/experiment-a-control-exchange.json
llm/experiment-a-control-request-bindings.jsonl
llm/experiment-a-control-requests.jsonl
metrics/experiment-a.jsonl
metrics/experiment-a-confirmatory.json
metrics/experiment-a-control-reference.json
metrics/experiment-a-control-baseline.json
metrics/experiment-a-hypothesis-estimands.json
metrics/experiment-a-oracle-slopes.jsonl
metrics/experiment-a-evidence-strength.json
metrics/experiment-a-raw-calibrated-scores.jsonl
metrics/experiment-a-reliability.jsonl
metrics/experiment-a-multiplicity.json
metrics/experiment-a-power.json
metrics/experiment-a-held-out-paraphrase-scores.jsonl
metrics/experiment-a-held-out-paraphrase-transfer.json
tables/experiment-a-brier.csv
tables/experiment-a-raw-calibrated-scores.csv
tables/experiment-a-reliability.csv
figures/experiment-a-update-magnitude.svg  # when controlled rows exist
metrics/gate-report.json
metrics/summary.json
```

Experiment A updater rows retain theta beliefs and an `exact_reference_id`.
The separate exact-reference file retains one full exact theta posterior and
theta×susceptibility posterior per trial, avoiding repetition across updater
rows. Join the two files on `exact_reference_id`.

Every Experiment A row also names `prior_stratum` and numeric
`prior_strength`. A configured concentration grid is crossed with every
matched mechanism while context and sampled response remain paired across
strata. `experiment-a-control-battery.json` content-binds the three positive
controls (volunteered preference, repeated balanced cross-context choices, and
direct correction) and three negative controls (indifference, random choice,
and target-nondistinguishing response). The battery itself remains a protocol,
not a result file: the one-step anchor runner does not fabricate outcomes for
signals its choice schema cannot faithfully encode.

`experiment-a-control-plan.json` separately materializes those signals as six
typed, content-addressed stimuli. The reference and baseline files are
complete deterministic diagnostics and explicitly are not external evidence.
The exchange manifest and two JSONL files contain a six-case
provenance-aware provider packet: the generic requests are consumable by the
existing OpenAI/OpenRouter commands, while the outer bindings retain plan,
battery, and stimulus hashes. Imported responses are scored outside the
immutable run with `cape-loop control-study analyze`; see
[Experiment A control execution](experiment-a-controls.md).

`experiment-a-confirmatory.json` groups the oracle slopes, fitted
evidence-strength ordering, clustered mechanism contrasts,
updater×mechanism interactions, raw/calibrated comparison, and optional
marginal regression. The regression artifact identifies itself as marginal OLS
with user-clustered CR1 covariance. It must not be reported as the proposal's
user-random-slope/scenario-random-intercept generalized mixed-effects model.

`experiment-a-hypothesis-estimands.json` separately freezes H1's
mechanism-wise directional and update-strength contrasts, H2's
aware-versus-unaware update-vector proximity criterion, and H7's update-error
superiority plus balanced/volunteered valid-learning noninferiority. Missing
LLM rows or volunteered direct-statement outcomes remain `null`, never imputed.
See [H1, H2, and H7 estimands](hypothesis-estimands.md).

Volunteered evidence is deliberately produced outside the immutable run.
`control-study h7-plan` writes:

```text
h7-volunteered-plan.json
h7-volunteered-request-bindings.jsonl
h7-volunteered-requests.jsonl
```

After complete provider collection, `control-study h7-review` writes one
`h7_volunteered_control_review` JSON artifact. It binds the verified source
run, all three plan files, response and accepted provider-audit files, every
derived `VolunteeredPreferenceUpdate`, and each provider-bound posterior. It
recomputes only volunteered valid learning and the overall Experiment A H7
status while retaining the source ACUE-superiority and balanced components by
hash. `control-study h7-verify` reconstructs the artifact from all inputs. The
review is not placed under the source run and always remains `not_claimed`.
See [H7 volunteered-preference controls](h7-volunteered-controls.md).

The raw/calibrated score and reliability files retain case-bound forecast
scores and one-vs-rest marginal-class reliability bins. Multiplicity uses Holm
over estimable non-intercept CR1 coefficients. Power is either a paired
user-cluster pilot simulation or an explicit `not_estimable` record; a
configured zero bootstrap count selects the recorded 200-replicate smoke
fallback.

The paraphrase suite fixes split families and content hashes. Held-out cases
bind surface responses to source trials, and the transfer artifact distinguishes
`verified: null` for missing required pairs from a computed true/false result.

### Experiment B

```text
events/experiment-b-trajectories.jsonl         # conditional
events/experiment-b-terminal-batteries.jsonl   # conditional
events/experiment-b-held-out-terminal-suites.jsonl # conditional
metrics/experiment-b-terminal.jsonl
metrics/experiment-b-native-decoders.jsonl
metrics/experiment-b-held-out-actions.jsonl
metrics/experiment-b-terminal-calibration.json
metrics/experiment-b-decomposition.jsonl
metrics/experiment-b-h7-mitigation.json
metrics/experiment-b-self-confirmation.jsonl
metrics/experiment-b-inference.json
metrics/experiment-b-power.json
metrics/experiment-b-llm-raw-calibrated-terminal.jsonl
metrics/experiment-b-llm-raw-calibrated-terminal-manifest.json
tables/experiment-b-decomposition.csv
tables/experiment-b-power.md
tables/experiment-b-llm-raw-calibrated-terminal.csv
decoder/external-requests.jsonl
decoder/truth-labels.researcher-only.jsonl
decoder/researcher-codebook.jsonl
decoder/design-manifest.json
metrics/gate-report.json
metrics/summary.json
```

The trajectory file includes evaluator-only latent truth, complete audit
records, shadow beliefs, action-influence counterfactual signatures, and full
native before/after state when applicable.

Exact evaluated updaters and the exact shadow retain their full
theta×susceptibility joint state before/after turns and at termination, not only
theta marginals.

`experiment-b-terminal-batteries.jsonl` is the common projection form of
heldout-terminal-v2 used for B/C evaluation.
`experiment-b-held-out-terminal-suites.jsonl` retains the corresponding
action-contract form. Their IDs, features, wording IDs, scenario families, and
four question types passed the training-overlap check. Each held-out action
score repeats the suite digest and names its transparent adapter kind.
`native_persona_action_reference` additionally binds the native state ID. These
are reference-projection action records, not opaque external natural-language
calls, and Gate 4 never counts them as native end-to-end actions. The two fixed
native decoder rows are likewise deterministic representation diagnostics,
not imported independent judgments.

Every terminal score embeds `profile_ece`,
`profile_calibration_sample_unit = "preference_attribute_forecast"`, a
prediction count, and fixed reliability bins.
`experiment-b-terminal-calibration.json` pools those bins by score projection,
updater, and decoder while naming trajectory/user dependence explicitly.

`experiment-b-inference.json` reports paired complete-trajectory estimands with
deterministic user-clustered primary intervals and paired-trajectory
sensitivity intervals. It includes cluster counts, the minimum-eight-user
adequacy decision, bootstrap count, and explicit `not_computed` status when
bootstrapping is disabled. The artifact states that this dependency-free
analysis is not a mixed-effects model or GLMM.

`experiment-b-h7-mitigation.json` compares matched ordinary full-context and
provenance-aware trajectories under soft profile conditioning with incorrect
initial profiles. It reports reductions in same-history attribution error and
the five-clause self-confirming-profile rate. It is only H7's closed-loop
component; the Experiment A mitigation and valid-learning criteria remain
separately required.

`experiment-b-power.json` is the checksum-retained, machine-readable
pilot-design calculation for the frozen full-context-versus-fitted-aware ×
soft-versus-balanced × incorrect-versus-correct terminal-error contrast. It
includes the exact formula and factor IDs, a digest of every contributing
trajectory/error input, complete-user inclusion and exclusion records, the
bounded simulation settings and assumptions, the 16/32/64/128 curve, Monte
Carlo standard errors and Wilson intervals, and the conservative 0.80
threshold decision. `experiment-b-power.md` renders the same information for a
human reviewer. Both files say `scientific_claim_status = "not_claimed"` and
prohibit an automatic sample-size commitment.

When LLM temperature calibration is active, the raw/calibrated terminal table
scores both cached probability vectors for the identical final request and
common terminal battery. It makes zero additional provider calls and preserves
the realized trajectory pairing. For a multi-turn history, however, the raw
row is only the raw terminal forecast conditional on the *calibrated active
history*. It is not a recursively raw trajectory: such a rerun may change later
priors, prompts, model outputs, and profile-conditioned actions. Each row and
the manifest carry this limitation, and these diagnostics do not replace gate
inputs.

The decoder request file is safe to send only by itself. The truth-label and
researcher-codebook files contain evaluator-only linkage and must remain
separate from decoder inputs. `design-manifest.json` records development/test
counts, the minimum two-source/distinct-family design, blinding status, and
`independence_claimed = false`.

### Experiment C

```text
events/experiment-c-fixed-histories.jsonl  # conditional
events/experiment-c-replays.jsonl          # conditional
events/experiment-c-endogenous.jsonl       # conditional
events/terminal-batteries.jsonl             # unconditional
decoder/experiment-c-external-requests.jsonl
decoder/experiment-c-truth-labels.researcher-only.jsonl
decoder/experiment-c-researcher-codebook.jsonl
decoder/experiment-c-external-design-manifest.json
metrics/experiment-c.jsonl
metrics/experiment-c-terminal-calibration.json
metrics/experiment-c-rankings.json
metrics/experiment-c-llm-raw-calibrated-terminal.jsonl
metrics/experiment-c-llm-raw-calibrated-terminal-manifest.json
tables/experiment-c-ranks.csv
tables/experiment-c-llm-raw-calibrated-terminal.csv
metrics/gate-report.json
metrics/summary.json
```

Fixed-history records contain the context, provenance, and observation once.
Replay records show every updater consuming those same event signatures.
Endogenous records include evaluator-only latent truth and complete native
before/after state where applicable.

`events/terminal-batteries.jsonl` retains the same heldout-terminal-v2-derived
diagnostic across all compared C systems: novel option IDs/features, scenario
families, wording IDs, and all four question types. The ranking evaluator uses
those feature vectors rather than the domain training option pool.

Experiment C metric rows identify `score_basis`. Structured rows rank the public
structured projection. Native rows rank the arithmetic mean of exactly two
blinded decoder scores, retain the public persona projection under
`system_projection_score`, and retain individual decoder scores and predictions
under `native_decoder_evaluations`.

Replay updater states and endogenous closed-loop records also retain full
theta×susceptibility joint state whenever the updater has one; the exact shadow
always does in closed loop.

`experiment-c-rankings.json` retains paired open-, closed-, and test
error-difference intervals, joint paired open-versus-closed
difference-of-differences intervals, and interval-supported partial orders. A
same-tier placement means interval evidence did not establish dominance, not
that equality was proved. Evaluation selection regret records the inferential
open/closed top tiers, mean/min/max test regret over every set pairing, and a
conservative envelope over paired closed-test intervals. Marginal rank
intervals do not establish a reversal.

`experiment-c-terminal-calibration.json` pools ECE/reliability records by
split, regime, updater, score projection, and decoder. Metric rows retain the
exact `ranking_score` calibration separately from the public system projection
and individual deterministic decoder projections.

The same object records `inference_unit`, the stable `alignment_key`,
development/test user-cluster counts, the shared domain × replicate component
layout, and the bootstrap method. Ranking inputs contain one paired value per
complete latent user; domains and trajectory replicates are never resampled as
independent rows.

The Experiment C raw/calibrated terminal files use the same cached-vector
estimand as Experiment B for fixed and endogenous histories. They are paired
same-request diagnostics only. Multi-turn raw counterfactual rankings require a
new recursive run with complete response coverage for the alternate prompts;
the cached table is therefore never substituted into
`experiment-c-rankings.json` or Gate 5.

Native C rows also produce a blinded external-decoder packet. A later
`experiment-c-decoder import` writes a separate checksum-bound review and
reruns rankings, ESR, and Gate 5 without modifying this source run. The exact
packet, review contents, calibration boundary, and verification commands are
documented in
[Experiment C external-decoder rescore](experiment-c-external-decoder.md).
The CLI requires either `--external-collection-dir DIR`, which validates and
binds the complete selected first-party collection, or
`--allow-reviewed-generic-decoders`, which explicitly records that all
origin/family/source metadata are caller-declared and makes no provider-
provenance assertion.
Publication uses an exclusive sibling lock and a durable same-parent stage;
the source and judgment inputs plus the staged review are reverified before one
atomic rename exposes the final directory.

Two or more independently seeded completed Experiment C runs can be supplied
to `experiment-c-robustness review`. The command verifies each source and
writes a separate atomic directory containing only:

```text
review.json
manifest.json
SHA256SUMS
```

The review binds every source checksum/config/ranking/gate/summary digest and
retains its positive clustered-bootstrap count and seed. It compares the three
point rankings, open/closed inferential top tiers and partial orders, Gate 5
decision/status, and ESR selection sets using exact rational agreement
fractions plus explicit disagreements. It never rewrites a source or infers a
scientific claim. See
[Experiment C multi-seed robustness](experiment-c-robustness.md).

### Sensitivity

```text
models/sensitivity-fits.jsonl
metrics/sensitivity.jsonl
metrics/sensitivity-decomposition.jsonl
metrics/sensitivity-grand.jsonl
metrics/sensitivity-phase-points.jsonl
metrics/sensitivity-phase-domains.jsonl
metrics/sensitivity-phase-boundaries.jsonl
metrics/sensitivity-phase-specification.json
tables/sensitivity.csv
events/sensitivity-trajectories.jsonl  # conditional
metrics/gate-report.json
metrics/summary.json
```

The compact checked-in grid has 81 points; the broader checked-in declaration
has 384. Every row names decision/shared-presentation values, independent
rank/default/suggestion multipliers, profile strength, prior uncertainty,
trajectory length, response-model family, and rule noise when applicable.
Primary rows are updater×policy×domain strata; decomposition rows retain paired
selection/attribution contrasts; grand rows are descriptive summaries.

Phase-point rows retain every declared criterion as true, false, or incomplete.
Grand and domain-phase rows retain profile-consistent suggestion opportunities,
rejections, and rejection rates. The frozen meaningful-region criterion uses
that rate and remains incomplete when there are no eligible suggestions.
Boundary rows are adjacent observed grid intervals where a criterion changes,
not interpolated thresholds. The specification retains the criterion
definitions and boundary axes. Replay, direct OpenAI, and OpenRouter are valid
external-LLM evidence modes when their complete exchange evidence is retained.
The event file contains evaluator-only truth in retained trajectories.

LLM sensitivity also writes
`llm/sensitivity-request-preflight.json`, `llm/requests.jsonl`, and
`llm/responses.jsonl`. Live modes additionally write
`llm/provider-audit.jsonl` and `llm/transport-attempts.jsonl`.

“Conditional” above means controlled by `[artifacts].retain_events`.

## Standalone protocol and analysis outputs

### Optional R mixed-effects analysis

The confirmatory R harness writes outside the immutable source run:

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

`input-manifest.json` binds every complete source run, its configuration/source
digests, its full checksum-manifest digest, the exact event and exclusion
inputs, the normalized-row digest, analysis source digests, factor levels, and
cluster counts. The runner recomputes the configuration digest from the
retained canonical `config.resolved.json` payload. Pooled repeats must share
the same `source_sha256`, factorial design, and, for B, horizon.

Experiment B's `analysis-rows.csv` has one row per retained turn. Its outcome
column keeps the protocol name `terminal_error`, but each value is marginal
Brier error reconstructed from that turn's `belief_after` and the trajectory's
top-level `theta`; stored zero-based turns become `1, ..., T`. The final
reconstructed value must equal the retained top-level `terminal_error`.

`diagnostics.json` records fixed-design rank, optimizer, raw gradient, the
curvature-scaled gradient used for convergence, Hessian, singularity, and
residual summaries; `random-effects.csv` records the fitted variance
components. Confidence-limit columns in the fixed-effect and contrast tables
are named `pointwise_unadjusted_confidence_lower` and
`pointwise_unadjusted_confidence_upper`, with standardized equivalents in the
contrast table. These are pointwise 95% intervals; Holm adjustment applies
only to the corresponding p-value families.

`analysis-result.json` is checked against the required closed top-level
contract documented by the analysis schema and uses `complete`,
`not_confirmatory`, or `not_estimable` independently of its fixed
`claim_status = not_claimed`. No fitted model is checked in. See
[Confirmatory mixed-effects analysis](mixed-effects-analysis.md).

### Gate 6 cross-run review

`gate6-review build` writes a separate atomic artifact outside every paired
source run:

```text
declaration.json
evidence/pairs.jsonl
metrics/gate-6.json
review.json
manifest.json
SHA256SUMS
```

The declaration binds at least two sensitivity-to-Experiment-A pairs by run ID
and checksum-manifest digest and records explicit researcher-controlled
family/source identities. Pair rows retain exact provider/requested/returned
model evidence, recomputed sensitivity clauses, and recomputed held-out
paraphrase transfer. `metrics/gate-6.json` contains exactly six tri-state
criteria and always retains `claim_status = "not_claimed"`.

Portable verification checks the retained artifact; `--reverify-sources`
additionally reopens every declared run and reproduces the pair evidence.
Publication uses a sibling exclusive lock, same-parent staging, source
reverification, and an atomic rename. See
[Gate 6 cross-run review](gate6-cross-run-review.md).

### Decoder, Gate 4, human, and correction-debt outputs

`decoder-study validate` prints a `DecoderImportAudit` with request/judgment
counts, coverage, per-request instance/family/source-descriptor counts, missing
request IDs, and the independence caveat. `decoder-study analyze` can write one
JSON object containing:

```text
evaluation_splits = ["test"]
calibration                       # per-family, development only
family_metrics                    # raw/calibrated Brier/NLL/accuracy/ECE/bins
agreement                         # cross-family argmax and total variation
source_design_audit
interpretation_boundary
```

Truth labels are inputs to the researcher-side analysis and must never be
copied into decoder-visible outputs.

The distinct-decoder live collector writes a separate evidence directory:

```text
collection-plan.json
transport-attempts.jsonl
provider-audit.jsonl
judgments.jsonl
execution-manifest.json
```

The native-action live collector writes its own separate evidence directory:

```text
requests.jsonl
collection-plan.json
transport-attempts.jsonl
provider-audit.jsonl
native-actions.jsonl
execution-manifest.json
```

For both collectors, the durable `started` transport-attempt journal entry is
flushed before the physical request. A validated, accepted provider-audit row
is then flushed before its reusable `judgments.jsonl` or
`native-actions.jsonl` row. These collection directories remain outside the
immutable source run; they are later supplied by digest to the Gate 4 import.

`gate-review import-native` validates a completed, checksum-valid Experiment B
run without modifying it. The selected automated path requires the exact
retained decoder request/truth packet, a complete official Anthropic/Gemini
five-file collection, a hash-bound responsible-researcher source review, and a
complete official OpenAI six-file native-action collection. The separate
output directory contains:

```text
gate-review.json
manifest.json
SHA256SUMS
```

The review includes source-run and external-input digests, decoder
calibration/test analysis, source-design and action-binding audits, per-
trajectory action scores, and recomputed Gate 4 criteria. Use
`gate-review verify REVIEW_DIR` to check the file and content bindings.
`claim_status` is always `not_claimed`.

The output is published under an exclusive sibling lock from a durable
same-parent stage. Before the atomic rename, import re-verifies the source run,
direct input snapshots, complete locked collection inventories and bytes, and
the staged artifact. A failed write, changed input, failed self-check, or raced
destination leaves no partial review directory; verification rejects
symlinked, missing, or unexpected entries.

The directory deliberately does not duplicate its evidence inputs. Retain the
request, truth-label, and source-review files plus the five decoder and six
native collection evidence files so all recorded digests can be checked and
the import can be recomputed. The positional judgment path is the exact
collection `judgments.jsonl`. Protect the truth-label file as researcher-only;
do not place it in decoder-visible material. Review verification checks the
immutable review files, while recomputation also requires the verified source
run and all retained inputs. The explicit
`--allow-reviewed-generic-decoders` mode omits the four decoder collection
sidecars and makes no official-provider provenance assertion.

`human-study generate` writes participant items, a researcher codebook, order
manifest, response schema, packet hash manifest, and an ethics warning.
`human-study analyze` can write condition summaries, the observed ordering,
paired participant-bootstrap contrasts, and import/exclusion counts. These
outputs do not contain or imply an ethics determination.

`human-study evidence-from-experiment-a` atomically writes strict
`human-model-evidence` JSONL outside the verified source run. Rows are bound to
the source metric digest and contain no synthesized volunteered condition.
`human-study compare` atomically writes one H8 JSON object containing input
digests, eligibility audit, source/mechanism contrasts, pair-complete cluster
counts, bootstrap bounds, explicit incomplete/null states, and
`claim_status = "not_claimed"`.

`correction-debt run OUTPUT.json --stage-gate-authorized` writes one standalone
protocol object with the protocol digest, adapter ID, every paired arm and
turn-level snapshot, pair-level debt rows, stage summaries, and
`claim_status = "not_claimed"`. The default adapter is identified as a
diagnostic reference. This standalone JSON is not automatically a verified run
directory; a paper workflow must retain and checksum it alongside its
authorization/provenance record.

## LLM exchange files

A successful run with an `llm_*` updater writes:

```text
llm/input-manifest.json
llm/responses.jsonl
llm/exchange-manifest.json
llm/requests.jsonl  # only when retain_prompts = true
models/llm-calibration.json
llm/development-raw-responses.jsonl       # temperature mode
llm/development-requests.jsonl            # temperature + retain_prompts
metrics/llm-development-calibration.jsonl # temperature mode
llm/test-raw-responses.jsonl              # temperature mode
llm/provider-audit.jsonl    # live mode only
llm/transport-attempts.jsonl # live mode only
llm/provider-manifest.json  # live mode only
```

In replay mode, `input-manifest.json` fingerprints the complete configured
response file before execution: configured path, byte SHA-256, parsed response
count, and model IDs. In live mode it records the declared provider, model role
and resolved model, reasoning effort, endpoint, and hard request/token budgets,
with `credential_retained = false`.

The exchange files describe requests and responses actually consumed by the
updaters. The exchange manifest lists request IDs, updater views, prompt hashes,
model IDs, execution mode, and probability-calibration mode.

With per-updater temperature calibration, development requests/responses are
collected from disjoint development users, one calibrator is fitted for each
LLM updater view, and its transformation is locked before test execution.
`llm/test-raw-responses.jsonl` preserves the pre-transformation test beliefs;
`llm/responses.jsonl` contains the active beliefs supplied downstream.
`models/llm-calibration.json` records `fitted_split = "development"` and
`test_labels_used = false`. With `llm.calibration = "none"`, the calibration
artifact explicitly records the ablation and raw/active providers coincide.

Live mode additionally retains final provider audits, every used physical
transport-attempt event, and a credential-free provider manifest. The manifest
records `provider_audit_sha256`, `transport_attempts_sha256`, their portable
paths, event/attempt counts, physical-request accounting, and no credentials.
External recovery journals live outside the run directory so a failed adaptive
call sequence can resume without rebilling already bound responses; their
filesystem paths are stripped from the completed run manifest.

The external adaptive journal is
`.llm-journals/<run-id>[/openrouter]/<role>/transport-attempts.jsonl`.
Static `llm execute-*` commands derive a sibling name from the requested audit
path, such as `provider-audit-transport-attempts.jsonl`, and report it as
`attempts_path`. A `started` event is fsynced before each HTTP dispatch and a
`settled` event afterward. Final settlements embed their accepted/rejected
audit, allowing audit/replay repair without a second call. Unresolved starts or
settled nonfinal sequences require manual billing review before retry.

Both provider budgets count physical HTTP attempts. Failed, invalid, HTTP, and
ambiguous attempts charge the full conservative reservation; valid final
responses charge reported usage within that reservation. Keyless plans judge
`within_declared_budget` against the all-retries maximum rather than only the
first attempt per logical request.

An accepted audit records a returned model equal to the configured model or its
dated snapshot. A completed response with a missing/different model label is
retained as `acceptance_status = "rejected_model_mismatch"` and is not copied
into `llm/responses.jsonl`.

## Summary and gate records

`metrics/summary.json` is experiment-specific and always includes
`scientific_claim_status = "not_claimed"`.

`metrics/gate-report.json` contains all six gate IDs. A gate not evaluated by
the current run has a `not-evaluated` criterion and computed status
`incomplete`. Even a gate with `computed_status =
"meets_computational_checks"` retains `claim_status = "not_claimed"`.

Experiment A now computes held-out paraphrase transfer when every required
domain/mechanism/updater case pair exists. A structured-only run leaves that
criterion incomplete rather than fabricating a full-context LLM comparison.

## Checksums

With `[artifacts].checksum_manifest = true`, finalization writes:

```text
SHA256SUMS
```

It lists every file present under the run except `SHA256SUMS` itself, using
relative paths and SHA-256 digests.

Verify with:

```bash
PYTHONPATH=src python -m cape_loop verify RUN_DIR
```

The verifier detects:

- a missing checksum file;
- malformed checksum lines;
- invalid SHA-256 strings;
- absolute, escaping, unsafe, or duplicate checksum paths;
- listed files that are missing;
- digest mismatches; and
- files present in the run but absent from the checksum manifest.

It also requires:

- parseable `manifest.json` with `status = "complete"`;
- a manifest run ID equal to the directory name;
- parseable, current-schema `config.resolved.json`;
- a manifest config digest equal to the resolved configuration digest; and
- a retained TOML that reparses to that digest, or a hash-bound programmatic
  config origin; and
- `metrics/summary.json`.

It does not:

- validate every event/metric record schema;
- compare the manifest source digest with the current source tree;
- prove experiment completeness;
- decide a gate; or
- authenticate external model identity.

If checksum generation is disabled, CLI verification fails with
`missing SHA256SUMS`.

## Deterministic frozen archive

Freeze only a completed run that already passes the verifier:

```bash
PYTHONPATH=src python -m cape_loop artifact freeze \
  RUN_DIR artifacts/paper-run.tar
```

This creates:

```text
artifacts/paper-run.tar
artifacts/paper-run.tar.manifest.json
```

The uncompressed tar normalizes every member to a regular file with mode
`0644`, timestamp `0`, UID/GID `0`, and empty owner/group names. Member paths
are rooted at the source run-directory name. The sidecar records the archive
SHA-256, source run ID, source manifest/checksum digests, resolved-config
digest, config-origin declaration, and file count. Destinations must be absent.

Verify both files with:

```bash
PYTHONPATH=src python -m cape_loop artifact verify \
  artifacts/paper-run.tar
```

Frozen verification checks the sidecar binding, archive digest, unique safe
paths, regular-file inventory, member count, and normalized tar metadata. It
also reparses an archived `config.source.toml` and requires its resolved digest
to match the sidecar. Without TOML, the sidecar must retain the complete
programmatic-origin binding. Verification does not rerun the study or turn a
smoke artifact into paper evidence.

## Failed and mutable output

When an exception occurs after run creation, the runner attempts to retain:

```text
manifest.json  # status = "failed"
failure.json
SHA256SUMS     # when enabled
```

Not every intended experiment file will exist. A failed directory is evidence
of a failed execution, not a completed cell.

Do not hand-edit run files. Any edit invalidates the checksum entry. Local
`runs/` are ignored by Git; deliberately released evidence belongs under
`artifacts/` with an artifact-specific README and provenance.
