# Detailed reproducibility guide

CAPE-Loop combines deterministic simulation with potentially nondeterministic
external models and human judgments. Reproducibility therefore means preserving
the appropriate canonical input at each boundary.

## Reproducibility levels

### Core deterministic replay

Given the same source, resolved configuration, semantic IDs, and Python behavior,
the simulator and reference inference should reproduce their records without
network access.

### External-model replay

The same provider call may not be reproducible. The validated response JSONL is
the canonical replay input. Reanalysis should not call the provider again.

### Human-study reanalysis

Participant responses are observations, not regenerable computations. Reanalysis
requires an appropriately consented, de-identified response dataset, codebook,
randomization manifest, and analysis configuration.

### Paper artifact reproduction

Tables and figures are deterministic transforms of retained complete runs.
Their manifest maps each output to source run IDs and checksums.

## Semantic-keyed randomness

Mutable random-number streams make paired branches fragile: an extra draw in one
policy changes every subsequent response. CAPE-Loop derives a random value from
a root seed and stable semantic parts.

Recommended namespaces include:

```text
population / user / preference
population / user / susceptibility
experiment / scenario / option-generation
trajectory / turn / option / choice-noise
trajectory / turn / policy
bootstrap / replicate / trajectory-selection
human-study / assignment / item-order
```

The same key must mean the same random variable. New purposes receive new
namespaces rather than reusing an existing key.

Common random numbers pair option-level noise across counterfactual branches
when an option has the same semantic identity. They do not force two scientifically
distinct natural-response draws to be identical when the estimand requires
independent sampling.

## Configuration identity

`AppConfig` is serialized to canonical JSON with sorted keys and compact
separators. Its SHA-256 digest determines the run suffix. This identity includes
resolved defaults.

An artifact should retain:

- the original TOML;
- `config.resolved.json`;
- configuration digest;
- schema version;
- source revision or source archive checksum.

The command prints the effective run directory when `--output-root` is used,
but the v1 manifest does not retain that non-scientific parent-directory
override as a separate field.

Changing only an output parent should not change scientific content. Changing a
seed, component, parameter, threshold, or retained-information behavior requires
a new configuration identity.

## Environment identity

The core records Python implementation/version, platform, and executable. For a
paper release also record:

- operating-system/container image digest;
- source revision;
- relevant environment variables excluding secrets;
- locale/timezone where formatting could matter;
- external harness dependency lock;
- provider execution metadata;
- analysis tool versions if non-core analysis is used.

The optional [mixed-effects analysis](mixed-effects-analysis.md) pins R and its
runtime dependency graph, records `sessionInfo()`, independently verifies every
source-run checksum, recomputes the configuration digest from the retained
canonical resolved-config payload, digests its normalized analysis rows, and
writes a separate `SHA256SUMS`. Pooled repeats must have the same source digest
and design, including an identical Experiment B horizon. B rows are
reconstructed for retained turns `1, ..., T`, and the final marginal Brier
error is checked against the retained terminal error. Do not copy analysis
outputs into or otherwise mutate the verified source run.

A container can aid portability but does not replace source and configuration
identity.

## Splits and leakage

Retain immutable identifiers for train, development, and test:

- complete latent user profiles;
- susceptibility types;
- options/templates;
- scenario families;
- paraphrases;
- terminal diagnostic items.

The runner enforces the latent-profile and susceptibility-group split and
consumes the option, dialogue, scenario, and paraphrase assignments in its
generators. Feature-matched atlas/beacon/cedar domain variants supply disjoint
train/development/test surface IDs. Each run writes
`metrics/split-leakage-audit.json`; any concrete overlap or manifest mismatch
aborts execution. Experiment A additionally invokes the content-addressed
paraphrase suite's test-leakage guard. See
[Data splits](data-splits.md).

When `retain_events` is enabled, fitted-likelihood training examples are
retained in `events/fitted-model-training.jsonl`; model bundles retain their
example count and training seed. Calibration records its development-only
transformation and example count. The generated user groups are disjoint from
test groups.

Do not tune simulator parameters, prompts, update thresholds, gate thresholds,
or system selection on the final test set and continue calling it held out.

## Static versus endogenous histories

Fixed-history evaluation serializes one canonical history and replays its event
IDs exactly to all updaters. Similar seeds are not enough.

Endogenous evaluation necessarily produces different histories because each
updater’s profile affects the policy. Pairing is retained through user-twin,
scenario, and semantic-noise identifiers.

The common terminal battery is a separate frozen manifest shared by both
regimes.

Every B/C terminal profile score retains top-label ECE and fixed reliability
bins with `preference_attribute_forecast` as the sample unit. The pooled
calibration artifacts preserve split/regime/updater/decoder grouping and state
that trajectory/user is the dependence unit; do not analyze the three
attributes per trajectory as independent users.

## Fitted models and calibration

Data-only fitted artifacts should retain:

- model kind/version;
- feature declaration identifying aware or unaware;
- coefficients/parameters;
- optimizer settings and any available convergence diagnostics;
- fit records or a deterministic generation identity and split;
- code/config digest.

Calibration creates a new belief layer and retains raw predictions. A released
summary contains both or states why one is absent.

## LLM execution

Follow [LLM exchange](llm-exchange.md). At minimum retain:

- exact request JSONL;
- prompt hash;
- validated response JSONL;
- model label and execution time;
- external harness identity;
- decoding parameters;
- missing, rejected, and retried request IDs.

`llm validate` checks response structure, not provider authenticity or request
completeness. The replay/experiment layer performs expected-ID and prompt-hash
matching. Static corpora can be planned and executed through the explicit
direct OpenAI or OpenRouter CLI; adaptive A–C execution uses the selected
journaled provider only with `--execute-live`. Runs fingerprint the supplied
response corpus or live model declaration in `llm/input-manifest.json`, and
`--allow-existing` requires that identity to match.

For OpenRouter, freeze and retain all of the following as one protocol identity:

- the exact canonical `author/model` slug, never an alias or auto router;
- the optional upstream provider/endpoint slug;
- `allow_fallbacks`, `require_parameters`, `data_collection`, and `zdr`;
- the app-attribution values, if any;
- the disabled response-cache and enabled router-metadata declarations;
- request/body/prompt hashes and conservative budget ceilings;
- requested/returned model plus selected upstream provider/model;
- routing strategy, attempt, and complete additive router metadata;
- body response ID, response-header generation ID, cache status, raw usage,
  timings, and acceptance status; and
- the redacted raw response and provider-neutral replay response.

Changing the `model` line in `configs/openrouter_gemini.toml` switches the
model, but it changes the resolved-config digest and creates a different run
identity. If the replacement is not served by the configured upstream route,
also change or clear that pin. Changing only a human-facing model description
is not sufficient. Endpoint support and routing policy are external state, so
preserve the observed route metadata rather than inferring it later from the
requested slug.

OpenRouter response caching is disabled in every prepared request. A reported
cache hit, fallback when disabled, model mismatch, non-direct strategy, or
material router pipeline is rejected before replay. Accepted audit rows are
written before response rows, and resumed audits are revalidated. The default
adaptive recovery path is
`.llm-journals/<run-id>/openrouter/<model-role>/`; successful runs copy their
used audit and provider manifest into `llm/`.

The OpenRouter audit deliberately records
`first_party_origin_claimed = false`. A gateway-reported upstream identity is
not equivalent to a direct request to that provider, and multiple models or
upstreams behind the same gateway do not establish statistically independent
errors. OpenRouter decoder artifacts must remain in the reviewed-generic
evidence branch. Strict Gate 4 continues to require the direct first-party
Anthropic/Gemini decoder collections and direct OpenAI native-action
collection.

For the declared primary/replication pair, use `llm evaluation-suite`. Its
default mode writes a credential-free combined plan; `--execute-live` is
required to run either role. The index binds both source configs and resolved
configs, retains separate run/journal paths and per-role ceilings, and labels
Terra as GPT-5.6 model-variant/tier replication rather than distinct-family
robustness.

## Human-study materials

Retain:

- generated participant items;
- separately protected researcher codebook;
- seed and assignment-order rule;
- survey rendering/version;
- protocol and consent versions;
- response validation rules;
- anonymization/transformation log;
- exclusions fixed before condition unblinding where possible.

The public release may need to omit the codebook or raw text until data
collection ends. State access conditions rather than publishing sensitive
material by default.

## Checksums

Run finalization writes `SHA256SUMS`. Verify before analysis and again after
transfer:

```bash
PYTHONPATH=src python -m cape_loop verify RUN_DIR
```

Verification checks every retained run file against `SHA256SUMS`, rejects
unlisted files, and rejects paths that escape the run directory. It also
requires a completed manifest, a matching directory/run ID, a resolved
configuration whose digest matches the manifest, and a summary record. It does
not establish scientific completeness. A release audit separately checks
expected factorial cells, schema validity, split integrity, and gate inputs.

## Reproducing a run

```bash
PYTHONPATH=src python -m cape_loop doctor
PYTHONPATH=src python -m cape_loop config validate configs/smoke.toml
PYTHONPATH=src python -m cape_loop run configs/smoke.toml \
  --output-root /tmp/cape-loop-runs
PYTHONPATH=src python -m cape_loop verify \
  /tmp/cape-loop-runs/<printed-run-id>
```

Do not predict the digest-derived run ID manually; use the path printed by
`run`.

For Experiment B or C, replace the config with:

```text
configs/closed_loop.toml
configs/evaluation.toml
configs/sensitivity.toml
```

Those configurations are larger than the smoke suite.

## Release audit checklist

### Scientific invariants

- [ ] Latent preference is fixed within every trajectory.
- [ ] Only simulator/evaluator code sees latent user state.
- [ ] Context and policy provenance are separate.
- [ ] Presentation utility never enters intrinsic welfare.
- [ ] Matched anchors retain identity and meet probability threshold.
- [ ] Same-history shadows observe identical events.
- [ ] Static histories are byte/record identical across updaters.
- [ ] Terminal diagnostics are exogenous and shared.
- [ ] All five self-confirmation conditions are present.

### Data and analysis

- [ ] `metrics/split-leakage-audit.json` passes for the executed concrete
      split/template families.
- [ ] Fitting uses training records only.
- [ ] Calibration uses development records only.
- [ ] Raw and calibrated results are separate.
- [ ] Trajectory/user is the resampling unit.
- [ ] Missing/failed cells and exclusions are retained.
- [ ] Bootstrap seeds and tie rules are recorded.

### Artifact integrity

- [ ] Source TOML and resolved JSON are present.
- [ ] Source/environment identity is present.
- [ ] External responses are replayable or limitations are stated.
- [ ] Human data have appropriate release authorization.
- [ ] `SHA256SUMS` verifies after final transfer.
- [ ] Tables/figures map to source run IDs.
- [ ] Smoke/pilot/paper status is explicit.
- [ ] No placeholder value is presented as a result.

The shorter release checklist is [REPRODUCIBILITY.md](../REPRODUCIBILITY.md).
