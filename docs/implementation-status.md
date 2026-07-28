# Implementation status

CAPE-Loop is an alpha research implementation. This page is the authoritative
boundary between executable repository software and evidence that still has to
be collected. No checked-in artifact establishes a paper hypothesis or claims
that a scientific gate has passed.

**Last reconciled:** 2026-07-28 against the source tree, command-line interface,
public configurations, generated schemas, tests, CI workflow, and available
machine-local diagnostics.

## Bottom line

| Question | Current answer |
| --- | --- |
| Can the repository generate its synthetic data and run offline? | **Yes.** Experiments A–C, sensitivity analysis, verification, and diagnostic gate reports are executable without an API key. |
| Can it call real models? | **Yes, with explicit authorization.** Direct OpenAI and OpenRouter adaptive execution, replay, selected decoder collection, and native-action collection are implemented and budget checked. |
| Is it ready for a bounded live pilot? | **Yes at the software level.** The public pilots fit the approved ceilings and require `--execute-live`; the operator must still review cost, model identity, routing, and evidence governance. |
| Is the paper study complete? | **No.** No eligible provider corpus, Gate 4 collection, human response dataset, confirmatory study fit, or paper result is checked in. |
| Has any scientific gate passed? | **No claim is made.** Generated gate artifacts always retain `claim_status = "not_claimed"`. |

## Status language

| Status | Meaning |
| --- | --- |
| **Implemented** | Executable in the current tree and covered by an offline automated test. |
| **Implemented diagnostic** | Executable and tested, but deliberately unable to establish a paper result by itself. |
| **Implemented optional analysis** | Executable and tested in a separate optional dependency environment; not required by the standard-library Python core. |
| **Provider-capable** | Can construct, budget, send, audit, and replay a live request, but has no eligible checked-in response corpus. |
| **External-evidence-dependent** | Protocol and validation code exist, but completion requires credentials, reviewed external sources, participants, or another outside input. |
| **Deferred** | Intentionally outside the present execution boundary. |

An implemented component is not an empirical finding. A transport smoke proves
connectivity, a verified run proves artifact integrity, and a computational
gate result proves only that the coded conditions evaluated as recorded.

## Verified repository snapshot

The current tree passes:

- `make check`, including compilation, runtime diagnosis, all public TOML
  validation, the static mixed-effects contract, and 373 offline tests across
  38 test modules;
- byte-for-byte parity for 31 generated JSON Schemas;
- Ruff, `git diff --check`, and resolution of all relative Markdown file
  targets; and
- validation of all 18 public configurations: one smoke, eight offline/source,
  and nine bounded live presets.

The standard CI matrix runs the core on Python 3.11, 3.12, 3.13, and 3.14. A
separate CI job using commit-SHA-pinned actions restores R 4.6.1 with locked
`lme4` 2.0-6, generates a synthetic verified Experiment A fixture, executes
the optional mixed-effects harness, and verifies its no-claim artifact.

These checks use no live provider. The R CI fit uses generated test data, not a
study corpus.

## Repository evidence boundary

The tracked repository contains source code, tests, schemas, configurations,
model-role declarations, and documentation. It does not contain:

- an API key or authorization header;
- an eligible OpenAI, OpenRouter, Anthropic, or Gemini response corpus;
- a completed external-decoder or native-action collection;
- participant responses;
- a confirmatory fit over an eligible study run;
- a paper-frozen empirical run archive; or
- completed author, venue, repository, DOI, or archive metadata.

Generated datasets and run artifacts are written below the configured output
root, normally `runs/`, and are intentionally ignored until a reviewed release
is assembled. Canonical scientific records are JSON and JSON Lines; CSV files
are derived analysis projections. See [Data model](data-model.md) for the
record graph and [Reproducibility](../REPRODUCIBILITY.md) for admission rules.

## Implemented scientific core

| Component | Status | Current boundary |
| --- | --- | --- |
| Runtime and configuration | Implemented | Python 3.11+ standard-library core; strict schema-versioned TOML; unknown and experiment-incompatible fields fail validation. |
| Synthetic population and domains | Implemented | Fixed latent users, heterogeneous presentation susceptibility, and travel and writing domains with three signed attributes. |
| Dataset surfaces and splits | Implemented | Generator-bound train/development/test option, dialogue, scenario, paraphrase, and terminal families with content digests and overlap audits. |
| Response models | Implemented | Random-utility and rule-based families; intrinsic utility remains separate from rank, default, and suggestion effects. |
| Exact inference | Implemented | Full finite preference×susceptibility posterior with an action-aware likelihood; public beliefs use the preference marginal. |
| Fitted references | Implemented | Parameter-count-matched aware and unaware likelihoods trained on the same records, with raw and development-calibrated bundles retained separately. |
| Randomness | Implemented | Semantic-keyed SHA-256 streams and option-keyed Gumbels provide deterministic common-random-number branches. |
| Structured and native state | Implemented | Structured beliefs plus inspectable episodic, semantic, persona, and provenance-linked native memory with content-addressed state identity. |
| Information boundaries | Implemented | Updaters receive only their declared response-only, full-context, or provenance-aware view; latent truth is restricted to the simulator and evaluator. |
| Run integrity | Implemented | Resolved configuration identity, source digest, manifests, streamed checksums, symlink rejection, run verification, and deterministic freeze/verify tooling. |

The durable data flow is:

```text
latent synthetic user
  -> policy provenance and separately retained visible context
  -> simulated, replayed, or live response
  -> declared updater information view
  -> structured profile or native-memory update
  -> evaluator-only metrics and no-claim gate records
```

## Experiment and analysis surface

| Component | Status | Current boundary |
| --- | --- | --- |
| Experiment A | Implemented | Matched provenance anchors, controlled and sampled modes, prior-strength strata, exact/fitted/LLM updater comparisons, H1/H2/H7 estimands, held-out paraphrases, clustered inference, Holm adjustment, and pilot-power diagnostics. |
| A six-control exchange | Implemented tooling / external-evidence-dependent | Content-bound positive and negative controls, reference/no-update diagnostics, provider request exchange, immutable review, and scoring exist; no external control corpus is checked in. |
| H7 volunteered-statement review | Implemented tooling / external-evidence-dependent | Complete planning, provider-audit binding, accepted-update conversion, and source-safe recomputation exist; no accepted provider evidence is checked in. |
| Experiment B | Implemented | Crossed initial profiles, policies, and updaters; endogenous trajectories; same-history action-aware shadows; terminal error, learning-confidence gain, decomposition, and pilot-power analysis. |
| Experiment C | Implemented | Fixed balanced, fixed biased, and endogenous regimes; common terminal battery; complete-user paired ranking, reversal, top-tier, and evaluation-selection-regret analysis. |
| Experiment C multi-seed review | Implemented diagnostic | Verifies 2–32 completed matched-source runs and records agreement and disagreements without selecting a favorable seed. |
| Sensitivity runner | Implemented | Cartesian and baseline-first one-at-a-time construction, phase classification, and observed-grid boundaries; public presets use the smaller OAT design and explicitly deny interaction estimability. |
| Calibration | Implemented | Development-only temperature calibration for fitted and LLM probability outputs, raw/active retention, test reliability diagnostics, and an explicit uncalibrated ablation. |
| Common terminal evaluation | Implemented | Held-out terminal-v2 battery shared by B/C structured and native systems; native representation diagnostics use two fixed blinded projections. |
| External native-state decoding | Implemented tooling / external-evidence-dependent | Blinded packets, source audit, development-only per-family calibration, held-out scoring, and agreement analysis exist; deterministic local projections are not independent external decoders. |
| Experiment C external rescore | Implemented tooling / external-evidence-dependent | Immutable import, calibration, native-only score replacement, full reranking, ESR, Gate 5 recomputation, and verification exist; no judgment corpus is checked in. |
| Confirmatory mixed effects | Implemented optional analysis | Locked R/`lme4` formulas, maximal random-effects structure, planned contrasts, diagnostics, and checksum-bound outputs exist; only a synthetic CI fixture has been fitted. |
| Human H8 workflow | Implemented tooling / external-evidence-dependent | Vignettes, assignment, blinding/codebook, strict import, exclusions, model-evidence conversion, and paired comparison exist; recruitment and human data are deferred. |
| Correction debt / H9 | Implemented diagnostic and deferred | The paired protocol and transparent reference adapter are tested; a real native/LLM adapter and empirical H9 execution remain stage-gated and outside the minimum paper. |

Detailed estimands and output records are documented in
[Experiments](experiments.md), [Metrics](metrics.md), and
[Data model](data-model.md).

## Provider and external-evidence surface

| Component | Status | Current boundary |
| --- | --- | --- |
| Provider-neutral replay | Implemented | Exact request IDs, prompt hashes, schema validation, complete coverage, and input-corpus fingerprints. |
| Direct OpenAI | Provider-capable | Responses API execution for profile writers and native actions with explicit authorization, origin locks, budgets, durable attempt journals, audit-first recovery, and resumability. |
| OpenRouter | Provider-capable | Chat Completions execution with exact canonical slugs, compatible strict schemas, routing/privacy controls, cache disablement, budget enforcement, durable journals, and multi-model collection. |
| Optional direct Anthropic and Gemini | Provider-capable | Separate first-party decoder collectors and validators remain available for origin replication; they use separate credentials and artifacts. |
| Adaptive live preflight | Implemented | Counts calibration, A paraphrases, experiment turns, sensitivity points, retries, and maximum output allocation before credential or artifact access. |
| Selected Gate 4 collection | External-evidence-dependent | Supports the complete Claude/Gemini OpenRouter decoder collection, OpenAI native-action collection, responsible-researcher source review, immutable import, and independent verification. |
| Alternate Gate 4 admission | External-evidence-dependent | Optional direct-first-party and explicitly authorized reviewed-generic decoder modes are implemented and kept provenance-distinct. |
| Gate 6 cross-run review | External-evidence-dependent | Revalidates declared live-model sensitivity/A pairs, provider and model identity, matched grids, paraphrase transfer, and all coded clauses; qualifying source runs are absent. |

OpenRouter validates router-reported provider/model display metadata; that
metadata is not an exact route-slug attestation. For the selected shared-gateway
decoder workflow, the collection plan, execution manifest, and validation
summary explicitly deny first-party origin, distinct transport origins, and
statistical independence.

## Public configurations and live bounds

The public matrix is intentionally minimal:

```text
configs/smoke.toml   1 quick offline check
configs/offline/     8 deterministic experiment/source presets
configs/live/        9 bounded provider presets
```

Local variants belong under ignored `configs/local/`. The R-only synthetic
fixture belongs to its analysis component at
`analysis/confirmatory-mixed-effects/fixtures/confirmatory_ci.toml`.

All three public Experiment A live presets use prior strengths `0.0` and `0.7`.
They are transport and estimability pilots, not powered confirmatory designs.

| Adaptive live pilot | Experiment calls | Calibration | A paraphrases | Physical-attempt bound | Maximum output allocation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Experiment A | 768 | 48 | 32 | 848 | 1,736,704 |
| Experiment B | 768 | 96 | 0 | 864 | 1,769,472 |
| Experiment C | 768 | 48 | 0 | 816 | 1,671,168 |
| Gate 6 OAT | 576 | 0 | 0 | 576 | 1,179,648 |

Each uses zero automatic retries and remains below the approved per-provider
ceiling of 900 physical attempts and 6,000,000 conservatively allocated tokens.

The offline Gate 4 source produces 640 decoder requests per model and 80
native-action requests. The offline Experiment C rescore source produces 360
decoder requests per model. These generation runs do not call a model.

## Current model roles

| Role | Current declaration | Transport |
| --- | --- | --- |
| Primary profile writer and Gate 4 native actions | `gpt-5.6-sol`, medium effort | Direct OpenAI |
| Profile-writer replication | `gpt-5.6-terra`, medium effort | Direct OpenAI |
| Generic decoder/pilot role | `gpt-5.6-luna`, low effort | Direct OpenAI |
| OpenRouter adaptive pilot | `google/gemini-3.6-flash`, minimal effort | OpenRouter |
| Gate 4 blinded decoder family 1 | `anthropic/claude-sonnet-5`, low effort | OpenRouter selected; direct optional |
| Gate 4 blinded decoder family 2 | `google/gemini-3.6-flash`, minimal effort | OpenRouter selected; direct optional |

Sol versus Terra is a model-variant replication, not distinct-family
robustness. Claude and Gemini are declared distinct model families, but their
selected requests share OpenRouter and therefore do not establish independent
transport or errors.

## Local live validation snapshot

Machine-local diagnostics have exercised direct OpenAI and one schema-valid
request to each selected OpenRouter decoder route. Some supporting evidence was
temporary and none is checked in, so these calls are not independently
verifiable release evidence and do not enter a dataset, hypothesis, or gate.

An earlier Claude diagnostic exposed a route-specific rejection of numeric JSON
Schema bounds. The compatible wire-schema subset and strict local finite-range
and vector-normalization checks are now covered by the offline suite.

Additional ignored diagnostics include verified Gate 4 and Experiment C source
packets, a three-seed Experiment C review, and a complete 19-point simulator
OAT run. They all retain `not_claimed` and are not distributed evidence. The
complete Gate 4 decoder/native-action collections and Experiment C external
rescore collection have not been executed.

## Gate state

Every completed A, B, C, or sensitivity experiment run writes
`metrics/gate-report.json` with six entries. Failed or interrupted attempts may
stop before that artifact exists. Gates outside a run's scope are explicitly
incomplete.

| Gate | Implemented computation and current evidence boundary |
| --- | --- |
| Gate 1 — learnable provenance gap | Experiment A evaluates fitted-aware versus fitted-unaware error separately in both domains, a material `llm_full_context`–aware gap in both domains for at least two non-balanced mechanisms, and held-out paraphrase transfer. Missing any required updater/domain/mechanism pair leaves it incomplete; no qualifying live evidence is checked in. |
| Gate 2 — nontrivial soft self-confirmation | Experiment B evaluates `llm_full_context` under eligible soft mechanisms with counter-profile options and requires at least eight user clusters plus lower 95% clustered interval bounds above zero. No passage is claimed. |
| Gate 3 — attribution beyond evidence selection | The same eligible `llm_full_context` trajectories must perform worse than their same-history action-aware shadows, with an adequate clustered interval above zero. No passage is claimed. |
| Gate 4 — native-system validity | Requires external blinded-decoder judgments and genuine hash-bound native end-to-end actions. The selected path requires complete Claude/Gemini OpenRouter evidence, complete OpenAI native actions, and responsible source review; optional direct-first-party and explicitly reviewed-generic modes also exist. No complete live collection has been imported. |
| Gate 5 — evaluation implication | Experiment C uses joint-paired complete-user differences and difference-of-differences for credible reversals, or interval-supported development tiers plus a conservative held-out ESR envelope. Descriptive correlations or rank bands cannot pass it alone. |
| Gate 6 — robustness | Sensitivity artifacts check declared-grid completion and phase boundaries; a separate immutable review binds qualifying live sensitivity/A runs and recomputes model-family and paraphrase clauses. No qualifying paper-scale review is checked in. |

`computed_status = "meets_computational_checks"` means only that every coded
Boolean available to that artifact evaluated true. It never changes
`claim_status` from `"not_claimed"`.

## Work still required for paper evidence

- Freeze the preregistration, estimands, model identities, provider routes,
  budgets, and release policy.
- Execute and admit eligible live Experiment A–C and Gate 6 collections.
- Collect the complete Gate 4 Claude/Gemini judgments and OpenAI native actions,
  then complete the responsible-researcher source review.
- Run the external Experiment C rescore and the declared multi-seed review.
- Obtain the required ethics determination before the deferred human study,
  then collect and analyze eligible responses if that study proceeds.
- Fit the confirmatory R models on frozen eligible runs and review diagnostics;
  the CI fixture is not a substitute.
- Freeze paper-facing archives and populate results only from verified evidence.
- Add authorship, ORCIDs, official repository URL, venue status, and DOI/archive
  metadata when available.

## Verification and readiness

Run the universal offline check from the repository root:

```bash
make check
```

It does not restore R dependencies, execute a live provider, or create a paper
result. For a generated run, also use:

```bash
PYTHONPATH=src python -m cape_loop verify runs/<run-id>
```

The repository is currently:

| Use | Ready? |
| --- | --- |
| Understand and inspect the complete protocol | Yes |
| Generate synthetic datasets | Yes |
| Run and verify offline Experiments A–C and sensitivity | Yes |
| Plan live requests without reading credentials | Yes |
| Execute an explicitly authorized bounded live pilot | Yes, subject to operator review and provider availability |
| Reproduce checked-in empirical findings | No; none are checked in |
| Make paper claims or mark a scientific gate passed | No |

Use [Getting started](getting-started.md) for the first run,
[Configuration](configuration.md) for the public presets,
[Live execution](live-execution.md) for provider commands and recovery, and
[Architecture](architecture.md) for the component and trust-boundary map.
