# Implementation status

CAPE-Loop is an alpha research implementation. This page is the authoritative
boundary between executable repository software and evidence that still has to
be collected. No checked-in artifact establishes a paper hypothesis or claims
that a scientific gate has passed.

**Last reconciled:** 2026-07-29 against the source tree, command-line interface,
public configurations, generated schemas, tests, CI workflow, and available
machine-local diagnostics.

## Bottom line

| Question | Current answer |
| --- | --- |
| Can the repository generate its synthetic data and run offline? | **Yes.** Experiments A–C, sensitivity analysis, verification, and diagnostic gate reports are executable without an API key. |
| Can ordinary analysis avoid the multi-gigabyte audit logs? | **Yes for newly generated A–C runs.** The runner writes checksum-bound compact rows alongside the full records; verified historical runs can be projected without mutation. |
| Is the scenario catalog ready for paper evidence? | **No.** Catalog loading, validation, without-replacement trajectory scheduling, prospective simulator calibration, rendered review packets, split auditing, and run binding are implemented. All 48 scenarios remain provisional, zero are approved, and human reviews and independent semantic calibration are incomplete. Six test scenarios per cell satisfy the 16-turn cyclic and v2 block-balanced exploratory requirements; custom unconstrained adaptive targeting can still exhaust a cell and is reported separately. |
| Does the simulator produce natural conversations? | **Mechanically yes; scientifically not yet approved.** Math selects the option, then the frozen bank renders an assistant/user exchange. The automated punctuation/article scan is clean, but it cannot establish naturalness, neutrality, or semantic validity; independent surface/scientific reviews remain incomplete. |
| Can a researcher read conversations beside their metrics? | **Yes.** Each A/B/C/sensitivity run writes exhaustive deduplicated JSONL plus a deterministic diverse Markdown preview of at most 100 trace records by default, with exact complete conversation, turn, and outcome counts and readable metric guidance. Hybrid runs contain the natural exchange; non-surface fixtures are marked unavailable. |
| Can a researcher inspect one complete live example first? | **Yes, as diagnostics.** `demo one-scenario` shows one frozen scenario and one live update; `demo experiment-b-case` shows matched balanced/profile-conditioned branches using 3, 6, 9, or 12 turns, with one logical update per turn in each of two policy branches. Both write natural-language metrics plus complete request/audit files, and neither can support a paper claim. |
| Can it call real models? | **Yes, with explicit authorization.** Direct OpenAI and OpenRouter adaptive execution, replay, selected decoder collection, and native-action collection are implemented and budget checked. |
| Is it ready for a bounded live pilot? | **Partly.** The transport, replay, audit, and analysis software is ready for bounded pilots, but the current public Experiment B live presets use only three turns. The multi-model diagnostic showed that this is too short to exercise the declared later-action mechanism; redesign B before collecting its pilot. |
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
  validation, the static mixed-effects contract, and the full offline test
  suite;
- byte-for-byte parity for 34 generated JSON Schemas, including the exhaustive
  conversation-log row contract;
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
model-role declarations, documentation, one explicitly provisional scenario
catalog, and its frozen candidate conversation bank. It does not contain:

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

New A–C runs contain both full reconstruction records and narrow
`analysis/*.jsonl` projections. The compact rows are derived from the same
evaluated records, so they add no observations and cannot change an experiment's
sample size or evidence status.

A–C and sensitivity runs also contain a normalized trace under
`conversations/`. Its JSONL exhaustively stores logical conversations with
their evaluations without repeating shared dialogue per updater. Its Markdown
is a bounded reading preview, not an analysis sample or an additional dataset.
Runs without a configured surface retain nullable dialogue rather than
inventing one.

## Implemented scientific core

| Component | Status | Current boundary |
| --- | --- | --- |
| Runtime and configuration | Implemented | Python 3.11+ standard-library core; strict schema-versioned TOML; unknown and experiment-incompatible fields fail validation. |
| Synthetic population and domains | Implemented | Fixed latent users, heterogeneous presentation susceptibility, and travel and writing domains with three signed attributes. Official presets opt into orthogonal v2 theta and susceptibility splits/allocation: coordinate levels and pairs are balanced in each split, and incomplete allocation blocks keep marginal counts within one user. A cached outcome-blind joint block search reduces avoidable theta–susceptibility association at official horizons while preserving those guarantees. Legacy and single-v2 sequences remain reproducible. |
| Scenario catalog and selection | Implemented / review-dependent | Strict schema and SHA-256 loading, 48 family-disjoint stimuli, deterministic paired selection, and within-trajectory sampling without replacement until a cell is exhausted are implemented. Version 1.4 explicitly counterbalances restricted-peer nuisance attribute, direction, their joint combinations, and three neutral conversation-frame families. Each test domain×attribute cell has six families, supporting the 16-turn cyclic and block-balanced exploratory references without reuse; custom unconstrained adaptive targeting retains the stricter worst-case capacity rule. `scenarios audit` reports planned capacity, counterbalancing, complete finite-support probabilities, overlap/surface warnings, and separate engineering/recorded-review/paper readiness without consuming outcomes. All stimuli remain simulation-and-pilot-only: 48 provisional, 0 approved, no completed human reviews, and not paper-eligible. Current writing options are direct category descriptors rather than full excerpts, so the supported estimand is deliberately narrow. |
| Hybrid conversation surfaces | Implemented / review-dependent | All 48 visible bases are outcome-blind project standardizations over three source-neutral frames: 16 uses per frame and two uses per frame in every six-scenario test domain×target cell. Every template carries `project-standardized-neutral-frame-v1-unreviewed`. The checked-in OpenRouter log preserves 24 historical candidate-authoring calls but does not establish provider authorship of the current visible text. Code enforces the frozen neutral/treatment contract at load time, assigns A/B by visible position after ranking, inserts only fixed default/suggestion text, and fixes `I choose {selected_name}.` after the response model chooses. Observations retain the exact exchange and surface ID. The automated hygiene scan is clean, but frame balance is not semantic validation; independent human review remains pending, so the bank is not paper-eligible. |
| Scenario human calibration | Packet implemented / evidence import pending | `scenarios audit` now emits both a metadata-visible researcher workbook and an opaque dialogue-only surface packet. Scientific readiness includes an explicit unverified-evidence blocker, so manually changing catalog review strings cannot approve a stimulus. A version-bound response importer, aggregation verifier, and promotion command are not implemented yet; completed independent reviews and pretests therefore require this remaining tooling before paper freeze. |
| Dataset surfaces and splits | Implemented | Catalog-bound train/development/test option, dialogue, and scenario families plus separate paraphrase and terminal families, with content digests and overlap audits. |
| Response models | Implemented | Random-utility and rule-based families; intrinsic utility remains separate from rank, default, and suggestion effects. |
| Exact inference | Implemented | Full finite preference×susceptibility posterior with an action-aware likelihood; public beliefs use the preference marginal. |
| Fitted references | Implemented | Parameter-count-matched aware and unaware likelihoods trained on the same records, with raw and development-calibrated bundles retained separately. |
| Randomness | Implemented | Semantic-keyed SHA-256 streams and option-keyed Gumbels provide deterministic common-random-number branches. |
| Structured and native state | Implemented | Structured beliefs plus inspectable episodic, semantic, persona, and provenance-linked native memory with content-addressed state identity. |
| Information boundaries | Implemented | Evaluated LLMs receive semantic attribute meanings and readable options, not numeric features, the target index, or semantic catalog option IDs. Full/provenance views receive the rendered dialogue under per-position `presented_option_N` aliases; response-only remains an ablation. Latent truth is restricted to the simulator and evaluator. |
| Run integrity | Implemented | Resolved configuration identity, source digest, manifests, streamed checksums, symlink rejection, run verification, and deterministic freeze/verify tooling. |
| Compact analysis projections | Implemented | New A, B, and C runs automatically retain one updater×trial, trajectory-turn, or evaluation/ranking row respectively; summary metadata and the run checksum bind the projection. Historical completed runs remain immutable and use a separately verified derived directory. |
| Human-readable conversation traces | Implemented | A, B, C, and sensitivity runs retain exhaustive deduplicated JSONL with compact analysis metrics plus a deterministic diverse Markdown preview capped at 100 trace records by default. Summary metadata exposes both paths and exact complete record, turn, and outcome counts; runs without a natural surface mark it unavailable. |

The durable data flow is:

```text
latent synthetic user
  -> policy provenance and separately retained visible context
  -> mathematical option choice
  -> frozen natural assistant/user rendering
  -> declared updater information view
  -> structured profile or native-memory update
  -> evaluator-only metrics and no-claim gate records
```

## Experiment and analysis surface

| Component | Status | Current boundary |
| --- | --- | --- |
| Experiment A | Implemented | Matched provenance anchors, paired scenario and opposite physical-order assignment across anchor directions, controlled and sampled modes, prior-strength strata, exact/fitted/LLM updater comparisons, H1/H2/H7 estimands, held-out paraphrases, clustered inference, Holm adjustment, and pilot-power diagnostics. |
| A six-control exchange | Implemented tooling / external-evidence-dependent | Content-bound positive and negative controls, reference/no-update diagnostics, provider request exchange, immutable review, and scoring exist; no external control corpus is checked in. |
| H7 volunteered-statement review | Implemented tooling / external-evidence-dependent | Complete planning, provider-audit binding, accepted-update conversion, and source-safe recomputation exist; no accepted provider evidence is checked in. |
| Experiment B | Implemented core / live-design revision required | Crossed initial profiles, policies, and updaters; endogenous trajectories; block-balanced entropy-guided exploratory coverage; same-history action-aware shadows; terminal error, learning-confidence gain, decomposition, and pilot-power analysis are implemented. The 12-turn offline reference exercises repeated attributes, but the current three-turn live presets do not and must be revised before a scientifically informative B pilot. |
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
The B entries are valid transport bounds, not an endorsement of their current
three-turn horizon. A six-turn, one-domain, `llm_full_context`-only development
pilot with eight users, four initial-profile conditions, two policies, and one
trajectory uses 384 experiment calls plus 24 calibration calls per model. A
matched two-domain form uses 768 plus 48, or 816 calls per model.

The offline Gate 4 source produces 640 decoder requests per model and 80
native-action requests. The offline Experiment C rescore source produces 360
decoder requests per model. These generation runs do not call a model.

## Current model roles

| Role | Current declaration | Transport |
| --- | --- | --- |
| Historical neutral-base/display-name candidate author | `anthropic/claude-sonnet-5` | Separate OpenRouter authoring command; not called per treatment or trial. The current visible bases were standardized afterward and are project-sourced. |
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

## Scenario calibration snapshot

The outcome-free test-split audit for a planned 16-turn trajectory reports
engineering readiness only. Across all 64 theta states, all 27 declared
susceptibility profiles, both anchor directions, and both counterbalanced
orders, the current response coefficients pass the prospective non-degeneracy
guardrails. Mean ranking, default, and suggestion increments are approximately
5.10, 5.47, and 4.74 percentage points. Every physical per-order anchor
probability is strictly between 0.05 and 0.95 for balanced, restricted, default,
and suggestion contexts. These are simulator properties, not LLM results and
not evidence that visible prose has equivalent semantic strength.
This exhaustive stress grid is not the realized test population: v2 assigns 16
theta and nine susceptibility profiles to the test split, and a run allocates
balanced orders from those supports rather than enumerating their 16×9
Cartesian product at ordinary sample sizes.

Each test domain×attribute cell has six scenarios, enough to avoid
within-trajectory reuse for the 16-turn cyclic reference under the
without-replacement schedule. The v2 exploratory policy now preserves the same
bound by covering all three attributes in each complete three-turn block;
unknown or custom unconstrained adaptive policies remain outside that
guarantee. Version 1.4 also declares and counterbalances both possible nuisance
attributes and both nuisance directions. The 72 test scenario-anchor instances
now contain 24 numeric signatures, each represented three times. Twenty-four
original scenario families were semantically revised to expose held-constant
and nuisance facts or correct surface asymmetries; 24 new test families were
added prospectively without consulting evaluated-model outcomes. Separately,
all 48 conversation bases were standardized outcome-blind onto three
source-neutral frames: each frame appears 16 times overall and twice in every
six-scenario test domain×target cell. Their common source is
`project-standardized-neutral-frame-v1-unreviewed`. The exhaustive
surface-hygiene scan now covers all 1,440 rendered combinations and reports
zero cases. The v1.4 test-split audit also
reports zero raw label-length, cross-split lexical-overlap, within-split lexical
redundancy, or exact task-family-reuse flags at the declared thresholds. These
heuristics make no semantic-similarity claim. This is a machine screen, not
approval: the pending
catalog review fields, two independent fact mappings, surface/scientific
reviews, neutral-choice and attractiveness pretests, and final freeze still
block scientific collection. Counterbalanced vectors and clean machine
warnings do not validate the visible prose. Historical candidate origins
remain separately recorded; the standardization does not constitute human
review.

## Local live validation snapshot

Machine-local diagnostics have exercised direct OpenAI and one schema-valid
request to each selected OpenRouter decoder route. Some supporting evidence was
temporary and none is checked in, so these calls are not independently
verifiable release evidence and do not enter a dataset, hypothesis, or gate.
The dedicated `demo one-scenario` path also completed its fresh-directory
Gemini/OpenRouter check with one executed request, zero resumed requests, and
one physical transport attempt; its ignored local walkthrough remains
diagnostic only.

On 2026-07-29, `demo experiment-b-case` completed matched three-turn
diagnostics for Gemini 3.6 Flash through OpenRouter, Claude Sonnet 5 through
OpenRouter, and GPT-5.6 Sol through direct OpenAI. Each retained six logical
updates with no retry. All accepted responses used the requested model and
direct route, and no three-turn trajectory recorded a profile-influenced
action.

Six-turn follow-ups completed for Gemini and GPT. Both retained two
profile-influenced presentation actions in the soft-policy branch, on turns 4
and 5, while the balanced branch retained none. The simulated choices and
regret did not change, and neither model produced a five-clause reportable
self-confirmation case. This shows that repeating each attribute can activate
the counterfactual policy-feedback check; it does not establish prevalence,
harm, a model ranking, or a hypothesis.

The Claude six-turn attempt is incomplete and excluded. Eight responses were
accepted; a ninth paid response was rejected because OpenRouter reported a
nonempty passed-moderation pipeline under the current strict route-integrity
rule. No retry occurred. Any revised treatment of pass-through moderation must
be defined prospectively and followed by a fresh complete run.

All of these artifacts are machine-local, diagnostic-only, source-unfrozen,
paper-ineligible, and claim-ineligible. They cannot support Gates 2 or 3, a
power decision, or a paper result.

On 2026-07-29, the bounded Gemini/OpenRouter Experiment A pilot completed in a
fresh ignored run and passed full artifact verification. It retained 768
natural conversation records, 4,608 compact analysis rows, zero excluded
matched sets, and 505 accepted physical calls using the direct Google route for
`google/gemini-3.6-flash` (785,304 provider-reported tokens and USD 1.636068
provider-reported cost). Gate 1 met its coded diagnostic checks and held-out
paraphrase transfer was complete, but the exact H1 and H2 decision criteria
were not met and H7 was incomplete. The run has only four user clusters, uses
the 200-bootstrap smoke fallback, remains local/pilot-only, and does not enter
paper evidence or establish a scientific claim.

The checked-in conversation-authoring log records 24 prepared and 24 completed
OpenRouter calls to pinned `anthropic/claude-sonnet-5`. Those calls created
historical candidate stimulus language only. The current visible bank was
project-standardized afterward and independently passes exact catalog coverage
and runtime validation; the log is not a provider-authorship claim for its
current text. Neither candidate authoring nor standardization generated
experiment observations, evaluated a profile writer, or changed the bank's
unreviewed, not-paper-eligible status.

An earlier Claude diagnostic exposed a route-specific rejection of numeric JSON
Schema bounds. The compatible wire-schema subset and strict local finite-range
and vector-normalization checks are now covered by the offline suite.

Additional ignored diagnostics include verified Gate 4 and Experiment C source
packets, a three-seed Experiment C review, and a complete 19-point simulator
OAT run. They all retain `not_claimed` and are not distributed evidence. The
complete Gate 4 decoder/native-action collections and Experiment C external
rescore collection have not been executed.

The hybrid renderer and authoring path are distinct from those diagnostics.
The checked-in standardized bank can be rendered offline, while the separate
OpenRouter candidate-authoring command requires explicit authorization. Its
model output contains only candidate neutral base wording and display names;
code adds the fixed treatments and choice reply. The historical
`.generation.jsonl` records those candidate calls, not provider authorship of
the current visible frames. Neither the bank nor its authoring log is an
evaluated profile-writer corpus. No completed human review makes the bank paper
evidence.

Some machine-local completed runs predate runner-native compact rows. They must
not be edited or backfilled because doing so would invalidate `SHA256SUMS`.
`artifact compact` instead derives `analysis-rows.jsonl` in a separate
source-bound directory, and `artifact verify-compact` verifies that projection.
It does not alter or promote the source run.

Verified ignored compact sidecars now exist locally for the completed primary
Experiment A run, the completed Experiment B run, and all three completed
Experiment C seed runs. Together they occupy about 97 MB, versus roughly 25 GB
for those full source runs. They remain local analysis conveniences, not
checked-in evidence or paper results.

On 2026-07-29, the seven separate incomplete or interrupted sensitivity attempt
directories were revalidated as unfinished (`created` or missing a manifest),
confirmed to have no completed checksum inventory, and permanently deleted with
the operator's explicit approval. This reclaimed 28.551 GiB. The similarly
named completed `simulator-sensitivity-full-725020f80ee4` run was excluded from
that cleanup and verified successfully both before and after it.

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

- Complete independent surface and scientific review of the scenario catalog
  and conversation bank using the generated review packet; resolve or reject
  semantic cross-loading, dominance, and semantic duplication without
  consulting experiment outcomes.
- Calibrate semantic strength with independent validation data, adjudicate any
  machine warnings introduced by future revisions while blinded to outcomes,
  and then freeze an approved paper-eligible catalog version before
  confirmatory collection.
- Freeze the preregistration, estimands, model identities, provider routes,
  budgets, and release policy.
- Replace the three-turn Experiment B live presets with a ceiling-safe design
  that repeats attributes, preserves at least eight independent user clusters,
  and defines the OpenRouter moderation policy prospectively.
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
| Render frozen natural conversations after mathematical choices | Yes |
| Treat the current scenario/conversation surfaces as paper-eligible | No; independent human reviews, neutral-choice/attractiveness and semantic calibration, approval, and a final freeze are pending |
| Run and verify offline Experiments A–C and sensitivity | Yes |
| Analyze A–C using compact rows without loading full audit events | Yes for new runs; use the derived compact-artifact workflow for immutable historical runs |
| Plan live requests without reading credentials | Yes |
| Execute an explicitly authorized bounded live pilot | Yes, subject to operator review and provider availability |
| Reproduce checked-in empirical findings | No; none are checked in |
| Make paper claims or mark a scientific gate passed | No |

Use [Getting started](getting-started.md) for the first run,
[Configuration](configuration.md) for the public presets,
[Live execution](live-execution.md) for provider commands and recovery, and
[Architecture](architecture.md) for the component and trust-boundary map.
