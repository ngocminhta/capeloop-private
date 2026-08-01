# Implementation status

CAPE-Loop is an alpha research implementation. This page is the authoritative
boundary between executable repository software and evidence that still has to
be collected. No checked-in artifact establishes a paper hypothesis or claims
that a scientific gate has passed.

**Last reconciled:** 2026-07-31 against the source tree, command-line interface,
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
| Is it ready for a bounded live pilot? | **Engineering-ready after operator review of the scenario and conversation surfaces.** Experiment B now freezes and admits an outcome-blind six-turn manipulation schedule, runs a multi-seed simulator-only audit before any model call, and has a credential-free four-arm model-suite plan. These are bounded calibration safeguards, not paper evidence or permission to use provisional stimuli unchanged. |
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

The primary evaluation object is now explicitly the updater–logging-policy
pair. Experiment B estimates policy-conditioned evidential legibility with
same-history exact-shadow gaps, the prospectively matched soft-minus-balanced
contrast, and paired exact-shadow SelectionCost. Soft minus exploratory is
retained only as a supporting adaptive whole-policy comparison.
Experiment A is the fixed-response calibration of a candidate mechanism for
those gaps, not a cross-experiment mediation analysis; Experiment B is the
distinct natural-response feedback arm. Model
heterogeneity is retained. DIR, strict five-clause self-confirmation, and
Experiment C selection consequences are secondary; correction debt, native
external evidence, and human evidence remain stage-gated or deferred.

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
| Scenario catalog and selection | Implemented / review-dependent | Strict schema loading, 48 family-disjoint stimuli, deterministic paired selection, and within-trajectory sampling without replacement are implemented. Version 1.5 retains the v1.4 nuisance/direction/frame counterbalancing and adds a predeclared `target_half_span`: the six test scenarios in every domain×attribute cell use ordered spans `0.10, 0.16, 0.24, 0.34, 0.46, 0.56`, while train/development use `0.50`. This supplies prospective subtle-to-pronounced difficulty support without consulting model outcomes. `scenarios audit` reports all strata, applies nondegeneracy bounds to the 30 nondecisive test scenarios, records the six `0.56` decisive controls separately, and reports capacity, counterbalancing, overlap/surface warnings, and separate engineering/review/paper readiness. All 48 stimuli remain provisional, zero are approved, and independent semantic and human review is still required. |
| Hybrid conversation surfaces | Implemented / review-dependent | All 48 visible bases are outcome-blind project standardizations over three source-neutral frames. Code enforces the frozen neutral/treatment contract, rejects assistant assertions of a user's general preference, assigns A/B by visible position, inserts only fixed default/suggestion text, and fixes `I choose {selected_name}.` after the response model chooses. It also rejects user replies that turn a local choice into a general-preference claim. The automated screen cannot establish naturalness or semantic validity; independent human review remains pending. |
| Scenario human calibration | Tooling implemented / external evidence pending | `scenarios audit --split all` emits a metadata-visible workbook, a digest-bound opaque item map, a blinded surface packet, one frozen review protocol, and fillable JSON contracts. `scenarios review-promote` strictly imports two distinct surface reviews, two distinct scientific fact mappings, a neutral-choice pretest, and a target-masked paired-attractiveness pretest. It recomputes the preregistered thresholds and writes a new paper catalog only when the complete bundle passes; otherwise it writes only the failure report. Catalog review strings are never accepted as evidence, sources are never changed in place, and no completed human/pretest evidence is checked in yet. |
| Dataset surfaces and splits | Implemented | Catalog-bound train/development/test option, dialogue, and scenario families plus separate paraphrase and terminal families, with content digests and overlap audits. |
| Response models | Implemented | Random-utility and rule-based families; intrinsic utility remains separate from rank, default, and suggestion effects. |
| Exact inference | Implemented | Full finite preference×susceptibility posterior with an action-aware likelihood; public beliefs use the preference marginal. The declared coefficients, uniform susceptibility support, and preference-prior boundary are retained in `models/exact-action-aware-reference.json`. |
| Fitted references | Implemented secondary robustness | Parameter-count-matched aware and unaware likelihoods are trained on the same records, with raw and development-calibrated bundles retained separately. They test learnability/misspecification and are not the primary controlled A oracle. |
| Randomness | Implemented | Semantic-keyed SHA-256 streams and option-keyed Gumbels provide deterministic common-random-number branches. |
| Structured and native state | Implemented | Structured beliefs plus inspectable episodic, semantic, persona, and provenance-linked native memory with content-addressed state identity. |
| Information boundaries | Implemented | Evaluated LLMs receive semantic attribute meanings and readable options, not numeric features, the target index, or semantic catalog option IDs. Full/provenance views receive the rendered dialogue under per-position `presented_option_N` aliases; response-only remains an ablation. Latent truth is restricted to the simulator and evaluator. |
| Run integrity | Implemented | Resolved configuration identity, source digest, manifests, streamed checksums, symlink rejection, run verification, and deterministic freeze/verify tooling. |
| Compact analysis projections | Implemented | New A, B, and C runs automatically retain one updater×trial, trajectory-turn, or evaluation/ranking row respectively. A uses row schema v2 with exact/fitted update errors and log-odds residuals; B includes prospective preference-strength and balanced-choice-margin strata. Historical completed runs remain immutable and use a separately verified derived directory. |
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
| Experiment A | Implemented / exact-oracle primary | Five matched mechanisms (balanced, restricted, ranking, default, suggested); controlled same-response primary track with a fail-closed invariant audit; target-writer signed calibration-residual contrasts and model/mechanism calibration curves computed from raw LLM vectors as primary; ExactACUE is secondary unsigned magnitude; factorized priors expose exactly the same marginals to oracle and model; fitted, temperature-scaled, and naturally sampled analyses are secondary robustness; held-out controlled paraphrases, clustered inference, Holm adjustment, and pilot-power diagnostics. Earlier directional H1/H2 outputs remain diagnostic only. |
| A six-control exchange | Implemented tooling / external-evidence-dependent | Content-bound positive and negative controls, reference/no-update diagnostics, provider request exchange, immutable review, and scoring exist; no external control corpus is checked in. |
| H7 volunteered/provenance review | Implemented tooling / external-evidence-dependent | Complete volunteered-statement planning, provider-audit binding, accepted-update conversion, source-safe recomputation, and full-context versus provenance-aware comparison exist; no accepted provider evidence is checked in. The provenance-aware view changes metadata and instruction together. A pure metadata × instruction source-attribution ablation, assistant-only input, and neutralized assistant wording require a versioned extension and are not currently claimed. |
| Experiment B | Implemented / bounded calibration design ready | Before any evaluated-model call, the runner writes prospective plan v2, fixes the scenario and role of every balanced/soft turn, and fails closed unless each paired six-turn trajectory contains at least two susceptible near-tie/marginal active turns, one decisive active control, two presentation mechanisms, retained counter-profile options, and trajectory ASM at or above the declared threshold. Correct and incorrect seeds share the same scenario/role/mechanism schedule, policy randomization, and response-noise draw; a neutral current profile uses and logs the frozen seed direction. Required active actions are checked again during execution. Audit v2 uses an exact local driver and reports 32-seed pooled, condition, domain, condition-by-domain, and role summaries without using evaluated-model output or changing admission; a descriptive active-turn role × mechanism × effective/planned direction × target × domain cross-tab reconciles to the pooled required-active count. Analysis v5 uses exact small-sample complete-user sign-flip tests, a primary Gate 3 intersection-union test, a post-Gate-3 three-claim Holm family, user-cluster bootstrap sensitivity, separate legibility and net-harm decisions, the exact total-effect decomposition invariant, and distinct relative CEC, absolute CEC, EAR, partial reinforcement, paired behavioral reinforcement, DIR, and strict self-confirmation endpoints. Exploratory is a separate policy comparator, not a literally turn-matched causal branch. |
| Experiment B model suite | Implemented / live evidence absent | The frozen OpenRouter primary panel is Gemini 3.6 Flash, GPT-5.6 Luna, and Mistral Large 3 (`mistralai/mistral-large-2512`), each on the full eight-user design. DeepSeek V4 Flash is a post-pilot targeted secondary replication restricted to incorrect-seed balanced versus soft trajectories. Every model is analyzed separately; no model outputs are pooled, no “any-model” claim is supported, and DeepSeek is outside the primary analysis set. The within-model hierarchy does not promote bounded calibration output into paper evidence. Planning is credential-free; paid arms run sequentially only with `--execute-live`. |
| Experiment C | Implemented secondary v1 | Fixed balanced, fixed biased, and endogenous regimes; common terminal battery; complete-user paired ranking, reversal, top-tier, and evaluation-selection-regret analysis. Scores are properties of updater–logging-policy pairs. A larger crossed fixed-logger design remains a possible versioned extension, not a current claim. |
| Experiment C multi-seed review | Implemented diagnostic | Verifies 2–32 completed matched-source runs and records agreement and disagreements without selecting a favorable seed. |
| Sensitivity runner | Implemented | Cartesian and baseline-first one-at-a-time construction; active `lambda = {0, .33, .67, 1}` profile-conditioning dose; explicit exposure/divergence/informative-strata checks; target-consistent selection and attribution gaps; exact expected and realized information; action profile consistency; ex-ante and realized choice divergence; balanced-minus-soft exact-shadow information/disconfirmation deficits; CEC, DIR, terminal error, and observed-grid boundaries. Positive-dose zero divergence or inadequate informative coverage fails the manipulation; monotonic outcomes are not assumed. Strict self-confirmation is secondary. Public presets use OAT and do not estimate axis interactions. |
| Calibration | Implemented | Development-only temperature calibration for fitted and LLM probability outputs, raw/active retention, test reliability diagnostics, and an explicit uncalibrated ablation. Experiment A uses raw LLM vectors for primary update metrics and temperature scaling only as a secondary diagnostic; B/C retain configured-active histories. |
| Common terminal evaluation | Implemented | Held-out terminal-v2 battery shared by B/C structured and native systems; native representation diagnostics use two fixed blinded projections. |
| External native-state decoding | Implemented tooling / external-evidence-dependent | Blinded packets, source audit, development-only per-family calibration, held-out scoring, and agreement analysis exist; deterministic local projections are not independent external decoders. |
| Experiment C external rescore | Implemented tooling / external-evidence-dependent | Immutable import, calibration, native-only score replacement, full reranking, ESR, Gate 5 recomputation, and verification exist; no judgment corpus is checked in. |
| Mixed-effects analysis | Implemented optional analysis | Locked R/`lme4` formulas, maximal random-effects structure, planned contrasts, diagnostics, and checksum-bound outputs exist; the B terminal-error model is supporting rather than paper-primary, and only a synthetic CI fixture has been fitted. |
| Human H8 workflow | Implemented tooling / external-evidence-dependent | Vignettes, assignment, blinding/codebook, strict import, exclusions, model-evidence conversion, and paired comparison exist; recruitment and human data are deferred. |
| Correction debt / H9 | Implemented diagnostic and deferred | The paired protocol and transparent reference adapter are tested; a real native/LLM adapter and empirical H9 execution remain stage-gated and outside the minimum paper. |

Detailed estimands and output records are documented in
[Experiments](experiments.md), [Metrics](metrics.md), and
[Data model](data-model.md).

Experiment B's executable within-model policy is
`experiment-b-within-model-gatekeeping-v1`: Gate 3 is the sole primary IUT;
only its rejection opens Holm correction over the fixed secondary claims for
Gate 2, incorrect-minus-correct seed moderation, and nested net harm. Missing
members remain in the family with p=1. All other B endpoints and every bounded
calibration run are descriptive/supporting. The policy is applied separately
to each model and authorizes neither pooling nor an “any-model” claim.

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
| Gate 6 cross-run review | External-evidence-dependent | Revalidates declared live-model sensitivity/A pairs, provider and model identity, matched grids, paraphrase coverage/invariance readiness, and all coded clauses; qualifying source runs are absent. |

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
They use five mechanisms and only the primary controlled-response track, with
controlled held-out paraphrases. They are transport and estimability pilots,
not powered confirmatory designs.

| Adaptive live pilot | Experiment calls | Calibration | A paraphrases | Physical-attempt bound | Maximum output allocation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Experiment A | 480 | 60 | 40 | 580 | 1,187,840 |
| Experiment B | 576 | 60 | 0 | 636 | 1,302,528 |
| Experiment C | 768 | 60 | 0 | 828 | 1,695,744 |
| Gate 6 OAT | 720 | 0 | 0 | 720 | 1,474,560 |

Each uses zero automatic retries and remains below the approved per-provider
ceiling of 900 physical attempts and 6,000,000 conservatively allocated tokens.
The A presets run 480 controlled same-response updates, 60 calibration updates,
and 40 controlled held-out-paraphrase updates. The B pair is the two-domain,
eight-user, six-turn, `llm_full_context`-only paid design over
correct/incorrect seeds and balanced, soft, and exploratory policies: 576
trajectory calls plus 60 calibration calls, or 636 per model. Local reference
updaters run in the same experiment
without provider calls. Repeat this frozen design separately for each selected
model family.

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

The bounded Experiment B model declaration is frozen in
`data/model-suites/experiment-b-bounded-calibration-v1.json`. Its primary trio
is `google/gemini-3.6-flash` (minimal), `openai/gpt-5.6-luna` (low), and
`mistralai/mistral-large-2512` (no reasoning parameter). The separately labeled
`deepseek/deepseek-v4-flash` arm is a post-pilot targeted replication of only
the incorrect-seed balanced-versus-soft contrast. Every model remains a
separate updater–policy analysis: there is no cross-model pooled result,
DeepSeek is outside the primary analysis set, model labels are not user
clusters, and the suite is not a model leaderboard.

## Scenario calibration snapshot

The outcome-free test-split audit reports engineering readiness only. Across
all 64 theta states, all 27 declared susceptibility profiles, both anchor
directions, and both counterbalanced orders, the 30 nondecisive test scenarios
pass the prospective non-degeneracy guardrails. Mean ranking, default, and
suggestion increments on that scope are approximately 7.09, 7.47, and 6.50
percentage points. Every physical per-order anchor probability there is
strictly between 0.05 and 0.95 for balanced, restricted, default, and suggestion
contexts. The six `0.56` scenarios are retained and reported separately as the
predeclared decisive-control stratum rather than being forced through an
informative-turn nondegeneracy criterion. These are simulator properties, not
LLM results and not evidence that visible prose has equivalent semantic
strength.
This exhaustive stress grid is not the realized test population: v2 assigns 16
theta and nine susceptibility profiles to the test split, and a run allocates
balanced orders from those supports rather than enumerating their 16×9
Cartesian product at ordinary sample sizes.

Each test domain×attribute cell has six scenarios, enough to avoid
within-trajectory reuse for the 16-turn cyclic reference under the
without-replacement schedule. The v2 exploratory policy now preserves the same
bound by covering all three attributes in each complete three-turn block;
unknown or custom unconstrained adaptive policies remain outside that
guarantee. Version 1.5 retains both possible nuisance
attributes and both nuisance directions. The 72 test scenario-anchor instances
now contain 72 numeric signatures because target half-span is part of the
numeric design. Twenty-four
original scenario families were semantically revised to expose held-constant
and nuisance facts or correct surface asymmetries; 24 new test families were
added prospectively without consulting evaluated-model outcomes. Separately,
all 48 conversation bases were standardized outcome-blind onto three
source-neutral frames: each frame appears 16 times overall and twice in every
six-scenario test domain×target cell. Their common source is
`project-standardized-neutral-frame-v1-unreviewed`. The exhaustive
surface-hygiene scan now covers all 1,440 rendered combinations and reports
zero cases. The v1.5 test-split audit also
reports zero raw label-length, cross-split lexical-overlap, within-split lexical
redundancy, or exact task-family-reuse flags at the declared thresholds. These
heuristics make no semantic-similarity claim. This is a machine screen, not
approval: the pending
catalog review evidence is still absent: two independent fact mappings,
surface/scientific reviews, neutral-choice and attractiveness pretests, and a
final promoted freeze still block scientific collection. The version-bound
contracts, aggregation checks, and non-mutating promotion command are now
implemented. Counterbalanced vectors and clean machine
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

On 2026-07-30, a broader ignored cross-model calibration used one simulated
travel user, one replicate, and five OpenRouter model families where endpoints
completed. It used the pre-redesign A directional H1/H2 diagnostics and a
sensitivity slice that changed reference coefficients without changing the
visible interaction. Those exploratory observations motivated the current
exact-oracle calibration, same-response audit, visible policy-dose checks,
continuous B endpoints, and six-turn live B presets. They are not evidence for
the redesigned estimands: the pilot remains one-user, incomplete across
endpoints, ignored, and claim-ineligible.

All of these artifacts are machine-local, diagnostic-only, source-unfrozen,
paper-ineligible, and claim-ineligible. They cannot support Gates 2 or 3, a
power decision, or a paper result.

On 2026-07-29, a bounded Gemini/OpenRouter Experiment A pilot completed under
the earlier design and passed its then-current artifact verification. It
predates the fifth ranking mechanism, exact-oracle primary analysis,
same-response audit, compact A schema v2, and current request plan. It therefore
cannot be admitted as evidence for the redesigned experiment, regardless of its
historical transport or diagnostic results.

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
packets, a three-seed Experiment C review, and a complete historical 19-point
simulator OAT run collected before the active policy-dose axis was added. They
all retain `not_claimed` and are not distributed evidence. The
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
`metrics/gate-report.json` with the six numbered gates. Schema v2 also retains
any nested decisions in `nested_gates`; Experiment B uses this for net profile
harm beneath Gate 3. Failed or interrupted attempts may stop before that
artifact exists. Gates outside a run's scope are explicitly incomplete.

| Gate | Implemented computation and current evidence boundary |
| --- | --- |
| Gate 1 — identifiable causal-provenance calibration | Outcome-neutral readiness gate: Experiment A requires a passing same-response audit, exact-oracle self-consistency, all declared domain×mechanism cells, nontrivial paired exact warranted-update separation from balanced for at least two non-balanced mechanisms in both domains, and complete/invariant controlled held-out paraphrases. It does not require an LLM failure. Fitted aware-versus-unaware and held-out Brier gaps are noncontrolling secondary diagnostics. Missing coverage leaves it incomplete; no qualifying live evidence is checked in. |
| Gate 2 — conditional behavioral feedback amplification | In the incorrect-initial-profile stratum, this downstream gate requires an active soft channel with counter-profile options, visible action divergence, natural-choice divergence, later action influence, and one-sided complete-user evidence that the soft-minus-balanced CEC contrast is positive. These four statistical components form an IUT, and the composite must survive the fixed post-Gate-3 Holm family. The CEC contrast is a **relative confidence penalty**, not by itself absolute confidence growth or reinforcement. Absolute soft CEC, EAR, partial reinforcement, paired behavioral reinforcement, and the strict five-clause endpoint remain descriptive/supporting. |
| Gate 3 — policy-conditioned evidential legibility | For `llm_full_context` in the incorrect-initial-profile stratum, one-sided paired complete-user tests must support `G_soft > 0`, `G_soft - G_balanced > 0`, and SelectionCost `< 0.02` on the marginal-Brier scale. This is the sole primary within-model IUT: its p-value is the maximum component p-value, and every component must reject at alpha `0.05`. Exact sign enumeration is used through 16 users; larger samples use bounded Monte Carlo sign patterns. User-cluster bootstrap intervals are sensitivity summaries, not the gate decision. No passage is currently claimed. |
| Nested Gate 3 — net profile harm | In the same incorrect-initial-profile stratum and in addition to Gate 3 legibility, this stricter decision requires one-sided evidence that `soft_minus_balanced_terminal_error > 0.02` and rejection in the fixed secondary Holm family. It is retained separately so a harder-to-interpret history is not mislabeled as net harmful when selection benefits offset attribution cost. |
| Gate 4 — native-system validity | Requires external blinded-decoder judgments and genuine hash-bound native end-to-end actions. The selected path requires complete Claude/Gemini OpenRouter evidence, complete OpenAI native actions, and responsible source review; optional direct-first-party and explicitly reviewed-generic modes also exist. No complete live collection has been imported. |
| Gate 5 — evaluation implication | Experiment C uses joint-paired complete-user differences and difference-of-differences for credible reversals, or interval-supported development tiers plus a conservative held-out ESR envelope. Descriptive correlations or rank bands cannot pass it alone. |
| Gate 6 — robustness | Sensitivity artifacts check declared-grid completion, visible treatment activation, prospective informative-strata occupancy, and observed phase boundaries. `lambda = 0` is a negative control; a positive dose with zero exposure, zero visible divergence, or fewer than the frozen informative-user/turn minimum fails the manipulation. Strict self-confirmation does not control the operational joint region. A separate review binds qualifying live sensitivity/A runs and recomputes model-family clauses plus outcome-neutral paraphrase coverage/invariance; none is checked in. |

`computed_status = "meets_computational_checks"` means only that every coded
Boolean available to that artifact evaluated true. It never changes
`claim_status` from `"not_claimed"`.

## Work still required for paper evidence

- **User/researcher action:** review the scenario catalog and every rendered
  conversation surface using the generated packet. Resolve semantic
  cross-loading, dominance, duplication, naturalness, and neutrality without
  consulting evaluated-model outcomes.
- Complete the two surface-review files, two scientific-review files,
  neutral-choice pretest, and target-masked attractiveness pretest produced by
  `scenarios audit --split all`. Run `scenarios review-promote` to aggregate
  them; it freezes a new paper-eligible catalog and companion bank only when
  every version-bound criterion passes.
- **User/researcher action:** freeze the preregistration, exact-oracle
  estimands, sample size/power decision, budgets, and release policy. The
  bounded-B model identities, primary/secondary boundary, manipulation roles,
  and the two `0.02` practical margins are already frozen in repository inputs;
  revise them only prospectively, before collection, if the manual stimulus
  review requires a new design version.
- Run the credential-free Experiment B manipulation audit and model-suite plan
  from the reviewed inputs, then add `--execute-live` only for the authorized
  collection. Keep every model in its own output subtree and never pool the
  targeted DeepSeek arm into the primary trio.
- Execute preregistered alternate seeds as separate robustness analyses.
  Different seeds must not be pooled to manufacture additional independent
  users; the confirmatory harness rejects that pooling.
- Treat Experiment C v1 as secondary and run it only after the core A/B
  collection is frozen; do not expand its logger design opportunistically after
  seeing A/B outcomes.
- Execute and admit the frozen Gate 6 collections after confirming that every
  positive policy dose activates the visible manipulation.
- Collect the complete Gate 4 Claude/Gemini judgments and OpenAI native actions,
  then complete the responsible-researcher source review.
- Run the external Experiment C rescore and the declared multi-seed review.
- Obtain the required ethics determination before the deferred human study,
  then collect and analyze eligible responses if that study proceeds.
- Fit the registered R models on frozen eligible runs and review diagnostics;
  retain the B terminal-error fit as supporting, and do not treat the CI
  fixture as evidence.
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
