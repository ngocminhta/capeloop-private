# Implementation status

CAPE-Loop is an alpha research implementation. This page distinguishes
executable software from empirical evidence. No checked-in file claims that a
paper hypothesis or scientific stage gate has passed.

**Last reconciled:** 2026-07-28 against the current source tree, CLI, checked-in
schemas/configurations, tests, and CI workflow.

## Status vocabulary

- **Implemented:** executable in the current source tree and covered by an
  offline test or end-to-end runner test.
- **Implemented diagnostic:** executable and tested, but deliberately emits
  `claim_status = "not_claimed"` and cannot establish a paper result by itself.
- **Implemented optional analysis:** executable and tested behind an isolated
  optional dependency environment; it is not required by the standard-library
  Python runtime.
- **Provider-capable:** the code can construct and execute a live request, but
  no external response or result is checked in.
- **External-evidence-dependent:** the protocol, validation, and analysis code
  exists, while completion requires credentials, reviewed external sources,
  human participants, or another outside input.
- **Diagnostic reference:** an inspectable adapter exercises the research
  protocol but is not evidence about an LLM or native system.
- **Not implemented:** the proposal requirement is absent from the executable
  path.

## Validation snapshot

The reconciled offline tree currently passes:

- 373 executed tests across 38 test modules with no network calls;
- validation of all 18 public TOML configurations;
- local Python compilation, Ruff, and `git diff --check`;
- byte-for-byte parity for 31 checked-in generated JSON Schemas;
- the static mixed-effects contract validator; and
- resolution of every relative Markdown file target.

The standard CI matrix runs the core suite on Python 3.11, 3.12, 3.13, and
3.14. A separate immutable-SHA-pinned CI job restores the frozen R
environment, validates the contract, generates a synthetic verified Experiment
A run, executes the R 4.6.1/lme4 harness, and checks the resulting artifact.
This validation establishes software behavior only, not empirical findings.

## Frozen reference defaults

The proposal leaves pilot parameters open. The executable reference uses these
explicit, replaceable defaults:

| Decision | Current reference |
| --- | --- |
| Runtime | Python 3.11+ standard-library core |
| Preference support | `{-2, -1, +1, +2}³` |
| Susceptibility support | `0.15`, `0.45`, `0.85` |
| Intrinsic response scale | `1.0` |
| Rank/default/suggestion scales | `0.35 / 0.80 / 0.65` |
| Minimum matched-choice probability | `0.05` |
| Primary profile error | Brier score over attribute marginals |
| Information gain | Prior entropy minus posterior entropy |
| Calibration | Temperature scaling on development data only |
| Run declaration | Strict schema-versioned TOML |
| Canonical scientific records | JSON and JSON Lines; CSV is a derived projection |
| Randomness | Semantic-keyed SHA-256 streams |

These values are software defaults, not preregistered scientific choices.
Paper-intended runs must freeze their resolved configuration, source identity,
split manifest, external inputs, and checksums.

The durable implementation boundary is:

```text
profile
  -> separately retained policy provenance and visible context
  -> simulated or replayed response
  -> declared updater information view
  -> structured profile or native-memory update
  -> evaluator-only metrics and no-claim gate records
```

Only the simulator and evaluator may access latent truth. Presentation affects
response probability, not intrinsic welfare or latent preference. External
providers receive only the view-specific content-addressed request.

## Current executable scope

| Capability | Status | Exact boundary |
| --- | --- | --- |
| Python 3.11+ standard-library core | Implemented | No third-party runtime dependency; live HTTP is isolated behind explicit opt-in |
| Strict schema-versioned TOML | Implemented | Unknown keys and experiment-incompatible combinations fail |
| Travel and writing domains | Implemented | Three signed attributes per domain |
| Semantic-keyed RNG and option-keyed Gumbels | Implemented | Deterministic common-random-number support |
| Random-utility response model | Implemented | Intrinsic and presentation utility are separate |
| Rule-based response model | Implemented | Selectable as a sensitivity-family axis; ordinary A–C configs use random utility |
| Exact finite action-aware inference | Implemented | Full preference×susceptibility joint retained; public belief is its theta marginal |
| Fitted aware/unaware likelihoods | Implemented | Shared training records and four fitted parameters each; outcome and function classes differ |
| Development temperature calibration | Implemented | Raw and active fitted bundles are retained separately |
| Executed data-split surfaces | Implemented | Atlas/beacon/cedar option, dialogue, and scenario families plus content-addressed paraphrases are generator-bound and overlap-audited |
| Matched anchor contexts | Implemented | Balanced, restricted, default, and suggested |
| Experiment A runner | Implemented | Controlled/naturally sampled modes, executable prior-strength strata, separate content-addressed six-control plan with reference/no-update diagnostics and provider exchange, exact-byte input binding and immutable control review, oracle slopes, evidence ordering, raw/calibrated reliability, clustered contrasts/interactions, CR1 marginal OLS, Holm, and pilot power; external control evidence and result claims remain absent |
| H1/H2/H7 estimands | Implemented diagnostic | Versioned artifacts encode H1 directional/strength contrasts, H2 aware-versus-unaware proximity on at least two mechanisms, H7 mitigation superiority, 80% balanced/volunteered valid-learning retention, and matched closed-loop reduction; absent LLM or volunteered outcomes remain explicitly incomplete |
| H7 volunteered direct-statement review | External-evidence-dependent | A verified A run deterministically yields every test-user/domain/attribute request crossed with full-context and provenance-aware writers; exact accepted OpenAI/OpenRouter audit coverage converts to paired `VolunteeredPreferenceUpdate` records and recomputes H7 in a separate hash-bound `not_claimed` artifact; source, plan, response, and audit bytes are snapshotted and rechecked before non-overwriting publication, without source mutation or imputation |
| Experiment B runner, same-history shadow, and pilot power | Implemented | Four initial-profile conditions and configured policy/updater crossing; complete-user frozen three-way terminal-error contrast with bounded 16/32/64/128 power curve and Monte Carlo uncertainty; calibrated LLM runs retain cached same-request raw/calibrated terminal scores without claiming a recursively raw trajectory |
| Confirmatory mixed-effects harness | Implemented optional analysis | Version-pinned R/lme4 pipeline implements both proposal formulas, maximal random effects, planned emmeans contrasts, pointwise intervals with Holm-adjusted p-values, canonical config/source validation, curvature-scaled gradient diagnostics, and checksum-bound JSON/CSV outputs; B is reconstructed one row per retained turn and no fit or claim is checked in |
| Experiment C runner | Implemented | Fixed balanced, fixed biased, and endogenous regimes; stable-key alignment and complete-user clustered ranking bootstrap; structured rows use their public projection while native rows use `mean_of_two_blinded_native_decoders`; cached raw/calibrated terminal diagnostics do not replace rankings |
| Experiment C external-decoder rescore | External-evidence-dependent | Exports every native fixed/endogenous terminal state; exact two-family judgments are calibrated on development only, change only declared native score fields, and then rerun the full all-system C ranking/ESR/Gate 5 analysis in a same-parent staged, fsynced, input-reverified, self-verified atomic artifact; the CLI requires a complete selected OpenRouter collection, a complete optional direct first-party collection, or an explicitly authorized reviewed-generic mode, and validates the declared mode before computation and again before publication; no real judgment corpus or result is checked in |
| Held-out paraphrase suite and Gate 1 adapter | Implemented | Test-only response families are content bound and leakage checked; completeness still depends on required updater/case pairs |
| Held-out terminal v2 suite | Implemented | Shared B/C diagnostic uses new option IDs/features/scenarios/wording and all four question types; B also retains explicit action bindings |
| B/C terminal calibration output | Implemented | Every profile score retains top-label ECE/reliability with one preference attribute as forecast unit; pooled artifacts preserve trajectory/user dependence |
| Sensitivity runner | Implemented | Cartesian and baseline-first one-at-a-time designs; independent mechanism, prior-uncertainty, trajectory, model-family/rule-noise axes plus explicit phase points and observed-grid boundary intervals; OAT artifacts deny interaction estimability |
| Native episodic/semantic/provenance-linked memory | Implemented | Content-addressed full state and policy-facing persona belief; episodic query-time inference and conservative semantic consolidation use distinct declared transition strengths |
| Two blinded native projections | Implemented | Direct-semantic and history-evidence deterministic views are both retained and evaluated; they are not independent judgments |
| External decoder exchange and analysis | External-evidence-dependent | Blinded requests, source audit, development-only per-family temperature calibration, test reliability/performance, and agreement are implemented; real judgments are absent |
| Common terminal projection battery | Implemented | Built from heldout-terminal-v2 and shared by every B/C comparison; decisions are still computed from projected beliefs |
| Joint-paired ranking and inferential-tier ESR | Implemented | Complete-user open/closed difference and difference-of-differences intervals determine credible reversals and top tiers; paired test intervals yield a conservative ESR envelope |
| Experiment C multi-seed robustness review | Implemented diagnostic | Verifies 2–32 completed distinct-seed runs with identical scientific config/source identity, compares point rankings, inferential tiers/orders, Gate 5, and ESR sets, and atomically retains exact agreement fractions/disagreements in a checksum-bound `not_claimed` artifact |
| Gate 6 cross-run robustness review | External-evidence-dependent | Verifies explicitly declared live-model sensitivity/Experiment A pairs, exact provider/model evidence, matched scientific grids, recomputed held-out paraphrase transfer, and all six tri-state clauses; family/source identity remains researcher-declared and never implies independence |
| Gate-report machinery | Implemented diagnostic | `claim_status` remains `not_claimed` |
| Run manifests, source digest, and checksums | Implemented | Verification checks complete status, run ID, exact retained resolved-config digest/schema, summary presence, source-config semantic equivalence, and exact agreement between TOML presence and the declared TOML/programmatic origin; payload hashes stream in bounded memory, control files are size-bounded, and any artifact-tree symlink fails closed; newer parser defaults do not rewrite historical digests |
| Deterministic paper-artifact freeze/verify | Implemented | Requires a verified run plus retained TOML source or hash-bound programmatic config origin; writes a lexically ordered normalized `.tar` outside the source run plus a bounded digest sidecar, and independently checks member/PAX metadata, checksums, manifest/config bindings, and the embedded summary |
| Public JSON Schema export | Implemented | CLI writes deterministic schema files |
| Minimal public configuration matrix | Implemented | 18 immutable presets remain: nine quick/offline/source designs and nine bounded live designs under `configs/live/`; transport-only smokes and superseded examples were removed, local variants are ignored under `configs/local/`, and the R-only synthetic fixture is owned by its analysis component |
| Universal adaptive LLM preflight | Implemented | One exact logical-completion and worst-case physical-attempt bound covers A/B/C/sensitivity, calibration, A paraphrases, all turns, retry expansion, and maximum output allocation; validation and live startup fail before credential/artifact access, and runs retain `llm/request-preflight.json` |
| LLM profile writers | Provider-capable | Strict replay plus explicit direct OpenAI Responses API or first-class OpenRouter Chat Completions execution, retry-expanded physical-attempt/token budgets, fsynced started/settled journals, audit-first recovery, nonblocking static execution locks, and fail-closed resumability; bounded paper pilots use zero automatic retries under 900 attempts/6M tokens |
| Bounded live A/B/C pilot configs | Provider-capable | A is bounded at 848 attempts/1,736,704 output tokens, B at 864/1,769,472 including 96 calibration updates, and C at 816/1,671,168 including 48 calibration updates; matched OpenAI/OpenRouter configs remain pilots and no paper run is claimed |
| Bounded Gate 4/C external source configs | Implemented | Offline B source yields 640 decoder requests/source and 80 eligible native-action requests; offline C source yields 360 decoder requests/source; neither generation step calls a model |
| OpenAI primary/replication suite orchestration | Provider-capable | Credential-free planning and explicit two-role execution use immutable matched configs, distinct run/journal paths, per-role ledgers, and a combined index; Terra is not distinct-family robustness |
| OpenRouter gateway execution | Provider-capable | Exact canonical model slugs, provider-compatible strict JSON Schema, cache disablement, router-metadata and selected-upstream validation, route/privacy controls, exact plan/execution request identity and body hashes, gateway/upstream audit schema, static/adaptive execution, and locked multi-model collection are implemented. Anthropic wire schemas omit the numeric-bound keywords rejected by the observed Bedrock route, while the local parser still requires every probability to be finite and in `[0,1]` and every vector to sum to one. The selected decoder defaults are `anthropic/claude-sonnet-5` at `low` effort and `google/gemini-3.6-flash` at `minimal` effort, each with its own journal under one OpenRouter gateway. Plans, durable attempts, provider audits, responses, judgments, aggregates, and the execution manifest are revalidated together; all artifacts deny strict first-party eligibility, distinct transport origins, and statistical independence while remaining eligible for responsible-researcher-reviewed shared-gateway admission |
| LLM probability calibration | External-evidence-dependent | Per-updater temperature is fitted on development users and locked for test; raw/active responses are retained; `none` is an explicit ablation |
| Human pragmatic-study packet, import, and H8 comparison | External-evidence-dependent | Fixed vignettes, blinding/codebook, consent/comprehension validation, strict source-bound model evidence, verified Experiment A conversion, pair-complete participant/test-user bootstrap, and an exact-byte-bound non-overwriting comparison artifact; no participants were recruited |
| Correction-debt protocol | Implemented diagnostic | Stage-gated exact paired arms and recovery/debt metrics are implemented with a transparent log-odds reference adapter; it is not evidence about an LLM or native system |
| Gate 4 external-evidence ingestion | External-evidence-dependent | Keyless plans fail before output/key access, selected ceilings are 900 attempts/6M tokens/1,024 output tokens per decoder with zero retries, and import independently rebuilds the plan. `gate-review import-native --openrouter-collection-dir` binds a verified completed B run to the complete selected Claude/Gemini OpenRouter collection and OpenAI native-action collection under shared locks. It labels the decoder evidence `selected_openrouter_gateway_collection`, requires responsible-researcher source review, and never upgrades gateway provenance to a first-party or independence claim. The OpenRouter and direct flags now reject opposite-kind artifacts, and the selected OpenRouter-to-Gate-4 transaction has end-to-end success, provenance-boundary, and rejected-source-review coverage. The direct Anthropic/Gemini validator and reviewed-generic alternative remain available |

## External, deferred, or result-producing work

The entries below are not silent implementation claims. Most require evidence,
credentials, ethics/governance decisions, paper-scale execution, or release
metadata that software alone cannot supply.

| Capability | Current boundary |
| --- | --- |
| Live model evidence | No API keys, provider responses, or live model results are checked in; ignored local OpenAI/OpenRouter transport smokes exist but are explicitly excluded from the study corpus and every hypothesis/gate |
| Complete Gate 1 evidence | Requires the declared full-context updater and complete held-out case pairs; structured-only smoke runs remain incomplete |
| External Experiment A control evidence | All six typed controls execute under transparent reference and no-update diagnostic executors, and exact-bound OpenAI/OpenRouter response exchange/scoring is available; no external provider response corpus or empirical control result is checked in, and anchor choices are never imputed as controls |
| Executed H7 volunteered control evidence | Direct-statement planning, provider-bound conversion, immutable review, and exact reverification are implemented; no accepted provider corpus or derived empirical review is checked in, so ordinary run artifacts retain the volunteered criterion as incomplete |
| Executed confirmatory mixed-effects result | The optional R harness is implemented, but no verified study run has been fitted and no model result or claim is checked in; standard A pilots have one prior level and are `not_estimable`, while the bounded mixed A pair includes two levels as a transport/estimability pilot rather than a powered confirmatory design; multi-turn B runs supply within-trajectory `turn` variation |
| Native end-to-end natural-language terminal evidence | The selected OpenAI `gpt-5.6-sol` path implements keyless planning, explicit live authorization, origin/budget locks, durable physical-attempt journals, resumable audit-first collection, and source-safe outputs; no eligible native-action corpus is checked in, and transparent projections remain ineligible |
| Selected OpenRouter decoder evidence | The selected repository workflow routes `anthropic/claude-sonnet-5` and `google/gemini-3.6-flash` through `OPENROUTER_API_KEY`. The complete gateway collection validator is implemented, but no accepted paper-scale judgment corpus is checked in. Distinct model families behind a shared gateway do not establish distinct transport origins or statistically independent errors |
| Optional direct first-party decoder evidence | Direct Anthropic `claude-sonnet-5` and Google `gemini-3.6-flash` collectors and their strict validator remain implemented for optional origin-validation or replication. They require their own runtime credentials and paid collection; no direct corpus is checked in, and separate providers/families still do not prove statistically independent errors |
| Executed Experiment C external rescore | Official-collection and reviewed-generic import modes, calibration, reranking, ESR, Gate 5, and verification are implemented; no external judgment corpus or derived rescore artifact is checked in |
| Empirical LLM calibration result | Per-updater development-only temperature calibration is executable for replay and live responses, with `none` as an explicit ablation; no response corpus/result is included |
| Human study execution | Ethics/IRB or exemption, approved consent, recruitment, compensation, survey hosting, privacy/retention policy, and collection remain external |
| Executed H8 result | Strict model-evidence conversion and human/model comparison are implemented; no eligible participant response corpus or resulting H8 artifact is checked in |
| Native/LLM correction-debt result | A real H9 native/LLM adapter, prerequisite gate review, retained evidence, and frozen analysis are intentionally deferred; H9 is stage-gated and excluded from the minimum paper |
| Confirmatory LLM phase diagram | Phase code and the broader simulator grid exist; current structured fallback rows are labeled proxies and cannot substitute for live LLM sweeps |
| Executed Gate 6 cross-run review | Exact pair/family/provider/model binding, clause recomputation, and immutable review are implemented; no paper-scale set of qualifying live-model source runs or derived review is checked in |
| Paper-scale Experiment C multi-seed evidence | The strict offline reviewer is implemented and tested against real runner artifacts; no paper-scale distinct-seed source set or resulting empirical review is checked in |
| Paper-frozen artifacts and empirical results | Not included |
| DOI, accepted-paper metadata, or named paper authors | Not included |

## Local-only execution boundary

Ignored local runs have exercised direct OpenAI and OpenRouter transport,
including schema-valid requests to the selected Claude and Gemini routes. The
supporting provider artifacts, identifiers, usage, and costs are intentionally
not published in this repository, so these observations are not independently
verifiable release evidence and do not enter any dataset or paper claim.

One earlier Claude request exposed a route-specific rejection of numeric JSON
Schema bounds. The wire-schema compatibility fix and strict local range and
normalization checks are covered by the offline suite. The complete Gate 4 and
Experiment C provider collections remain unexecuted.

## Local bounded source-run snapshot

Two ignored, fully offline source runs now complete the packet-generation side
of the external workflows:

| Source run | Verified contents | Zero-retry keyless plan |
| --- | --- | --- |
| `experiment-b-gate4-source-pilot-cbfddb31b0a2` | 320 trajectories; 640 decoder requests/source; 80 native-action requests | OpenRouter Claude-low 4,469,348 tokens; OpenRouter Gemini-minimal 4,201,828; OpenAI native actions 2,396,907 |
| `experiment-c-external-rescore-pilot-558e7b78e7aa` | 360 decoder requests/source | OpenRouter Claude-low 2,709,130 tokens; OpenRouter Gemini-minimal 2,558,650 |

Every request count is below 900 and every conservative token plan is below
6,000,000. Both source runs verify and retain `not_claimed`; neither plan read
a credential or called a provider. The selected decoder collection requires
only `OPENROUTER_API_KEY`; optional direct replication uses separately supplied
Anthropic and Gemini credentials.

The ignored three-seed Experiment C review at
`artifacts/local/experiment-c-multiseed` also re-verifies against seeds 1729,
271828, and 314159. Only one of nine predeclared ranking conclusions was
unanimous across those runs. That retained disagreement is diagnostic
robustness information with `claim_status = "not_claimed"`, not a paper claim
or a reason to select a favorable seed.

The complete local OAT artifact at
`runs/simulator-sensitivity-full-725020f80ee4` completed under recorded source
digest `aec47ca2…` and remains independently verifiable with the current
verifier. Later Gate 4/Experiment C collection hardening makes it a
source-bound historical artifact rather than a current-tree execution; the
sensitivity runner and configuration did not change. It retains all 19
declared points, 29,184 synthetic
trajectories, 304 stratified rows, 152 decomposition rows, 19 phase rows, 38
domain-phase rows, and 137 boundary rows. The local directory is approximately
23 GiB because full trajectory events are retained. It contains no LLM model
IDs or provider calls, records that OAT interactions are not estimable, and
leaves Gate 6 `incomplete` with `claim_status = "not_claimed"`.

Older interrupted attempts remain in explicitly named directories, including
partial trees from earlier source revisions and a source-bound attempt
stopped by a 20-minute watchdog. Even when a partial tree contains event or
metric rows, it lacks a complete manifest, summary, and
checksum set and is ineligible for analysis.

Sensitivity configs admit replay, direct OpenAI, and OpenRouter `llm_*`
updaters only with uncalibrated operation, retained prompts/events, and a
retry-expanded preflight that proves the complete adaptive grid fits its
declared physical-request ceiling before credential access or artifact
creation. The hard cumulative token ceiling is reserved and enforced before
each request; no exact whole-grid prompt-token total is claimed because later
prompts depend on earlier responses. Requests are content-bound per grid point
and are not reused across changed dynamics. Sensitivity still requires
`experiment.turns = 1`; executed lengths come only from
`sensitivity.trajectory_lengths`. Phase results name whether the target is the
declared live LLM, a replayed LLM, or a structured proxy.

`configs/offline/sensitivity.toml` is a 19-point baseline-first OAT simulator
design. It covers broad marginal
perturbations but records that interaction effects are not estimable. The
matched OpenAI/OpenRouter Gate 6 pilots use 11 OAT points, 36 summed trajectory
turns, and a 576-physical-attempt upper bound per provider with retries
disabled. Neither live Gate 6 pilot has been executed as paper evidence.

## Supported CLI

```text
doctor
config validate CONFIG
run CONFIG [--output-root DIR] [--allow-existing] [--execute-live]
           [--resume-failed-live]
verify RUN_DIR
schema export [DEST]
llm validate RESPONSES.jsonl
llm models
llm plan REQUESTS.jsonl
llm execute-openai REQUESTS.jsonl RESPONSES.jsonl AUDIT.jsonl --execute-live
llm plan-openrouter REQUESTS.jsonl
llm execute-openrouter REQUESTS.jsonl RESPONSES.jsonl AUDIT.jsonl \
  --execute-live
llm evaluation-suite PRIMARY.toml REPLICATION.toml [--execute-live]
artifact freeze RUN_DIR ARCHIVE.tar
artifact verify ARCHIVE.tar
decoder-study validate REQUESTS.jsonl JUDGMENTS.jsonl
decoder-study analyze REQUESTS.jsonl JUDGMENTS.jsonl TRUTH.jsonl
decoder-study plan-openai REQUESTS.jsonl
decoder-study execute-openai REQUESTS.jsonl OUTPUT_DIR --execute-live
decoder-study plan-openrouter REQUESTS.jsonl \
  [--model MODEL [--additional-model MODEL]] \
  [--model-reasoning-effort MODEL=EFFORT]
decoder-study execute-openrouter REQUESTS.jsonl OUTPUT_DIR \
  [--model MODEL [--additional-model MODEL]] \
  [--model-reasoning-effort MODEL=EFFORT] --execute-live
decoder-study plan-distinct REQUESTS.jsonl [--output PLAN.json]
decoder-study execute-distinct REQUESTS.jsonl OUTPUT_DIR --execute-live
native-action plan-openai RUN_DIR [--output PLAN.json]
native-action execute-openai RUN_DIR OUTPUT_DIR --execute-live
gate-review import-native RUN REQUESTS JUDGMENTS TRUTH NATIVE_COLLECTION \
  SOURCE_REVIEW OUTPUT \
  (--openrouter-collection-dir DIR | --external-collection-dir DIR | \
   --allow-reviewed-generic-decoders)
gate-review verify REVIEW_DIR
gate6-review build DECLARATION.json OUTPUT_DIR
gate6-review verify REVIEW_DIR [--reverify-sources]
experiment-c-decoder import RUN_DIR JUDGMENTS.jsonl OUTPUT_DIR \
  (--openrouter-collection-dir DIR | --external-collection-dir DIR | \
   --allow-reviewed-generic-decoders)
experiment-c-decoder verify REVIEW_DIR [--source-run RUN_DIR]
experiment-c-robustness review OUTPUT_DIR SOURCE_RUN SOURCE_RUN [...]
experiment-c-robustness verify REVIEW_DIR [--source-run SOURCE_RUN ...]
human-study generate OUTPUT_DIR [--assignment-id ID] [--seed INTEGER]
human-study analyze RESPONSES.jsonl CODEBOOK.json
human-study evidence-from-experiment-a RUN_DIR EVIDENCE.jsonl --source ID=UPDATER
human-study compare RESPONSES.jsonl CODEBOOK.json EVIDENCE.jsonl OUTPUT.json \
  --primary-llm-source-id ID
correction-debt run OUTPUT.json --stage-gate-authorized
control-study analyze BINDINGS.json RESPONSES.jsonl OUTPUT.json \
  [--source-descriptor TEXT]
control-study h7-plan RUN_DIR OUTPUT_DIR
control-study h7-review RUN_DIR PLAN_DIR RESPONSES.jsonl \
  PROVIDER_AUDIT.jsonl OUTPUT.json
control-study h7-verify RUN_DIR PLAN_DIR RESPONSES.jsonl \
  PROVIDER_AUDIT.jsonl REVIEW.json
```

Use:

```bash
PYTHONPATH=src python -m cape_loop <command>
```

Packaging declares the `cape-loop` console script, but source-tree invocation is
the documented reproducibility path.

## Gate state

Every run writes `metrics/gate-report.json` with six gate entries. Gates not
evaluated by that run are explicitly incomplete.

| Gate | Current implementation behavior |
| --- | --- |
| Gate 1 — learnable provenance gap | Experiment A generates and scores leakage-checked test paraphrases. The criterion is complete only with all required domains/mechanisms and paired fitted-aware/`llm_full_context` scores; structured-only smoke runs remain incomplete |
| Gate 2 — nontrivial soft self-confirmation | Experiment B targets only `llm_full_context`; response-only/provenance-aware LLM variants are controls. Computational passage also requires at least eight user clusters and 95% user-clustered intervals above zero for mean LCG and five-clause profile rate. The offline reference config lacks this updater, while both bounded B live pilots include it; no qualifying live result is claimed |
| Gate 3 — attribution beyond evidence selection | Same `llm_full_context` boundary as Gate 2; computational passage requires an adequate 95% user-clustered same-history attribution interval above zero. No passage is claimed |
| Gate 4 — native-system validity | Experiment B restricts to incorrect-seed, soft-profile-conditioned native trajectories with counter-profile alternatives and checks retained state plus the matched native failure; responsibly reviewed distinct-family blinded decoders and genuine native end-to-end actions are mandatory incomplete prerequisites. The selected repository workflow admits only a complete audit-bound Claude/Gemini OpenRouter collection plus the complete OpenAI native-action collection and a responsible-researcher source review. It explicitly does not claim first-party decoder origin, distinct transport origins, or statistically independent errors. Direct first-party decoder collection remains an optional stricter provenance tier; deterministic projections and reference actions remain ineligible |
| Gate 5 — evaluation implication | Experiment C uses joint paired complete-user open/closed error differences for reversals and interval-supported top tiers plus paired test envelopes for ESR; descriptive tau, rank bands, and reversal probabilities cannot pass it alone |
| Gate 6 — robustness | The sensitivity runner checks declared-grid completion and retains phase classifications/boundaries. A separate immutable offline review verifies explicitly paired live-LLM sensitivity/Experiment A runs, exact model/provider evidence, multiple caller-declared families, and recomputed held-out paraphrase transfer; it retains `claim_status = "not_claimed"` and leaves family taxonomy, independence, preregistration, and paper review to responsible researchers |

`computed_status = "meets_computational_checks"` means only that every coded
Boolean criterion evaluated true. All gate records retain
`claim_status = "not_claimed"`.

## Calibration state

For every non-sensitivity run, the runner retains:

```text
models/raw-fitted-likelihoods.json
models/fitted-likelihoods.json
models/calibration.json
models/held-out-response-diagnostics.json
```

The first file is the training fit. The second is the active bundle used by
fitted updaters after optional development-only temperature scaling. The
calibration artifact records separate aware and unaware transformations.
Experiment A reruns its fitted analysis against raw and active bundles, retains
case-bound forecast scores and one-vs-rest marginal-class reliability bins,
and records the comparison in both JSONL and CSV. This is simulator reliability
evidence; it is distinct from the LLM calibration path.
Sensitivity runs retain the corresponding raw/active/calibration records inside
each row of `models/sensitivity-fits.jsonl`.

For any configured `llm_*` updater, the runner can fit a separate temperature
for each updater view from disjoint development-user responses before A/B/C
test execution. It retains:

```text
models/llm-calibration.json
llm/development-raw-responses.jsonl
llm/development-requests.jsonl       # when prompts are retained
metrics/llm-development-calibration.jsonl
llm/test-raw-responses.jsonl         # temperature mode
llm/responses.jsonl                  # active calibrated responses
```

`llm.calibration = "none"` is an explicit uncalibrated ablation. The
calibration artifact records `test_labels_used = false`; actual calibration
parameters and results still require an imported or live response corpus.

## Native-state and decoder-scoring state

When event retention is enabled, Experiment B and endogenous Experiment C
trajectory artifacts serialize complete native state before and after each turn
and at the terminal point; fixed-history Experiment C replay artifacts retain
the complete terminal state. Without event retention those full-state
artifacts are intentionally absent, and Gate 4's state-retention criterion
fails. Native state includes base belief, all episodes, semantic claims,
persona belief, persona text, and its content-addressed state ID.

Every native terminal state is projected through both fixed blinded views when
evaluation is requested. Decoder rows omit updater identity and latent truth
from the decoder payload, then the evaluator scores the two deterministic
belief projections using latent truth outside the decoder boundary. These
views test representation sensitivity; they are not statistically independent
decoders.

Accordingly, an ordinary Experiment B run leaves Gate 4 incomplete even if both
projections are present and agree. Its action rows are transparent
structured/persona references, not native end-to-end executions. Passing the
computational gate requires separately imported, reviewed decoder evidence and
recorded native-system actions bound to the exact terminal suite.

Experiment B also writes a separate external-decoder packet containing blinded
development/test requests, researcher-only truth labels, a researcher codebook,
and a design manifest. The analysis path refuses test-label calibration: one
temperature is fitted per external decoder family from development rows, then
raw/calibrated performance, ECE/reliability bins, and source agreement are
reported for held-out test rows. No external judgment corpus is checked in.

In the ordinary Experiment C runner, the top-level ranking fields for a native
system are the arithmetic mean of its two deterministic blinded projection
scores and use
`score_basis = "mean_of_two_blinded_native_decoders"`. The public persona
projection is retained separately as `system_projection_score`; individual
projection predictions remain nested. Structured systems use their public
projection directly.

The append-only external-decoder rescore is a different artifact boundary. It
changes only the declared native score fields, uses the arithmetic mean of
exactly two development-calibrated external decoder families, and records
`mean_of_exactly_two_calibrated_external_decoder_families` in `score_basis`.
It preserves non-native rows and the completed source run.

## Verification

The universal standard-library offline check is:

```bash
make check
```

It compiles the Python source and tests, runs the runtime doctor and complete
offline unittest suite, validates every checked-in TOML configuration, and
executes the static mixed-effects contract validator. It does not restore R
dependencies or fit a model.

The current 373-test executed snapshot covers core schemas,
context/provenance separation,
response and welfare separation, exact and fitted inference, population splits,
matched anchors, Experiment A controls and H1/H2/H7 estimands, H7 exact-byte
provider evidence, H8 comparison, closed-loop/shadow behavior, Experiment B
power, fixed-history identity, Experiment C external rescoring and multi-seed
review, native decoders/actions, Gate 4 and Gate 6 review transactions, LLM
replay and provider request construction, physical-attempt accounting,
concurrency locks, crash recovery, schema export parity, held-out suites,
confirmatory statistics, correction-debt pairing, inferential ranking,
artifact freezing, universal A/B/C/sensitivity live preflight, bounded pilot
configuration counts, sensitivity construction/phase inference, and end-to-end
Experiment A, B, and C artifacts.

During this reconciliation, maintainers also ran Ruff, `git diff --check`,
relative-Markdown-target resolution, and a tracked-content credential scan.
These are release-audit checks rather than hidden steps inside `make check`;
none invokes a live provider.

Tests validate implementation contracts. They do not constitute an empirical
study, a successful scientific gate, provider evaluation, independent decoder
dataset, human judgment dataset, mixed-effects analysis, or paper conclusion.

## Completion boundary

The M1–M5 software infrastructure milestone is complete in the current tree:
the deterministic scientific core, Experiments A–C, native-memory diagnostics,
external exchange/admission paths, optional mixed-effects harness, schemas,
examples, CI, and artifact verification are implemented and tested.

For a candidate release, the software completion criteria are:

- `make check` passes without network access;
- `doctor` reports a supported runtime;
- the offline smoke run completes and verifies;
- every retained run has an explicit config/source/split identity;
- component information boundaries remain enforced;
- provider and human dependencies are labeled as external;
- generated gates retain `claim_status = "not_claimed"`; and
- no placeholder, smoke, or pilot value is presented as a paper result.

This milestone does not complete the study. Provider-scale collection,
responsible decoder-source review, native actions, eligible human evidence,
confirmatory fits on preregistered runs, paper-frozen artifacts, authorship,
repository/venue metadata, and DOI/archive registration remain external or
result-producing work.
