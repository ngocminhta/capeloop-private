# Implementation status

CAPE-Loop is an alpha research implementation. This page distinguishes
executable software from empirical evidence. No checked-in file claims that a
paper hypothesis or scientific stage gate has passed.

## Status vocabulary

- **Implemented:** executable in the current source tree and covered by an
  offline test or end-to-end runner test.
- **Provider-capable:** the code can construct and execute a live request, but
  no external response or result is checked in.
- **External-evidence-dependent:** the protocol, validation, and analysis code
  exists, while collection requires credentials, independent sources, human
  participants, or another outside input.
- **Diagnostic reference:** an inspectable adapter exercises the research
  protocol but is not evidence about an LLM or native system.
- **Not implemented:** the proposal requirement is absent from the executable
  path.

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
| Experiment A runner | Implemented | Controlled/naturally sampled modes, executable prior-strength strata, fixed six-case control protocol, oracle slopes, evidence ordering, raw/calibrated reliability, clustered contrasts/interactions, CR1 marginal OLS, Holm, and pilot power; control outcomes and result claims remain absent |
| Experiment B runner and same-history shadow | Implemented | Four initial-profile conditions and configured policy/updater crossing; calibrated LLM runs retain cached same-request raw/calibrated terminal scores without claiming a recursively raw trajectory |
| Confirmatory mixed-effects harness | Implemented optional analysis | Version-pinned R/lme4 pipeline implements both proposal formulas, maximal random effects, planned emmeans contrasts, pointwise intervals with Holm-adjusted p-values, canonical config/source validation, curvature-scaled gradient diagnostics, and checksum-bound JSON/CSV outputs; B is reconstructed one row per retained turn and no fit or claim is checked in |
| Experiment C runner | Implemented | Fixed balanced, fixed biased, and endogenous regimes; stable-key alignment and complete-user clustered ranking bootstrap; cached raw/calibrated terminal diagnostics do not replace rankings |
| Held-out paraphrase suite and Gate 1 adapter | Implemented | Test-only response families are content bound and leakage checked; completeness still depends on required updater/case pairs |
| Held-out terminal v2 suite | Implemented | Shared B/C diagnostic uses new option IDs/features/scenarios/wording and all four question types; B also retains explicit action bindings |
| B/C terminal calibration output | Implemented | Every profile score retains top-label ECE/reliability with one preference attribute as forecast unit; pooled artifacts preserve trajectory/user dependence |
| Sensitivity runner | Implemented | Independent mechanism, prior-uncertainty, trajectory, model-family/rule-noise axes plus explicit phase points and observed-grid boundary intervals |
| Native episodic/semantic/provenance-linked memory | Implemented | Content-addressed full state and policy-facing persona belief; episodic query-time inference and conservative semantic consolidation use distinct declared transition strengths |
| Two blinded native projections | Implemented | Direct-semantic and history-evidence deterministic views are both retained and evaluated; they are not independent judgments |
| External decoder exchange and analysis | External-evidence-dependent | Blinded requests, source audit, development-only per-family temperature calibration, test reliability/performance, and agreement are implemented; real judgments are absent |
| Common terminal projection battery | Implemented | Built from heldout-terminal-v2 and shared by every B/C comparison; decisions are still computed from projected beliefs |
| Joint-paired ranking and inferential-tier ESR | Implemented | Complete-user open/closed difference and difference-of-differences intervals determine credible reversals and top tiers; paired test intervals yield a conservative ESR envelope |
| Gate-report machinery | Implemented diagnostic | `claim_status` remains `not_claimed` |
| Run manifests, source digest, and checksums | Implemented | Verification also checks complete status, run ID, resolved-config digest/schema, and summary presence |
| Deterministic paper-artifact freeze/verify | Implemented | Requires a verified run plus retained TOML source or hash-bound programmatic config origin; writes normalized `.tar` plus a digest sidecar |
| Public JSON Schema export | Implemented | CLI writes deterministic schema files |
| LLM profile writers | Provider-capable | Strict replay plus explicit direct OpenAI Responses API or first-class OpenRouter Chat Completions execution, budgets, audit-first journals, and resumability; no live study result is included |
| OpenAI primary/replication suite orchestration | Provider-capable | Credential-free planning and explicit two-role execution use immutable matched configs, distinct run/journal paths, per-role ledgers, and a combined index; Terra is not distinct-family robustness |
| OpenRouter gateway execution | Provider-capable | Exact canonical model slugs, strict JSON Schema, cache disablement, router-metadata and selected-upstream validation, route/privacy controls, gateway/upstream audit schema, static/adaptive execution, and multi-model decoder collection are implemented; every record denies first-party provenance and decoder manifests deny strict Gate 4 eligibility and statistical independence |
| LLM probability calibration | External-evidence-dependent | Per-updater temperature is fitted on development users and locked for test; raw/active responses are retained; `none` is an explicit ablation |
| Human pragmatic-study packet, import, and analysis | External-evidence-dependent | Fixed vignettes, blinding/codebook, consent/comprehension metadata validation, summaries, and paired bootstrap contrasts; no participants were recruited |
| Correction-debt protocol | Diagnostic reference | Stage-gated exact paired arms and recovery/debt metrics are implemented with a transparent log-odds adapter |
| Gate 4 external-evidence ingestion | External-evidence-dependent | `gate-review import-native` binds a verified completed B run to the complete selected Anthropic/Gemini and OpenAI collections, validates plans/journals/audits/manifests and every digest under collection locks, or explicitly labels a reviewed-generic decoder alternative; writes a separate checksum-bound review and never mutates the run |

## Known missing work

| Capability | Current status |
| --- | --- |
| Live model evidence | No API keys, paid calls, provider responses, or live model results are checked in |
| Complete Gate 1 evidence | Requires the declared full-context updater and complete held-out case pairs; structured-only smoke runs remain incomplete |
| Executed Experiment A control outcomes | The six-case positive/negative protocol is frozen, but direct-statement, longitudinal, indifference, randomized-response, and real correction outcomes require their declared executors/evidence; the repository does not impute them from anchor choices |
| Executed confirmatory mixed-effects result | The optional R harness is implemented, but no verified study run has been fitted and no model result or claim is checked in; current one-level Experiment A pilots lack the required `prior_strength` variation and are `not_estimable`, while ordinary multi-turn B runs supply within-trajectory `turn` variation |
| Native end-to-end natural-language terminal evidence | The OpenAI provider path now implements keyless planning, explicit live authorization, origin/budget locks, durable physical-attempt journals, resumable audit-first collection, and source-safe outputs; no paid call or empirical native-action record is checked in, and transparent projections remain ineligible |
| Genuinely distinct decoder evidence | Requests and analysis are ready; no external model/human judgments are included, and model/version or OpenRouter upstream-route metadata alone cannot prove independent errors |
| Empirical LLM calibration result | Per-updater development-only temperature calibration is executable for replay and live responses, with `none` as an explicit ablation; no response corpus/result is included |
| Human study execution | Ethics/IRB or exemption, approved consent, recruitment, compensation, survey hosting, privacy/retention policy, and collection remain external |
| Native/LLM correction-debt result | Protocol and reference adapter exist; a real system adapter and prerequisite gate review are still required |
| Confirmatory LLM phase diagram | Phase code and the broader simulator grid exist; current structured fallback rows are labeled proxies and cannot substitute for live LLM sweeps |
| Paper-frozen artifacts and empirical results | Not included |
| DOI, accepted-paper metadata, or named paper authors | Not included |

Sensitivity configs reject `llm_*` updaters because changed dynamics would
change the adaptive request sequence and create an uncontrolled cost surface.
They also require `experiment.turns = 1`; the grid's executed lengths come only
from `sensitivity.trajectory_lengths`. Phase results name whether the target is
the declared LLM or a structured proxy.

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
decoder-study plan-openrouter REQUESTS.jsonl [--additional-model MODEL]
decoder-study execute-openrouter REQUESTS.jsonl OUTPUT_DIR \
  [--additional-model MODEL] --execute-live
decoder-study plan-distinct REQUESTS.jsonl [--output PLAN.json]
decoder-study execute-distinct REQUESTS.jsonl OUTPUT_DIR --execute-live
native-action plan-openai RUN_DIR [--output PLAN.json]
native-action execute-openai RUN_DIR OUTPUT_DIR --execute-live
gate-review import-native RUN REQUESTS JUDGMENTS TRUTH NATIVE_COLLECTION \
  SOURCE_REVIEW OUTPUT \
  (--external-collection-dir DIR | --allow-reviewed-generic-decoders)
gate-review verify REVIEW_DIR
human-study generate OUTPUT_DIR [--assignment-id ID] [--seed INTEGER]
human-study analyze RESPONSES.jsonl CODEBOOK.json
correction-debt run OUTPUT.json --stage-gate-authorized
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
| Gate 2 — nontrivial soft self-confirmation | Experiment B targets only `llm_full_context`; response-only/provenance-aware LLM variants are controls. Computational passage also requires at least eight user clusters and 95% user-clustered intervals above zero for mean LCG and five-clause profile rate. The checked-in config lacks this updater, so the report is incomplete |
| Gate 3 — attribution beyond evidence selection | Same `llm_full_context` boundary as Gate 2; computational passage requires an adequate 95% user-clustered same-history attribution interval above zero. No passage is claimed |
| Gate 4 — native-system validity | Experiment B restricts to incorrect-seed, soft-profile-conditioned native trajectories with counter-profile alternatives and checks retained state plus the matched native failure; imported independent blinded decoders and genuine native end-to-end actions are mandatory incomplete prerequisites. Strict provenance accepts only the complete direct first-party Anthropic/Gemini decoder and OpenAI native-action collections; deterministic projections, reference actions, and OpenRouter shared-gateway collections are ineligible |
| Gate 5 — evaluation implication | Experiment C uses joint paired complete-user open/closed error differences for reversals and interval-supported top tiers plus paired test envelopes for ESR; descriptive tau, rank bands, and reversal probabilities cannot pass it alone |
| Gate 6 — robustness | Sensitivity checks declared-grid completion and retains phase classifications/boundaries, but live-LLM confirmation and scientific review remain external |

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

## Native-state state

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

For Experiment C, the top-level ranking fields for a native system are the
arithmetic mean of exactly those two decoder scores and use
`score_basis = "mean_of_two_blinded_native_decoders"`. The public persona
projection is retained separately as `system_projection_score`; individual
decoder predictions remain nested. Structured systems use their public
projection directly.

## Verification

The current offline checks are:

```bash
make check
```

They cover core schemas, context/provenance separation, response and welfare
separation, exact and fitted inference, population splits, matched anchors,
closed-loop/shadow behavior, fixed-history identity, metrics, native decoders,
LLM replay/provider request construction without network calls, schema export,
held-out suites, confirmatory statistics, external decoder/human import
contracts, correction-debt pairing, inferential ranking, artifact freezing,
sensitivity construction/phase inference, and end-to-end Experiment A and B
artifacts.

Tests validate implementation contracts. They do not constitute an empirical
study, a successful scientific gate, provider evaluation, independent decoder
dataset, human judgment dataset, mixed-effects analysis, or paper conclusion.
