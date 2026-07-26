# Component reference

This page describes the executable components in `src/cape_loop`. The central
boundary is:

```text
profile
  → policy provenance + visible context
  → simulated observation
  → permitted updater view
  → profile/native-state update
  → evaluator-only metrics
```

The complete causal chain is retained, but only the simulator and evaluator may
read latent preference and susceptibility.

## Canonical records

Module: `schemas.py`

| Record | Responsibility |
| --- | --- |
| `LatentUser` | Evaluator/simulator-only fixed preference `theta` and presentation susceptibility |
| `Option` | Stable ID, three intrinsic features, label, and domain |
| `InteractionContext` | Displayed options, exact ranking, default, suggestion, wording, question, scenario, turn, and target attribute |
| `PolicyProvenance` | Policy ID/version, profile snapshot, semantic seed, applied mechanism/profile-conditioning flag, and optional config digest |
| `Observation` | Selected option, optional constrained surface response, and choice-noise key |
| `ProfileUpdate` | Structured belief snapshots, native state ID/persona snapshots, and written delta |
| `InteractionRecord` | Context → provenance → observation → optional update |
| `TrajectoryRecord` | Ordered interaction records; excludes latent truth by design |

Context and provenance are distinct dataclasses. A default can affect the user
and belongs in context. The profile snapshot that caused an action is internal
policy provenance and is never embedded in context.

See [Data model](data-model.md) for field-level records.

## Belief state

Modules: `beliefs.py`, `inference.py`

The preference support is:

```text
{-2, -1, +1, +2}³
```

`PreferenceBelief` stores the complete 64-state joint over preference.
`MarginalPreferenceBelief` is a three-row projection. `JointThetaPsiBelief`
stores the theta-major joint over preference and a finite susceptibility
support.

The exact action-aware updater retains `JointThetaPsiBelief` in
`UpdaterState.joint_belief`; the public `belief` is required to equal its theta
marginal. Experiment A exposes both updater and exact-reference joint states in
memory; its runner deduplicates retained exact references into
`events/experiment-a-exact-references.jsonl` and links updater rows by
`exact_reference_id`. Closed-loop records retain evaluated joint state when
present and the exact shadow joint before/after each turn and at termination.
Experiment C replay states serialize the same updater state.

Exact inference enumerates and normalizes the declared finite likelihood. It is
an oracle under the simulator assumptions, not a claim about real users.

## Domains

Module: `domains.py`

Each `DomainSpec` has three attributes, eight full-pool options, and six isolated
directional options.

| Domain | Attribute | Negative / positive |
| --- | --- | --- |
| Travel | `price` | budget / premium |
| Travel | `setting` | central / comfort |
| Travel | `planning` | convenience / flexibility |
| Writing | `length` | concise / detailed |
| Writing | `tone` | formal / conversational |
| Writing | `spelling` | british / american |

Full options cover every direction combination. Isolated options change one
feature and support matched anchors and terminal probes. Domain records contain
no policy, user, or outcome.

## Population and splits

Modules: `population.py`, `splits.py`

The split manifest assigns complete theta groups and complete three-channel
susceptibility groups to train, development, or test. `generate_users` draws
only combinations assigned to the requested split and creates stable IDs/order.

The manifest also assigns named option, dialogue, scenario, and paraphrase
template families. The runner consumes feature-matched but surface-disjoint
atlas/beacon/cedar domain variants for train/development/test, and aborts if
their concrete option IDs, dialogue IDs, or scenario families overlap.
`metrics/split-leakage-audit.json` retains the executed checks. Terminal
identifiers are reserved for test.
`heldout.py` supplies content-addressed development/test surface templates and
rejects a paraphrase family that crosses splits. Experiment A renders the
test-only templates from retained controlled-anchor sources, binds every case
to its source context and selected option, and evaluates the fitted-aware
reference plus `llm_full_context` when that updater is present. A generated
suite is a leakage-checked test instrument; Gate 1 remains incomplete when the
required LLM case pairs are absent.

See [Data splits](data-splits.md) for the exact generator-to-manifest binding.

Initial structured profiles are `correct`, `incorrect`, `uncertain`, and
`empty`. Empty is uniform; uncertain has weak truth-aligned sign mass.

## Semantic randomness

Module: `rng.py`

`semantic_digest`, `semantic_seed`, `uniform`, `gumbel`, and weighted-choice
helpers derive output from a root seed and canonical semantic keys. They do not
use a mutable global generator.

The primary response model uses option-keyed Gumbels. Overlapping options in
paired branches therefore receive the same option noise even if option order or
branch execution order differs. Experiment A's naturally sampled mechanisms
deliberately add the action-context identity to that key, producing independent
draws as required by its estimand; closed-loop counterfactual policy twins
retain common random numbers.

## Response and welfare

Module: `response.py`

`RandomUtilityModel` combines:

```text
beta × intrinsic_utility(theta, option)
+ rank_scale × user.rank_susceptibility
+ default_scale × user.default_susceptibility
+ suggestion_scale × user.suggestion_susceptibility
```

Presentation terms enter simulated choice logits only. `intrinsic_utility` and
`regret` use `theta · option.features` and never use rank, default, or
suggestion.

`RuleBasedResponseModel` is a directly tested noisy maximizing alternative.
Ordinary A–C runs instantiate `RandomUtilityModel`. The sensitivity runner can
cross `random_utility` and `rule_based`, with a separately declared rule-noise
axis, so alternative-family results remain identifiable in every grid row.

## Matched elicitation

Module: `elicitation.py`

`build_matched_anchor_set` constructs exactly four contexts:

```text
balanced
restricted
default
suggested
```

It holds the anchor option fixed. Default and suggestion use the same option set,
ranking, wording, question, scenario, turn, and target as balanced. Restricted
uses a same-direction peer. `eligible` tests the held anchor probability in
every context.

## Policies

Module: `policies.py`

| Policy | Current behavior |
| --- | --- |
| `balanced` | Cycles target attributes and randomizes the order of both directions |
| `soft_profile_conditioned` | Keeps both directions; rotates rank/default/suggestion and probabilistically applies a profile-consistent treatment |
| `exploratory` | Chooses the highest-entropy attribute and presents both directions |
| `fixed_bias` | Uses a fixed negative-direction rank/default independent of the evaluated profile |
| `hard_filter` | Stress condition showing only same-direction alternatives |

Every policy receives a belief, never `LatentUser`. It returns
`PolicyAction(context, provenance)`. Provenance records the profile snapshot even
for policies that do not use it, so independence must be established from the
policy implementation and tests rather than inferred from snapshot absence.

## Fitted likelihoods and calibration

Modules: `fitting.py`, `training.py`, `calibration.py`

The aware conditional-logit model has four coefficients: intrinsic, rank,
default, and suggestion. The unaware signed-semantic model also has four
coefficients while its context features are fixed to zero. This matches
parameter capacity while withholding context.

Both fits use the same randomized training choice records and deterministic Adam
implementation. Their response-space diagnostics have different outcomes:
option identity for aware and semantic sign for unaware.

Temperature calibration uses development records only and fits separate aware
and unaware transformations. The runner retains both the raw training bundle
and the active post-calibration bundle; it never overwrites the raw artifact.

## Updater protocol

Module: `updaters.py`

Every updater declares an `UpdateViewKind`, creates immutable `UpdaterState`,
and returns `UpdateResult`.

| View | Supplied information |
| --- | --- |
| `response_only` | Observation, selected option, and target attribute |
| `full_context` | Response-only fields plus the complete visible context |
| `provenance_aware` | Full context plus policy provenance |

`make_update_view` creates these projections centrally. An updater rejects the
wrong view kind and duplicate event IDs.

Implemented structured updater IDs are:

```text
no_update
exact_action_aware
fitted_action_aware
fitted_action_unaware
response_only
full_context_blind
provenance_discount
provenance_aware
conservative
```

Three `llm_*` IDs use the same state protocol with hash-bound offline responses.
Three native IDs use opaque `NativeMemoryState`.

## Native memory and decoder boundaries

Modules: `native.py`, `decoder_study.py`

`episodic_memory`, `semantic_memory`, and `provenance_linked_memory` retain
content-addressed state. The closed-loop runner preserves complete native state
before/after each turn and at termination.

The episodic adapter replays raw episodes at query time with update strength
`1.25`. The semantic adapters consolidate incrementally with conservative
strength `0.90`; the provenance-linked variant additionally discounts using
explicit causal provenance. These constants are implementation parameters, not
empirical estimates.

`direct_semantic_v1` and `history_evidence_v1` create distinct blinded payloads
with a shared pseudonymous state ID. The decoder boundary rejects system,
updater, memory-kind, truth, and user identifiers, and replaces raw event/source
IDs with decoder-local pseudonyms. Both decoders are deterministic repository
projections, not independent human or model judgments.

Experiment C ranks a native system using the arithmetic mean of the two decoder
scores. Its row records:

```text
score_basis = "mean_of_two_blinded_native_decoders"
system_projection_score
native_decoder_evaluations[2]
```

The mean score has no single predicted action sequence; individual decoder
predictions remain nested. Structured systems use
`score_basis = "system_structured_projection"`.

The two repository decoders are still deterministic representation checks.
They are not counted as independent external judgments. `decoder_study.py`
defines a separate evidence path:

1. build a content-addressed request from a blinded native-state payload;
2. retain truth labels and the trajectory codebook in researcher-only files;
3. import judgments bound to the exact request hash and source metadata;
4. audit per-request coverage, source descriptors, instance IDs, and decoder
   family IDs;
5. fit one temperature per decoder family on development labels only; and
6. report raw and calibrated Brier/NLL/accuracy/ECE, reliability bins, and
   cross-family agreement on held-out test judgments.

Distinct metadata makes a proposed source design auditable, but does not prove
independent errors. The deterministic repository projections are excluded from
this external evidence path.

See [Native memory](native-memory.md).

## First-party distinct external decoder providers

Module: `external_decoder_providers.py`

Gate 4's default external pair is Anthropic `claude-sonnet-5` and Google
`gemini-3.6-flash`. They use separate first-party providers and model families:
that is a stronger source design than two OpenAI variants, but it is not an
automatic claim that their errors are statistically independent.

`ExternalDecoderProviderConfig` locks each credential to the provider's
official HTTPS origin by default. Custom routing needs two deliberate changes:
the custom-origin opt-in and a dedicated, non-default credential variable.
Planning constructs every provider body, content digest, request ID, and
conservative token reservation without reading an environment variable.

Anthropic requests use the Messages API with a strict JSON-schema output
format; Gemini requests use `generateContent` with a JSON response schema.
Both response paths reject missing/inconsistent provider model identities,
malformed or incomplete probability vectors, non-completion states, and
unexpected candidates. The resulting records are ordinary
`ExternalDecoderJudgment` rows accepted by the existing decoder analysis and
Gate 4 importer.

Collection records a durable `started`/`settled` journal around every physical
HTTP attempt, and every retry consumes the per-provider request/token budget.
The dependency-free HTTP transport refuses redirects and reads at most 16 MiB
plus one overflow-detection byte for either success or error bodies. An
oversized body is never retained or reflected; its attempt is charged
conservatively and stopped without a retry.
After a successful response, the redacted provider audit, including its
validated judgment, is flushed before the judgment JSONL row. An exclusive
cross-platform advisory output lock prevents concurrent collectors. Resume
restores each provider's
separate physical-attempt/token ledger, repairs a final crash-truncated JSONL
tail or an interrupted judgment append when safe, and does not repeat an
accepted call. An unresolved attempt or rejected model identity stops automatic
resume for manual review and never becomes a judgment.

See [Gate 4 live collection](gate4-live-collection.md) for commands,
credentials, budgets, and the responsible-researcher review boundary.

## LLM replay

Module: `llm_exchange.py`; adapter: `LLMReplayUpdater`

Requests are information-view-safe and content hashed. Responses are strict
marginal distributions keyed by request ID and repeated prompt hash.
`ReplayProvider.complete` performs local lookup and hash equality; it never
contacts a provider.

The adapter projects observations and contexts onto model-visible semantic
fields: choice-noise, event, context, scenario, turn, user, profile-condition,
policy, and CRN audit labels are absent from response-only/full-context
requests. Request IDs are keyed by prompt hash rather than audit event ID.

The configured response corpus is fingerprinted before run-directory creation.
`llm/input-manifest.json` records its configured path, byte SHA-256, parsed
response count, and model IDs. Run reuse requires that manifest to match the
currently configured corpus exactly.

Before A/B/C test execution, `runner.py` can fit one temperature per configured
LLM updater on disjoint development users. `TemperatureCalibratedProvider`
applies only that updater's locked transformation, while the runner retains
development raw responses, calibration metrics, test raw responses, and active
calibrated responses separately. `llm.calibration = "none"` is an explicit
ablation. The code enforces `test_labels_used = false`; meaningful fitted
temperatures still require real imported or live responses.

Sensitivity rejects all `llm_*` updaters because sequential adaptive requests
cannot be safely reused across changed grid dynamics.

See [LLM exchange](llm-exchange.md).

## Live OpenAI provider and paper suite

Modules: `openai_provider.py`, `evaluation_suite.py`

`OpenAIResponsesProvider` converts the same content-bound `LLMRequest` into a
strict Structured Outputs request. Preparing or planning a request does not
read a credential. An actual call requires explicit live authorization, reads
only the configured environment variable, and reserves that provider
instance's request/token budget before transport. Static and adaptive adapters
append the provider audit before exposing the replay response. The shared
transport refuses redirects and applies the same 16 MiB wire-body ceiling to
successful and error responses.

The returned model must equal the requested model or its dated snapshot. A
missing or different label is charged and retained as a rejected audit, but is
never written as replay input; the journal then requires manual review.

`evaluation_suite.py` treats the checked primary and replication TOML files as
immutable inputs. It verifies their matched design and fixed Sol/Terra roles,
derives different content-addressed run and journal paths, and writes one
identity-locked index. Planning is keyless; `--execute-live` creates a fresh
provider ledger for each role. Terra is explicitly a GPT-5.6
model-variant/tier replication rather than distinct-family robustness.

## Live OpenRouter gateway provider

Module: `openrouter_provider.py`

`OpenRouterChatProvider` converts the provider-neutral `LLMRequest` into a
non-streaming request to OpenRouter's
`/api/v1/chat/completions` endpoint. It is a first-class gateway adapter rather
than an `OpenAIResponsesProvider` custom-base-URL configuration. One canonical
`author/model` slug is mandatory; aliases, route variants, `-latest`, and
`openrouter/auto` are rejected. Changing that one slug is the complete model
switch.

Planning is credential-free. An authorized call reads only
`OPENROUTER_API_KEY` (or the explicitly configured environment-variable name),
reserves the request/token budget, sends strict JSON Schema, disables
OpenRouter response caching, and opts into router metadata. The default route
policy disables fallbacks, requires every request parameter, denies providers
that OpenRouter marks as data collecting, and optionally constrains both
`provider.order` and `provider.only` to one upstream slug. ZDR and attribution
headers are separately configurable.

The parser requires one stopped choice, exact returned-model equality, one
selected upstream endpoint, a direct routing strategy, an upstream model
matching the returned model, no disallowed fallback, no cache hit, and no
material router-pipeline stage. It parses
`choices[0].message.content` as JSON and revalidates all probability vectors
locally. A completed response that fails route/model acceptance is durably
audited but cannot enter replay.

`OpenRouterProviderResult.to_audit_record()` implements
`openrouter-provider-audit.schema.json`. The record distinguishes the gateway
from the observed upstream route with `gateway`, `model_requested`,
`model_returned`, `upstream_provider`, `upstream_model`, `routing_strategy`,
`routing_attempt`, and the additive `routing_metadata`. It also retains
provider/generation IDs when returned, cache status, raw usage, timings and
hashes, a redacted raw response, and the provider-neutral replay response.
Resume revalidates those identities and route-acceptance rules before using an
accepted audit.

The runner places its recovery files under
`.llm-journals/<run-id>/openrouter/<model-role>/` and copies only used records
into the checksummed run's `llm/` directory. The decoder CLI can place multiple
exact models in separate `journals/<model-digest>/` directories and emit one
combined `judgments.jsonl` plus execution manifest.

Every audit fixes `first_party_origin_claimed = false`; decoder manifests also
fix `strict_gate4_eligible = false` and
`statistical_independence_claimed = false`. A selected upstream label does not
turn a shared-gateway call into direct first-party provenance. OpenRouter
profile-writer and reviewed-generic decoder execution are implemented
capabilities, but strict Gate 4 still requires the direct first-party decoder
and native-action collections documented in
[Gate 4 live collection](gate4-live-collection.md).

## Native end-to-end action provider

Module: `native_action_provider.py`

`build_native_action_requests` starts only from a checksum-valid completed
Experiment B run with retained events. It selects the exact Gate 4-eligible
incorrect-seed, soft-profile-conditioned native trajectories, verifies each
terminal native-state ID, and binds that state to the domain's exact retained
held-out terminal suite.

The declared native system,
`cape-loop-openai-native-agent-v1`, sends that complete retained state and suite
to OpenAI `gpt-5.6-sol` at medium reasoning. The model must emit one strict,
item-bound action for every terminal item. The adapter does not derive an action
from a repository belief or persona projection. Returned actions are validated
against item hashes, wording templates, question types, and displayed options
before they become `NativeTerminalActionRecord` rows with
`adapter_kind = "native_end_to_end_recorded"`.

Planning is credential-free. Live collection requires `--execute-live`, the
environment-only `OPENAI_API_KEY`, and the declared hard request/token
ceilings; retries consume physical-attempt budget. The collector binds an exact
collection plan, keeps outputs outside the immutable source run, holds an
exclusive cross-platform advisory output lock, and fsyncs `started`/`settled`
events around every HTTP attempt. The 16 MiB transport ceiling also applies
here; overflow creates a body-free conservative settlement. The provider audit
is flushed before its reusable action row, and
resume can reconstruct an interrupted row from that audit without another
accepted call. An unresolved attempt requires manual review. Gate 4 admits only
the complete six-file collection, not the action file alone; even a validated
collection does not by itself pass Gate 4 or create a paper claim.

## Native and structured evaluation

Modules: `experiments/closed_loop.py`, `experiments/evaluation.py`,
`heldout.py`

The closed-loop runner:

1. asks a policy for context/provenance;
2. evaluates per-attribute counterfactual action influence relative to the
   initial marginal and counts it only while wrong mass remains strengthened;
3. samples the fixed latent user;
4. updates the evaluated system;
5. updates the exact same-history shadow;
6. records structured joint/marginal and native state;
7. computes evaluator-only turn diagnostics; and
8. retains an audit `TrajectoryRecord`.

The common B/C terminal-ranking battery is constructed without a system or policy
argument by wrapping `heldout-terminal-v2`. It forbids defaults/suggestions,
has a content digest, and preserves all four terminal question types. Its
option IDs, feature vectors, wording-template IDs, and scenario-family IDs are
checked against training-domain material. Both the public system projection and
both native decoder projections can be scored against exactly that battery.

Experiment B additionally exercises the explicit terminal-action contract over
the same v2 suite. An action must repeat the exact item digest, question type,
and wording ID. The current structured and native adapters are transparent
reference projections from the public belief or native persona belief. They
exercise the action contract, but are not a claim that an opaque external
system answered natural-language terminal prompts end to end.

## Metrics, statistics, gates, and power

Modules: `metrics.py`, `statistics.py`, `gates.py`, `power.py`

Implemented metrics include marginal Brier, joint NLL, marginal KL/L1,
action-conditioned update error, update direction, entropy/information,
false/laundered confidence, selection cost, attribution cost, and the five-part
self-confirmation predicate.

Experiment A statistics include directional log-odds oracle-update slopes,
fitted evidence-strength ordering, raw/calibrated forecast and reliability
comparisons, user-clustered paired bootstrap contrasts and interactions,
marginal OLS with user-clustered CR1 covariance, Holm correction, and paired
cluster pilot-power simulation. The CR1 model is a dependency-free marginal
robustness analysis. It is explicitly **not** the proposal's
user-random-slope/scenario-random-intercept generalized mixed-effects model,
which is implemented separately by the version-pinned optional
[R mixed-effects harness](mixed-effects-analysis.md). No fitted R result is
checked in.

Experiment A also crosses an explicit prior-concentration factor. Each level is
a declared mixture of the uniform joint prior and truth-aligned mass; the same
prior is supplied to every updater/mechanism in a matched cell, and natural
responses share the same semantic noise key across prior strata. The primary
marginal OLS includes numeric `prior_strength` when more than one level is
configured. A content-addressed control battery fixes all three positive and
three negative protocols from the proposal, but labels them as protocol-only
until their direct-statement, longitudinal, correction, indifference, or
randomized-response executors produce observations.

Experiment B statistics apply deterministic percentile bootstraps to
complete-trajectory paired estimands. Primary intervals resample equally
weighted latent-user clusters; paired-trajectory intervals are retained only
as a sensitivity view. The artifact covers evidence selection, same-history
attribution, SCI, LCG, five-clause profile rate, and later-action-influence
rate, and marks fewer than eight user clusters inadequate. This analysis is
also explicitly not a mixed-effects model or GLMM.

For temperature-calibrated LLM updaters in B/C, `llm_outcomes.py` performs a
strictly local paired diagnostic: it recovers the terminal content-addressed
request from the audit record and scores the cached raw and calibrated vectors
on the common battery. It performs no provider call. A multi-turn raw vector is
conditional on the calibrated active history, so the artifact requires a full
recursive counterfactual rerun before any raw-trajectory, raw-ranking, or
raw-gate claim.

Evaluation statistics include ranks with ties, Kendall tau-b, bootstrap rank
intervals, pairwise reversal/tie probabilities, paired open/closed/test
difference intervals, joint open-versus-closed difference-of-differences, and
interval-supported partial-order tiers. Evaluation selection regret uses the
complete inferential development top tiers and reports the mean, minimum,
maximum, and paired-test interval envelope over every open-top-tier ×
closed-top-tier pair; it does not break an unresolved comparison by numeric
tolerance or system ID.

Experiment C aligns every system row by the stable user/domain/replicate design
key, rejects duplicates or incomplete layouts, and reduces all domains and
trajectory replicates to one value per complete latent-user cluster before
paired resampling. Its ranking output records that independent unit and the
effective development/test cluster counts.

Power helpers also retain Benjamini–Hochberg utilities for legacy analyses;
Experiment A's wired confirmatory family uses Holm correction.

Gate reports have `computed_status` and a separate, fixed
`claim_status = "not_claimed"`. Missing evidence produces an incomplete
criterion rather than an inferred failure or success.

Experiment B Gates 2/3 target only `llm_full_context`; the response-only and
provenance-aware LLM variants remain controls. Gate 2 requires adequate
user-clustered LCG and five-clause-rate intervals above zero; Gate 3 requires
an adequate user-clustered same-history attribution interval above zero.
Missing or inadequate interval evidence cannot computationally pass. Gate 4
requires all seven serialized terminal `NativeMemoryState` fields, imported
blind judgments from at least two genuinely distinct and independently
reviewed decoder sources, and genuine hash-bound native end-to-end actions.
The two deterministic repository projections and persona/reference actions
are retained but explicitly cannot satisfy those external criteria.
The run remains immutable after completion. `gate_review.py` and
`gate-review import-native` bind a validated five-file selected external
decoder collection and six-file native-action collection to a verified
completed Experiment B run, recompute Gate 4, and write a separate
checksum-bound review. Plans, durable attempts, accepted audits, terminal
judgments/actions, manifests, origins/models, approved ceilings, and every
evidence digest are checked under shared collection locks. An explicit
reviewed-generic decoder mode remains available without provider-collection
provenance. Ordinary Experiment B gate reports remain incomplete until that
external import exists.

## Runner, artifacts, reporting, and schemas

Modules: `runner.py`, `artifacts.py`, `release.py`, `gate_review.py`,
`reporting.py`, `schema_export.py`

`runner.py` validates the experiment contract, prepares fits/splits, dispatches
the runner, writes exact filenames, captures failures, and finalizes checksums.

`RunArtifacts` constrains write paths to the run directory. `verify_run` rejects
malformed digests, absolute/escaping paths, duplicate checksum paths, missing or
modified listed files, and files present in the run but absent from
`SHA256SUMS`. It also requires complete manifest status, matching directory/run
ID, a current-schema resolved configuration whose digest matches the manifest,
and a summary file. A retained TOML config is reparsed and must resolve to the
same digest; otherwise the manifest must carry a descriptor and digest for its
programmatic config origin.

`freeze_run` first requires a checksum-valid completed run, then writes a
deterministic uncompressed tar with normalized ownership, modes, and timestamps.
Its adjacent manifest binds the archive digest, source run ID, source manifest,
source checksum file, and member count. `verify_frozen_artifact` checks that
sidecar, archive digest, safe unique member paths, regular-file inventory, and
normalized tar metadata, and independently validates the archived TOML/config
digest binding when a TOML source is present.

Reporting writes deterministic CSV and simple SVG. It does not contain a
separate post-hoc report CLI.

Schema export writes the public JSON Schemas under `schemas/` or another
specified destination.

## Human collection and pragmatic analysis

Modules: `human_study.py`, `decoder_study.py`, `verbalization.py`

The human-study component constructs fixed study items, hides condition/source
IDs in participant records, deterministically orders items, and builds a
separate codebook. The collection contract validates de-identified participant
codes, assignment/display bindings, fixed consent and blinding versions,
comprehension status, response time, and 1–7 ratings. Analysis includes only
consented, comprehension-passing rows and reports condition summaries, an
observed evidence-strength ordering, and paired participant bootstrap
contrasts.

These are collection and analysis contracts, not a recruitment or survey
service. A consent-version field does not confer ethics approval. Institutional
review or exemption, approved consent language, recruitment, compensation,
privacy/retention policy, hosting, and actual data collection remain external.

The verbalizer allows a small set of choice acknowledgements and rejects
surface responses that invent unsupported general-preference language.

## Correction debt

Module: `correction_debt.py`

The correction-debt protocol crosses four correction placements with exactly
paired false-seed and equally strong correct-seed arms. Both arms receive the
same explicit correction and balanced-recovery schedule. It retains pair-level
profile, text, behavior, derived-memory, recovery-time, corrective-evidence,
and recovery-AUC measurements before aggregation; an individual turn is never
treated as an independent unit.

Execution requires `stage_gate_authorized=True` or the CLI
`--stage-gate-authorized` acknowledgement. The shipped log-odds adapter is an
inspectable diagnostic reference that tests the protocol. It is not an LLM or
native-memory finding, and the stage-gate acknowledgement is not evidence that
an earlier scientific gate passed.

## Robustness and phase records

Module: `sensitivity.py`

Sensitivity points can vary decision noise, shared presentation strength,
independent rank/default/suggestion multipliers, initial-profile strength,
prior uncertainty, trajectory length, response-model family, and rule noise.
Every row retains the full point identity and its raw/active fitted bundle.

Phase criteria are explicit threshold/relation records. Classification
preserves missing inputs as incomplete, and boundary inference reports only
observed adjacent grid intervals where a criterion changes. Those intervals are
descriptive grid boundaries, not interpolated causal thresholds. A phase target
that falls back to a structured proxy is labeled as such; confirmatory
LLM-phase evidence still requires live external runs.

## Information access

| Component | Latent truth | Context | Provenance | Own profile/state |
| --- | ---: | ---: | ---: | ---: |
| Domain | No | Constructs options | No | No |
| Policy | No | Constructs | Writes | Public belief only |
| Response model | Yes | Reads | No | No |
| Response-only updater | No | Selected option only | No | Yes |
| Full-context updater | No | Reads | No | Yes |
| Provenance-aware updater | No | Reads | Reads | Yes |
| Exact shadow | No | Reads | Not required by likelihood | Joint shadow state |
| Native decoder | No | Only blinded retained evidence | Blinded/omitted | Blinded state view |
| External decoder source | No | Receives only `decoder/external-requests.jsonl` | Omitted | Blinded content payload |
| Native action provider | No | Exact held-out terminal suite | Only provenance inside retained native memory | Exact retained native state |
| Human participant | No latent simulator truth | Receives participant vignette/display ID | Condition/source hidden | No system profile |
| Evaluator | Yes | Reads retained records | Reads retained records | Reads outputs |
| Reporter/verifier | No new truth access | Reads files | Reads files | Reads files |

Artifact event files may contain evaluator-only truth because they support
scientific scoring. They must not be fed back to policies, updaters, decoders,
or external prompts.
