# Architecture

CAPE-Loop is organized around causal and information boundaries. Its central
requirement is not a particular class hierarchy: the complete chain from an
agent action to a user response and later profile update must remain auditable,
while each runtime component receives only the information required for its
declared role.

This guide is the canonical overview of the implemented system. See
[Scientific design](scientific-design.md) for the formal claim,
[Data model](data-model.md) for retained records, [Configuration](configuration.md)
for executable settings, and [Implementation status](implementation-status.md)
for evidence that still depends on external collection.

## End-to-end flow

```text
 profile ──► interaction policy ──► [visible context C | provenance P]
                                              │
 evaluator-only latent user θ, ψ ─────────────┤
                                              ▼
                             mathematical response model ──► selected option Y
                                              │                         │
                                              └────────────┬────────────┘
                                                           ▼
                                             frozen conversation bank
                                                           │
                                                assistant + user turns
                                                           ▼
                                             declared information view
                                                           │
                             ┌─────────────────────────────┴─────────────┐
                             ▼                                           ▼
                   evaluated profile writer                  same-history shadow
                             │
                             ▼
                     event store and evaluator
```

`C` is the user-visible `InteractionContext`. `P` is the separately recorded
`PolicyProvenance` explaining why that context was produced. The response model
needs latent truth to simulate a choice; policies and updaters never receive
it. Only after the option is fixed does the renderer select a reviewed
per-scenario template and produce the natural two-turn exchange. Rendering is
deterministic and cannot alter the option. The evaluator may read truth only
after the action, observation, and update have been fixed.

The trajectory runner owns the order:

1. obtain the evaluated system's current public profile;
2. ask the policy for visible context and separate provenance;
3. mathematically sample or replay the selected option;
4. render the exact assistant presentation and local user choice from the
   frozen conversation bank;
5. construct exactly the updater's declared information view;
6. update the evaluated structured or native state;
7. update the same-history action-aware shadow;
8. retain the structured choice, rendered dialogue, state, and event linkage;
   and
9. pass the retained record and latent truth to evaluation.

Static evaluation generates a history once and replays the identical events to
every updater. Endogenous evaluation creates one trajectory per updater because
the updater's profile can change later policy actions.

## Design invariants

The implementation enforces five cross-cutting invariants:

- **Truth isolation.** Only the simulator and evaluator read latent preference
  or presentation susceptibility.
- **Context/provenance separation.** What the user saw is distinct from the
  internal reason the policy showed it.
- **Choice/language separation.** The response model fixes the option before
  language rendering; a conversation template cannot choose, explain, or
  change that option.
- **Declared views.** Response-only, full-context, and provenance-aware inputs
  are constructed centrally rather than by individual updaters.
- **Natural model views.** Evaluated LLMs receive readable descriptions and a
  semantic attribute codebook, never numeric feature vectors or the target
  index used by the evaluator.
- **Semantic pairing.** Random values derive from a root seed and stable
  semantic keys, not execution order.
- **Evidence before claims.** A software capability, request plan, smoke run,
  or valid checksum is not a paper result or a passed stage gate.

## Component contracts

### Canonical records

The Python records in `schemas.py` and exported JSON Schemas define boundaries
between generation, inference, external execution, and analysis.

| Record | Responsibility |
| --- | --- |
| `LatentUser` | Fixed evaluator/simulator-only preference and presentation susceptibility |
| `Option` | Stable option ID, label, domain, and three intrinsic features |
| `InteractionContext` | Task prompt, displayed options, ranking, default, suggestion, wording, scenario, turn, and target |
| `PolicyProvenance` | Policy/version, profile snapshot, semantic key, applied mechanism, and profile-conditioning flag |
| `Observation` | Selected option, exact assistant/user surface, surface ID, and choice-noise key |
| `ProfileUpdate` | Structured belief or native-state before/after references plus written delta |
| `InteractionRecord` | Context, provenance, observation, and optional update |
| `TrajectoryRecord` | Ordered audit events; latent truth is excluded by design |

A default can affect the simulated user and therefore belongs in visible
context. The profile snapshot that caused a policy action is internal
provenance and is never smuggled into context. Every public boundary has an
explicit version carrier, although some nested records inherit the version of
their containing artifact.

### Domain, population, and split construction

A `DomainSpec` supplies a controlled option space with exactly three primary
preference dimensions, intrinsic feature vectors, complete feasible pools,
balanced alternatives, same-direction alternatives, and validation. The
implemented travel and writing domains each provide full direction-combination
options plus isolated one-feature options for anchors and terminal probes.

Population generation follows a split manifest that assigns complete preference
and susceptibility groups to train, development, or test. Named option,
dialogue, scenario, and paraphrase families are also split. Concrete overlap is
checked at execution, and the result is retained in the split-leakage audit.
Terminal identifiers are test-only.

Initial structured profiles are `correct`, `incorrect`, `uncertain`, and
`empty`. Empty is uniform; uncertain is weakly aligned with truth. These labels
describe initialized state, not evidence about a real person.

### Semantic randomness, response, and welfare

`rng.py` derives uniform, Gumbel, and weighted-choice draws from the global seed
plus canonical semantic keys. Overlapping options in paired branches therefore
receive common option noise even if branches execute in a different order.
Experiments that require independent draws add the action-context identity to
the key explicitly.

The primary response model is finite multinomial random utility:

```text
beta × intrinsic_utility(theta, option)
+ rank_scale × rank_susceptibility
+ default_scale × default_susceptibility
+ suggestion_scale × suggestion_susceptibility
```

Presentation terms affect simulated choice logits only. Welfare and regret use
intrinsic utility, `theta · option.features`, so the benchmark never rewards a
system merely for exploiting a default, rank, or suggestion. A directly tested
rule-based response family supports robustness analysis through the same
interface.

### Frozen hybrid conversation surfaces

`[scenarios] conversation_file` loads a `ConversationTemplateBank`. Each
scenario has four neutral display names and one neutral base presentation.
Balanced, restricted, and ranking all use that same base wording; only the
visible option pair or order changes. Code derives the default form by
inserting a fixed default sentence and the suggested form by inserting a fixed
suggestion sentence. It also fixes the user template to
`I choose {selected_name}.`

The neutral authoring inputs are generated once, outside experimental
execution, with:

```bash
cape-loop conversations generate-openrouter \
  data/scenarios/scenario-catalog-v1.json \
  data/scenarios/conversation-templates-v1.json \
  --model anthropic/claude-sonnet-5 --execute-live
```

The transient OpenRouter response contains only `display_names` and a neutral
`base_template`. The base uses `{prompt}`, both visible name placeholders, and
both visible description placeholders exactly once, then ends in a question;
it contains no treatment placeholder. Code expands it into the five
core-compatible stored `presentation_templates`. The three neutral forms are
identical before option and order substitution. For the default and suggested
forms, code inserts the corresponding fixed treatment sentence immediately
after `{prompt}`. The stored `choice_template` is always
`I choose {selected_name}.`

The authoring call may improve neutral fluency but cannot see latent users,
assign a choice, write treatment wording, vary the user reply, or run the
evaluated profile-writing task. A run reuses the frozen bank rather than asking
the authoring model to rewrite every trial.

The evaluated writer is a separate model call. Its model-facing projection
uses readable option descriptions and a domain-specific codebook such as
“`-2` strongly favors lower-cost; `+2` strongly favors higher-cost.” It omits
internal feature vectors, target indices, split labels, and randomness keys.
Full-context and provenance-aware views receive the exact assistant/user
dialogue; response-only receives the user reply and selected readable option as
an intentional information ablation.

### Matched elicitation and policies

The elicitation constructor creates balanced, restricted, defaulted, and
suggested contexts while holding the anchor option fixed. It validates option
identity, feature invariance, treatment references, alternative direction, and
anchor eligibility.

Policies receive a declared public profile view, domain/scenario state, and the
semantic random source. They return both visible context and provenance.
Implemented families cover balanced/randomized, softly profile-conditioned,
exploratory, fixed mildly biased, and hard-filter stress conditions. A policy
must not inspect latent truth except in explicitly labeled diagnostic
generation that is never treated as a deployable condition.

### Belief, fitting, calibration, and updater protocol

Structured preference support is:

```text
{-2, -1, +1, +2}³
```

`PreferenceBelief` stores the normalized 64-state joint distribution;
attribute marginals are projections. The exact action-aware reference can also
retain a theta-by-susceptibility joint distribution. Exact inference enumerates
the declared finite likelihood and is an oracle only under simulator
assumptions.

The fitted aware and unaware likelihoods are capacity matched. Both are trained
on the same declared training records; calibration is a separate temperature
transformation fitted on development records only. Raw and calibrated bundles
remain distinct, and test labels cannot inform either stage.

Every updater declares an `UpdateViewKind`, constructs immutable state, and
returns an auditable update:

| View | Supplied information |
| --- | --- |
| `response_only` | Observation and selected option; internal mathematical references also receive the scoring coordinate |
| `full_context` | Response-only fields plus complete visible context |
| `provenance_aware` | Full context plus policy provenance |

The central view builder rejects the wrong view kind and duplicate events.
Structured, LLM-replay, and native-memory adapters all use this protocol. The
external-LLM projection is narrower than the internal Python view: it removes
the scoring coordinate and numeric features before constructing any model
request.

### Evaluation, statistics, and artifacts

The evaluator is the sole general consumer of latent truth. It computes
structured error, welfare, selection/attribution decompositions, ranking
statistics, and the multi-clause false-self-confirmation predicate. It cannot
alter profiles or policies.

Experiment-level statistics use declared pairing and cluster units. The
dependency-free Python analyses provide clustered contrasts, bootstrap
intervals, reliability and ranking summaries, gate diagnostics, and bounded
power utilities. The optional version-pinned R harness implements the
confirmatory mixed-effects formulas; its directory contains protocol and
software, not a fitted result.

`RunArtifacts` writes canonical JSON/JSONL, resolved configuration,
environment identity, manifests, and checksums. Verification rejects malformed
or escaping paths, symbolic links, duplicate entries, missing or unexpected
files, checksum mismatch, incomplete manifests, and inconsistent configuration
identity. Reporting consumes retained records and cannot resample users, change
state, or call a model. A verified run can be frozen into a deterministic tar
whose sidecar binds the source run and archive digest.

Experiments A–C also project narrow analysis rows during the same runner
execution:

```text
evaluated in-memory record
  ├── full event/metric record for reconstruction
  ├── compact analysis row for ordinary statistics
  └── deduplicated conversation trace
          └── deterministic Markdown preview
```

Neither projection branch calls a simulator or provider or creates a second
observation. A writes one updater×trial analysis row but groups conversation
evaluations under one trace per trial. B flattens each retained trajectory for
analysis while keeping the same trajectory as one multi-turn conversation
record. C writes one analysis row per evaluation but stores a fixed history
once across its replayed updaters. Sensitivity keeps its existing aggregate
analysis and uses the B-style trace with a sensitivity-point condition.

The conversation JSONL is exhaustive. It excludes latent truth, feature
vectors, posterior arrays, and native-memory payloads, so size grows with
unique natural exchanges and scalar evaluations rather than reconstruction
state. The companion Markdown deterministically chooses a diverse preview of
at most 100 trace records by default and states exact complete
record/turn/outcome counts plus readable metric labels and interpretation
guidance. All outputs are finalized in the same run and covered by the same
`SHA256SUMS`.

An immutable historical run cannot acquire new files without invalidating its
inventory. The compact-artifact adapter therefore verifies the source and
writes a separate `analysis-rows.jsonl`, manifest, and checksum inventory
bound to that source. This derived directory is not a full run and cannot
promote the source's evidence status.

## Information-access matrix

| Component | Latent user | Visible context | Policy provenance | Profile/state | Observation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Domain | No | Constructs options | No | No | No |
| Policy | No | Constructs | Writes | Declared public view | No |
| Response model | Yes | Reads | No | No | Writes |
| Conversation renderer | No | Reads readable fields | Treatment only | No | Reads fixed choice; writes language |
| Response-only updater | No | Selected option only | No | Own state | Reads |
| Full-context updater | No | Reads rendered dialogue | No | Own state/history | Reads |
| Provenance-aware updater | No | Reads rendered dialogue | Reads declared metadata | Own state/history | Reads |
| Exact shadow | No | Reads | Not required by likelihood | Joint shadow state | Reads |
| Native decoder | No | Blinded retained evidence only | Blinded/omitted | Blinded native view | If retained in view |
| External decoder source | No | Blinded request payload only | Omitted | Blinded native view | No |
| OpenRouter gateway/upstream | No | Model-facing selected view only | Only when that view permits it | Request prior or blinded state | Only when that view permits it |
| Native action provider | No | Exact held-out terminal suite | Only fields retained inside native state | Exact terminal native state | No |
| Human participant | No simulator truth | Participant vignette/display | Condition/source hidden | No system profile | Writes rating |
| Evaluator | Yes | Reads retained records | Reads retained records | Reads outputs | Reads |
| Reporter/verifier | No new truth access | Reads files | Reads files | Reads files | Reads files |

Evaluator event files may contain truth for scoring. They must never be fed
back to a policy, updater, decoder, participant prompt, or external provider.
Tests are expected to fail when a component receives a broader object than this
matrix allows.

## Structured and native tracks

The tracks share domains, policies, response simulation, trajectory execution,
terminal batteries, and evaluation:

```text
                       ┌─► structured updater ─► probability metrics
common event stream ───┤
                       └─► native updater ─────► blinded decoders
                                               └► terminal actions
```

Structured evaluation gives direct numerical comparability. Native evaluation
checks whether the same behavior appears in inspectable persistent state rather
than only in an artificial probability interface.

### Native memory state

Every immutable `NativeMemoryState` contains:

```text
memory_kind
base_belief
episodes[]
semantic claims[]
persona_belief
persona_text
state_id
```

`state_id` is the SHA-256 digest of canonical state contents. The policy sees
only `persona_belief`; it does not receive episodes, semantic claims, state
IDs, provenance details, or latent truth.

| Variant | Required view | State behavior |
| --- | --- | --- |
| `episodic_memory` | Full context | Retains events and replays them from the base belief at query time with strength `1.25` |
| `semantic_memory` | Full context | Consolidates selected-direction claims incrementally with strength `0.90` |
| `provenance_linked_memory` | Provenance aware | Retains explicit causal fields and discounts evidence using applied mechanisms and profile conditioning |

The constants are implementation parameters, not fitted empirical estimates.
Every episode links the selected direction to visible mechanisms and an
evidence weight; only the provenance-linked variant adds internal policy and
causal fields. No variant stores latent preference or susceptibility.

When event retention is enabled, endogenous trajectories retain complete native
state before and after each turn and at termination. Fixed-history replay
retains the complete terminal state. A state digest is useful evidence only
when the referenced full state remains available.

### Two blinded deterministic decoders

Every native terminal state is projected through both fixed decoders:

| Decoder | Blinded payload |
| --- | --- |
| `direct_semantic_v1` | Persona summary, direction confidence, and semantic claims |
| `history_evidence_v1` | Base prior marginals and complete episode evidence history |

Both use a pseudonymous state ID. The boundary rejects system/updater IDs,
memory kind, user ID, and latent-truth fields; event and claim-source IDs are
rewritten to decoder-local pseudonyms. The direct decoder reconstructs a joint
belief from semantic confidence, while the history decoder replays blinded
episodes. Neither calls an LLM.

Evaluation always retains both outputs. Native Experiment C ranking uses the
arithmetic mean of their scores and records
`score_basis = "mean_of_two_blinded_native_decoders"` plus the public system
projection and both nested decoder evaluations. The averaged score has no
single action sequence.

These decoders are deterministic representation checks, not independent human
or model judgments. External decoder evidence follows a separate
content-addressed path with blinded requests, researcher-only truth/codebooks,
complete source audits, development-only calibration, and held-out scoring.

### Native end-to-end actions

The repository also implements a separate provider adapter for model-mediated
terminal actions. It starts from a checksum-valid completed Experiment B run,
verifies each retained native state, and binds it to the exact held-out terminal
suite. The declared `cape-loop-openai-native-agent-v1` sends that state and
suite directly to OpenAI `gpt-5.6-sol` at medium reasoning. It must return one
item-bound action per terminal case.

This adapter does not rewrite memory and does not infer actions from the public
persona projection. A complete collection can be imported as Gate 4 evidence;
the local transparent action projection cannot substitute for it.

## Live-provider process boundary

Network execution is outside the trusted simulation core:

```text
credential-free whole-design preflight
    │ exact request allocation fits reviewed ceilings
    ▼
content-addressed request builder
    ├────────► direct provider ─────────► attempt + provider audit
    └────────► OpenRouter gateway ──────► route + gateway audit
                                             │
                                             ▼
strict validator/importer
    │ canonical replay response or external judgment/action
    ▼
ordinary updater and evaluation path
```

The external harness cannot choose its latent-information view. Request content
and output schema are fixed and hashed before dispatch. Planning is
credential-free; a live call requires an explicit execution flag and reads
only the configured environment variable. Every physical attempt is journaled
as `started` and `settled`, charged against request/token ceilings, and audited
before a reusable response, judgment, or action is appended.

Available routes are:

| Route | Implemented use | Provenance boundary |
| --- | --- | --- |
| Offline replay | Structured LLM writers and retained provider outputs | No network; request and prompt hashes must match |
| Direct OpenAI | Profile writers and the native terminal-action system | Official origin by default; exact returned-model validation |
| OpenRouter | Switchable profile writers and selected Claude/Gemini decoder families | Shared gateway; reported upstream metadata is retained but is not first-party attestation |
| Optional direct Anthropic/Gemini | Decoder-family origin replications | Separate evidence mode; not required for the selected workflow |

### Selected OpenRouter decoder collection

The selected external pair is:

| Model | Reasoning | Gate 4 requests | Gate 4 conservative tokens | Experiment C requests | Experiment C conservative tokens |
| --- | --- | ---: | ---: | ---: | ---: |
| `anthropic/claude-sonnet-5` | `low` | 640 | 4,469,348 | 360 | 2,709,130 |
| `google/gemini-3.6-flash` | `minimal` | 640 | 4,201,828 | 360 | 2,558,650 |

Each source has an approved hard ceiling of 900 physical attempts, 6,000,000
conservative tokens, 1,024 maximum output tokens per request, and zero
automatic retries in the selected plans. Gate 4's corresponding direct-OpenAI
native-action plan contains 80 requests and reserves 2,396,907 conservative
tokens.

Claude and Gemini are distinct model families but both calls traverse
OpenRouter. Every selected collection therefore fixes:

```text
provenance_mode = "selected_openrouter_gateway_collection"
first_party_origin_claimed = false
distinct_transport_origins = false
statistical_independence_claimed = false
```

The collector uses a separate locked journal per model and one combined
judgment manifest. It disables cache reuse, disallows fallback, requires
supported parameters and exact canonical model identity, requests router
metadata, and validates the selected upstream route. An OpenRouter-reported
upstream name or model snapshot never upgrades gateway evidence to direct
origin. Admission requires the complete collection and a responsible
researcher review of shared dependencies; loose judgment rows are insufficient.

The structured-output schema is adapted to the selected route's supported
JSON Schema subset. In particular, Anthropic requests omit numeric `minimum`
and `maximum` keywords rejected by the observed Amazon Bedrock route. The wire
schema still describes the inclusive `[0,1]` requirement, and acceptance
remains fail-closed because local parsing independently requires finite
in-range probabilities and a sum-to-one vector for every attribute.

The optional direct adapters call Anthropic `claude-sonnet-5` and Google
`gemini-3.6-flash` through their own official APIs. They retain distinct
provenance and are optional origin replications, not silently interchangeable
with the selected shared-gateway corpus.

### Transport and recovery behavior

Direct and gateway transports refuse redirects, bound response bodies, validate
strict structured output locally, and reject missing or substituted model
identity. A charged response that fails validation is retained as rejected
audit evidence and never enters replay.

Output-level advisory locks prevent concurrent commands from duplicating paid
calls. Resume reconstructs a safely interrupted append from a settled accepted
audit when possible. An unresolved physical attempt, inconsistent identity, or
ambiguous crash window stops automatic resume for manual review. Exact
operations and import commands live in [Live execution](live-execution.md).

## Repository layout

The layout is deliberately summarized by responsibility rather than maintained
as a volatile file-by-file catalog:

```text
src/cape_loop/                 core records, simulation, providers, evaluation
└── experiments/               Experiment A/B/C estimands and runners
configs/
├── smoke.toml                 fast offline validation
├── offline/                   deterministic studies and source packets
└── live/                      budget-bounded provider pilots
data/                          synthetic-data policy, manifests, model suites
schemas/                       deterministic public interchange schemas
analysis/confirmatory-mixed-effects/
                               optional version-pinned R analysis contract
tests/                         deterministic offline unit and integration tests
docs/                          scientific, operational, and status references
examples/                      small instructional programs, never evidence
runs/                          ignored local output; each run has conversations/
artifacts/                     curated checksum-verified release bundles
paper/                         manuscript, figure, and table provenance
```

The Python package separates records/domain generation, policies and response
models, inference/updaters, provider transports, experiment runners,
statistics/gates, and artifact/release code. `runner.py` is the orchestration
boundary; it is not a second implementation of those components.

Checked-in configurations are executable declarations, not preregistrations or
results. `data/model-suites/` declares provider roles, not model output.
Generated schemas must be regenerated from the exporter rather than edited by
hand. The optional R directory contains its own locked environment because it
is outside the dependency-free Python core. CI runs offline and never reads API
credentials.

## Extension points

Extensions must preserve the same information and evidence boundaries:

| Extension | Required contract |
| --- | --- |
| Domain | Controlled features/options, intrinsic-utility meaning, split metadata, anchors, and system-independent terminal items |
| Response model | Finite normalized probabilities, semantic-key sampling, serialized identity, and welfare/presentation separation |
| Policy | Declared public-profile input plus separate valid context/provenance output; no latent truth |
| Structured updater | One declared central information view, normalized output, retained before/after record, and split-safe fitting/calibration |
| Native memory | Versioned content-addressed state, source-linked deltas, public persona projection, blinded decoder compatibility, and terminal action interface |
| LLM/provider adapter | Content-hashed request, strict output, environment-only secret, attempt/audit journal, hard budgets, offline replay, and explicit origin semantics |
| Metric/statistic | Versioned formula, analysis unit, missing/tie rules, aggregation/resampling unit, and hand-checkable fixture |
| Experiment | Predeclared factorial cells, estimand/controls, pairing, splits, metrics/gates, incomplete-cell retention, a declared compact-row unit that does not change sample size, and the common artifact writer |
| Schema | Explicit compatibility decision, deterministic export, consumer updates, and version tests |

Choose stable IDs and versions, reject unknown configuration, retain component
identity in manifests, and add both invalid-boundary tests and an end-to-end
smoke. Tests must encode invariants, never a desired empirical conclusion.

The contributor checklist and type-specific test expectations are in
[Contributing](../CONTRIBUTING.md#extending-cape-loop).

## Failure behavior

Scientifically invalid states fail closed:

- non-normalized or non-finite beliefs or model probabilities;
- context references to nonexistent options;
- context/provenance field mixing or latent-truth leakage;
- changed anchor features across matched variants;
- static-history mismatch across compared systems;
- use of test IDs in fitting or calibration;
- unmatched request/response identifiers or prompt hashes;
- missing or inconsistent provider, gateway, or upstream model identity;
- disallowed fallback, cache replay, redirect, or material transformation;
- exceeded request, token, output, or response-body limits;
- unresolved live-attempt journals or unsafe concurrent output access;
- missing action-influence evidence for a self-confirmation label;
- incomplete external evidence presented as a gate result;
- artifact checksum, path-safety, or configuration-identity failure.

The CLI returns a nonzero status and retains enough failure or journal evidence
to diagnose an interrupted operation. It never silently coerces a different
scientific condition. The exact executable/evidence boundary is maintained in
[Implementation status](implementation-status.md).
