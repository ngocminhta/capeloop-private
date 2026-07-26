# Architecture

CAPE-Loop is organized around information boundaries. The central requirement is
not a particular class hierarchy; it is that the full causal chain remains
inspectable while each component sees only the information appropriate to its
role.

## End-to-end flow

```text
                              evaluator-only
                         ┌──────────────────────┐
                         │ latent user θ, ψ     │
                         └──────────┬───────────┘
                                    │
                                    ▼
┌─────────┐   profile   ┌───────────────┐   C, P   ┌───────────────┐
│ updater │────────────►│ interaction   │─────────►│ response model│
└────▲────┘             │ policy        │          └───────┬───────┘
     │                  └───────────────┘                  │ Y
     │                                                     ▼
     │             ┌─────────────────────────────────────────────┐
     └─────────────│ information-view builder + update recorder  │
                   └──────────────────┬──────────────────────────┘
                                      │
                                      ▼
                   ┌─────────────────────────────────────────────┐
                   │ event store, shadow inference, evaluation   │
                   └─────────────────────────────────────────────┘
```

`C` is the user-visible `InteractionContext`; `P` is separately recorded
`PolicyProvenance`. The response model needs latent truth to simulate a choice.
The updater never does.

## Layer responsibilities

### Schemas

Python record validators and the generated JSON Schemas collectively
define the boundary between generation, inference, external models, and
analysis. Their current coverage includes:

- domain options and feature vectors;
- latent user states;
- visible contexts;
- policy provenance;
- observations;
- structured beliefs;
- native memory snapshots;
- update and trajectory records;
- LLM exchange envelopes;
- direct-provider and OpenRouter gateway audit records;
- experiment metrics and run manifests.

Every external boundary has an explicit version carrier, but not every nested
or experiment-specific dictionary contains a `schema_version` key: some derive
their version from a generated schema `$id`, a containing artifact, or the
software release. [Data model](data-model.md) lists the concrete carriers and
the records that do not yet have standalone JSON Schemas. Incompatible changes
require a new boundary version; no migration tool is currently provided.

### Domain

A domain defines the semantic names and controlled option space for exactly
three primary preference dimensions. It supplies:

- option pools and complete feasible sets;
- feature vectors used by intrinsic utility;
- balanced and same-direction alternatives for anchor construction;
- domain-specific validation.

Policies and experiment builders, not `DomainSpec`, assign scenario/template
identifiers and construct the versioned terminal battery from these options.
The domain does not select an action, simulate a user, or update a profile.

### Semantic random source

Random draws are derived from a global seed plus stable semantic keys. The
random source supplies deterministic uniform or Gumbel values without depending
on call order. This is necessary for common-random-number pairing across policy
branches.

### Response model

The response model maps latent user, visible context, and semantic random key to
choice probabilities and an observation. The primary implementation is finite
multinomial random utility. Robustness models can replace it through the same
interface.

Presentation contributions are kept separate from intrinsic utility so welfare
and regret cannot accidentally reward defaults or persuasion.

### Elicitation constructor

The matched-anchor constructor creates balanced, restricted, defaulted, and
suggested variants. It verifies:

- invariant anchor identity and attributes;
- the expected direction of alternatives;
- valid ranking/default/suggestion references;
- a sufficient response probability in every matched condition.

It is also responsible for producing both controlled identical-response and
naturally sampled examples without conflating their estimands.

### Interaction policy

Policies receive a declared profile view, domain/scenario state, and random
source. They return:

1. a visible interaction context; and
2. a separate provenance record explaining the policy, profile snapshot,
   actually applied mechanism, whether the profile caused it, version, and
   semantic random key.

Implemented policy families include balanced/randomized, softly
profile-conditioned, exploratory, fixed mildly biased, and hard-filter stress
conditions. A policy must not inspect latent truth except in explicitly labeled
diagnostic generation code that is never treated as a deployable condition.

### Belief and inference

The exact posterior retains a normalized joint distribution over preference and
susceptibility states. Attribute marginals are derived views, not the internal
state.

Fitted aware and unaware likelihood models are trained on declared training
records. Their serialized bundle includes the schema version, training-record
count, training seed, and both fitted models; it does not retain the individual
training-record IDs. Calibration is a separate transformation fit only on
development records.

### Profile updater

An updater has an identifier, information-view requirement, initial-state
constructor, update operation, and serializable public report. Its input is
constructed centrally:

- response-only;
- full visible context;
- provenance-aware.

This prevents individual adapters from quietly reading the trajectory store or
latent state. Native updaters may return an opaque memory state, but must also
produce an auditable update record.

### Trajectory runner

The runner owns turn order:

1. obtain current profile state;
2. ask the policy for context and provenance;
3. sample or replay the observation;
4. create the updater’s permitted view;
5. update the evaluated system;
6. update the same-history action-aware shadow;
7. record immutable before/after state and action influence;
8. pass truth only to the evaluator.

Static evaluation uses a generated history once and replays the identical
events to every updater. Endogenous evaluation creates a separate trajectory per
updater because its profile controls future actions.

### Native memory and decoders

Native memory adapters retain episodic events, consolidated semantic/persona
state, or provenance-linked claims. Decoder inputs exclude system identity and
latent truth. Two fixed, distinct deterministic projections plus the terminal
battery are retained so a result cannot depend on a favorable single
conversion. They are sensitivity views, not independent human judgments.

### Evaluator

The evaluator is the sole general consumer of latent truth. It computes
structured scores, welfare, selection/attribution decompositions, ranking
statistics, and five-part self-confirmation labels. It does not alter profiles
or policies.

### Artifact writer and reporter

The artifact writer serializes canonical JSON/JSONL records, environment and
configuration identity, and checksums. During a run, report generation consumes
already-computed retained records and does not alter profiles, resample data, or
call a model. Finalization then writes the completion manifest and checksum
inventory.

## Information-access matrix

| Component | Latent user | Visible context | Policy provenance | Profile | Observation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Domain | No | Constructs options | No | No | No |
| Policy | No | Constructs | Writes | Declared view | No |
| Response model | Yes | Yes | No | No | Writes |
| Response-only updater | No | Selected option only | No | Own state | Yes |
| Full-context updater | No | Yes | No | Own state/history | Yes |
| Provenance-aware updater | No | Yes | Declared metadata | Own state/history | Yes |
| Shadow updater | No | Yes | No | Shadow state | Yes |
| Native decoder | No | If retained in blinded memory | Blinded/declared | Native state | If retained |
| External decoder provider | No | Blinded request payload only | Omitted | Blinded native-state view | No |
| OpenRouter gateway/upstream | No | Only fields in the selected LLM or blinded-decoder view | Only for the provenance-aware LLM view | Request prior or blinded state | Selected observation when included by that view |
| Native action provider | No | Held-out terminal suite | Only fields inside retained native memory | Exact retained native state | No |
| Evaluator | Yes | Yes | Yes | Yes | Yes |
| Reporter | Only retained metric truth fields | Retained records | Retained records | Retained reports | Retained records |

Tests should fail when a component receives a broader object than this table
allows.

## Structured and native tracks

The two tracks share domain, policy, response, trajectory, and evaluator
infrastructure.

```text
                       ┌─► structured updater ─► probability metrics
common event stream ───┤
                       └─► native updater ─────► blinded decoders
                                               └► terminal behavior
```

Structured evaluation supplies direct numerical comparability. Native evaluation
checks that the same failure appears in persistent agent state rather than only
in an artificial probability interface.

## LLM process boundary

External LLM execution is intentionally outside the trusted core:

```text
CAPE-Loop Python request builder
    │ schema-versioned, content-hashed request
    ├────────► direct first-party provider ──► provider audit
    └────────► OpenRouter gateway ──► upstream route + gateway audit
                                            │
                                            ▼
CAPE-Loop validator/importer
    │ canonical replay records
    ▼
ordinary updater/evaluation path
```

The external harness cannot choose its own latent information view. The request
payload is complete and hashed; imported responses must match it.

This is a process boundary, not a claim that any configured evaluation has
already run. The CLI now orchestrates credential-free planning and explicitly
authorized, budgeted live execution for direct OpenAI profile writers,
OpenRouter-routed profile writers and reviewed-generic decoders, direct
Anthropic and Gemini external decoders, and the direct OpenAI-backed native
action system. Provider audits are written before reusable responses,
judgments, or actions. The ordinary simulation/replay core remains network-free
and treats retained, hash-bound records as its canonical input.

The OpenRouter branch has a distinct provenance layer. It requires one exact
model, disables gateway response caching, requests router metadata, and retains
the reported selected upstream provider/model plus routing attempt and
strategy. The validator rejects unexpected model substitution, fallback, cache
hits, or material pipeline transformations before producing replay input. The
gateway audit nevertheless sets `first_party_origin_claimed = false`: an
upstream label reported by OpenRouter is not the same evidence as a request
sent directly to that provider. Multiple routes or models behind one gateway
also do not establish statistically independent errors. Strict Gate 4 therefore
uses only the separate direct first-party collection branches.

The external projection strips audit-only choice-noise and
context/event/scenario/turn identifiers. Model-facing request IDs are derived
from the prompt hash, and policy surface-template IDs are neutral, so hidden
condition and common-random-number labels do not cross the full-context
boundary.

## Failure behavior

Scientific-invalid states should fail loudly:

- non-normalized or non-finite beliefs;
- context references to nonexistent options;
- context/provenance field mixing;
- changed anchor features across matched variants;
- unmatched request and response identifiers;
- missing or inconsistent gateway/upstream model identity;
- disallowed gateway fallback, cache replay, or request/response transformation;
- static-history mismatch across systems;
- use of test IDs in fitting or calibration;
- missing action-influence evidence for a self-confirmation label;
- artifact checksum failure.

The CLI should return a nonzero exit status and leave enough manifest/log
information to diagnose an interrupted run. It should not silently coerce a
scientifically different condition.
