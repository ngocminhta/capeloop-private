# Native persistent memory

The native track implements three inspectable persistent-state variants. They
run inside the same policy–response–update loop as structured updaters; they do
not call an external LLM.

## State model

Every `NativeMemoryState` is immutable and contains:

```text
memory_kind
base_belief
episodes[]
semantic claims[]
persona_belief
persona_text
state_id
```

`state_id` is the SHA-256 digest of the canonical state contents. Supplying a
nonmatching ID is rejected.

The policy sees only `persona_belief`. It does not receive episodes, semantic
claims, state IDs, provenance details, or latent truth.

## Episode record

A native episode stores:

- event and selected-option IDs;
- target attribute and selected direction;
- displayed directions;
- visible presentation mechanisms;
- numeric evidence weight;
- optional surface response; and
- for provenance-linked memory only, policy ID, profile snapshot, explicit
  presentation mechanism, and whether profile conditioning was applied.

It never stores latent preference or susceptibility.

## Implemented variants

### `episodic_memory`

- Required view: full visible context.
- Retains every episode.
- Does not create semantic claims.
- Replays the complete episode list against the base belief to obtain the
  current persona belief.
- Assigns each event evidence weight `1.0`; it is intentionally not
  provenance-aware. Query-time evidence strength is `1.25`.

### `semantic_memory`

- Required view: full visible context.
- Retains every episode.
- Consolidates a semantic claim for the selected direction.
- Each claim records all supporting event IDs and cumulative evidence weight.
- Updates from the current persona belief with event weight `1.0` and
  conservative consolidation strength `0.90`.

This is the implemented full-context-blind native condition.

### `provenance_linked_memory`

- Required view: provenance aware.
- Retains each episode with policy ID, policy profile snapshot, and explicit
  causal-mechanism fields.
- Creates provenance-linked semantic claims.
- Discounts evidence for restricted exposure, first ranking, default,
  suggestion, and an explicitly profile-conditioned, profile-consistent causal
  mechanism using fixed code-level factors. The causal check uses provenance
  fields, not a policy-ID substring.

This is a diagnostic reference condition, not a proposed production memory
architecture.

All three updaters reject a broader or narrower `UpdateView` than declared and
reject consuming an event ID twice.

## Profile update record

Each native update emits the same `ProfileUpdate` schema as structured systems.
Its native before/after fields contain:

```text
state_id
persona_text
```

The written delta names the triggering event and evidence weight. The updater's
opaque state contains the complete `NativeMemoryState`.

## Full-state retention

When `[artifacts].retain_events = true`, the closed-loop trajectory runner
serializes the complete opaque native state:

- before every turn;
- after every turn; and
- at the terminal point.

Those full snapshots contain base/persona beliefs, all episodes, all claims,
persona text, and content-addressed ID. Experiment B writes them inside
`events/experiment-b-trajectories.jsonl`; endogenous Experiment C writes them
inside `events/experiment-c-endogenous.jsonl` when event retention is enabled.

With event retention enabled, fixed-history Experiment C replay records retain
the complete terminal native state in `events/experiment-c-replays.jsonl`.
Their per-interaction `ProfileUpdate` records retain state ID/persona pairs
rather than a separate full opaque snapshot at each replayed turn.

This distinction matters when auditing an artifact: a content ID proves which
state was referenced only when the corresponding full state is also retained or
available.

## Two blinded decoder views

Every native terminal state has two fixed deterministic decoder views:

| Decoder | Payload |
| --- | --- |
| `direct_semantic_v1` | Persona summary, direction confidence, and semantic claims |
| `history_evidence_v1` | Base prior marginals and the complete episode evidence history |

Both views share a pseudonymous state ID derived from the native state digest.
Their payload validator rejects fields named:

```text
system
system_id
updater
updater_id
memory_kind
latent_truth
truth
user_id
```

Episode IDs and semantic-claim source IDs are also rewritten to decoder-local
`event-NNNN` pseudonyms, so audit strings that contain user, initial-condition,
policy, or updater labels are not exposed merely under an allowed key.

The direct decoder reconstructs a joint belief from sign confidence. The history
decoder replays blinded episodes from prior marginals. These are separate
representations, not prompt paraphrases, and neither calls an LLM.

`decode_native_state` always returns both results. Evaluation does not select
one decoder after observing scores.

## Terminal evaluation

The evaluator—not the decoder—has access to latent truth. It evaluates each
decoded belief on the same per-domain terminal battery used for the system's
public persona belief.

Experiment B retains one row per trajectory and decoder in:

```text
metrics/experiment-b-native-decoders.jsonl
```

Experiment C nests both decoder evaluations in each native system's row in:

```text
metrics/experiment-c.jsonl
```

For Experiment C ranking, a native system's top-level profile error, behavioral
accuracy, cross-context accuracy, and regret are the arithmetic mean of exactly
the two decoder scores. The row marks this with:

```text
score_basis = "mean_of_two_blinded_native_decoders"
```

The system's public persona projection is retained as
`system_projection_score`, and both individual decoder records retain their own
predicted option IDs. The averaged top-level score has no single prediction
sequence, so its `predicted_option_ids` is empty.

Each decoder evaluation records:

- decoder ID;
- pseudonymous state ID;
- profile Brier;
- behavioral accuracy;
- tie-excluded and fractional behavioral accuracy;
- cross-context accuracy;
- mean intrinsic regret;
- predicted terminal option IDs;
- predicted-utility and intrinsic-utility tie counts; and
- evaluated-item count.

The common battery is policy-independent, contains no defaults or suggestions,
and is identified by a content digest.

## What is not implemented

The current native state, consolidation, persona projection, and both local
decoders are deterministic Python implementations. The repository includes an
external LLM-decoder exchange/analysis path and a stage-gated correction-debt
runner, but no checked-in external judgments, production LLM-backed native
memory writer, embedding store, vector database, or production-agent connector.

Accordingly, a native failure would support a claim only about these inspectable
reference loops. No current artifact claims that such a failure was observed.
