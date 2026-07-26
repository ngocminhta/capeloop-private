# Extending CAPE-Loop

Extensions are welcome when they preserve causal provenance, information
boundaries, replayability, and honest result status. Start with a small
deterministic fixture and add the full experiment only after its invariants are
tested.

## General checklist

For every new component:

1. choose a stable ID and version;
2. document inputs, outputs, and latent-truth access;
3. use schema-versioned JSON-compatible state;
4. derive randomness from semantic keys;
5. add configuration validation and reject unknown values;
6. add unit and end-to-end smoke tests;
7. retain component identity in the run manifest;
8. update [Components](components.md),
   [Repository map](repository-map.md), and
   [Implementation status](implementation-status.md);
9. avoid encoding a desired empirical conclusion into tests.

## Add a domain

A complete research-domain extension requires coordinated changes to
`DomainSpec`, policy/elicitation builders, terminal-battery construction, and
split metadata. Together those changes supply:

- three controlled preference dimensions or a clearly versioned generalization;
- option features and complete feasible pools;
- intrinsic-utility interpretation;
- balanced and same-direction alternatives;
- anchor candidates;
- scenario, dialogue, and paraphrase template families;
- versioned exogenous terminal diagnostic items;
- split metadata.

Tests must establish:

- feature order and names are stable;
- presentation fields do not enter intrinsic features;
- matched anchors preserve identity;
- target dimensions are actually distinguished by alternatives;
- terminal items are system-independent, and any claimed held-out content is
  disjoint from training content by an enforced split;
- verbalizations do not add unsupported preference claims.

A domain with a different dimensionality requires metric/config/schema review,
not just another registry entry.

## Add a response model

Implement probability and sampling operations from latent user and visible
context. The model must expose its parameters and serialize its identity.

Tests should cover:

- finite normalized probabilities;
- zero probability for unavailable options;
- deterministic semantic-key sampling;
- expected behavior when presentation scales are zero;
- strict welfare/presentation separation;
- anchor eligibility;
- broad parameter edges.

Do not give a response model policy provenance unless provenance is genuinely
visible to the user and represented in the context.

## Add a policy

A policy receives current declared profile state and scenario/domain inputs, and
returns both context and separate provenance.

Document whether it is:

- balanced/randomized;
- softly profile-conditioned;
- exploratory;
- fixed independent logging;
- hard-filter stress;
- another declared mechanism.

Tests must show that it cannot read latent truth, records the exact profile
snapshot used, produces valid option references, and obeys its availability
claims. A supposedly fixed policy must generate identical histories for every
evaluated updater.

## Add a structured updater

Declare one information view:

- `response_only`;
- `full_context`;
- `provenance_aware`.

The central view builder, rather than the updater, should construct its input.
Return normalized attribute distributions and an auditable before/after record.

If the updater is fitted, retain fit IDs and enforce split restrictions. If it
is calibrated, preserve raw output and fit calibration on development records
only.

## Add an external LLM writer

Use `LLMRequest.build` and provider-neutral response JSONL. Do not add a
provider-specific client to the standard-library core.

An external adapter should:

- preserve request ID and prompt hash;
- record exact provider/model/decoding metadata in a sidecar manifest;
- return the strict response schema;
- retain failure/retry outcomes;
- never include credentials;
- support offline replay.

Validate with:

```bash
PYTHONPATH=src python -m cape_loop llm validate responses.jsonl
```

See [LLM exchange](llm-exchange.md).

## Add a native memory adapter

Define:

- native state kind/version;
- update view;
- before/after serialization;
- written delta and source-event links;
- policy-facing persona/profile projection;
- two blinded decoder views or compatibility with existing decoders;
- terminal behavioral interface.

The adapter must record evidence that a state change affected a later action if
it is used in self-confirmation analysis.

## Add a metric

A metric needs:

- stable name and version;
- mathematical definition;
- level: turn, trajectory, user, pair, or run;
- truth-access requirement;
- raw/calibrated applicability;
- missing/tie/boundary handling;
- aggregation and resampling unit;
- deterministic fixture with hand-checkable output.

Update [Metrics](metrics.md). Never change an existing formula without changing
its version.

## Add an experiment

An experiment runner must enumerate its expected factorial cells before
execution and retain incomplete/failed cells. It declares:

- estimand and controls;
- population/splits;
- policies/updaters/response models;
- pairing/randomness;
- terminal battery;
- metrics and gates;
- stage-gated interpretation.

Use the common artifact writer. A custom script that emits only a summary CSV is
not a reproducible experiment implementation.

## Add or change schemas

Export current schemas:

```bash
PYTHONPATH=src python -m cape_loop schema export schemas
```

For incompatible changes:

1. increment the schema version;
2. retain or document migration behavior;
3. add old-version rejection/compatibility tests;
4. update example records and exchange consumers;
5. regenerate exported schemas deterministically.

Do not reuse a field name with new semantics under the same version.

## Contribution verification

Before opening a pull request:

```bash
make check
PYTHONPATH=src python -m cape_loop config validate configs/smoke.toml
PYTHONPATH=src python -m cape_loop run configs/smoke.toml \
  --output-root /tmp/cape-loop-contribution-check
```

Remove or retain local output outside the repository as appropriate; do not
commit an ad hoc run. Complete the scientific-invariant checklist in the pull
request template.
