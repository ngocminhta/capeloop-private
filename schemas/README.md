# Public interchange schemas

These JSON Schemas are deterministic exports of
`cape_loop.schema_export.SCHEMAS`. Regenerate them with:

```bash
PYTHONPATH=src python -m cape_loop schema export schemas
```

The schemas describe release-facing records, while Python dataclasses enforce
additional cross-field invariants such as ranking permutations, displayed-option
references, and probability normalization. JSON Schema validation alone is
therefore necessary but not sufficient for a scientifically valid run.

The directory covers the core run and LLM records plus public held-out,
external-decoder, provider-audit, human-collection, and Gate 4 review
contracts. Gate 4 publishes strict schemas for recorded native actions,
responsible-researcher decoder-source review, and the immutable review object.
Other experiment-specific event/metric rows, beliefs, native memory states,
split manifests, and model records do not yet have standalone exported schemas.
Experiment A therefore uses an explicitly documented normalized artifact pair:
updater rows in `events/experiment-a.jsonl` join by `exact_reference_id` to one
full exact state per trial in
`events/experiment-a-exact-references.jsonl`.

Version placement is record-specific. User state, LLM request/response, and run
manifest carry `schema_version: 1`. Interaction, trajectory, and human-rating
records do not; their v1 boundary is the schema `$id`. Nested options, contexts,
provenance, observations, profile updates, and beliefs are not recursively
given a `schema_version`.

Schema IDs are stable `urn:cape-loop:schema:…:v1` identifiers. A breaking
record change requires a new version boundary rather than silently modifying
released artifacts. See [the concrete data model](../docs/data-model.md) for
field names, array ordering, runner wrappers, manifests, and retention rules.

Every checked-in schema validates as a standalone document. In particular,
`trajectory.schema.json` bundles the interaction-record contract under
`$defs`, so a validator does not need a custom resolver for the CAPE URNs.
The separate `interaction-record.schema.json` remains available when validating
individual interaction rows.
