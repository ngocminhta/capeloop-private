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
external-decoder, provider-audit, human-collection, H8 model-evidence, and Gate 4 review
contracts. Gate 4 publishes strict schemas for recorded native actions,
responsible-researcher decoder-source review, the immutable review object, and
the audit-first Anthropic/Gemini decoder and OpenAI native-action provider
journals. The Gate 4 review input schema always requires all six native
collection evidence entries. Its decoder branch requires either all four
selected-provider sidecars alongside `decoder_judgments`, or none of those
sidecars for the explicitly reviewed-generic mode.
Experiment C additionally publishes a researcher-only native-state/metric-row
codebook contract and a calibrated external-family terminal-score contract.
The runtime requires exactly two distinct family/source rows per native state,
development-only calibration, exact source-run/row/state/battery hashes, and an
unchanged non-native row set before producing a separate immutable reranking
artifact. See
[Experiment C external-decoder rescore](../docs/experiments.md#experiment-c-external-decoder-rescore).
Other experiment-specific event/metric rows, beliefs, native memory states,
split manifests, and model records do not yet have standalone exported schemas.
Experiment A therefore uses an explicitly documented normalized artifact pair:
updater rows in `events/experiment-a.jsonl` join by `exact_reference_id` to one
full exact state per trial in
`events/experiment-a-exact-references.jsonl`.

`openrouter-provider-audit.schema.json` is the public v1 contract for one
OpenRouter Chat Completions audit row. It keeps `provider`/`gateway` fixed to
`openrouter`, separates requested and returned model labels from the reported
`upstream_provider`/`upstream_model`, and retains routing strategy, routing
attempt, additive router metadata, generation/cache identifiers, usage,
timings, hashes, redacted raw response, and provider-neutral replay response.
Its `first_party_origin_claimed` field is always `false`. Schema validity
therefore demonstrates record shape, not direct first-party origin, upstream
authentication, statistical independence, or Gate 4 admission. A single audit
row is never sufficient evidence. The selected shared-gateway path additionally
requires the complete two-model collection, exact coverage validation, and a
hash-bound responsible-researcher source review.

`llm-provider-transport-attempt.schema.json` is the crash-safe physical-call
contract shared by direct OpenAI and OpenRouter execution. A `started` event is
fsynced before dispatch; its paired `settled` event records conservative token
accounting and embeds any final accepted/rejected provider audit. An unresolved
start or a settled sequence without a final audit requires manual billing
review before another call.

`human-model-evidence.schema.json` is the strict held-out exchange for H8. Each
row binds one nonnegative update-toward-the-claim measurement to its source
run, source record, and source-artifact SHA-256. The only admitted measurement
is the positive part of the anchor-directional log-odds update, so an update
away from the preference claim cannot be relabeled as positive evidence.

The four `h7-volunteered-*` schemas cover the direct-statement workflow. The
request binding joins one model-visible prompt to its withheld source user and
case; the collection plan binds the exhaustive case/role crossing to a
verified Experiment A run; the evidence record binds an accepted OpenAI or
OpenRouter response to its directional log-odds update; and the review records
the immutable source components plus the recomputed volunteered and overall H7
criteria. Runtime validation additionally requires exact corpus coverage,
same-provider/model pairing, source-run checksum verification, and
`claim_status = "not_claimed"`.

Version placement is record-specific. User state, LLM request/response, and run
manifest plus provider-audit sidecars carry `schema_version: 1`. Interaction,
trajectory, and human-rating records do not; their v1 boundary is the schema
`$id`. Nested options, contexts, provenance, observations, profile updates, and
beliefs are not recursively given a `schema_version`.

Schema IDs are stable `urn:cape-loop:schema:…:v1` identifiers. A breaking
record change requires a new version boundary rather than silently modifying
released artifacts. See [the concrete data model](../docs/data-model.md) for
field names, array ordering, runner wrappers, manifests, and retention rules.

Every checked-in schema validates as a standalone document. In particular,
`trajectory.schema.json` bundles the interaction-record contract under
`$defs`, so a validator does not need a custom resolver for the CAPE URNs.
The separate `interaction-record.schema.json` remains available when validating
individual interaction rows.
