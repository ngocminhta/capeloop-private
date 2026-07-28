# Data directory policy

This directory contains only small, reviewable, redistributable inputs and
model-suite declarations. It is not the default destination for generated
runs, provider responses, or participant data.

The canonical description of dataset generation, storage, formats, splits,
records, and artifacts is [the data model](../docs/data-model.md).

## What belongs here

Suitable tracked content includes:

- tiny synthetic fixtures used by offline tests;
- versioned option, scenario, dialogue, paraphrase, split, or terminal-battery
  manifests;
- public schema examples;
- data-generation metadata and checksums; and
- reviewed provider/model declarations that contain no credentials or model
  outputs.

Every tracked data file must identify its source or project-generated status,
format/schema version, applicable license, and expected consumer.

Generated populations, trajectories, events, models, and metrics belong in the
ignored, content-addressed `runs/<run-id>/` tree. Curated evidence may be copied
or frozen under `artifacts/` only with a manifest, checksums, and provenance
statement.

## Checked-in model declarations

The two current metadata files are:

- [`model-suites/openai-gpt-5.6.json`](model-suites/openai-gpt-5.6.json):
  OpenAI Sol/medium as primary, Terra/medium as model-variant replication, and
  Luna/low for generic decoder/pilot work; and
- [`model-suites/gate4-native-and-distinct-decoders.json`](model-suites/gate4-native-and-distinct-decoders.json):
  OpenAI Sol/medium for native actions and OpenRouter
  `anthropic/claude-sonnet-5` at low effort plus
  `google/gemini-3.6-flash` at minimal effort for the selected decoder pair.

These files are protocol and provenance metadata, not observations. The
selected decoder families share OpenRouter, so the declaration denies
first-party origin, distinct transport origins, and statistical independence.
Direct Anthropic and Gemini adapters remain optional origin replications.

## External provider records

Provider request/response JSONL belongs inside a completed run's `llm/`
directory or in a separate checksum-bound collection. Adaptive recovery
journals stay outside the immutable run, normally under:

```text
<output-root>/.llm-journals/<run-id>/<role>/
<output-root>/.llm-journals/<run-id>/openrouter/<role>/
```

Do not move recovery journals into `data/`. Keep raw/active calibration
records, provider audits, attempt journals, and collection manifests together
so the development/test and retry boundaries remain auditable.

Credentials are environment-only. [`.env.example`](../.env.example) contains
empty variable names; a populated `.env`, authorization header, API key,
billing identifier, or unrelated account metadata must never be committed.
See [Live execution](../docs/live-execution.md) for provider and release rules.

## Human data

Do not commit participant responses, platform identifiers, contact details,
free-text comments, or researcher codebooks by default. A human-data release
requires:

- an appropriate ethics approval, exemption, or determination;
- consent compatible with the intended release;
- data minimization, de-identification, and re-identification review;
- privacy, access, deletion, and retention rules;
- a codebook and transformation/exclusion record; and
- explicit license and release authorization.

`human-study generate` creates blinded study materials only. It does not
recruit, host, compensate, collect, approve, or authorize participant data.
See [Ethics and limitations](../docs/ethics-and-limitations.md).

## Formats and review

Prefer canonical JSON/JSONL and include SHA-256 checksums for frozen manifests.
CSV may be supplied as a derived readable projection, not as the sole
scientific record. Avoid opaque executable or binary serialization.

When adding tracked data, document:

- who or what created it;
- whether it is synthetic, external-model, or human;
- the generation/collection configuration;
- schema and semantic meaning;
- license or provider terms;
- privacy and security review;
- expected consumers; and
- reproducibility and checksum information.

Released runs and provider-derived artifacts must follow
[REPRODUCIBILITY.md](../REPRODUCIBILITY.md) and the
[artifact policy](../artifacts/README.md).
