# Data

This directory is for small, reviewable, redistributable inputs and manifests.
It is not the default destination for generated runs, external-model raw data, or
human participant records.

## Suitable tracked contents

- tiny synthetic fixtures used by offline tests;
- versioned train/development/test split manifests;
- option, scenario, dialogue, and paraphrase template manifests;
- terminal diagnostic battery manifests;
- schema examples;
- data-generation metadata and checksums.

All tracked data must have a documented source, schema version, and license or
project-generated status.

## Generated synthetic data

Latent populations and trajectories are deterministically generated from a
configuration, semantic IDs, and seed. Large generated outputs belong under the
ignored `runs/` tree. A curated subset may move to `artifacts/` only with a
manifest and checksums.

Synthetic does not automatically mean harmless: natural-language templates can
still contain third-party text or encode stereotypes. Review new fixtures before
release.

## External LLM records

Request and response JSONL normally live in the corresponding run’s `llm/`
directory. Live adaptive recovery journals deliberately live outside the
checksummed run, by default under
`<output-root>/.llm-journals/<run-id>/<model-role>/`, so a failed attempt can
reuse completed provider calls. Do not move those journals into `data/` merely
to make them visible.

The only checked-in provider declaration is
[`model-suites/openai-gpt-5.6.json`](model-suites/openai-gpt-5.6.json): GPT-5.6
Sol/medium for the primary writer, GPT-5.6 Terra/medium for replication, and
GPT-5.6 Luna/low for decoder/pilot work. It is configuration metadata, not a
live-result dataset. No live provider execution was performed to create the
checked-in repository.

Credentials must remain environment-only. [`.env.example`](../.env.example)
contains an empty variable name template; never commit a populated `.env`,
Authorization header, or API key. Live commands retain provider audit metadata
and replay-compatible beliefs, not the credential.

The executor permits only the official `https://api.openai.com` origin by
default. A custom HTTPS endpoint requires the separate
`allow_custom_base_url = true` or `--allow-custom-base-url` opt-in, which means
the configured environment credential will be sent to that reviewed endpoint
during live execution. Record the exact endpoint and security review in release
provenance; do not mistake HTTPS alone for provider identity.

Temperature-calibrated A/B/C runs retain a deliberate raw/derived boundary in
their run artifact:

```text
models/llm-calibration.json
llm/development-raw-responses.jsonl
metrics/llm-development-calibration.jsonl
llm/test-raw-responses.jsonl
llm/responses.jsonl
```

The fitted temperature is view-specific and uses declared development users
only. Development metrics contain raw and calibrated Brier scores;
`llm/test-raw-responses.jsonl` preserves uncalibrated provider values, and
`llm/responses.jsonl` contains the calibrated values used at test/runtime. Keep
these files together when publishing derived results so reviewers can verify
that no test labels entered fitting. They belong in a verified run or curated
artifact, not as loose files under `data/`.

Before releasing any provider-derived record, check:

- provider redistribution terms;
- removal of credentials and account metadata;
- direct/indirect personal data;
- exact model and prompt identity;
- exact provider endpoint and whether custom-endpoint opt-in was enabled;
- declared request/token ceilings and execution dates;
- retry, refusal, parse, and partial-run outcomes;
- failure and missing-response records.

Parsed beliefs alone may be releasable when raw provider output is not; document
that limitation. Two decoder variants from one provider may count as distinct
model labels in a mechanical source audit, but they do not prove statistically
independent judgment.

## Human participant data

Do not commit participant responses, platform identifiers, contact details,
free-text comments, or researcher codebooks by default.

A human-data release requires:

- appropriate ethics review or documented determination;
- consent compatible with release;
- de-identification and re-identification review;
- data minimization and retention plan;
- access/license terms;
- a data statement and codebook;
- documentation of exclusions and transformations.

The `human-study generate` command creates study materials only:

```bash
PYTHONPATH=src python -m cape_loop human-study generate OUTPUT_DIR
```

Its output is not approval to collect or publish responses.

## Proposed layout

When manifests are added, use:

```text
data/
├── README.md
├── fixtures/
├── manifests/
│   ├── splits/
│   ├── templates/
│   └── terminal-batteries/
└── licenses/
```

Avoid opaque binary or executable serialization. Prefer canonical JSON/JSONL and
include SHA-256 checksums for frozen manifests.

## Adding data

In a pull request, state:

- who or what created the data;
- whether it is synthetic, external-model, or human;
- collection/generation configuration;
- schema and semantic meaning;
- applicable license/terms;
- privacy review;
- expected consumers;
- reproducibility and checksum information.

See [Data model](../docs/data-model.md),
[Ethics and limitations](../docs/ethics-and-limitations.md), and
[Reproducibility](../REPRODUCIBILITY.md).
