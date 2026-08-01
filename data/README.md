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
- small frozen model-assisted language assets with explicit provenance and
  review status;
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

## Canonical scenario catalog

[`scenarios/scenario-catalog-v1.json`](scenarios/scenario-catalog-v1.json) is
the canonical interaction-stimulus input used by every checked-in
configuration. Version 1.5.0 contains 48 unique split-disjoint families:
six train, six development, and 36 test scenarios, covering both domains
and all three attributes. Each test domain×attribute cell has six families,
enough for a 16-turn three-attribute cyclic or v2 block-balanced exploratory
trajectory without reuse. A custom adaptive policy that can select one
attribute repeatedly has a larger, policy-specific capacity requirement.
Version 1.5.0 preserves the earlier outcome-blind semantic revisions and
nuisance counterbalancing, and adds a declared `target_half_span`. Every test
domain×attribute cell uses spans `0.10`, `0.16`, `0.24`, `0.34`, `0.46`, and
`0.56`; train/development use `0.50`. This gives the prospective manipulation
planner a subtle-to-pronounced difficulty grid. No
evaluated-model output or experiment result was consulted. Runs copy and
checksum-bind the exact consumed bytes; generated interactions and responses
still belong under `runs/`, not here.

The catalog is deliberately `frozen-development` and
`simulation-and-pilot-only`. All 48 scenarios are provisional, none has
completed the required human surface and scientific reviews, and none is
paper-eligible. Its project-authored synthetic provenance is explicit in the
file. The
normative good-scenario and review policy is in
[Scientific design](../docs/scientific-design.md#scenario-catalog-and-quality-policy);
the exact fields, coverage, and run artifacts are in
[Data model](../docs/data-model.md#scenario-catalog-input). The independently
generated held-out terminal-v2 batteries remain separate from this catalog.

## Frozen conversation bank

[`scenarios/conversation-templates-v1.json`](scenarios/conversation-templates-v1.json)
contains one small natural-language template family for each of the 48
scenarios. The current visible bases were outcome-blind project-standardized
onto three source-neutral frame families. Each frame appears exactly 16 times,
and each six-scenario test domain×attribute cell contains two instances of
each frame. The bank-level and all per-template `source` fields are
`project-standardized-neutral-frame-v1-unreviewed`. This balances question
frame across test cells; it is not a human naturalness or neutrality
validation.

The historical generation log still records 24 candidate bases and display
name pools produced by the pinned Claude route through OpenRouter. Those
provider outputs were authoring inputs before project standardization, not the
declared source of the current visible frame text. Other historical candidate
inputs were project-produced and correctly have no fabricated provider
records.
Repository code supplies the fixed default/suggestion wording and fixed local
reply, and the mathematical simulator—not an authoring model—selects the
option. Runtime treats the stored names as one scenario-specific A–D pool,
assigns A/B by displayed position, and replaces internal option IDs with
`presented_option_N` in evaluated-model prompts.

The writing bank currently presents direct, controlled category descriptors
rather than complete alternative passages. That is useful for the narrow
mechanistic estimand, but it is not evidence that the same results generalize
to natural document excerpts. An excerpt-based robustness bank must be
authored, frozen, reviewed, and calibrated separately before making that
broader claim.

[`scenarios/conversation-templates-v1.generation.jsonl`](scenarios/conversation-templates-v1.generation.jsonl)
is the readable 24-request/24-result authoring log. It contains no credential
or authorization header. It is historical provenance for candidate language;
it is not a claim that the provider authored the current standardized frames,
and it is not an experiment-response file. The current bank remains a
candidate research stimulus: it is valid for simulation and bounded pilots,
but independently unreviewed and not paper-eligible. Before a public evidence
release, confirm applicable provider-output terms and complete the surface and
scientific reviews described in
[Scientific design](../docs/scientific-design.md#scenario-catalog-and-quality-policy).

Before using either file, run the outcome-free `cape-loop scenarios audit`
command documented in [Getting started](../docs/getting-started.md#inspect-and-validate-the-scenario-catalog).
Its JSON report and Markdown review packet are derived artifacts and belong
under `artifacts/` or another ignored work directory, not in `data/`.

## Checked-in model declarations

The three current metadata files are:

- [`model-suites/openai-gpt-5.6.json`](model-suites/openai-gpt-5.6.json):
  OpenAI Sol/medium as primary, Terra/medium as model-variant replication, and
  Luna/low for generic decoder/pilot work; and
- [`model-suites/gate4-native-and-distinct-decoders.json`](model-suites/gate4-native-and-distinct-decoders.json):
  OpenAI Sol/medium for native actions and OpenRouter
  `anthropic/claude-sonnet-5` at low effort plus
  `google/gemini-3.6-flash` at minimal effort for the selected decoder pair; and
- [`model-suites/experiment-b-bounded-calibration-v1.json`](model-suites/experiment-b-bounded-calibration-v1.json):
  the frozen Experiment B OpenRouter panel—Gemini 3.6 Flash, GPT-5.6 Luna, and
  Mistral Large 3 (`mistralai/mistral-large-2512`) as separate full-design
  primary arms, plus an incorrect-seed balanced-versus-soft DeepSeek V4 Flash
  secondary replication selected after the pilot.

These files are protocol and provenance metadata, not observations. The
selected decoder families share OpenRouter, so the declaration denies
first-party origin, distinct transport origins, and statistical independence.
Direct Anthropic and Gemini adapters remain optional origin replications.
The Experiment B suite likewise treats every model as its own analysis: no
cross-model estimator pools the primary arms, and DeepSeek remains outside the
primary analysis set. Its declaration records project authorship, source
status, Apache-2.0 licensing, and the CLI consumer explicitly.

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
