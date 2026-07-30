# Getting started

CAPE-Loop has a Python 3.11+ standard-library core. Offline validation,
simulation, replay planning, and artifact verification require no API key and
make no network request. A provider can be called only by a command carrying
the explicit `--execute-live` flag.

Run commands from the repository root.

## Check the checkout

```bash
python --version
PYTHONPATH=src python -m cape_loop doctor
PYTHONPATH=src python -m cape_loop --help
make check
```

`make check` compiles the Python source/tests, runs the runtime doctor,
validates every checked-in TOML file, checks the static mixed-effects contract,
and executes the offline unittest suite. Tests include schema-export parity.
Ruff, Markdown-link resolution, and `git diff --check` are separate release
audits. Passing any of these establishes software behavior only, not a paper
finding.

The source-tree invocation used throughout this repository is:

```text
PYTHONPATH=src python -m cape_loop ...
```

Packaging also exposes a `cape-loop` console script, but the source-tree form
is the reproducibility baseline.

## Choose the task

| Goal | Start here | Provider call |
| --- | --- | ---: |
| Understand the study | [Scientific design](scientific-design.md) and [Experiments](experiments.md) | No |
| Inspect one natural conversation, real model update, and its metrics | [Run one understandable live scenario](#run-one-understandable-live-scenario) | Exactly one |
| Inspect one matched, multi-turn Experiment B user case | [Run one multi-turn Experiment B case](#run-one-multi-turn-experiment-b-case) | 6 logical by default; at most 6 physical |
| Audit and review scenario calibration | [Inspect and validate the scenario catalog](#inspect-and-validate-the-scenario-catalog) | No |
| Inspect components and information boundaries | [Architecture](architecture.md) | No |
| Validate or edit TOML | [Configuration](configuration.md) | No |
| Author the frozen natural-language scenario bank | [Inspect and validate the scenario catalog](#inspect-and-validate-the-scenario-catalog) | Only with `--execute-live` |
| Generate the synthetic dataset | [Produce an offline dataset](#produce-an-offline-dataset) | No |
| Understand records, storage, and formats | [Data model](data-model.md) | No |
| Replay retained model responses | [Live execution](live-execution.md#request-response-and-replay-contract) | No |
| Plan or call OpenAI/OpenRouter | [Live execution](live-execution.md) | Only with `--execute-live` |
| Check what is actually complete | [Implementation status](implementation-status.md) | No |

## Run one understandable live scenario

Use this workflow when you want to see one complete hybrid-simulator example
before reading or launching a factorial experiment. It requires an OpenRouter
credential and always makes exactly one paid model request:

```bash
set -a
source .env
set +a
PYTHONPATH=src python -m cape_loop demo one-scenario \
  artifacts/one-scenario-demo --execute-live
```

The CLI never loads `.env` itself. Omitting `--execute-live` fails before the
command reads `OPENROUTER_API_KEY`, and the output directory must not already
exist. The one-request guard has zero retries, uses only the official
OpenRouter endpoint, permits no endpoint or model fallback, limits model
output to 2,048 tokens, and caps conservative total-token accounting at
10,000.

The defaults are:

| Control | Default |
| --- | --- |
| Model | `google/gemini-3.6-flash` |
| Scenario | `travel-scenario-atlas-lodging-price-01` |
| Mechanism | `balanced` |
| Seed | `1729` |

You may select another frozen scenario or supported OpenRouter model:

```bash
PYTHONPATH=src python -m cape_loop demo one-scenario \
  artifacts/one-scenario-claude \
  --model anthropic/claude-sonnet-5 \
  --scenario-id travel-scenario-atlas-lodging-price-01 \
  --mechanism suggested \
  --seed 1729 \
  --execute-live
```

`--mechanism` accepts `balanced`, `restricted`, `default`, or `suggested`.
The mathematical response model selects an option first. The frozen
conversation bank then renders the assistant presentation and the constrained
user reply. Finally, the selected OpenRouter model receives the full-context
view and returns one structured profile update. A local exact action-aware
reference requires no provider call.

The output directory contains:

```text
conversation.md
conversation.jsonl
result.json
llm/
├── requests.jsonl
├── responses.jsonl
├── provider-audit.jsonl
├── provider-attempts.jsonl
└── provider-manifest.json
```

Read `conversation.md` first. It identifies each role, shows the natural
exchange, names the evaluated model and its information view, and places
readable metrics beside the result. `conversation.jsonl` contains one
canonical conversation-trace record. `result.json` is the complete compact
machine-readable walkthrough; the `llm/` files preserve the one request,
response, transport audit, and provider execution manifest. Standard output
is a compact JSON receipt with status, provider-call count, selected
model/scenario/choice, key metrics, and the readable-log path.

This command intentionally does not run matched treatments, multiple users,
multiple updaters, calibration, or inference. Its artifacts are marked
demonstration/debugging only, not paper-eligible, and not claim-eligible. Use a
reviewed configuration for an actual experiment.

## Run one multi-turn Experiment B case

Experiment B cannot be illustrated faithfully with only one turn: it asks
whether a stored profile shapes later evidence and actions. The bounded
diagnostic therefore fixes one simulated travel user with an intentionally
incorrect initial profile and runs two matched branches:

- three turns under the balanced policy by default; and
- three turns under the soft-profile-conditioned policy by default.

The evaluated `llm_full_context` updater makes one selected-model update after
each turn, for six logical updates by default and at most six paid calls; an
identical content-addressed request can be reused locally. The same-history
exact action-aware shadow runs locally and makes no provider call. Load the
ignored credential and choose a new output directory:

```bash
set -a
source .env
set +a
PYTHONPATH=src python -m cape_loop demo experiment-b-case \
  artifacts/experiment-b-case --execute-live
```

Defaults pin `google/gemini-3.6-flash` to `google-vertex/global`, disable
retries and fallback, and use a 40-second per-call timeout. The output layout
matches the one-scenario diagnostic: read `conversation.md` first, then use
`result.json`, `conversation.jsonl`, and the `llm/` request, response, audit,
attempt, and provider-manifest files for exact inspection.

For this multi-turn command, `llm/logical-request-events.jsonl` contains one
small binding row for every logical update. `llm/requests.jsonl` contains each
distinct full request once, so it remains compatible with the repository's
strict request reader even when a logical update reuses identical prompt
content.

Use `--turns 3`, `6`, `9`, or `12` to select a complete attribute cycle. Two
policy branches make the logical update ceiling `2 × turns`; physical calls
can be lower when content-addressed prompts are identical. The timeout is per
physical call, so longer cycles can take more than five minutes.

The three-turn default is the shortest transport and output-format check. It
shows every attribute once, but normally gives an update no later action on the
same attribute to influence. Use at least `--turns 6` when checking the
later-action mechanism; the 2026-07-29 multi-model diagnostic first observed
profile-influenced presentation actions at that horizon.

OpenRouter model switching uses the same command. Specify the model's intended
reasoning effort explicitly; for example:

```bash
PYTHONPATH=src python -m cape_loop demo experiment-b-case \
  artifacts/experiment-b-case-claude \
  --model anthropic/claude-sonnet-5 \
  --reasoning-effort low \
  --execute-live
```

The same command supports the direct OpenAI primary writer:

```bash
PYTHONPATH=src python -m cape_loop demo experiment-b-case \
  artifacts/experiment-b-case-openai \
  --provider openai \
  --model gpt-5.6-sol \
  --reasoning-effort medium \
  --execute-live
```

This is one **simulated-user case**, not one repeated frozen scenario. At six
or more turns, each branch revisits attributes so a later-action effect is
possible. The command is marked diagnostic-only, paper-ineligible, and
claim-ineligible; the full Experiment B pilot still requires a revised,
reviewed TOML design, adequate users, calibration, and inference.

## Validate a configuration

Configurations are strict, schema-versioned TOML:

```bash
PYTHONPATH=src python -m cape_loop config validate configs/smoke.toml
```

Validation prints the fully resolved JSON configuration. It rejects unknown
keys, invalid component identifiers, invalid ranges, experiment-incompatible
fields, and a live workload whose retry-expanded attempt or maximum-output
allocation exceeds its declared ceilings.

The configuration directory contains four groups:

- offline A/B/C and smoke runs;
- offline sensitivity and multi-seed robustness runs;
- bounded OpenAI/OpenRouter transport pilots; and
- offline Gate 4/Experiment C source-packet generators.

The exact inventory, defaults, and per-experiment contracts are kept in one
place: [Configuration](configuration.md). Checked-in settings are software
references or explicitly labeled pilots, not preregistrations or final power
decisions.

## Inspect and validate the scenario catalog

The checked-in configurations bind
[`data/scenarios/scenario-catalog-v1.json`](../data/scenarios/scenario-catalog-v1.json)
by SHA-256. First check that it is valid JSON:

```bash
python -m json.tool data/scenarios/scenario-catalog-v1.json > /dev/null
```

Then run the same strict parser and digest check used before experiment
execution:

```bash
PYTHONPATH=src python -c 'from cape_loop import load_config; from cape_loop.scenarios import load_scenario_catalog; c = load_config("configs/smoke.toml"); x = load_scenario_catalog(c.scenarios.catalog_file, expected_sha256=c.scenarios.catalog_sha256); print(x.catalog.coverage_report())'
```

The current report has 48 scenarios and 48 families: one train, one
development, and six test scenarios for each domain×attribute cell. It also
reports 48 provisional scenarios, zero approved scenarios, and
`paper_eligible = false`. This is deliberate: the catalog is usable for
simulation and bounded pilots, but its independent surface and scientific
human reviews are incomplete. The acceptance policy is in
[Scientific design](scientific-design.md#scenario-catalog-and-quality-policy).

Generate the complete prospective audit and human-review packet before a live
experiment:

```bash
PYTHONPATH=src python -m cape_loop scenarios audit \
  configs/live/experiment_b_openrouter.toml \
  artifacts/scenario-audit-test16 \
  --split test --turns 16
```

No API key is read and no model is called. The output directory must be new:

- `scenario-audit.json` contains the declared response-model grid, per-cell
  no-repeat capacity, machine warnings, review counts, and separate
  engineering/recorded-scientific/paper readiness;
- `scenario-review.md` is the detailed researcher packet: it exposes feature
  roles and gives six labeled preview surfaces per scenario for fact alignment
  and scientific inspection;
- `scenario-surface-review-blinded.md` uses only opaque item/surface labels and
  rendered dialogue so independent reviewers can rate naturalness and
  neutrality without seeing targets, features, roles, splits, or mechanism
  labels; and
- the machine audit separately renders all 40
  anchor×order×mechanism×selection cases per scenario for hygiene checks.

The current test bank passes the code-only finite-support probability
guardrails and has enough unique stimuli for a 16-turn cyclic trajectory
without reuse. The same capacity holds for the v2 exploratory policy because
it covers every attribute once per complete three-turn block. Unknown
unconstrained adaptive policies have a stricter worst-case requirement. Its
automated, surface, and scientific reviews remain incomplete. Machine
warnings are review prompts, never automatic approval. Do not edit or replace
stimuli after inspecting evaluated-model outcomes. Catalog review-status
strings are not verified evidence; scientific readiness remains false until a
future version-bound human-evidence importer verifies the completed protocol.

`config validate` checks the `[scenarios]` field syntax; the strict loader above
also reads, hashes, and validates the external catalog. A hash mismatch aborts
before provider or run-artifact construction.

The companion
[`data/scenarios/conversation-templates-v1.json`](../data/scenarios/conversation-templates-v1.json)
contains one frozen template family per scenario. Its 48 current visible bases
were project-standardized outcome-blind onto three source-neutral frames: each
frame appears 16 times and twice within each six-scenario test
domain×target cell. Every template has source
`project-standardized-neutral-frame-v1-unreviewed`. A separate historical log
records 24 OpenRouter-assisted candidate-authoring calls; it does not attribute
the current standardized text to that provider. Code supplies the fixed
treatment sentence and `I choose {selected_name}.` A/B are assigned by visible
position at runtime, and structured evaluated-model inputs receive only
`presented_option_N` aliases, not semantic catalog IDs. To generate a new
OpenRouter-assisted candidate bank:

```bash
cape-loop conversations generate-openrouter \
  data/scenarios/scenario-catalog-v1.json \
  data/scenarios/conversation-templates-v1.json \
  --model anthropic/claude-sonnet-5 --execute-live
```

This is a separate authoring task, not an experiment run. It also writes
`data/scenarios/conversation-templates-v1.generation.jsonl` for readable
request/output provenance. Review the resulting language before using
`[scenarios] conversation_file` to select it. Do not rerun the authoring model
independently for every simulated user or trial.

The current catalog and bank remain simulation-and-pilot-only. Their
independent human surface and scientific reviews are pending, so
`paper_eligible` remains false.

## Run and verify the smoke configuration

```bash
PYTHONPATH=src python -m cape_loop run configs/smoke.toml
```

The command prints JSON containing the exact `run_dir`, whether an existing
artifact was reused, and a summary. The default directory is content-addressed:

```text
runs/<run-name>-<first-12-hex-of-resolved-config-SHA256>/
```

Verify the printed path:

```bash
PYTHONPATH=src python -m cape_loop verify runs/<printed-run-id>
```

Verification checks the completion manifest, run/config identity,
`config.source.toml` origin, summary, symlink policy, and exact `SHA256SUMS`
inventory. It checks integrity, not scientific completeness.

Runs never overwrite an existing deterministic destination. To reuse an
existing completed run:

```bash
PYTHONPATH=src python -m cape_loop run \
  configs/smoke.toml --allow-existing
```

Reuse succeeds only when the artifact verifies and its source/config/input
identity matches the current request.

## Produce an offline dataset

The latent users, contexts, choices, updates, and reference metrics are
synthesized deterministically by CAPE-Loop rather than downloaded. OpenRouter
supplies only frozen neutral base wording and display names; code adds the
fixed treatment sentence and user reply. The author never sees the latent user
and never chooses an option.

Generate the A/B/C release-candidate set:

```bash
PYTHONPATH=src python -m cape_loop run configs/offline/experiment_a.toml
PYTHONPATH=src python -m cape_loop run configs/offline/experiment_b.toml
PYTHONPATH=src python -m cape_loop run configs/offline/experiment_c.toml
PYTHONPATH=src python -m cape_loop run configs/offline/experiment_c_seed_271828.toml
PYTHONPATH=src python -m cape_loop run configs/offline/experiment_c_seed_314159.toml
```

Verify every printed directory separately. The canonical dataset is the set of
verified run directories, not a flattened table assembled later.

The default storage and primary formats are:

```text
runs/<run-id>/             # ignored local run
  inputs/scenario-catalog.json
  inputs/scenario-catalog-manifest.json
  inputs/conversation-templates.json
  inputs/conversation-templates-manifest.json
  population/*.jsonl       # synthetic latent-user rows
  events/*.jsonl           # complete interactions/state; large audit records
  analysis/*.jsonl         # compact A/B/C rows for ordinary analysis
  models/*.json            # fitted/calibrated model records
  metrics/*.json[l]        # canonical metric evidence
  metrics/scenario-coverage.json
  metrics/scenario-consumption.json
  tables/*.csv             # derived readable projections
  manifest.json
  SHA256SUMS
```

In a hybrid event, inspect the observation fields together:

```text
selected_option    mathematical simulator output
assistant_message  frozen natural presentation of the visible options
surface_response   constrained local reply, such as "I choose Hotel A."
surface_id         exact scenario/presentation/ranking/choice surface
```

Full event JSONL retains those fields. If an evaluated LLM updater is active,
its separate requests and responses are under `llm/`. The authoring
`.generation.jsonl` remains beside the source bank in `data/scenarios/`; it is
not repeated once per event.

The three runner-native compact files are:

| Experiment | File | One row means |
| --- | --- | --- |
| A | `analysis/experiment-a-rows.jsonl` | One evaluated updater×trial pair, including controlled and naturally sampled response modes |
| B | `analysis/experiment-b-turns.jsonl` | One retained turn from one closed-loop trajectory |
| C | `analysis/experiment-c-rows.jsonl` | One fixed-history or endogenous evaluation/ranking row |

Experiment A also always writes
`analysis/experiment-a-exclusions.jsonl`, one row per excluded matched set, so
the confirmatory loader can audit exclusions even when raw events are not
retained.

These rows are projections of records the runner already produced. They do not
create another user, interaction, LLM request, or observation, and therefore
do not increase the sample size. Their purpose is to leave the large posterior,
shadow-state, and native-memory payloads in `events/` while exposing the scalar
identifiers and outcomes used by ordinary analysis. Sensitivity already writes
compact aggregate records under `metrics/` and `tables/`, so it has no
additional `analysis/` file.

`data/` contains only small tracked inputs and declarations.
`artifacts/` is for curated, checksum-bound release evidence. See
[Data model](data-model.md) for splits, fields, joins, complete output
lifecycle, and the external/human-data boundary.

Generate the broader offline simulator robustness artifact separately:

```bash
PYTHONPATH=src python -m cape_loop run configs/offline/sensitivity.toml
```

It is a 19-point baseline-first one-at-a-time design. It measures marginal
departures from the baseline and explicitly does not estimate interactions
among sensitivity axes.

## Inspect a run

Start with:

```text
manifest.json
config.resolved.json
splits.json
inputs/scenario-catalog-manifest.json
metrics/scenario-coverage.json
metrics/scenario-consumption.json
metrics/summary.json
metrics/gate-report.json
conversations/<experiment>.md
conversations/<experiment>.jsonl
SHA256SUMS
```

Here `<experiment>` is `experiment-a`, `experiment-b`, `experiment-c`, or
`sensitivity`. Read the Markdown conversation preview first when you want to
understand what happened in ordinary language. It shows Assistant/User turns,
conditions, and metrics for a deterministic, diverse sample of at most 100
trace records by default. Its header states the exact total number of
conversation records, turns, and outcomes/evaluations in the run and defines
the metric names.

Use the matching JSONL when you need every conversation. It is exhaustive and
deduplicated: shared Experiment A trials and Experiment C fixed histories are
stored once with their updater evaluations grouped beside them. The Markdown
file is intentionally only a preview, so never use its displayed-record count
as the experiment sample size.

The same values are available programmatically in `metrics/summary.json` as
`conversation_log_artifact`, `conversation_log_markdown_artifact`,
`conversation_record_count`, `conversation_turn_count`,
`conversation_outcome_count`, and `conversation_markdown_preview_count`.

For A, B, or C, inspect the matching `analysis/*.jsonl` for exploratory tables
and scripts; sensitivity already has compact `metrics/*.jsonl` and
`tables/sensitivity.csv`. Opening a multi-gigabyte `events/*.jsonl` audit is
unnecessary for routine analysis. Then inspect experiment-specific metrics and
any derived CSV. Important interpretation rules are:

- `claim_status` or `scientific_claim_status` remains `not_claimed`;
- controlled identical-response and naturally sampled rows are different
  estimands;
- fixed histories and endogenous histories are different regimes;
- raw and calibrated model records remain separate;
- test labels never fit a calibrator; and
- an implemented gate computation is not evidence that the gate passed.

The exact record contracts and output locations are in
[Data model](data-model.md). Metric definitions are in [Metrics](metrics.md).

New runs write their compact rows before finalization, so `SHA256SUMS` and the
ordinary `verify` command cover them. Do not copy a new file into an already
completed run: that would invalidate its exact inventory. Instead, derive and
verify a separate compact directory from an immutable historical run:

```bash
PYTHONPATH=src python -m cape_loop artifact compact \
  runs/<run-id> artifacts/<compact-id>
PYTHONPATH=src python -m cape_loop artifact verify-compact \
  artifacts/<compact-id>
```

The output contains `analysis-rows.jsonl`, `manifest.json`, and `SHA256SUMS`.
Its manifest binds the verified source run, checksums, and exact exporter-source
digest. This is a derived analysis convenience, not a new dataset and not a
paper-evidence promotion. The optional R harness accepts one such directory per
historical `--run` through `--compact-bundle`; see its
[operator guide](../analysis/confirmatory-mixed-effects/README.md).

## Export public schemas

Regenerate the checked-in interchange schemas:

```bash
PYTHONPATH=src python -m cape_loop schema export schemas
```

Export to another destination by replacing `schemas`. Python constructors also
enforce cross-field invariants, so JSON Schema validity alone is necessary but
not sufficient.

## Plan live work before spending

Use increasing-cost tiers:

```text
offline validate and keyless plan
  -> one-time conversation-bank authoring and review, when changing surfaces
  -> small static transport smoke
  -> six-update adaptive smoke
  -> reviewed bounded A/B/C or Gate 6 pilot
  -> paper collection only after preregistration and evidence review
```

| Tier | Purpose | Typical upper bound |
| --- | --- | ---: |
| Offline | Generate synthetic data, validate, plan, replay | 0 calls |
| Conversation authoring | Create a candidate frozen bank; separate from experiments | Explicit OpenRouter calls |
| Static smoke | Endpoint, schema, identity, audit, replay conversion | 6 attempts |
| Adaptive smoke | Provider/runner integration | 6 logical updates |
| Bounded pilot | Reviewed A/B/C or Gate 6 design | 576–864 attempts/provider |
| Paper collection | Frozen design and release plan | Separately approved |

The approved pilot ceiling is 900 physical attempts and 6,000,000
conservatively accounted tokens per provider ledger. Bounded pilots use zero
automatic retries so their full design fits. Do not raise the approved ceiling
to make a plan pass; reduce design size, retries, or output allowance.

The current source can plan all selected workflows. Provider capability does
not imply that every selected endpoint/model combination has completed a
current-source live check or that an eligible corpus exists. Review the dated
[Implementation status](implementation-status.md#local-live-validation-snapshot)
and [Live execution](live-execution.md#current-live-boundary) before spending.

## Load credentials safely

The CLI does not automatically parse `.env`. If you use the local ignored file:

```bash
set -a
source .env
set +a
```

Never print or commit the values. The main variables are:

```text
OPENAI_API_KEY       direct OpenAI writers and native actions
OPENROUTER_API_KEY   conversation authoring, OpenRouter writers, and selected Claude/Gemini decoders
ANTHROPIC_API_KEY    optional direct decoder replication
GEMINI_API_KEY       optional direct decoder replication
```

Planning remains keyless even when no variable is set.

## Run a bounded provider workflow

Conversation authoring and profile writing are two distinct roles:

```text
OpenRouter conversation author
  -> neutral base wording + display names
  -> code-expanded frozen template bank
  -> mathematical choice + offline rendering
  -> evaluated profile writer configured under [llm]
```

Changing the authoring model changes the stimulus bank and therefore requires
new review. Changing `[llm].model` changes the evaluated system but does not
rewrite the conversation bank.

Inspect the declared OpenAI roles:

```bash
PYTHONPATH=src python -m cape_loop llm models
```

Plan a static OpenRouter request without reading a key:

```bash
PYTHONPATH=src python -m cape_loop llm plan-openrouter \
  requests.jsonl --model google/gemini-3.6-flash \
  --max-requests 6 --max-retries 0
```

Run an adaptive pilot only after its resolved configuration and preflight are
reviewed. To change its model or route, first copy the preset into ignored
`configs/local/` and operate on that copy:

```bash
mkdir -p configs/local
cp configs/live/experiment_a_openrouter.toml \
  configs/local/experiment_a_openrouter.toml

PYTHONPATH=src python -m cape_loop config validate \
  configs/local/experiment_a_openrouter.toml

PYTHONPATH=src python -m cape_loop run \
  configs/local/experiment_a_openrouter.toml --execute-live
```

The complete guide covers:

- direct OpenAI and OpenRouter static execution;
- adaptive A/B/C and Gate 6 pilots;
- model switching and route constraints;
- development-only calibration;
- attempt journals, locks, and safe recovery;
- Experiment A controls and H7 direct statements;
- selected Claude/Gemini decoder collection;
- OpenAI native actions and Gate 4 import;
- Experiment C external rescoring; and
- optional direct decoder replications.

See [Live execution](live-execution.md).

## Generate Gate 4 and Experiment C request packets offline

These source runs make no provider call:

```bash
PYTHONPATH=src python -m cape_loop run \
  configs/offline/gate4_source.toml

PYTHONPATH=src python -m cape_loop run \
  configs/offline/experiment_c_rescore_source.toml
```

The Gate 4 source produces 640 blinded decoder requests per selected model and
80 native-action requests. The Experiment C source produces 360 decoder
requests per model. Verify the source before using its request packet:

```bash
PYTHONPATH=src python -m cape_loop verify runs/<printed-run-id>
```

Generating a request packet is not the same as collecting provider responses.

## Recover a failed adaptive run

Inspect the retained `failure.json`, correct the cause, and resume only the
same configuration:

```bash
PYTHONPATH=src python -m cape_loop run CONFIG.toml \
  --execute-live --resume-failed-live
```

The runner archives the failed directory and reconciles its external
audit-first journal. An unresolved physical attempt requires manual billing
review and is not automatically resent.

## Freeze a release artifact

Freeze only a completed verified run:

```bash
PYTHONPATH=src python -m cape_loop artifact freeze \
  runs/<run-id> artifacts/<artifact-id>.tar
PYTHONPATH=src python -m cape_loop artifact verify \
  artifacts/<artifact-id>.tar
```

Freezing normalizes and hash-binds the archive; it does not turn a smoke,
pilot, incomplete gate, or synthetic diagnostic into paper evidence.
`artifact freeze` preserves the complete run, including raw audit events. Use
`artifact compact` only for the smaller derived analysis directory; it does not
replace the full archive required for forensic reproduction.

## Human-study materials

Generate a deterministic blinded packet:

```bash
PYTHONPATH=src python -m cape_loop human-study generate \
  /tmp/cape-loop-study --assignment-id pilot-template --seed 1729
```

This creates study materials and a researcher codebook. It does not provide
ethics approval, recruit participants, host a survey, collect responses, or
authorize release. Review [Ethics and limitations](ethics-and-limitations.md)
before any human collection.

## Recommended execution order

1. Run `make check`.
2. Validate the selected TOML.
3. Generate and verify the offline source data.
4. Inspect keyless provider plans and hard ceilings.
5. If needed, run a six-request static transport smoke.
6. If adaptive integration needs validation, run one six-update smoke.
7. Review audits, returned models/routes, failures, and actual usage.
8. Treat each bounded A/B/C, Gate 4, or Gate 6 collection as a separate
   authorization and spending decision.
9. Build immutable reviews only from complete verified source/evidence
   directories.
10. Freeze and release only after scientific, ethics, provenance, and
    authorship review.

Before interpreting any result, compare it against
[Implementation status](implementation-status.md) and preserve the no-claim
boundary.
