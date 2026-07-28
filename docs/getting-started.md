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
| Inspect components and information boundaries | [Architecture](architecture.md) | No |
| Validate or edit TOML | [Configuration](configuration.md) | No |
| Generate the synthetic dataset | [Produce an offline dataset](#produce-an-offline-dataset) | No |
| Understand records, storage, and formats | [Data model](data-model.md) | No |
| Replay retained model responses | [Live execution](live-execution.md#request-response-and-replay-contract) | No |
| Plan or call OpenAI/OpenRouter | [Live execution](live-execution.md) | Only with `--execute-live` |
| Check what is actually complete | [Implementation status](implementation-status.md) | No |

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

The core dataset is synthesized by CAPE-Loop's deterministic generator. It is
not downloaded and is not written by Codex or an external model.

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
  population/*.jsonl       # synthetic latent-user rows
  events/*.jsonl           # interactions and trajectories
  models/*.json             # fitted/calibrated model records
  metrics/*.json[l]         # canonical metric evidence
  tables/*.csv              # derived readable projections
  manifest.json
  SHA256SUMS
```

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
metrics/summary.json
metrics/gate-report.json
SHA256SUMS
```

Then inspect experiment-specific JSON/JSONL and any derived CSV. Important
interpretation rules are:

- `claim_status` or `scientific_claim_status` remains `not_claimed`;
- controlled identical-response and naturally sampled rows are different
  estimands;
- fixed histories and endogenous histories are different regimes;
- raw and calibrated model records remain separate;
- test labels never fit a calibrator; and
- an implemented gate computation is not evidence that the gate passed.

The exact record contracts and output locations are in
[Data model](data-model.md). Metric definitions are in [Metrics](metrics.md).

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
  -> small static transport smoke
  -> six-update adaptive smoke
  -> reviewed bounded A/B/C or Gate 6 pilot
  -> paper collection only after preregistration and evidence review
```

| Tier | Purpose | Typical upper bound |
| --- | --- | ---: |
| Offline | Generate synthetic data, validate, plan, replay | 0 calls |
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
OPENROUTER_API_KEY   OpenRouter writers and selected Claude/Gemini decoders
ANTHROPIC_API_KEY    optional direct decoder replication
GEMINI_API_KEY       optional direct decoder replication
```

Planning remains keyless even when no variable is set.

## Run a bounded provider workflow

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
