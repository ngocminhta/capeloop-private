# H7 volunteered-preference controls

H7 asks whether provenance-aware instructions reduce causal-attribution error
without suppressing valid learning. Balanced choices supply one valid-learning
condition inside Experiment A. A volunteered preference is a different
observation type: it is a direct user-originated statement with no option set,
default, ranking, or suggestion. It must not be synthesized from a choice row.

The repository therefore implements volunteered evidence as an external,
provider-neutral collection and a derived review. The source Experiment A run
remains checksummed and unchanged throughout.

## Components

| Component | Responsibility |
| --- | --- |
| `h7_control_review.py` | Deterministic plan construction, source-run verification, provider-audit validation, update conversion, derived H7 recomputation, and exact reverification |
| `hypothesis_estimands.py` | Frozen paired-valid-learning estimand through `analyze_h7_volunteered_updates()` |
| `control-study h7-plan` | Build requests from a verified Experiment A run without reading a credential |
| `llm plan` / `llm plan-openrouter` | Inspect exact OpenAI/OpenRouter request bodies and hard ceilings |
| `llm execute-openai` / `llm execute-openrouter` | Perform explicitly authorized, resumable live collection |
| `control-study h7-review` | Require complete accepted provider evidence and write a new review artifact |
| `control-study h7-verify` | Reload every input, recompute the review, and reject any difference |

## Frozen case construction

The plan reads only a complete, checksum-verified Experiment A run. It requires:

- `population/users.jsonl`;
- `metrics/experiment-a.jsonl`;
- `metrics/experiment-a-hypothesis-estimands.json`;
- `manifest.json`, `config.resolved.json`, and `SHA256SUMS`; and
- controlled-anchor coverage for both `llm_full_context` and
  `llm_provenance_aware`.

There is no case sampling or author-selected subset. For every retained test
user, every configured domain row, and each of the three preference
attributes, the plan generates one direct statement from the simulator's
latent preference direction. For example:

```text
I generally prefer premium.
I generally prefer concise.
```

Each case is crossed with exactly two updater views:

| Updater | View | Additional provenance metadata |
| --- | --- | --- |
| `llm_full_context` | `full_context` | None |
| `llm_provenance_aware` | `provenance_aware` | `response_source = user`; `elicitation_provenance = user_originated_unprompted` |

Both conditions receive the same uniform prior, statement, target attribute,
and contextual record. The request ID, prompt digest, case digest, source-user
row digest, and request-binding digest are retained. The target direction used
for scoring is withheld in the analysis binding; it is not added as a special
answer label to the model prompt.

The independent statistical unit remains the complete latent user. Multiple
domain/attribute cases from one user remain in that user's bootstrap cluster.
At least two independent test users are required.

## Build and inspect the plan

Use an already complete Experiment A primary or replication run:

```bash
PYTHONPATH=src python -m cape_loop control-study h7-plan \
  runs/<verified-experiment-a-run> \
  artifacts/h7-primary-plan
```

The output directory must not already exist and must be outside the source run.
It contains:

```text
h7-volunteered-plan.json
h7-volunteered-request-bindings.jsonl
h7-volunteered-requests.jsonl
```

The JSON plan embeds the source-run hashes, all cases, all withheld request
bindings, exact coverage counts, and `plan_sha256`. The request JSONL is the
provider-neutral corpus accepted by the existing provider commands.

Inspect OpenAI request bodies and the retry-expanded ceiling without reading a
key:

```bash
PYTHONPATH=src python -m cape_loop llm plan \
  artifacts/h7-primary-plan/h7-volunteered-requests.jsonl \
  --role primary \
  --max-requests 900 \
  --max-total-tokens 6000000
```

Or inspect an explicit OpenRouter model and route:

```bash
PYTHONPATH=src python -m cape_loop llm plan-openrouter \
  artifacts/h7-primary-plan/h7-volunteered-requests.jsonl \
  --model google/gemini-2.5-pro \
  --upstream-provider Google \
  --max-requests 900 \
  --max-total-tokens 6000000
```

Always review the reported physical-attempt and conservative-token bounds
before authorizing a live call.

## Collect provider evidence

OpenAI:

```bash
PYTHONPATH=src python -m cape_loop llm execute-openai \
  artifacts/h7-primary-plan/h7-volunteered-requests.jsonl \
  artifacts/h7-primary-responses.jsonl \
  artifacts/h7-primary-provider-audit.jsonl \
  --role primary \
  --max-requests 900 \
  --max-total-tokens 6000000 \
  --execute-live
```

OpenRouter:

```bash
PYTHONPATH=src python -m cape_loop llm execute-openrouter \
  artifacts/h7-primary-plan/h7-volunteered-requests.jsonl \
  artifacts/h7-primary-responses.jsonl \
  artifacts/h7-primary-provider-audit.jsonl \
  --model google/gemini-2.5-pro \
  --upstream-provider Google \
  --max-requests 900 \
  --max-total-tokens 6000000 \
  --execute-live
```

The CLI does not load `.env` automatically. Export the selected credential into
the process environment before live execution, and never commit `.env` or a
provider key.

One review holds provider and returned model fixed across every case and both
updater conditions. To evaluate another model, collect a separate corpus and
produce a separate review. OpenRouter evidence remains labeled as gateway
evidence; it does not claim direct first-party origin.

## Create and verify the derived review

```bash
PYTHONPATH=src python -m cape_loop control-study h7-review \
  runs/<verified-experiment-a-run> \
  artifacts/h7-primary-plan \
  artifacts/h7-primary-responses.jsonl \
  artifacts/h7-primary-provider-audit.jsonl \
  artifacts/h7-primary-review.json

PYTHONPATH=src python -m cape_loop control-study h7-verify \
  runs/<verified-experiment-a-run> \
  artifacts/h7-primary-plan \
  artifacts/h7-primary-responses.jsonl \
  artifacts/h7-primary-provider-audit.jsonl \
  artifacts/h7-primary-review.json
```

Review is strict:

- the source run must pass `SHA256SUMS` verification;
- plan JSON, binding JSONL, and request JSONL must exactly match regeneration;
- response and accepted provider-audit coverage must be exact;
- duplicate, missing, unexpected, or rejected rows fail;
- request ID, prompt hash, request-body hash, raw-response hash, embedded replay
  response, provider identity, and returned model are bound;
- the ordinary and provenance-aware outcome for each case must use the same
  provider and model; and
- every case must have both updater outcomes.

The verified source artifacts, all three plan representations, provider
responses, and provider audit are parsed from exact regular-file byte
snapshots. Their recorded hashes bind those parsed bytes; symlinks and
non-files are rejected, and every named path is rechecked immediately before
the immutable review is published.

No incomplete collection is analyzed. No balanced-choice value, average,
zero, or other placeholder is substituted for a missing direct statement.

For each accepted response the converter computes the target-direction
log-odds change from the frozen uniform prior and emits:

```text
case_id
user_id
updater_id
directional_log_odds_update
```

These are the exact `VolunteeredPreferenceUpdate` records consumed by H7's
cluster-bootstrap valid-learning analysis. The derived artifact reuses the
source run's already checksummed ACUE-superiority and balanced-valid-learning
components, recomputes volunteered valid learning, and then recomputes the
Experiment A H7 completeness and criterion. It records every input digest,
the provider-bound evidence, the recomputation scope, and its own
`review_sha256`.

The review always keeps:

```text
claim_status = "not_claimed"
source_run_modified = false
missing_values_imputed = false
```

A complete or passing Experiment A H7 review is not the full H7 paper claim.
The separate Experiment B closed-loop self-confirmation mitigation component,
paper sample adequacy, multiplicity review, and the declared release process
remain required.

## Public schemas

The external contracts are exported as:

- `h7-volunteered-request-binding.schema.json`;
- `h7-volunteered-collection-plan.schema.json`;
- `h7-volunteered-evidence.schema.json`; and
- `h7-volunteered-review.schema.json`.

`python -m cape_loop schema export schemas` regenerates these files from the
same in-package schema registry used by the reproducibility tests.
