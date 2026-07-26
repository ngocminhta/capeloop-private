# LLM exchange and live execution

CAPE-Loop supports two model-execution modes behind the same hash-bound
request/response contract:

```text
replay:  retained response JSONL ──► prompt-hash check ──► profile update
openai:  CAPE request ──► budgeted Responses API call ──► journal ──► profile update
runner:  development raw outputs ──► temperature fit ──► calibrated test/runtime update
```

Replay remains the portable scientific record. Live execution is an opt-in data
collection mechanism around it: successful provider outputs are converted to
the same strict `LLMResponse` objects and can subsequently be replayed offline.

No live provider execution was performed to create the checked-in repository.
There are no checked-in API credentials or live-model results.

## Declared model suite

Inspect the versioned defaults without a credential:

```bash
PYTHONPATH=src python -m cape_loop llm models
```

| Role | Default model | Reasoning effort | Intended use |
| --- | --- | --- | --- |
| `primary` | `gpt-5.6-sol` | `medium` | Confirmatory profile-writer evaluation |
| `replication` | `gpt-5.6-terra` | `medium` | Cost-balanced GPT-5.6 model-variant/tier replication |
| `decoder` | `gpt-5.6-luna` | `low` | High-volume blinded decoding and pilots |

The role names are repository policy, not claims that one model is universally
best. For a causal comparison among response-only, full-context, and
provenance-aware writers, hold both the model and reasoning effort fixed across
all information views. A `--model` or `[llm].model` override is allowed, but it
must be reported as a protocol change.

The Terra role is a replication across a declared GPT-5.6 model variant/tier.
It is not evidence of robustness to a distinct model family, provider, or
independent error process.

The machine-readable declaration is
[`data/model-suites/openai-gpt-5.6.json`](../data/model-suites/openai-gpt-5.6.json).
The selection and wire contract were checked against OpenAI's official
[latest-model guide](https://developers.openai.com/api/docs/guides/latest-model),
[model catalog](https://developers.openai.com/api/docs/models), and
[Structured Outputs guide](https://developers.openai.com/api/docs/guides/structured-outputs)
on the declaration's `resolved_on` date. Recheck those sources before changing
models or starting a later collection wave.

## Safety boundary

A live request requires every one of these conditions:

1. the command carries `--execute-live`;
2. the provider configuration has live execution enabled internally;
3. the configured credential environment variable exists;
4. the endpoint is the official `https://api.openai.com` origin, unless a
   separate custom-endpoint opt-in is present;
5. the next request fits under both `max_requests` and
   `max_total_tokens`; and
6. the provider response satisfies the strict beliefs schema; and
7. the returned model label is the requested label or its `YYYY-MM-DD` dated
   snapshot.

Configuration validation, `llm models`, `llm plan`, replay, tests, and ordinary
non-LLM runs do not read a key or contact a provider. The key is read only from
the environment immediately before an authorized HTTP request. It is never
accepted as a CLI value, TOML value, request field, journal field, or run
artifact.

The default endpoint guard accepts the official
`https://api.openai.com` origin (with an optional trailing slash) and rejects a
different host, port, or path. `--base-url` alone cannot bypass this boundary.
For static/decoder CLI work, `--allow-custom-base-url`; for an adaptive run,
`[llm].allow_custom_base_url = true` is also required. This is a credential
routing decision, not merely a compatibility flag: an authorized live request
will send the value from `api_key_env` to the reviewed custom HTTPS endpoint.
Custom hosts require a dedicated environment-variable name other than
`OPENAI_API_KEY` (for example `CAPE_LOOP_PROXY_KEY`). Do not enable the route
for an endpoint you do not administer or explicitly trust.

Copy [`.env.example`](../.env.example) only as a naming reminder. The CLI does
not load `.env` files automatically; export the value in the invoking shell:

```bash
export OPENAI_API_KEY='...'
```

Never commit the populated value or paste it into an experiment config.

## Static request workflow

### Plan without a key

Given an existing request JSONL corpus, prepare the exact request bodies and
check conservative ceilings without reading a credential:

```bash
PYTHONPATH=src python -m cape_loop llm plan requests.jsonl \
  --role primary \
  --max-requests 500 \
  --max-total-tokens 2050000
```

The plan reports the resolved model and reasoning effort, body hashes, request
count, conservative maximum token reservation, and whether the corpus fits the
declared ceilings. The conservative estimate treats every UTF-8 request byte as
an input token, adds framing headroom, and reserves `max_output_tokens`; it is a
safety bound, not a billing estimate.

`llm plan` does not export adaptive experiment prompts. Later prompts in a
closed loop depend on earlier model-written profiles, so use the adaptive runner
for that case.

### Execute or resume

After reviewing the plan and cost ceiling:

```bash
PYTHONPATH=src python -m cape_loop llm execute-openai \
  requests.jsonl \
  responses.jsonl \
  provider-audit.jsonl \
  --role primary \
  --max-requests 500 \
  --max-total-tokens 2050000 \
  --execute-live
```

The positional outputs must be different paths. The executor is resumable:
existing responses are checked against request IDs and prompt hashes, completed
audit records are reconciled into a missing response line, and only absent
requests are sent. The audit line is appended before the replay response so an
interruption after provider success does not require another provider call.
Trailing partial JSONL records are repaired on resume and reported.

A missing or inconsistent returned model label fails closed. The completed call
is charged to that role's budget and retained as a
`rejected_model_mismatch` provider-audit line, but no replay response is
written. A later resume stops for manual review instead of silently retrying or
accepting the wrong model.

When neither output exists, the live command also preflights the complete
corpus and rejects an over-budget design before its first network call. On a
resume, journaled usage is restored and each remaining request is checked
against the remaining ledger.

The same provider flags are accepted by `llm plan` and `llm execute-openai`:

```text
--role {primary,replication,decoder}
--model MODEL
--reasoning-effort {none,low,medium,high,xhigh,max}
--api-key-env NAME
--base-url HTTPS_URL
--allow-custom-base-url
--timeout-seconds FLOAT
--max-retries INTEGER
--max-output-tokens INTEGER
--max-requests INTEGER
--max-total-tokens INTEGER
```

Defaults match the `[llm]` configuration table described in
[Configuration](configuration.md). Omitting `--allow-custom-base-url` requires
the exact official OpenAI origin; providing the flag permits the configured
credential to be sent to the reviewed custom HTTPS endpoint during live
execution.

## Adaptive experiment workflow

[`configs/openai_primary.toml`](../configs/openai_primary.toml) is an executable
pilot configuration for the primary model role;
[`configs/openai_replication.toml`](../configs/openai_replication.toml) is its
matched replication-role counterpart. Both are two-user pilots, not paper power
settings, and both keep their selected model fixed across all three LLM
information views. Each declares hard ceilings of 900 requests and 6,000,000
conservatively estimated tokens so the development calibration probe and all
three test/runtime views fit within the declared ledger. First inspect the
resolved configuration and the model declaration:

```bash
PYTHONPATH=src python -m cape_loop config validate \
  configs/openai_primary.toml
PYTHONPATH=src python -m cape_loop llm models
```

Only after reviewing the experiment size, model, reasoning effort, and hard
ceilings, authorize execution:

```bash
PYTHONPATH=src python -m cape_loop run \
  configs/openai_primary.toml \
  --execute-live
```

An OpenAI-mode configuration containing an `llm_` updater fails closed when
`--execute-live` is absent. The runner executes requests adaptively, so the next
prompt may incorporate the preceding model-written belief. Identical
content-addressed requests share a retained response and are not sent twice.

### Primary + replication suite command

Plan the complete two-role paper suite without reading a key:

```bash
PYTHONPATH=src python -m cape_loop llm evaluation-suite \
  configs/openai_primary.toml \
  configs/openai_replication.toml \
  --output-root runs
```

The command requires the checked configs to remain a matched design after
normalizing only their run name and declared role/model fields. It records both
source-file hashes and resolved-config hashes, derives two different immutable
run IDs and journal directories, retains each config's request/token ceilings,
and writes a combined index outside both run artifacts. For the checked pilot,
the deterministic upper bound is 752 provider calls per role: 576 test audit
calls, 144 development-calibration calls, and at most 32 held-out paraphrase
calls. The index records 148 calls of headroom under the 900-request ceiling
and fails before live execution if a changed design exceeds that ceiling.

An existing index path is identity-locked. A different suite/config/run
identity cannot overwrite it. Replanning may refresh only the same index while
its status remains `planned`; a completed index can be revisited for verified
reuse only with `--allow-existing`. Failed or interrupted indexes must be
preserved and handled through explicit recovery instead of being silently
replaced.

After reviewing that index and both ceilings, the same command can explicitly
authorize collection:

```bash
PYTHONPATH=src python -m cape_loop llm evaluation-suite \
  configs/openai_primary.toml \
  configs/openai_replication.toml \
  --output-root runs \
  --execute-live
```

Each role receives a new provider and budget ledger; responses and audits stay
under that role's own run/journal identity. The combined index links results but
does not merge them. If a role fails, the index records the failing role and
stops; use the ordinary failed-run recovery workflow before re-indexing
verified role runs.

By default the recovery journal is outside the checksummed run directory:

```text
<output-root>/
├── .llm-journals/<run-id>/<model-role>/
│   ├── provider-audit.jsonl
│   └── responses.jsonl
└── <run-id>/
```

Setting `[llm].journal_dir` changes the journal root. A successful run copies
the audit records actually used into `llm/provider-audit.jsonl`, writes a
credential-free `llm/provider-manifest.json`, and retains consumed exchange
records. The external journal is deliberately kept so a failed process can
resume without rebilling completed requests.

The checked-in pilots explicitly set:

```toml
base_url = "https://api.openai.com"
allow_custom_base_url = false
```

Changing the URL without also changing the Boolean fails configuration
validation. Setting the Boolean to `true` should be reviewed as a security
change because the live runner will direct the configured API credential to
that HTTPS endpoint.

### Development-only probability calibration

For every Experiment A, B, or C run containing an `llm_` updater, the default
`[llm].calibration = "temperature"` inserts a calibration stage before any
test/runtime updater evaluation:

1. select the first `calibration_users` users from the declared development
   split;
2. execute a fixed matched provenance probe spanning balanced, restricted,
   default, and suggested contexts with naturally sampled responses;
3. collect the model's raw probability vectors independently for each LLM
   updater view;
4. fit one scalar temperature per updater ID using only those development
   users' latent labels; and
5. wrap the raw provider so all subsequent test/runtime beliefs use the
   corresponding fitted temperature.

The system does not pool response-only, full-context, and provenance-aware
calibrators, and it does not use test labels in fitting. Calibration changes
only the reported probability vector, not the request, information boundary,
model, or reasoning effort. Setting `calibration = "none"` skips the
development probe and uses raw vectors directly.

The calibrated run retains the boundary explicitly:

| Artifact | Contents |
| --- | --- |
| `models/llm-calibration.json` | One fitted development temperature per updater view, fitted-split declaration, and `test_labels_used = false` |
| `llm/development-raw-responses.jsonl` | Uncalibrated model outputs from the development probe |
| `metrics/llm-development-calibration.jsonl` | Development raw and calibrated Brier scores for each probe row |
| `llm/test-raw-responses.jsonl` | Uncalibrated outputs underlying test/runtime updates |
| `llm/responses.jsonl` | Active calibrated responses consumed by experiment updaters |

Experiments B and C also write
`metrics/experiment-{b,c}-llm-raw-calibrated-terminal.jsonl` and a neighboring
manifest. These files score the cached raw and calibrated vectors for the
identical terminal prompt on the common battery; scoring performs no provider
call. In a multi-turn run, that prompt was generated on the calibrated active
history. The raw row is therefore a same-request diagnostic, not a recursively
raw counterfactual trajectory. The row sets
`full_counterfactual_rerun_required = true`, and the manifest records that
rankings and gates were not replaced. A valid raw-trajectory comparison would
need a new recursive execution and response coverage for every alternate
prompt/action branch.

When prompt retention is enabled, development requests are also retained in
`llm/development-requests.jsonl`. Both replay and OpenAI runner modes use this
same calibration boundary; a replay corpus must therefore cover the
development probe as well as the test/runtime requests. The standalone
`llm execute-openai` command only collects the supplied raw request corpus—it
does not fit temperatures by itself.

### Recover a failed live run

A failed run directory is retained with `status = "failed"` and `failure.json`;
it is not silently overwritten. After correcting the transient cause, resume
the exact same configuration with:

```bash
PYTHONPATH=src python -m cape_loop run \
  configs/openai_primary.toml \
  --execute-live \
  --resume-failed-live
```

This option requires both OpenAI mode and `--execute-live`. It verifies that the
existing destination is a failed artifact for the same resolved
configuration, moves it to:

```text
<output-root>/.failed-runs/<run-id>-attempt-NNN/
```

and creates a fresh artifact at the deterministic run path while reusing the
external audit/response journal. It refuses a completed artifact, a mismatched
configuration, or a missing failed destination. The command result reports the
archived failed-run path.

## OpenAI decoder collection

External decoder requests have their own keyless budget plan:

```bash
PYTHONPATH=src python -m cape_loop decoder-study plan-openai \
  decoder-requests.jsonl \
  --roles replication decoder \
  --max-requests 500 \
  --max-total-tokens 2050000
```

The output reports request counts and conservative maximum tokens separately
for each role, does not read a credential, and exits nonzero if either role is
over its own ceiling. Then, after review, use the live helper:

```bash
PYTHONPATH=src python -m cape_loop decoder-study execute-openai \
  decoder-requests.jsonl \
  decoder-output \
  --roles replication decoder \
  --max-requests 500 \
  --max-total-tokens 2050000 \
  --execute-live
```

The default variants are GPT-5.6 Terra (`replication`) and GPT-5.6 Luna
(`decoder`). Each role has a separate resumable journal. The output includes
`judgments.jsonl` and `execution-manifest.json`, which explicitly sets
`statistical_independence_claimed` to `false`.

For a fresh output corpus, live execution performs the same per-role
whole-corpus budget check and rejects an over-budget design before the first
network call. A resumed corpus restores journaled usage and applies the
remaining hard ledger during execution.

Different model IDs can satisfy a mechanical source-diversity audit, but two
OpenAI variants share provider infrastructure and may share training lineage,
failure modes, and institutional incentives. They do **not** prove independent
judgment. Treat them as operational replication sources; use independently
administered or human judgments for a strong independence claim.

## Replay configuration

The three LLM updater IDs are:

```text
llm_response_only
llm_full_context
llm_provenance_aware
```

For offline replay:

```toml
[experiment]
updaters = ["llm_full_context"]

[llm]
mode = "replay"
responses_file = "responses.jsonl"
calibration = "temperature"
calibration_users = 1
```

Validate the record shapes first:

```bash
PYTHONPATH=src python -m cape_loop llm validate responses.jsonl
```

The runner parses and fingerprints the complete file before building the
updater registry. During each update it reconstructs the request, looks up
`<updater-id>:<prompt-sha256>`, and rejects a different prompt hash.
`llm/input-manifest.json` retains the configured path, exact byte SHA-256,
record count, reported model labels, and calibration declaration.
`--allow-existing` requires the current corpus manifest to equal the retained
one.

LLM updaters remain forbidden in `kind = "sensitivity"` runs because each grid
point changes adaptive dynamics and can change later prompts.

## Information views

### Response only

The request contains the current prior marginals, selected option and
attributes, optional surface response, and target attribute. It omits
unselected options, ranking, default, suggestion, wording, and policy
provenance.

### Full context

The request adds semantic user-visible context: domain, displayed options,
ranking, default, suggestion, policy-neutral surface-template ID, question
type, and target attribute. It omits policy provenance and audit IDs that can
carry hidden condition labels.

### Provenance aware

The request adds structured policy provenance and instructs the writer to
interpret the response conditionally on the elicitation process.

No view contains latent preference or susceptibility. Audit-only
choice-randomness keys are also omitted.

## Hash-bound records

`LLMRequest.build(...)` hashes:

```text
system_instruction + "\n" + canonical_JSON(payload)
```

and uses a content-addressed request ID:

```text
<updater_id>:<prompt_sha256>
```

The hash binds a response to bytes constructed by CAPE-Loop. It is not a
provider signature and does not authenticate a model label.

Each response JSONL line contains schema version `1`, request ID, prompt hash,
model ID, three complete four-value probability vectors, and an optional raw
response hash. Parsing rejects unknown fields, duplicate IDs, non-finite or
out-of-range probabilities, incomplete support, and vectors that do not sum to
one within `1e-6`.

`llm validate` proves only structural validity. It does not prove coverage,
provider identity, correct information blinding, independence, or scientific
validity. Actual replay checks every consumed response against the locally
reconstructed request.

## Budgets, retries, and retained records

`max_requests` and `max_total_tokens` are hard preflight ceilings. Before a live
call, the executor reserves the conservative maximum; after a successful call,
it commits provider-reported usage when available. Resumed usage is restored
into the ledger, so restarting does not reset the ceiling. `max_output_tokens`
limits each response. `max_retries` controls retries for transient transport
errors and retryable HTTP statuses; it does not relax either budget.

For an A/B/C runner with temperature calibration, the same ledger covers both
the development probe and test/runtime requests. Size the ceiling for both;
`calibration_users` is therefore an execution-cost parameter as well as an
analysis choice.

Live requests use deterministic content-based idempotency and client request
IDs, strict Structured Outputs, and `store = false`. Audit records retain
request/body hashes, model request/return labels, provider response IDs,
timestamps, attempts, safe response metadata, usage, and replay-compatible
beliefs. They never retain the Authorization header or credential value.

HTTPS validates transport to the named endpoint; it does not establish that a
custom endpoint is OpenAI or safe. The custom-base-URL opt-in deliberately
prevents a config typo or unreviewed proxy from silently receiving the
credential.

Provider-side behavior, pricing, availability, and idempotency retention are
external conditions. Review current provider terms before collection and set
ceilings you are willing to spend; the conservative token limit is not a
currency-denominated spending cap.

## Release responsibilities

Before releasing provider-derived data, document:

- the exact model roles and any overrides;
- reasoning effort, execution dates, and software revision;
- request, retry, refusal, truncation, and parse outcomes;
- journal and artifact checksums;
- provider redistribution terms; and
- which independence claims are and are not supported.

Do not release keys, authorization headers, billing identifiers, unrelated
account metadata, or private conversation content. Replay makes analysis
repeatable from retained records; it does not make the original provider
execution deterministic or independently authenticated.
