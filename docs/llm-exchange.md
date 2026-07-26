# LLM exchange and live execution

CAPE-Loop supports three model-execution modes behind the same hash-bound
request/response contract:

```text
replay:      retained response JSONL ──► prompt-hash check ──► profile update
openai:      CAPE request ──► direct Responses API ──► audit journal ──► profile update
openrouter:  CAPE request ──► gateway Chat Completions ──► route audit ──► profile update
runner:      development raw outputs ──► temperature fit ──► calibrated test/runtime update
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

## Direct OpenAI safety boundary

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

## OpenRouter gateway workflow

OpenRouter uses a dedicated `OpenRouterChatProvider`, not the direct OpenAI
provider with a changed base URL. It prepares
`POST https://openrouter.ai/api/v1/chat/completions` requests with:

```text
model = one exact canonical author/model slug
stream = false
response_format.type = "json_schema"
response_format.json_schema.strict = true
max_completion_tokens = configured output ceiling
provider = explicit routing/privacy preferences
X-OpenRouter-Metadata: enabled
X-OpenRouter-Cache: false
```

The response's structured JSON remains text in
`choices[0].message.content`; CAPE-Loop parses and validates it locally. The
request selects no model-fallback array. Aliases beginning with `~`,
colon-suffixed route variants, `-latest` labels, and `openrouter/auto` are
rejected so one model switch is exactly one value:

```bash
--model google/gemini-3.6-flash
```

Select and re-verify the canonical slug and endpoint support in OpenRouter's
[model catalog](https://openrouter.ai/models) before a collection wave.

Plan a static corpus without reading `OPENROUTER_API_KEY`:

```bash
PYTHONPATH=src python -m cape_loop llm plan-openrouter \
  requests.jsonl \
  --model google/gemini-3.6-flash \
  --upstream-provider google-ai-studio \
  --max-requests 500 \
  --max-total-tokens 2050000
```

The optional upstream slug is placed in both `provider.order` and
`provider.only`. Use a full endpoint-variant slug when a region or specialized
endpoint matters. Without it, OpenRouter may choose among eligible providers,
but the selected route is still retained. Defaults set
`allow_fallbacks = false`, `require_parameters = true`, and
`data_collection = "deny"`; `--zdr` adds a zero-data-retention requirement.
The corresponding flags `--allow-fallbacks`,
`--allow-unsupported-parameters`, and `--data-collection allow` weaken those
defaults deliberately. The data-collection and ZDR fields filter using
OpenRouter's endpoint classifications; retain the values, but verify current
provider policies independently rather than treating the flags as a legal or
institutional guarantee.

After reviewing the model, endpoint, route policy, current pricing, and hard
ceilings:

```bash
export OPENROUTER_API_KEY='...'
PYTHONPATH=src python -m cape_loop llm execute-openrouter \
  requests.jsonl responses.jsonl openrouter-provider-audit.jsonl \
  --model google/gemini-3.6-flash \
  --upstream-provider google-ai-studio \
  --max-requests 500 \
  --max-total-tokens 2050000 \
  --execute-live
```

`--http-referer` and `--app-title` add optional attribution headers. According
to OpenRouter's [app-attribution documentation](https://openrouter.ai/docs/app-attribution),
the referer is the attribution identifier and a title alone does not create an
app entry. These headers label usage; they do not authenticate provider
provenance.

An accepted response must:

1. be one non-streaming `chat.completion` with one stopped choice;
2. return the exact requested model;
3. include opted-in `openrouter_metadata`;
4. report the requested model and `strategy = "direct"`;
5. mark exactly one upstream endpoint selected, with its model equal to the
   top-level returned model;
6. use routing attempt one when fallbacks are disabled;
7. not be an OpenRouter response-cache hit; and
8. have no nonempty router pipeline that materially transformed the request or
   response.

OpenRouter documents that cache hits omit `openrouter_metadata`; consequently,
a replayed cache response fails the required-metadata check even before the
explicit cache-status acceptance check can apply.

A completely parsed but unacceptable response is charged and retained as
`rejected_openrouter_identity`, but it does not become replay input. Accepted
audits are written before replay responses, and resume revalidates the retained
raw response, route metadata, model identity, cache status, and transformation
policy before reuse.

The audit schema separates gateway from upstream identity:

```text
provider = "openrouter"
gateway = "openrouter"
model_requested / model_returned
upstream_provider / upstream_model
routing_strategy / routing_attempt / routing_metadata
provider_response_id / generation_id / cache_status
usage / timing / request and response hashes
raw_response / replay_response
first_party_origin_claimed = false
```

`routing_metadata` is additive and should be decoded permissively. The adapter
captures the body response ID and `X-Generation-Id` response header when
present; it does not itself call
`GET /api/v1/generation`. The raw response `usage` object is retained,
including prompt/completion/total counts, token-detail objects, and cost fields
when OpenRouter returns them. The budget commits a valid `total_tokens`, falls
back to prompt plus completion counts, and otherwise charges the conservative
reservation; missing optional detail fields do not erase that reservation.
OpenRouter's official
[Chat Completions reference](https://openrouter.ai/docs/api/api-reference/chat/send-chat-completion-request),
[Structured Outputs guide](https://openrouter.ai/docs/guides/features/structured-outputs),
[provider-routing guide](https://openrouter.ai/docs/guides/routing/provider-selection),
[router-metadata guide](https://openrouter.ai/docs/guides/features/router-metadata),
[response-caching guide](https://openrouter.ai/docs/guides/features/response-caching),
[usage-accounting guide](https://openrouter.ai/docs/cookbook/administration/usage-accounting),
and [generation lookup](https://openrouter.ai/docs/api/api-reference/generations/get-generation)
define the external wire behavior. Endpoint support and provider policies can
change; recheck them before collection.

For adaptive execution, change only the exact `model` line in
[`configs/openrouter_gemini.toml`](../configs/openrouter_gemini.toml), validate
the file, and run it with explicit authorization:

```bash
PYTHONPATH=src python -m cape_loop config validate \
  configs/openrouter_gemini.toml
PYTHONPATH=src python -m cape_loop run \
  configs/openrouter_gemini.toml --execute-live
```

The OpenRouter runner journal adds a gateway segment:

```text
<output-root>/.llm-journals/<run-id>/openrouter/<model-role>/
├── provider-audit.jsonl
└── responses.jsonl
```

Used audits and a credential-free provider manifest are copied into the
successful run in the same `llm/` locations used by direct OpenAI execution.
No live result is checked in.

For reviewed-generic external decoder collection, one command can journal
multiple exact models separately:

```bash
PYTHONPATH=src python -m cape_loop decoder-study plan-openrouter \
  decoder-requests.jsonl \
  --model anthropic/claude-sonnet-4.5 \
  --additional-model google/gemini-3.6-flash

PYTHONPATH=src python -m cape_loop decoder-study execute-openrouter \
  decoder-requests.jsonl openrouter-decoder-output \
  --model anthropic/claude-sonnet-4.5 \
  --additional-model google/gemini-3.6-flash \
  --execute-live
```

The output has one `journals/<model-digest>/` directory per model,
`judgments.jsonl`, and `execution-manifest.json`. Different model labels or
reported upstream routes behind the same OpenRouter gateway do not prove
independent errors. The manifest consequently fixes
`first_party_origin_claimed = false`, `strict_gate4_eligible = false`, and
`statistical_independence_claimed = false`. These sources may enter the
explicit reviewed-generic decoder branch, but cannot replace strict Gate 4's
direct first-party Anthropic/Gemini decoder collection.

## Direct OpenAI adaptive experiment workflow

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
`llm/development-requests.jsonl`. Replay, direct OpenAI, and OpenRouter runner
modes use this same calibration boundary; a replay corpus must therefore cover
the development probe as well as the test/runtime requests. The standalone
`llm execute-openai` and `llm execute-openrouter` commands collect only their
supplied raw request corpus—they do not fit temperatures by themselves.

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

This option requires direct OpenAI or OpenRouter mode plus `--execute-live`. It
verifies that the existing destination is a failed artifact for the same
resolved configuration, moves it to:

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

## Distinct-family Gate 4 collection

The default Gate 4 decoder workflow crosses both provider and model family:
Anthropic `claude-sonnet-5` and stable Google `gemini-3.6-flash`. Plan the exact
two-source corpus without reading either credential:

This strict path calls the two first-party APIs directly. An OpenRouter decoder
collection remains a shared-gateway, reviewed-generic source even when its
router metadata names Anthropic or Google; it cannot substitute for either
first-party collection and does not establish statistically independent
errors.

```bash
PYTHONPATH=src python -m cape_loop decoder-study plan-distinct \
  runs/EXPERIMENT-B/decoder/external-requests.jsonl \
  --output artifacts/gate4-decoder-plan.json
```

The planner locks the official Anthropic and Google origins, constructs strict
provider-specific JSON-schema requests, records every request-body hash, and
checks the default ceilings of 900 physical transport attempts and 6,000,000
conservatively charged tokens independently for each source. Every retry
consumes another attempt-budget unit. It reports `credential_read = false`;
the environment-variable names are recorded, but their values are neither read
nor retained.

After reviewing the plan, export keys only in the invoking shell:

```bash
export ANTHROPIC_API_KEY='...'
export GEMINI_API_KEY='...'

PYTHONPATH=src python -m cape_loop decoder-study execute-distinct \
  runs/EXPERIMENT-B/decoder/external-requests.jsonl \
  artifacts/gate4-distinct-decoders \
  --execute-live
```

The command writes the exact plan, a durable physical-attempt journal, a
provider audit, import-compatible judgments, and an execution manifest with
file digests. A `started` event is fsynced before each HTTP request. Accepted
audits are durably appended before judgments, so an interrupted judgment append
can be reconstructed without another accepted call. Returned provider/model
identities, request and prompt hashes, strict probabilities, blinding fields,
usage, and resume configuration are all validated. An unresolved attempt or a
returned-model mismatch requires manual review.

The official origins are `https://api.anthropic.com` and
`https://generativelanguage.googleapis.com`. A custom route needs both its
provider-specific `--allow-custom-...-base-url` flag and a dedicated
non-default credential-variable name; changing only the URL is insufficient.

Distinct provider/family metadata is necessary but does not establish
statistically independent errors. The execution manifest explicitly preserves
`statistical_independence_claimed = false` and requires a responsible
researcher to assess provider ownership, lineage, infrastructure, likely
training overlap, common prompt/schema effects, and the intended claim scope.

Gate 4 also requires end-to-end actions from retained native memory. Plan and
execute the OpenAI-backed native system separately:

```bash
PYTHONPATH=src python -m cape_loop native-action plan-openai \
  runs/EXPERIMENT-B \
  --output artifacts/gate4-native-action-plan.json

PYTHONPATH=src python -m cape_loop native-action execute-openai \
  runs/EXPERIMENT-B \
  artifacts/gate4-native-actions \
  --execute-live
```

This action workflow sends the complete retained native state and held-out
terminal suite to `gpt-5.6-sol`, then requires one schema-bound action for every
item. It is a model-mediated native action adapter, not a deterministic
belief-to-action projection. Its `transport-attempts.jsonl` is written around
each physical request, and its accepted audit is written before
`native-actions.jsonl`; together they support the same conservative
crash-recovery and no-duplicate-accepted-call reconciliation.

The exact Anthropic and Gemini request/response fields, origin and key guards,
budget accounting, recovery behavior, native-action artifacts, researcher
attestation, and final `gate-review import-native` command are documented in
[Gate 4 live collection](gate4-live-collection.md).

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
is sent as the provider-side generation ceiling, but it is not trusted as a
wire-size guarantee. The direct-provider and OpenRouter transports read at most
16 MiB plus one overflow-detection byte for either success or HTTP-error
bodies. An oversized body is not retained or reflected and is charged
conservatively without a retry. `max_retries` controls only retry cases that
the selected transport treats as unambiguous and transient; it does not relax
either budget.

For an A/B/C runner with temperature calibration, the same ledger covers both
the development probe and test/runtime requests. Size the ceiling for both;
`calibration_users` is therefore an execution-cost parameter as well as an
analysis choice.

Direct OpenAI requests use deterministic content-based idempotency/client
request IDs, strict Structured Outputs, and `store = false`. OpenRouter
requests also derive a stable local client request ID, but Chat Completions has
no documented general idempotency guarantee; ambiguous transport outcomes stop
for manual review instead of being retried automatically. Its accepted
responses must be direct, uncached, untransformed, and identity-consistent.
Audit records retain request/body hashes, model request/return labels, provider
response IDs, timestamps, attempts, safe response metadata, usage, and
replay-compatible beliefs. OpenRouter audits additionally retain selected
upstream and routing metadata. Neither path retains the Authorization header or
credential value.

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
