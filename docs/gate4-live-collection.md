# Gate 4 live collection

This guide describes the two live evidence streams required by Gate 4:
independent blinded decoding of retained native memory and native end-to-end
terminal actions. The repository implements collection, validation, audit, and
resume mechanics. It does not turn provider metadata into a scientific
independence claim.

No live Anthropic, Google, or OpenAI call was made while creating this
repository. No provider judgment, native action, or Gate 4 result is checked
in. The current state is `protocol_ready`, and all generated manifests retain
`claim_status = "not_claimed"`.

## Evidence flow

```text
verified Experiment B run
├── retained blinded decoder requests
│   ├── Anthropic Claude judgment ─┐
│   └── Google Gemini judgment ────┼── researcher source review
│                                  │
└── retained native states         │
    └── OpenAI native actions ─────┘
                                       │
                                       ▼
                             gate-review import-native
                                       │
                                       ▼
                       immutable, checksum-bound review
```

The two branches test different things:

- external decoders test whether separate model families can recover the
  preference represented by a blinded native state;
- native actions test whether a model operating directly from that retained
  native state can complete every held-out terminal item; and
- the responsible-researcher review states whether the decoder sources are
  genuinely distinct enough for the exact claim being considered.

Gate 4 needs all three. A decoder judgment is not a native action, and a
provider-produced action does not establish decoder-source independence.

## Required inputs

Begin with a completed, verified Experiment B run. Its retained decoder packet
contains:

| File | Visibility | Purpose |
| --- | --- | --- |
| `decoder/external-requests.jsonl` | Decoder-visible | Blinded, content-addressed native-state requests |
| `decoder/truth-labels.researcher-only.jsonl` | Researcher-only | Development calibration and held-out scoring labels |
| `decoder/researcher-codebook.jsonl` | Researcher-only | Joins pseudonyms to eligible trajectories |

Never send the truth-label or researcher-codebook files to a model provider.
The collection commands accept only the external request file or the verified
run directory and construct their provider payloads from the allowed surface.
The request and judgment schemas are
[`external-decoder-request.schema.json`](../schemas/external-decoder-request.schema.json)
and
[`external-decoder-judgment.schema.json`](../schemas/external-decoder-judgment.schema.json).

## Distinct-family decoder pair

The default Gate 4 pair intentionally crosses provider and model family:

| Source | Pinned model | Family ID | Default key variable | Official origin |
| --- | --- | --- | --- | --- |
| Anthropic | `claude-sonnet-5` | `anthropic-claude` | `ANTHROPIC_API_KEY` | `https://api.anthropic.com` |
| Google | `gemini-3.6-flash` | `google-gemini` | `GEMINI_API_KEY` | `https://generativelanguage.googleapis.com` |

The choices and wire contracts were resolved on 2026-07-26 against
Anthropic's official [model overview](https://platform.claude.com/docs/en/about-claude/models/overview),
[Messages API](https://platform.claude.com/docs/en/api/messages/create),
[structured-output guide](https://platform.claude.com/docs/en/build-with-claude/structured-outputs),
and [effort guide](https://platform.claude.com/docs/en/build-with-claude/effort);
and Google's official [model catalog](https://ai.google.dev/gemini-api/docs/models),
[GenerateContent reference](https://ai.google.dev/api/generate-content),
[structured-output guide](https://ai.google.dev/gemini-api/docs/generate-content/structured-output),
and [thinking guide](https://ai.google.dev/gemini-api/docs/generate-content/thinking).
Recheck those sources before a later collection wave.

Crossing providers is stronger source separation than using two variants from
one provider. It still does not prove independent errors: the sources can
share public training material, evaluation conventions, prompt-induced
failure modes, or infrastructure dependencies.

### Keyless plan

Prepare every request body, verify both whole-corpus ceilings, and write a
credential-free plan:

```bash
PYTHONPATH=src python -m cape_loop decoder-study plan-distinct \
  runs/EXPERIMENT-B/decoder/external-requests.jsonl \
  --output artifacts/gate4-decoder-plan.json
```

Planning:

- sorts request IDs deterministically;
- reads no environment credential;
- constructs both provider bodies and their SHA-256 bindings;
- records model, endpoint, instance and family IDs, source references, and the
  credential-variable names but never their values;
- reserves each UTF-8 request byte as one input token, adds 512 framing tokens,
  and reserves the full output allowance;
- exits before any network activity if either source's initial corpus exceeds
  its own transport-attempt or token ceiling; and
- records the initial attempt count, maximum attempts per logical request,
  theoretical all-retry maximum, and whether every possible retry would fit.

The default ceilings are 900 actual HTTP transport attempts and 6,000,000
conservatively reserved tokens per source, with 1,024 maximum output tokens per
attempt. Every retry consumes another request-budget unit and another token
reservation; the ceiling is not a count of logical decoder items. Execution
hard-stops before an attempt that would exceed either bound. These are safety
ceilings, not billing forecasts. Override them only after reviewing the
generated plan:

```text
--max-output-tokens INTEGER
--max-requests-per-source INTEGER
--max-total-tokens-per-source INTEGER
--timeout-seconds FLOAT
--max-retries INTEGER
```

### Live collection

Export keys in the invoking shell. Do not pass them as arguments, put them in
TOML, paste them into an issue, or commit them:

```bash
export ANTHROPIC_API_KEY='...'
export GEMINI_API_KEY='...'
```

Execute only after reviewing the plan and approved ceilings:

```bash
PYTHONPATH=src python -m cape_loop decoder-study execute-distinct \
  runs/EXPERIMENT-B/decoder/external-requests.jsonl \
  artifacts/gate4-distinct-decoders \
  --execute-live
```

Both the internal provider configuration and the command-line authorization
must enable live execution. Missing `--execute-live`, a missing key, an invalid
origin, an over-budget request, an invalid response, or a model-identity
mismatch fails closed.

The command uses the same plan logic before reading either key. It stores
`collection-plan.json` in the output directory and refuses to continue if an
existing plan has a different request, model, origin, or budget identity.

### Common decoder contract

An `ExternalDecoderRequest` binds:

```text
request_id
pseudonymous_state_id
representation_id
evaluation_split
rubric_version
instruction
payload
request_sha256
```

The provider sees only the instruction and this payload:

```json
{
  "decoder_request": {
    "representation_id": "blinded-native-content-v1",
    "rubric_version": "native-profile-decoder-v1",
    "payload": {}
  }
}
```

The request adapter does not send the evaluation split, pseudonymous state ID,
local request ID, truth labels, codebook, system identity, updater identity,
memory-kind label, latent truth, or user ID. The request validator recursively
rejects forbidden fields before preparation.

Both providers must return exactly one `beliefs` object with these keys:

```json
{
  "beliefs": {
    "attribute_1": {"-2": 0.1, "-1": 0.2, "+1": 0.3, "+2": 0.4},
    "attribute_2": {"-2": 0.1, "-1": 0.2, "+1": 0.3, "+2": 0.4},
    "attribute_3": {"-2": 0.1, "-1": 0.2, "+1": 0.3, "+2": 0.4}
  }
}
```

Every probability must be finite and in `[0, 1]`; each four-value row must sum
to one within `1e-6`; missing, additional, Boolean, nonnumeric, and nonfinite
values are rejected. Provider-side structured output constrains the object
shape, and local `LLMResponse` validation independently enforces the complete
semantic probability contract.

The local binding chain is:

```text
ExternalDecoderRequest.request_sha256
  └── provider-neutral prompt_sha256
      └── exact provider request_body_sha256
          └── validated LLMResponse
              └── ExternalDecoderJudgment(request_id, request_sha256)
```

The resulting judgment declares `judgment_origin = "external_model"`, provider
instance and family IDs, a source descriptor, and both blinding flags. It is
directly consumable by `decoder-study validate`, `decoder-study analyze`, and
`gate-review import-native`.

### Anthropic wire contract

The Anthropic request is:

```text
POST https://api.anthropic.com/v1/messages
content-type: application/json
anthropic-version: 2023-06-01
x-api-key: read only at authorized execution
```

Its body uses:

- `model = "claude-sonnet-5"`;
- `max_tokens = 1024` by default;
- `thinking.type = "disabled"` because this is a bounded classification task;
- the decoder instruction in `system`;
- canonical decoder payload JSON in one user text block; and
- `output_config.effort = "low"` with
  `output_config.format.type = "json_schema"`.

The integration deliberately omits non-default sampling parameters because
Claude Sonnet 5 rejects them. The response is accepted only when it identifies
itself as a Messages API assistant message, has a nonempty message ID, returns
the exact configured model ID, ends with `stop_reason = "end_turn"`, and
contains exactly one text output block whose JSON passes the common contract.

Usage accounting includes input, cache-creation input, cache-read input, and
output token fields when present. `request-id` is retained as safe provider
metadata.

### Gemini wire contract

The Google request is:

```text
POST https://generativelanguage.googleapis.com/v1beta/models/
     gemini-3.6-flash:generateContent
content-type: application/json
x-goog-api-key: read only at authorized execution
```

Its body uses:

- the decoder instruction in `systemInstruction`;
- canonical decoder payload JSON in one user part;
- `maxOutputTokens = 1024` by default;
- `thinkingConfig.thinkingLevel = "low"`; and
- `generationConfig.responseFormat.text` with
  `mimeType = "application/json"` and the bound `schema`.

The Gemini 3.6 request deliberately omits `candidateCount`, `temperature`, and
other sampling controls instead of relying on obsolete fields or unsupported
defaults.

The response is rejected when `promptFeedback` reports a block. Otherwise it
must include `modelVersion`, `responseId`, exactly one candidate,
`finishReason = "STOP"`, model-role content, and exactly one non-thinking text
part whose JSON passes the common contract. If `modelStatus.modelStage` is
present, it must be `STABLE`. The returned model must be the configured ID,
optionally with Google's `models/` resource prefix.

`usageMetadata.totalTokenCount` supplies actual usage when present.
`x-request-id` is retained as safe provider metadata.

### Origin and credential locks

By default each key can be sent only to its provider's exact official HTTPS
origin. User information, a port, a path, a query, a fragment, or a different
host is rejected.

A reviewed proxy needs two explicit controls:

```bash
--anthropic-base-url https://anthropic-proxy.example.org \
--allow-custom-anthropic-base-url \
--anthropic-api-key-env CAPE_LOOP_ANTHROPIC_PROXY_KEY
```

The equivalent Gemini flags are `--gemini-base-url`,
`--allow-custom-gemini-base-url`, and `--gemini-api-key-env`. A custom host
cannot use the default provider credential-variable name. This prevents a URL
override alone from redirecting a general first-party credential.

Credentials are read only after live authorization and budget reservation,
immediately before HTTP execution. Authorization headers never appear in a
prepared request, plan, judgment, manifest, or audit. Provider bodies and
errors are recursively redacted if a response unexpectedly echoes a secret.

### Retries, accounting, and resume

Each logical request permits four retries by default, for at most five physical
attempts, with bounded exponential backoff and jitter. HTTP 408, 409, 425, 429,
and 5xx responses are retryable; other HTTP failures fail immediately.
`Retry-After` in seconds or HTTP-date form is honored. Every attempted
transport is charged one request and its conservative token reservation unless
settled provider usage supplies the charged token count.

The output directory contains:

| File | Contents |
| --- | --- |
| `collection-plan.json` | Exact keyless request/model/origin/budget plan |
| `transport-attempts.jsonl` | Fsynced `started`/`settled` record for every physical HTTP attempt |
| `provider-audit.jsonl` | Safe provider metadata, usage, raw response, validated replay response, and embedded judgment |
| `judgments.jsonl` | Import-compatible `ExternalDecoderJudgment` rows |
| `execution-manifest.json` | File digests, source-design validation, budget summary, and no-claim declarations |

The collector also leaves the opaque
`.external-decoder-command.lock` and
`.external-decoder-collection.lock` markers. Do not delete them: Gate import
requires both and acquires them in the collector's outer-to-inner order before
reading the five evidence files.

An exclusive output-directory lock prevents concurrent collectors from
reconciling or appending the same corpus. Before every physical HTTP call, a
durable `started` attempt event is flushed and fsynced; its `settled` event
records the outcome and charged usage. An unresolved `started` event has
unknown billing status and blocks automatic resume for manual review. If a
process ends after a settled transient attempt but before the logical request
obtains an embedded accepted/rejected final audit, a later process also blocks
automatic retry. Only a crash-truncated final JSONL tail is repaired
automatically; earlier corruption fails closed.

For every successful call, the complete audit line is flushed and fsynced
before its judgment line. On resume, an accepted matching audit can reconstruct
a missing judgment without another paid request. Existing attempts and audits
are rebound to the current request hash, prompt hash, body hash, model, family,
source descriptor, and client request ID; mismatches are rejected.

A paid response with the wrong returned model is stored as
`rejected_identity_mismatch`, produces no judgment, and stops future automatic
resume for manual review. Any malformed or partial interior JSONL record fails
closed instead of being silently accepted; only the crash-truncated final tail
described above is repaired.

## Native end-to-end actions

Gate 4 also needs actions emitted from retained native memory, rather than a
local deterministic projection. The checked implementation defines the native
system as `cape-loop-openai-native-agent-v1`: OpenAI `gpt-5.6-sol` receives the
complete retained native state plus the held-out terminal suite and returns one
strictly bound action per item.

Plan against a verified Experiment B run without reading `OPENAI_API_KEY`:

```bash
PYTHONPATH=src python -m cape_loop native-action plan-openai \
  runs/EXPERIMENT-B \
  --output artifacts/gate4-native-action-plan.json
```

The plan binds each eligible trajectory to its native-state digest, held-out
suite digest, and exact request-body digest. It uses the same default ceiling
of 900 physical transport attempts and 6,000,000 conservatively reserved
tokens, with at most 4,096 output tokens per attempt. Retries consume both
budgets independently.

After reviewing the plan and exporting `OPENAI_API_KEY`, authorize collection:

```bash
PYTHONPATH=src python -m cape_loop native-action execute-openai \
  runs/EXPERIMENT-B \
  artifacts/gate4-native-actions \
  --execute-live
```

The output contains:

| File | Contents |
| --- | --- |
| `requests.jsonl` | Content-addressed native-state and terminal-suite requests |
| `collection-plan.json` | Exact source/model/origin/retry/budget identity |
| `transport-attempts.jsonl` | Fsynced `started`/`settled` record for every physical HTTP attempt |
| `provider-audit.jsonl` | OpenAI model, response, usage, timing, and embedded action records |
| `native-actions.jsonl` | Gate-import-compatible native terminal action records |
| `execution-manifest.json` | Source run, native system/version, file hashes, budgets, and no-claim status |

The opaque `.collection.lock` marker is likewise a required operational part
of the native collection and is held through Gate validation and review
creation.

The provider must return every exact item ID, item hash, wording-template ID,
and question type. Choice items must select a displayed option; direct probes
must return a declared direction. The action record binds the native state,
system/version, suite, execution trace, and all actions. Accepted audit records
are written before actions, and matching accepted audits repair an interrupted
action append without a second call.

The native-action collector also holds an exclusive output lock. An unresolved
or incompletely audited physical attempt requires manual review before retry,
and a rejected charged response must be preserved in a separately reviewed
recovery directory. Both collection commands reject plan or result outputs
inside the immutable source run before creating their destination.

This is a real model-mediated action adapter, not either deterministic
reference adapter. It still evaluates the declared OpenAI-backed native system;
do not generalize its result to a different deployment or memory runtime.

## Responsible-researcher source review

`execute-distinct` verifies that two provider/model family labels cover the
requests. It deliberately writes:

```text
distinct_provider_model_families = true
statistical_independence_claimed = false
responsible_researcher_source_review_required = true
claim_status = "not_claimed"
```

An eligible Gate 4 import additionally needs a researcher-authored
`decoder-source-review.json` matching
[`decoder-source-review.schema.json`](../schemas/decoder-source-review.schema.json).
It must bind the exact request and judgment file digests and contain:

- review ID, responsible researcher ID, and review timestamp;
- an overall `eligible_distinct_sources` or `not_eligible` decision;
- one assessment per decoder instance, including family, origin,
  `eligible_for_gate4`, and concrete dependency notes; and
- one assessment for every co-occurring source pair, with lexicographically
  sorted instance IDs, a scope-specific genuine-distinctness decision, and a
  rationale.

The review should address provider ownership, model-family lineage, serving
infrastructure, likely training-data overlap, common prompt/schema effects,
calibration/adjudication dependencies, and the exact scientific claim. The
software validates completeness, hashes, and internal consistency. It cannot
verify the researcher's investigation or make the determination automatically.

Keep `decoder/truth-labels.researcher-only.jsonl` access-controlled. The source
review and native-action evidence may reveal operational metadata even though
the model-facing decoder payload is blinded.

## Import and verify

After collecting both evidence streams and completing the source review,
import them beside, never inside, the immutable source run:

```bash
PYTHONPATH=src python -m cape_loop gate-review import-native \
  runs/EXPERIMENT-B \
  runs/EXPERIMENT-B/decoder/external-requests.jsonl \
  artifacts/gate4-distinct-decoders/judgments.jsonl \
  runs/EXPERIMENT-B/decoder/truth-labels.researcher-only.jsonl \
  artifacts/gate4-native-actions \
  decoder-source-review.json \
  artifacts/GATE4-REVIEW \
  --external-collection-dir artifacts/gate4-distinct-decoders

PYTHONPATH=src python -m cape_loop gate-review verify \
  artifacts/GATE4-REVIEW
```

The importer independently verifies the source run, exact retained request and
truth packet, decoder calibration and test coverage, source-review bindings,
native-state and suite bindings, and recorded action coverage. For the selected
automated path it rebuilds both collection plans, validates every physical
attempt, provider audit, judgment/action row, actual model and origin, and both
portable execution manifests. It holds the decoder collector's outer and inner
locks and the native collector lock through validation, input hashing, and
review creation. It creates a new checksum-bound review and never mutates the
Experiment B run.

Import accepts lower collection budgets but rejects upward overrides beyond
the approved ceilings: per external source, 900 attempts, 6,000,000 total
tokens, and 1,024 output tokens; for native actions, 900 attempts, 6,000,000
total tokens, and 4,096 output tokens.

Retain the exact request, truth-label, and source-review files plus all five
decoder and six native collection evidence files named by the review's
digests. `gate-review verify` verifies the review directory itself; full
recomputation also needs the verified source run and those exact inputs. A
checksum-valid review is still `not_claimed` until the authors review the
analysis, limitations, and claim scope.

For reviewed human or other generic decoder evidence, replace
`--external-collection-dir ...` with
`--allow-reviewed-generic-decoders`. This explicit alternative retains
blinding, coverage, analysis, and responsible-researcher review checks, but
records no official automated-provider collection provenance.

See [External evidence boundaries](external-evidence.md) for the general
admission policy and [LLM exchange and live execution](llm-exchange.md) for the
profile-writer and replay workflows.
