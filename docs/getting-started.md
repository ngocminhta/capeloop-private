# Getting started

CAPE-Loop's executable core requires Python 3.11 or newer and has no runtime
dependency outside the standard library. Inspection, planning, validation,
replay, and deterministic simulation make no provider calls and require no
credentials. Only commands carrying the explicit `--execute-live` flag can
perform a live provider request.

Run them from the repository root.

## Check the checkout

```bash
python --version
PYTHONPATH=src python -m cape_loop doctor
PYTHONPATH=src python -m cape_loop --help
```

`doctor` reports the package version, Python version, domain registry, dependency
policy, and whether the core requires a network. It does not validate a run
configuration or inspect an artifact directory.

Run the complete offline test suite with:

```bash
make check
```

The equivalent direct commands are:

```bash
PYTHONPATH=src python -m cape_loop doctor
PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py' -v
```

A passing suite validates the checked-in software fixtures. It is not evidence
for a paper hypothesis.

## Supported commands

The current CLI surface is:

```text
doctor
config validate CONFIG
run CONFIG [--output-root DIR] [--allow-existing] [--execute-live]
           [--resume-failed-live]
verify RUN_DIR
schema export [DEST]
artifact freeze RUN_DIR ARCHIVE
artifact verify ARCHIVE
llm models
llm validate RESPONSES.jsonl
llm plan REQUESTS.jsonl [provider options]
llm execute-openai REQUESTS.jsonl RESPONSES.jsonl AUDIT.jsonl
                   [provider options] --execute-live
llm plan-openrouter REQUESTS.jsonl [OpenRouter options]
llm execute-openrouter REQUESTS.jsonl RESPONSES.jsonl AUDIT.jsonl
                       [OpenRouter options] --execute-live
decoder-study validate REQUESTS JUDGMENTS
decoder-study analyze REQUESTS JUDGMENTS TRUTH_LABELS
decoder-study plan-openai REQUESTS [provider options]
decoder-study execute-openai REQUESTS OUTPUT_DIR
                             [provider options] --execute-live
decoder-study plan-openrouter REQUESTS [OpenRouter options]
decoder-study execute-openrouter REQUESTS OUTPUT_DIR
                                 [OpenRouter options] --execute-live
decoder-study plan-distinct REQUESTS [--output PLAN]
decoder-study execute-distinct REQUESTS OUTPUT_DIR --execute-live
native-action plan-openai RUN_DIR [--output PLAN]
native-action execute-openai RUN_DIR OUTPUT_DIR --execute-live
gate-review import-native RUN REQUESTS JUDGMENTS TRUTH NATIVE_COLLECTION \
  SOURCE_REVIEW OUTPUT \
  (--external-collection-dir DIR | --allow-reviewed-generic-decoders)
gate-review verify REVIEW_DIR
human-study generate OUTPUT_DIR [--assignment-id ID] [--seed INTEGER]
human-study analyze RESPONSES CODEBOOK [--output OUTPUT]
correction-debt run OUTPUT --stage-gate-authorized
```

Invoke a command as `PYTHONPATH=src python -m cape_loop ...`. Packaging also
declares a `cape-loop` console script, but the source-tree form is the
reproducibility baseline.

## Validate a configuration

Configurations are strict, schema-versioned TOML:

```bash
PYTHONPATH=src python -m cape_loop config validate configs/smoke.toml
```

Validation prints the fully resolved configuration as JSON. It rejects unknown
keys, unknown component identifiers, invalid ranges, and combinations that the
selected experiment runner would ignore. It does not run an experiment or
establish scientific validity.

The checked-in configurations are:

| File | Runner |
| --- | --- |
| `configs/smoke.toml` | Experiment A provenance audit across three prior-strength strata |
| `configs/closed_loop.toml` | Experiment B closed loop |
| `configs/evaluation.toml` | Experiment C evaluation validity |
| `configs/sensitivity.toml` | Simulator sensitivity grid |
| `configs/sensitivity_full.toml` | Broader alternative-model robustness grid |
| `configs/openai_primary.toml` | GPT-5.6 Sol Experiment A live pilot |
| `configs/openai_replication.toml` | GPT-5.6 Terra matched live pilot |
| `configs/openrouter_gemini.toml` | OpenRouter-routed Gemini Experiment A live pilot |

See [Configuration](configuration.md) before changing a value. These files
contain reference software settings, not preregistered paper parameters.
The OpenAI and OpenRouter configurations are two-user pilot designs, not paper
power settings or completed runs. Each keeps its selected model fixed across
the three LLM information views and declares hard ceilings of 900 requests and
6,000,000 conservatively estimated tokens. The ceiling covers a one-user
development calibration probe plus all three test/runtime views.

## Run and verify the smoke configuration

```bash
PYTHONPATH=src python -m cape_loop run configs/smoke.toml
```

The smoke configuration intentionally crosses three prior-strength strata and
materializes full audit/reliability artifacts. On a typical laptop it can take
roughly one to two minutes; `make check` is the faster software-only check.

The command prints JSON containing `run_dir`, `reused`, and a summary. The
directory name combines `run.name` with the first 12 characters of the resolved
configuration's SHA-256 digest.

Verify the path printed by the command:

```bash
PYTHONPATH=src python -m cape_loop verify runs/<run-id>
```

Verification checks `SHA256SUMS`, rejects unsafe or duplicate checksum paths,
and rejects both missing listed files and extra unlisted files. It also requires
a complete manifest whose run ID matches the directory, a current-schema
`config.resolved.json` whose digest matches the manifest, and
`metrics/summary.json`. It does not validate every event/metric record schema or
decide a scientific gate.

Runs do not overwrite an existing deterministic directory. To return an
existing result, all of the following must hold:

- `--allow-existing` was supplied;
- checksums verify;
- `metrics/summary.json` exists;
- `manifest.json` has the SHA-256 of the current executable source tree; and
- for an LLM replay run, `llm/input-manifest.json` exactly matches the
  currently configured response corpus.

Example:

```bash
PYTHONPATH=src python -m cape_loop run configs/smoke.toml --allow-existing
```

If those checks fail, reuse fails rather than repairing or replacing the
directory.

## Plan and execute model evaluation

Show the repository's declared suite:

```bash
PYTHONPATH=src python -m cape_loop llm models
```

The defaults are GPT-5.6 Sol at medium reasoning for the `primary`
profile-writer role, GPT-5.6 Terra at medium reasoning for `replication`, and
GPT-5.6 Luna at low reasoning for the `decoder`/pilot role. Keep the model and
reasoning effort fixed across response-only, full-context, and
provenance-aware views within a causal comparison.

Terra is a GPT-5.6 model-variant/tier replication, not a distinct-family
robustness check. Plan both checked role configs and their isolated outputs in
one credential-free command:

```bash
PYTHONPATH=src python -m cape_loop llm evaluation-suite \
  configs/openai_primary.toml configs/openai_replication.toml \
  --output-root runs
```

After reviewing the combined index and both configs' hard ceilings, add
`--execute-live` to authorize the two isolated runs.

For an existing static request corpus, perform keyless preflight first:

```bash
PYTHONPATH=src python -m cape_loop llm plan requests.jsonl \
  --role primary \
  --max-requests 900 \
  --max-total-tokens 6000000
```

This constructs request bodies, reports their hashes and conservative token
reservation, and exits nonzero when the corpus exceeds a declared ceiling. It
does not read `OPENAI_API_KEY`. After reviewing the plan and provider pricing,
export the key in your shell and authorize execution:

```bash
export OPENAI_API_KEY='...'
PYTHONPATH=src python -m cape_loop llm execute-openai \
  requests.jsonl responses.jsonl provider-audit.jsonl \
  --role primary \
  --max-requests 900 \
  --max-total-tokens 6000000 \
  --execute-live
```

The key is read only from the environment immediately before a request and is
never retained. The executor rejects a fresh over-budget corpus before its
first network call, writes each provider audit record before its
replay-compatible response, and resumes already completed request IDs from
those files.

The default endpoint is locked to the official
`https://api.openai.com` origin. `--base-url` with a different host, port, or
path is rejected unless the same command also carries
`--allow-custom-base-url`. That opt-in means an authorized live execution sends
the configured environment credential to the reviewed custom HTTPS endpoint:

```bash
PYTHONPATH=src python -m cape_loop llm plan requests.jsonl \
  --base-url https://reviewed-provider.example \
  --api-key-env CAPE_LOOP_PROXY_KEY \
  --allow-custom-base-url
```

Planning remains keyless, but the custom origin must name a dedicated
credential variable rather than `OPENAI_API_KEY`. A subsequent
`llm execute-openai` with those same endpoint and credential-name flags does
send that credential there. Treat the flag as a security approval, not a
routine compatibility switch. Decoder `plan-openai` and `execute-openai` use
the same guard.

Adaptive experiments should use the runner:

```bash
PYTHONPATH=src python -m cape_loop config validate \
  configs/openai_primary.toml
PYTHONPATH=src python -m cape_loop run \
  configs/openai_primary.toml \
  --execute-live
```

The runner refuses an OpenAI-mode LLM updater without `--execute-live`. Its
durable audit-first journal defaults to
`<output-root>/.llm-journals/<run-id>/<model-role>/`, outside the incomplete run
artifact, so completed requests survive interruption.

Adaptive runs configure the same boundary in TOML. The checked-in pilots use:

```toml
base_url = "https://api.openai.com"
allow_custom_base_url = false
```

Changing `base_url` requires the separate Boolean opt-in and directs the
configured credential to that reviewed HTTPS endpoint during live execution.

Both checked-in pilots declare:

```toml
[llm]
calibration = "temperature"
calibration_users = 1
```

Before test/runtime evaluation, the runner executes a fixed matched provenance
probe for that declared development user and collects raw probabilities for
each LLM updater view. It fits one temperature per view using development
labels only, then applies the matching calibrator to subsequent outputs. It
never uses test labels for fitting. Use `calibration = "none"` only as an
explicit protocol choice when raw probabilities are intended.

After a successful calibrated run, inspect:

```text
models/llm-calibration.json
llm/development-raw-responses.jsonl
metrics/llm-development-calibration.jsonl
llm/test-raw-responses.jsonl
llm/responses.jsonl
```

The metrics file compares raw and calibrated Brier scores on development probe
rows. `llm/test-raw-responses.jsonl` preserves the uncalibrated values
underlying runtime/test updates, while `llm/responses.jsonl` contains the
calibrated responses consumed by the experiment. These artifacts make the
development/test boundary auditable.

If that attempt fails, first inspect `failure.json`. After correcting the
cause, resume the same configuration:

```bash
PYTHONPATH=src python -m cape_loop run \
  configs/openai_primary.toml \
  --execute-live \
  --resume-failed-live
```

The command verifies that the existing deterministic destination is a failed
artifact for the same configuration, moves it to
`<output-root>/.failed-runs/<run-id>-attempt-NNN/`, creates a fresh run
artifact, and reuses the external provider journal. It refuses to move a
completed or mismatched run.

No live call was made to create this repository. The checked-in implementation
and pilot configurations are execution capability, not empirical evidence.

## Plan and execute through OpenRouter

OpenRouter is a separate first-class gateway mode using
`https://openrouter.ai/api/v1/chat/completions`. It is not configured by
pointing the direct OpenAI provider at a custom URL. Start with a keyless static
plan:

```bash
PYTHONPATH=src python -m cape_loop llm plan-openrouter \
  requests.jsonl \
  --model google/gemini-3.6-flash \
  --upstream-provider google-ai-studio \
  --max-requests 500 \
  --max-total-tokens 2050000
```

The planner reads no credential. It prints the exact model, endpoint, provider
preferences, per-request body hashes, conservative token reservation, and
whether the corpus is within both hard ceilings. To switch models, change only
the canonical `--model author/model` value. Aliases, `openrouter/auto`,
colon-suffixed routes, and `-latest` labels are rejected.

After reviewing the model's current OpenRouter endpoints and pricing, export
the gateway key in the invoking shell and explicitly authorize the call:

```bash
export OPENROUTER_API_KEY='...'
PYTHONPATH=src python -m cape_loop llm execute-openrouter \
  requests.jsonl \
  responses.jsonl \
  openrouter-provider-audit.jsonl \
  --model google/gemini-3.6-flash \
  --upstream-provider google-ai-studio \
  --max-requests 500 \
  --max-total-tokens 2050000 \
  --execute-live
```

The key is never written to either output. The adapter submits strict JSON
Schema, disables OpenRouter response caching, requests router metadata, and
defaults to `provider.allow_fallbacks = false`,
`provider.require_parameters = true`, and
`provider.data_collection = "deny"`. `--zdr` further requires a
zero-data-retention endpoint. `--allow-fallbacks`,
`--allow-unsupported-parameters`, or `--data-collection allow` deliberately
weakens those defaults and must be reported as a protocol change.
`--http-referer` and `--app-title` control optional public app attribution; they
are not provenance authentication.

The accepted response must contain exactly one selected upstream endpoint and
must retain the requested/returned model, selected upstream provider/model,
routing strategy and attempt, additive router metadata, provider response ID,
`X-Generation-Id` when present, cache status, usage, timings, hashes, redacted
raw response, and replay-compatible beliefs. Model changes, unexpected
fallback, cache hits, and nonempty transformation pipelines fail closed and
never become usable replay rows. A completely parsed model/routing identity
mismatch is retained as a rejected audit; a response missing required metadata
can fail earlier during parsing. OpenRouter documents these fields in its
[Chat Completions reference](https://openrouter.ai/docs/api/api-reference/chat/send-chat-completion-request),
[provider-routing guide](https://openrouter.ai/docs/guides/routing/provider-selection),
[router-metadata guide](https://openrouter.ai/docs/guides/features/router-metadata),
and [usage-accounting guide](https://openrouter.ai/docs/cookbook/administration/usage-accounting).
The implementation captures generation IDs but does not automatically query
the separate [generation record endpoint](https://openrouter.ai/docs/api/api-reference/generations/get-generation).

For adaptive profile-writer execution, validate the checked-in reproducible
route example:

```bash
PYTHONPATH=src python -m cape_loop config validate \
  configs/openrouter_gemini.toml
```

Change the model slug to switch models:

```toml
# Change this canonical author/model slug.
model = "google/gemini-3.6-flash"
```

That example pins `openrouter_upstream_provider = "google-ai-studio"`. Keep the
pin when the replacement model is served by that endpoint. For a different
model family, set the correct OpenRouter provider slug or clear the value to
allow router selection; the accepted upstream route is retained in every
audit. The standalone `plan-openrouter` and `execute-openrouter` commands
default to no upstream pin, so `--model author/model` is their only required
model-switching argument.

Then authorize the adaptive run:

```bash
PYTHONPATH=src python -m cape_loop run \
  configs/openrouter_gemini.toml \
  --execute-live
```

The external recovery journal defaults to
`<output-root>/.llm-journals/<run-id>/openrouter/<model-role>/`. A successful
run copies used rows into `llm/provider-audit.jsonl` and writes
`llm/provider-manifest.json`; it does not imply that this repository contains a
live result.

OpenRouter can also collect one or more reviewed-generic decoder sources:

```bash
PYTHONPATH=src python -m cape_loop decoder-study plan-openrouter \
  decoder-requests.jsonl \
  --model anthropic/claude-sonnet-4.5 \
  --additional-model google/gemini-3.6-flash

PYTHONPATH=src python -m cape_loop decoder-study execute-openrouter \
  decoder-requests.jsonl \
  openrouter-decoder-output \
  --model anthropic/claude-sonnet-4.5 \
  --additional-model google/gemini-3.6-flash \
  --execute-live
```

Each exact model receives a separate journal under
`OUTPUT_DIR/journals/<model-digest>/`; the output also contains
`judgments.jsonl` and `execution-manifest.json`. All sources still share the
OpenRouter gateway, and a reported upstream provider is not a direct
first-party origin record. The manifest therefore fixes
`first_party_origin_claimed = false`, `strict_gate4_eligible = false`, and
`statistical_independence_claimed = false`. Strict Gate 4 remains limited to
the direct first-party Anthropic/Gemini decoder collection and direct OpenAI
native-action collection described below.

## Plan and execute OpenAI decoder variants

Budget both default decoder sources without a key:

```bash
PYTHONPATH=src python -m cape_loop decoder-study plan-openai \
  decoder-requests.jsonl \
  --roles replication decoder \
  --max-requests 500 \
  --max-total-tokens 2050000
```

Only after reviewing the per-role ceilings:

```bash
PYTHONPATH=src python -m cape_loop decoder-study execute-openai \
  decoder-requests.jsonl decoder-output \
  --roles replication decoder \
  --max-requests 500 \
  --max-total-tokens 2050000 \
  --execute-live
```

Fresh execution is rejected before the first network call if either role's
corpus exceeds its own ceiling. Each role has a separate resumable journal.
The default pair—GPT-5.6 Terra and GPT-5.6 Luna—provides two operational model
variants, but shared provider infrastructure and possible lineage mean it does
not prove statistically independent judgment. The execution manifest records
`statistical_independence_claimed = false`.

See [LLM exchange](llm-exchange.md) for exact provider fields, request hashes,
recovery semantics, and scientific interpretation.

## Collect the selected Gate 4 model evidence

The checked-in Gate 4 model suite uses:

- OpenAI `gpt-5.6-sol` at medium reasoning as the declared native action
  system;
- Anthropic `claude-sonnet-5` as one blinded external decoder; and
- Google `gemini-3.6-flash` as the other blinded external decoder.

The frozen declaration, official source links, origins, credential names, and
per-source ceilings are in
[`data/model-suites/gate4-native-and-distinct-decoders.json`](../data/model-suites/gate4-native-and-distinct-decoders.json).
First produce and verify a completed Experiment B run. Then create both plans;
neither command reads a credential:

```bash
PYTHONPATH=src python -m cape_loop decoder-study plan-distinct \
  runs/<experiment-b-run>/decoder/external-requests.jsonl \
  --output artifacts/gate4-decoder-plan.json
PYTHONPATH=src python -m cape_loop native-action plan-openai \
  runs/<experiment-b-run> \
  --output artifacts/gate4-native-action-plan.json
```

Review the exact request counts, body hashes, official origins, selected models,
and default ceilings of 900 physical transport attempts and 6,000,000
conservatively charged tokens per source. Retries consume those ceilings. When
you are ready to fund collection, export the three keys in your own shell; do
not add values to `.env.example` or commit a populated `.env`:

```bash
export OPENAI_API_KEY='...'
export ANTHROPIC_API_KEY='...'
export GEMINI_API_KEY='...'
```

Authorize the two resumable collections explicitly:

```bash
PYTHONPATH=src python -m cape_loop decoder-study execute-distinct \
  runs/<experiment-b-run>/decoder/external-requests.jsonl \
  artifacts/gate4-decoder-collection \
  --execute-live
PYTHONPATH=src python -m cape_loop native-action execute-openai \
  runs/<experiment-b-run> \
  artifacts/gate4-native-actions \
  --execute-live
```

Both outputs remain outside the immutable run. The external collection produces
`judgments.jsonl`, `provider-audit.jsonl`, `transport-attempts.jsonl`, its
keyless plan, and an execution manifest. The native collection produces its
bound collection plan, exact requests, physical-attempt journal, audit records,
`native-actions.jsonl`, and execution manifest. A `started` attempt is durable
before each HTTP request, and accepted audit rows are written before reusable
judgment/action rows so an interrupted append can be repaired without another
accepted call. An unresolved attempt stops automatic resume for manual review.

Separate first-party providers and model families satisfy the repository's
mechanical source-diversity check; they do not prove independent errors. A
responsible researcher must inspect the retained source descriptors and complete
a `decoder-source-review` record. Only then import all exact evidence files:

```bash
PYTHONPATH=src python -m cape_loop gate-review import-native \
  runs/<experiment-b-run> \
  runs/<experiment-b-run>/decoder/external-requests.jsonl \
  artifacts/gate4-decoder-collection/judgments.jsonl \
  runs/<experiment-b-run>/decoder/truth-labels.researcher-only.jsonl \
  artifacts/gate4-native-actions \
  artifacts/gate4-source-review.json \
  artifacts/gate4-review \
  --external-collection-dir artifacts/gate4-decoder-collection
PYTHONPATH=src python -m cape_loop gate-review verify \
  artifacts/gate4-review
```

Keep the request, truth-label, and source-review inputs plus the complete
five-file decoder and six-file native collections beside or within the release
evidence bundle. The positional judgment path must name the decoder
collection's byte-identical `judgments.jsonl`. The review stores every evidence
file's hash and filename but does not copy them. Standalone automated judgment
or native-action files are not eligible for official-provider provenance. For
reviewed human or other generic sources, use the explicit
`--allow-reviewed-generic-decoders` alternative instead of
`--external-collection-dir`; that mode records no provider-collection claim. A
valid review still records
`claim_status = "not_claimed"` until the paper's full scientific decision
process is completed. See
[Gate 4 live collection](gate4-live-collection.md).

## Choose another output parent

`--output-root` changes only where the run directory is created:

```bash
PYTHONPATH=src python -m cape_loop run configs/smoke.toml \
  --output-root /tmp/cape-loop-runs
```

The override is not folded into the resolved configuration digest. The original
TOML is copied into the run as `config.source.toml`, and the resolved form is
stored as `config.resolved.json`.

## Export public JSON Schemas

Export to the checked-in default directory:

```bash
PYTHONPATH=src python -m cape_loop schema export
```

Or use another destination:

```bash
PYTHONPATH=src python -m cape_loop schema export /tmp/cape-loop-schemas
```

The command writes the currently defined interchange schemas and prints every
written path.

## Validate offline LLM replay records

```bash
PYTHONPATH=src python -m cape_loop llm validate responses.jsonl
```

This validates the response file's syntax and record shape only. It does not
call a model, export requests, prove request coverage, or authenticate the
reported model identifier. Actual replay binds each consumed response to the
locally reconstructed request through both `request_id` and `prompt_sha256`.
See [LLM exchange](llm-exchange.md).

## Generate a human-study material packet

```bash
PYTHONPATH=src python -m cape_loop human-study generate /tmp/cape-loop-study \
  --assignment-id pilot-template \
  --seed 1729
```

The command creates:

```text
README.md
human-rating.schema.json
order-manifest.json
packet-manifest.json
participant-items.jsonl
researcher-codebook.json
```

`OUTPUT_DIR` must be absent or empty. Generation refuses a nonempty directory
instead of overwriting or mixing packet files.

This is deterministic material generation, not survey deployment. It does not
recruit participants, collect ratings, analyze responses, or provide ethics
approval.

## Interpretation checklist

Before treating any run as evidence:

- inspect `metrics/summary.json` and `metrics/gate-report.json`;
- confirm `scientific_claim_status` or `claim_status` remains `not_claimed`;
- compare the split, calibration, terminal-battery, and source identities;
- distinguish controlled anchors from naturally sampled responses;
- distinguish fixed histories from endogenous trajectories;
- retain raw and active fitted-model artifacts separately; and
- require a complete, verified held-out paraphrase result before treating Gate
  1 as passed; the presence of the execution pipeline alone is not verification.

See [Outputs](outputs.md) for exact filenames and
[Implementation status](implementation-status.md) for current boundaries.
