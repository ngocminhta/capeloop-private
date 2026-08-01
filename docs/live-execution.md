# Live execution and external evidence

This is the canonical operator guide for replaying retained model outputs,
calling supported providers, and admitting external evidence into CAPE-Loop.
It separates four execution modes that must not be conflated:

| Mode | Network or key | What it produces |
| --- | --- | --- |
| Conversation authoring | Yes, with explicit authorization | Candidate neutral bases/display names and a readable generation log |
| Offline simulation | No | Synthetic users, mathematical choices, frozen rendered conversations, profiles, metrics, and request packets |
| Replay | No | The same experiment path driven by retained, hash-bound model responses |
| Live collection | Yes, with explicit authorization | Provider responses, audits, and transport journals that can later be replayed |

The mathematical simulator does not ask a model to invent users or choose
responses. A separate OpenRouter command may author a candidate conversation
bank, after which experiment runtime renders that frozen bank offline. The
evaluated profile writer is another role, called only by a live provider
command or by a run containing an `llm_*` updater in `openai` or `openrouter`
mode. Every live path requires `--execute-live`.

No credential, eligible paper-scale provider corpus, Gate 4 judgment corpus,
native-action corpus, or human response dataset is checked into the repository.
The authoritative software/evidence boundary and dated local validation
snapshot are in [Implementation status](implementation-status.md).

## Current live boundary

Provider support means the repository can construct, budget, send, audit, and
replay an explicitly authorized request. It does not mean every selected model
and endpoint combination has completed a current-source live check. In
particular, the selected Gate 4 pair is budget-validated and provider-capable
but does not have a complete eligible decoder collection. Consult the dated
[Implementation status](implementation-status.md#local-live-validation-snapshot)
before starting a collection wave.

That operational state does not change the scientific boundary:

- software tests and keyless plans establish executable protocol behavior;
- a transport smoke establishes only that a narrow request path was exercised;
- an accepted corpus must still pass complete request, audit, provenance, and
  source-review validation; and
- every generated review retains `claim_status = "not_claimed"`.

## Credentials and authorization

The CLI does not automatically load `.env`. If a trusted, ignored `.env` is
used, load it into the invoking shell without printing it:

```bash
set -a
source .env
set +a
```

Never commit `.env`, paste a key into TOML or JSON, pass a key as a command
argument, or copy an authorization header into an artifact.

| Workflow | Environment variable |
| --- | --- |
| Direct OpenAI writers and native actions | `OPENAI_API_KEY` |
| OpenRouter conversation authoring, writers, and selected Claude/Gemini decoders | `OPENROUTER_API_KEY` |
| Optional direct Anthropic decoder replication | `ANTHROPIC_API_KEY` |
| Optional direct Gemini decoder replication | `GEMINI_API_KEY` |

A configured key alone cannot initiate a request. Live execution also requires
`--execute-live`, a valid provider origin, a complete budget preflight, and a
request that is not already safely represented in the recovery journal.

Official origins are locked by default:

```text
OpenAI       https://api.openai.com
OpenRouter   https://openrouter.ai/api
Anthropic    https://api.anthropic.com
Gemini       https://generativelanguage.googleapis.com
```

A custom HTTPS origin requires the corresponding explicit custom-origin flag
and a dedicated, non-reserved credential-variable name. That opt-in authorizes
sending the named credential to the reviewed endpoint; it is a security
decision, not a routine compatibility switch.

## Declared models

Inspect the checked-in OpenAI role declaration without a key:

```bash
PYTHONPATH=src python -m cape_loop llm models
```

| Role | Model | Effort | Use |
| --- | --- | --- | --- |
| Neutral-base/display-name author | OpenRouter `anthropic/claude-sonnet-5` | Command-declared | One call per scenario; no treatment or choice authoring |
| `primary` | `gpt-5.6-sol` | `medium` | Primary profile-writer evaluation |
| `replication` | `gpt-5.6-terra` | `medium` | GPT-5.6 model-variant/tier replication |
| `decoder` | `gpt-5.6-luna` | `low` | Generic decoder and pilot workloads |

Sol versus Terra is not distinct-family robustness. Within one causal
comparison, keep model and effort fixed across `llm_response_only`,
`llm_full_context`, and `llm_provenance_aware`.

OpenRouter is a first-class gateway provider. Its static and adaptive commands
accept one exact canonical `author/model` slug, making it easy to switch
models. Aliases beginning with `~`, `openrouter/auto`, colon-suffixed route
variants, and `-latest` labels are rejected. When an adaptive configuration
pins an upstream endpoint, changing to a model not served there also requires
changing or clearing `openrouter_upstream_provider`. Preserve the reviewed
presets: copy one into ignored `configs/local/`, change the model/route there,
then validate the resolved local file before execution.

The canonical Experiment B panel is frozen in
`data/model-suites/experiment-b-bounded-calibration-v1.json`:

| Analysis set | Model | Effort | Conditions | Maximum attempts |
| --- | --- | --- | --- | ---: |
| Primary | `google/gemini-3.6-flash` | `minimal` | full B design | 636 |
| Primary | `openai/gpt-5.6-luna` | `low` | full B design | 636 |
| Primary | `mistralai/mistral-large-2512` | omitted | full B design | 636 |
| Targeted secondary | `deepseek/deepseek-v4-flash` | omitted | incorrect seed; balanced versus soft only | 252 |

The primary trio was frozen before the bounded multi-user calibration. DeepSeek
is explicitly post-pilot and secondary because the quick pilot motivated its
targeted replication. Analyze every model separately; never pool the DeepSeek
arm into the primary panel, and do not treat model names as independent user
clusters. Claude Sonnet 5 is not in the primary trio because its completed
quick-pilot route exposed a nonempty moderation pipeline that the current
integrity contract rejects.

Plan the complete suite without a credential or model call:

```bash
PYTHONPATH=src python -m cape_loop experiment-b model-suite \
  configs/live/experiment_b_openrouter.toml \
  --output-root runs/experiment-b-suite
```

The JSON printed to standard output resolves every arm's model, route,
conditions, isolated output directory, request count, and token ceiling. This
is planning only: it reports `live_execution = false` and does not read
`OPENROUTER_API_KEY`. To authorize all four paid runs, one after another, add
the flag explicitly:

```bash
PYTHONPATH=src python -m cape_loop experiment-b model-suite \
  configs/live/experiment_b_openrouter.toml \
  --output-root runs/experiment-b-suite \
  --execute-live
```

`--execute-live` on this suite command authorizes four sequential runs, not one
smoke request. Each arm has its own provider ledger and output subtree.

### Conversation-template authoring

Generate a candidate bank only when creating or revising scenario language:

```bash
cape-loop conversations generate-openrouter \
  data/scenarios/scenario-catalog-v1.json \
  data/scenarios/conversation-templates-v1.json \
  --model anthropic/claude-sonnet-5 --execute-live
```

This produces:

```text
data/scenarios/conversation-templates-v1.json
data/scenarios/conversation-templates-v1.generation.jsonl
```

Each OpenRouter call returns four neutral `display_names` and one neutral
`base_template`. It does not author separate treatments or a user decision.
Code expands the result into the bank's five stored
`presentation_templates`: balanced/restricted/ranking share the base; default
and suggested add only a fixed treatment sentence. Code sets
`choice_template` exactly to `I choose {selected_name}.`

The current command makes one bounded authoring call per catalog scenario. The
JSONL sibling is a readable generation record. Neither is a profile-writer
response corpus. Human surface and scientific review remain pending for the
current bank, so it is suitable only for simulation and bounded pilots and
remains `paper_eligible = false`.

Experiment configs select the reviewed/frozen candidate through
`[scenarios] conversation_file`. A consuming run copies it to
`inputs/conversation-templates.json`, writes
`inputs/conversation-templates-manifest.json`, and renders it without a
provider call.
Do not regenerate language once per user, condition, or trial: that would add
an uncontrolled model-dependent treatment.

### Scenario review and promotion

Before treating the catalog as paper evidence, create a whole-catalog kit with
no provider calls:

```bash
PYTHONPATH=src python -m cape_loop scenarios audit \
  configs/live/experiment_b_openrouter.toml \
  artifacts/scenario-review-kit --split all --turns 16
```

Keep `review-item-map.json` with the researchers. Give blinded reviewers
copies of the corresponding JSON templates; those copies already contain the
opaque item IDs and visible material. Collect exactly two completed surface
reviews, two completed scientific reviews, one neutral-choice pretest, and one
masked-attractiveness pretest in an otherwise empty evidence directory. The
reviewers must be distinct as declared by the protocol, and the two scientific
reviewer IDs must also approve the actual masked materials.

Then verify and, only if every rule passes, derive new reviewed inputs:

```bash
PYTHONPATH=src python -m cape_loop scenarios review-promote \
  configs/live/experiment_b_openrouter.toml \
  artifacts/scenario-review-kit \
  artifacts/scenario-review-responses \
  artifacts/scenario-reviewed-release \
  --catalog-version 1.6.0 --frozen-on YYYY-MM-DD
```

This command makes no model call and never edits `data/scenarios/`. It always
retains the imported evidence and aggregate report in a new output directory.
If a review or pretest misses a threshold, no reviewed catalog is written;
revise and version the source prospectively rather than replacing a failed
stimulus after seeing experiment outcomes. On a full pass, the companion bank
also receives a new bank ID and protocol-bound reviewed source; an unchanged
`unreviewed` development-bank identity is never presented as reviewed.

### Selected Gate 4 systems

The repository-selected Gate 4 stack is:

| Role | Model and transport | Effort | Key |
| --- | --- | --- | --- |
| Native end-to-end actions | Direct OpenAI `gpt-5.6-sol` | `medium` | `OPENAI_API_KEY` |
| Blinded decoder family 1 | OpenRouter `anthropic/claude-sonnet-5` | `low` | `OPENROUTER_API_KEY` |
| Blinded decoder family 2 | OpenRouter `google/gemini-3.6-flash` | `minimal` | `OPENROUTER_API_KEY` |

Claude and Gemini are distinct declared model families, but both requests pass
through one OpenRouter gateway. Selected artifacts therefore retain:

```text
provenance_mode = "selected_openrouter_gateway_collection"
shared_gateway = true
distinct_transport_origins = false
first_party_origin_claimed = false
statistical_independence_claimed = false
responsible_researcher_source_review_required = true
```

An upstream display label reported by OpenRouter is not proof that CAPE-Loop
called a first-party endpoint directly. A responsible researcher must review
model lineage, shared infrastructure, likely training overlap, prompt/schema
effects, and the intended claim scope.

Direct Anthropic `claude-sonnet-5` and direct Gemini `gemini-3.6-flash`
collectors remain implemented as optional origin replications. They are not
prerequisites for the selected OpenRouter workflow, are retained in separate
directories, and do not by themselves prove statistically independent errors.

## Experiment B manipulation audit

Before any paid Experiment B collection, build the prospective matched schedule
and stress it across local simulator response seeds:

```bash
PYTHONPATH=src python -m cape_loop experiment-b manipulation-audit \
  configs/live/experiment_b_openrouter.toml \
  artifacts/experiment-b-manipulation-audit
```

This command cannot call an LLM: it has no `--execute-live` option, reads no API
credential, and reports `llm_calls = 0`. The optional `--response-seeds N`
changes only the number of local simulator draws; the config default is 32.
Use a smaller override for a smoke check and reserve 64 for a one-time final
offline audit when runtime permits. It writes:

```text
experiment-b-manipulation-plan.json
experiment-b-manipulation-plan.md
experiment-b-offline-manipulation-audit.json
```

The plan is outcome-blind. It predeclares at least two informative active turns,
one decisive active control, both default and suggestion mechanisms, retained
counter-profile options, and a minimum trajectory active susceptibility mass.
Correct and incorrect initial-profile branches share the same frozen
scenario-role-mechanism schedule and exogenous randomization key. Required
active turns follow the current profile direction; an exactly neutral current
profile uses the frozen initial-profile direction and logs that fallback. The
audit then describes expected information, predicted and simulated choice
susceptibility, visible-action divergence, realized exact-shadow SelectionCost,
fallback use, condition/domain and role-specific summaries, and
attribute/direction coverage. Its simulated outcomes never revise or admit the
already frozen plan. For transparency, the audit also emits a descriptive
active-turn cross-tab over role, mechanism, runtime effective direction, frozen
planned direction, target attribute, and domain. Its counts must reconcile with
the pooled active-instruction and role totals, but the cells are not additional
post-outcome admission gates.

Passing establishes only that the treatment reaches suitable decisions under
the declared simulator. It does not establish an LLM effect, behavioral
reinforcement, natural language quality, or paper eligibility. With
`manipulation.planning_mode = "required"`, every ordinary live B run repeats the
planning and offline audit inside its run directory and refuses the evaluated
model calls if the prospective schedule is not ready.

## Budget preflight

Planning and validation read no credential. The universal adaptive preflight
counts:

- development calibration requests;
- every experiment updater request;
- Experiment A held-out paraphrases;
- every trajectory turn and sensitivity point; and
- the `max_retries + 1` physical-attempt expansion.

It rejects a design before provider construction, credential access, journal
creation, or run-directory creation when either the physical-attempt ceiling
or maximum output allocation is too small. Adaptive input length depends on
earlier model outputs, so the provider ledger conservatively enforces the
cumulative token ceiling before every attempt.

The approved pilot ceiling is 900 physical HTTP attempts and 6,000,000
conservatively accounted tokens per provider ledger. Bounded paper pilots use
zero retries; do not raise these ceilings merely to make a design pass.

| Adaptive pilot | Physical-attempt bound | Maximum output allocation |
| --- | ---: | ---: |
| Experiment A | 580 | 1,187,840 |
| Experiment B | 636 | 1,302,528 |
| Experiment C | 828 | 1,695,744 |
| Gate 6 OAT | 720 | 1,474,560 |

The A bound is 480 controlled same-response updates, 60 development
calibration updates, and 40 controlled held-out-paraphrase updates. The B and C
bounds each include the same five-mechanism 60-request calibration probe.

The listed B presets use that two-domain, eight-user,
`llm_full_context`-only six-turn design. They cross correct/incorrect seeds with
balanced, soft, and exploratory policies, revisit each preference dimension,
retain local reference updaters, and fit at 636 calls including calibration.
Run one selected model per completed source run; the preset is a bounded
calibration design, not a preregistered paper sample.

The bounded offline Gate 4 source produces 640 decoder requests per model and
80 native-action requests. Its selected zero-retry plan reserves 4,469,348
Claude tokens, 4,201,828 Gemini tokens, and 2,396,907 OpenAI native-action
tokens. The Experiment C external-rescore source produces 360 decoder requests
per model and reserves 2,709,130 Claude tokens and 2,558,650 Gemini tokens.
These are conservative keyless reservations, not usage or cost.

Validate an adaptive configuration without loading a key:

```bash
PYTHONPATH=src python -m cape_loop config validate CONFIG.toml
```

Static and decoder planners similarly construct the exact request bodies,
hashes, retry-expanded attempt counts, and conservative token reservations
without reading a credential.

## Request, response, and replay contract

An `LLMRequest` hashes:

```text
system_instruction + "\n" + canonical_JSON(payload)
```

Its request ID is content-addressed by updater ID and prompt SHA-256. A
provider-neutral response repeats the request ID and prompt hash and contains
three complete four-value probability vectors. Parsing rejects unknown fields,
duplicate IDs, missing support values, non-finite probabilities, and rows that
do not sum to one within tolerance. Accepted provider decimals that are within
that public tolerance but outside the tighter executable-belief invariant are
normalized in the provider-neutral replay record; the verbatim provider
payload and its SHA-256 remain retained separately in `raw_response`.

The three model information views are:

| View | Included | Withheld |
| --- | --- | --- |
| Response only | Prior, semantic attribute codebook, local user reply, selected readable option | Assistant turn, unselected options, presentation, provenance |
| Full context | Response-only fields plus exact assistant/user dialogue and readable displayed options | Policy provenance and hidden audit labels |
| Provenance aware | Full context plus structured policy provenance | Latent preference, susceptibility, and numeric simulator encoding |

No model view receives latent truth, numeric option features, the target
attribute index, choice-randomness keys, or hidden condition/CRN identifiers.
The codebook explains each domain attribute and the meanings of `-2`, `-1`,
`+1`, and `+2` in ordinary language.

Validate a retained response corpus:

```bash
PYTHONPATH=src python -m cape_loop llm validate responses.jsonl
```

This checks record shape only. Actual replay reconstructs every request and
requires exact request-ID and prompt-hash coverage. The run fingerprints the
complete corpus in `llm/input-manifest.json`.

For A, B, and C, `llm.calibration = "temperature"` first collects a fixed
development-only probe and fits one scalar temperature per updater view. Test
labels are never used. A calibrated run retains:

```text
models/llm-calibration.json
llm/development-raw-responses.jsonl
metrics/llm-development-calibration.jsonl
llm/test-raw-responses.jsonl
llm/test-calibrated-responses.jsonl
llm/responses.jsonl
```

For Experiment A, `llm/responses.jsonl` and every primary update/error metric
use the raw vector; the calibrated test file is secondary diagnostics only.
For B/C, `llm/responses.jsonl` contains the configured active calibrated
vectors used by the realized histories.

`calibration = "none"` is an explicit raw-probability ablation. Sensitivity
LLM runs require this uncalibrated mode plus retained prompts and events.

## Transport audits, locks, and recovery

Every physical request is journaled before transport:

```text
started  -> HTTP dispatch -> settled
```

The `started` event is flushed and fsynced before dispatch. A final settlement
embeds the accepted or rejected provider audit. Accepted audits are persisted
before reusable responses, judgments, or actions, allowing a crash-interrupted
append to be repaired without another accepted call.

Request budgets count physical attempts. Failed, invalid, HTTP-error, and
ambiguous attempts consume one attempt and conservatively reserve the full
token amount unless settled provider usage supplies a valid charged count. An
unresolved start, ambiguous transport result, or settled sequence without a
final audit requires manual billing review before another call.

Static and collection commands hold nonblocking file locks over reconciliation,
dispatch, and final writes. Concurrent invocations aimed at one output fail
before credential access or duplicate dispatch. Opaque lock files are
coordination state, not scientific evidence.

Adaptive recovery journals live outside the immutable run:

```text
<output-root>/.llm-journals/<run-id>/<role>/
<output-root>/.llm-journals/<run-id>/openrouter/<role>/
```

After a failed adaptive run, inspect `failure.json`, correct the cause, and
resume the exact same configuration:

```bash
PYTHONPATH=src python -m cape_loop run CONFIG.toml \
  --execute-live --resume-failed-live
```

The command archives the failed destination under `.failed-runs/`, recreates
the deterministic run path, and reuses only safely reconciled journal entries.
It refuses a completed artifact or mismatched configuration.

## Static provider checks

Plan and execute a direct OpenAI request corpus:

```bash
PYTHONPATH=src python -m cape_loop llm plan requests.jsonl \
  --role primary --max-requests 6 --max-retries 0

PYTHONPATH=src python -m cape_loop llm execute-openai \
  requests.jsonl responses.jsonl provider-audit.jsonl \
  --role primary --max-requests 6 --max-retries 0 --execute-live
```

Plan and execute the same provider-neutral corpus through OpenRouter:

```bash
PYTHONPATH=src python -m cape_loop llm plan-openrouter \
  requests.jsonl --model google/gemini-3.6-flash \
  --max-requests 6 --max-retries 0

PYTHONPATH=src python -m cape_loop llm execute-openrouter \
  requests.jsonl responses.jsonl openrouter-audit.jsonl \
  --model google/gemini-3.6-flash \
  --max-requests 6 --max-retries 0 --execute-live
```

The OpenRouter adapter sends provider-compatible strict JSON Schema, disables
gateway caching, requests router metadata, and defaults to no fallbacks,
required parameter support, and denied provider data collection. Anthropic
requests omit numeric `minimum` and `maximum` schema keywords because the
observed Amazon Bedrock route rejects them; the schema description retains the
inclusive `[0,1]` contract, and local parsing still rejects non-finite,
out-of-range, or non-normalized belief vectors. The audit retains
requested/returned model, selected upstream display identity, route
strategy/attempt, cache status, provider/generation IDs, usage, timings,
hashes, redacted raw response, and the provider-neutral replay response. Model
substitution, disallowed fallback, cache hits, or material router
transformations are rejected.

## Adaptive A/B/C and Gate 6 pilots

Use the runner for adaptive profile writing:

```bash
PYTHONPATH=src python -m cape_loop run \
  configs/live/experiment_a_openai.toml --execute-live

PYTHONPATH=src python -m cape_loop run \
  configs/live/experiment_a_openrouter.toml --execute-live
```

The matched bounded configs are:

| Purpose | Direct OpenAI | OpenRouter |
| --- | --- | --- |
| Experiment A | `configs/live/experiment_a_openai.toml` | `configs/live/experiment_a_openrouter.toml` |
| Experiment B | `configs/live/experiment_b_openai.toml` | `configs/live/experiment_b_openrouter.toml` |
| Experiment C | `configs/live/experiment_c_openai.toml` | `configs/live/experiment_c_openrouter.toml` |
| Gate 6 OAT | `configs/live/sensitivity_openai.toml` | `configs/live/sensitivity_openrouter.toml` |

These are ceiling-safe pilots, not preregistered sample sizes or completed
paper evidence. Verify every successful run:

```bash
PYTHONPATH=src python -m cape_loop verify runs/<printed-run-id>
```

For Experiment B, `experiment.bootstrap_replicates = 0` is point-estimate-only:
clustered intervals and one-sided directional decisions are marked
`not_computed`, so Gates 2 and 3 cannot pass. A positive value enables both the
bootstrap intervals and paired complete-user sign-flip decisions. It sets the
bootstrap resample count, not the sign-pattern count; sign patterns are derived
separately from the number of complete user clusters.

The B presets freeze `selection_noninferiority_margin = 0.02` and
`net_harm_margin = 0.02` on the marginal-Brier terminal-error scale. Gate 3 is a
policy-conditioned legibility gate evaluated in the incorrect-initial-profile
stratum: positive soft same-history attribution gap, positive
soft-minus-balanced attribution-gap contrast, and a one-sided SelectionCost
noninferiority decision below `0.02`. The nested net-profile-harm gate uses the
same incorrect-seed contrast and additionally requires a one-sided
soft-minus-balanced updater terminal-error decision above `0.02`. Neither gate
is decided from a point-estimate sign alone. `decomposition_tolerance = 1e-12`
only verifies the algebraic error decomposition and is not an effect-size
threshold.

The public A presets run the primary `controlled_anchor` track over balanced,
restricted, ranking, default, and suggested contexts. They hold the user
response fixed, require the same-response audit to pass, and use the exact
action-aware oracle as the primary reference. The oracle uses a uniform prior
over the susceptibility support prospectively assigned to the test split and
does not receive an individual's latent susceptibility. Read
`metrics/experiment-a-exact-calibration.json`,
`metrics/experiment-a-same-response-audit.json`, and
`models/exact-action-aware-reference.json` before the fitted-reference
robustness artifacts. The earlier directional H1/H2 outputs are diagnostic,
not the primary decision rule.

For live sensitivity, `profile_conditioning_strength = 0` is a no-treatment
negative control. Every positive dose must produce both nonzero treatment
exposure and nonzero visible-action divergence; otherwise the point is marked
as a failed manipulation. Strict self-confirmation remains a secondary
endpoint.

The OpenAI primary/replication pair can be planned together without a key:

```bash
PYTHONPATH=src python -m cape_loop llm evaluation-suite \
  configs/live/experiment_a_openai.toml \
  configs/live/experiment_a_openai_replication.toml \
  --output-root runs
```

Add `--execute-live` only after reviewing its two isolated ledgers and output
paths.

## Experiment A controls and H7

Every Experiment A run emits a six-case provider-neutral control packet.
Collect the request file with either static provider command, then score the
complete corpus outside the immutable source run:

```bash
PYTHONPATH=src python -m cape_loop control-study analyze \
  RUN/llm/experiment-a-control-request-bindings.jsonl \
  control-responses.jsonl control-analysis.json \
  --source-descriptor "reviewed provider collection"
```

H7's volunteered direct-statement control is also external to the source run:

```bash
PYTHONPATH=src python -m cape_loop control-study h7-plan \
  RUN artifacts/h7-plan

# Collect artifacts/h7-plan/h7-volunteered-requests.jsonl with
# llm execute-openai or llm execute-openrouter.

PYTHONPATH=src python -m cape_loop control-study h7-review \
  RUN artifacts/h7-plan responses.jsonl provider-audit.jsonl h7-review.json

PYTHONPATH=src python -m cape_loop control-study h7-verify \
  RUN artifacts/h7-plan responses.jsonl provider-audit.jsonl h7-review.json
```

The review requires complete test-user/domain/attribute coverage in both
full-context and provenance-aware arms and exact accepted provider-audit
bindings. It never imputes missing outcomes or mutates the source run.

## Gate 4 collection and admission

First generate the bounded source packet offline:

```bash
PYTHONPATH=src python -m cape_loop run \
  configs/offline/gate4_source.toml
PYTHONPATH=src python -m cape_loop verify runs/<printed-source-run>
```

The source packet contains:

```text
decoder/external-requests.jsonl
decoder/truth-labels.researcher-only.jsonl
decoder/researcher-codebook.jsonl
```

Send only `external-requests.jsonl` to a decoder. Truth labels and the
researcher codebook are evaluator-only.

Plan the selected pair without a key:

```bash
PYTHONPATH=src python -m cape_loop decoder-study plan-openrouter \
  runs/<source-run>/decoder/external-requests.jsonl
```

After reviewing the current status, exact bodies, and ceilings, the live
command is:

```bash
PYTHONPATH=src python -m cape_loop decoder-study execute-openrouter \
  runs/<source-run>/decoder/external-requests.jsonl \
  artifacts/gate4-openrouter-decoders --execute-live
```

It writes separate per-model journals plus:

```text
collection-plan.json
transport-attempts.jsonl
provider-audit.jsonl
judgments.jsonl
execution-manifest.json
```

Gate 4 also needs genuine actions emitted from retained native state. The
declared `cape-loop-openai-native-agent-v1` sends the complete retained native
state and exact held-out suite to OpenAI Sol; it does not project a local
belief into an action:

```bash
PYTHONPATH=src python -m cape_loop native-action plan-openai \
  runs/<source-run>

PYTHONPATH=src python -m cape_loop native-action execute-openai \
  runs/<source-run> artifacts/gate4-native-actions --execute-live
```

The native collection retains its request plan, attempt journal, provider
audits, `native-actions.jsonl`, and execution manifest.

Before import, a named responsible researcher must create a hash-bound
`decoder-source-review` covering each source and source pair. Then import all
evidence into a new directory:

```bash
PYTHONPATH=src python -m cape_loop gate-review import-native \
  runs/<source-run> \
  runs/<source-run>/decoder/external-requests.jsonl \
  artifacts/gate4-openrouter-decoders/judgments.jsonl \
  runs/<source-run>/decoder/truth-labels.researcher-only.jsonl \
  artifacts/gate4-native-actions \
  decoder-source-review.json \
  artifacts/GATE4-REVIEW \
  --openrouter-collection-dir artifacts/gate4-openrouter-decoders

PYTHONPATH=src python -m cape_loop gate-review verify \
  artifacts/GATE4-REVIEW
```

The importer rebuilds the selected request plans, verifies every source-run and
collection binding, calibrates and scores decoder evidence on the declared
splits, checks action coverage, and publishes a staged checksum-bound review
without modifying the source. `--external-collection-dir` is reserved for the
optional validated direct-provider collection.
`--allow-reviewed-generic-decoders` explicitly admits reviewed human or other
generic sources without claiming automated-provider provenance.

Plan and execute the optional direct-origin replication separately:

```bash
PYTHONPATH=src python -m cape_loop decoder-study plan-distinct \
  runs/<source-run>/decoder/external-requests.jsonl

PYTHONPATH=src python -m cape_loop decoder-study execute-distinct \
  runs/<source-run>/decoder/external-requests.jsonl \
  artifacts/gate4-direct-origin-replication --execute-live
```

That collection uses the direct Anthropic and Gemini credentials, retains its
own journals and manifest, and must never be silently merged with or relabeled
as the selected OpenRouter corpus.

## Experiment C external rescore

Create and verify the bounded source offline:

```bash
PYTHONPATH=src python -m cape_loop run \
  configs/offline/experiment_c_rescore_source.toml
```

Collect its blinded decoder packet with the selected OpenRouter workflow, then
build a separate immutable rescore:

```bash
PYTHONPATH=src python -m cape_loop experiment-c-decoder import \
  runs/<experiment-c-source> \
  artifacts/experiment-c-openrouter-decoders/judgments.jsonl \
  artifacts/EXPERIMENT-C-RESCORE \
  --openrouter-collection-dir artifacts/experiment-c-openrouter-decoders

PYTHONPATH=src python -m cape_loop experiment-c-decoder verify \
  artifacts/EXPERIMENT-C-RESCORE \
  --source-run runs/<experiment-c-source>
```

Development labels fit one temperature per family. Held-out judgments change
only declared native score fields, after which the complete all-system
ranking, ESR, and Gate 5 analysis is recomputed. The source run is immutable.

## Gate 6 cross-run review

Gate 6 combines explicit live-model sensitivity/Experiment A source pairs. A
researcher-authored declaration must bind each verified run, exact
provider/model evidence, and the declared family/source taxonomy:

```bash
PYTHONPATH=src python -m cape_loop gate6-review build \
  gate6-declaration.json artifacts/GATE6-REVIEW

PYTHONPATH=src python -m cape_loop gate6-review verify \
  artifacts/GATE6-REVIEW --reverify-sources
```

The reviewer recomputes the six tri-state clauses and the outcome-neutral
held-out paraphrase coverage/invariance check. It does not infer family
identity, statistical independence, or a paper claim from model labels.

## Human and confirmatory evidence

The repository can generate blinded human-study materials and validate
de-identified imports, but it does not supply ethics approval, consent,
recruitment, hosting, compensation, privacy policy, or participant responses.
See [Ethics and limitations](ethics-and-limitations.md) before any collection.

The optional R/lme4 harness consumes verified source runs and writes a separate
checksum-bound analysis. Its complete operator contract is
[`analysis/confirmatory-mixed-effects/README.md`](../analysis/confirmatory-mixed-effects/README.md).
Executable formulas do not constitute a fitted result.

## Evidence admission and release

Before treating an external corpus as evidence, retain:

- exact requests, prompt/body hashes, model and reasoning declarations;
- every accepted, rejected, missing, and failed attempt;
- durable transport journals and provider audits;
- raw and active/calibrated responses;
- development-only calibration and test separation;
- complete source-run and collection checksums;
- declared provider/gateway/origin limitations;
- responsible source review where required; and
- `claim_status = "not_claimed"` until the authors complete the scientific
  decision process.

Provider metadata demonstrates only what the adapter observed and validated.
It is not a cryptographic provider signature, an independence guarantee, or a
scientific result. Freeze only verified runs or reviews, document redistribution
terms and privacy constraints, and map every released number to its exact
source artifact.
