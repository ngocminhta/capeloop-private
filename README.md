# CAPE-Loop

**Causal Attribution of Preference Evidence in Closed-Loop Agents**

CAPE-Loop is the reference implementation and evaluation harness for the paper
proposal **“You Chose What I Showed You: Causal Provenance and Self-Confirming
User Profiles in LLM Agents.”**

The project asks a narrow causal question: when a persistent agent updates a user
profile from a choice or acceptance, does it account for the options, ranking,
default, suggestion, and policy that helped produce that response?

```text
current profile
      │
      ▼
interaction policy ──► visible choice context ──► user response
      ▲                                               │
      └──────────── persistent profile update ◄───────┘
```

A response can be compatible with a user's preference without being independent
evidence for it. Choosing a budget hotel from a balanced budget-versus-premium
set is more diagnostic than accepting the same hotel when it was preselected,
recommended, or shown only beside other budget options. CAPE-Loop preserves that
elicitation history and measures whether profile writers use it appropriately.

## Repository status

This repository contains research infrastructure, not completed empirical
claims. It implements controlled and closed-loop evaluations, produces
machine-readable metrics and gate reports, and supports hash-bound offline
replay plus explicitly authorized, budget-limited OpenAI, OpenRouter,
Anthropic, and Google API execution.
It does **not** contain fabricated paper results or imply that any scientific
stage gate has passed. Consult
[implementation status](docs/implementation-status.md) before interpreting an
available component or artifact. The exact bounded pilot matrix, credential
requirements, provider contracts, and recommended order are in
[live execution](docs/live-execution.md).

No API key or live provider response is checked in. Local direct-OpenAI and
OpenRouter profile-writer smokes validated the transport paths, but they are
not study data. One strict selected-Claude request and one strict
selected-Gemini request have also completed through OpenRouter and passed local
replay validation after the Anthropic schema was adapted to the selected
route's supported subset. These one-request diagnostics are not an eligible
decoder corpus; the full paper collection remains a separate, reviewed task.
The human study is deferred, and every diagnostic report remains
`claim_status = "not_claimed"`.

The paper design is in [the proposal](docs/proposal.md). The engineering
completion boundary is in
[implementation status](docs/implementation-status.md).

## What CAPE-Loop provides

- Fixed latent users with three preference dimensions and heterogeneous
  susceptibility to ranking, defaults, and agent suggestions.
- Travel-planning and writing-assistance domains with controlled option
  attributes.
- Explicit separation between the user-visible interaction context and the
  internal policy provenance that generated it.
- Exact action-aware Bayesian inference under the declared response model, with
  the complete preference×susceptibility joint retained.
- Fitted action-aware and four-parameter action-unaware baselines trained on
  the same records (parameter-count matched, with different outcome classes).
- Matched anchor-option audits that hold the selected item constant while
  changing its causal provenance, with executable prior-concentration strata
  and a separate content-bound six-case positive/negative control execution
  and provider-exchange path.
- Exhaustive H7 volunteered direct-statement planning plus strict
  OpenAI/OpenRouter audit binding, paired update conversion, and immutable
  source-safe recomputation; see
  [the experiment protocols](docs/experiments.md#h7-volunteered-preference-control).
- Endogenous profile–policy–response loops with a same-history action-aware
  shadow posterior.
- Fixed-history versus closed-loop evaluation using a common exogenous terminal
  diagnostic battery.
- Immutable offline Experiment C cross-seed review with exact agreement
  fractions, retained disagreements, and source-run checksum bindings.
- Structured beliefs and inspectable native episodic, semantic/persona, and
  provenance-linked memory.
- Provider-neutral JSON Lines exchange, prompt-hash-checked offline replay,
  opt-in direct OpenAI Responses API execution, and first-class OpenRouter Chat
  Completions execution with hard request/token budgets and resumable
  audit-first journals.
- One credential-free, fail-closed adaptive request preflight shared by
  Experiments A, B, C, and sensitivity. It counts calibration, held-out
  paraphrases, trajectory turns, and retry expansion before a live provider is
  constructed, and records the result in `llm/request-preflight.json`.
- Development-only temperature calibration for LLM probability vectors, fitted
  separately for each information view with raw and calibrated records retained.
  B/C additionally score cached raw/calibrated terminal vectors on the same
  realized history; multi-turn rows explicitly do not claim a recursively raw
  counterfactual trajectory.
- Deterministic human evidence-strength material packets, strict de-identified
  imports, and analysis helpers; no recruitment, hosting, or ethics approval.
- A selected Gate 4 collection stack: OpenAI `gpt-5.6-sol` acting directly
  from retained native memory, plus the exact
  `anthropic/claude-sonnet-5` and `google/gemini-3.6-flash` OpenRouter slugs as
  two blinded, distinct-model-family decoder sources, using Claude `low` and
  Gemini `minimal` reasoning.
- An optional, contract-validated R confirmatory pipeline for the proposal's
  mixed-effects models, while the dependency-free Python CR1 analysis remains
  available as a marginal robustness check.
- Versioned JSON artifacts, semantic-keyed random streams, checksums, and
  result-free gate reports.
- Executed, feature-matched train/development/test surface families with a
  retained concrete
  [leakage audit](docs/data-model.md#splits-and-leakage-controls).
- A checked-in simulator-sensitivity design over decision noise, shared and
  mechanism-specific presentation strength, profile strength, prior
  uncertainty, trajectory length, response-model family, and rule noise.

The standard-library-only core requires Python 3.11 or newer. Offline commands
never require credentials. A provider call is possible only through a live
command or a `mode = "openai"`/`mode = "openrouter"` run with the explicit
`--execute-live` flag.

## Quick start

From the repository root:

```bash
PYTHONPATH=src python -m cape_loop doctor
PYTHONPATH=src python -m unittest discover -s tests
PYTHONPATH=src python -m cape_loop --help
PYTHONPATH=src python -m cape_loop llm models
```

Validate and run a checked-in TOML configuration:

```bash
PYTHONPATH=src python -m cape_loop config validate configs/smoke.toml
PYTHONPATH=src python -m cape_loop run configs/smoke.toml
```

Use the live workflow in increasing-cost tiers:

```text
offline validate/plan
  -> six-request static transport smoke
  -> six-update adaptive smoke
  -> reviewed bounded A/B/C or Gate 6 pilot
  -> paper collection only after preregistration and evidence review
```

The repository never implicitly starts a paid run. Live commands require
`--execute-live`; the paper pilots keep the approved per-provider ceiling at
900 physical HTTP attempts and 6,000,000 conservatively accounted tokens.
Their `max_retries = 0` setting is intentional: it ensures the entire design,
not only its first attempts, fits those ceilings. Do not raise the approved
ceilings to make a design pass; reduce the design, retry count, or output
allowance.

The run command prints the output directory. Verify it before analysis:

```bash
PYTHONPATH=src python -m cape_loop verify runs/<run-id>
```

Use `make help` for repository shortcuts. The exact command and configuration
surface is documented in [Getting started](docs/getting-started.md) and
[Configuration](docs/configuration.md).

## Scientific design

CAPE-Loop holds the latent user preference fixed within a trajectory:

```text
theta[t + 1] = theta[t] = theta
```

Presentation changes response probabilities, not the user's intrinsic utility.
The implementation therefore keeps these records separate:

1. **Latent user state** — preference and presentation susceptibility, visible
   only to the simulator and evaluator.
2. **Interaction context** — displayed options, order, default, wording,
   suggestion, and question visible to the user.
3. **Policy provenance** — the policy, profile snapshot, version, and seed that
   caused the context.
4. **Observation** — selected option and semantically constrained surface
   response.
5. **Profile update** — before/after beliefs or native memory and a complete
   audit record.

Updaters receive an explicit information view:

| View | Information supplied |
| --- | --- |
| Response-only | Selected option and its attributes |
| Full-context | Response plus the complete visible elicitation context |
| Provenance-aware | Full context plus structured policy provenance |

Only evaluators may access latent truth. See
[Scientific design](docs/scientific-design.md) and
[Architecture](docs/architecture.md) for the full trust boundary.

## Evaluation tracks

**Structured-belief evaluation** asks every updater for comparable probability
distributions and measures proper scores, posterior divergence, update error,
and confidence. Fitted-model calibration parameters, raw/active bundles,
separate raw/calibrated outcomes, reliability rows, and confirmatory summaries
are retained. Exact updaters retain the full preference×susceptibility joint;
marginals remain the common public projection.

**Native-memory evaluation** lets systems retain their ordinary episodic,
semantic, persona, or provenance-linked states. Two fixed deterministic blinded
views and a common terminal behavioral battery keep the diagnostic from
silently selecting one favorable projection. Experiment C ranks a native system
by the mean of exactly those two decoder scores while retaining its public
persona projection separately. Those repository projections are diagnostics,
not independent decoders, and the transparent persona/reference choices are
not native end-to-end actions; Gate 4 remains incomplete until both evidence
types are imported. B/C terminal profile scores include ECE and reliability
bins with one preference attribute as the forecast unit.

Experiments are organized around:

- **Experiment A:** one-step causal-provenance sensitivity;
- **Experiment B:** false-profile self-confirmation and the decomposition of
  evidence selection from evidential attribution;
- **Experiment C:** fixed-history versus endogenous closed-loop system
  evaluation, plus a separate
  [two-family external-decoder rescore](docs/experiments.md#experiment-c-external-decoder-rescore)
  and
  [multi-seed robustness review](docs/experiments.md#experiment-c-multi-seed-robustness);
- **Experiment D support:** human pragmatic evidence-strength study materials.

Correction-debt, held-out paraphrase-transfer, and broader alternative-model
robustness code paths are implemented, but their existence is not evidence that
the corresponding paper stages were executed or passed.
[Experiments](docs/experiments.md) explains the cells, controls, and output
contracts.

## Dataset

CAPE-Loop does not download or repackage an existing benchmark. Its core
dataset is a procedurally generated, fully synthetic population of fixed latent
users interacting with controlled travel-planning and writing-assistance
choice environments. A configuration and semantic random keys determine latent
preferences, presentation susceptibility, option sets, policies, responses,
profile updates, and train/development/test membership. Runs retain the
generated population, causal interaction records, split manifest, exact
held-out surface families, and held-out terminal suites.

External LLM responses, external decoder judgments, and human ratings are not
part of the checked-in dataset. The repository generates content-addressed
requests and strict import contracts for them; real provider outputs must be
collected under explicit budgets, and human data collection remains deferred.
See [Data model and dataset production](docs/data-model.md) and the
[data directory policy](data/README.md).

## External LLM evaluation

Core and replay runs do not require an SDK or API key. CAPE-Loop can inspect the
declared model suite, dry-run a static request corpus without reading a
credential, execute that corpus resumably, or execute adaptive experiment
requests through the runner:

```bash
PYTHONPATH=src python -m cape_loop llm models
PYTHONPATH=src python -m cape_loop llm plan requests.jsonl --role primary
PYTHONPATH=src python -m cape_loop llm execute-openai \
  requests.jsonl responses.jsonl provider-audit.jsonl \
  --role primary --execute-live
PYTHONPATH=src python -m cape_loop run \
  configs/live/experiment_a_openai.toml --execute-live
PYTHONPATH=src python -m cape_loop llm plan-openrouter \
  requests.jsonl --model google/gemini-3.6-flash
PYTHONPATH=src python -m cape_loop llm execute-openrouter \
  requests.jsonl responses.jsonl openrouter-audit.jsonl \
  --model google/gemini-3.6-flash --execute-live
PYTHONPATH=src python -m cape_loop run \
  configs/live/experiment_a_openrouter.toml --execute-live
PYTHONPATH=src python -m cape_loop llm evaluation-suite \
  configs/live/experiment_a_openai.toml \
  configs/live/experiment_a_openai_replication.toml \
  --output-root runs
PYTHONPATH=src python -m cape_loop decoder-study plan-openai \
  decoder-requests.jsonl
PYTHONPATH=src python -m cape_loop decoder-study plan-openrouter \
  runs/EXPERIMENT-B/decoder/external-requests.jsonl
PYTHONPATH=src python -m cape_loop decoder-study execute-openrouter \
  runs/EXPERIMENT-B/decoder/external-requests.jsonl \
  artifacts/gate4-openrouter-decoders --execute-live
PYTHONPATH=src python -m cape_loop native-action plan-openai \
  runs/EXPERIMENT-B
PYTHONPATH=src python -m cape_loop gate-review import-native \
  runs/EXPERIMENT-B requests.jsonl \
  artifacts/gate4-openrouter-decoders/judgments.jsonl truth.jsonl \
  artifacts/gate4-native-actions source-review.json artifacts/GATE4-REVIEW \
  --openrouter-collection-dir artifacts/gate4-openrouter-decoders
PYTHONPATH=src python -m cape_loop gate-review verify \
  artifacts/GATE4-REVIEW
PYTHONPATH=src python -m cape_loop experiment-c-decoder import \
  runs/EXPERIMENT-C \
  artifacts/experiment-c-openrouter-decoders/judgments.jsonl \
  artifacts/EXPERIMENT-C-RESCORE \
  --openrouter-collection-dir artifacts/experiment-c-openrouter-decoders
PYTHONPATH=src python -m cape_loop experiment-c-decoder verify \
  artifacts/EXPERIMENT-C-RESCORE --source-run runs/EXPERIMENT-C
PYTHONPATH=src python -m cape_loop experiment-c-robustness review \
  artifacts/experiment-c-multiseed runs/C-SEED-1 runs/C-SEED-2
PYTHONPATH=src python -m cape_loop experiment-c-robustness verify \
  artifacts/experiment-c-multiseed
PYTHONPATH=src python -m cape_loop gate6-review build \
  gate6-declaration.json artifacts/GATE6-REVIEW
PYTHONPATH=src python -m cape_loop gate6-review verify \
  artifacts/GATE6-REVIEW --reverify-sources
PYTHONPATH=src python -m cape_loop llm validate responses.jsonl
```

The CLI does not load `.env` automatically. If credentials are stored in a
local ignored `.env`, load them into the current shell explicitly before a live
command, for example:

```bash
set -a
source .env
set +a
```

Never commit `.env`, print its values, or copy a key into TOML, JSON, a command
argument, a log, or a paper artifact.

The Gate 6 declaration and evidence rules are documented in
[Experiments](docs/experiments.md#gate-6-cross-run-review).

The checked-in roles are GPT-5.6 Sol at medium reasoning for the primary writer,
GPT-5.6 Terra at medium reasoning for a GPT-5.6 model-variant/tier replication,
and GPT-5.6 Luna at low reasoning for blinded decoder/pilot workloads. Terra is
not presented as distinct-family robustness. Within a comparison, keep the
model and reasoning effort fixed across information views. Keys are read only
from the configured environment-variable name and are never written to
artifacts. By default credentials can be sent only to the official
`https://api.openai.com` origin. A different HTTPS endpoint is rejected unless
the separate `allow_custom_base_url = true` or `--allow-custom-base-url` opt-in
is present and a dedicated key environment variable other than
`OPENAI_API_KEY` is selected; enabling it sends that credential to the reviewed
endpoint. Every live path requires `--execute-live`, enforces the declared
request and conservative token ceilings, and uses audit-first JSONL journals to
resume completed requests without rebilling them. Request ceilings count
physical HTTP attempts, and keyless plans require the retry-expanded worst case
to fit. Each call is surrounded by an fsynced transport-attempt journal; an
unknown/unresolved or settled-nonfinal prior outcome stops automatic resume for
manual billing review, while a final embedded audit repairs an interrupted
public audit/replay append without another call.
Static corpus execution also holds a nonblocking sibling file lock across
reconciliation and every append, so a concurrent local invocation fails before
credential access or duplicate dispatch.

OpenRouter is configured as a gateway, not as an OpenAI-compatible custom URL.
Set `OPENROUTER_API_KEY` in the invoking shell and pass one
`--model author/model` to switch a static routed model. For adaptive runs, copy
[`configs/live/experiment_a_openrouter.toml`](configs/live/experiment_a_openrouter.toml)
into the ignored `configs/local/` directory, then change its `model` line. If
the replacement is not served by the pinned Google Vertex global route, also
change or clear `openrouter_upstream_provider`. Aliases, route variants,
`openrouter/auto`, and `-latest` labels are rejected for reproducible
evaluation. The adapter sends provider-compatible strict JSON Schema, disables
gateway response caching, requests router metadata, retains the selected
upstream provider and model, and rejects model changes, unexpected fallback,
cache hits, or material router transformations. Anthropic/OpenRouter requests
omit numeric-bound schema keywords rejected by the observed Amazon Bedrock
route; the local parser still enforces finite `[0,1]` probabilities and
sum-to-one vectors. Its artifacts always record
`first_party_origin_claimed = false`. Even when OpenRouter reports an upstream
Anthropic, Google, or OpenAI route, that shared-gateway record is not a direct
first-party record and does not establish independent errors. The selected
Gate 4 workflow admits only the complete validated gateway collection under
`selected_openrouter_gateway_collection`, followed by responsible-researcher
source review; loose OpenRouter outputs do not qualify. See
[Live execution](docs/live-execution.md) for the exact audit and routing
contract.

Completed live runs copy the used provider evidence into
`llm/provider-audit.jsonl` and `llm/transport-attempts.jsonl`; both digests and
portable paths are retained in `llm/provider-manifest.json`. OpenRouter audits
separately retain the submitted provider constraint/preferences and the router
display identity, and explicitly do not treat that display label as exact
endpoint-slug attestation.

The checked-in primary and replication configurations are two-user pilot
designs, not paper power settings or completed runs. Both compare all three LLM
information views with one fixed model/effort setting and declare hard ceilings
of 900 requests and 6,000,000 conservative tokens. Their default
`calibration = "temperature"` stage first collects a fixed matched-provenance
probe on one declared development user, fits one temperature per LLM updater
view using development labels only, and applies those calibrators to test/run
outputs. If an adaptive live run fails, rerunning the same configuration with
`--execute-live --resume-failed-live` archives the failed artifact under
`.failed-runs/` and resumes its external journal into a fresh deterministic
run path.

Additional bounded configurations cover the remaining live and external
evidence workflows:

| Purpose | Configurations | Exact bounded workload |
| --- | --- | ---: |
| A live profile writing | `configs/live/experiment_a_{openai,openrouter}.toml` | 848 physical attempts per provider |
| B live closed loop | `configs/live/experiment_b_{openai,openrouter}.toml` | 768 trajectory + 96 calibration = 864 attempts per provider |
| C live evaluation validity | `configs/live/experiment_c_{openai,openrouter}.toml` | 768 evaluation + 48 calibration = 816 attempts per provider |
| Gate 4 source generation | `configs/offline/gate4_source.toml` | Offline; 640 decoder requests per source and 80 native actions |
| C external rescore source | `configs/offline/experiment_c_rescore_source.toml` | Offline; 360 decoder requests per source |
| Gate 6 live OAT | `configs/live/sensitivity_{openai,openrouter}.toml` | 576-attempt upper bound per provider |

These are ceiling-safe pilots, not power commitments or completed paper
experiments. Configuration validation and live startup both recompute the
whole-design preflight and fail before credential access when a workload no
longer fits.

The runner retains the fitted parameters in
`models/llm-calibration.json`, development raw responses in
`llm/development-raw-responses.jsonl`, development raw-versus-calibrated metrics
in `metrics/llm-development-calibration.jsonl`, and runtime/test raw responses
in `llm/test-raw-responses.jsonl`; calibrated responses remain in the ordinary
LLM exchange artifacts. This separation makes the no-test-label fitting
boundary auditable.

`llm evaluation-suite` is the paper-facing two-role orchestrator. Without
`--execute-live` it reads no credential, validates the immutable matched
primary/replication configs, and writes a combined index containing both config
hashes, distinct run IDs, isolated journal locations, and each role's own hard
ceilings. The checked pilot design has a conservative upper bound of 848 calls
per role, leaving 52 calls of headroom under the 900-request ceiling. Adding
`--execute-live` runs both configs under separate provider ledgers and updates
that index; it never merges their response or audit files.

Two OpenAI decoder variants remain useful operational checks, but the selected
Gate 4 pair uses `anthropic/claude-sonnet-5` and
`google/gemini-3.6-flash` through OpenRouter. These are distinct model
families behind one shared gateway, not two independently authenticated
first-party transports, and their errors are not claimed to be statistically
independent. Gate 4 therefore requires a named responsible researcher to
review the shared gateway, model lineage, prompt, training-data, and
adjudication dependencies before import. The native-action path sends exact
retained native memory and the exact terminal suite to OpenAI Sol and accepts
only model-produced, schema-bound actions—not a local persona or belief
projection. See
[Live execution](docs/live-execution.md) and
[Configuration](docs/configuration.md).

The repository-selected automated decoder collection requires
`OPENROUTER_API_KEY`; the native action collection separately requires
`OPENAI_API_KEY`. Direct
`ANTHROPIC_API_KEY` and `GEMINI_API_KEY` collectors remain available as
optional first-party-origin replications, not prerequisites for the selected
OpenRouter workflow.

The broader simulator robustness configuration is now a baseline-first,
19-point one-at-a-time design. It varies every declared axis while explicitly
recording that interactions among axes are not estimable. The two live Gate 6
pilots use smaller 11-point one-at-a-time grids and remain separate from the
simulator-only robustness run.

## Run artifacts

A run is intended to be self-describing:

```text
runs/<run-id>/
├── config.resolved.json
├── config.source.toml
├── manifest.json
├── environment.json
├── splits.json        # Experiments A–C
├── population/        # Experiments A–C
├── events/
├── metrics/
│   └── summary.json
├── models/
├── tables/
├── figures/
├── llm/
└── SHA256SUMS
```

Exact experiment files vary. Canonical scientific records are JSON or JSON
Lines whose version boundary is carried by the row, a generated schema `$id`,
the containing artifact, or the software release; see the data-model inventory
for the carrier used by each record. The foundational manifest records
configuration identity, source revision when available, and completion state;
the checksum verifier rejects changed/missing files, unsafe or duplicate
manifest paths, and extra unlisted files. It also requires complete status,
matching run/config identity, a current-schema resolved config, and a summary.
Run directories are ignored by Git.
Curated release artifacts belong under `artifacts/` and must include a manifest
and provenance statement. `artifact freeze` additionally requires the retained
`config.source.toml`, or a manifest entry that explicitly identifies and
hash-binds a programmatically constructed configuration.

See [Reproducibility](REPRODUCIBILITY.md), [Data model](docs/data-model.md), and
[Artifact policy](artifacts/README.md).

## Repository guide

| Path | Purpose |
| --- | --- |
| `src/cape_loop/` | Library and command-line implementation |
| `analysis/` | Optional external statistical pipelines with frozen contracts |
| `configs/smoke.toml` | Fast offline quickstart |
| `configs/offline/` | Deterministic study and source-packet designs |
| `configs/live/` | Budget-bounded OpenAI/OpenRouter pilots |
| `schemas/` | Exported interchange schemas |
| `tests/` | Unit, integration, statistical-contract, and regression tests |
| `examples/` | Small public API and workflow examples |
| `docs/` | Scientific, architectural, and operational documentation |
| `data/` | Small fixtures and versioned data manifests, not private data |
| `artifacts/` | Curated, checksum-verified release artifacts |
| `paper/` | Paper-facing figure/table guidance and, when released, sources |

The stable component and directory map is in
[Architecture](docs/architecture.md#repository-layout).

## Scope and interpretation

CAPE-Loop diagnoses preference-inference behavior. It is not:

- a new recommendation algorithm;
- a claim that an agent changes the user's latent preference;
- a universal normative model of human choice;
- evidence that a hypothesis holds merely because the code can compute its
  metric;
- authorization to conduct or release a human-participant study without the
  required ethics review and consent.

Exact inference is optimal only under the declared simulator. Human judgments
validate pragmatic evidential ordering rather than reveal metaphysical
preferences. Please read [Ethics and limitations](docs/ethics-and-limitations.md)
before adapting the benchmark to real user data.

## Contributing

Contributions are welcome, especially new domain fixtures, response-model
robustness checks, updater adapters, and validation tests. Scientific invariants
and latent-information boundaries must remain explicit. Start with
[CONTRIBUTING.md](CONTRIBUTING.md) and the
[architecture extension points](docs/architecture.md#extension-points).

Security and private-data concerns should follow [SECURITY.md](SECURITY.md), not
public issue reports.

## Citation and license

Citation metadata is provided in [CITATION.cff](CITATION.cff). It intentionally
does not invent a DOI, repository URL, author list, or publication status.

Code is licensed under the Apache License 2.0; see [LICENSE](LICENSE) and
[NOTICE](NOTICE). Released data or third-party materials may have separate terms
documented beside the relevant artifact.
