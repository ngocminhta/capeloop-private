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
replay plus explicitly authorized, budget-limited OpenAI Responses API
execution.
It does **not** contain fabricated paper results or imply that any scientific
stage gate has passed. Consult
[implementation status](docs/implementation-status.md) before interpreting an
available component or artifact.

No live provider execution was performed to create the checked-in repository,
and no API key is included. Live-model, external-decoder, and human-study
results remain data-collection tasks; implemented code paths do not make those
results exist or establish a scientific claim.

The paper design is in [the proposal](docs/proposal.md). The engineering
contract is in [the implementation plan](docs/implementation-plan.md).

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
  and a content-bound positive/negative control protocol.
- Endogenous profile–policy–response loops with a same-history action-aware
  shadow posterior.
- Fixed-history versus closed-loop evaluation using a common exogenous terminal
  diagnostic battery.
- Structured beliefs and inspectable native episodic, semantic/persona, and
  provenance-linked memory.
- Provider-neutral JSON Lines exchange, prompt-hash-checked offline replay, and
  an opt-in OpenAI Responses API executor with hard request/token budgets and
  resumable audit-first journals.
- Development-only temperature calibration for LLM probability vectors, fitted
  separately for each information view with raw and calibrated records retained.
  B/C additionally score cached raw/calibrated terminal vectors on the same
  realized history; multi-turn rows explicitly do not claim a recursively raw
  counterfactual trajectory.
- Deterministic human evidence-strength material packets, strict de-identified
  imports, and analysis helpers; no recruitment, hosting, or ethics approval.
- Versioned JSON artifacts, semantic-keyed random streams, checksums, and
  result-free gate reports.
- Executed, feature-matched train/development/test surface families with a
  retained concrete [leakage audit](docs/data-splits.md).
- A checked-in simulator-sensitivity grid over decision noise, presentation
  strength, profile strength, and trajectory length.

The standard-library-only core requires Python 3.11 or newer. Offline commands
never require credentials. A provider call is possible only through a live
command or an `mode = "openai"` run with the explicit `--execute-live` flag.

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
  evaluation;
- **Experiment D support:** human pragmatic evidence-strength study materials.

Correction-debt, held-out paraphrase-transfer, and broader alternative-model
robustness code paths are implemented, but their existence is not evidence that
the corresponding paper stages were executed or passed.
[Experiments](docs/experiments.md) explains the cells, controls, and output
contracts.

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
  configs/openai_primary.toml --execute-live
PYTHONPATH=src python -m cape_loop llm evaluation-suite \
  configs/openai_primary.toml configs/openai_replication.toml \
  --output-root runs
PYTHONPATH=src python -m cape_loop decoder-study plan-openai \
  decoder-requests.jsonl
PYTHONPATH=src python -m cape_loop gate-review import-native \
  runs/EXPERIMENT-B requests.jsonl judgments.jsonl truth.jsonl \
  recorded-actions.jsonl source-review.json artifacts/GATE4-REVIEW
PYTHONPATH=src python -m cape_loop gate-review verify \
  artifacts/GATE4-REVIEW
PYTHONPATH=src python -m cape_loop llm validate responses.jsonl
```

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
resume completed requests without rebilling them.

The checked-in primary and replication configurations are two-user pilot
designs, not paper power settings or completed runs. Both compare all three LLM
information views with one fixed model/effort setting and declare hard ceilings
of 900 requests and 6,000,000 conservative tokens. Their default
`calibration = "temperature"` stage first collects a fixed matched-provenance
probe on one declared development user, fits one temperature per LLM updater
view using development labels only, and applies those calibrators to test/run
outputs. If an adaptive live run fails, rerunning the same configuration with
`--execute-live --resume-failed-live` archives the failed artifact under `.failed-runs/` and
resumes its external journal into a fresh deterministic run path.

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
ceilings. The checked pilot design has a conservative upper bound of 752 calls
per role, leaving 148 calls of headroom under the 900-request ceiling. Adding
`--execute-live` runs both configs under separate provider ledgers and updates
that index; it never merges their response or audit files.

Two OpenAI decoder variants are useful operational checks, but common provider
infrastructure and model lineage mean they do not prove independent judgment.
Independent human or externally administered decoding remains necessary for a
strong independence claim. See [LLM exchange](docs/llm-exchange.md) and
[Configuration](docs/configuration.md).

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
| `configs/` | Reviewable experiment and smoke configurations |
| `schemas/` | Exported interchange schemas |
| `tests/` | Unit, integration, statistical-contract, and regression tests |
| `examples/` | Small public API and workflow examples |
| `docs/` | Scientific, architectural, and operational documentation |
| `data/` | Small fixtures and versioned data manifests, not private data |
| `artifacts/` | Curated, checksum-verified release artifacts |
| `paper/` | Paper-facing figure/table guidance and, when released, sources |

The detailed component inventory is in
[Repository map](docs/repository-map.md).

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
[CONTRIBUTING.md](CONTRIBUTING.md) and [Extending CAPE-Loop](docs/extending.md).

Security and private-data concerns should follow [SECURITY.md](SECURITY.md), not
public issue reports.

## Citation and license

Citation metadata is provided in [CITATION.cff](CITATION.cff). It intentionally
does not invent a DOI, repository URL, author list, or publication status.

Code is licensed under the Apache License 2.0; see [LICENSE](LICENSE) and
[NOTICE](NOTICE). Released data or third-party materials may have separate terms
documented beside the relevant artifact.
