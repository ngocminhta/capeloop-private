# Repository map

This page maps the current checkout. It describes files that exist now, not a
planned future layout.

## Top-level files

| Path | Purpose |
| --- | --- |
| `README.md` | Project scope, status, quick start, and documentation entry points |
| `pyproject.toml` | Alpha package metadata, Python `>=3.11`, empty dependency list, and `cape-loop` entry point |
| `Makefile` | `doctor`, `test`, and `check` aliases using `PYTHONPATH=src` |
| `REPRODUCIBILITY.md` | Release-level reproduction checklist |
| `LICENSE`, `NOTICE` | Apache License 2.0 and attribution notice |
| `CITATION.cff` | Entity-authored software citation without DOI/repository/publication claims |
| `CONTRIBUTING.md` | Development and scientific-integrity requirements |
| `CODE_OF_CONDUCT.md` | Community behavior and enforcement policy |
| `SECURITY.md` | Private vulnerability and sensitive-data reporting guidance |
| `.env.example` | Environment-variable naming reference for live provider credentials; not automatically loaded |
| `.gitignore` | Python, editor, credential, build, and local-run exclusions |

## Source package

`src/cape_loop/` is the standard-library implementation.

| Module | Responsibility |
| --- | --- |
| `__init__.py` | Package version and public `load_config`/`AppConfig` imports |
| `__main__.py` | `python -m cape_loop` entry point |
| `cli.py` | Command parsing, run dispatch, replay/live LLM helpers, decoder/human analysis, correction-debt diagnostics, schema export, and artifact freeze/verify |
| `config.py` | Strict TOML sections, enumerations, defaults, and per-experiment contracts |
| `schemas.py` | Latent, option, context, provenance, observation, update, and trajectory dataclasses |
| `domains.py` | Travel/writing attributes and stable full/isolated option pools |
| `rng.py` | Canonical semantic digests, seeds, uniform/Gumbel draws, and weighted choice |
| `response.py` | Random-utility and rule-based response models; intrinsic welfare/regret |
| `elicitation.py` | Matched balanced/restricted/default/suggested anchor construction |
| `beliefs.py` | Marginal, theta-joint, and theta×susceptibility joint beliefs |
| `inference.py` | Exact finite Bayesian updates and likelihood wrappers |
| `fitting.py` | Aware/unaware feature models and deterministic Adam fitting |
| `training.py` | Randomized training records, model bundles, diagnostics, and bundle calibration |
| `calibration.py` | Development-only temperature fitting |
| `population.py` | Split-constrained latent users and four initial-profile seeds |
| `splits.py` | Deterministic group split manifests and terminal-template checks |
| `policies.py` | Balanced, soft, exploratory, fixed-bias, and hard-filter policies |
| `updaters.py` | Update views/state protocol and structured/LLM replay updater registry |
| `native.py` | Three native memory variants and two deterministic blinded decoders |
| `metrics.py` | Belief, confidence, decomposition, and self-confirmation metrics |
| `statistics.py` | Clustered bootstrap/contrasts, CR1 marginal OLS, multiplicity, reliability, ranking intervals, inferential partial orders, and set-valued selection regret |
| `gates.py` | Result-free machine-readable stage-gate diagnostics |
| `power.py` | Pilot power and multiple-testing helpers |
| `sensitivity.py` | Multi-axis random-utility/rule-based grid points, phase criteria, and observed-grid boundary inference |
| `file_lock.py` | Dependency-free POSIX/Windows advisory lock abstraction for live collections and coherent evidence review |
| `llm_exchange.py` | Strict request/response records and hash-bound local replay |
| `openai_provider.py` | Explicit-opt-in Responses API execution, strict structured output, retry/budget controls, audit journaling, and resumability |
| `openrouter_provider.py` | Explicit-opt-in OpenRouter Chat Completions execution, strict structured output, route/cache/metadata validation, gateway/upstream audit, and resumability |
| `evaluation_suite.py` | Immutable matched primary/replication planning, isolated live dispatch, per-role budget lineage, and combined indexing |
| `heldout.py` | Leakage-guarded surface paraphrases, Gate 1 transfer criterion, terminal-v2 items, action bindings, and scoring |
| `decoder_study.py` | Blinded external decoder exchange, source audit, development calibration/test reliability, and de-identified human collection analysis |
| `external_decoder_providers.py` | Explicit-opt-in Anthropic/Gemini decoder collection with blinded payloads, budgets, journals, and resume |
| `native_action_provider.py` | Explicit-opt-in OpenAI execution of native-state terminal actions with source-run/state/suite bindings |
| `human_study.py` | Study items, deterministic blinding/order, codebooks, and base rating validation |
| `correction_debt.py` | Stage-gated paired correction/recovery protocol and transparent reference adapter |
| `gate_review.py` | Strict recorded-action/source-review contracts, verified Experiment B evidence binding, recomputed Gate 4, and immutable review verification |
| `verbalization.py` | Constrained choice acknowledgements |
| `artifacts.py` | Run creation, config/source identity, SHA-256 writing, and strict verification |
| `release.py` | Deterministic verified-run tar freezing and sidecar verification |
| `reporting.py` | Dependency-free CSV aggregation and SVG lines |
| `schema_export.py` | Public JSON Schema definitions and deterministic export |
| `runner.py` | Study preparation, experiment dispatch, confirmatory/held-out/decoder output writing, gate reports, live provider orchestration, failure capture, and run reuse |
| `py.typed` | PEP 561 typing marker |

`src/cape_loop/experiments/` contains:

| File | Responsibility |
| --- | --- |
| `provenance.py` | Experiment A rows, matched exclusions, and audit execution |
| `closed_loop.py` | Experiment B trajectories, shadow state, action influence, decomposition, and predicate |
| `evaluation.py` | Experiment C fixed histories, heldout-terminal-v2 battery adapter, native score basis, inferential ranking, and set-valued ESR |
| `__init__.py` | Public experiment exports |

See [Components](components.md) for inputs, outputs, and information boundaries.

## Configurations

`configs/` contains:

| File | Kind |
| --- | --- |
| `smoke.toml` | `provenance_audit` |
| `closed_loop.toml` | `closed_loop` |
| `evaluation.toml` | `evaluation_validity` |
| `sensitivity.toml` | Compact 81-point `sensitivity` grid |
| `sensitivity_full.toml` | Broader 384-point multi-axis/model-family sensitivity grid |
| `openai_primary.toml` | Explicit-opt-in primary live-provider Experiment A pilot |
| `openai_replication.toml` | Matched GPT-5.6 model-variant/tier replication pilot |
| `openrouter_gemini.toml` | Explicit-opt-in OpenRouter/Gemini routed Experiment A pilot with exact model and upstream-route controls |

All are strict executable declarations. The live-provider configs are request
plans, not evidence that a provider call was made. The OpenRouter config records
a shared-gateway route and is not strict Gate 4 first-party provenance. There
is no checked-in paper-frozen configuration/result pair.

## Optional analysis

`analysis/confirmatory-mixed-effects/` is an isolated R 4.6.1 project for the
two proposal mixed-effects formulas. It contains:

| Path | Purpose |
| --- | --- |
| `analysis-spec.json` | Frozen formulas, coding, estimands, contrasts, diagnostics, and failure policy |
| `run_analysis.R` | Strict source-run verification, row preparation, fitting, and artifact entry point |
| `R/io.R` | Checksum/run validation, schema checks, factor construction, and deterministic writers |
| `R/model.R` | Maximal lmerTest fits, diagnostics, emmeans contrasts, and machine-readable tables |
| `renv.lock`, `restore.R`, `DESCRIPTION` | Exact R/runtime dependency environment and installation |
| `analysis-result.schema.json` | Public result-object contract |
| `validate_contract.py` | Standard-library static validation when R is unavailable |
| `README.md` | Operator, statistical, failure, and output reference |

The directory contains protocol/software only, not a fitted result.

## Tests

The offline `unittest` modules are flat under `tests/`:

| File | Main coverage |
| --- | --- |
| `test_core.py` | Schemas, domains, semantic RNG, response normalization, welfare separation |
| `test_inference.py` | Beliefs, exact enumeration, aware/unaware fitting and capacity |
| `test_population_training.py` | Split populations, seed semantics, deterministic training, identifying fixture |
| `test_experiments.py` | Matched provenance, views, A/B/C, shadow/CRN, fixed histories, native ranking basis |
| `test_metrics.py` | ACUE, belief/decomposition formulas, five-clause predicate, ranking/ESR |
| `test_native.py` | Native state auditability, replayability, view rejection, and both blinded decoders |
| `test_gates_power.py` | Claim-free gate status, closed-loop clauses, power/multiplicity |
| `test_sensitivity.py` | Independent grid axes, model-family construction, phase classification/boundaries |
| `test_support.py` | Config contracts, calibration leakage, LLM replay/corpus identity, artifacts, schemas, language, study materials |
| `test_runner.py` | End-to-end Experiment A/B artifacts, gate adapter, checksums, and safe reuse |
| `test_confirmatory.py` | Experiment A oracle slopes, evidence ordering, clustered analyses, reliability, multiplicity, and pilot power |
| `test_ranking_inference.py` | Paired difference intervals, inferential tiers, and set-valued ESR |
| `test_missing_workflows.py` | Held-out surface/terminal contracts, external decoder/human analysis, and correction-debt pairing |
| `test_external_schemas.py` | Public held-out, decoder, human-collection, and provider-audit schemas |
| `test_openai_provider.py` | Provider request bodies, strict parsing, retry/budget/audit behavior, and resumability with no live network |
| `test_openrouter_integration.py` | OpenRouter request/routing/cache/metadata validation, mocked live execution, config/CLI integration, audits, and resumability |
| `test_evaluation_suite.py` | Credential-free suite planning, immutable role configs, isolated paths, explicit dispatch, and per-role ceilings |
| `test_live_integration.py` | Config/runner live-adapter integration under mocked provider execution |
| `test_release.py` | Deterministic tar freeze, config-origin gate, and sidecar verification |
| `test_gate_review.py` | Gate 4 action/source contracts, fail-closed external-evidence import, source-run non-mutation, and review checksums |
| `test_external_decoder_providers.py` | Keyless planning and mocked Anthropic/Gemini decoder execution, validation, budgets, audit, and resume |
| `test_native_action_provider.py` | Keyless planning and mocked native-action execution, source bindings, audit ordering, and recovery |
| `test_provider_cli.py` | Gate 4 model-manifest/default alignment, live authorization gates, immutable-source destinations, and command-wide locking |
| `test_file_lock.py` | Exclusive/shared lock behavior plus mocked Windows byte-range semantics |
| `test_mixed_effects_contract.py` | R-free validation of formulas, package locks, result status/claim boundary, and output inventory |

`test_support.py` also provides shared test helpers. Tests validate software
contracts, not paper findings.

## Exported schemas

`schemas/` contains:

```text
README.md
human-rating.schema.json
human-collection.schema.json
interaction-record.schema.json
llm-request.schema.json
llm-response.schema.json
openai-provider-audit.schema.json
openrouter-provider-audit.schema.json
heldout-paraphrase-case.schema.json
heldout-paraphrase-evaluation.schema.json
heldout-paraphrase-criterion.schema.json
external-decoder-request.schema.json
external-decoder-judgment.schema.json
external-decoder-provider-audit.schema.json
external-decoder-transport-attempt.schema.json
decoder-truth-label.schema.json
decoder-source-review.schema.json
native-terminal-action-record.schema.json
native-action-provider-audit.schema.json
native-action-transport-attempt.schema.json
gate4-review-artifact.schema.json
run-manifest.schema.json
trajectory.schema.json
user-state.schema.json
```

Regenerate them with:

```bash
PYTHONPATH=src python -m cape_loop schema export
```

Do not edit generated JSON independently of `schema_export.py`.

## Examples

`examples/structured_update.py` demonstrates a small structured update.
`examples/inspect_run.py` reads a retained run. Examples are instructional and
are not benchmark evidence.

## Documentation

| File | Purpose |
| --- | --- |
| `docs/index.md` | Documentation index |
| `docs/proposal.md` | Paper proposal and intended scientific claims |
| `docs/implementation-plan.md` | Engineering plan and reference defaults |
| `docs/scientific-design.md` | Formal scientific framing |
| `docs/architecture.md` | Causal data flow and trust boundary |
| `docs/dataset-card.md` | Synthetic benchmark provenance, contents, splits, reproduction, and limitations |
| `docs/data-model.md` | Record schemas and linkage |
| `docs/data-splits.md` | Executed split assets and leakage audit |
| `docs/configuration.md` | Exact TOML fields and strict contracts |
| `docs/getting-started.md` | Supported source-tree workflows |
| `docs/experiments.md` | Current A/B/C/sensitivity/human material behavior |
| `docs/native-memory.md` | Full native state and two-decoder evaluation |
| `docs/llm-exchange.md` | Hash-bound offline replay |
| `docs/gate4-live-collection.md` | Distinct-family decoder and real native-action provider collection |
| `docs/mixed-effects-analysis.md` | Optional R confirmatory formulas, planned contrasts, diagnostics, and artifacts |
| `docs/metrics.md` | Metric definitions |
| `docs/outputs.md` | Exact run files and verification |
| `docs/components.md` | Runtime component contracts |
| `docs/repository-map.md` | This file inventory |
| `docs/implementation-status.md` | Implemented and missing capabilities |
| `docs/external-evidence.md` | Boundaries and admission checklist for provider, decoder, human, mixed-effects, and paper evidence |
| `docs/reproducibility.md` | Detailed determinism guidance |
| `docs/ethics-and-limitations.md` | Interpretation, privacy, and human-study constraints |
| `docs/extending.md` | Extension guidance |

The proposal describes intended studies. The implementation-status page and
source code determine what is executable today.

## Data, artifacts, and paper directories

| Path | Policy |
| --- | --- |
| `data/README.md` | Data placement and privacy policy |
| `data/fixtures/` | Small redistributable fixtures when added |
| `data/manifests/` | Checked-in data identities when added |
| `data/model-suites/` | Versioned provider-role declarations; not model outputs |
| `artifacts/README.md` | Curated evidence-bundle requirements |
| `paper/README.md` | Figure/table/manuscript provenance policy |
| `paper/figures/`, `paper/tables/` | Paper-facing outputs only when backed by retained artifacts |

Local run directories are written under `runs/` by default and ignored by Git.
No checked-in path currently contains paper results.

## Repository automation

`.github/` contains:

- `workflows/ci.yml` — dependency-free Python 3.11–3.14 doctor and unittest
  checks;
- issue forms for bugs, features, and scientific/reproducibility concerns; and
- the pull-request scientific-integrity checklist.

CI does not install or call an LLM provider.
