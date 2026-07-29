# CAPE-Loop documentation

CAPE-Loop audits whether a persistent profile writer interprets a response in
light of the options, presentation, and agent policy that caused it to be
observed. The documentation is intentionally organized around a small set of
canonical guides; protocol details should be added to these files instead of
creating one-off pages.

## Choose a path

- **Run the code:** [Getting started](getting-started.md), then
  [Configuration](configuration.md).
- **Understand the paper:** [Proposal](proposal.md), then
  [Scientific design](scientific-design.md).
- **Understand an experiment or gate:** [Experiments](experiments.md), then
  [Metrics and estimands](metrics.md).
- **Understand full audit records, compact analysis rows, and generated data:**
  [Data model](data-model.md).
- **Use OpenAI, OpenRouter, replay, external decoders, or native actions:**
  [Live execution](live-execution.md).
- **Audit what is actually ready:** [Implementation status](implementation-status.md)
  and the root [reproducibility contract](../REPRODUCIBILITY.md).

## Canonical guides

| Guide | Scope |
| --- | --- |
| [Getting started](getting-started.md) | Install, validate, generate and inspect full or compact synthetic outputs, run experiments, and approach live pilots safely |
| [Configuration](configuration.md) | Complete TOML and strict-validation reference |
| [Architecture](architecture.md) | End-to-end flow, component contracts, information boundaries, native memory, repository layout, and extension points |
| [Scientific design](scientific-design.md) | Causal question, evaluation tracks, controls, and claim boundaries |
| [Experiments](experiments.md) | Experiments A–C, sensitivity, human-study support, correction debt, controls, external rescoring, and Gates 1–6 |
| [Metrics and estimands](metrics.md) | Metric formulas, inference units, uncertainty, H1/H2/H7, and confirmatory-analysis relationship |
| [Data model](data-model.md) | Dataset production, splits, schemas, full-versus-compact records, record joins, run outputs, and artifact lifecycle |
| [Live execution](live-execution.md) | Replay and provider workflows, model selection, budgets, journals, collection, admission, and recovery |
| [Ethics and limitations](ethics-and-limitations.md) | Interpretation, privacy, human-study, provider, and release limits |
| [Implementation status](implementation-status.md) | Tested capabilities, external dependencies, local diagnostic boundary, gates, and completion criteria |
| [Paper proposal](proposal.md) | Frozen scientific intent and planned claims; result placeholders remain unfilled until eligible evidence exists |

The optional confirmatory R/lme4 operator and statistical contract lives beside
its code in
[analysis/confirmatory-mixed-effects/README.md](../analysis/confirmatory-mixed-effects/README.md).

## Repository policies

- [Reproducibility](../REPRODUCIBILITY.md)
- [Data directory and release policy](../data/README.md)
- [Artifact policy](../artifacts/README.md)
- [Schema directory](../schemas/README.md)
- [Contributing](../CONTRIBUTING.md)
- [Security](../SECURITY.md)

## Status language

- **Implemented:** executable and covered by an offline automated test.
- **Implemented diagnostic:** executable, but intentionally
  `claim_status = "not_claimed"`.
- **Provider-capable:** live request construction and execution exist, but no
  eligible response corpus is checked in.
- **External-evidence-dependent:** completion requires credentials, reviewed
  external sources, human participants, or another outside input.
- **Planned:** described by the proposal but not executable in the current
  source tree.

An implemented component is not an empirical result. Smoke runs validate
plumbing, frozen archives validate identity, and gate reports validate declared
computations; none of those facts alone establishes a paper claim.
