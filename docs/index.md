# CAPE-Loop documentation

CAPE-Loop is a causal audit for persistent profile updating in agents. It tests
whether a profile writer interprets a response conditional on the options and
agent actions that caused that response to be observed.

The shortest path through the documentation is:

1. [Getting started](getting-started.md) — run the offline checks and a smoke
   configuration.
2. [Scientific design](scientific-design.md) — understand the claim and the
   experimental controls.
3. [Architecture](architecture.md) — follow information through the simulator,
   updater, and evaluator.
4. [Experiments](experiments.md) and [Metrics](metrics.md) — understand each
   evaluation and its outputs.
5. [Reproducibility](reproducibility.md) — preserve and audit a run.

## Reference

| Document | What it answers |
| --- | --- |
| [Repository map](repository-map.md) | Where does each component live? |
| [Component reference](components.md) | What does each runtime component do? |
| [Dataset card](dataset-card.md) | What data is generated, how is it split, and what external evidence is absent? |
| [Data model](data-model.md) | What records connect policy to memory update? |
| [Data splits](data-splits.md) | How are train, development, and test surfaces made disjoint and audited? |
| [Configuration](configuration.md) | How is a run declared in TOML? |
| [LLM exchange](llm-exchange.md) | How are external model requests represented and supplied response corpora replayed? |
| [Native memory](native-memory.md) | How are non-probabilistic memory systems evaluated? |
| [Gate 4 live collection](gate4-live-collection.md) | How are distinct-family decoder judgments and real native terminal actions collected safely? |
| [Mixed-effects analysis](mixed-effects-analysis.md) | How are the proposal's confirmatory models fit and audited in R? |
| [Outputs](outputs.md) | Which files does a run produce? |
| [Extending](extending.md) | How do I add a domain, policy, updater, or metric? |
| [Ethics and limitations](ethics-and-limitations.md) | What should not be inferred or released? |
| [Implementation status](implementation-status.md) | Which planned capabilities are present? |

## Project documents

- [Paper proposal](proposal.md): scientific source of truth and planned claims.
- [Implementation plan](implementation-plan.md): engineering contract and frozen
  reference defaults.
- [Root reproducibility checklist](../REPRODUCIBILITY.md): release checklist.
- [Data policy](../data/README.md): tracked and untracked data.
- [Artifact policy](../artifacts/README.md): evidence packaging.
- [Paper directory policy](../paper/README.md): figure, table, and manuscript
  provenance.

## Status convention

Documentation uses these labels:

- **Implemented:** exercised by source code and an offline automated test.
- **Exchange-supported:** CAPE-Loop can generate or consume the required records,
  but an external provider or human study is needed.
- **Stage-gated:** code may exist, but the corresponding scientific analysis must
  not be promoted unless its prerequisite gate passes.
- **Planned:** described by the proposal but not part of the current executable
  surface.

“Implemented” never means that a hypothesis is true. Scientific results require
retained runs and statistical analysis.
