# Experiment C multi-seed robustness review

The Experiment C runner performs a complete-latent-user clustered bootstrap
inside one run. The multi-seed reviewer checks whether its ranking conclusions
are reproduced across independently seeded, otherwise identical completed
runs. It is entirely offline and does not read credentials or call a provider.

This component satisfies the software side of proposal test 17. A reviewer
artifact is still a diagnostic: it cannot establish that the empirical study
was adequately powered, that Gate 5 passed, or that a paper claim is warranted.
Every output therefore retains:

```json
{"claim_status": "not_claimed"}
```

## Inputs and admission checks

`experiment-c-robustness review` accepts between 2 and 32 source run
directories. Each source must:

- pass the ordinary `SHA256SUMS`, manifest, resolved-config, and completion
  checks;
- have `experiment.kind = "evaluation_validity"`;
- retain `metrics/experiment-c-rankings.json`,
  `metrics/gate-report.json`, and `metrics/summary.json`;
- declare a positive `experiment.bootstrap_replicates`;
- use the complete-latent-user inference unit and stable
  `(split, regime, user_id, domain_id, replicate)` alignment key;
- retain one seeded clustered-bootstrap rank summary and the complete paired
  interval sets for every configured updater pair; and
- have a distinct nonnegative `run.seed`.

All sources must have byte-equivalent resolved scientific configuration after
removing only:

```text
run.name
run.seed
run.output_root
```

Updater order/set, all response and inference parameters, artifact settings,
LLM settings, `thresholds.ranking_tie_tolerance`, and every other field must
remain identical. The retained executable source-tree SHA-256 must also match.
This prevents a code or design change from being mislabeled as random-seed
variation.

The reviewer accepts completed source runs, not an external-decoder rescore
directory. External rescores have their own immutable review contract and
should first be repeated under a separately preregistered cross-seed design if
they are to become a new robustness input type.

## Compared dimensions

Nine dimensions are declared in code before any source values are read:

1. fixed-balanced development point ranking;
2. fixed-biased development point ranking;
3. endogenous closed-loop development point ranking;
4. fixed-balanced inferential top tier;
5. endogenous closed-loop inferential top tier;
6. fixed-balanced inferential partial order;
7. endogenous closed-loop inferential partial order;
8. Gate 5 criterion decision together with its computed status; and
9. the open-selected and closed-selected ESR development sets.

The reviewer does not choose a favorable metric or seed. For every dimension it
reports:

- every distinct value pattern and its bound run IDs/seeds;
- whether the result is unanimous;
- the exact modal stability proportion;
- the exact pairwise agreement proportion; and
- every disagreeing run pair with both retained values.

Proportions are stored as integer numerator, integer denominator, and a reduced
rational string such as `2/3`. No rounded decimal controls a decision. The
overall section reports the exact fraction of the nine dimensions that are
unanimous, but it deliberately sets `scientific_claim_inferred = false`.

Each source run remains one bootstrap-evidence unit. Its configured seed,
positive replicate count, cluster counts, inference method, and a digest of the
retained bootstrap summaries/intervals are copied into the review. Bootstrap
draws are not pooled, rerun, or treated as independent paper replications.

## Create and verify

First create at least two Experiment C configurations that differ only in the
three permitted run fields, especially `run.seed`, then execute and verify each
normally. Create the separate review with:

```bash
PYTHONPATH=src python -m cape_loop experiment-c-robustness review \
  artifacts/experiment-c-multiseed \
  runs/experiment-c-seed-1-... \
  runs/experiment-c-seed-2-...
```

Verify its self-contained checksums and semantic aggregates:

```bash
PYTHONPATH=src python -m cape_loop experiment-c-robustness verify \
  artifacts/experiment-c-multiseed
```

For the stronger check, repeat `--source-run` for every bound source. The
command re-verifies each original run and compares its current artifact
digests, observations, config identity, source identity, and seed to the
review:

```bash
PYTHONPATH=src python -m cape_loop experiment-c-robustness verify \
  artifacts/experiment-c-multiseed \
  --source-run runs/experiment-c-seed-1-... \
  --source-run runs/experiment-c-seed-2-...
```

Supplying only a subset fails because the exact source set is part of the
artifact identity.

## Output and immutability

The command refuses an existing output path, symlinked source/output leaf, a
source nested under the output, duplicate paths, duplicate seeds, incompatible
configs, unsafe checksum paths, unknown JSON fields at the ranking/gate
boundary, non-finite JSON numbers, or incomplete pair coverage.

It stages and fsyncs all bytes before atomically publishing this directory:

```text
artifacts/experiment-c-multiseed/
├── review.json
├── manifest.json
└── SHA256SUMS
```

`review.json` contains the exact source bindings, bootstrap evidence,
predeclared comparisons, disagreements, rational proportions, interpretation
boundary, and a content-derived artifact ID. `manifest.json` binds that review,
its normalized scientific-config digest, and every source `SHA256SUMS` digest.
The checksum file covers both retained JSON files and rejects extra files or
subdirectories.

The implementation verifies every source both before and after staging. It
writes nothing under a source run. Later verification with `--source-run`
detects a source that changed after review creation.

## Interpretation boundary

Unanimous results mean only that the nine serialized conclusions were exactly
equal across the supplied verified seeds. Disagreement is retained rather than
averaged away. Either outcome requires researcher review of sample size,
bootstrap diagnostics, seed coverage, external evidence, and the preregistered
claim rule. The artifact never changes a source Gate 5 report and never turns
`meets_computational_checks` into a scientific claim.
