# Experiment C external-decoder rescore

Experiment C finishes and checksums its original rankings before external
decoder judgments exist. The external-decoder workflow is therefore
append-only: a completed run exports blinded requests, and a later import writes
a different immutable directory. It never edits the source run.

This workflow applies only to the three native updater families:

- `episodic_memory`
- `semantic_memory`
- `provenance_linked_memory`

Structured updater rows remain byte-equivalent JSON objects in the rescored
metric file. The deterministic native projections retained by the original run
remain useful diagnostics, but they are not admitted as external judgments.

## Components and information boundaries

Each completed Experiment C run writes:

```text
decoder/experiment-c-external-requests.jsonl
decoder/experiment-c-truth-labels.researcher-only.jsonl
decoder/experiment-c-researcher-codebook.jsonl
decoder/experiment-c-external-design-manifest.json
```

Send only `experiment-c-external-requests.jsonl` to decoder providers. Its
payload contains the content-only native representation and a pseudonymous
state ID. Runtime validation recursively rejects protected payload keys,
including user, updater, memory-kind, and latent-truth fields.

The truth-label and codebook files are evaluator-only. The codebook binds every
request to:

- the exact `(split, regime, replicate, user, domain, updater)` row key;
- the complete source metric-row SHA-256;
- the common terminal battery ID and SHA-256;
- the complete retained native-state SHA-256; and
- the exact fixed-replay or endogenous source-record SHA-256.

The design manifest binds all three packet files, the full C metric-row set,
the terminal-battery set, and the source run identity. A run without retained
native states is marked ineligible for this import.

## Collect the two decoder families

The first-party distinct-family collector can consume the exported requests:

```bash
cape-loop decoder-study plan-distinct \
  RUN_DIR/decoder/experiment-c-external-requests.jsonl

cape-loop decoder-study execute-distinct \
  RUN_DIR/decoder/experiment-c-external-requests.jsonl \
  DECODER_COLLECTION_DIR \
  --execute-live
```

Collection is a separate, explicitly authorized step. The rescore command
itself performs no network calls and reads no API keys.

The import admits one fixed pair covering the whole packet. Every request must
have exactly two `external_model` judgments with distinct instance IDs, family
IDs, and source descriptors. The CLI requires one provenance mode:

- `--external-collection-dir` verifies that the positional judgment file is
  byte-identical to the complete selected Anthropic/Gemini first-party
  collection and validates its plan, attempt journal, provider audits,
  execution manifest, and evidence-file digests; or
- `--allow-reviewed-generic-decoders` accepts the family, instance, source, and
  `external_model` origin fields only as caller-declared metadata and makes no
  provider-provenance claim.

In both modes the metadata constraints establish design eligibility only; they
do not prove statistical independence.

## Import, calibrate, and rerank

```bash
cape-loop experiment-c-decoder import \
  RUN_DIR \
  DECODER_COLLECTION_DIR/judgments.jsonl \
  C_EXTERNAL_REVIEW_DIR \
  --external-collection-dir DECODER_COLLECTION_DIR
```

The command fails closed unless the completed run and every request, truth,
codebook, row, battery, and native-state hash verify exactly. The official
mode also fails unless the complete locked collection verifies and its
`judgments.jsonl` exactly matches the positional file.

For a manual or other generic corpus, the explicit alternative is:

```bash
cape-loop experiment-c-decoder import \
  RUN_DIR \
  GENERIC_JUDGMENTS.jsonl \
  C_EXTERNAL_REVIEW_DIR \
  --allow-reviewed-generic-decoders
```

This mode records `provenance_mode = "reviewed_generic_judgments"`,
`provider_provenance_validated = false`, and
`caller_declared_source_metadata_only = true`. It must not be described as a
first-party provider collection.

After provenance admission, the importer:

1. fits one temperature per decoder family using development labels only;
2. applies the frozen family calibrators to development and test judgments;
3. converts each calibrated marginal forecast to an independent-joint belief;
4. scores both beliefs on the exact common terminal battery;
5. replaces each native row's ranking score with the arithmetic mean of exactly
   those two family/source scores;
6. proves that no non-native row changed; and
7. reruns the complete-user paired ranking analysis, evaluation selection
   regret, and Gate 5.

User-supplied calibrators are not accepted. The calibration implementation
rejects test-label fitting, and the review records the expected count of three
development attribute forecasts per request and family.

## Immutable review artifact

The import refuses an existing or symlinked output leaf, a symlinked source-run
or judgment input, an output inside the immutable source run, and a symlinked
output parent. It holds an exclusive sibling lock named:

```text
.<output-name>.external-rescore.lock
```

The complete review is first written to a randomly named staging directory on
the same filesystem as the final destination. Every JSON/JSONL/checksum file is
flushed and fsynced, followed by the staging directories. Before publication,
the importer:

1. re-verifies the completed source run and requires its exact initial
   source-binding object to remain unchanged;
2. re-resolves the original source and judgment paths and rejects substitution
   by a symlink or another target;
3. requires the judgment file bytes to equal the initially parsed bytes; and
4. when official collection provenance was selected, revalidates that complete
   locked collection and its exact judgment binding; and
5. invokes the ordinary review verifier against the staged directory and the
   exact source run.

Only a fully verified stage is atomically renamed to the requested output path.
Any calculation, write, fsync, input-reverification, or staged-verification
failure removes the staging directory and releases the lock without exposing a
partial output. An existing destination is never reused or overwritten.
A stale lock after an operating-system or machine crash must be removed only
after confirming no import process is active and no final output exists.

The output directory contains:

```text
inputs/external-requests.jsonl
inputs/truth-labels.researcher-only.jsonl
inputs/researcher-codebook.jsonl
inputs/judgments.jsonl
metrics/external-decoder-scores.jsonl
metrics/experiment-c-rescored.jsonl
metrics/calibration.json
metrics/decoder-analysis.json
metrics/experiment-c-rankings.json
metrics/gate-5.json
review.json
manifest.json
SHA256SUMS
```

`external-decoder-scores.jsonl` retains each calibrated marginal belief and
common-battery score. `experiment-c-rescored.jsonl` retains all original rows;
native rows use
`score_basis = "mean_of_exactly_two_calibrated_external_decoder_families"`.
The source-run manifest, checksum manifest, C metrics, battery file, and decoder
design manifest are all SHA-256-bound by the review. `review.json` also retains
`validation.source_design.provenance_mode`,
`validation.source_design.provider_provenance_validated`, the generic-mode
caller-declaration flag, and—only for the official mode—the complete collection
input/summary bindings.

Verify the review alone, or also re-verify its exact source run:

```bash
cape-loop experiment-c-decoder verify C_EXTERNAL_REVIEW_DIR

cape-loop experiment-c-decoder verify C_EXTERNAL_REVIEW_DIR \
  --source-run RUN_DIR
```

Successful validation does not promote a scientific claim. The review,
manifest, and recomputed Gate 5 all retain `claim_status = "not_claimed"`.
Verification also rejects a symlinked review/source leaf, symlinked retained
file, unsafe checksum path, missing file, extra file, digest mismatch, or
source-binding mismatch.

## Public schemas

- `experiment-c-decoder-codebook.schema.json` describes the protected row/state
  binding.
- `experiment-c-external-score.schema.json` describes one calibrated
  family-specific belief and terminal-battery score.
- Existing external request, judgment, and truth-label schemas continue to
  govern the exchange files.
