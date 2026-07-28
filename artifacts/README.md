# Artifacts

This directory is for deliberately curated, checksum-verified release bundles.
Ordinary local experiment output belongs under the ignored `runs/` directory.

At repository initialization there are no empirical paper artifacts. This README
must not be interpreted as evidence that any hypothesis or stage gate has been
supported.

## Artifact classes

Every bundle declares one status:

- **smoke:** validates execution and artifact structure only;
- **pilot:** used to tune implementation or power assumptions, not final test
  evidence;
- **paper:** frozen evidence for a named paper table, figure, or claim;
- **reanalysis:** derived from an existing frozen artifact without new data
  collection.

## Required bundle contents

```text
artifacts/<bundle-id>/
├── README.md
├── manifest.json
├── source-config.toml
├── config.resolved.json
├── environment.json
├── runs-or-records/
├── tables/
├── figures/
└── SHA256SUMS
```

An external archive may use a different physical layout, but its manifest must
provide equivalent information.

Command-generated append-only review artifacts (for example Gate 4 reviews,
Experiment C external rescoring, and the Experiment C multi-seed diagnostic)
use their command-specific minimal checksum-bound layouts while under active
analysis. A paper release must place or reference that verified review from a
curated bundle satisfying the structure above; the small review directory
alone is not a paper bundle.

The bundle README states:

- purpose and status;
- source revision/archive digest;
- run IDs and configuration digest;
- population and split manifests;
- expected and completed cells;
- external LLM or human inputs;
- generation and verification commands;
- exclusions and known limitations;
- applicable code/data/provider terms;
- mapping to paper table, figure, or text identifiers.

## Acceptance rules

Do not add a bundle when:

- a run manifest is incomplete;
- checksums fail;
- expected factorial cells are silently absent;
- test labels informed fitting/calibration;
- external responses cannot be matched to requests;
- human data lack appropriate release authorization;
- values are placeholders or manually copied without source records;
- a smoke/pilot run would be mistaken for final evidence.

## Verification

For a standard run:

```bash
PYTHONPATH=src python -m cape_loop verify RUN_DIR
```

For a curated bundle, verify its top-level `SHA256SUMS` using an appropriate
checksum tool and then verify included run directories individually.

Checksum validity establishes file integrity, not scientific validity. Review
the bundle manifest and stage-gate evidence separately.

## Paper linkage

The [paper directory](../paper/README.md) refers to bundle and run IDs; it does
not own independent copies of numerical results. Regenerated tables and figures
should match the artifact checksums or be labeled as a new reanalysis.

See [Run outputs and artifacts](../docs/data-model.md#run-directory-and-output-lifecycle)
and [Reproducibility](../REPRODUCIBILITY.md).
