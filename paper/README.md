# Paper materials

This directory is reserved for released manuscript sources and the
machine-readable linkage between paper figures/tables and CAPE-Loop artifacts.

No accepted-paper status, author list, venue metadata, DOI, or numerical result
is asserted here. The current scientific design is
[docs/proposal.md](../docs/proposal.md).

## Provenance rule

Every paper number, table, and figure must be generated from a named,
checksum-verified artifact. Do not hand-copy values into paper sources without a
manifest link.

A future paper manifest should record:

```json
{
  "schema_version": 1,
  "paper_version": "draft identifier",
  "outputs": [
    {
      "paper_id": "figure-1",
      "artifact_bundle": "bundle identifier",
      "source_run_ids": ["run identifier"],
      "generation_command": "documented deterministic command",
      "output_sha256": "..."
    }
  ]
}
```

## Suggested layout when manuscript sources are released

```text
paper/
├── README.md
├── manifest.json
├── main.tex
├── sections/
├── bibliography.bib
├── figures/
├── tables/
└── Makefile
```

Publisher or ACL style files may have separate redistribution terms and should
not be copied into this repository without permission. Record third-party file
sources and versions.

## Result discipline

- `[TBD]` proposal placeholders are not results.
- Smoke and pilot values are not final paper evidence.
- Gate machinery does not establish gate passage.
- Missing, failed, and excluded cells must remain documented.
- Raw and calibrated results must be distinguishable.
- Controlled identical-response and naturally sampled analyses must be labeled.
- Fixed-history and closed-loop regimes must be labeled.
- Changes after unblinding final test results require a transparent new analysis
  version.

Citation metadata for the software is in [CITATION.cff](../CITATION.cff).
Paper citation metadata should be added only when real authorship and publication
details are available.
