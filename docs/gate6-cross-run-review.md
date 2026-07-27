# Cross-run Gate 6 robustness review

`kind = "sensitivity"` evaluates the simulator-facing Gate 6 clauses inside one
run. One run cannot establish multiple-model-family replication or bind the
same model to Experiment A's held-out natural-language paraphrases. The
`gate6-review` command performs that join offline.

It makes no provider calls and never changes a source run. Its output is a new,
atomic, checksum-bound artifact with `claim_status = "not_claimed"`.

## Required source evidence

Each declared family needs an explicit pair:

1. a complete, checksum-valid `kind = "sensitivity"` run whose phase target is
   the live `llm_full_context` updater; and
2. a complete, checksum-valid `kind = "provenance_audit"` Experiment A run
   using the same actual requested/returned model and provider evidence.

Both runs must retain prompts, responses, the LLM exchange manifest, provider
manifest, accepted provider audit, settled physical-attempt journal, and
development-only or no-calibration manifest. The sensitivity run must retain
its Gate 6 report, phase/domain rows, and fitted-model rows. Experiment A must
retain the fixed paraphrase suite, cases, paired scores, and transfer
criterion.

The importer recomputes the sensitivity clauses from the resolved grid and
phase records. It also recomputes Experiment A's transfer criterion and checks
that every `llm_full_context` paraphrase score hashes the belief produced by
the corresponding retained provider response.

Sensitivity runs must have the same scientific configuration. Only run name,
seed, output location, and LLM provider/model/transport fields may differ.
Calibration remains part of the matched scientific configuration.

## Strict declaration

Create a JSON file with exactly this shape:

```json
{
  "schema_version": 1,
  "artifact_kind": "gate6-cross-run-declaration",
  "declaration_id": "gate6-review-v1",
  "review_authority": {
    "responsible_researcher_id": "researcher-controlled-id",
    "reviewed_at_utc": "2026-07-27T12:00:00Z",
    "preregistration_reference": "URL, DOI, registry ID, or explicit not-preregistered record",
    "family_assignments_declared_before_outcome_review": true,
    "source_identities_reviewed": true
  },
  "statistical_independence_claimed": false,
  "pairs": [
    {
      "pair_id": "sol-pair",
      "family_id": "openai-gpt-5.6",
      "sensitivity_run": {
        "path": "runs/SENSITIVITY-SOL",
        "run_id": "SENSITIVITY-SOL",
        "sha256sums_sha256": "LOWERCASE_SHA256_OF_SOURCE_SHA256SUMS"
      },
      "experiment_a_run": {
        "path": "runs/EXPERIMENT-A-SOL",
        "run_id": "EXPERIMENT-A-SOL",
        "sha256sums_sha256": "LOWERCASE_SHA256_OF_SOURCE_SHA256SUMS"
      },
      "model_binding": {
        "provider_id": "openai",
        "provider_source_id": "openai-first-party-responses",
        "requested_model_id": "PINNED_REQUEST_MODEL",
        "response_model_id": "EXACT_RETURNED_MODEL",
        "upstream_provider_id": null,
        "upstream_model_id": null
      }
    },
    {
      "pair_id": "gemini-pair",
      "family_id": "google-gemini-3",
      "sensitivity_run": {
        "path": "runs/SENSITIVITY-GEMINI",
        "run_id": "SENSITIVITY-GEMINI",
        "sha256sums_sha256": "LOWERCASE_SHA256_OF_SOURCE_SHA256SUMS"
      },
      "experiment_a_run": {
        "path": "runs/EXPERIMENT-A-GEMINI",
        "run_id": "EXPERIMENT-A-GEMINI",
        "sha256sums_sha256": "LOWERCASE_SHA256_OF_SOURCE_SHA256SUMS"
      },
      "model_binding": {
        "provider_id": "openrouter",
        "provider_source_id": "reviewed-openrouter-google-route",
        "requested_model_id": "google/PINNED-MODEL",
        "response_model_id": "EXACT_RETURNED_MODEL",
        "upstream_provider_id": "EXACT_RETAINED_DISPLAY_PROVIDER",
        "upstream_model_id": "EXACT_RETAINED_UPSTREAM_MODEL"
      }
    }
  ]
}
```

Relative run paths are resolved from the declaration file. `run_id` and
`sha256sums_sha256` prevent a path from being silently swapped before import.
At least two pairs, unique family IDs, distinct sensitivity and Experiment A
runs, and distinct actual response model IDs are required. Unknown fields are
rejected.

For OpenRouter, upstream values are the exact retained response metadata. They
do not turn a display provider label into proof of a physical route. The
separate `provider_source_id` and `family_id` are responsible-researcher
declarations; the software deliberately does not infer either from names.

## Build and verify

```bash
PYTHONPATH=src python -m cape_loop gate6-review build \
  gate6-declaration.json artifacts/GATE6-REVIEW

PYTHONPATH=src python -m cape_loop gate6-review verify \
  artifacts/GATE6-REVIEW

PYTHONPATH=src python -m cape_loop gate6-review verify \
  artifacts/GATE6-REVIEW --reverify-sources
```

The output directory must not exist and must be outside every source run.
Source and output leaf symlinks, symlinks inside source/review trees, checksum
tampering, missing response coverage, ambiguous model identity, unresolved
transport attempts, incomplete paraphrase coverage, and source mutation all
fail closed. A sibling exclusive publication lock prevents two local builders
from validating the same absent destination and racing to publish it.

The artifact contains:

```text
declaration.json
evidence/pairs.jsonl
metrics/gate-6.json
review.json
manifest.json
SHA256SUMS
```

The ordinary verifier is portable and checks retained checksums and semantic
recomputation. `--reverify-sources` additionally reopens all declared runs,
verifies their `SHA256SUMS`, and reproduces every retained pair-evidence row.

## Six-clause decision rule

The review emits exactly the proposal's six Gate 6 criteria:

- another response model;
- broad simulator parameters;
- both domains;
- multiple caller-declared LLM families;
- held-out natural-language paraphrases; and
- exact and fitted action-aware references.

The four within-sensitivity clauses use a conservative conjunction across
pairs: any explicit failure is `false`, all passing is `true`, and missing
evidence remains `null`. Multiple-family replication requires a surviving
complete LLM phase result for every distinct declared family/model binding.
Paraphrase transfer uses the recomputed complete Experiment A result for every
pair.

Even when all six computational checks pass, the result is not a paper claim.
The responsible researcher must defend the family taxonomy and source
identity, preserve any preregistration timing, assess shared-provider or
shared-training dependence, and freeze the paper analysis. Distinct metadata
or model IDs are never described as statistical independence.
