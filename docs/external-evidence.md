# External evidence boundaries

CAPE-Loop separates executable research infrastructure from observations that
the repository cannot create honestly on its own. Passing an input validator,
running a smoke configuration, or freezing an archive establishes a software
or identity property. None of those actions establishes a paper finding.

## Evidence states

Use these states in run and artifact documentation:

| State | Meaning |
| --- | --- |
| `protocol_ready` | Requests, schemas, split rules, and analysis code exist |
| `awaiting_external_input` | A credential, judgment source, participant dataset, or outside statistical fit is required |
| `import_validated` | Hashes, schemas, bindings, and declared eligibility metadata passed |
| `analyzed_not_claimed` | The declared computation ran; scientific review and stage-gate interpretation remain |
| `frozen_not_claimed` | A checksum-valid run was deterministically archived; identity is fixed but no claim is implied |
| `paper_evidence` | Authors have reviewed prerequisites, exclusions, uncertainty, limitations, and frozen provenance for a named claim |

The code's gate reports intentionally stop at computational status and retain
`claim_status = "not_claimed"`.

## External-input matrix

| Evidence | Repository supplies | Must come from outside | Minimum retained record |
| --- | --- | --- | --- |
| Live profile-writer evaluation | Strict requests/outputs, model-role config, budgets, retries, audit journal, replay, and development-only per-updater temperature calibration | API credential, spending authorization, actual provider responses | Resolved model/version, prompt/request hashes, response IDs, reasoning parameters, usage, timestamps, raw/active responses, calibration, audit |
| Gate 1 paraphrase transfer | Split-safe surface suite, source/case hashes, fitted-aware scoring, completeness logic | Complete `llm_full_context` responses for every required case | Suite digest, case records, paired updater scores, transfer criterion |
| External native decoding | Blinded requests, researcher-only truth/codebook, source audit, development calibration, test metrics | At least two genuinely distinct judgment sources per request | Request/judgment hashes, instance/family/source descriptors, calibration, raw/calibrated reliability and agreement |
| Native end-to-end terminal actions | Held-out v2 action schema, exact item/content bindings, scoring, transparent reference adapters, and an origin/budget-locked resumable OpenAI collection path | Paid or replayed actions actually emitted by the declared native system; none are checked in | Native state/system version, suite digest, per-item bound actions, transport-attempt journal, live/replay mode, execution audit, score |
| Human pragmatic ordering | Blinded packet, codebook, collection schema, eligibility checks, paired analysis | Ethics determination, approved consent, recruitment, hosting, compensation, collected de-identified ratings | Protocol versions, determination ID, packet digest, exclusions, analysis, privacy/retention statement |
| Confirmatory mixed effects | Version-pinned R harness for both exact formulas, turn-level B reconstruction, maximal random effects, validation, contrasts, diagnostics, and result schema; clustered CR1 remains separate | Verified preregistered study runs with A prior-strength variation, executed R fit, and responsible statistical review | Canonical config/input/source digests, formula, random effects, software/version, optimizer/scaled-gradient convergence, pointwise intervals, contrasts, multiplicity family |
| Correction debt on a studied system | Stage-gated exact-pair protocol and diagnostic adapter | Prerequisite gate review plus a real LLM/native adapter and results | Gate-review record, adapter/version, protocol digest, arms, pair debts, stage summaries |
| Paper release | Run verification and deterministic tar/sidecar | Author approval, frozen paper configuration/results, authorship and repository/DOI metadata | Archive/sidecar, artifact README, claim-to-run map, author/venue metadata |

## Live model admission

Never paste or commit an API key. Store it only in the environment variable
named by the resolved configuration. Before adding `--execute-live`, review:

- provider, exact model or version field, model role, and reasoning effort;
- maximum request count, output tokens per request, and total-token ceiling;
- whether prompts may be retained and released;
- destination and recovery-journal policy; and
- the consequence of partial failure and the conditions for
  `--resume-failed-live`.

The run must retain the credential-free provider audit and the exact
replay-compatible responses it consumed. Provider code, a dry-run plan, or a
mocked test is `protocol_ready`, not a model result.

With `llm.calibration = "temperature"`, the run first uses only development
users to fit one temperature for each updater information view. It must retain
development raw responses/metrics, the locked calibrators, test raw responses,
and active calibrated responses. With `llm.calibration = "none"`, record the
uncalibrated ablation explicitly. Executable calibration code is not an
empirical calibration result until provider or replay responses exist.

## Decoder admission

Only `decoder/external-requests.jsonl` is decoder-visible. Never send:

```text
decoder/truth-labels.researcher-only.jsonl
decoder/researcher-codebook.jsonl
```

Each imported judgment must repeat the bound request digest and declare a
decoder instance, decoder family, source descriptor, and blinding attestation.
Development labels may fit one temperature per family. Test labels may be used
only after calibration is fixed, for held-out performance/reliability and
agreement.

Distinct instance, family, and source strings are auditable design metadata.
They do not prove statistically independent errors. Sources sharing training,
provider infrastructure, prompts, or adjudication should be described
accordingly. The two deterministic decoders in `native.py` are representation
projections and never count toward this requirement.

Gate 4 also requires an explicit responsible-researcher review that the
imported sources are genuinely distinct for the claimed scope. Passing the
metadata validator alone is necessary but not sufficient; shared provider,
training, prompting, or adjudication dependencies must be documented.

## Native action admission

`structured_profile_action_reference` and
`native_persona_action_reference` verify the terminal action contract, but
they are deterministic belief projections. They never count as native
end-to-end evidence. An eligible import must use
`adapter_kind = "native_end_to_end_recorded"`, bind every action to the exact
suite item hash/wording/question type, name the native state and system
version, and identify whether execution was recorded live or replayed from a
frozen provider/system trace.

The [Gate 4 live-collection guide](gate4-live-collection.md) documents the
implemented keyless planning and explicitly authorized OpenAI native-action
collector. Its physical-attempt budgets, durable journal, output lock, and
manual-review stops make collection resumable; they are infrastructure, not
empirical evidence. No provider-produced native action is checked in.

Import both evidence classes into a new review directory; never place them
inside the completed run:

```bash
PYTHONPATH=src python -m cape_loop gate-review import-native \
  runs/EXPERIMENT-B \
  decoder-requests.jsonl \
  artifacts/gate4-distinct-decoders/judgments.jsonl \
  decoder-truth.jsonl \
  artifacts/gate4-native-actions decoder-source-review.json \
  artifacts/GATE4-REVIEW \
  --external-collection-dir artifacts/gate4-distinct-decoders
PYTHONPATH=src python -m cape_loop gate-review verify \
  artifacts/GATE4-REVIEW
```

The request and truth files must exactly match the verified run's retained
packet. Official automated decoder evidence is accepted only as the complete
selected Anthropic/Gemini collection. The importer rebuilds its provider plan,
validates the physical-attempt journal, accepted provider audits, exact
judgments, portable execution manifest, and all five evidence-file digests
while holding both collector locks. The positional judgment file must be
byte-identical to that collection's `judgments.jsonl`.

Gate 4 accepts native actions only as the complete output directory
from the implemented official-origin OpenAI `gpt-5.6-sol`/medium collector.
The importer rebuilds its plan and requests from the verified run, validates
the transport-attempt journal and every accepted provider audit, matches
`native-actions.jsonl` to the embedded audited records, and validates the
execution manifest. All six collection-file digests are recorded in the
review. A standalone action JSONL—even if each row is self-hashed and conforms
to `native-terminal-action-record.schema.json`—is ineligible.

Reviewed human or other generic decoder sources remain available only through
the explicit `--allow-reviewed-generic-decoders` alternative. That path
retains source-review and coverage checks but records
`reviewed_generic_import`; it does not assert automated provider provenance.

`decoder-source-review.json` uses `decoder-source-review.schema.json`. It binds
the exact request/judgment file digests and records one responsible
researcher's eligibility decision, dependency notes for every source, and a
genuine-distinctness determination for every co-occurring source pair. The
importer validates the full development/test decoder analysis, but only
test-split judgments cover eligible Gate 4 trajectories.

The output contains `gate-review.json`, `manifest.json`, and `SHA256SUMS`.
Import fails if the destination already exists or is inside the source run.
The recomputed Gate 4 can meet computational checks, but both the artifact and
the gate retain `claim_status = "not_claimed"`. A metadata attestation is
auditable evidence, not proof that a source or action was honestly produced.

The review stores digests and validation results; it does not copy the
evidence. A selected-provider evidence bundle must retain the request,
truth-label, and source-review files, all five external-decoder collection
files, and all six native-action collection files named by those digests.
Keep truth labels researcher-only and access-controlled rather than placing
them in decoder-visible or public material. `gate-review verify`
verifies the review directory itself; full recomputation additionally needs
the verified source run and those separately retained exact input bytes.

## Human evidence admission

Before generating a live survey from the packet, the responsible investigators
must supply:

- an ethics/IRB approval, exemption, or documented determination;
- approved consent text and a stable consent version;
- recruitment, eligibility, compensation, and stopping rules;
- a survey platform and assignment protocol;
- a comprehension-check policy fixed before analysis; and
- de-identification, access, privacy, deletion, and retention procedures.

The importer validates declared metadata and codebook bindings. It cannot
verify that consent was informed, compensation was delivered, or the platform
implemented the approved protocol. Raw identifying data must not enter this
repository. Release only the minimum de-identified evidence allowed by the
determination and consent.

## Statistical admission

Experiment A's marginal OLS with user-clustered CR1 covariance is useful for
dependency-free auditing and smoke analysis. It is not interchangeable with
the proposal's user-random-slope/scenario-random-intercept generalized
mixed-effects model.

The optional [mixed-effects analysis](mixed-effects-analysis.md) implements
both proposal formulas in a version-pinned R environment. It verifies source
run checksums and canonical resolved-config digests, requires pooled repeats to
share the same source digest and design, refuses fixed-effect rank deficiency,
preserves the maximal random-effects structure, and emits separately
checksum-bound results. Experiment A needs at least two `prior_strength`
values; the current one-level pilots are `not_estimable`. Experiment B derives
one row per retained turn by rescoring each `belief_after` against top-level
`theta`, normalizes turns to `1, ..., T`, and verifies that the last result
equals the retained terminal error. Its repeats must have identical horizons;
different horizons are not pooled to manufacture `turn` variation. For a
confirmatory claim, execute that harness on preregistered verified runs and
retain:

- exact input-row and exclusion digests;
- formula, link, fixed effects, random effects, contrasts, and coding;
- software/package versions and numerical settings;
- raw and curvature-scaled gradients, convergence, singularity, and
  variance-component diagnostics;
- interval and multiplicity procedures; and
- a machine-readable result linked back to the source run.

If the mixed model is not estimable, report that fact and the predeclared
fallback. Do not silently promote the CR1 result to confirmatory status.
The A target-versus-aware ACUE contrasts assess whether the target's error
contrast is nonzero relative to the aware reference, not the direction of
target belief updating required for H1. Likewise, B's declared terminal-error
interaction alone does not establish all five self-confirmation clauses.
Reported confidence intervals are unadjusted pointwise intervals; Holm
adjustment applies to p-values only.

## Freeze and release

First verify the completed run, then freeze it:

```bash
PYTHONPATH=src python -m cape_loop verify runs/<run-id>
PYTHONPATH=src python -m cape_loop artifact freeze \
  runs/<run-id> artifacts/<artifact-id>.tar
PYTHONPATH=src python -m cape_loop artifact verify \
  artifacts/<artifact-id>.tar
```

The deterministic tar and sidecar fix file identity and normalized archive
metadata. The artifact README must still name:

- whether the run is smoke, pilot, or paper evidence;
- every external input and its admissibility record;
- missing cells, exclusions, and failed attempts;
- the exact claims, tables, or figures the archive supports; and
- authorship, repository URL, DOI/venue state, and release date.

At present the repository includes no live API results, genuinely distinct
decoder judgment corpus, human participant dataset, full mixed-effects result,
paper-frozen results, named paper authors, or DOI.
