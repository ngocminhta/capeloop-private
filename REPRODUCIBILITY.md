# Reproducing CAPE-Loop runs

This document is the release-level reproducibility checklist. The conceptual
reasoning behind each requirement is described in
[the detailed reproducibility guide](docs/reproducibility.md).

## Supported environment

The core implementation targets Python 3.11 or newer and uses only the Python
standard library. Run commands from the repository root with:

```bash
PYTHONPATH=src python -m cape_loop ...
```

First record the local environment and run the offline checks:

```bash
python --version
PYTHONPATH=src python -m cape_loop doctor
PYTHONPATH=src python -m unittest discover -s tests
```

No core test or smoke run should require network access, an API key, or a
provider SDK.

## Reproduce a checked-in configuration

1. Start from a clean source checkout or record all local modifications.
2. Select the exact TOML configuration named by the artifact or paper manifest.
3. Validate it before execution:

   ```bash
   PYTHONPATH=src python -m cape_loop config validate configs/smoke.toml
   ```

4. Run it without modifying the configuration:

   ```bash
   PYTHONPATH=src python -m cape_loop run configs/smoke.toml
   ```

5. Verify the generated manifest and checksums:

   ```bash
   PYTHONPATH=src python -m cape_loop verify runs/<run-id>
   ```

6. Compare the resolved configuration, split manifest, code identity, and
   environment record before comparing metrics.

The configuration path above is a smoke example. Paper artifacts must name a
frozen configuration and must not silently inherit local defaults.

## Required contents of a reproducible run

A releaseable run must retain:

- the original TOML configuration in the release bundle and the run's fully
  resolved JSON configuration;
- configuration schema version and digest;
- global seed and semantic random-key scheme;
- code version or source-tree digest;
- Python and operating-system information;
- population and terminal-battery identities plus the declared
  scenario/template split manifest, including paraphrase-suite and terminal-v2
  content digests when those paths are used;
- event-level causal-chain records;
- fitted-model and calibration parameters;
- metric definitions and metric records;
- external-model request and response identities, if used;
- decoder request/judgment hashes, source metadata, development-only
  calibration, and separately retained truth linkage, if used;
- human assignment/codebook identity, consent/blinding/comprehension versions,
  exclusions, and ethics determination, if human evidence is used;
- a gate report that states outcomes without rewriting the gate criteria;
- SHA-256 checksums for retained artifacts.

If any item cannot be released, the artifact README must say why and describe
how an authorized evaluator can regenerate or inspect it.

## Determinism and common random numbers

CAPE-Loop derives random values from semantic keys rather than a mutable global
sequence. A draw is identified by stable fields such as:

```text
(run seed, experiment, user, scenario, branch, turn, option, purpose)
```

This makes execution order irrelevant and permits counterfactual policy branches
to reuse randomness where their options overlap. Reproduction requires both the
seed and the semantic identifiers; recording only a final pseudorandom-generator
state is insufficient.

Bit-for-bit identity may still differ if external LLMs, manually collected human
judgments, or a changed Python implementation are involved. In those cases the
retained response records are the canonical replay input.

## External LLM runs

The provider-neutral JSONL replay remains the canonical reproducible input.
The repository also has an explicit-opt-in, standard-library Responses API
executor for static request files and adaptive experiment runs. Before any
paid call:

1. validate the config or request JSONL;
2. inspect the resolved model role, reasoning effort, maximum output tokens,
   request count, and total-token ceiling;
3. set the named API-key environment variable locally;
4. use the dry-run plan command for static requests; and
5. add `--execute-live` only after the model and hard budgets are approved.

For the checked primary/replication pair, plan both immutable configs together
without reading a key:

```bash
PYTHONPATH=src python -m cape_loop llm evaluation-suite \
  configs/openai_primary.toml configs/openai_replication.toml \
  --output-root runs
```

Add `--execute-live` to that same command only after reviewing its combined
index. The two roles retain separate run IDs, journals, and provider ledgers.
GPT-5.6 Terra is a model-variant/tier replication, not distinct-family
robustness.

Live adaptive runs journal a response before exposing it to the updater and
retain an attempt audit. The recovery journal is outside the run directory,
allowing `--resume-failed-live` to preserve the failed run and reuse already
hash-bound responses. A completed run copies only the responses/audits it used
and removes credential and external-journal paths from its provider manifest.

Whether imported or live, retain exact request IDs, prompt hashes, resolved
model identifiers, reasoning/decoding parameters, parsed output, usage
metadata, timestamps, and provider request IDs. A replay run fingerprints the
complete response corpus in `llm/input-manifest.json`; verified reuse requires
that fingerprint to match.

When `llm.calibration = "temperature"`, collect the declared development-user
responses first, fit one temperature per updater view, and lock it before test
execution. Retain `models/llm-calibration.json`, development raw responses and
metrics, test raw responses, and active calibrated responses. Confirm that the
artifact says `fitted_split = "development"` and `test_labels_used = false`.
`llm.calibration = "none"` is an explicit ablation and must be labeled in any
comparison.

Do not record API keys, authorization headers, or unrelated provider account
metadata. A model alias that silently changes over time is not a sufficient
identifier; record every version field the provider exposes. The executor
accepts the requested label or its dated snapshot only. A missing or
inconsistent returned label is retained as a rejected audit and never exposed
as replay input.

No API key, live response corpus, or paid model result is checked in. Provider
code and mocked tests establish execution contracts only.

## External decoder evidence

Experiment B may emit blinded development/test decoder requests alongside two
files that must remain researcher-only:

```text
decoder/external-requests.jsonl
decoder/truth-labels.researcher-only.jsonl
decoder/researcher-codebook.jsonl
```

Send only the request material to a decoder. Imported judgments must repeat the
request hash and record decoder instance, family, source descriptor, and
blinding attestation. The source audit checks complete per-request coverage and
distinct metadata. It does not prove statistically independent errors.

Fit decoder temperature calibration on development labels only, then lock it
before reporting raw/calibrated test Brier, NLL, accuracy, ECE/reliability, and
cross-source agreement. The two deterministic repository decoders are
representation checks and cannot satisfy the external-source requirement.

## Human evidence

The human-study CLI generates blinded assignments and can validate/analyze
de-identified response JSONL. Before collection, freeze:

- the institutional ethics/IRB approval or exemption determination;
- approved consent language and version;
- recruitment, eligibility, compensation, and stopping rules;
- assignment/blinding/comprehension protocol versions;
- privacy, access, and retention policy; and
- the survey implementation and researcher codebook.

Analysis excludes participants without declared consent or a passing
comprehension check and retains those counts. Software validation of
`consented = true` is not proof that valid consent occurred. The repository
does not recruit, host, compensate, or collect participants.

## Calibration and split integrity

Calibration parameters and fitted response models may use only the declared
training and development records. Test users and terminal-diagnostic labels must
remain inaccessible. A release must retain the split manifest and fitting
records or deterministic generation identity so leakage can be checked.

The runner enforces group-disjoint latent users. Experiment A's rendered
paraphrases additionally enforce family-disjoint train/development/test
templates and content-bind test cases to their source trials. Experiment B's
terminal-v2 suite rejects overlap in option ID, feature vector, wording ID, and
scenario family. Retaining these checks establishes instrument separation; it
does not by itself establish a successful transfer result.

Raw and calibrated results are distinct outputs. Never replace raw model
probabilities in place.

The Experiment A dependency-free regression uses a marginal OLS fit with
user-clustered CR1 covariance. Reproducing it does not reproduce the proposal's
user-random-slope/scenario-random-intercept generalized mixed-effects model.
Any paper confirmatory analysis must retain the external model formula,
software/version, optimizer, convergence diagnostics, variance components,
contrast definitions, multiplicity family, and exact input row digest.

## Paper artifacts

An artifact supporting a table, figure, or statement must include:

- the table/figure identifier;
- its source run identifiers;
- aggregation command or deterministic report invocation;
- exclusions and stage-gate dependencies;
- checksums;
- a clear distinction between smoke outputs and paper evidence.

Freeze a checksum-valid run into a deterministic archive with:

```bash
PYTHONPATH=src python -m cape_loop artifact freeze \
  runs/<run-id> artifacts/<artifact-id>.tar
PYTHONPATH=src python -m cape_loop artifact verify \
  artifacts/<artifact-id>.tar
```

The freeze command also requires either the byte-for-byte
`config.source.toml` or a manifest entry that identifies and hash-binds a
programmatically constructed config. A resolved config without traceable origin
is not paper-freezable.

The adjacent `.manifest.json` binds the archive digest, source run ID, source
manifest/checksum digests, and member count. Normalized tar metadata makes the
same verified source run byte-reproducible. Freezing authenticates artifact
identity; it does not establish scientific adequacy.

The repository must not present generated placeholder values as results. See
[artifacts/README.md](artifacts/README.md) and
[paper/README.md](paper/README.md).

## Reproducibility audit

Before a release:

```bash
make check
PYTHONPATH=src python -m cape_loop run configs/smoke.toml \
  --output-root /tmp/cape-loop-release-check
PYTHONPATH=src python -m cape_loop verify /tmp/cape-loop-release-check/<run-id>
```

Then manually confirm:

- latent truth was visible only to simulators and evaluators;
- interaction context and policy provenance remained separate;
- intrinsic welfare excluded presentation effects;
- fixed-history inputs were identical across updaters;
- terminal diagnostics were exogenous;
- calibration did not use test labels;
- held-out paraphrase and terminal-v2 overlap checks passed;
- decoder truth/codebook files were never exposed to decoder sources;
- decoder calibration used development labels only and source diversity was not
  described as proof of independence;
- human evidence had an external ethics determination and approved collection
  protocol;
- every claimed self-confirming case passed all five defining conditions;
- CR1 marginal OLS was not mislabeled as the confirmatory mixed-effects model;
- no headline was asserted merely because a software gate existed.

Report discrepancies in the artifact documentation. Do not edit retained output
until it matches an expected claim.

See [External evidence boundaries](docs/external-evidence.md) for the
admission checklist separating executable protocol support from evidence that
must be collected or fitted outside the repository.
