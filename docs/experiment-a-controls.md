# Experiment A control execution

Experiment A has three positive controls and three negative controls:

| Control | Polarity | Executed construction | Expected target diagnostic |
| --- | --- | --- | --- |
| Volunteered preference | Positive | One user-originated, option-free preference statement | Probability mass moves toward the stated direction |
| Repeated balanced choices | Positive | Three independently worded, scenario-disjoint balanced choices with the same selected direction | Evidence accumulates toward the repeated direction |
| Direct correction | Positive | One explicit correction applied after the same fixed false-profile seed | Target mass moves toward the correction |
| Indifference | Negative | One explicit indifference response to an opposing pair | No target-direction update |
| Registered random choices | Negative | Three choices from a precommitted SHA-256 randomization device independent of the user and presentation | No target-direction update |
| Target-nondistinguishing choice | Negative | One choice between options with identical target features and different non-target features | No target-direction update |

The fixed high-level descriptions still live in
`experiment-a-controls-v1`. The executable layer is
`experiment-a-control-execution-v1`, implemented in
`src/cape_loop/control_study.py`. It does not coerce these signals into the
main one-step `Observation` choice schema. Direct statements, indifference,
and randomized responses keep their own typed event records.

## Components

`ExperimentAControlPlan` is the root protocol artifact. It binds:

- the original control-battery ID and SHA-256 digest;
- six ordered concrete stimuli;
- test-split status;
- target attribute and direction;
- the control-specific prior;
- every surface response, option, selected option, scenario, wording template,
  and response source;
- the randomization registration; and
- positive and negative diagnostic tolerances.

Each `ExperimentAControlStimulus` and the complete plan have independent
content digests. Changing a response, threshold, option, prior, event order, or
battery case invalidates the old digest.

`ControlRandomizationRegistration` fixes
`sha256-modulo-option-count-v1`, public seed material, and three draws before
the choices are constructed. A draw hashes the registration digest, event ID,
and option count. The retained draw digest allows an auditor to reproduce each
selected option. The generator never reads latent preference, context,
provider output, or an API key.

`ControlExecutionReport` retains exact control coverage and one
`ControlExecutionOutcome` per required stimulus. Every outcome binds the plan,
stimulus, executor, prior, posterior, directional mass change, criterion, and
evidence source. Provider outcomes additionally bind request ID, prompt digest,
model ID, and response digest.

Every plan, report, and outcome retains:

```text
claim_status = "not_claimed"
```

A complete or passing control report is not automatically a paper result.

## Diagnostic executors

Run the checked implementation from Python:

```python
from cape_loop.control_study import (
    build_experiment_a_control_plan,
    run_diagnostic_control_executions,
)

plan = build_experiment_a_control_plan()
reference, no_update = run_diagnostic_control_executions(plan)
```

The reference is intentionally transparent:

- a declared sign-likelihood update handles the volunteered statement;
- three sequential declared sign-likelihood updates handle repeated balanced
  choices;
- `ReferenceLogOddsCorrectionAdapter` and the versioned correction-debt
  protocol handle direct correction; and
- a target no-update transition handles all three negative controls.

The returned `reference_binding` records the selected reference family and its
parameters. Direct correction also records the correction-debt adapter ID,
protocol version and digest, before/after state digests, and before/after wrong
mass.

This reference is a software and protocol diagnostic. It is labeled
`diagnostic_reference`, has `is_live_evidence = false`, and must not be counted
as an evaluated system or external model.

The second executor is an identity no-update baseline. It should satisfy the
three negative controls and, by construction, should not respond to the three
positive controls. This behavior verifies that the positive controls can
separate a responsive reference from a nonresponsive baseline. The baseline is
labeled `diagnostic_baseline`, not empirical evidence.

## Provider-neutral LLM exchange

Build a six-control provenance-aware packet:

```python
from cape_loop.control_study import (
    build_control_llm_exchange,
    build_experiment_a_control_plan,
    write_control_provider_requests,
    write_control_request_bindings,
)

plan = build_experiment_a_control_plan()
exchange = build_control_llm_exchange(
    plan,
    updater_id="llm_control_provenance_aware",
    view="provenance_aware",
)
write_control_provider_requests(
    "control-provider-requests.jsonl",
    exchange,
)
write_control_request_bindings(
    "control-request-bindings.jsonl",
    exchange,
)
```

`control-provider-requests.jsonl` uses the repository's ordinary
`LLMRequest` schema. It can therefore be passed to the existing offline replay,
OpenAI, or OpenRouter execution path. `control-request-bindings.jsonl` retains
the outer control ID, plan digest, battery digest, stimulus digest, and binding
digest. Those audit labels are deliberately absent from the model-visible
payload: the model sees the actual response, context, and permitted provenance,
not `positive`, `negative`, an expected outcome, or the control ID.

The exchange only emits requests whose declared information view can preserve
the intended control:

| Information view | Emitted controls | Omitted controls and reason |
| --- | ---: | --- |
| `response_only` | 3 | Repeated balanced and nondistinguishing signals need their visible contexts; random choice needs its registered provenance |
| `full_context` | 5 | Random choice needs its registered generation provenance |
| `provenance_aware` | 6 | None |

An omitted control is recorded with a reason. It is never silently converted
into a semantically different request.

Existing provider commands can execute the generic request file. Live
execution remains explicitly opt-in and subject to the provider's request,
retry, and token budgets. This control module does not load `.env`, read an API
key, or bypass those safeguards.

After responses are available, score them through the same bindings:

```bash
PYTHONPATH=src python -m cape_loop control-study analyze \
  RUN/llm/experiment-a-control-request-bindings.jsonl \
  control-provider-responses.jsonl \
  control-provider-analysis.json \
  --source-descriptor "reviewed provider collection artifact"
```

The command reconstructs the fixed plan, infers the single declared updater
and information view from the bindings, and requires the regenerated bindings
to equal the supplied JSONL exactly. It then validates exact response coverage
and writes the analysis atomically. Existing outputs are refused. The output
envelope records SHA-256 digests of both input files, the plan and exchange
digests, the scored report, and `claim_status = "not_claimed"`.
Bindings and responses are parsed from the same immutable byte snapshots used
for those SHA-256 digests. Symlink or non-file inputs are rejected, and both
named paths are checked again immediately before publication.

The equivalent Python API is:

```python
from cape_loop.control_study import execute_control_llm_exchange
from cape_loop.llm_exchange import ReplayProvider, read_responses

responses = read_responses("control-provider-responses.jsonl")
report = execute_control_llm_exchange(
    plan,
    exchange,
    ReplayProvider(responses),
    execution_mode="provider_replay",
    source_descriptor="reviewed provider collection artifact",
)
```

Coverage is exact: duplicate, missing, or unexpected request IDs fail before
scoring, as does a prompt-digest mismatch. A `ReplayProvider` cannot be labeled
`provider_live`. A genuinely live provider call must be supplied by the caller
and labeled `provider_live`; the outcome then sets `is_live_evidence = true`.
Both modes still retain `claim_status = "not_claimed"`.

## Evidence interpretation

The four evidence labels are intentionally non-interchangeable:

| Execution mode | Evidence class | Live evidence |
| --- | --- | --- |
| `deterministic_reference` | `diagnostic_reference` | No |
| `deterministic_no_update_baseline` | `diagnostic_baseline` | No |
| `provider_replay` | `external_model_response` | No live call in this execution |
| `provider_live` | `external_model_response` | Yes |

Reference and baseline values verify construction and scoring. They cannot fill
a missing model response. Replayed responses remain distinguishable from calls
made during the current execution. Neither a criterion nor complete coverage
promotes any of these artifacts into a scientific claim.

## Relationship to H7 volunteered valid learning

The fixed six-control study proves that an explicit volunteered preference is
representable and scoreable, but its single protocol case is not substituted
for H7's user-clustered positive-control estimand. H7 uses a separate exhaustive
collection generated from every retained Experiment A test user, domain, and
attribute. That path requires paired `llm_full_context` and
`llm_provenance_aware` provider outcomes, converts them to
`VolunteeredPreferenceUpdate` records, and recomputes H7 in a derived artifact.

Use `control-study h7-plan`, `h7-review`, and `h7-verify` as documented in
[H7 volunteered-preference controls](h7-volunteered-controls.md). Neither
workflow imputes an absent control from anchor-choice rows.

## Test coverage

`tests/test_control_study.py` checks:

- deterministic plan and random-choice reproduction;
- all six construction invariants;
- content-hash tamper rejection;
- complete reference and baseline coverage;
- correction-debt adapter/protocol bindings;
- expected positive and negative reference behavior;
- view-specific semantic omissions;
- absence of audit-label leakage into model prompts;
- exact provider response and prompt binding;
- replay-versus-live evidence labeling; and
- round-trippable binding and generic provider JSONL.
