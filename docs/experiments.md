# Experiments

The repository has four executable run kinds:

```text
provenance_audit
closed_loop
evaluation_validity
sensitivity
```

It also provides standalone LLM/decoder exchange, human packet/analysis,
correction-debt diagnostic, and artifact-freezing commands. This document
describes what the current runners execute and retain. It reports no
experimental outcome.

The stable object of study is the **updater–logging-policy pair**. Experiment A
tests whether an updater assigns warranted weight to evidence generated under
different elicitation mechanisms. Experiment B tests how an updater and an
adaptive interaction policy jointly determine the evidence that is collected.
The pair-level construct is **policy-conditioned evidential legibility**: a
history can remain informative to exact inference while a particular writer
translates it inaccurately. This is operationalized by policy-specific
same-history attribution gaps, not by a composite policy score.
Experiment C tests whether conclusions about an updater change with the policy
that produced its evaluation history. Strict five-clause self-confirmation is a
strong conditional downstream outcome, not the premise on which the other
experiments depend.

## Shared preparation

Experiments A–C call the same preparation path before their main runner:

1. load and checksum-validate the configured scenario catalog, when enabled;
2. load the travel and/or writing domain;
3. build a deterministic split manifest;
4. generate disjoint train, development, and test latent-user groups;
5. generate randomized training interactions across the five provenance
   mechanisms;
6. fit four-parameter aware and unaware likelihood models on the same training
   interactions;
7. generate a distinct development interaction set;
8. optionally fit separate aware and unaware temperatures on development
   outcomes; and
9. when `[scenarios] conversation_file` is configured, render each
   mathematically fixed choice through the frozen per-scenario conversation
   bank; and
10. retain the split, population, raw model, active model, calibration, and
   development diagnostic records.

The active fitted bundle is the raw bundle when calibration is `none`; otherwise
its coefficient vectors are divided by the fitted temperatures. Both versions
are retained. The aware response diagnostic is over displayed option identity,
whereas the unaware diagnostic is over signed semantic direction. Those NLLs
have different outcome spaces and are not directly comparable.

When `llm_*` updaters are configured, a second calibration boundary applies:
the runner first collects/imports naturally sampled responses for the declared
number of development users, fits one temperature per updater view, and locks
those transformations before the A/B/C test run. Development raw responses,
calibration metrics, test raw responses, and active calibrated responses are
retained separately. `llm.calibration = "none"` is a deliberate uncalibrated
ablation. No empirical calibration exists until a replay corpus or explicitly
authorized live responses are available.

Exact action-aware updater state is a full joint over preference and
susceptibility. Its public preference belief is the theta marginal. Runners
retain the joint whenever an exact evaluated updater or exact shadow is present.
`models/exact-action-aware-reference.json` records the declared response
coefficients, full uniform susceptibility support and weights, and supplied
preference-prior boundary used by the exact reference. Split membership is not
leaked into that prior.

The split manifest assigns identifiers for preference groups, susceptibility
groups, option templates, dialogue templates, scenario families, and
paraphrase templates. User generation and the interaction generators consume
those assignments: atlas, beacon, and cedar surface families are respectively
reserved for training, development, and test while retaining matched intrinsic
features. Every run writes a concrete overlap/binding audit. Terminal template
identifiers are reserved for test. Experiment A materializes a
separate content-addressed paraphrase suite, rejects family overlap, renders
test-only response surfaces from retained source trials, and binds each result
to the source context and selected option. Experiment B independently checks
terminal-v2 option IDs, feature vectors, wording IDs, and scenario families
against its training material.

Official v2 population construction balances each theta and susceptibility
coordinate within incomplete allocation blocks, then applies an outcome-blind
joint block search to reduce cross-coordinate contingency imbalance and linear
association at the declared sample horizons. This controls avoidable
population-composition dependence without changing arm assignment or claiming
that the finite synthetic population is independent.

### Data splits and leakage controls

A split is a collection of complete semantic groups, not a random row
partition. Training, development, and test are disjoint on all six declared
axes:

| Axis | Executable boundary |
| --- | --- |
| Latent preference | The complete three-attribute `theta` tuple belongs to one split |
| Susceptibility | The complete ranking/default/suggestion tuple belongs to one split |
| Option template | Feature-matched atlas, beacon, and cedar families belong to train, development, and test |
| Dialogue template | Each split has a distinct opaque visible wording-template ID |
| Scenario family | Each complete catalog family and its visible surfaces belong to one split |
| Paraphrase template | Each content-addressed family belongs to exactly one split |

Atlas, beacon, and cedar are intentionally opaque so that split labels do not
enter model-visible text. Their option features remain matched. The preparation
path fits response likelihoods on atlas interactions, fits temperatures and
computes held-out response diagnostics on beacon interactions, and reserves
cedar assets for A–C test evaluation. The terminal-v2 suite is an additional
test-only family with novel IDs, feature values, scenarios, and wording.

`build_default_paraphrase_suite()` supplies train/development/test language
families. Before Experiment A renders held-out test surfaces, the runner rejects
overlap with every train/development template ID, content hash, and literal
pattern.

Every A–C run retains:

```text
splits.json
metrics/split-leakage-audit.json
events/fitted-model-training.jsonl       # when events are retained
events/fitted-model-development.jsonl    # when events are retained
```

The audit names concrete asset counts, pairwise overlap sets, all
manifest-to-generator bindings, the paraphrase-suite digest, and its status.
Any overlap or inconsistent generator binding aborts the run. This proves the
software split; it cannot prove that researchers avoided looking at test
results while tuning prompts or thresholds, which remains a preregistration and
release-review obligation.

### Catalog-backed deterministic selection

Every checked-in configuration binds the canonical 1.5.0 catalog and its exact
SHA-256. Selection first filters by domain, split, and target attribute. For a
longitudinal history, it derives one semantic-keyed permutation for that
trajectory and cell, then consumes the permutation without replacement until
the cell is exhausted. A later cycle repeats only when the planned horizon
exceeds available scenarios. Single matched cases use the same deterministic
selection contract. Neither path inspects latent preference, current profile,
updater output, response, model performance, or sensitivity-grid coordinates.
The catalog replaces generic surfaces inside existing cells; it is not an
additional Cartesian factor and does not by itself increase the model-call
budget.

The six test scenarios per cell cover a 16-turn trajectory for the declared
three-attribute cyclic policies. `exploratory` v2 is adaptive only within a
block-balance constraint: it chooses among the least-exposed attributes using
current marginal entropy, keeps target counts within one at every prefix, and
therefore visits all three attributes once in each three-turn block. It has the
same \(\lceil T/3\rceil\) per-cell no-repeat requirement. A custom or unknown
adaptive policy without this constraint is still audited conservatively at
\(T\) scenarios per eligible cell. When a declared design exceeds its pool,
the runner cycles deterministically and reports the reuse rather than
pretending it had no repeat.

- **Training/development:** the existing example schedule selects
  deterministically from its own atlas/beacon pool before fitting or
  calibration.
- **Experiment A:** selection cycles over test-user index within each
  domain-by-attribute cell. The two anchor directions for one user and target
  reuse the same scenario, and the anchor appears once in each physical
  display position across that pair. All mechanisms, response modes, prior
  strengths, and updaters retain that scenario/order assignment. Across
  users, scenario counts within each anchor direction differ by at most one.
- **Experiment B:** the without-replacement order is keyed by the common
  trajectory-pair key. Policy/updater twins therefore retain the same scenario
  whenever they have reached the same occurrence of an attribute, while
  presentation and endogenous response can still differ.
- **Experiment C:** development and test use their distinct catalog pools.
  Every updater replays each fixed history verbatim; fixed logger twins and
  same-target endogenous updater branches use paired scenario schedules.
- **Sensitivity:** every grid point reuses the same frozen catalog and semantic
  schedule. Parameters may change choices, but a grid coordinate cannot select
  a different scenario.

Every catalog-backed run retains the consumed bytes, input manifest, and
availability and realized-consumption reports described in
[Data model](data-model.md#scenario-catalog-input). The held-out terminal-v2
batteries used by Experiments B and C remain separate generated evaluation
instruments and are not selected from the interaction catalog.

### Hybrid conversation protocol

The scenario catalog defines the controlled meanings; the companion
`[scenarios] conversation_file` defines their frozen natural-language
realizations. The mathematical random-utility or rule-based model always
selects the option first. Runtime then renders one assistant turn that presents
the visible options, followed by the fixed user sentence
`I choose {selected_name}.`

For example, a restricted lodging surface is:

> **Assistant:** Here are two lower-cost hotel options for your trip. Hotel A
> is a standard room in a mixed-use neighborhood. Hotel B is a standard room
> in a quiet outer neighborhood. Which would you like?
>
> **User:** I choose Hotel A.

The template-authoring model neither sees the latent user nor chooses Hotel A.
It may draft a candidate neutral base presentation and display names. The
current visible bases were subsequently project-standardized outcome-blind
onto three source-neutral frames, balanced across test cells, and remain
unreviewed. Balanced, restricted, and ranking share the selected base wording.
Code adds only the fixed default or suggestion sentence when that treatment is
present and fixes the user reply. The resulting templates are reviewed as
scenario inputs and reused across trials; runtime introduces no
authoring-model call per event.

Full-context and provenance-aware evaluated LLMs receive the rendered dialogue,
readable option descriptions, and a semantic attribute codebook. They do not
receive numeric feature vectors, the internal target-attribute index, or
catalog option IDs. Structured options use position aliases such as
`presented_option_1`; visible A/B names are also assigned by displayed position
after ranking, so neither identifier permanently encodes a preference
direction.
Response-only receives the local reply and selected readable option as the
deliberately information-poor ablation. The evaluated writer remains the model
configured in `[llm]`; it is separate from the conversation author.

Every run writes a normalized trace with associated metrics under
`conversations/`. Configured hybrid runs contain the exact rendered exchanges;
legacy/programmatic runs explicitly mark the natural surface unavailable. The
JSONL is exhaustive and deduplicated by the experiment's logical conversation
unit. The matching Markdown is a deterministic, diverse preview of at most 100
trace records by default; it reports the complete record, turn, and
outcome/evaluation counts and provides readable metric labels and
interpretation guidance. The preview is for reading, not a substitute for the
complete JSONL or canonical metric artifacts. Each run summary names both paths
and records the complete conversation, turn, and outcome counts plus the
displayed preview count; the exact summary keys are listed in
[Data model](data-model.md#conversation-logs-and-readable-previews).

Before spending on a live run, generate the prospective scenario packet:

```bash
PYTHONPATH=src python -m cape_loop scenarios audit \
  configs/live/experiment_b_openrouter.toml \
  artifacts/scenario-audit-b6 \
  --split test --turns 6
```

This command makes no provider call and consumes no experiment outcome. Use
`scenario-review.md` as the metadata-visible researcher workbook,
`scenario-surface-review-blinded.md` for independent surface ratings, and
`scenario-audit.json` for capacity, counterbalancing, overlap candidates,
surface-hygiene flags, and complete finite-support simulator probabilities. A
successful command is not scientific approval: readiness is reported
separately for engineering, scientific-pilot, and paper use, and catalog status
strings alone never count as verified human evidence.

## Experiment A: causal-provenance calibration

Use:

```bash
PYTHONPATH=src python -m cape_loop run configs/smoke.toml
```

### Executed design

For every selected test user, domain, target attribute, anchor direction,
prior-strength stratum, mechanism, response mode, and updater, the runner
constructs a matched anchor set from one deterministically selected cedar
scenario.

Scenario and physical order are paired before any response is sampled. For
each user–domain–target cell, the negative- and positive-anchor cases use the
same scenario; their display orders are opposite, so the anchor is first once
and second once. The same assignment is then reused across mechanisms,
response modes, prior strengths, and updaters. This prevents scenario identity
or a one-sided position imbalance from being mistaken for an
anchor-direction effect.

`experiment.prior_strengths` crosses a declared concentration factor. At level
`s`, the user-specific prior is `(1-s)` uniform joint mass plus `s`
truth-aligned mass. The latent population balances the aligned values, and all
updaters/mechanisms in a matched stratum receive the identical prior. Context
and natural-response semantic noise keys are shared across prior strata, so
only prior concentration changes. The default `[0.0]` is the uniform pilot;
`[0.0, 0.35, 0.70]` is a documented multi-level example.

Supported mechanisms are:

| ID | Context change |
| --- | --- |
| `balanced` | Anchor and opposite-direction option; no default or suggestion |
| `restricted` | Anchor and same-direction option |
| `ranking` | The balanced pair with its displayed order reversed |
| `default` | Balanced pair with the anchor preselected |
| `suggested` | Balanced pair with the anchor recommended |

The anchor ID and features are invariant. A matched set is excluded when the
declared simulator gives the held anchor less than
`response_model.minimum_matched_probability` in any requested mechanism.
Exclusions are retained when event retention is enabled.

`controlled_anchor` is the primary **same-response provenance track**. It
supplies the same anchor observation in every mechanism while the choice set,
display order, default, or suggestion changes. With the hybrid surface bank,
the anchor keeps the same stable display name and exact local user sentence
across mechanisms. The assistant base wording is shared. Restricted changes
the option pair, ranking reverses its order, and only default or suggestion
inserts a fixed treatment sentence.

Before analysis, `metrics/experiment-a-same-response-audit.json` verifies
complete mechanism coverage, the identical selected anchor and local reply,
the invariant prior, and the invariant anchor identity for every matched
updater cell. A failed audit aborts the run. This track identifies how much
evidential weight the updater assigns to provenance; it is not an average
treatment effect on user choice.

`naturally_sampled` samples separately from the declared response distribution
for each context and is retained as a secondary A robustness track. The primary
natural-response feedback-loop test is Experiment B, where policy-dependent
actions can change later responses and histories. The three public live A
presets intentionally request only `controlled_anchor`; the offline A preset
retains both modes.

The configured policy must be exactly `balanced`; mechanism variation is
constructed by the elicitation layer rather than a policy trajectory.
Experiment A is strictly one-step:

```text
trajectories_per_cell = 1
turns = 1
bootstrap_replicates = 2000  # paper-grade example
```

The resampling count may be any non-negative integer. Zero is reserved for
smoke runs and selects a recorded 200-replicate fallback; confirmatory
configurations should set the count explicitly.

### Metrics and artifacts

The runner writes:

```text
analysis/experiment-a-rows.jsonl            # one compact updater×trial row
analysis/experiment-a-exclusions.jsonl      # always-retained exclusions
conversations/experiment-a.jsonl            # exhaustive, one record per trial
conversations/experiment-a.md               # diverse human preview, at most 100
events/experiment-a.jsonl                  # when retain_events = true
events/experiment-a-exact-references.jsonl # when retain_events = true
events/experiment-a-exclusions.jsonl       # when retain_events = true
events/experiment-a-held-out-paraphrases.jsonl # when retain_events = true
models/experiment-a-control-battery.json
models/experiment-a-control-plan.json
models/held-out-paraphrase-suite.json
llm/experiment-a-control-exchange.json
llm/experiment-a-control-request-bindings.jsonl
llm/experiment-a-control-requests.jsonl
metrics/experiment-a.jsonl
metrics/experiment-a-confirmatory.json
metrics/experiment-a-control-reference.json
metrics/experiment-a-control-baseline.json
metrics/experiment-a-same-response-audit.json
metrics/experiment-a-exact-calibration.json
metrics/experiment-a-hypothesis-estimands.json
metrics/experiment-a-oracle-slopes.jsonl
metrics/experiment-a-exact-oracle-slopes.jsonl
metrics/experiment-a-evidence-strength.json
metrics/experiment-a-raw-calibrated-scores.jsonl
metrics/experiment-a-reliability.jsonl
metrics/experiment-a-multiplicity.json
metrics/experiment-a-power.json
metrics/experiment-a-held-out-paraphrase-scores.jsonl
metrics/experiment-a-held-out-paraphrase-transfer.json
tables/experiment-a-brier.csv
tables/experiment-a-raw-calibrated-scores.csv
tables/experiment-a-reliability.csv
figures/experiment-a-update-magnitude.svg  # when controlled rows exist
metrics/gate-report.json
metrics/summary.json
```

The compact A file uses row schema v2. It includes whichever response modes the
configuration executes and exposes identifiers, mechanism, prior strength,
`analysis_track`, `reference_basis = "exact_action_aware"`, exact and fitted
update errors, system/exact/fitted log-odds updates, and
`calibration_residual = system_log_odds_update - exact_log_odds_update`.
The full event and exact-reference files remain the source for reconstructing
complete posteriors and causal chains. A compact row is a projection of an
evaluated trial, not an additional trial.
`analysis/experiment-a-exclusions.jsonl` mirrors the versioned exclusion rows
independently of raw event retention so confirmatory admission never silently
drops an excluded matched set.

The A conversation trace stores the assistant/user exchange once per
`trial_id` and groups the updater evaluations beneath that record. This avoids
copying identical dialogue once for every updater while retaining the source
record linkage and the A metrics used in analysis.

Metric rows include marginal Brier score, fitted-aware reference Brier,
excess Brier, exact- and fitted-reference action-conditioned update errors,
exact and fitted log-odds updates, marginal KL, update-direction accuracy,
update magnitude, and evidence weight. For controlled rows, the exact
action-aware posterior is primary: the selected anchor is fixed by the
same-response design, and the oracle evaluates its warranted likelihood under
the declared simulator response model while integrating uniformly over the
susceptibility support prospectively assigned to the test split. It never uses
the realized user's susceptibility or empirical test frequencies. The runner
records the support and every prior weight in
`models/exact-action-aware-reference.json`. The fitted action-aware reference
is retained as a secondary learnability and model-misspecification robustness
analysis.

Experiment A's crossed truth-aligned prior is an independent joint reconstructed
from the same attribute marginals included in the LLM request. The exact
reference therefore receives no hidden cross-attribute prior information. The
primary A system posterior is also always the raw LLM vector. Development-fitted
temperature scaling is retained only in secondary forecast-calibration outputs,
so an unchanged raw prior cannot become an apparent primary update.

To avoid repeating the exact reference once per updater,
`experiment-a.jsonl` retains each updater row's theta belief projections and an
`exact_reference_id`, while `experiment-a-exact-references.jsonl` stores one
full exact theta and theta×susceptibility reference per trial. Join those files
on `exact_reference_id`. The in-memory experiment row also exposes
`posterior_theta_psi` for an exact evaluated updater.

The confirmatory bundle adds:

- primary pooled and mechanism-specific calibration curves of system log-odds
  update against the warranted exact action-aware update, each with
  user-clustered bootstrap intervals; ideal calibration has intercept `0`,
  slope `1`, and residual RMSE `0`;
- primary target-writer signed calibration-residual contrasts comparing each
  non-balanced mechanism with balanced presentation of the same response;
- secondary target-writer ExactACUE magnitude contrasts on those same matched
  responses;
- separately labeled fitted-aware slopes as secondary learnability and
  misspecification robustness;
- a data-derived fitted evidence-strength ordering across mechanisms;
- raw-primary versus temperature-scaled-secondary forecast scores and
  one-vs-rest marginal-class reliability bins;
- a marginal OLS model with user-clustered CR1 covariance;
- Holm correction over its estimable non-intercept coefficient family; and
- paired user-cluster pilot-power simulation when enough complete differences
  exist.

The separate hypothesis-estimand artifact retains the earlier directional H1
over-update contrast, H2 distance-to-unaware contrast, and H7
mitigation/valid-learning component for diagnostic continuity. H1/H2 no longer
control the primary claim: they test extreme directional failure patterns and
must not be interpreted as a universal provenance-blindness test. The primary
claim is model- and mechanism-specific causal-provenance **miscalibration**
against the exact oracle. The formulas and incomplete-data rules are in
[Metrics](metrics.md).

The dependency-free CR1 regression is an auditable marginal robustness
analysis. It is **not** the proposal's confirmatory generalized mixed-effects
model with user random slopes and scenario random intercepts. The optional
[R mixed-effects harness](../analysis/confirmatory-mixed-effects/README.md)
implements that exact model, planned contrasts, source-digest checks, and
convergence/singularity diagnostics. Executing it on verified paper runs and
reviewing any inferential claim remain a separate statistical stage. A
configured bootstrap count of zero uses 200 replicates as an explicitly
recorded smoke fallback, not a paper default.

### Bounded live Experiment A presets

Each public live A preset crosses four users, two domains, three attributes, two
anchor directions, two prior strengths, five mechanisms, and one
`llm_full_context` updater in `controlled_anchor` mode:

```text
480 controlled experiment updates
+ 60 five-mechanism development-calibration updates
+ 40 controlled held-out-paraphrase updates
= 580 logical requests and, with zero retries, 580 physical attempts
```

At 2,048 maximum output tokens per attempt, the preflight allocation is
1,187,840 output tokens. These are bounded estimability pilots, not a powered
sample or paper evidence.

### Six-control execution protocol

Human-derived evidence strength is never fabricated. The fitted ordering
artifact records volunteered strength as unavailable until eligible external
judgments are imported. Independently, `experiment-a-controls-v1` fixes three
positive and three negative controls:

| Control | Polarity | Executed construction | Expected diagnostic |
| --- | --- | --- | --- |
| Volunteered preference | Positive | One user-originated, option-free preference statement | Mass moves toward the stated direction |
| Repeated balanced choices | Positive | Three independently worded, scenario-disjoint balanced choices in the same direction | Evidence accumulates toward that direction |
| Direct correction | Positive | One explicit correction after the same fixed false-profile seed | Target mass moves toward the correction |
| Indifference | Negative | One explicit indifference response to an opposing pair | No target-direction update |
| Registered random choices | Negative | Three choices from a precommitted SHA-256 device independent of user and presentation | No target-direction update |
| Target-nondistinguishing choice | Negative | Options share the target feature and differ only elsewhere | No target-direction update |

The executable layer, `experiment-a-control-execution-v1`, uses typed statement,
indifference, randomization, and correction events. It never coerces them into
the one-step choice `Observation` schema or relabels anchor choices as controls.

`ExperimentAControlPlan` content-binds the high-level battery, six ordered
stimuli, test-split status, target and direction, control prior, surfaces,
options, selections, scenarios, wording, response sources, randomization
registration, and tolerances. Every stimulus and the complete plan has an
independent digest. The registered random device uses
`sha256-modulo-option-count-v1`; each draw hashes the registration digest,
event ID, and option count and never reads latent preference, provider output,
or a credential.

Each outcome binds the plan, stimulus, executor, prior/posterior,
directional-mass change, criterion, and evidence source. Provider outcomes also
bind request, prompt, model, and response digests. All plans, reports, and
outcomes retain `claim_status = "not_claimed"`.

Two deterministic executors validate the protocol:

- the transparent reference applies declared sign-likelihood updates to
  volunteered and repeated-balanced evidence, a versioned log-odds correction
  adapter to direct correction, and no target update to negative controls; and
- the identity baseline leaves all six unchanged, so it should pass the
  negative but not the positive controls.

They are respectively labeled `diagnostic_reference` and
`diagnostic_baseline`, with `is_live_evidence = false`; neither can replace an
external model response.

The provider-neutral control exchange emits three response-only cases, five
full-context cases, or all six provenance-aware cases. A view omits any control
whose required context/provenance it cannot represent, with an explicit reason.
Model-visible payloads contain the actual interaction but not polarity,
expected outcome, or control ID.

After collecting responses through the ordinary replay/OpenAI/OpenRouter
exchange, analyze exact bindings with:

```bash
PYTHONPATH=src python -m cape_loop control-study analyze \
  RUN/llm/experiment-a-control-request-bindings.jsonl \
  control-provider-responses.jsonl \
  control-provider-analysis.json \
  --source-descriptor "reviewed provider collection artifact"
```

The analyzer reconstructs the fixed plan, requires exact request coverage and
prompt bindings, parses immutable regular-file snapshots, refuses an existing
output, and rechecks every input before atomic publication. Execution evidence
is kept in four non-interchangeable classes:

| Execution | Evidence class | Live evidence |
| --- | --- | --- |
| Transparent reference | `diagnostic_reference` | No |
| Identity baseline | `diagnostic_baseline` | No |
| Provider replay | `external_model_response` | No call in this execution |
| Live provider | `external_model_response` | Yes |

Provider setup, budgets, and collection commands are centralized in
[Live execution](live-execution.md).

### H7 volunteered-preference control

The six-control plan contains one volunteered case for protocol diagnosis. H7's
valid-learning estimand instead exhaustively generates one direct statement for
every retained test user, configured domain, and preference attribute, crossed
with exactly `llm_full_context` and `llm_provenance_aware`. The two views share
the uniform prior, statement, target, and context; only the provenance-aware
view receives:

```text
response_source = user
elicitation_provenance = user_originated_unprompted
```

The plan withholds the latent target direction from model-visible prompts and
binds every case to its source user, request, prompt, and plan digest. There is
no sampling or author-selected subset, and the complete user remains the
independent unit.

Build the immutable provider-neutral plan from a verified Experiment A run:

```bash
PYTHONPATH=src python -m cape_loop control-study h7-plan \
  runs/EXPERIMENT-A \
  artifacts/h7-volunteered-plan
```

The output contains:

```text
h7-volunteered-plan.json
h7-volunteered-request-bindings.jsonl
h7-volunteered-requests.jsonl
```

Collect one fixed provider/returned-model corpus for both updater views as
described in [Live execution](live-execution.md), then derive and reverify the
review:

```bash
PYTHONPATH=src python -m cape_loop control-study h7-review \
  runs/EXPERIMENT-A \
  artifacts/h7-volunteered-plan \
  artifacts/h7-responses.jsonl \
  artifacts/h7-provider-audit.jsonl \
  artifacts/h7-review.json

PYTHONPATH=src python -m cape_loop control-study h7-verify \
  runs/EXPERIMENT-A \
  artifacts/h7-volunteered-plan \
  artifacts/h7-responses.jsonl \
  artifacts/h7-provider-audit.jsonl \
  artifacts/h7-review.json
```

Review requires the checksum-valid source run; exact regeneration of the plan,
bindings, and requests; complete accepted response/audit coverage; and matching
request, prompt, request-body, raw-response, provider, and returned-model
identity. Every case must have both updater views. Missing direct statements
are never filled with balanced choices, the diagnostic case, an average, zero,
or any other placeholder.

Each accepted posterior becomes a
`VolunteeredPreferenceUpdate(case_id, user_id, updater_id,
directional_log_odds_update)`. The new artifact reuses the source run's bound
ACUE-superiority and balanced-valid-learning components, recomputes volunteered
valid learning and the Experiment A H7 criterion, and leaves the source run
unchanged. It records `claim_status = "not_claimed"`,
`source_run_modified = false`, and `missing_values_imputed = false`. The
closed-loop Experiment B component is still required for any full H7 claim.
Exact formulas are in [Metrics](metrics.md).

### Held-out paraphrase readiness and Gate 1

The held-out suite has train/development/test surface families and rejects any
family that crosses splits. Experiment A renders test cases from controlled
source trials without changing the selected option or visible context. It
evaluates the fitted-aware updater and, when configured, `llm_full_context`.
Every case and score carries content hashes tying the surface text to its source
and suite version.

Gate 1 is outcome neutral. It checks the same-response audit, exact-oracle
self-consistency, complete declared mechanism/domain coverage, and whether the
exact warranted update differs nontrivially from balanced for at least two
non-balanced mechanisms in both domains. It does not require
`llm_full_context` to make an error.

For held-out surfaces, the controlling criterion is complete required
case/updater coverage plus structural invariance: every paraphrase of a source
must retain its selected option, domain, mechanism, and visible-context
binding. The artifact is schema v2 and preserves `verified = null` when
required pairs are missing. Historical fitted-aware versus full-context Brier
gaps and `qualifying_mechanisms` remain in the artifact as descriptive
compatibility diagnostics, but neither controls Gate 1. Fitted
aware-versus-unaware learnability is likewise reported outside the gate. Thus
a structured-only smoke run is incomplete, while a complete response corpus
can exercise the full readiness check. `claim_status` remains `not_claimed` in
either case.

## Experiment B: evidential legibility and behavioral feedback

Use:

```bash
PYTHONPATH=src python -m cape_loop run configs/offline/experiment_b.toml
```

To inspect the closed-loop mechanics before launching a pilot, the
demonstration-only command

```bash
PYTHONPATH=src python -m cape_loop demo experiment-b-case \
  artifacts/local/experiment-b-case --execute-live
```

runs one incorrect-profile user through matched balanced and
soft-profile-conditioned branches, with three turns by default and one
`llm_full_context` logical call per turn. The command also accepts 6, 9, or 12
turns per branch; content-identical logical requests can share one physical
provider response. Its output is not a sample from the factorial design and is
never paper- or claim-eligible. Three turns exercise transport and structured
updates, but normally do not revisit an attribute after it is updated. Use at
least six turns to inspect the declared later-action mechanism.

The bounded live pair is
`configs/live/experiment_b_openai.toml` and
`configs/live/experiment_b_openrouter.toml`. Both cross correct/incorrect seeds
with balanced, soft-profile-conditioned, and exploratory policies, and evaluate
one `llm_full_context` model at a time alongside local reference updaters. Each
uses eight users, two domains, and six turns, so every preference dimension is
revisited once. The preflight bound is 576 trajectory updates plus 60
five-mechanism development-calibration calls: 636 physical attempts with
retries disabled.
Run the same frozen design separately for each model family; never combine
provider calls from different models into one nominal updater. The request and
token ceilings are defined and preflighted in
[Live execution](live-execution.md).

For the OpenRouter calibration, first build and stress the manipulation without
a key or model call:

```bash
PYTHONPATH=src python -m cape_loop experiment-b manipulation-audit \
  configs/live/experiment_b_openrouter.toml \
  artifacts/experiment-b-manipulation-audit
```

The output contains the complete JSON schedule, a readable active-turn table,
and the multi-seed simulator audit. The plan is admitted from ex-ante inputs
only; simulated outcomes describe it but never reselect scenarios or change
admission. A local exact action-aware updater evolves adaptive policy state;
target-writer behavioral reinforcement is deliberately `not_evaluated` because
the audit has no evaluated LLM output.

The model suite is also a dry plan unless live execution is stated explicitly:

```bash
PYTHONPATH=src python -m cape_loop experiment-b model-suite \
  configs/live/experiment_b_openrouter.toml \
  --output-root runs/experiment-b-suite
```

It freezes Gemini 3.6 Flash, GPT-5.6 Luna, and Mistral Large 3
(`mistralai/mistral-large-2512`) as the full primary panel. DeepSeek V4 Flash is
a post-pilot secondary replication of the incorrect-seed balanced-versus-soft
contrast. Every model is analyzed separately; no model outputs are pooled, and
DeepSeek is outside the primary analysis set. Adding `--execute-live`
authorizes all four isolated sequential OpenRouter runs.

### Executed crossing

The runner crosses:

```text
domain
× test user
× initial profile condition
× replicate
× configured policy
× configured updater
```

The four supported initial profile conditions are:

```text
correct
incorrect
uncertain
empty
```

`experiment.initial_profile_conditions` selects a nonempty subset. Both
checked-in Experiment B presets use `correct` and `incorrect`; `uncertain` and
`empty` remain available for separate diagnostics. The presets select balanced, softly
profile-conditioned, and exploratory policies where the budget permits. Other
known policies may be selected by a different valid closed-loop config. The
strict contract requires the ranking/default/suggestion mechanism declaration,
naturally sampled responses, and a nonnegative bootstrap count. The checked-in
offline configuration uses 2,000 replicates. A count of zero is permitted only
to make smoke and integration runs inexpensive; it emits point estimates marked
`not_computed` and cannot computationally pass Gates 2 or 3.

Every trajectory has:

- one fixed latent preference and susceptibility state;
- an evaluated updater;
- an exact action-aware shadow initialized from the same profile;
- the same observed context and choice supplied to evaluated and shadow paths;
- option-keyed common random numbers shared across paired branches; and
- evaluator-only counterfactual policy calls that reset one accumulated
  attribute update to test whether strengthening above the seeded wrong mass
  changed a later action signature.

Experiment A is the fixed-response attribution arm: it holds the selected item
and reply constant and already includes balanced, restricted, ranking, default,
and suggestion contexts. Experiment B is the natural-response feedback arm:
the presentation may change the sampled choice. An A-side attribution failure
does not establish a changed response, and a B-side choice change does not by
itself establish updater misattribution. The outputs keep those paths separate.

The existing `llm_provenance_aware` condition tests the combined effect of
structured provenance metadata and a provenance-aware instruction. It is not a
pure source-label ablation. Crossing metadata absent/present with instruction
absent/present, assistant-only input, and neutralized assistant wording is a
versioned extension and is not silently added to the current primary design.

Turn records retain evaluated joint state when that updater has one and always
retain the exact shadow's joint state before and after the event. Terminal
evaluated/shadow joint states are retained separately.

Before a natural response is drawn, every turn records two prospective
difficulty labels:

- latent target-preference strength is `weak` for `|theta| = 1` and `strong`
  for `|theta| = 2`; trajectory rows also retain the three attribute strata
  and label a user `mixed` when both magnitudes occur; and
- the top-two choice-probability gap under the matched balanced action is
  `near_tie` below `0.20`, `marginal` from `0.20` to below `0.50`, and
  `decisive` at or above `0.50`.

These strata are computed from the frozen latent user and declared response
model before observing the sampled choice. They support prespecified
heterogeneity summaries without treating seeds as independent users or
post-selecting cases that happened to diverge. For configurations with
`manipulation.planning_mode = "required"`, the outcome-blind scheduler fails
before a model call unless every balanced/soft six-turn pair has at least two
near-tie/marginal active turns, one decisive active control, two active
mechanisms, both preference directions available, and sufficient active
susceptibility mass (ASM). Realized choice divergence is never an admission
condition.

Every action also records three prospective mechanism descriptors:

- structural profile consistency in `[-1, 1]` and its difference from the
  paired balanced counterfactual;
- exact expected information gain about the preference marginal, obtained by
  integrating over every possible displayed response before sampling one; and
- for paired binary actions with the same option IDs, the exact shared-noise
  probability that the presentation changes the choice relative to balanced
  (shared Gumbels for random utility; shared inverse-CDF noise for the
  rule-based sensitivity model).

These are reported beside realized exact information, visible-action
divergence, and realized-choice divergence. A null choice-probability value
means the actions do not share an identical binary choice set, not that
susceptibility is zero. Reports retain the comparable-turn denominator and the
separate changed-choice-set rate so policies are not compared on hidden
coverage differences.

The soft policy always displays both preference directions. Required active
turns use the predeclared scenario and mechanism and must visibly diverge from
balanced; the promoted direction follows the current profile, with planner
admission calibrated conservatively across either possible profile direction.
If an updater produces an exactly neutral target expectation, the active turn
uses the initial-profile direction frozen before outcomes and records that
fallback explicitly. Correct and incorrect initial-profile conditions reuse one
condition-invariant scenario-role-mechanism and exogenous-randomization schedule
per domain-user-replicate group, so their moderation contrast does not change
the underlying stimuli.
Adaptive-observation turns continue to rotate ranking, default, and suggestion
and apply treatment with a probability determined by current profile
confidence. Runtime records the planned role, mechanism, susceptibility bound,
and execution match on every scheduled turn and fails on a required mismatch.

For sensitivity runs, a numeric policy-strength dose multiplies that legacy
adaptive treatment probability. A dose of `0` is the balanced-action negative
control, and `1` exactly reproduces the ordinary soft-policy implementation.
The ordinary unplanned policy has no numeric dose override. Required-plan
execution uses the `v5-condition-matched-active-turns` provenance version; adaptive
turns retain the same confidence-dependent behavior inside that schedule.

The exploratory policy is `v3-balanced-coverage-shared-neutral-ranking`. At each turn it restricts
selection to the least-exposed attribute or attributes, then chooses the one
with greatest current marginal entropy, with a deterministic index tie-break.
Consequently, every complete three-turn block covers all three attributes and
the prefix exposure counts can never differ by more than one. The order within
the block can respond to the current public profile; total coverage cannot
collapse onto one dimension.

### Decomposition and predicate

Balanced versus profile-conditioned shadow paths supply evidence-selection
comparisons. Evaluated belief versus its same-history shadow supplies
attribution comparisons.

After every update, trajectory construction asserts that the evaluated updater
and exact shadow have consumed the identical ordered event-ID history. A
history mismatch aborts the run instead of emitting a reassuring flag.

Every soft-policy turn also receives an evaluator-only balanced-policy action
signature constructed from the same user, scenario, profile, turn, and semantic
random seed. This creates direct, paired indicators for whether the visible
action and observed choice diverged from their balanced counterparts. It is a
policy contrast, not another user interaction.

Experiment B's primary and supporting endpoints are continuous. The sole
primary claim is Gate 3's conjunction of the soft-policy attribution gap, its
paired soft-minus-balanced contrast, and evidence-selection cost. Gate 2, seed
moderation, and nested net harm form the gated Holm-adjusted secondary family.
Terminal, decomposition, and inference artifacts additionally report the
supporting measures below:

- **policy-specific attribution gap** `G_policy`: evaluated-system terminal
  error minus exact same-history shadow error;
- **primary attribution-gap contrast:** the prospectively schedule-matched
  `G_soft - G_balanced`;
- **secondary seed moderation:** the incorrect-minus-correct moderation of that
  contrast;
- **supporting whole-policy comparator:** `G_soft - G_exploratory`; exploratory
  target and scenario selection remains adaptive, so this is not a turn-matched
  causal branch;
- **error amplification ratio (EAR):** terminal marginal Brier error divided by
  initial marginal Brier error;
- **cumulative excess confidence (CEC):** cumulative system-minus-exact-shadow
  log-odds confidence gain, averaged over attributes that were wrong initially;
- **paired CEC contrast:** soft-policy CEC minus balanced-policy CEC for the
  same user/domain/updater/replicate; this is Gate 2's controlling **relative
  confidence penalty**, not by itself absolute amplification or reinforcement;
- **action-aware information deficit:** exploratory exact-shadow information
  gain minus soft-policy exact-shadow information gain;
- **disconfirmation deficit (DD):** exploratory exact-shadow disconfirming
  log-evidence minus soft-policy disconfirming log-evidence for the initially
  false profile direction;
- **evidence-selection cost:** profile-policy shadow error minus
  balanced-policy shadow error;
- **same-history attribution cost:** evaluated-system error minus exact-shadow
- **total updater-policy effect:** soft terminal updater error minus balanced
  terminal updater error, which must equal SelectionCost plus the soft-minus-
  balanced attribution-gap contrast within the configured tolerance;
- **partial reinforcement-event rate:** the fraction of turns on which a visible
  profile-aligned action differs from balanced, the selected response supports
  the false direction, and the evaluated updater gains more false confidence
  than its exact same-history shadow; and
- **paired behavioral-reinforcement rate:** events divided by active false-
  profile-aligned soft-treatment opportunities, additionally requiring a
  same-turn soft-versus-balanced choice change toward the false profile. The
  rate is null with no opportunities.

The runner additionally reports **Disconfirmation Inversion Rate (DIR)**. An
opportunity is an initially false attribute-turn on which the exact shadow
reduces false-sign confidence; an inversion occurs when the evaluated updater
increases that same false-sign confidence. DIR divides inversions by
opportunities and remains null when no opportunity exists. It has no
profile-action or behavior-change clause and therefore remains distinct from
the reinforcement-event and strict self-confirmation rates.

EAR is undefined when initial error is numerically zero, and exploratory
deficits are available only when a matched exploratory trajectory exists.
These continuous measures can show partial feedback-loop formation without
claiming a stable self-confirming equilibrium. The original strict five-clause
self-confirmation rate remains a secondary endpoint and diagnostic gate; a null
strict rate does not erase an updater-side attribution or evidence-selection
failure.

The runner's inference-v5 artifact first reduces repeated rows to equally
weighted complete-user means. Its primary directional decisions use one-sided
paired sign-flip inference: all $2^n$ signs are enumerated for at most 16 users;
larger samples use 16,384 deterministic Monte Carlo patterns, include the
observed assignment, and use a plus-one correction. The minimum is eight users
and alpha is `0.05`. Sign exchangeability around the tested null margin is the
required assumption.

Deterministic 95% percentile user-cluster bootstrap intervals are sensitivity
summaries. Complete paired-trajectory resampling is an additional sensitivity
and must not be interpreted as making repeated trajectories independent users.
These analyses are not a GLMM or a user-level mixed-effects model. The latter
is implemented by the separate,
version-pinned [R mixed-effects harness](../analysis/confirmatory-mixed-effects/README.md)
and must be fitted on verified paper runs in the declared R environment.
Its scenario random intercept uses the actual scenario displayed on each
retained turn. Its separate CRN-set intercept uses the common key shared by the
counterfactual policy/updater twin set. The CRN set is not a substitute
scenario ID: endogenous branches can remain paired by random numbers while
targeting different attributes and therefore displaying different scenarios.

### Pilot power for the frozen three-way interaction

Every Experiment B run also writes a bounded pilot-design power curve for the
proposal's primary `Updater × Policy × InitialProfile` terminal-error
interaction. The frozen scalar contrast is:

```text
[(target soft - target balanced)
 - (fitted-aware soft - fitted-aware balanced)] at incorrect initialization
-
[(target soft - target balanced)
 - (fitted-aware soft - fitted-aware balanced)] at correct initialization
```

`target` is `llm_full_context` when present and otherwise the deterministic
`full_context_blind` reference. The analysis is `not_estimable` rather than
changing the estimand when neither target is configured or fewer than two
complete users remain. Each included domain×replicate stratum must contain all
eight updater×policy×initial-profile cells. Stratum contrasts are averaged
within latent user, so repeated domains and trajectories never become
independent units.

The simulator centers the complete-user pilot contrasts, adds back their pilot
mean as the declared target effect, resamples complete users, and applies a
two-sided normal-reference one-sample test at alpha 0.05. It evaluates the
frozen candidate user counts 16, 32, 64, and 128. The configured
`experiment.bootstrap_replicates` supplies the requested simulation count, with
a 200-replicate smoke minimum and a hard 10,000-replicate ceiling. Every point
reports its binomial Monte Carlo standard error and 95% Wilson interval.

The computational decision rule identifies the first candidate whose lower
95% Monte Carlo bound is at least 0.80. That result is advisory: the artifact
sets `automatic_sample_size_commitment = false`, requires investigator review,
and does not replace preregistration or the optional confirmatory mixed-effects
model. It is pilot-design evidence, never empirical support for a paper claim.

A self-confirmation assessment is reportable only when the implemented
five-clause predicate is satisfied: wrong initialization, policy-conditioned
evidence, excess confidence beyond the shadow, later action influence, and
terminal persistence under the declared thresholds.

The later-action clause requires the wrong mass to remain above its initial
level when the action is selected. An action difference caused only by
correcting or weakening the profile below its wrong initial seed does not count
as strengthening-driven self-confirmation.

LCG and shadow equivalence use separate operational checks. LCG compares
system-versus-shadow log-odds gain against
`thresholds.laundered_confidence_gain`; shadow equivalence compares terminal
wrong-direction probability mass against
`thresholds.shadow_equivalence_tolerance`.

Computing that predicate does not establish that any case occurred.

### Native and terminal evaluation

After every trajectory, the runner evaluates the terminal belief on a common
per-domain `heldout-terminal-v2` battery. For native updaters, it also runs both
fixed blinded decoders and evaluates each decoded belief on that battery.

Experiment B also retains the underlying v2 suite and exercises its explicit
action contract. The suite's option IDs, feature vectors, wording-template IDs,
and scenario-family IDs do not overlap fitted-model training material.
Terminal actions must cover it exactly and repeat each item's content digest,
wording ID, and question type. The runner records:

- `structured_profile_action_reference`, projected from the public structured
  belief; and
- `native_persona_action_reference`, projected directly from the native
  state's policy-facing persona belief and bound to its state ID.

These transparent reference adapters test the held-out action contract and
score choice, direct-probe, and cross-context items. They are not opaque
end-to-end natural-language calls and do not establish natural-language
generalization by themselves.

When event retention is enabled, closed-loop trajectory records retain the
complete native state before and after every turn and at the terminal point.

For every native terminal state, the runner also generates external-decoder
materials for disjoint development and test users:

```text
decoder/external-requests.jsonl
decoder/truth-labels.researcher-only.jsonl
decoder/researcher-codebook.jsonl
decoder/design-manifest.json
```

The request payload excludes system/updater IDs, memory kind, user ID, initial
profile, and latent truth. The truth and codebook files must not be sent to a
decoder. The design requires at least two decoder instances, distinct source
descriptors, and distinct decoder families per request, while explicitly
stating that metadata cannot prove independent errors.

External judgments can be hash-validated and analyzed separately. Calibration
fits one temperature per decoder family on development labels only; held-out
test analysis reports raw/calibrated Brier, NLL, accuracy, ECE/reliability bins,
and cross-family argmax/total-variation agreement. No such judgment corpus is
included with the repository.

```bash
PYTHONPATH=src python -m cape_loop decoder-study validate \
  REQUESTS.jsonl JUDGMENTS.jsonl
PYTHONPATH=src python -m cape_loop decoder-study analyze \
  REQUESTS.jsonl JUDGMENTS.jsonl TRUTH-LABELS.jsonl --output analysis.json
```

`decoder-study plan-openrouter` constructs the selected Claude/Gemini
shared-gateway plan without reading a key. The corresponding live command is
explicitly authorized and budget-bound as described in
[Live execution](live-execution.md). Direct-provider and two-role OpenAI
adapters remain optional alternatives; distinct role or model labels still do
not establish statistically independent decoder errors.

### Artifacts

```text
analysis/experiment-b-turns.jsonl               # one compact retained turn
conversations/experiment-b.jsonl                 # exhaustive trajectory traces
conversations/experiment-b.md                    # diverse human preview, at most 100
events/experiment-b-trajectories.jsonl         # when retain_events = true
events/experiment-b-terminal-batteries.jsonl   # when retain_events = true
events/experiment-b-held-out-terminal-suites.jsonl # when retain_events = true
metrics/experiment-b-terminal.jsonl
metrics/experiment-b-prospective-strata-occupancy.json
metrics/experiment-b-native-decoders.jsonl
metrics/experiment-b-held-out-actions.jsonl
metrics/experiment-b-terminal-calibration.json
metrics/experiment-b-decomposition.jsonl
design/experiment-b-manipulation-plan.json
design/experiment-b-manipulation-plan.md
design/experiment-b-offline-manipulation-audit.json  # live LLM configurations
metrics/experiment-b-h7-mitigation.json
metrics/experiment-b-self-confirmation.jsonl
metrics/experiment-b-inference.json
metrics/experiment-b-power.json
metrics/experiment-b-llm-raw-calibrated-terminal.jsonl
metrics/experiment-b-llm-raw-calibrated-terminal-manifest.json
tables/experiment-b-decomposition.csv
tables/experiment-b-power.md
tables/experiment-b-llm-raw-calibrated-terminal.csv
decoder/external-requests.jsonl
decoder/truth-labels.researcher-only.jsonl
decoder/researcher-codebook.jsonl
decoder/design-manifest.json
metrics/gate-report.json
metrics/summary.json
```

The always-retained turn and terminal rows contain the legibility/action fields
even when raw trajectory retention is disabled. The raw event record adds
complete beliefs, gain vectors, visible contexts, responses, and policy
provenance for forensic reconstruction. `experiment-b-inference.json` is
schema v5 (`experiment-b-clustered-randomization-v5`). Its `directional_tests`
retain the null margin, alternative, alpha, p-value, decision, cluster count,
exact-versus-Monte-Carlo status, and sign-pattern count. Older completed runs
remain valid under their original schema and do not acquire these estimands
retroactively.

Its `multiplicity` result executes
`experiment-b-within-model-gatekeeping-v1`. Gate 3 is the primary IUT and uses
the maximum of its three component p-values. Only after that conjunction
rejects does Holm operate on the frozen secondary family: the Gate 2 IUT,
incorrect-minus-correct moderation, and nested net harm. Missing members remain
in the family with p=1. Supporting endpoints and bounded calibration are
descriptive, and every model run is analyzed separately without pooling or an
“any-model” decision. The gate report consumes these adjusted decisions; it
cannot replace configured selection or net-harm margins with artifact values.

The compact B file flattens each retained trajectory into one row per turn. It
keeps the trajectory/user/domain/updater/policy/initial-condition identifiers,
the common-random-number key, prospective user/target preference-strength
strata, the balanced-action target and choice-probability margin/stratum,
per-turn system/shadow Brier and attribution gap, exact expected and realized
information, action profile consistency and balanced advantage, ex-ante
paired-choice divergence, DIR opportunities/inversions, retained terminal
error, and same-history shadow indicator. It omits the repeated full joint belief and native-memory payloads
that make the raw trajectory audit large. The number of compact rows is the
number of retained turns, not a larger experimental sample.

The paper-primary clustered inference is not the supporting R terminal-error
model. It is written directly to `metrics/experiment-b-inference.json` from the
source trajectories. Historical `artifact compact` sidecars retain only the
core fields needed by the R model, so they cannot reconstruct the new
same-history, action-characterization, or DIR estimands on their own.

The B conversation trace instead keeps each trajectory as one readable record:
its turns appear in order with per-turn metrics, followed by the terminal
evaluation. It omits posterior arrays, latent truth, and native-memory payloads.

Each Experiment B terminal row includes profile Brier and projected behavioral
scores, exact-shadow error, same-history attribution gap, exact-shadow
improvement, expected and realized information, action profile consistency,
ex-ante paired-choice divergence, DIR counts/rate, same-history
shadow-to-system marginal KL, preference-dimension
coverage and time-to-full-coverage, displayed-option diversity, distinct
selected-option count, profile-conditioned exposure rate, mechanism
count/evenness, prospective weak/strong and balanced-margin summaries,
cumulative action-aware information gain, total intrinsic regret, and the
explicitly defined false-stable attribute rate and trajectory flag for
incorrect-seed trajectories. It also includes top-label profile ECE and fixed
reliability bins over its three preference-attribute forecasts. The pooled
calibration artifact groups these records by public projection or deterministic
decoder and states that trajectory/user, not attribute, is the dependence unit.

Held-out action rows include the suite ID/digest, adapter kind, action bindings,
behavioral accuracy, cross-context accuracy, and intrinsic regret. They must be
interpreted separately from the common projection-battery row and from external
decoder evidence.

For a temperature-calibrated LLM run, the raw/calibrated terminal table scores
both cached vectors for the same final prompt and common battery, adding no
provider calls. Multi-turn rows are conditional on the calibrated active
history and set `full_counterfactual_rerun_required = true`; they are not a
recursive raw trajectory and do not replace Gate 2/3 inputs.

The H7 mitigation artifact pairs `llm_full_context` and
`llm_provenance_aware` under soft profile conditioning with incorrect initial
profiles. It tests reductions in same-history attribution error and the
five-clause self-confirming-profile rate. It does not, by itself, establish
H7; the Experiment A superiority and balanced/volunteered retention criteria
remain required.

### Gate 2–4 behavior

The Gate 2/3 runner targets only `llm_full_context`, the central ordinary
full-context writer. `llm_response_only` and `llm_provenance_aware` are controls
and are not pooled into the gate decision. The checked-in
`configs/offline/experiment_b.toml` has no `llm_full_context` updater, so its
target set is empty and Gates 2 and 3 are incomplete. Both bounded live B
pilots include that target, but neither has been executed as a paper result.

With correctly configured replay or explicitly authorized live LLM updaters,
the code can compute the declared checks, but still records
`claim_status = "not_claimed"`. Inference v5 identifies policy-specific
same-history gaps, their policy/seed contrasts, SelectionCost, and the total
soft-minus-balanced updater error as the primary continuous family; EAR, CEC,
reinforcement, DIR, and disconfirmation/information deficits explain distinct
parts of the result.

Within the incorrect-initial-profile stratum, Gate 2 requires an active soft
manipulation, visible action divergence, some natural-choice divergence, later
action influence, and one-sided complete-user evidence for a positive
soft-minus-balanced CEC relative penalty. Gate 3 uses the same stratum for the
separate policy-conditioned-legibility conjunction: positive soft-policy $G$,
positive soft-minus-balanced $G$, and SelectionCost below the frozen `0.02`
noninferiority margin. The nested `gate-3-net-profile-harm` additionally
requires incorrect-seed total soft-minus-balanced updater error above `0.02`.
It is serialized under `nested_gates` rather than replacing or renumbering Gate
3. Bootstrap intervals remain sensitivity evidence. With
`bootstrap_replicates = 0`, both the intervals and directional gate decisions
are `not_computed`; too few users makes the directional decision inadequate.

Gate 4 first restricts to incorrect-seed, soft-profile-conditioned native
trajectories whose choices retained both preference directions. It checks
complete retained terminal state and whether a semantic five-clause failure was
observed relative to its matched, equal-strength provenance-linked control.
Two additional criteria deliberately remain incomplete in an ordinary runner
output: every eligible state must have imported, blind, source-design-eligible
judgments from at least two genuinely distinct decoder sources, and the native
system itself must have produced hash-bound `native_end_to_end_recorded`
terminal actions. The two repository decoder projections and the
`structured_profile_action_reference`/`native_persona_action_reference` rows
are diagnostics but are explicitly ineligible for those criteria. Episodic
memory remains evaluated but is excluded from the causal contrast because its
transition strength differs.

`_gate_4_for_b` has an internal post-validation evidence-injection boundary so
the criterion logic can be audited without weakening it. Completed runs remain
immutable. `gate-review import-native` validates and binds the complete
selected OpenRouter `anthropic/claude-sonnet-5` plus
`google/gemini-3.6-flash` decoder collection and complete OpenAI native-action
collection to eligible trajectory IDs, recomputes Gate 4, and writes a
separate checksum-bound artifact. Every per-model plan, reasoning setting,
physical attempt, accepted gateway audit, judgment/action record, execution
manifest, and evidence-file digest is checked under the collection locks. The
gateway collection retains `first_party_origin_claimed = false`; the explicit
reviewed-generic alternative does not claim validated provider provenance. The
import never rewrites the run's gate report. Ordinary Experiment B runs remain
incomplete until that import exists.

For the selected evidence packet, use the separate offline
`configs/offline/gate4_source.toml`. It produces 320 trajectories, 640
blinded decoder requests for each model family, and 80 eligible native actions
without calling a provider. The selected zero-retry plans must fit the approved
per-source hard ceilings documented in [Live execution](live-execution.md).
The two decoder rows are distinct model families, not distinct transport
origins or a statistical-independence claim. Current local validation and
collection state is reported only in
[Implementation status](implementation-status.md).

## Experiment C: logging-policy-dependent evaluation validity

Experiment C remains a secondary version-1 study. It asks whether system
rankings and selected systems change with the policy that generated the
evaluation history; it is not needed to identify Experiment A's updater-side
provenance calibration or Experiment B's feedback-loop decomposition.

Use:

```bash
PYTHONPATH=src python -m cape_loop run configs/offline/experiment_c.toml
```

The bounded live pair is `configs/live/experiment_c_openai.toml` and
`configs/live/experiment_c_openrouter.toml`. Each crosses the same seven local
systems plus one `llm_full_context` writer over both splits, domains, and all
three regimes. Its whole-design bound is 768 evaluation calls plus 60
five-mechanism calibration calls, or 828 physical attempts with retries
disabled and hard ceilings validated before credential access.

For external-decoder reranking, the separate offline
`configs/offline/experiment_c_rescore_source.toml` preserves all seven local
systems and produces 360 requests for each external decoder family. Its
selected zero-retry OpenRouter plan must fit the approved per-model ceilings.
Generating the source packet performs no provider call and is not an external
rescore result.

### Required policies and regimes

The configuration must contain exactly:

```text
balanced
fixed_bias
soft_profile_conditioned
```

For every development/test user, domain, and replicate, the runner creates:

1. `fixed_balanced` — one balanced history generated independently of evaluated
   updaters, then replayed to every updater;
2. `fixed_biased` — one mildly biased history generated from a fixed logger,
   then replayed to every updater; and
3. `endogenous_closed_loop` — one soft-profile-conditioned trajectory per
   updater.

Fixed histories have a content digest and per-event signatures. The runner
asserts that every updater receives identical fixed event objects. Endogenous
histories differ by design because the updater's profile affects later actions.

Accordingly, the primary evaluation object is
`Score(updater, logging_policy)`, not a single policy-independent updater
score. The current executable `v1` comparison contains fixed balanced and
fixed-bias loggers plus an endogenous soft-policy deployment branch. It can
measure absolute open-loop optimism and paired system-selection error for those
regimes. A larger crossed set of balanced, exploratory, randomized, and
profile-conditioned fixed loggers would be a versioned Experiment C extension,
not something the current artifacts imply.

### Terminal battery

`build_terminal_battery` wraps the versioned `heldout-terminal-v2` suite into a
common projection battery per domain:

- balanced preference choices;
- counterbalanced matched choices;
- neutral direct probes; and
- two cross-context composite choices.

Defaults and suggestions are forbidden in the battery. Its content digest is
identical across compared systems. The evaluator scores profile Brier,
full-credit/tie-excluded/fractional behavioral accuracy, cross-context
accuracy, intrinsic regret, and explicit predicted/intrinsic tie counts. Every
structured, decoder, and ranking score also reports top-label multiclass ECE
and fixed reliability bins. The calibration sample unit is one
preference-attribute forecast; trajectory/user grouping remains the dependence
unit for inferential work.

The v2 constructor uses novel option IDs and feature vectors, scenario families,
and wording-template IDs, and preserves all four question types. It validates
those identities against training-domain material before creating the battery;
the ranking path no longer reuses the domain training option pool.

The deterministic evaluator still acts on the held-out option feature vectors:
it projects a structured or decoded belief to a choice and does not interpret
natural-language wording. Thus this is a genuinely novel exogenous diagnostic
for every C ranking, while an opaque end-to-end system response remains a
separate external action-adapter requirement.

### Ranking

The runner computes open/fixed-biased/closed mean errors, ranks with declared
ties, Kendall's tau-b (nullable when undefined), bootstrap rank intervals,
pairwise reversal and tie probabilities, paired open/closed/test
error-difference intervals, joint open-versus-closed
difference-of-differences intervals, inferential partial orders, open-loop
optimism, and evaluation selection regret. A credible reversal requires the
joint paired bootstrap to resolve the system pair in opposite directions in
open and closed evaluation and to resolve the corresponding regime shift.
Marginal rank-interval non-overlap and reversal probability remain descriptive;
neither can satisfy Gate 5 by itself.

Before computing those quantities, systems are aligned by the stable
`(split, regime, user_id, domain_id, replicate)` pairing key, with
`updater_id` as the compared system dimension. Duplicate or missing system rows
are fatal. Within each split/regime, every user must retain the same complete
domain × replicate layout. The analyzer then reduces each system to one mean
per latent user and resamples those paired user clusters; all domains and
trajectory replicates for a user therefore move together in every bootstrap
draw. Input row order cannot affect the result. The ranking artifact records
the cluster counts, component layout, alignment key, independent unit, and
bootstrap method.

Development open-loop and closed-loop paired error-difference intervals each
define an inferential top tier. Disjoint-user test closed-loop intervals
determine regret for every open-top-tier × closed-top-tier pairing. The
artifact reports the descriptive mean/minimum/maximum and a conservative
envelope over the paired test intervals. Gate 5 requires the minimum and the
envelope lower bound to clear the practical threshold; an unresolved near tie
therefore cannot be promoted by a numeric tolerance or stable system ID.

For a structured system, top-level metric/ranking fields use its public
structured projection and record
`score_basis = "system_structured_projection"`. For a native system, they use
the arithmetic mean of exactly two fixed blinded projection scores and record
`score_basis = "mean_of_two_blinded_native_decoders"`.
`system_projection_score` retains the public persona projection separately, and
the two individual decoder scores and prediction sequences remain nested.
Because the mean has no single action sequence, the native row's top-level
`predicted_option_ids` is empty.

These quantities are descriptive until a retained analysis and uncertainty
review supports an interpretation.

### Artifacts

```text
analysis/experiment-c-rows.jsonl            # one compact evaluation/ranking row
conversations/experiment-c.jsonl            # exhaustive deduplicated histories
conversations/experiment-c.md               # diverse human preview, at most 100
events/experiment-c-fixed-histories.jsonl  # when retain_events = true
events/experiment-c-replays.jsonl          # when retain_events = true
events/experiment-c-endogenous.jsonl       # when retain_events = true
events/terminal-batteries.jsonl             # always retained
metrics/experiment-c.jsonl
metrics/experiment-c-terminal-calibration.json
metrics/experiment-c-rankings.json
metrics/experiment-c-llm-raw-calibrated-terminal.jsonl
metrics/experiment-c-llm-raw-calibrated-terminal-manifest.json
tables/experiment-c-ranks.csv
tables/experiment-c-llm-raw-calibrated-terminal.csv
metrics/gate-report.json
metrics/summary.json
```

The compact C file keeps one row for each existing fixed-history or endogenous
evaluation. It retains the split/regime/replicate/system identifiers, four
ranking scores, score basis, and history/battery digests while leaving nested
native decoder payloads and full replay/trajectory state in the canonical
metric and event records. It does not re-evaluate a system or add a ranking
observation.

The C conversation trace stores each fixed history once and groups all updater
evaluations that replayed it. Endogenous histories remain separate because the
evaluated updater can change later policy actions and dialogue. This
normalization is the main reason the trace is much smaller than a naive
updater-by-history transcript export.

When event retention is enabled, replay records retain complete terminal native
state when present and endogenous records retain complete native state at every
turn. Both native decoder evaluations remain nested in Experiment C metric rows
regardless of that event-artifact setting.

Replay `UpdaterState` records and endogenous trajectory records also retain
theta×susceptibility joint state for exact updaters; endogenous records retain
the exact shadow joint throughout.

C's raw/calibrated terminal files have the same same-request scope as B. A
multi-turn raw counterfactual may require alternate recursive prompts and, in
the endogenous regime, alternate actions. These rows therefore never replace
the active calibrated ranking or Gate 5 inputs.

Gate 5 evaluates joint-paired reversal and inferential-top-tier
selection-regret checks. Point Kendall tau, marginal rank intervals, and raw
reversal probabilities remain visible but are not gate-sufficient. The gate
never changes `claim_status` from `not_claimed`.

### Experiment C external-decoder rescore

Experiment C finishes and checksums its original ranking before external
judgments exist. Rescoring is append-only: it writes a separate immutable
review and never edits the source run.

Only these native updater families are rescored:

```text
episodic_memory
semantic_memory
provenance_linked_memory
```

Structured rows remain byte-equivalent JSON objects. The fixed local native
projections remain diagnostics and are not admitted as external judgments.

Every eligible source run exports:

```text
decoder/experiment-c-external-requests.jsonl
decoder/experiment-c-truth-labels.researcher-only.jsonl
decoder/experiment-c-researcher-codebook.jsonl
decoder/experiment-c-external-design-manifest.json
```

Only the external-request file is provider-visible. Its payload contains the
blinded native representation and pseudonymous state ID. The truth and codebook
files remain evaluator-only. The codebook binds every request to its exact
split/regime/replicate/user/domain/updater row, complete metric-row digest,
terminal-suite digest, native-state digest, and fixed-replay or endogenous
source-record digest. The design manifest binds all packet files, source C
rows, batteries, and source-run identity.

The selected collection obtains exactly one
`anthropic/claude-sonnet-5` and one `google/gemini-3.6-flash` judgment for every
request through OpenRouter. The bounded source configuration exports 360
requests per model. With zero retries, its complete plans must remain below the
approved 900-attempt and 6,000,000-token per-model ceilings. Both models share
the gateway; neither first-party origin nor statistically independent errors
are claimed. Direct Anthropic/Gemini collection remains an optional separate
origin replication.

Provider collection is documented in [Live execution](live-execution.md).
Import the selected complete collection with:

```bash
PYTHONPATH=src python -m cape_loop experiment-c-decoder import \
  RUN_DIR \
  DECODER_COLLECTION_DIR/judgments.jsonl \
  C_EXTERNAL_REVIEW_DIR \
  --openrouter-collection-dir DECODER_COLLECTION_DIR
```

Exactly one provenance mode is mandatory:

- `--openrouter-collection-dir` validates the complete selected gateway
  collection, including exact models, per-model reasoning, journals, route and
  cache identity, audits, manifest, and judgment digest;
- `--external-collection-dir` validates the optional direct-origin
  Anthropic/Gemini collection; or
- `--allow-reviewed-generic-decoders` treats instance, family, source, and
  origin metadata as caller-declared and makes no provider-provenance claim.

The two collection-directory flags reject the opposite collection kind; they
are not aliases. Every request must have exactly two `external_model`
judgments with distinct instance, family, and source descriptors. Those
metadata conditions establish design eligibility, not independence.

After provenance admission, the importer:

1. fits one temperature per decoder family on development labels only;
2. freezes and applies those calibrators to development and test judgments;
3. converts each calibrated marginal forecast to an independent-joint belief;
4. scores both beliefs on the exact common terminal battery;
5. replaces each native row's ranking score with the arithmetic mean of exactly
   the two family/source scores;
6. proves that no non-native row changed; and
7. reruns complete-user paired ranking, ESR, and Gate 5.

User-supplied calibrators and test-label fitting are rejected. The selected
gateway review records
`provenance_mode = "selected_openrouter_gateway_collection"`,
`gateway_provenance_validated = true`,
`provider_provenance_validated = false`,
`first_party_origin_claimed = false`, `shared_gateway = true`,
`distinct_transport_origins = false`, and
`statistical_independence_claimed = false`.

The importer rejects existing/symlinked destinations, outputs inside the source
run, unsafe inputs, and source/judgment mutation. It stages and fsyncs every
file, reverifies the source and collection, invokes the ordinary verifier on
the stage, and publishes only by same-filesystem atomic rename. Its output is:

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

Verify it alone or bind verification back to the exact source:

```bash
PYTHONPATH=src python -m cape_loop experiment-c-decoder verify \
  C_EXTERNAL_REVIEW_DIR

PYTHONPATH=src python -m cape_loop experiment-c-decoder verify \
  C_EXTERNAL_REVIEW_DIR --source-run RUN_DIR
```

The source run, review, recomputed Gate 5, and all collection artifacts remain
`not_claimed`.

### Experiment C multi-seed robustness

The offline multi-seed reviewer admits two to 32 checksum-valid completed
Experiment C source runs. Every source must retain a positive clustered
bootstrap, complete paired intervals, the complete-latent-user inference unit,
the stable alignment key, rankings, Gate 5 report, summary, and a distinct
nonnegative seed.

Scientific configurations must be byte-equivalent after removing only:

```text
run.name
run.seed
run.output_root
```

Updater sets/order, response and inference parameters, artifact settings, LLM
settings, tie tolerance, and executable source digest must remain identical.
External-decoder review directories are not accepted as source runs under this
v1 protocol.

Nine comparison dimensions are fixed before reading outcomes:

1. fixed-balanced development point ranking;
2. fixed-biased development point ranking;
3. endogenous closed-loop development point ranking;
4. fixed-balanced inferential top tier;
5. endogenous inferential top tier;
6. fixed-balanced inferential partial order;
7. endogenous inferential partial order;
8. Gate 5 decision together with computation status; and
9. open-selected and closed-selected ESR development sets.

For every dimension the review retains each distinct value pattern and its
source runs/seeds, unanimity, exact modal stability, exact pairwise agreement,
and every disagreeing run pair. Proportions are integer numerator/denominator
and reduced rational strings; rounded decimals never control a decision.
Bootstrap draws are not pooled, rerun, or treated as independent paper
replications.

Create and verify the review:

```bash
PYTHONPATH=src python -m cape_loop experiment-c-robustness review \
  artifacts/experiment-c-multiseed \
  runs/experiment-c-seed-1 \
  runs/experiment-c-seed-2

PYTHONPATH=src python -m cape_loop experiment-c-robustness verify \
  artifacts/experiment-c-multiseed

PYTHONPATH=src python -m cape_loop experiment-c-robustness verify \
  artifacts/experiment-c-multiseed \
  --source-run runs/experiment-c-seed-1 \
  --source-run runs/experiment-c-seed-2
```

Supplying only a subset of bound sources fails. The immutable output contains
`review.json`, `manifest.json`, and `SHA256SUMS`; it refuses existing outputs,
duplicate paths/seeds, incompatible configurations, unsafe checksum paths,
unknown boundary fields, non-finite numbers, incomplete pairs, and source
mutation. Unanimity means only exact equality across the supplied verified
seeds. It does not establish adequate power, Gate 5 passage, or a paper claim.

## Sensitivity grid

Use:

```bash
PYTHONPATH=src python -m cape_loop run configs/offline/sensitivity.toml
```

The checked-in config is a baseline-first, 22-point one-at-a-time declaration:

```text
1 all-baseline random-utility point
+ 2 nonbaseline values on each of decision noise, shared presentation,
  rank, default, suggestion, profile strength, and trajectory length
+ 1 nonbaseline prior-uncertainty value
+ 3 nonbaseline profile-conditioning policy doses
+ 3 rule-based baseline points, one per declared rule-noise value
= 22 points
```

Random-utility points do not carry a rule-noise value. Rule-based points are
repeated once per declared rule-noise value at the numeric baseline, keeping
family identity explicit. This design estimates marginal departures from the
baseline; it does not estimate interactions among sensitivity axes.

Each point:

- builds the declared random-utility or noisy rule-based response model;
- regenerates and fits raw aware/unaware models using the same population,
  split identities, contexts, and semantic choice-noise draws as every other
  point;
- calibrates the active models when configured;
- runs Experiment B with only the `incorrect` initial profile and the configured
  balanced/soft policies; and
- records updater×policy×domain error, information, regret,
  self-confirmation, and attribution diagnostics, paired decomposition rows,
  and a separate grand summary;
- injects declared prior uncertainty without changing split identity; and
- retains information gain and regret both cumulatively and per turn so
  trajectory-length comparisons do not rely on mechanically larger totals.

`profile_conditioning_strength_values` is behaviorally active. For a numeric
dose `lambda`, the soft policy applies its otherwise unchanged
confidence-dependent treatment with probability:

```text
lambda × ordinary_soft_policy_probability
```

Thus `0` yields balanced visible actions, intermediate values produce weak and
moderate exposure, and `1` is the exact legacy/full-strength baseline. This
axis changes what the user and updater see; it is distinct from
`presentation_multipliers`, which changes simulated user susceptibility to a
visible treatment. The `1` endpoint retains the ordinary
`v2-neutral-profile-tie` policy
provenance string, so provenance-aware prompts are identical to ordinary
full-strength runs; only intermediate/null-dose points use the versioned dose
label. Because the null dose is intended to remove the harmful
mechanism, it is a negative control and is not required to lie inside a
declared all-level harmful region. The complete dose axis is retained for
manipulation summaries and phase-boundary inference; it is not silently added
to the version-1 Gate 6 broad-simulator-parameter clause.

Every point receives an explicit visible-manipulation classification. At
`lambda = 0`, both profile-conditioned treatment exposure and visible-action
divergence must be zero; otherwise the negative control fails. At every
positive dose, both quantities must be positive. A positive-dose point with
zero visible divergence is a failed manipulation, even if a response-model
coefficient or reference calculation changed. Null points are never eligible
for the positive-dose operational region.

The runner supports `llm_response_only`, `llm_full_context`, and
`llm_provenance_aware` in replay, direct OpenAI, or OpenRouter mode. One shared
content-addressed provider follows the entire grid, so repeated prompts reuse
the same bound response while genuinely changed histories produce distinct
request hashes. LLM sensitivity has a stricter contract:

- `llm.calibration = "none"`; no point-specific probability calibrator is fit;
- prompts and trajectory events must both be retained;
- replay responses must cover every prompt actually reached by the adaptive
  grid; and
- before live execution, the exact logical update bound is multiplied by
  `max_retries + 1` and must fit within `llm.max_requests`.

Transport-only sensitivity smoke presets are intentionally not part of the
public configuration matrix. Provider construction, retry accounting, and
content-addressed reuse are exercised by offline tests; operators can derive a
temporary bounded diagnostic under ignored `configs/local/` when needed.
`experiment.turns` must be `1`; each point's actual trajectory length comes
only from `[sensitivity].trajectory_lengths`.

Artifacts are:

```text
models/sensitivity-fits.jsonl
conversations/sensitivity.jsonl          # exhaustive trajectory traces
conversations/sensitivity.md             # diverse human preview, at most 100
metrics/sensitivity.jsonl
metrics/sensitivity-decomposition.jsonl
metrics/sensitivity-grand.jsonl
metrics/sensitivity-prospective-strata-occupancy.jsonl
metrics/sensitivity-phase-points.jsonl
metrics/sensitivity-phase-domains.jsonl
metrics/sensitivity-phase-boundaries.jsonl
metrics/sensitivity-phase-specification.json
tables/sensitivity.csv
events/sensitivity-trajectories.jsonl  # when retain_events = true
llm/request-preflight.json             # when an LLM updater is configured
llm/sensitivity-request-preflight.json # when an LLM updater is configured
llm/requests.jsonl                     # mandatory for LLM sensitivity
llm/responses.jsonl                    # when an LLM updater is configured
llm/provider-audit.jsonl               # live modes
llm/transport-attempts.jsonl           # live modes
metrics/gate-report.json
metrics/summary.json
```

Sensitivity does not write another `analysis/` projection. Its existing
`metrics/sensitivity*.jsonl` files are already aggregated by declared grid
point and analysis stratum, and `tables/sensitivity.csv` is the compact readable
projection. The large optional `events/sensitivity-trajectories.jsonl` remains
the reconstruction audit. Its conversation trace uses the same closed-loop
format as B, with one record per trajectory and the full sensitivity point,
including `sensitivity_point_id`, retained among the conditions.

Each model row contains raw and active fitted bundles, calibration, and
development diagnostics. `sensitivity.jsonl` and its CSV are stratified by
domain, policy, and updater; `sensitivity-decomposition.jsonl` contains paired
policy contrasts; both attribute-assessment and trajectory/profile rate
denominators are named explicitly; `sensitivity-grand.jsonl` is descriptive
only. Terminal rows also expose EAR, CEC, reinforcement-event rate, and direct
visible-action/choice divergence diagnostics. Ordinary Experiment B retains
exploratory-minus-soft information and disconfirmation deficits. Sensitivity
uses only balanced and soft branches and therefore reports the corresponding
balanced-minus-soft exact-shadow deficits without additional model calls.

`sensitivity-prospective-strata-occupancy.jsonl` records one pre-response
coverage report per grid point for the selected phase target. An active-dose
point must have positive exposure, positive visible divergence, and at least
two incorrect-profile users per domain whose soft trajectory contains at least
two visibly divergent near-tie/marginal turns. Failure blocks the
trajectory-level mechanism interpretation; rows and outcomes remain retained.

Phase criteria are declared by metric, relation, and threshold in the resolved
configuration. The operational joint region requires an activated visible
manipulation, positive evidence-selection cost, adequate fitted-aware response
calibration, a positive soft-minus-balanced same-history attribution-gap
contrast, and at least 20%
rejection of profile-consistent suggestions by default. The opportunity and
rejection counts are retained per point; a point with no eligible suggestion
remains incomplete rather than receiving a zero rate. Strict wrong-profile
self-confirmation is listed in the phase specification as a secondary endpoint
and does not control the operational joint region. Boundary rows identify
adjacent observed grid values where a criterion or joint-region label changes
while all other boundary axes are fixed. They are observed-grid intervals, not
interpolated causal thresholds.

The phase target prefers `llm_full_context`, then
`llm_provenance_aware`, `llm_response_only`, and the structured
`full_context_blind` proxy. Every grand row states the selected updater,
whether it is an external LLM, and whether execution was replay or live.
Only a live `llm_full_context` target can populate the confirmatory joint-region
field. Other LLM/proxy targets remain diagnostic. In particular, at
`lambda = 0` a provenance-aware updater can still see the soft branch's
metadata, so that cell is only a visible-action negative control, not a
full-prompt no-treatment control.

Gate 6 now reports the proposal's six clauses separately. Another-response-
model, broad-parameter, both-domain, and exact/fitted-reference clauses are
computed from completed point and domain phase rows. Multiple LLM families and
held-out paraphrases remain explicitly incomplete inside any one run. Model
labels or wording templates are never promoted into family identity,
independence, or a paper claim.

### Gate 6 cross-run review

The offline `gate6-review` joins the evidence that one sensitivity run cannot
contain. Each caller-declared family requires:

1. a complete checksum-valid sensitivity run whose phase target is the live
   `llm_full_context` updater; and
2. a complete checksum-valid Experiment A run using the same requested and
   returned model/provider evidence.

Both runs must retain prompts, responses, exchange/provider manifests, accepted
provider audit, settled physical-attempt journal, and development-only or
no-calibration manifest. Sensitivity contributes Gate 6, phase/domain, and
fitted-model rows; Experiment A contributes its fixed held-out paraphrase
suite, bound cases/scores, and readiness result. The importer recomputes both
the within-sensitivity clauses and the held-out coverage/invariance check,
including the binding from each LLM score to its retained provider response.

All family pairs use the same scientific design. Only run name, seed, output
location, and LLM provider/model/transport fields may differ. Calibration,
updater design, response parameters, phase rules, and every other scientific
field remain matched.

The declaration is written before review and binds:

- a responsible-researcher ID, timestamp, preregistration record, and whether
  family assignments preceded outcome inspection;
- `statistical_independence_claimed = false`;
- a unique pair and caller-declared family ID;
- exact sensitivity/A run IDs plus each source `SHA256SUMS` digest; and
- provider source, requested model, returned model, and any retained
  OpenRouter upstream display metadata.

At least two distinct family/model bindings and distinct source runs are
required. OpenRouter upstream labels are retained gateway metadata, not proof
of a physical route. The software validates declaration consistency but does
not infer family lineage or genuine independence.

Build and verify the separate artifact:

```bash
PYTHONPATH=src python -m cape_loop gate6-review build \
  gate6-declaration.json artifacts/GATE6-REVIEW

PYTHONPATH=src python -m cape_loop gate6-review verify \
  artifacts/GATE6-REVIEW

PYTHONPATH=src python -m cape_loop gate6-review verify \
  artifacts/GATE6-REVIEW --reverify-sources
```

The output contains:

```text
declaration.json
evidence/pairs.jsonl
metrics/gate-6.json
review.json
manifest.json
SHA256SUMS
```

It refuses incomplete responses, unresolved transport attempts, ambiguous model
identity, source mutation, missing paraphrase coverage, unsafe paths, or
incompatible grids. The six tri-state clauses are exactly:

1. another response model;
2. broad simulator parameters;
3. both domains;
4. multiple caller-declared LLM families;
5. held-out natural-language paraphrases; and
6. exact and fitted action-aware references.

The four within-sensitivity clauses use a conservative conjunction across
pairs: any explicit failure is `false`, all passing is `true`, and missing
evidence remains `null`. Multiple-family replication requires a surviving
complete phase result for every declared binding; paraphrase transfer must
recompute completely for every pair.

All outputs remain `claim_status = "not_claimed"`. Even six computational
passes require researcher defense of the family taxonomy, source identity,
shared-provider/training dependence, preregistration timing, and paper analysis.
Provider collection and request ceilings are documented in
[Live execution](live-execution.md).

## Human pragmatic-study collection and analysis

Generate a packet with:

```bash
PYTHONPATH=src python -m cape_loop human-study generate OUTPUT_DIR \
  --assignment-id template \
  --seed 1729
```

The CLI constructs 10 fixed vignettes: five conditions in each of the travel
and writing domains. Conditions are volunteered, balanced, restricted, default,
and suggested.

The participant file hides source item IDs and condition labels. The separate
researcher codebook maps display IDs back to them. The packet also includes an
order manifest, rating schema, file-hash manifest, and a warning README.
The output directory must be absent or empty; generation refuses to mix with or
overwrite an existing nonempty packet.

After an authorized external collection, analyze de-identified JSONL with:

```bash
PYTHONPATH=src python -m cape_loop human-study analyze \
  RESPONSES.jsonl RESEARCHER-CODEBOOK.json
```

The importer binds every display ID to the assignment codebook, rejects
duplicate participant/item rows and changing participant metadata, and checks
the expected assignment-protocol, consent, and blinding versions. Analysis uses
only consented participants who passed the declared comprehension check. It
reports condition summaries, the observed evidence-strength ranking, response
times, and paired-participant bootstrap contrasts for the proposal's
volunteered/balanced and mechanism comparisons.

The code does not deploy a survey, recruit or compensate participants, approve
consent language, define a data-retention policy, or confer ethics/IRB approval.
Those decisions must be supplied by the responsible institution before
collection. A metadata field saying `consented = true` is validated as data; it
is not proof that valid consent occurred.

For H8, convert one verified Experiment A run's held-out controlled-anchor
metrics, then run the atomic human/model comparison:

```bash
PYTHONPATH=src python -m cape_loop human-study \
  evidence-from-experiment-a RUN_DIR model-evidence.jsonl \
  --source fitted=fitted_action_aware \
  --source primary=llm_full_context

PYTHONPATH=src python -m cape_loop human-study compare \
  RESPONSES.jsonl RESEARCHER-CODEBOOK.json model-evidence.jsonl h8.json \
  --primary-llm-source-id primary
```

The converter accepts only the fitted-aware updater and actual `llm_*`
updaters, verifies converted user/domain pairs against the retained test
population, and emits the positive part of the anchor-directional log-odds
update. It never invents the volunteered condition absent from Experiment A.
Every row binds its source run, record, and metric-file digest. H8 pairs
balanced and policy observations within scenario, averages scenarios inside
participant/test-user clusters, and independently bootstraps those clusters.
Missing sources, mechanisms, or adequate cluster counts stay `incomplete` with
`criterion_met = null`; all outputs retain `claim_status = "not_claimed"`.

## Stage-gated correction debt

Run the diagnostic reference only after explicitly acknowledging that the
prerequisite gate evidence has been reviewed:

```bash
PYTHONPATH=src python -m cape_loop correction-debt run \
  correction-debt.json --stage-gate-authorized
```

The protocol crosses four placements of an explicit correction:
before reinforcement, after one reinforcing interaction, after repeated
reinforcement, and after recurrent consolidation. Every false-seed arm has an
equally strong correct-seed control with the same correction and the same
semantic balanced-recovery evidence keys. Output retains all arm snapshots,
pair-level debts, and per-stage summaries for:

- profile recovery turn and corrective evidence to recovery;
- recovery-error area under the curve and terminal profile error;
- textual and behavioral recovery timing; and
- persistent wrong-derived-memory count.

The independent unit is the pair, not an individual recovery turn. The default
adapter is a transparent log-odds protocol reference, not an LLM or native
memory. A real H9 native/LLM adapter is intentionally deferred, and H9 remains
stage-gated and excluded from the minimum paper. The authorization flag only
prevents accidental early execution; it does not prove that a scientific gate
passed. A future H9 result requires a real system adapter, retained evidence,
and frozen review.

## Evidence still external

The repository contains no checked-in API keys or paper-intended live model
response corpus, genuinely distinct external decoder judgments, recruited
human participants, ethics determination, confirmatory mixed-effects result,
or paper results. Local OpenAI/OpenRouter transport smokes are execution checks
only. Authors, repository/DOI, and accepted-paper metadata are also unset. See
[Implementation status](implementation-status.md) for the current evidence
inventory and [Ethics and limitations](ethics-and-limitations.md) for
collection and claim boundaries.

No smoke, gate, packet, or checked-in configuration is an empirical result.
