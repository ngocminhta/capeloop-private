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

## Shared preparation

Experiments A–C call the same preparation path before their main runner:

1. load the travel and/or writing domain;
2. build a deterministic split manifest;
3. generate disjoint train, development, and test latent-user groups;
4. generate randomized training interactions across the four provenance
   mechanisms;
5. fit four-parameter aware and unaware likelihood models on the same training
   interactions;
6. generate a distinct development interaction set;
7. optionally fit separate aware and unaware temperatures on development
   outcomes; and
8. retain the split, population, raw model, active model, calibration, and
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

The complete contract is documented in [Data splits](data-splits.md).

## Experiment A: provenance audit

Use:

```bash
PYTHONPATH=src python -m cape_loop run configs/smoke.toml
```

### Executed design

For every selected test user, domain, target attribute, anchor direction,
prior-strength stratum, mechanism, response mode, and updater, the runner
constructs a matched anchor set.

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
| `default` | Balanced pair with the anchor preselected |
| `suggested` | Balanced pair with the anchor recommended |

The anchor ID and features are invariant. A matched set is excluded when the
declared simulator gives the held anchor less than
`response_model.minimum_matched_probability` in any requested mechanism.
Exclusions are retained when event retention is enabled.

`controlled_anchor` supplies the same anchor observation in every mechanism.
This is a functional provenance-sensitivity control, not an average treatment
effect. `naturally_sampled` samples from the declared response distribution for
each context.

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
metrics/experiment-a-hypothesis-estimands.json
metrics/experiment-a-oracle-slopes.jsonl
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

Metric rows include marginal Brier score, fitted-aware reference Brier,
excess Brier, action-conditioned update error, marginal KL, update-direction
accuracy, update magnitude, and evidence weight.

To avoid repeating the exact reference once per updater,
`experiment-a.jsonl` retains each updater row's theta belief projections and an
`exact_reference_id`, while `experiment-a-exact-references.jsonl` stores one
full exact theta and theta×susceptibility reference per trial. Join those files
on `exact_reference_id`. The in-memory experiment row also exposes
`posterior_theta_psi` for an exact evaluated updater.

The confirmatory bundle adds:

- directional log-odds update slope against the fitted-aware reference, with a
  user-clustered bootstrap interval;
- a data-derived fitted evidence-strength ordering across mechanisms;
- raw-versus-calibrated forecast scores and one-vs-rest marginal-class
  reliability bins;
- user-clustered paired mechanism contrasts and updater×mechanism interactions;
- a marginal OLS model with user-clustered CR1 covariance;
- Holm correction over its estimable non-intercept coefficient family; and
- paired user-cluster pilot-power simulation when enough complete differences
  exist.

The separate hypothesis-estimand artifact prevents those general-purpose ACUE
analyses from being mistaken for H1 or H2. It reports H1's anchor-directional
and update-strength contrasts, H2's explicit distance-to-unaware versus
distance-to-aware comparison, and H7's mitigation/valid-learning component.
See [H1, H2, and H7 estimands](hypothesis-estimands.md) for the frozen formulas
and incomplete-data rules.

The dependency-free CR1 regression is an auditable marginal robustness
analysis. It is **not** the proposal's confirmatory generalized mixed-effects
model with user random slopes and scenario random intercepts. The optional
[R mixed-effects harness](mixed-effects-analysis.md) now implements that exact
model, planned contrasts, source-digest checks, and convergence/singularity
diagnostics. Executing it on verified paper runs and reviewing any inferential
claim remain a separate statistical stage. A configured bootstrap count of
zero uses 200 replicates as an explicitly recorded smoke fallback, not a paper
default.

Human-derived evidence strength is not fabricated. The fitted ordering artifact
records that volunteered-control strength is unavailable until eligible
external judgments are imported. Separately,
`experiment-a-control-battery.json` fixes the proposal's volunteered, repeated
balanced, direct-correction, indifferent, randomized-choice, and
target-nondistinguishing protocols. It remains explicitly
`fixed_protocol_not_scored_by_one_step_choice_runner`: those signals are not
relabeled anchor choices.

The runner now materializes a separate content-addressed six-control plan,
transparent reference and no-update baseline reports, and a provenance-aware
provider-neutral request packet. Reference/baseline outcomes are labeled
diagnostic rather than external evidence. Imported provider responses are
scored only after exact request, prompt, plan, and stimulus binding through
`cape-loop control-study analyze`. See
[Experiment A control execution](experiment-a-controls.md).

H7's user-clustered volunteered valid-learning criterion is intentionally
separate from that single fixed protocol case. `control-study h7-plan`
exhaustively creates direct statements for every retained test
user/domain/attribute and crosses the ordinary full-context and
provenance-aware views. `h7-review` admits only complete accepted
OpenAI/OpenRouter evidence, converts the bound posteriors to
`VolunteeredPreferenceUpdate`, and recomputes H7 in a new immutable artifact;
`h7-verify` repeats the full computation. See
[H7 volunteered-preference controls](h7-volunteered-controls.md).

### Held-out paraphrase transfer and Gate 1

The held-out suite has train/development/test surface families and rejects any
family that crosses splits. Experiment A renders test cases from controlled
source trials without changing the selected option or visible context. It
evaluates the fitted-aware updater and, when configured, `llm_full_context`.
Every case and score carries content hashes tying the surface text to its source
and suite version.

Gate 1 checks aware/unaware Brier ordering by domain, a full-context gap,
mechanism transfer, domain coverage, and the held-out criterion. The held-out
criterion preserves `verified = null` when required case/updater pairs are
missing; it never treats missing LLM evidence as failure or success. Thus a
structured-only smoke run is incomplete, while a complete response corpus can
exercise the full computational check. `claim_status` remains `not_claimed` in
either case.

## Experiment B: closed-loop self-confirmation

Use:

```bash
PYTHONPATH=src python -m cape_loop run configs/closed_loop.toml
```

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

The four initial profile conditions are fixed in code:

```text
correct
incorrect
uncertain
empty
```

The checked-in configuration selects balanced, softly
profile-conditioned, and exploratory policies. Other known policies may be
selected by a different valid closed-loop config. The strict contract requires
the ranking/default/suggestion mechanism declaration, naturally sampled
responses, and a nonnegative bootstrap count. The checked-in confirmatory
configuration uses 2,000 replicates. A count of zero is permitted only to make
smoke and integration runs inexpensive; it emits point estimates marked
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

Turn records retain evaluated joint state when that updater has one and always
retain the exact shadow's joint state before and after the event. Terminal
evaluated/shadow joint states are retained separately.

The soft policy always displays both preference directions. Across turns it
rotates ranking, default, and suggestion channels and applies the
profile-consistent treatment with a probability determined by current profile
confidence.

### Decomposition and predicate

Balanced versus profile-conditioned shadow paths supply evidence-selection
comparisons. Evaluated belief versus its same-history shadow supplies
attribution comparisons.

The runner reports deterministic 95% percentile-bootstrap intervals for
evidence-selection cost, profile- and balanced-policy attribution costs, the
self-confirmation interaction, cumulative LCG, five-clause profile rate, and
later-action-influence rate. The primary interval resamples complete latent
users after reducing repeated domains and trajectories to equally weighted
user means. A complete paired-trajectory resampling is retained as a
sensitivity analysis. It must not be interpreted as making repeated
trajectories independent users. Fewer than eight user clusters is explicitly
marked `insufficient_clusters`.

These are paired, cluster-aware nonparametric intervals. They are not a GLMM or
a user-level mixed-effects model; the latter remains a separate proposal
analysis requiring an appropriate statistical environment.

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

`decoder-study plan-openai` estimates the two-source request budget without a
key. `decoder-study execute-openai ... --execute-live` is an optional collection
adapter; using two declared model roles still does not establish statistically
independent decoder errors.

### Artifacts

```text
events/experiment-b-trajectories.jsonl         # when retain_events = true
events/experiment-b-terminal-batteries.jsonl   # when retain_events = true
events/experiment-b-held-out-terminal-suites.jsonl # when retain_events = true
metrics/experiment-b-terminal.jsonl
metrics/experiment-b-native-decoders.jsonl
metrics/experiment-b-held-out-actions.jsonl
metrics/experiment-b-terminal-calibration.json
metrics/experiment-b-decomposition.jsonl
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

Each Experiment B terminal row includes profile Brier and projected behavioral
scores, same-history shadow-to-system marginal KL, preference-dimension
coverage and time-to-full-coverage, displayed-option diversity, distinct
selected-option count, profile-conditioned exposure rate, mechanism
count/evenness, cumulative action-aware information gain, total intrinsic
regret, and the explicitly defined false-stable attribute rate and trajectory
flag for incorrect-seed trajectories. It also includes top-label profile ECE
and fixed reliability bins over its three preference-attribute forecasts. The
pooled calibration artifact groups these records by public projection or
deterministic decoder and states that trajectory/user, not attribute, is the
dependence unit.

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
`closed_loop.toml` has no `llm_full_context` updater, so its target set is empty
and Gates 2 and 3 are incomplete.

With correctly configured replay or explicitly authorized live LLM updaters,
the code can compute the declared checks, but still records
`claim_status = "not_claimed"`. Gate 2 additionally requires adequate
user-clustered intervals whose lower bounds are above zero for both mean LCG
and the five-clause profile rate. Gate 3 requires an adequate user-clustered
same-history attribution interval whose lower bound is above zero. Missing
bootstrap evidence makes these gates incomplete; insufficient clusters or an
interval crossing zero makes the corresponding gate fail its computational
checks.

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
selected Anthropic/Gemini decoder collection and complete OpenAI native-action
collection to eligible trajectory IDs, recomputes Gate 4, and writes a separate
checksum-bound artifact. Every plan, physical attempt, accepted audit,
judgment/action record, execution manifest, and evidence-file digest is
checked under the collectors' shared locks. The explicit reviewed-generic
decoder alternative does not claim provider-collection provenance. The import
never rewrites the run's gate report. Ordinary Experiment B runs remain
incomplete until that import exists.

## Experiment C: evaluation validity

Use:

```bash
PYTHONPATH=src python -m cape_loop run configs/evaluation.toml
```

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

### Reproduction across random seeds

The separate offline reviewer admits two to 32 checksum-valid completed
Experiment C runs with distinct seeds and otherwise identical scientific
configuration/source identity. It compares all three point rankings,
open/closed inferential top tiers and partial orders, Gate 5 decision/status,
and ESR selection sets. It reports exact rational stability proportions and
every disagreement without pooling bootstrap draws or creating a claim. See
[Experiment C multi-seed robustness](experiment-c-robustness.md) for its CLI,
strict admission checks, artifact layout, and interpretation boundary.

## Sensitivity grid

Use:

```bash
PYTHONPATH=src python -m cape_loop run configs/sensitivity.toml
```

The compact checked-in `configs/sensitivity.toml` product is:

```text
3 decision-noise values
× 3 presentation multipliers
× 3 initial-profile strengths
× 3 trajectory lengths
= 81 points
```

All additional axes default to neutral singleton values, so this compact config
retains its original identity. `configs/sensitivity_full.toml` is the broader
384-point executable declaration:

```text
2 decision-noise values
× 1 shared presentation multiplier
× 2 rank multipliers
× 2 default multipliers
× 2 suggestion multipliers
× 2 initial-profile strengths
× 2 prior-uncertainty values
× 2 trajectory lengths
× (1 random-utility point + 2 rule-noise points)
= 384 points
```

Random-utility points do not carry a rule-noise value. Rule-based points are
repeated once per declared rule-noise value, keeping family identity explicit.

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

`configs/sensitivity_llm_openrouter_smoke.toml` is a one-point, two-domain
transport smoke configuration. It is not a robustness grid or paper result.
`experiment.turns` must be `1`; each point's actual trajectory length comes
only from `[sensitivity].trajectory_lengths`.

Artifacts are:

```text
models/sensitivity-fits.jsonl
metrics/sensitivity.jsonl
metrics/sensitivity-decomposition.jsonl
metrics/sensitivity-grand.jsonl
metrics/sensitivity-phase-points.jsonl
metrics/sensitivity-phase-domains.jsonl
metrics/sensitivity-phase-boundaries.jsonl
metrics/sensitivity-phase-specification.json
tables/sensitivity.csv
events/sensitivity-trajectories.jsonl  # when retain_events = true
llm/sensitivity-request-preflight.json # when an LLM updater is configured
llm/requests.jsonl                     # mandatory for LLM sensitivity
llm/responses.jsonl                    # when an LLM updater is configured
llm/provider-audit.jsonl               # live modes
llm/transport-attempts.jsonl           # live modes
metrics/gate-report.json
metrics/summary.json
```

Each model row contains raw and active fitted bundles, calibration, and
development diagnostics. `sensitivity.jsonl` and its CSV are stratified by
domain, policy, and updater; `sensitivity-decomposition.jsonl` contains paired
policy contrasts; both attribute-assessment and trajectory/profile rate
denominators are named explicitly; `sensitivity-grand.jsonl` is descriptive
only.

Phase criteria are declared by metric, relation, and threshold in the resolved
configuration. The fifth frozen criterion requires the phase-target users to
reject at least 20% of profile-consistent suggestions by default. The
opportunity and rejection counts are retained per point; a point with no
eligible suggestion remains incomplete rather than receiving a zero rate.
Boundary rows identify adjacent observed grid values where a criterion or
joint-region label changes while all other boundary axes are fixed. They are
observed-grid intervals, not interpolated causal thresholds.

The phase target prefers `llm_full_context`, then
`llm_provenance_aware`, `llm_response_only`, and the structured
`full_context_blind` proxy. Every grand row states the selected updater,
whether it is an external LLM, and whether execution was replay or live.

Gate 6 now reports the proposal's six clauses separately. Another-response-
model, broad-parameter, both-domain, and exact/fitted-reference clauses are
computed from completed point and domain phase rows. Multiple independent LLM
families and held-out paraphrases remain explicitly incomplete inside any one
run. Once complete live-model sensitivity and Experiment A runs exist,
`gate6-review build` can bind explicit family/source declarations to exact
provider evidence, recompute the held-out transfer, and aggregate all six
clauses in a separate checksum-bound artifact. Model labels or ordinary
wording templates are never promoted into family identity, independence, or a
paper claim.

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
memory. The authorization flag only prevents accidental early execution; it
does not prove that a scientific gate passed. A paper result requires a real
system adapter, retained evidence, and frozen review.

## Evidence still external

The repository contains no API keys or live model responses, genuinely distinct
external decoder judgments, recruited human participants, ethics determination,
full generalized mixed-effects fit, or paper results. Authors, repository/DOI,
and accepted-paper metadata are also unset. See
[External evidence boundaries](external-evidence.md).

No smoke, gate, packet, or checked-in configuration is an empirical result.
