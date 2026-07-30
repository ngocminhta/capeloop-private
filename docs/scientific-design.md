# Scientific design

This page is the concise implementation-facing scientific contract. The
paper-facing narrative remains in [the proposal](proposal.md), exact formulas
and decision quantities are in [Metrics](metrics.md), executable study and
review procedures are in [Experiments](experiments.md), and provider operations
are in [Live execution](live-execution.md). None of those documents treats
software completion as an empirical result.

## The claim

CAPE-Loop studies **performative observation**, not changing preferences. A
latent user remains fixed within a trajectory:

\[
\theta_{t+1} = \theta_t = \theta.
\]

The current persistent profile can nevertheless influence which evidence will
be observed:

\[
\text{profile}
\rightarrow
\text{agent action}
\rightarrow
\text{policy-conditioned response}
\rightarrow
\text{profile update}.
\]

The audit asks whether the updater conditions its evidential weight on that
elicitation process. A user response is an observation, but it is not generally
independent of the policy that produced the choice environment.

The project does not claim to introduce a recommendation controller, memory
architecture, user simulator, or preference-changing mechanism.

## Latent user

Each domain defines three signed preference dimensions:

\[
\theta = (\theta_1,\theta_2,\theta_3), \qquad
\theta_j \in \{-2,-1,+1,+2\}.
\]

The sign is the preference direction and magnitude is weak or strong. The finite
support keeps joint posterior enumeration tractable.

Each simulated user also has presentation susceptibility:

\[
\psi =
\langle
\psi_{\text{rank}},
\psi_{\text{default}},
\psi_{\text{suggest}}
\rangle.
\]

Susceptibility changes choice probability. It does not enter intrinsic welfare.

Official configurations use prospective orthogonal split/allocation policies
for both latent components. The 64 theta profiles are partitioned into 32
train, 16 development, and 16 test profiles with balanced coordinate levels
and coordinate pairs. The 27 susceptibility profiles are partitioned into nine
profiles per split with the same strength-two balance. Deterministic blocked
orders keep each coordinate's realized counts within one user for incomplete
blocks. When both v2 policies are active, a deterministic outcome-blind joint
block search also reduces cross-coordinate categorical and linear association
over the official sample horizons while preserving all marginal guarantees.
This avoids letting an arbitrary index pairing distort marginal presentation
effects, but it cannot make very small cross-tables independent. This is
controlled design balance, not an empirical claim about the prevalence or
independence of preferences and susceptibility in people.

## Context and provenance

The visible context is:

\[
C_t =
\langle
\mathcal X_t,r_t,d_t,w_t,s_t,q_t
\rangle,
\]

where the fields are displayed options, ranking, default, wording, explicit
suggestion, and question type.

Policy provenance is a separate record:

\[
P_t =
\langle
\pi_t,M_t,\text{policy version},\text{random key}
\rangle.
\]

This boundary serves two purposes:

1. `InteractionContext` contains factors that may affect the response.
2. `PolicyProvenance` explains why the agent created that context and whether
   the evidence may be self-generated.

A full-context profile writer receives \(C_t\), not hidden policy internals. A
provenance-aware condition receives a declared structured view of \(P_t\).

## User response

The primary response model is multinomial random utility:

\[
P(Y_t=x\mid\theta,\psi,C_t)
\propto
\exp\left[
\beta\theta^\top\phi(x)+\delta_\psi(C_t,x)
\right].
\]

Here \(\phi(x)\) is the intrinsic option feature vector and
\(\delta_\psi(C_t,x)\) contains position, default, and suggestion effects.

Intrinsic welfare is evaluated separately:

\[
u_\theta(x)=\theta^\top\phi(x).
\]

Consequently, a strong default may increase selection probability without
increasing user welfare. Regret compares intrinsic utility with the complete
feasible option pool, not only the displayed set.

The primary runtime is hybrid:

```text
fixed latent user
  -> mathematical response model selects an option
  -> frozen neutral base renders the assistant turn
  -> code states "I choose {selected_name}."
  -> evaluated profile writer updates its profile
```

The structured choice is therefore sampled before natural-language
verbalization. The authoring model never receives latent truth and never chooses
for the user. A model-assisted workflow may draft candidate base presentations
and display names, but the current 48 visible bases were subsequently
standardized by the project, outcome-blind, onto three source-neutral frame
families. Each frame appears exactly 16 times and twice within every
six-scenario test domain-by-target cell; every template has source
`project-standardized-neutral-frame-v1-unreviewed`. Balanced, restricted, and
ranking share the selected neutral frame. Code adds only the fixed
default/suggestion sentence and fixes the user reply. Runtime does not call an
authoring model for experimental rows. Frame balance is a design control, not
human validation of the prose.

For example, a restricted lodging context can be rendered as:

> **Assistant:** Here are two lower-cost hotel options for your trip. Hotel A
> is a standard room in a mixed-use neighborhood. Hotel B is a standard room
> in a quiet outer neighborhood. Which would you like?
>
> **User:** I choose Hotel A.

Full-context and provenance-aware evaluated models see this dialogue and a
semantic codebook for the domain attributes. They do not see numeric feature
vectors or the target index. Response-only deliberately omits the assistant
turn and unselected option as an ablation.

Internal catalog option IDs are control-plane identifiers. They may contain
split, attribute, and direction tokens for auditing, so they must never enter
an evaluated-model prompt. Runtime projects them to
`presented_option_1`, `presented_option_2`, and so on. Likewise, the concrete
display-name stem comes from the frozen bank, but A and B are assigned by
visible position after ranking rather than permanently attached to preference
direction. A matched provenance set keeps the same mapping; counterbalanced
orders exchange it.

## Scenario catalog and quality policy

Official configurations bind the versioned
[`cape-loop-scenarios-v1`](../data/scenarios/scenario-catalog-v1.json)
catalog before constructing a run. A scenario is an experimental stimulus:
it supplies a plausible low-stakes task, controlled option surfaces, declared
feature vectors, and matched same-direction alternatives. It does not supply a
latent user, response, preferred answer, policy treatment, or result.

A scenario is acceptable only when all of the following hold:

- every visible distinction is represented by a declared feature or is held
  constant, the target direction and three-coordinate feature order are
  correct, and the same anchor is unchanged across matched mechanisms;
- the balanced pair opposes the target direction, each restricted peer keeps
  that direction while varying only its explicitly declared nuisance
  coordinate and sign, and no option is objectively dominant or implausible;
- the prompt and labels are grammatical, choice-neutral, similarly specific,
  and contain no profile, hypothesis, split, mechanism, popularity,
  recommendation, moral, prestige, or expected-answer cue;
- the stimulus contains no real-person data, real brand dependency,
  time-sensitive factual claim, or copied third-party text;
- the complete scenario family, including revisions, translations, and
  paraphrases, belongs to one split, with no exact or near-duplicate visible
  surface in another split; and
- it passes structural, feature, identifier, split, surface-overlap,
  probability-eligibility, and planned-coverage checks.

The travel dimensions are declared preference **bundles**, not atomic causal
attributes: budget versus premium tier, central/compact versus
quieter/roomier setting, and coordinated convenience versus self-directed
flexibility. A stimulus or paper statement must use that scope. It must not
interpret the current `price`, `setting`, or `planning` key as an isolated
effect of money, distance, noise, or service quality. Writing dimensions are
length, tone, and spelling style, but their contextual appropriateness must
still be reviewed.

The present writing stimuli expose these categories directly in option
descriptions rather than presenting complete candidate passages. This supports
a narrow, high-internal-validity test of whether profile updates condition on
the causal context around declared preference evidence. It does not support a
broad claim about subtle style inference, natural drafting, or comparative
text quality. Any such claim requires a separately versioned and split-disjoint
excerpt robustness bank that passes the same blinded review and pretest before
its outcomes are inspected.

Before any scientifically interpretable pilot, freeze these outcome-independent
calibration rules:

1. Two independent reviewers map every visible fact to the target bundle, the
   single declared nuisance bundle, or a fact held constant. Every disagreement
   and cross-loading must be resolved; an unmodeled fact rejects the stimulus.
2. In a separate blinded stimulus check, at least 90% of judgments must recover
   the intended target direction and no more than 10% may assign either option
   to an unintended dimension.
3. After target language is masked, option descriptions must have a word-count
   ratio between 0.85 and 1.15, equal fact counts, and no unmatched
   recommendation, prestige, superlative, or quality cue. Neutral-attractiveness
   ratings must fall inside a preregistered equivalence margin; the recommended
   standardized margin is 0.20.
4. A neutral choice pretest must give every balanced option at least 20% of
   choices as a hard dominance screen. For primary calibrated stimuli,
   preregister a stronger 30–70% compatibility target with an uncertainty or
   equivalence rule and a sample size chosen for that rule. The 20% floor alone
   is not evidence of semantic equivalence and is not a requirement for an
   exact 50/50 result.
5. Exhaustive rendering across scenario, mechanism, order, anchor direction,
   and selected option must have zero punctuation, article, placeholder, or
   treatment-isolation error. Independent surface reviewers must rate
   naturalness and neutrality at least 4/5 at the scenario median.
6. Token Jaccard at or above 0.60 over scenario-specific prompts and option
   facts is a conservative machine trigger for blinded cross-split
   near-duplicate adjudication; 0.65 is the corresponding within-selected-split
   redundancy trigger. Neither threshold is automatic rejection. Shared
   conversation scaffolding and position-assigned names are excluded. Semantic
   families, revisions, and paraphrases remain in one split.
7. Scenario allocation is without replacement within each trajectory until a
   domain-by-attribute cell is exhausted. A horizon of \(T\) turns therefore
   needs at least \(\lceil T/3\rceil\) scenarios in every used cell for the
   declared three-attribute cyclic schedules. The v2 exploratory policy has the
   same bound: it selects among the least-exposed dimensions and therefore
   covers all three once per complete three-turn block. A custom or unknown
   policy that can repeatedly target one attribute conservatively needs \(T\)
   per cell. The present six test scenarios per cell support the maximum
   16-turn cyclic and block-balanced exploratory references without reuse.
   This capacity result does not approve their content.
8. Equivalent numeric feature magnitudes require independent human evidence
   that visible semantic contrasts have acceptably similar choice log-odds.
   The evaluated models and A/B/C outcomes may not supply this calibration.
   Otherwise, use prospectively validated scenario-specific strengths or
   narrow the claim to fixed-catalog performance.
9. Every human-review or pretest artifact must bind the exact catalog and
   conversation-bank digests plus scenario IDs and revisions. Before collecting
   judgments, preregister rater allocation, minimum sample size or precision
   target, randomization, exclusions, aggregation unit, uncertainty interval,
   and the rule for revisions. Point thresholds alone are not a sample-size
   plan, and additional ratings or replacement stimuli may not be chosen after
   inspecting evaluated-model outcomes.

The command

```bash
cape-loop scenarios audit CONFIG OUTPUT_DIR --split test --turns 16
```

implements the outcome-free structural, capacity, rendered-surface, overlap,
and simulator checks and writes a complete human-review packet. Its simulator
guardrails retain the legacy order-averaged estimands, requiring balanced
probabilities in 0.10–0.90 and restricted probabilities in 0.20–0.80. They
also require both binary responses to remain strictly above the configured
0.05 floor in every physical display order and mechanism, and require mean
incremental ranking/default/suggestion effects in 0.02–0.20 over the complete
declared finite support. Version 1.4 explicitly varies both possible nuisance
attributes and both nuisance directions. The 72 test scenario-anchor instances
therefore contain 24 unique numeric signatures, each represented three times.
Prose still does not enter the mathematical response model. Within each
six-item domain-by-target cell, both nuisance-attribute and
direction marginals are 3/3 and the four joint combinations occur once or
twice. These guardrails prevent a degenerate or forcing simulator, but they do
not calibrate the semantic strength of prose and never promote a review field
automatically.

Reject a candidate when a feature mapping is ambiguous, a salient difference
is unmodeled, one option is a generally better version of the other, treatment
is encoded in the wording, cross-split reuse is plausible, or content was
chosen or edited after inspecting test outcomes. Rejection must not be hidden
by silently drawing a replacement that produced a more favorable result.

LLMs may draft candidate scenario wording and one neutral conversation base
from a locked semantic specification. A candidate conversation author supplies
only neutral display names and a base presentation containing the declared
placeholders. Code supplies all default/suggestion text and the exact
`I choose {selected_name}.` reply after the mathematical simulator chooses.
The author may not assign the split, generate latent users, choose an option,
write treatment-specific language, explain a choice, evaluate itself, or
approve its own stimulus. The historical OpenRouter authoring command and
readable `.generation.jsonl` record candidate inputs and responses; they are
not evidence that the provider authored the current standardized visible text,
and the workflow is never rerun per experimental row. Project-produced
candidate or standardized wording must carry explicit authorship and edit
provenance without fabricated provider logs. Record the interface/provider,
exact model when exposed, edits, standardization, and unavailable provenance
explicitly. Never record credentials.

The review sequence is:

1. assign the whole semantic family to a split and lock its feature contract;
2. draft the candidate independently of experiment outcomes;
3. run automated schema, invariant, overlap, probability, and coverage checks;
4. obtain a surface review for naturalness and neutrality;
5. obtain a scientific review for feature alignment, tradeoff validity, and
   non-dominance; and
6. freeze the reviewed bytes and checksum before confirmatory or
   paper-evidence collection.

Reviewers receive the semantic specification and rendered packet, but not
evaluated-model identities, prompts, responses, or experiment results.
Scenario authors and conversation-author models cannot be their own sole
reviewers. All rejected candidates and revisions remain in the review record;
do not redraw stimuli until a preferred hypothesis result appears.

A scientific-content change requires a new scenario revision and catalog
version, a new freeze, and a new checksum-bound run. Paper eligibility requires
the automated and human reviews; machine validation alone is insufficient. A
reviewed release uses the coherent catalog state `frozen-paper` plus
`paper-eligible`, which the strict loader accepts only when every scenario is
approved and all recorded reviews pass.
The current 1.4.0 catalog and companion conversation bank are intentionally
development inputs containing provisional model-assisted drafts. They are
eligible for simulation and bounded pilots only. Their independent human
surface and scientific reviews are incomplete, `paper_eligible` remains false,
and they support no paper claim. Version 1.4's nuisance counterbalancing is a
machine-checked design property only; it is not human evidence of feature
alignment, naturalness, neutrality, non-dominance, or equivalent semantic
strength. Exact fields and the run binding are documented in
[Data model](data-model.md#scenario-catalog-input).

## Inference references

### Exact action-aware posterior

The exact reference updates the joint state:

\[
p_{t+1}(\theta,\psi)
\propto
p_t(\theta,\psi)
P(Y_t\mid\theta,\psi,C_t).
\]

It knows the declared simulator family and coefficients, but not the realized
latent user. It is Bayes-optimal only under that model.

### Fitted action-aware updater

The fitted aware reference learns a conditional response model from randomized
training interactions while receiving the choice set and presentation context.
It tests whether provenance adjustment is learnable without simulator
coefficients.

### Fitted action-unaware updater

The four-parameter unaware reference is fit to the same observations but cannot
condition on the option set, ranking, default, or suggestion. This is
parameter-count matching, not equality of function class or outcome space: the
aware model is conditional over displayed options, while the unaware model
predicts a binary semantic direction. It models the specific information loss
that CAPE-Loop audits.

The primary practical comparison is against the fitted action-aware reference.
The exact posterior remains a diagnostic upper reference.

## Matched provenance controls

An anchor set preserves the selected option and response while manipulating
elicitation context:

| Condition | Other option | Default | Suggestion | Selection |
| --- | --- | --- | --- | --- |
| Balanced | Matched counter-direction option | None | None | Anchor |
| Restricted | Same-direction option | None | None | Anchor |
| Default | Matched counter-direction option | Anchor | None | Anchor |
| Suggested | Matched counter-direction option | None | Anchor | Anchor |

Anchor identity and features must be byte-for-byte or value-for-value invariant.
The held response must have probability above the configured threshold under
every context. This avoids making the audit depend on nearly impossible
counterfactual observations.

Experiment A pairs scenario and physical order prospectively. Within each
user–domain–target cell, both anchor directions use the same catalog scenario,
and the two cases put their anchor in opposite display positions. Mechanism,
response-mode, prior-strength, and updater comparisons reuse that assignment.
Across users, the deterministic scenario cycle keeps counts within each
anchor-direction cell balanced to within one.

There are two distinct analyses:

- **Controlled identical-response audit:** a functional diagnostic of update
  sensitivity, not an average causal effect.
- **Naturally sampled audit:** one marginally correct response is sampled per
  action context and evaluated with proper scores under realistic frequencies.
  Each matched mechanism receives an independent context-specific Gumbel draw,
  as required by the Experiment A contract. Common random numbers are reserved
  for counterfactual policy twins in closed-loop experiments.

## Selection versus attribution

Every closed-loop trajectory has a shadow action-aware posterior updated from
the exact same contexts and responses as the evaluated updater. It does not
control the policy.

The exploratory reference may adapt the order of dimensions using current
marginal entropy, but not their aggregate coverage: it chooses only among the
least-exposed dimensions, so every complete three-turn block contains all
three. This prevents an entropy-driven trajectory from silently becoming a
single-dimension exposure design.

- **Evidence-selection cost** compares action-aware shadows under
  profile-conditioned and balanced evidence collection.
- **Evidential-attribution cost** compares an updater with its same-history
  action-aware shadow.
- **Self-confirmation interaction** asks whether updater blindness adds more
  error under a profile-conditioned policy than under a balanced policy for
  initially wrong profiles.

This decomposition distinguishes insufficient information from excessive
interpretation of the information actually collected.

## Definition of false self-confirmation

An episode is counted only if all five conditions hold:

1. the profile remains materially wrong;
2. probability mass on the seeded wrong direction increases;
3. cumulative confidence gain exceeds the same-history action-aware gain by a
   declared threshold;
4. the strengthened profile affects a later action;
5. the shadow posterior does not acquire equivalent confidence.

Persistence of a false seed alone is not self-confirmation.

For the descriptive **false-stable rate**, an initially incorrect attribute
counts as false-stable when its terminal wrong-direction mass remains above
`materially_wrong_mass` and no retained turn's wrong-direction mass departs from
the initial value by more than `false_stability_tolerance`. This longitudinal
persistence diagnostic is reported separately and never satisfies the
five-clause definition by itself.

## Evaluation tracks

### Track A: structured belief

Updaters expose attribute-level probability distributions. The track supports
Brier score, negative log likelihood, posterior divergence, action-conditioned
update error, direction accuracy, and confidence. Experiment A retains and
scores raw and active fitted/LLM forecasts separately, including reliability
bins. B/C retain paired raw/calibrated terminal forecasts from each realized
calibrated-history request without another provider call. Those B/C rows are a
local calibration estimand, not a recursively uncalibrated counterfactual
trajectory and never replace ranking or gate inputs.

### Track B: native persistent memory

The executable native families are `episodic_memory`, `semantic_memory`, and
`provenance_linked_memory`. Systems keep their native representation, including
the policy-facing persona projection where present. The ordinary offline
evaluator uses two distinct fixed deterministic blinded projections and a
shared exogenous terminal battery.

Those local projections are diagnostics. Gate 4 instead requires two external
evidence streams:

1. the exact retained terminal state is decoded blindly by
   `anthropic/claude-sonnet-5` and `google/gemini-3.6-flash` through the
   selected OpenRouter collection; and
2. `cape-loop-openai-native-agent-v1`, backed by OpenAI `gpt-5.6-sol`, receives
   the complete retained native state and held-out suite and returns one bound
   terminal action per item.

Plans, per-model reasoning settings, responses, physical-attempt journals,
provider audits, state/suite bindings, and imports are hash-bound and
validated. No eligible paper corpus is checked in. Both decoder families share
one gateway, so first-party decoder origin, distinct transport origins, and
statistical independence are not claimed. Gate 4 remains incomplete until the
collections are executed and a responsible researcher accepts the sources for
the stated claim. Direct Anthropic/Gemini adapters remain optional
first-party-origin replications.

## Positive and negative controls

Positive controls include explicit volunteered preferences, repeated balanced
cross-context choices, and direct correction. Negative controls include
indifference, random choices, and responses that do not identify the target
dimension.

`experiment-a-controls-v1` content-binds all six constructions and their
expected diagnostics. The one-step choice runner deliberately leaves their
outcomes unscored when a faithful direct-statement, longitudinal,
randomized-response, indifference, or real-system correction executor is not
present; it does not relabel anchor choices as these controls.

Hard option filtering is a stress test. The primary self-confirmation result
requires soft conditioning with counter-profile alternatives still available.

## Stage gates

The implementation emits evidence for the proposal’s gates but does not
automatically turn a gate outcome into a claim:

1. **Learnable provenance gap:** aware must outperform unaware and practical
   writers must leave material room.
2. **Nontrivial soft self-confirmation:** effect under soft presentation with
   positive excess confidence and downstream action influence.
3. **Attribution beyond selection:** updater worse than its same-history shadow.
4. **Native validity:** a matched native failure appears in an inspectable
   persistent loop; every eligible state has imported blind judgments from two
   responsibly reviewed genuinely distinct decoder sources and hash-bound
   actions emitted directly by the native system. Shared-gateway metadata is
   never promoted to first-party origin or independence.
5. **Evaluation implication:** either a joint-paired complete-user analysis
   resolves a reversal and its regime shift, or the minimum inferential-top-tier
   selection regret and conservative paired-test interval envelope clear the
   declared practical threshold. Descriptive rank disagreement cannot pass the
   gate.
6. **Robustness:** verified sensitivity/Experiment A pairs cover another
   response model, broad parameters, both domains, at least two
   researcher-declared LLM families, held-out paraphrases, and exact/fitted
   aware references. The cross-run review does not infer family identity or
   statistical independence from model names.

Until retained evidence supports a gate, documentation and reports must call the
associated conclusion unestablished.
