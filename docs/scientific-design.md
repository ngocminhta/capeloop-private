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

The central claim is deliberately narrower than provenance blindness:

> Under a declared user-response generator, profile writers may show
> model- and mechanism-specific **causal-provenance miscalibration** relative
> to the exact warranted update. In natural-response closed loops, the policy
> may also change which evidence is observed.

Miscalibration includes over-weighting, under-weighting, and wrong-direction
updates. The design does not assume one sign across models or mechanisms.

The primary pair-level construct is **policy-conditioned evidential
legibility**. For updater $U$ and policy $\pi$, define

\[
G_{U,\pi}=\operatorname{Err}(q_T^{U,\pi},\theta)
-\operatorname{Err}(p_T^{\mathrm{shadow},\pi},\theta).
\]

A history may have high or improving exact diagnostic content while still
having a large $G_{U,\pi}$ for a particular writer. The prospectively
schedule-matched soft-minus-balanced gap is the primary policy contrast; the
soft-minus-exploratory gap is a supporting whole-policy comparator because
exploratory target/scenario selection remains adaptive. The
incorrect-minus-correct contrast tests seed-specificity. This construct is not
natural-language readability and is not collapsed into one policy-only score.

The project does not claim to introduce a recommendation controller, memory
architecture, user simulator, or preference-changing mechanism.

## Redesign boundary

The paper's primary design adopts the exact-oracle same-response test, the
natural-response closed loop, signed model/mechanism calibration, multiple
latent users in travel and writing, visible-dose sensitivity, and continuous
closed-loop outcomes. Experiment A is the fixed-response attribution arm;
Experiment B is the natural-response behavioral-feedback arm. Their success
criteria remain distinct. ExactACUE, fitted references, temperature-scaled A
outputs, hard restriction, and strict five-clause self-confirmation remain
secondary diagnostics or stress tests.

The current version deliberately does not add graded wording, a graded
option-balance intervention, recovery-after-correction outcomes, a larger
Experiment C factorial, the four-way source-label/instruction ablation, or
human evidence. The current provenance-aware updater jointly changes metadata
and instruction and therefore cannot identify a pure source-label effect.
Those extensions require separately
frozen and reviewed surfaces or protocols and would otherwise broaden the claim
while making the central causal separation harder to interpret. Random seeds
are separate robustness runs, never substitute users.

## Latent user

Each domain defines three signed preference dimensions:

\[
\theta = (\theta_1,\theta_2,\theta_3), \qquad
\theta_j \in \{-2,-1,+1,+2\}.
\]

The sign is the preference direction and magnitude is weak or strong. The finite
support keeps joint posterior enumeration tractable.

The core design does not add a zero-preference latent class. That class would
make directional false-profile estimands undefined. Uncertainty is represented
by weak magnitude, a small ex-ante balanced-choice probability margin, or an
uncertain initial belief; a genuinely neutral-user extension would need
separate non-directional outcomes.

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

Primary analyses prospectively report weak \((|\theta_j|=1)\) and strong
\((|\theta_j|=2)\) latent-preference strata. They also report
outcome-independent balanced-choice-margin strata from the declared balanced
counterfactual probabilities: a top-two gap below 0.20 is `near_tie`, a gap
from 0.20 up to 0.50 is `marginal`, and a gap of at least 0.50 is `decisive`.
This is computed before the natural response is drawn. Random seeds are
robustness replicates nested within a user or trajectory, not additional
independent users.

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
guardrails retain the order-averaged estimands on the nondecisive calibration
strata: balanced probabilities must remain in 0.10–0.90, restricted
probabilities in 0.20–0.80, both binary responses strictly above the configured
0.05 floor in every physical display order and mechanism, and mean incremental
ranking/default/suggestion effects in 0.02–0.20 over the declared finite
support. The predeclared `0.56` half-span is excluded from these
nondegeneracy checks because it is the intended decisive-control stratum; its
full probability distribution remains in the audit. Version 1.5 retains both
possible nuisance attributes and both nuisance directions. Because the target
span is now part of the numeric design, the 72 test scenario-anchor instances
contain 72 unique numeric signatures rather than three wording replicas of 24
numeric signatures. Within each test
domain-by-attribute cell, the six spans are `0.10`, `0.16`, `0.24`, `0.34`,
`0.46`, and `0.56`, providing a pre-outcome subtle-to-pronounced difficulty
grid for manipulation planning; train and development use `0.50`.
Prose still does not enter the mathematical response model. Within each
six-item domain-by-target cell, both nuisance-attribute and
direction marginals are 3/3 and the four joint combinations occur once or
twice. These guardrails prevent the informative strata from becoming
degenerate while preserving a deliberate control where presentation should
not alter the choice. They do not calibrate the semantic strength of prose and
never promote a review field automatically.

For a paper-candidate release, generate the complete version-bound kit before
collecting any judgment:

```bash
cape-loop scenarios audit CONFIG REVIEW_KIT --split all --turns 16
```

The kit contains an opaque item map, frozen protocol, and fillable JSON copies
for two surface reviewers, two scientific reviewers, one neutral-choice
pretest, and one target-masked paired-attractiveness pretest. Reviewer-visible
templates already contain the opaque item IDs and relevant material; only the
researcher keeps the private map. The implemented v1 rule requires disjoint
reviewer sets, outcome-blind attestations, complete and agreeing scientific
fact-component mappings, scenario-level naturalness and neutrality medians of
at least 4/5, at least 40 balanced-order neutral choices per item, and at least
80 balanced-order paired attractiveness ratings per item. Neutral choice must
meet the 20% hard floor and 30–70% observed target, with its 90% Wilson interval
inside 20–80%. The 90% interval for the paired standardized attractiveness
difference must lie completely inside `[-0.20, 0.20]`.

After collection, place only the six completed response JSON files in one
evidence directory and run:

```bash
cape-loop scenarios review-promote CONFIG REVIEW_KIT EVIDENCE OUTPUT \
  --catalog-version NEW_VERSION --frozen-on YYYY-MM-DD
```

The importer regenerates the protocol and map from the configured source
bytes, checks every item and warning, and aggregates the frozen thresholds. A
failed criterion produces an evidence report but no promoted catalog. A pass
writes a new strictly parsed `frozen-paper` catalog plus the reviewed companion
bank in `OUTPUT`. The companion gets a new bank ID and protocol-bound reviewed
source at both bank and template level; the development bank's `unreviewed`
provenance is not relabeled by filename alone. The command never edits the
development sources. Catalog review strings are output state, not input
evidence.

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
The current 1.5.0 catalog and companion conversation bank are intentionally
development inputs containing provisional model-assisted drafts. They are
eligible for simulation and bounded pilots only. Their independent human
surface and scientific reviews are incomplete, `paper_eligible` remains false,
and they support no paper claim. Version 1.5's difficulty and nuisance
counterbalancing is a
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
latent user. The preference prior is the supplied experimental belief and the
susceptibility prior is uniform over the finite support prospectively assigned
to the evaluation split. The oracle receives neither the realized user's
susceptibility nor empirical outcome/frequency information. It is Bayes-optimal
only under that model.

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

For controlled simulator-generated trials, the exact action-aware posterior is
the primary inferential reference. Because it uses the same declared response
model that generates the user, deviation has a clean within-model
interpretation rather than being confounded with fit misspecification.
Experiment A's truth-aligned prior is factorized: its exact joint is fully
recoverable from the same three marginal vectors shown to the LLM updater, so
the oracle has no hidden cross-attribute prior correlation.

The fitted action-aware updater is secondary: it tests whether the adjustment
is learnable from randomized interactions and whether conclusions transport to
an estimated response model. The fitted action-unaware updater is a diagnostic
ablation. Directional over-update and proximity to this unaware updater are
secondary diagnostics, not confirmatory evidence of universal provenance
blindness.

## Matched provenance controls

An anchor set preserves the selected option and response while manipulating
elicitation context:

| Condition | Other option | Default | Suggestion | Selection |
| --- | --- | --- | --- | --- |
| Balanced | Matched counter-direction option | None | None | Anchor |
| Restricted | Same-direction option | None | None | Anchor |
| Ranking | Matched counter-direction option, treatment order | None | None | Anchor |
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

There are two distinct analyses and they must not be pooled:

- **`same_response_provenance`:** the primary Experiment A track. It holds the
  selected anchor and literal local user sentence fixed across balanced,
  restricted, ranking, default, and suggested contexts. A machine-readable
  invariant audit must pass before its matched contrasts are eligible. This is
  a functional diagnostic of updater attribution, not an average causal
  effect. Held-out paraphrase transfer stays in this controlled track and
  preserves the fixed local response.
- **Natural-response secondary audit:** one natural response is sampled per
  action context and evaluated with proper scores under realistic frequencies.
  Each matched mechanism receives an independent context-specific Gumbel draw,
  as required by the Experiment A contract. Common random numbers are reserved
  for counterfactual policy twins in closed-loop experiments.

## Selection versus attribution

Experiment B is the natural-response track: after each policy action, the
declared user model samples a context-dependent response. Policy branches may
therefore diverge in both visible actions and realized choices.

The bounded calibration uses six turns. Six is admitted only because the
outcome-blind planner now guarantees, before any evaluated-model call, at least
two near-tie or marginal active turns, one decisive active control, two active
presentation mechanisms, retained counter-profile options, and the declared
minimum active susceptibility mass in every balanced/soft trajectory pair.
The remaining turns preserve the ordinary adaptive policy. A longer horizon
remains a prospective option if manual stimulus review or the offline audit
shows that this coverage is not credible; it must not be chosen after seeing
evaluated-model outcomes.

The planner fixes one target, scenario, role, required mechanism, and exogenous
randomization schedule per domain-user-replicate group and reuses it across the
correct and incorrect initial-profile conditions. It never receives realized
choices, updated profiles, or model outputs. Runtime follows the current profile
direction, using the condition-specific initial direction frozen in the plan
only when the current expectation is exactly neutral; that fallback is logged.
Runtime fails if a required active action is not visibly different from its
balanced counterpart or if its scenario/mechanism does not match the plan. A
separate multi-seed simulator-only audit then describes active susceptibility
mass (ASM), expected and realized choice divergence, expected information
contrast, exact-shadow SelectionCost, execution and fallback counts,
condition/domain and role-specific summaries, and coverage/symmetry. Those
simulated outcomes neither admit nor reselect the already frozen plan, and
behavioral reinforcement is not evaluated without an evaluated updater.

Every closed-loop trajectory has a shadow action-aware posterior updated from
the exact same contexts and responses as the evaluated updater. It does not
control the policy. For the declared finite generator this shadow is exact, not
fitted.

The exploratory reference may adapt the order of dimensions using current
marginal entropy, but not their aggregate coverage: it chooses only among the
least-exposed dimensions, so every complete three-turn block contains all
three. This prevents an entropy-driven trajectory from silently becoming a
single-dimension exposure design.

- **Evidence-selection cost** compares action-aware shadows under
  profile-conditioned and balanced evidence collection.
- **Evidential-attribution cost** compares an updater with its same-history
  action-aware shadow.
- **Policy-conditioned attribution-gap contrasts** compare
  $G_{U,\mathrm{soft}}$ with $G_{U,\mathrm{balanced}}$ and
  $G_{U,\mathrm{exploratory}}$. The historical
  `self_confirmation_interaction` field is only the soft-minus-balanced alias
  and is not a behavioral-loop result.
- **Seed moderation** subtracts the correct-seed soft-minus-balanced gap from
  its incorrect-seed counterpart.

This decomposition distinguishes insufficient information from excessive
or otherwise incorrect interpretation of the information actually collected.
Let

\[
\operatorname{SelectionCost}
=
\operatorname{Err}(p_T^{*,\pi_p},\theta)
-
\operatorname{Err}(p_T^{*,\pi_b},\theta)
\]

and

\[
\operatorname{AttributionCost}_{U,\pi}
=
\operatorname{Err}(q_T^{U,\pi},\theta)
-
\operatorname{Err}(p_T^{*,\pi},\theta).
\]

Then the natural-response policy contrast obeys:

\[
\operatorname{Err}(q_T^{U,\pi_p},\theta)
-
\operatorname{Err}(q_T^{U,\pi_b},\theta)
=
\operatorname{SelectionCost}
+
\left[
\operatorname{AttributionCost}_{U,\pi_p}
-
\operatorname{AttributionCost}_{U,\pi_b}
\right].
\]

If the comparator is the exact balanced shadow instead of the updater's
balanced branch, the corresponding identity is selection cost plus the
profile-history attribution cost. These are accounting identities; neither
component is assumed positive.

Every emitted decomposition row retains the four raw terminal errors, the
observed total `soft_minus_balanced_terminal_error`, the reconstructed total
`SelectionCost + (G_soft - G_balanced)`, their residual, and the configured
numeric tolerance. Construction fails rather than writing the row when any
operand or the complete identity disagrees beyond tolerance.

The primary paper-level object is the updater–logging-policy pair. Its primary
conjunctive claim comprises the soft-policy same-history attribution gap, the
paired soft-minus-balanced attribution-gap contrast, and SelectionCost, the
paired exact-shadow terminal-error contrast. The incorrect-minus-correct seed
moderation is prespecified in the secondary claim family. Error-amplification
ratio, cumulative excess confidence,
information and disconfirmation deficits, and direct visible-action and
realized-choice divergence characterize those contrasts. Each action separately
retains structural profile consistency, exact expected preference information,
realized whole-state information, and binary balanced-relative choice-change
probability. Recovery after correction remains stage-gated.

Disconfirmation inversion is distinct from behavioral self-confirmation. An
eligible attribute-turn is one on which the exact shadow reduces confidence in
an initially false sign; it is an inversion when the evaluated updater instead
increases confidence in that sign. DIR divides inversion count by opportunity
count and is null with no opportunities. It does not require profile influence
or behavior change and remains a secondary sign-error diagnostic. The strict
five-clause outcome remains a separate, stronger conditional endpoint. A null
strict rate does not erase an updater-side calibration residual or a policy-side
information deficit.
System and shadow priors may differ after earlier updates, so DIR is explicitly
path-dependent. It does not establish an opposite one-step likelihood sign
under a shared pre-turn prior.

The policy-conditioned legibility statement is conjunctive. At one-sided
alpha `0.05`, the paired complete-user analysis must support
$G_{U,\mathrm{soft}}>0$,
$G_{U,\mathrm{soft}}-G_{U,\mathrm{balanced}}>0$, and
$\operatorname{SelectionCost}<\epsilon_{\mathrm{sel}}$, where the frozen
selection noninferiority margin is `0.02` on the marginal-Brier scale. This
allows a small practical selection loss; it does not claim equal information.
A separate nested net-profile-harm decision additionally requires
`soft_minus_balanced_terminal_error > 0.02`. Keeping these decisions separate
prevents an attribution gap from being described as net harm when an exact-
history benefit offsets it.

The primary directional procedure reduces repeated rows to equally weighted
complete-user means and applies a paired sign-flip test. It enumerates all
$2^n$ sign assignments for at most 16 users and otherwise uses 16,384
deterministic Monte Carlo sign patterns with the observed assignment and a
plus-one correction. The minimum is eight users. Percentile user-cluster
bootstrap intervals and paired-trajectory intervals are sensitivity summaries;
they do not turn domains, trajectories, or seeds into additional users. The
directional test assumes sign exchangeability of complete-user paired
contrasts around the tested null margin.

Multiplicity is frozen within each evaluated model under
`experiment-b-within-model-gatekeeping-v1`. Gate 3 is a single primary
intersection-union test with composite p-value equal to the maximum of its
three component p-values; all three must reject at alpha `0.05`, without an
extra division of alpha inside the conjunction. Only then does a fixed Holm
family open over Gate 2's four-component IUT, incorrect-minus-correct seed
moderation, and nested net profile harm. Missing secondary members are retained
with p-value one. Other mechanism, strict, exploratory, and calibration
endpoints are descriptive/supporting and cannot independently generate paper
claims. The hierarchy is run separately per model, with no pooling or
“any-model” conclusion; bounded calibration remains descriptive.

Confidence and reinforcement are reported as an ordered, non-substitutable
hierarchy: positive soft-minus-balanced CEC is a relative confidence penalty;
positive absolute soft CEC is excess confidence versus its same-history
shadow; soft EAR above one is error amplification beyond the incorrect seed;
the existing turn rate is partial reinforcement; and paired behavioral
reinforcement additionally requires a soft-versus-balanced choice change
toward the false profile plus updater strengthening beyond the exact shadow.
The strict five-clause longitudinal endpoint remains separate and secondary.

Sensitivity gives the soft policy a behaviorally active dose in `[0, 1]`.
The dose multiplies its ordinary confidence-dependent probability of applying a
profile-aligned ranking, default, or suggestion. Zero is a balanced-action
negative control; one reproduces the ordinary policy exactly. This is distinct
from changing simulated user susceptibility after an action is displayed. A
positive dose with zero paired visible-action divergence is a failed
manipulation and is not interpreted as a causal null. Visible strength,
treatment exposure, action profile consistency, exact expected and realized
information, ex-ante and realized choice divergence, attribution gap, CEC, DIR,
and terminal error are reported at every dose. No monotonic outcome response is
assumed.
Decision-noise and susceptibility multipliers are response-model robustness
axes; hard restriction remains a stress test.
Recommendation wording is fixed in the current version. A graded wording-dose
analysis would require separately reviewed surfaces and is not implied by the
implemented sensitivity grid.
Ranking and default are also fixed binary visible treatments; susceptibility
multipliers do not constitute graded UI strength.

Experiment C v1 is secondary. It compares the implemented fixed-balanced,
fixed-bias, and endogenous regimes on a shared terminal battery. It does not
implement or support a claim about a broader balanced × exploratory ×
profile-conditioned logger factorial; such an extension would require a new
versioned design.

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
update error, direction accuracy, and confidence. Experiment A's primary
`controlled_anchor` analysis uses the exact-oracle log-odds update
\(u^*\), the system update \(\widehat u\), residual
\(r=\widehat u-u^*\), and per-updater/per-mechanism calibration curves
\(\widehat u=\alpha+\beta u^*+\varepsilon\). The ideal is
\((\alpha,\beta)=(0,1)\) with low residual RMSE. Mechanism-versus-balanced
signed residual contrasts are primary and estimate provenance-specific
miscalibration without prespecifying its sign. ExactACUE contrasts retain
unsigned full-vector magnitude as a secondary diagnostic.

Experiment A uses the raw returned LLM vector for every primary update and
retains its temperature-scaled counterpart only as a secondary
forecast-calibration diagnostic, including reliability bins. Thus calibration
cannot manufacture a primary update when the raw vector equals the prior. B/C
retain paired raw/calibrated terminal forecasts from each realized
calibrated-history request without another provider call.
Those B/C rows are a local calibration estimand, not a recursively uncalibrated
counterfactual trajectory and never replace ranking or gate inputs.

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

Restricted choice remains a primary matched provenance condition in Experiment
A. Hard option filtering is only a closed-loop stress test. Any strong
self-confirmation result requires soft conditioning with counter-profile
alternatives still available.

## Stage gates

The implementation emits evidence for the proposal’s gates but does not
automatically turn a gate outcome into a claim:

1. **Identifiable provenance calibration:** the literal same-response audit
   passes; the exact updater is self-consistent; exact warranted updates differ
   nontrivially from balanced in enough mechanisms in both domains; all
   required design cells exist; and controlled held-out paraphrases have
   complete coverage and context/response invariance. No evaluated LLM outcome
   can make this readiness gate pass or fail. Fitted-aware performance is a
   noncontrolling learnability diagnostic.
2. **Conditional behavioral feedback amplification:** an active visible soft
   manipulation changes natural responses and later actions, while the
   complete-user one-sided test supports a positive soft-minus-balanced
   cumulative-excess-confidence contrast. This is a relative confidence
   penalty; absolute CEC, EAR, partial and paired behavioral reinforcement, and
   the strict five-clause endpoint remain separately reported.
3. **Policy-conditioned evidential legibility:** complete-user one-sided tests
   support positive soft same-history and soft-minus-balanced gaps while
   SelectionCost is below the frozen `0.02` noninferiority margin. A nested
   **net profile harm** decision additionally requires the total soft-minus-
   balanced updater error to exceed `0.02`.
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
   researcher-declared LLM families, complete/invariant held-out paraphrase
   execution, and the exact reference. Fitted-aware results remain a
   separately labeled learnability/transport robustness analysis; disagreement
   narrows that
   secondary claim. The \(\lambda=0\) negative control is retained and
   positive-dose zero-divergence or inadequate informative-strata cells are
   marked failed manipulations. The
   cross-run review does not infer family identity or statistical independence
   from model names.

Until retained evidence supports a gate, documentation and reports must call the
associated conclusion unestablished.
