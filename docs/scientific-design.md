# Scientific design

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

The structured choice is sampled before any natural-language verbalization.
Verbalizers may paraphrase the chosen action but may not add unsupported general
preference statements.

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

Systems keep their native representation. The current evaluator uses two
distinct fixed deterministic blinded projections and a shared exogenous
terminal battery. Battery choices are produced from each projected belief, not
through a separate native-system action API. The proposal's independent
decoder judgments and native end-to-end action evaluation therefore remain
future empirical work. Gate 4 encodes both as incomplete prerequisites: local
decoder projections and structured/persona action references are never
accepted as substitutes.

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
4. **Native validity:** failure appears in an inspectable persistent loop and
   is supported by imported independent blinded decoders plus genuine native
   end-to-end terminal actions.
5. **Evaluation implication:** a joint-paired complete-user reversal or
   inferential-top-tier selection regret clears its declared threshold.
6. **Robustness:** transfer across response models, parameters, domains, model
   families, and paraphrases.

Until retained evidence supports a gate, documentation and reports must call the
associated conclusion unestablished.
