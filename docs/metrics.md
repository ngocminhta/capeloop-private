# Metrics

This document defines the intended quantities and their interpretation. The
current runner emits wide, experiment-specific metric rows rather than a
generic name/version envelope; their concrete keys are documented in
[Data model](data-model.md). A formula change therefore requires an explicit
artifact/schema or software-version boundary and must not be applied silently
to an existing result.

Let \(J=3\) be the number of preference attributes and
\(\mathcal V=\{-2,-1,+1,+2\}\). Let \(q_j(v)\) be an updater’s marginal belief
and \(\theta_j\) the latent value. Let \(p^A\) denote the fitted action-aware
reference unless a metric is explicitly labeled exact-oracle.

## Structured profile error

### Marginal multiclass Brier score

\[
\operatorname{BS}(q,\theta)
=
\frac{1}{J}
\sum_{j=1}^{J}
\sum_{v\in\mathcal V}
\left(q_j(v)-\mathbf 1[\theta_j=v]\right)^2.
\]

Lower is better. The averaging convention and support are stored with the metric
version. Brier is the primary profile error in the reference implementation.

### Negative log likelihood

\[
\operatorname{NLL}(q,\theta)
=
-\frac{1}{J}
\sum_{j=1}^{J}\log q_j(\theta_j).
\]

Assigning zero probability to the truth yields infinite loss. If finite
reporting uses a probability floor, the floor must be configured and named in
the metric rather than applied silently.

### Posterior divergence

For structured writers, marginal fitted-aware divergence is:

\[
D_{\mathrm{KL}}(p^A\Vert q)
=
\frac{1}{J}
\sum_j\sum_v p^A_j(v)
\log\frac{p^A_j(v)}{q_j(v)}.
\]

Joint KL is reported only when both systems expose comparable joint
distributions. Exact-oracle and fitted-aware divergence are separate fields.

### Excess proper-score error

\[
\operatorname{ExcessBrier}
=
\operatorname{BS}(q,\theta)
-
\operatorname{BS}(p^A,\theta).
\]

Positive values mean worse realized proper-score error than the fitted-aware
reference. This metric depends on latent truth and is evaluator-only.

## Update sensitivity

### Action-Conditioned Update Error

Let \(\Delta q=q_{t+1}-q_t\) and
\(\Delta p^A=p^A_{t+1}-p^A_t\), flattened across attribute marginals:

\[
\operatorname{ACUE}_t
=
\left\|\Delta q-\Delta p^A\right\|_1.
\]

ACUE measures the evidence increment rather than only terminal agreement. The
L1 norm is taken over all 12 attribute/value marginal components, matching the
proposal definition; it is not divided by \(J\).

### Update-direction accuracy

For each attribute/value component whose aware-reference change exceeds a
declared tolerance, record whether the updater changes it in the same direction.
Components below tolerance are excluded and their count retained. Direction
accuracy without a tolerance is unstable near zero.

### Update magnitude

\[
\operatorname{Magnitude}_t(q)
=
\frac{1}{J}\lVert q_{t+1}-q_t\rVert_1.
\]

Matched-context contrasts compare magnitude for the same response across
balanced, restricted, defaulted, and suggested contexts.

### Directional log odds

For preference direction, collapse positive values:

\[
q_j^+ = q_j(+1)+q_j(+2),\qquad
\ell(q_j)=\log\frac{q_j^+}{1-q_j^+}.
\]

Boundary probabilities use a declared clipping constant for numerical reporting.
The raw probability remains retained.

The oracle-update slope regression is:

\[
\widehat{\Delta\ell}
=\alpha+\beta\Delta\ell^A+\varepsilon.
\]

Mechanism-specific residuals and uncertainty matter alongside whether
\(\beta\) is near one.

ACUE, direction accuracy, magnitude, and the oracle slope are general-purpose
diagnostics. The paper hypotheses use additional frozen contrasts: H1 compares
anchor-directional log-odds and its absolute strength with fitted-aware
updates; H2 compares update-vector distance to fitted-aware and fitted-unaware
references; H7 combines mitigation superiority with positive-control
noninferiority. The exact formulas and decision rules follow below.

### Evidence-strength ordering

Within matched sets, rank update strength across volunteered preference,
balanced choice, default/suggestion acceptance, and restricted choice. The
fitted response model and human validation determine the expected middle
ordering; the software does not hard-code a universal ordering as truth.

## Frozen H1, H2, and H7 estimands

The implementation in
`src/cape_loop/experiments/hypothesis_estimands.py` freezes the executable
decision quantities for H1, H2, and H7. Experiment A writes
`metrics/experiment-a-hypothesis-estimands.json`; Experiment B writes
`metrics/experiment-b-h7-mitigation.json`.

Every artifact retains `claim_status = "not_claimed"`. A computed criterion is
not a paper claim and does not replace sample-adequacy, preregistration,
multiplicity, robustness, or author review.

### Shared pairing, uncertainty, and incomplete evidence

The primary response mode is `controlled_anchor`. Within each mechanism, the
visible context, anchor, prior, user, domain, target attribute, direction, and
prior-strength stratum are held fixed across updaters.

The independent unit is the complete latent user. Each contrast:

1. forms a paired difference within a matched trial;
2. averages all eligible rows within user; and
3. applies a deterministic percentile bootstrap over complete users.

The interval is 95%. The run's configured bootstrap count is used; a smoke
configuration declaring zero uses the runner's recorded 200-replicate fallback.
At least two independent user clusters are required. With fewer clusters or
missing required cells, `criterion_met = null` and
`computed_status = "incomplete"`; absence is never converted into failure,
success, or a zero effect.

The policy-conditioned mechanisms are exactly:

```text
restricted
default
suggested
```

### H1 — Directional over-update

ACUE cannot establish H1 by itself: distance from the fitted-aware update may
reflect over-update, under-update, or a wrong-direction update.

For controlled anchor direction \(s\) on attribute \(j\), define:

\[
\Delta\ell_s(q)
=
\operatorname{logit}q_{t+1,j}(s)
-
\operatorname{logit}q_{t,j}(s),
\]

where \(q_j(s)\) is total marginal mass on the two latent values with sign
\(s\). Clipping at \(10^{-6}\) is only a numerical log-odds diagnostic; it does
not change the retained belief.

For mechanism \(m\), H1 compares the ordinary full-context writer with the
fitted-aware reference:

\[
\delta^{\mathrm{dir}}_m
=
\mathbb E_{\mathrm{user}}
\left[
\Delta\ell_s(q^{\mathrm{full}}_m)
-
\Delta\ell_s(p^A_m)
\right],
\]

\[
\delta^{\mathrm{strength}}_m
=
\mathbb E_{\mathrm{user}}
\left[
\left|\Delta\ell_s(q^{\mathrm{full}}_m)\right|
-
\left|\Delta\ell_s(p^A_m)\right|
\right].
\]

A mechanism meets the executable H1 criterion only when both complete-user
bootstrap lower bounds are strictly above zero. The joint H1 artifact is
complete only when restricted, default, and suggested mechanisms all have
adequate evidence, and it passes only when all three mechanism criteria pass.
Individual mechanism rows remain reportable when the joint result is
incomplete.

### H2 — Closer to action-unaware inference

H2 compares update vectors, not terminal posteriors or response-model training
losses. Let \(\Delta q\) be the 12-component vector of marginal probability
increments over three attributes and four values. For one matched trial:

\[
d_A
=
\left\|\Delta q^{\mathrm{full}}-\Delta p^A\right\|_1,
\qquad
d_U
=
\left\|\Delta q^{\mathrm{full}}-\Delta p^U\right\|_1.
\]

The mechanism-wise proximity advantage is:

\[
\gamma_m=\mathbb E_{\mathrm{user}}[d_A-d_U].
\]

A positive value means the full-context writer is closer to fitted
action-unaware inference. A mechanism qualifies only when the 95%
complete-user bootstrap lower bound for \(\gamma_m\) is strictly above zero.
H2 requires at least two qualifying mechanisms. Separate clustered intervals
for \(d_A\) and \(d_U\), qualifying mechanisms, missing mechanisms, and
inadequate-cluster mechanisms are retained. Both fitted-aware and
fitted-unaware rows are mandatory for an evaluable mechanism.

### H7 — Mitigation without loss of valid learning

H7 has three non-substitutable components:

1. Experiment A update-error superiority under policy-conditioned evidence;
2. retention of valid learning under both balanced and volunteered evidence;
3. Experiment B reduction of closed-loop attribution error and
   self-confirming-profile rate.

#### Update-error superiority

For each policy-conditioned mechanism:

\[
\rho_m
=
\mathbb E_{\mathrm{user}}
\left[
\operatorname{ACUE}(q^{\mathrm{full}}_m)
-
\operatorname{ACUE}(q^{\mathrm{provenance}}_m)
\right].
\]

Positive values favor the provenance-aware writer. A mechanism qualifies when
its 95% complete-user bootstrap lower bound is strictly above zero. At least two
policy-conditioned mechanisms must qualify.

#### Balanced and volunteered noninferiority

Let \(u_F\) be the ordinary full-context directional log-odds update and \(u_P\)
the provenance-aware update. The frozen retention fraction is 0.80:

\[
\eta_c
=
\mathbb E_{\mathrm{user}}
\left[
u_{P,c}-0.80u_{F,c}
\right],
\qquad
c\in\{\mathrm{balanced},\mathrm{volunteered}\}.
\]

A condition qualifies only when:

- the ordinary full-context positive-control update has a 95% lower bound
  strictly above zero; and
- the 95% lower bound for \(\eta_c\) is at least zero.

Balanced values come from matched Experiment A rows. Volunteered preferences
are direct user-originated statements with no option set, default, ranking, or
suggestion. They cannot be synthesized from a choice row. The external review
converts each exactly bound provider response into:

```text
case_id
user_id
updater_id
directional_log_odds_update
```

Ordinary and provenance-aware records are paired by `case_id` and must share
`user_id`. Missing direct statements are never imputed from balanced choices,
the six-control diagnostic battery, an average, or zero.

#### Closed-loop self-confirmation mitigation

Experiment B pairs ordinary full-context and provenance-aware trajectories
using their common-random-number key under:

```text
policy_id = "soft_profile_conditioned"
initial_profile_condition = "incorrect"
```

It emits:

\[
\kappa_{\mathrm{attr}}
=
\mathbb E_{\mathrm{user}}
\left[
(\operatorname{Err}_F-\operatorname{Err}^{\mathrm{shadow}}_F)
-
(\operatorname{Err}_P-\operatorname{Err}^{\mathrm{shadow}}_P)
\right],
\]

\[
\kappa_{\mathrm{profile}}
=
\Pr_F(\text{at least one reportable self-confirming attribute})
-
\Pr_P(\text{at least one reportable self-confirming attribute}).
\]

Both 95% lower bounds must be strictly above zero. The profile indicator uses
the existing five-clause definition and is not reconstructed from terminal
error. Experiment A and B artifacts explicitly name their still-required
counterparts; a full H7 claim must join compatible frozen paper runs and
satisfy every component.

### Estimand artifact interpretation

The Experiment A artifact retains the analysis identity, independent unit,
bootstrap count, confidence level, frozen constants, and H1/H2/H7 results. The
Experiment B artifact retains the two closed-loop contrasts, pairing condition,
missing reason, and computation status.

An externally collected volunteered control is stored in a separate immutable
review. It binds the verified source run, regenerated plan, response and
provider-audit files, every converted update, and the recomputed H7 component.
It always records:

```text
claim_status = "not_claimed"
source_run_modified = false
missing_values_imputed = false
```

A formula, pairing unit, noninferiority margin, mechanism set, or decision-rule
change requires a new schema version. Existing artifacts must not be silently
reinterpreted.

## Calibration and information

### Calibration

Report raw and development-calibrated scores separately. Reliability summaries
group predicted probability assigned to the realized class using the retained
binning rule. The implemented calibration choices are per-view temperature
scaling and the explicitly labeled `none` ablation; retain the selected mode
and fitted temperature parameters.

No calibration summary may use test labels to select parameters or bins.

### Entropy

For a discrete belief \(q\):

\[
H(q)=-\sum_z q(z)\log q(z).
\]

Use the joint state \(z=(\theta,\psi)\) when available and label marginal-sum
entropy otherwise.

### Action-aware information gain

\[
\operatorname{IG}_t
=
H(p_t^A)-H(p_{t+1}^A).
\]

Cumulative action-aware information gain describes how diagnostic the collected
history is under the declared aware model. A single noisy observation can
increase realized entropy; aggregate interpretation must not assume every
turn-level value is nonnegative.

## Selection and attribution

Let `Err` be the configured primary profile error, normally marginal Brier.

### Evidence-selection cost

\[
\operatorname{SelectionCost}
=
\operatorname{Err}(p_T^{\text{shadow, profile policy}},\theta)
-
\operatorname{Err}(p_T^{\text{shadow, balanced}},\theta).
\]

The paired shadows use their corresponding policy histories. Positive values
indicate that profile-conditioned evidence collection left an aware inference
system with more error.

### Evidential-attribution cost

\[
\operatorname{AttributionCost}_{U,\pi}
=
\operatorname{Err}(q_T^{U,\pi},\theta)
-
\operatorname{Err}(p_T^{\text{shadow},\pi},\theta).
\]

Both terms use the identical contexts and responses. Positive values isolate
error beyond evidence selection. Experiment B uses its exact finite
action-aware shadow for this reference; other analyses must name any fitted
reference explicitly rather than treating it as exact.

### Self-Confirmation Interaction

For incorrect initial profiles:

\[
\begin{aligned}
\operatorname{SCI}_{\text{wrong}}
=&
\left[
\operatorname{Err}(q_T^{U,\pi_p},\theta)
-
\operatorname{Err}(p_T^{A,\pi_p},\theta)
\right]\\
&-
\left[
\operatorname{Err}(q_T^{U,\pi_b},\theta)
-
\operatorname{Err}(p_T^{A,\pi_b},\theta)
\right],
\end{aligned}
\]

where \(\pi_p\) is profile-conditioned and \(\pi_b\) balanced. A positive value
means the attribution gap is larger under profile-conditioned collection.

## False-profile confidence

For a dimension with seeded wrong direction \(W_j\):

\[
q_t^{\text{wrong},j}
=
\sum_{v\in W_j}q_{t,j}(v).
\]

### False Confidence Gain

\[
\operatorname{FCG}_{t,j}
=
\operatorname{logit}q_{t+1}^{\text{wrong},j}
-
\operatorname{logit}q_t^{\text{wrong},j}.
\]

### Laundered Confidence Gain

\[
\operatorname{LCG}_{t,j}
=
\operatorname{FCG}_{t,j}^{\text{system}}
-
\operatorname{FCG}_{t,j}^{\text{shadow-aware}}.
\]

Cumulative LCG is the sum over turns under the metric’s declared aggregation.
Boundary clipping is declared and applied identically to system and shadow.

### False stable and self-confirming rates

A seeded attribute is **false-stable** when it remains materially wrong at
terminal time and its wrong-direction mass never departs from the initial value
by more than `false_stability_tolerance` at any retained turn. This dedicated
stability threshold is distinct from numerical `direction_tolerance`. A
trajectory is a false-stable profile when at least one eligible seeded
attribute is false-stable.

A **false self-confirming profile** additionally requires:

1. terminal/material wrongness;
2. increased wrong-direction mass;
3. cumulative LCG above a declared threshold;
4. recorded influence on a subsequent action;
5. no equivalent shadow confidence.

The profile rates use all eligible incorrectly seeded trajectories as their
denominator and count a trajectory once if any attribute qualifies.
Attribute-assessment rates are retained under explicitly named
`*_attribute_rate` fields and must not be called profile rates.

Clauses 3 and 5 are not duplicate thresholds. Clause 3 uses cumulative
system-minus-shadow log-odds gain. Clause 5 treats the shadow as equivalent when
its terminal wrong-direction probability mass is within the configured
`shadow_equivalence_tolerance` of (or exceeds) the system's terminal wrong
mass.

## Policy and user outcomes

### Intrinsic regret

\[
\operatorname{Regret}_t
=
\max_{x\in\mathcal X_t^{\mathrm{full}}}
\theta^\top\phi(x)
-
\theta^\top\phi(Y_t).
\]

The complete feasible pool is required. Presentation bonuses never enter this
metric.

### Preference-dimension coverage

Coverage is the fraction of domain preference dimensions for which the history
contains at least one target-attribute context displaying both negative and
positive directions. Merely targeting a dimension under a one-direction hard
filter does not count. `turns_to_full_preference_coverage` records the first
one-based turn at which all dimensions qualify, or null if they never do.

### Option diversity

Experiment B reports the number of distinct selected options and
`displayed_option_diversity`, the number of distinct displayed option IDs
divided by the six isolated directional options in a domain. This v1 diagnostic
is a normalized support count, not an entropy/evenness measure.

Because balanced and soft policies can share the same option support, terminal
rows also retain profile-conditioned exposure rate, distinct presentation
mechanism count, and normalized evenness across observed mechanism counts.
Together these distinguish exposure support from the presentation treatment.

### Terminal behavioral accuracy

Accuracy is computed on the shared exogenous battery against the response
implied by latent intrinsic preference under each diagnostic item. Among
projected-utility maximizers within `1e-12`, the stable lexicographically largest
option ID is selected. Any truly utility-maximizing option counts as correct.
Rows separately retain predicted-utility tie count, intrinsic-utility tie
count, and evaluated-item count. They report three complementary accuracies:
full-credit accuracy, accuracy excluding intrinsically tied items, and
fractional accuracy that awards `1 / number_of_true_maximizers` when the
prediction is one of several tied optima.

### Cross-context generalization

Measure terminal performance on held-out scenario and paraphrase families that
apply the same preference dimensions in a new context. It is not the same as
memorizing a direct preference probe.

## Evaluation validity

### Kendall rank agreement

Use Kendall’s \(\tau_b\) to handle ties between updater scores. Bootstrap
intervals resample complete latent-user clusters and recompute all ranks. For
Experiment C, systems are first aligned by user, domain, and replicate. All
domains and trajectory replicates belonging to one user remain together in
every draw.

### Pairwise reversal probability

For every system pair, estimate the fraction of paired bootstrap replicates in
which the performance ordering differs between open-loop and closed-loop
evaluation. Report ties separately or according to a declared tolerance. This
is descriptive. A Gate 5 reversal additionally requires a joint paired
complete-user interval analysis: the open and closed error-difference
intervals must clear the tie region in opposite directions, and the
closed-minus-open difference-of-differences interval must clear it in the same
direction.

### Interval-supported tiers and selection

Separate open- and closed-development partial orders use paired
error-difference intervals. A system dominates another only when the interval
clears zero beyond `ranking_tie_tolerance`; a shared tier means dominance was
not established, not that equality was proved. Bootstrap rank intervals and
pairwise tie probabilities are retained separately and do not define the
inferential tiers.

`evaluation_selection_regret` selects every system in the open and closed
inferential top tiers and reports mean/minimum/maximum test regret over the
resulting Cartesian product. Paired closed-test difference intervals yield a
conservative interval envelope over that set. Numeric proximity to a sample
minimum is not treated as inferential equivalence.

### Terminal profile calibration

Every B/C terminal profile score reports top-label multiclass ECE and ten
fixed-width reliability bins. One prediction is the highest-probability class
for one preference attribute, so each trajectory contributes exactly three
`preference_attribute_forecast` units per decoded or structured profile.
Pooled artifacts aggregate the bin numerators and denominators. They label the
trajectory/user as the dependence unit and provide descriptive calibration
only; attribute forecasts are not independent user-level replicates.

### Open-loop optimism

For a system \(s\), a simple error-scale form is:

\[
\operatorname{Optimism}(s)
=
\operatorname{Err}_{\text{closed}}(s)
-
\operatorname{Err}_{\text{open}}(s).
\]

Positive values mean fixed-history error understated closed-loop error. The
logging regime and split must accompany the value.

### Evaluation Selection Regret

Choose systems on development records:

\[
s_{\text{open}}^*
=\arg\min_s\operatorname{Err}_{\text{open,dev}}(s),
\qquad
s_{\text{closed}}^*
=\arg\min_s\operatorname{Err}_{\text{closed,dev}}(s).
\]

Evaluate that choice on held-out closed-loop test records:

\[
\operatorname{ESR}
=
\operatorname{Err}_{\text{closed,test}}(s_{\text{open}}^*)
-
\operatorname{Err}_{\text{closed,test}}(s_{\text{closed}}^*).
\]

System selection and final assessment must use different records. If
development intervals leave multiple systems in either top tier, retain the
full set. A substantial-ESR gate check requires both the worst descriptive
pairing and the conservative paired-test interval envelope to exceed the
declared practical threshold.

## Confirmatory mixed-effects interpretation

The optional R harness fits the proposal's exact maximal models, but its
contrasts remain narrower than the paper hypotheses. Experiment A's ACUE
contrasts do not test the direction or magnitude of the target's belief update.
Experiment B's terminal-error interaction does not alone establish all five
self-confirmation clauses.

Reported mixed-effects intervals use the explicit
`pointwise_unadjusted_confidence_lower` and
`pointwise_unadjusted_confidence_upper` fields, including their standardized
counterparts. They are pointwise 95% intervals; Holm adjustment applies to
p-values, not confidence limits. The complete formulas, planned contrast
families, diagnostics, failure semantics, and output contract are in the
[confirmatory mixed-effects README](../analysis/confirmatory-mixed-effects/README.md).

## Human evidence-strength metrics

Human validation reports condition-wise evidence-strength distributions,
matched-set paired contrasts, ordinal/rank agreement, and uncertainty at the
annotator and scenario level. Comparisons with updater magnitude use the same
matched items.

Human ratings are pragmatic judgments, not latent-truth labels.

H8 operationalizes cross-scale provenance sensitivity as

\[
D=(E_{\text{balanced}}-E_{\text{policy}})/E_{\text{balanced}},
\qquad
\Delta_{H8}=D_{\text{human}}-D_{\text{model}}.
\]

Human ratings are shifted from 1–7 to a zero-at-no-support scale before this
ratio; model rows must use the nonnegative positive part of the
anchor-directional log-odds update and attest that zero means no update toward
the claim. Balanced-zero pairs are undefined and excluded rather than
regularized. Matched scenarios are averaged within participant or held-out
test-user cluster, clusters receive equal weight, and the two independent
samples are bootstrapped separately. The primary ordinary-LLM criterion is
evaluable only with at least eight pair-complete clusters in both samples for
all three policy mechanisms and one independently complete fitted-aware source.

## Statistical unit and uncertainty

The independent unit is a complete latent user or trajectory. Turns are
repeated observations, not independent samples. Experiment C uses the complete
latent user as its unit: all domain × trajectory-replicate rows are validated
as one cluster, reduced together, and never split across bootstrap draws.

Release analyses should retain:

- trajectory-level paired bootstrap intervals;
- user/scenario grouping identifiers for mixed-effects analysis;
- effect sizes and confidence intervals;
- raw and calibrated results;
- simulation-based power inputs;
- multiplicity correction for declared secondary comparisons;
- held-out profile/template results.

Bootstrap seeds are semantic-keyed and retained. Ranking summaries must be
reproducible across execution order.
