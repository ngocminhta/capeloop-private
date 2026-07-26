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

### Evidence-strength ordering

Within matched sets, rank update strength across volunteered preference,
balanced choice, default/suggestion acceptance, and restricted choice. The
fitted response model and human validation determine the expected middle
ordering; the software does not hard-code a universal ordering as truth.

## Calibration and information

### Calibration

Report raw and development-calibrated scores separately. Reliability summaries
group predicted probability assigned to the realized class, with binning or
isotonic/temperature parameters declared in the configuration.

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

## Human evidence-strength metrics

Human validation reports condition-wise evidence-strength distributions,
matched-set paired contrasts, ordinal/rank agreement, and uncertainty at the
annotator and scenario level. Comparisons with updater magnitude use the same
matched items.

Human ratings are pragmatic judgments, not latent-truth labels.

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
