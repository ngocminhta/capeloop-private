# Metrics

This document defines the intended quantities and their interpretation. The
current runner emits wide, experiment-specific metric rows rather than a
generic name/version envelope; their concrete keys are documented in
[Data model](data-model.md). A formula change therefore requires an explicit
artifact/schema or software-version boundary and must not be applied silently
to an existing result.

Let \(J=3\) be the number of preference attributes and
\(\mathcal V=\{-2,-1,+1,+2\}\). Let \(q_j(v)\) be an updater’s marginal belief
and \(\theta_j\) the latent value. Let \(p^*\) denote the exact action-aware
posterior under the declared generating response model and \(p^F\) the fitted
action-aware posterior. Experiment A's primary `same_response_provenance`
metrics use \(p^*\); fitted-reference metrics are explicitly labeled
secondary.

## Experiment A prior-information equivalence

For truth-aligned prior strength \(s\in[0,1)\), Experiment A defines

\[
q_{s,j}(v)=\frac{1-s}{|\mathcal V|}
            +s\,\mathbf 1[v=\theta_j],
\qquad
q_s(\theta')=\prod_{j=1}^{J}q_{s,j}(\theta'_j).
\]

The LLM receives all \(q_{s,j}\) and the exact updater receives their
factorized joint \(q_s\). Thus neither starts with cross-attribute information
unavailable to the other. The oracle's nuisance prior over susceptibility is
uniform over the support prospectively assigned to the evaluation split.

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

For structured writers, the primary controlled-trial marginal divergence is:

\[
D_{\mathrm{KL}}(p^*\Vert q)
=
\frac{1}{J}
\sum_j\sum_v p^*_j(v)
\log\frac{p^*_j(v)}{q_j(v)}.
\]

Joint KL is reported only when both systems expose comparable joint
distributions. Fitted-aware divergence substitutes \(p^F\) and is a separate
secondary field.

### Excess proper-score error

\[
\operatorname{ExactExcessBrier}
=
\operatorname{BS}(q,\theta)
-
\operatorname{BS}(p^*,\theta).
\]

Positive values mean worse realized proper-score error than the exact
action-aware reference. It can be derived from the retained exact posterior and
latent truth. The legacy emitted field `excess_brier` uses \(p^F\) and is
secondary. Both depend on latent truth and are evaluator-only.

## Update sensitivity

### Exact Action-Conditioned Update Error

Let \(\Delta q=q_{t+1}-q_t\) and
\(\Delta p^*=p^*_{t+1}-p^*_t\), flattened across attribute marginals:

\[
\operatorname{ExactACUE}_t
=
\left\|\Delta q-\Delta p^*\right\|_1.
\]

ExactACUE measures the evidence increment rather than only terminal agreement.
The L1 norm is taken over all 12 attribute/value marginal components, matching
the proposal definition; it is not divided by \(J\). It is an unsigned
secondary magnitude diagnostic: it cannot distinguish over-updating from
under-updating and is not Experiment A's confirmatory mixed-effects outcome.
The legacy field `acue` applies the same formula to \(p^F\); it remains a
secondary learnability and misspecification diagnostic and must not be silently
pooled with `exact_acue`.

### Update-direction accuracy

The retained `update_direction_accuracy` field compares component-wise changes
with the fitted action-aware reference and is secondary. Components whose
reference change falls below a declared tolerance are excluded and their count
is retained. Primary signed interpretation uses \(u^*\), \(\widehat u\), and
\(r\) below. Direction accuracy without a tolerance is unstable near zero.

### Update magnitude

\[
\operatorname{Magnitude}_t(q)
=
\frac{1}{J}\lVert q_{t+1}-q_t\rVert_1.
\]

Matched-context contrasts compare magnitude for the same response across
balanced, restricted, ranking, defaulted, and suggested contexts.

### Directional log odds

For preference direction, collapse positive values:

\[
q_j^+ = q_j(+1)+q_j(+2),\qquad
\ell(q_j)=\log\frac{q_j^+}{1-q_j^+}.
\]

More generally,
\(q_j(s)=\sum_{v:\operatorname{sign}(v)=s}q_j(v)\) for
\(s\in\{-1,+1\}\).

Boundary probabilities use a declared clipping constant for numerical reporting.
The raw probability remains retained.

For target direction \(s\), Experiment A begins with the same supplied prior
\(p_t^*=q_t=q_0\). The primary warranted log-odds update is:

\[
u^*
=
\operatorname{logit}p^*_{t+1,j}(s)
-
\operatorname{logit}p^*_{t,j}(s),
\]

\[
\widehat u
=
\operatorname{logit}q_{t+1,j}(s)
-
\operatorname{logit}q_{t,j}(s),
\qquad
r=\widehat u-u^*.
\]

The signed calibration residual \(r\) is positive for an update stronger than
warranted in the anchor direction, negative for a weaker update, and must be
interpreted with \(u^*\) to identify wrong-direction behavior. It does not
presuppose a universal over-update effect.

For updater \(k\) and mechanism \(m\), fit:

\[
\widehat u_{k,m}
=
\alpha_{k,m}+\beta_{k,m}u^*+\varepsilon.
\]

The ideal is \(\alpha=0\), \(\beta=1\), and low residual RMSE. The artifact
retains each coefficient, RMSE, user-cluster count, and complete-user bootstrap
interval. A mechanism slice with no reference-update variation is
`not_estimable`; no artificial slope is assigned.

The primary provenance contrast compares each non-balanced mechanism with
balanced for the predeclared target writer:

\[
\Delta r_m
=
\mathbb E_{\mathrm{user}}
\left[
r_{\mathrm{target},m}
-
r_{\mathrm{target},\mathrm{balanced}}
\right],
\]

As a secondary unsigned magnitude diagnostic, report:

\[
\Delta e_m
=
\mathbb E_{\mathrm{user}}
\left[
|\widehat u_{\mathrm{target},m}-u^*_{m}|
-
|\widehat u_{\mathrm{target},\mathrm{balanced}}
-u^*_{\mathrm{balanced}}|
\right].
\]

Both are paired within the literal same-response matched set. \(\Delta r_m\)
is primary and removes the target writer's balanced-condition error.
\(\Delta e_m\) is secondary because its magnitude does not identify direction.
Calibration curves remain model- and mechanism-specific; model heterogeneity
is not collapsed into one direction.

A parallel regression replacing \(u^*\) with the fitted-aware update is
secondary robustness. Directional over-update relative to \(p^F\) and distance
to a fitted action-unaware updater are also secondary diagnostics; they are not
the primary H1/H2 hypotheses.

### Evidence-strength ordering

Within matched sets, rank update strength across volunteered preference,
balanced choice, changed ranking, default/suggestion acceptance, and restricted
choice. For simulated trials, the exact declared response model determines the
warranted condition-specific updates. The software does not hard-code a
universal middle ordering; fitted-model and human judgments are secondary
validation.

## Primary Experiment A estimands

The primary response mode is `controlled_anchor` and the analysis track is
`same_response_provenance`. Within a matched set, the anchor, literal local
user response, prior, user, domain, target attribute, direction, and
prior-strength stratum are held fixed. The mechanisms are:

```text
balanced
restricted
ranking
default
suggested
```

The exact-reference analysis records `reference_basis =
"exact_action_aware"`. A separate invariant audit must confirm that the local
response is literally unchanged across the matched mechanisms. A failed or
incomplete audit makes the corresponding same-response contrast ineligible.
Held-out paraphrase rows remain `controlled_anchor` rows and are analyzed
against the same exact primary reference.

### Outcome-neutral Gate 1 quantities

Gate 1 establishes that the controlled design can identify provenance
calibration; it does not test whether an evaluated LLM fails. For matched set
\(s\), domain \(d\), and mechanism \(m\), define the exact warranted-update
separation from balanced as

\[
D_{m,d}
=
\frac{1}{|S_d|}
\sum_{s\in S_d}
\left|u^*_{s,m}-u^*_{s,\mathrm{balanced}}\right|.
\]

The design-side separation criterion requires \(D_{m,d}>\delta_{\mathrm{id}}\)
in both domains for at least two non-balanced mechanisms. The implementation
defaults to \(\delta_{\mathrm{id}}=0.01\); the paper run must freeze that
identifiability margin before execution. All declared domain×mechanism cells
must also be present.

Exact-oracle self-consistency requires both
\(\max_s\mathrm{ExactACUE}_{\mathrm{exact},s}\le\epsilon\) and
\(\max_s|u_{\mathrm{exact},s}-u_s^*|\le\epsilon\), with the numerical default
\(\epsilon=10^{-10}\). The remaining conditions are a passing same-response
audit and complete held-out controlled-paraphrase coverage with invariant
selected-option and visible-context bindings. Fitted-aware versus
fitted-unaware scores and held-out Brier gaps are secondary diagnostics and do
not control this gate.

### H1 — Exact-oracle causal-provenance calibration

H1 is an estimation target, not a universal directional-over-update test.
Report \(\alpha_{k,m}\), \(\beta_{k,m}\), residual RMSE, mean signed residual,
mean absolute log-odds residual, and ExactACUE for every eligible
updater–mechanism cell. Compare intervals with the ideal
\((\alpha,\beta,\mathrm{RMSE})=(0,1,0)\). The primary report is continuous and
does not assign a binary “materially miscalibrated” label. Any later binary
label requires a separately preregistered practical margin.

### H2 — Provenance-specific residual heterogeneity

For each treatment
\(m\in\{\text{restricted},\text{ranking},\text{default},\text{suggested}\}\),
the confirmatory family reports the target writer's paired primary
\(\Delta r_m\) contrast defined above. Report \(\Delta e_m\) as secondary
unsigned magnitude and other updater families as labeled controls or
model-specific descriptive curves. This isolates interpretation of provenance
because the response is fixed. A nonzero contrast may be positive or negative
and does not imply that the model ignored context.

### Pairing, strata, and uncertainty

The independent unit is the complete latent user. Each contrast:

1. forms a paired difference within a complete matched set;
2. averages eligible observations within user; and
3. resamples complete users, preserving domains, weak/strong
   \(\theta\)-strata, balanced-choice-margin strata, and nested replicates.

Random seeds are robustness replicates, not independent users. At least two
user clusters are required for an interval. Missing mechanisms, incomplete
matched sets, invariant-audit failures, or inadequate reference-update
variation produce an explicit incomplete/not-estimable status, never a zero
effect.

For closed-loop turn \(t\), compute the balanced counterfactual response
probabilities before drawing the response and define:

\[
m_t=p_{t,(1)}^{*,\mathrm{balanced}}
-p_{t,(2)}^{*,\mathrm{balanced}},
\]

where \((1)\) and \((2)\) are the two largest probabilities. The prospective
strata are `near_tie` for \(m_t<0.20\), `marginal` for
\(0.20\le m_t<0.50\), and `decisive` otherwise. Target preference strength is
`weak` for \(|\theta_j|=1\) and `strong` for \(|\theta_j|=2\); a user's
trajectory-level label is `mixed` when its coordinates contain both.

### Secondary fitted-reference diagnostics

The legacy fitted-aware directional-over-update and fitted-unaware-proximity
estimands remain available in
`metrics/experiment-a-hypothesis-estimands.json`. They answer useful
learnability and model-misspecification questions, but they are secondary and
must not be described as the primary H1/H2 results. The fitted and exact
reference bases are never pooled.

## H7 mitigation estimands

H7 asks whether explicit provenance improves calibration without suppressing
warranted learning. It has three components:

1. lower ExactACUE or absolute exact log-odds residual under restricted,
   ranking, default, or suggested evidence;
2. retained learning under balanced and genuinely volunteered evidence; and
3. reduced same-history attribution cost or continuous closed-loop
   amplification under soft profile conditioning.

For Experiment A:

\[
\rho_m
=
\mathbb E_{\mathrm{user}}
\left[
\operatorname{ExactACUE}(q^{\mathrm{full}}_m)
-
\operatorname{ExactACUE}(q^{\mathrm{provenance}}_m)
\right].
\]

Positive values favor the provenance-aware writer. Balanced noninferiority
uses the preregistered retention fraction. Volunteered controls must be direct
user-originated statements with no option set, default, ranking, or suggestion;
they cannot be synthesized from a choice row or imputed when missing.

Experiment B compares same-history attribution gaps, EAR, cumulative excess
confidence, and information/disconfirmation deficits. Recovery is absent
unless its separate stage-gated protocol is activated.
Reduction in the strict five-clause self-confirming-profile rate is a secondary
mitigation endpoint, not a required substitute for all continuous components.

Every estimand artifact retains `claim_status = "not_claimed"`. A computed
criterion does not replace sample adequacy, preregistration, multiplicity,
robustness, or author review. A formula, pairing unit, margin, mechanism set, or
decision rule change requires a new schema/version boundary; existing
artifacts are not silently reinterpreted.

## Calibration and information

### Calibration

Report raw and development-calibrated scores separately. Experiment A's
confirmatory updater outputs and exact-oracle residuals always use the raw LLM
vector; its temperature-scaled vector is a secondary forecast-calibration
diagnostic. This avoids counting a temperature-induced departure from an
unchanged prior as an evidential update. B/C use the configured active vector
for the realized history and retain paired terminal diagnostics. Reliability
summaries group predicted probability assigned to the realized class using the
retained binning rule. The implemented calibration choices are per-view
temperature scaling and the explicitly labeled `none` ablation; retain the
selected mode and fitted temperature parameters.

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
H(p_t^*)-H(p_{t+1}^*).
\]

Cumulative action-aware information gain describes how diagnostic the collected
history is under the exact declared aware model. A single noisy observation
can increase realized entropy; aggregate interpretation must not assume every
turn-level value is nonnegative. A fitted-reference information measure, if
used, must be separately labeled.

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

Define the policy-specific gap

\[
G_{U,\pi}=\operatorname{AttributionCost}_{U,\pi}.
\]

This is the operational measure of **policy-conditioned evidential
legibility** for updater $U$: it conditions the writer's error on the exact
diagnostic content of the same policy-generated history. The primary paired
schedule-matched contrast is

\[
G_{U,\mathrm{soft}}-G_{U,\mathrm{balanced}}
\]

The separately reported whole-policy comparator is

\[
G_{U,\mathrm{soft}}-G_{U,\mathrm{exploratory}}.
\]

They are stored as `soft_minus_balanced_attribution_gap` and
`soft_minus_exploratory_attribution_gap`; the component fields are
`profile_attribution_cost`, `balanced_attribution_cost`, and
`exploratory_attribution_cost`. Positive values mean the soft history is less
legible to that updater relative to the stated reference policy. Exploratory
retains genuine adaptive target/scenario selection, so its contrast is not
interpreted as a turn-matched causal effect and is outside the primary matched
family. No composite legibility index is constructed.

### Policy-conditioned attribution-gap contrast

For incorrect initial profiles:

\[
\begin{aligned}
\operatorname{SCI}_{\text{wrong}}
=&
\left[
\operatorname{Err}(q_T^{U,\pi_p},\theta)
-
\operatorname{Err}(p_T^{*,\pi_p},\theta)
\right]\\
&-
\left[
\operatorname{Err}(q_T^{U,\pi_b},\theta)
-
\operatorname{Err}(p_T^{*,\pi_b},\theta)
\right],
\end{aligned}
\]

where \(\pi_p\) is profile-conditioned and \(\pi_b\) balanced. A positive value
means the attribution gap is larger under profile-conditioned collection.
The retained `self_confirmation_interaction` field is a backward-compatible
alias for this arithmetic. It must not be interpreted as evidence of a changed
user response or behavioral self-confirmation.

The complete natural-response policy contrast obeys the exact accounting
identity:

\[
\begin{aligned}
&\operatorname{Err}(q_T^{U,\pi_p},\theta)
-
\operatorname{Err}(q_T^{U,\pi_b},\theta)\\
&\quad =
\operatorname{SelectionCost}
+
\left[
\operatorname{AttributionCost}_{U,\pi_p}
-
\operatorname{AttributionCost}_{U,\pi_b}
\right].
\end{aligned}
\]

When the comparator is the exact balanced shadow rather than the updater's
balanced branch:

\[
\operatorname{Err}(q_T^{U,\pi_p},\theta)
-
\operatorname{Err}(p_T^{*,\pi_b},\theta)
=
\operatorname{SelectionCost}
+
\operatorname{AttributionCost}_{U,\pi_p}.
\]

These identities separate changes in the evidence stream from interpretation
of the realized history. They do not require either component to be positive.

The total updater-policy effect is retained explicitly as
`soft_minus_balanced_terminal_error`. Every decomposition row also retains the
four raw terminal operands, the reconstructed total
`evidence_selection_cost + soft_minus_balanced_attribution_gap`, the numeric
residual, the configured tolerance, and `decomposition_identity_passed`.
Artifact construction fails if the selection term, either attribution term,
the attribution interaction, or the total identity misses tolerance. This is
an accounting invariant, not a test result.

The correct-profile control produces the paired moderation estimand

\[
\left(G_{U,\mathrm{soft}}-G_{U,\mathrm{balanced}}\right)_{\mathrm{incorrect}}
-
\left(G_{U,\mathrm{soft}}-G_{U,\mathrm{balanced}}\right)_{\mathrm{correct}}.
\]

It is retained as
`incorrect_minus_correct_soft_balanced_attribution_gap` and uses complete
latent users as clusters.

The policy-conditioned-legibility decision is conjunctive and is evaluated in
the incorrect-initial-profile stratum. One-sided paired complete-user tests
must support $G_{U,\mathrm{soft}}>0$,
$G_{U,\mathrm{soft}}-G_{U,\mathrm{balanced}}>0$, and
$\operatorname{SelectionCost}<\epsilon_{\mathrm{sel}}$. The frozen
noninferiority margin is $\epsilon_{\mathrm{sel}}=0.02$ on the marginal-Brier
scale. This permits a small practical selection loss and does not claim equal
mutual information. The separate nested net-harm decision additionally
requires the incorrect-seed
`soft_minus_balanced_terminal_error > 0.02`; net harm is not required for the
narrower legibility result.

This conjunction is the sole primary Experiment B claim within a model. It is
implemented as an intersection-union test whose composite p-value is the
maximum of the three component p-values. Thus every component must reject at
one-sided alpha `0.05`; no additional alpha division is applied within the
conjunction. Its rejection opens a frozen three-member Holm family containing
the Gate 2 four-component IUT, the incorrect-minus-correct moderation, and
nested net profile harm. A missing member is conservatively entered as p=1.
The emitted `multiplicity` object records raw and adjusted p-values, ranks,
thresholds, activation, and decisions under
`experiment-b-within-model-gatekeeping-v1`. This hierarchy is evaluated
separately for each model; it does not pool models or create an any-model
claim, and bounded calibration remains descriptive.

For the primary directional decision, observations are first reduced to
equally weighted complete-user means. The one-sided sign-flip reference is
enumerated exactly over all $2^n$ signs for $n\le16$; larger samples use
16,384 deterministic Monte Carlo sign patterns, include the observed
assignment, and apply a plus-one correction. Alpha is `0.05` and at least eight
users are required. Two-sided percentile user-cluster bootstrap intervals are
sensitivity summaries. A paired-trajectory bootstrap is a further sensitivity
and never increases the independent-user count. The sign-flip interpretation
requires sign exchangeability of complete-user paired contrasts around the
tested null margin.

## Primary and supporting continuous closed-loop estimands

The primary conjunction is the soft-policy same-history attribution gap, its
schedule-matched soft-minus-balanced contrast, and evidence-selection cost.
The incorrect-minus-correct moderation is in the multiplicity-controlled
secondary family. The soft-minus-exploratory gap is a supporting whole-policy
comparator because exploratory target and scenario choices are intentionally
adaptive. Gate 2, moderation, and nested net harm are the only secondary
confirmatory claims; the measures below are otherwise supporting
continuous mechanism outcomes: they explain action content, amplification,
information loss, or recovery without requiring the secondary strict
five-clause endpoint to occur.

### Action-level policy characterization

Policy names do not substitute for measured actions. Before sampling the user
response, every turn records:

* `profile_consistency_score`, the average signed alignment in `[-1, 1]` of
  option-set composition, first rank, default, and suggestion with the current
  profile direction on the target attribute. Positive values favor the
  profile, negative values favor its opposite, and zero is neutral or the
  current profile has no directional expectation.
* `profile_consistency_advantage_over_balanced`, the preceding score minus the
  score of the paired balanced counterfactual action generated at the same
  turn from the same profile and semantic random key.
* `expected_action_aware_information_gain`, the ex-ante preference-state
  entropy reduction

  \[
  H[p^*(\theta\mid H_t)]-
  \mathbb E_{Y_t}H[p^*(\theta\mid H_t,C_t,Y_t)].
  \]

  The expectation enumerates all displayed responses under the exact current
  joint preference–susceptibility belief and declared response model, but the
  entropy target is the preference marginal. It is distinct from
  `action_aware_information_gain`, the realized whole-state entropy reduction
  after the sampled response.
* `ex_ante_balanced_choice_divergence_probability`, the exact probability that
  shared-noise paired choices differ for binary actions exposing the same two
  option IDs. Random utility uses shared Gumbels; the rule-based sensitivity
  model uses a shared inverse-CDF draw. It is the absolute difference in either
  option's choice probability. The value is null unless the action and balanced
  counterfactual expose the identical binary choice set. Its trajectory mean
  is conditional on comparable turns;
  `ex_ante_balanced_choice_comparable_turn_count` and its rate retain that
  denominator, while `balanced_choice_set_divergence_rate` separately reports
  turns whose option-ID sets differ.

Trajectory outputs retain sums or means of these quantities. They distinguish
profile consistency, diagnostic content, and behavioral susceptibility without
asserting that one caused another or collapsing them into a composite score.

The paired expected-information contrasts are

\[
\operatorname{ExpectedPrefIGDeficit}_{\mathrm{exp-soft}}
=\sum_t \operatorname{EIG}(C_t^{\mathrm{exploratory}})
-\sum_t \operatorname{EIG}(C_t^{\mathrm{soft}})
\]

and

\[
\operatorname{ExpectedPrefIGDeficit}_{\mathrm{bal-soft}}
=\sum_t \operatorname{EIG}(C_t^{\mathrm{balanced}})
-\sum_t \operatorname{EIG}(C_t^{\mathrm{soft}}).
\]

They are stored as `expected_preference_information_gain_deficit` and
`balanced_expected_preference_information_gain_deficit`. Positive values mean
the named comparator offered more expected preference information than soft
conditioning. Unlike `action_aware_information_gain_deficit`, these are ex-ante
preference-marginal quantities and do not depend on the one response that was
sampled.

### Error amplification ratio

\[
\operatorname{EAR}
=
\frac{\operatorname{BS}(q_T,\theta)}
{\operatorname{BS}(q_0,\theta)}.
\]

`initial_error` retains the denominator and `error_amplification_ratio` retains
the ratio. A value above one means the trajectory worsened its own seed; below
one means it improved. The ratio is null when initial error is numerically zero
and should be interpreted primarily for incorrect seeds. It complements the
paired SelectionCost: a soft policy may be worse than balanced while both
branches still improve an intentionally bad initial profile.

### Exploratory disconfirmation deficit

For initially false attribute \(j\), the exact shadow's evidence against the
false sign is

\[
D_{\pi,j}
=
-\sum_t \operatorname{FCG}^{\mathrm{shadow}}_{t,j}.
\]

The paired trajectory measure is

\[
\operatorname{DD}
=
\frac{1}{|J_{\mathrm{false}}|}
\sum_{j\in J_{\mathrm{false}}}
\left(D_{\mathrm{exploratory},j}-D_{\mathrm{profile},j}\right).
\]

The field is `disconfirmation_evidence_deficit_log_odds`. Positive values mean
the exploratory policy collected more action-aware evidence against the false
seed. `action_aware_information_gain_deficit` separately stores exploratory
minus profile-conditioned realized whole-state entropy reduction. Ex-ante
expected preference information gain is now retained separately per action and
trajectory; the two quantities must not be conflated.

Sensitivity runs do not add an exploratory branch. They instead retain the
same quantities with the balanced policy as comparator:

\[
\operatorname{BalancedDD}
=D_{\mathrm{balanced}}-D_{\mathrm{profile}},
\qquad
\operatorname{BalancedIGDeficit}
=IG_{\mathrm{balanced}}-IG_{\mathrm{profile}}.
\]

The fields are
`balanced_disconfirmation_evidence_deficit_log_odds` and
`balanced_action_aware_information_gain_deficit`. They are computed from
already retained exact-shadow trajectories and require no extra provider calls.

### Partial reinforcement-event rate

A turn is a reinforcement event only when:

1. the assigned profile-conditioned treatment produces a visible action that
   differs from its balanced counterfactual and promotes an initially false
   direction;
2. the simulated user selects that direction;
3. the evaluated updater increases false-direction confidence; and
4. that increase exceeds the exact same-history shadow increase.

`reinforcement_event_count / number_of_turns` is retained as
`reinforcement_event_rate`; turns without an initially false attribute make
the trajectory rate null. This is a continuous/partial-loop diagnostic and
must not be called self-confirmation. The paired decomposition additionally
retains `visible_action_divergence_rate` and
`observed_choice_divergence_rate`, separating an assigned/visible intervention
from a changed simulated response.

### Paired behavioral-reinforcement rate

This stronger paired endpoint uses only active, false-profile-aligned soft
treatment turns as opportunities. An event requires all of the following on
the same matched turn:

1. the soft and balanced simulated choices differ;
2. the soft choice moves toward the initially false profile direction while
   the balanced choice does not; and
3. the evaluated updater strengthens that false direction both absolutely and
   beyond its exact same-history shadow.

The decomposition row retains
`behavioral_reinforcement_event_count`,
`behavioral_reinforcement_opportunity_count`, and the nullable
`behavioral_reinforcement_rate`. The rate is null with no opportunities. It is
not computed by the simulator-only manipulation audit: a local exact updater
evolves adaptive policy state, but there is no evaluated target-writer output.
A positive partial rate does not imply this paired
behavioral endpoint, and neither substitutes for the later-action clause in the
strict five-clause self-confirmation definition.

### Disconfirmation Inversion Rate

DIR is an updater-side sign-error diagnostic and is not a behavioral-loop
predicate. For every initially false attribute $j$ and turn $t$, let
$g^*_{tj}$ and $g^U_{tj}$ be the exact-shadow and evaluated-updater
false-direction log-odds gains. With the configured `direction_tolerance`
$\tau$,

\[
O_{tj}=\mathbb 1[g^*_{tj}< -\tau],\qquad
I_{tj}=O_{tj}\mathbb 1[g^U_{tj}>\tau].
\]

Then

\[
\operatorname{DIR}_{U,\pi}
=
\frac{\sum_{t,j}I_{tj}}{\sum_{t,j}O_{tj}}.
\]

`disconfirmation_opportunity_count`,
`disconfirmation_inversion_count`, and
`disconfirmation_inversion_rate` are retained together; the rate is null when
the denominator is zero. `disconfirmation_inversion_turn_rate` additionally
reports the fraction of turns with at least one inversion. No profile-action or
choice-change clause is added, because doing so would mix attribution with
behavioral feedback. DIR discards magnitude and is secondary to cumulative
excess confidence and the continuous same-history attribution gap.
Because the system and shadow priors can diverge after the first turn, DIR is
a path-dependent trajectory diagnostic. It does not by itself prove that the
current observation received an opposite one-step likelihood sign under a
common pre-turn prior.

### Deferred/stage-gated recovery after correction

When the stage-gated correction protocol is active, report terminal residual
error, turns to recovery, and area under the post-correction error curve
relative to the pre-reinforcement branch. The same explicit correction and
subsequent balanced contexts must be paired across reinforcement depths.
Failure to activate the correction stage leaves these outcomes missing, not
zero.

### Sensitivity manipulation checks

The primary sensitivity axis is visible policy strength and exposure:
profile-conditioning dose, assigned treatment rate, and paired visible-action
divergence. The \(\lambda=0\) cell is a negative control. A cell with
\(\lambda>0\) but zero visible-action divergence is labeled a failed
manipulation and is excluded from causal policy-dose interpretation.
Report the number of active treatments falling in the ex-ante `near_tie` and
`marginal` choice strata. The frozen manipulation-adequacy threshold is at
least two visibly divergent informative turns in a soft trajectory and at
least two qualifying incorrect-profile users in every domain. This is a
pre-response coverage rule, not an outcome filter or an independent sample-size
claim. A six-turn pilot is not silently promoted to the 12-turn paper design.
Every dose row reports exact expected preference information, exact realized
information and posterior-error improvement, the same-history attribution gap,
profile-consistency score and balanced advantage, ex-ante paired-choice
divergence probability, visible-action and realized-choice divergence, CEC,
DIR with its opportunity count, and terminal error. Monotonicity is not assumed:
a profile-consistent action can be informative to the exact updater while
remaining poorly interpreted by an evaluated LLM writer.

Decision-noise, ranking/default/suggestion susceptibility, and other numeric
response-model multipliers are robustness axes. They are not visible
interventions unless they change the rendered action or realized response.
Hard restriction is a stress test, not evidence for the soft-loop claim.
Recommendation wording is fixed; no wording-strength estimand is defined until
separately reviewed graded surfaces exist.
Ranking and default are fixed binary visible treatments. Numeric susceptibility
multipliers do not define graded visible-treatment strength.

This restriction applies to the closed-loop headline. Restricted choice
remains an eligible primary mechanism in the controlled same-response audit.

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
For initially false attributes this is exactly cumulative excess confidence
(CEC) in clipped log-odds units. The raw per-attribute vector remains
`cumulative_lcg`; `mean_cumulative_excess_confidence_log_odds` is its mean over
attributes whose initial seed places majority mass on the wrong sign.

The paired policy contrast used by Gate 2 is

\[
\Delta\operatorname{CEC}_{\mathrm{soft-bal}}
=\operatorname{CEC}_{\mathrm{soft}}
-\operatorname{CEC}_{\mathrm{balanced}},
\]

stored as `soft_minus_balanced_excess_confidence_log_odds` in paired
decomposition rows and clustered inference. A positive value is a **relative
confidence penalty**: soft retained more false-direction confidence than
balanced. It does not by itself show that false confidence grew in absolute
terms or that behavior reinforced it.

Interpret the endpoints in this order and do not collapse them:

- soft-policy CEC above zero is absolute excess confidence relative to the
  exact same-history shadow;
- soft-policy EAR above one means terminal error exceeds the deliberately
  incorrect initial profile;
- `reinforcement_event_rate` above zero is partial reinforcement and does not
  require a paired soft-versus-balanced choice change;
- `behavioral_reinforcement_rate` above zero additionally requires that paired
  choice change toward the false profile; and
- the strict five-clause self-confirming profile additionally requires later
  action influence and terminal persistence.

### Secondary false-stable and self-confirming rates

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

These metrics describe secondary Experiment C v1. The implemented regimes are
fixed balanced, fixed bias, and endogenous closed loop, evaluated on a shared
terminal battery. They do not imply an unimplemented factorial of balanced,
exploratory, and profile-conditioned logging policies.

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

The optional R harness filters controlled `same_response_provenance` rows to
the predeclared target writer and uses `calibration_residual` as Experiment A's
primary mixed-effects outcome. Its four primary contrasts compare restricted,
ranking, default, and suggested presentation with balanced presentation.
`exact_acue` is a required descriptive secondary magnitude; fitted-reference
ACUE is secondary learnability robustness. Experiment B's terminal-error
interaction does not alone establish all five self-confirmation clauses; it
also does not by itself establish the continuous-outcome and
selection–attribution decomposition claims.

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
Weak/strong latent-preference and prospectively computed balanced-choice-margin
strata are retained in analysis. Random seeds and trajectory replicates are
nested robustness observations and never increase the independent-user count.

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
