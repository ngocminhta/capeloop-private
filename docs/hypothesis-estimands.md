# H1, H2, and H7 estimands

This document freezes the executable decision quantities for proposal
hypotheses H1, H2, and H7. The implementation is
`src/cape_loop/experiments/hypothesis_estimands.py`. Experiment A writes
`metrics/experiment-a-hypothesis-estimands.json`; Experiment B writes
`metrics/experiment-b-h7-mitigation.json`.

These files always retain `claim_status = "not_claimed"`. A computed criterion
is not, by itself, a paper claim. Paper-grade sample adequacy, preregistration,
multiplicity handling, robustness checks, and author review remain separate.

## Shared pairing and uncertainty

The primary response mode is `controlled_anchor`. Within each mechanism, the
visible context, selected anchor, prior, user, domain, target attribute,
direction, and prior stratum are held fixed across updaters.

The independent unit is the complete latent user. Each contrast first forms a
paired difference within a matched trial, then averages rows within user, then
uses a deterministic percentile bootstrap over complete users. The default
interval is 95% with the run's configured bootstrap count. A smoke
configuration with zero configured replicates uses the runner's recorded
200-replicate fallback.

At least two independent user clusters are required for a computed criterion.
Insufficient clusters produce `criterion_met = null` and
`computed_status = "incomplete"` rather than a failed or passed claim.

The three policy-conditioned mechanisms are frozen as:

```text
restricted
default
suggested
```

## H1: directional over-update

ACUE is a distance from the fitted-aware update. A positive ACUE can arise from
over-updating, under-updating, or updating in the wrong direction, so an ACUE
contrast alone cannot establish H1.

For target attribute \(j\) and the controlled anchor direction \(s\), define:

\[
\Delta\ell_s(q)
=
\operatorname{logit} q_{t+1,j}(s)
-
\operatorname{logit} q_{t,j}(s),
\]

where \(q_j(s)\) is the total marginal mass on the two latent values with sign
\(s\). Clipping at \(10^{-6}\) is used only for this numerical log-odds
diagnostic; retained beliefs are unchanged.

For every mechanism \(m\), H1 emits two full-context-minus-aware estimands:

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

A mechanism meets the executable H1 criterion only when both cluster-bootstrap
lower bounds are strictly above zero. The H1 artifact is complete only when
restricted, default, and suggested mechanisms are all available with adequate
clusters; its joint criterion requires all three mechanism-wise criteria.
Mechanism rows remain reportable individually even when the joint artifact is
incomplete.

## H2: closer to action-unaware inference

H2 compares update vectors, not terminal posteriors or likelihood-model
training losses. Let \(\Delta q\) denote the 12-component vector of marginal
probability increments over three attributes and four values. For each matched
trial:

\[
d_A
=
\left\|
\Delta q^{\mathrm{full}}-\Delta p^A
\right\|_1,
\qquad
d_U
=
\left\|
\Delta q^{\mathrm{full}}-\Delta p^U
\right\|_1.
\]

The mechanism-wise proximity advantage is:

\[
\gamma_m = \mathbb E_{\mathrm{user}}[d_A-d_U].
\]

Positive values mean that the full-context writer is closer to fitted
action-unaware inference. A mechanism qualifies only when the 95%
complete-user bootstrap lower bound for \(\gamma_m\) is strictly above zero.
The H2 criterion requires at least two qualifying mechanisms, exactly matching
the proposal. The artifact also retains separate clustered intervals for
\(d_A\) and \(d_U\), the qualifying mechanism names, missing mechanisms, and
inadequate-cluster mechanisms.

The fitted-aware reference is retained on every Experiment A row. The
fitted-unaware updater row must also be present for H2; its absence leaves the
corresponding mechanism incomplete.

## H7: mitigation without loss of valid learning

H7 has three distinct components. None may substitute for another:

1. Experiment A update-error superiority under policy-conditioned evidence.
2. Retention of valid learning under balanced and volunteered evidence.
3. Experiment B reduction of closed-loop attribution error and
   self-confirming-profile rate.

### Update-error superiority

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

Positive values favor the provenance-aware mitigation. A mechanism qualifies
when the 95% complete-user bootstrap lower bound is strictly above zero. The
frozen criterion requires at least two qualifying policy-conditioned
mechanisms.

### Balanced and volunteered noninferiority

The positive-control outcome is the anchor- or statement-aligned directional
log-odds update. Let \(u_F\) be the ordinary full-context update and \(u_P\)
the provenance-aware update. The frozen retention fraction is:

```text
0.80
```

This permits at most a 20% relative loss in valid directional learning. For
each positive-control condition, the noninferiority estimand is:

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

- the unmitigated full-context positive-control update has a 95% lower bound
  strictly above zero, establishing that the control contains observable
  positive learning; and
- the 95% lower bound for \(\eta_c\) is at least zero.

Balanced outcomes come from the matched Experiment A rows. Volunteered
preferences are direct user-originated statements and cannot be represented
faithfully as a one-step option choice. They enter through explicit
`VolunteeredPreferenceUpdate` records with:

```text
case_id
user_id
updater_id
directional_log_odds_update
```

The ordinary and provenance-aware records are paired by `case_id` and must
share `user_id`. The provider-neutral direct-statement executor and immutable
review are implemented by `control-study h7-plan`, `h7-review`, and
`h7-verify`. They generate every test-user/domain/attribute case, require exact
accepted OpenAI or OpenRouter audit coverage, and convert each bound posterior
into these records. See
[H7 volunteered-preference controls](h7-volunteered-controls.md).

The ordinary run artifact remains incomplete until that external collection is
performed. The derived review is written outside the source run and recomputes
the volunteered and overall Experiment A H7 criteria without modifying the
source. If any case, updater arm, response hash, or provider-audit row is
missing or inconsistent, review fails. No value is inferred from balanced
choices or the six-control diagnostic battery.

### Closed-loop self-confirmation mitigation

Experiment B pairs ordinary full-context and provenance-aware trajectories by
their common-random-number key under:

```text
policy_id = "soft_profile_conditioned"
initial_profile_condition = "incorrect"
```

It emits two ordinary-minus-mitigated contrasts:

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
the existing five-clause self-confirmation definition; it is not reconstructed
from terminal error alone.

The Experiment B artifact explicitly notes that the Experiment A superiority
and both valid-learning controls are still required. Conversely, the
Experiment A artifact notes that the closed-loop component is still required.
A full H7 claim must join compatible, frozen paper runs and satisfy every
component.

## Artifact interpretation

`metrics/experiment-a-hypothesis-estimands.json` contains:

```text
schema_version
analysis
independent_unit
bootstrap_replicates
confidence_level
frozen_decision_constants
hypotheses.H1
hypotheses.H2
hypotheses.H7
claim_status
```

Each hypothesis or mechanism uses:

- `criterion_met = true` for a computed criterion that met its frozen rule;
- `criterion_met = false` for a computed criterion that did not;
- `criterion_met = null` when required rows, controls, or clusters are absent.

`metrics/experiment-b-h7-mitigation.json` contains the two closed-loop
contrasts, the frozen pairing condition, missing reason when applicable,
computed status, and `claim_status = "not_claimed"`.

An externally collected volunteered control produces
`h7-volunteered-review.schema.json`. It binds the verified Experiment A source,
the regenerated plan, response and provider-audit files, every
`VolunteeredPreferenceUpdate`, the provider-bound evidence rows, and the
recomputed H7 component. It always retains `claim_status = "not_claimed"`,
`source_run_modified = false`, and `missing_values_imputed = false`.

The schema version changes if a formula, pairing unit, margin, mechanism set,
or decision rule changes. Existing artifacts must never be silently
reinterpreted under a newer estimand.
