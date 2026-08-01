# Final Paper Proposal

## **You Chose What I Showed You: Policy-Dependent Evidence and Causal-Provenance Miscalibration in LLM User Profiles**

### **CAPE-Loop: Causal Attribution of Preference Evidence in Closed-Loop Agents**

**Paper type:** Analysis and evaluation paper
**Target:** ACL main conference
**Primary contribution:** Joint evaluation of persistent-profile updaters and the interaction policies that generate their evidence
**Not a contribution:** A new recommendation algorithm, memory architecture, user simulator, or personalization controller

> **Document status.** This is the scientific proposal and claim plan, not a
> results report. The repository implements the declared software protocols,
> but the paper-scale live-model, external-decoder, native-action, human, and
> confirmatory-analysis evidence is not checked in. Every `[TBD]` below is
> therefore intentional. See [Implementation status](implementation-status.md)
> for the current executable and evidence boundaries.

## One-sentence pitch

CAPE-Loop tests whether persistent-profile quality depends on the interaction
policy that generated its evidence, and whether LLM profile writers calibrate
each update to that evidence's causal provenance.

# Abstract

Personalized LLM agents increasingly update persistent user profiles from interaction outcomes such as choices, revisions, acceptances, and implicit feedback. These observations are not exogenous: the agent's current profile influences which alternatives are displayed, how they are ranked, what default is selected, and which option the agent recommends. A profile may therefore shape the evidence subsequently used to update that same profile.

We study **causal-provenance miscalibration**: a mismatch between the update a
profile writer applies and the evidential weight warranted by the context that
elicited the response. Miscalibration may be over-weighting, under-weighting,
or wrong-direction updating, and may differ by mechanism and model; it does
not require the writer to ignore context. For example, selecting an
inexpensive hotel from a balanced premium-versus-budget choice provides
stronger evidence of a general price preference than accepting the same hotel
when it was preselected, explicitly recommended, or presented alongside only
other inexpensive options.

We introduce **CAPE-Loop**, a controlled evaluation of action-conditioned preference inference and closed-loop profile formation. CAPE-Loop maintains a fixed latent user, explicitly logs the option set, ranking, default, recommendation, wording, and policy provenance that generated each response, and provides three inference references: a Bayes-optimal posterior under the declared response model, a fitted action-aware updater that learns the response model from training interactions, and a capacity-matched fitted action-unaware updater. Controlled anchor-option sets hold the selected item and response constant while varying only their causal provenance, while naturally sampled interactions evaluate average proper-score performance under the response distribution.

We then cross initial profile correctness, interaction policy, and memory
updater in closed loop. This separates **evidence-selection error**, where
profile-conditioned actions change the diagnostic evidence collected, from
**evidential-attribution error**, where profile writers assign incorrect weight
to the evidence that was collected. We call the resulting pair-level question
**policy-conditioned evidential legibility**: a history may remain informative
to exact action-aware inference while being harder for a particular profile
writer to translate into an accurate persistent profile. Every trajectory is
accompanied by an exact shadow action-aware posterior updated on the same
observations, enabling turn-level measurement of confidence gained beyond what
the interaction warrants. Finally, as a secondary evaluation-validity analysis, we compare
systems under fixed balanced histories, fixed mildly biased histories, and
endogenous closed-loop histories using an identical exogenous terminal
diagnostic battery.

Across `[UPDATERS]`, `[MODEL FAMILIES]`, `[DOMAINS]`, and `[TRAJECTORIES]`,
we find `[TBD]`. Relative to the exact action-aware update under the declared
generator, profile writers show `[TBD]` mechanism- and model-specific
calibration residuals; profile-conditioned interaction changes `[TBD]` in the
evidence stream; and false initial profiles show `[TBD]` continuous
amplification. Strict self-confirmation and fixed-versus-closed-loop selection
effects remain separately stage-gated. These results `[TBD INTERPRETATION]`.

# 1. Central scientific claim

The paper does **not** claim that the agent changes the user's underlying preferences.

The latent user state remains fixed during each trajectory:

$$
\theta_{t+1}=\theta_t=\theta.
$$

The paper studies **performative observation**:

$$
\text{current profile}
\rightarrow
\text{agent action}
\rightarrow
\text{policy-conditioned observation}
\rightarrow
\text{profile update}.
$$

The precise claim is:

> **Persistent-profile performance is a property of the
> updater–interaction-policy pair. Profile-conditioned actions can change the
> informativeness of later user evidence, and profile writers may assign that
> evidence mechanism- and model-specific weight that differs from action-aware
> inference.**

The claim hierarchy is:

1. **Primary pair-level claim:** persistent-profile accuracy is a property of
   the updater–interaction-policy pair. The primary closed-loop decomposition
   jointly reports policy contrasts in the exact same-history attribution gap
   and SelectionCost, the paired exact-shadow terminal-error contrast.
2. **Strong mechanism claim:** a policy-conditioned history can remain
   practically noninferior for exact-shadow terminal profile error—under the
   frozen SelectionCost margin—yet produce a larger attribution gap for a
   specific LLM profile writer. This is narrower than claiming equal mutual
   information. A separate nested result tests whether the evaluated writer's
   total soft-minus-balanced terminal error also exceeds a frozen practical-harm
   margin. Experiment A separately tests a candidate one-step provenance
   mechanism—model- and mechanism-specific over-weighting, under-weighting, or
   wrong-direction updating—without serving as a mediation analysis for B.
3. **Model-specific finding:** models may differ in whether they overreact,
   attenuate, or invert disconfirming evidence. Heterogeneity is retained
   rather than averaged into a universal LLM claim.
4. **Conditional downstream claims:** behavioral self-confirmation,
   logging-policy-dependent system selection, correction debt, native-system
   validity, and human evidence are secondary or stage-gated and are reported
   only if their separate protocols are executed and gates are met.

This is different from merely observing that recommendation policies create feedback loops. LLM recommender feedback loops, interactive preference elicitation, response-conditioned user feedback, and memory–action coupling already exist as research areas. Echoes in the Loop studies accumulated bias and self-reinforcing exposure in LLM recommender pipelines; PEPPER evaluates interactive preference elicitation; IEvoAgent models dependence between an agent response and subsequent user feedback; and PersonaAgent explicitly couples personalized memory, persona-mediated actions, and memory refinement. ([arXiv][1])

CAPE-Loop instead asks:

> **Given the same user response, does a persistent profile writer assign the correct amount of evidence after accounting for the choice environment and agent action that produced it?**

# 2. Defensible novelty statement

The paper should use the following language:

> Existing work has studied closed-loop recommendation, interactive preference elicitation, response-conditioned feedback, and memory–action adaptation. CAPE-Loop studies a distinct inferential problem: whether persistent conversational profile writers condition preference updates on the causal provenance of user signals. We combine identical-response provenance controls, learned and exact action-aware inference targets, explicit false-profile seeding, crossed selection–attribution interventions, and open-loop versus closed-loop evaluation under a shared terminal diagnostic distribution.

The novelty does **not** come from any one component. It comes from the complete causal audit:

$$
\text{elicitation context}
\rightarrow
\text{evidential weight}
\rightarrow
\text{persistent profile}
\rightarrow
\text{future elicitation}
\rightarrow
\text{self-confirmation}.
$$

# 3. Research Questions

## RQ1: Causal-provenance sensitivity

**Do persistent profile writers assign the warranted evidential weight to the
same response when it follows balanced exposure, restricted exposure, a
changed ranking, a default, or an agent-authored suggestion?**

The primary comparison is against the **exact action-aware posterior under the
declared generator**. The fitted action-aware updater is a secondary test of
learnability and reference-model misspecification.

## RQ2: Evidence selection versus evidence attribution

**How much closed-loop profile error arises because the agent changes the
evidence stream, and how much arises because the memory updater assigns
incorrect weight to the evidence that was collected?**

The crossed policy–updater design and shadow posterior provide the decomposition.

## RQ3: False self-confirmation

**Does an initially wrong profile amplify, lose corrective information, or
recover slowly when observations are generated by actions selected using that
profile?**

Continuous outcomes are primary. A false profile is called strictly
self-confirming only if it gains excess confidence, influences later actions,
and satisfies every other registered clause; mere persistence is insufficient.

## RQ4: Evaluation validity

**Do fixed-history evaluations rank profile writers differently from closed-loop evaluation, and can an open-loop benchmark select an updater that is materially worse after deployment?**

The most consequential secondary Experiment C result would be a credible
model-selection failure, not merely a score gap.

## Deferred RQ5: Pragmatic validity

**Do humans distinguish evidence produced through free choice from acceptance produced through restricted options, defaults, or agent suggestions more strongly than LLM profile writers do?**

Human judgments would validate the pragmatic evidential ordering; they are not
treated as access to metaphysical “true preferences.” Recruitment and human
evidence are deferred beyond the minimum paper and require ethics approval.

## Secondary RQ: Correction debt

**After a false profile has accumulated endogenous supporting observations, does the same explicit correction produce slower or incomplete recovery?**

This is stage-gated and excluded from the minimum paper.

# 4. Formal Setting

## 4.1 Latent user

Each domain contains three preference dimensions:

$$
\theta=(\theta_1,\theta_2,\theta_3),
$$

with:

$$
\theta_j\in\{-2,-1,+1,+2\}.
$$

The four values represent weak and strong preferences in either direction. This gives:

$$
|\Theta|=4^3
$$

latent preference profiles per domain while keeping exact posterior inference tractable.

The primary signed-error design does not add \(\theta_j=0\): a truly neutral
latent value would remove the directional truth needed by false-profile and
recovery estimands. “Uncertain” cases are instead represented prospectively by
weak preferences, small balanced-choice probability margins, or uncertain
initial beliefs. A neutral-user extension would require separately defined
non-directional outcomes.

Each user also has a susceptibility vector:

$$
\psi=
\langle
\psi_{\text{rank}},
\psi_{\text{default}},
\psi_{\text{suggest}}
\rangle,
$$

representing heterogeneous sensitivity to presentation effects.

The primary paper holds $\theta$ fixed during each trajectory.

## 4.2 Action context and policy provenance

The action context visible to the user is:

$$
C_t=
\left\langle
\mathcal X_t,
r_t,
d_t,
w_t,
s_t,
q_t
\right\rangle,
$$

where:

* $\mathcal X_t$: displayed options and attributes;
* $r_t$: ranking or position;
* $d_t$: selected default;
* $w_t$: wording or framing;
* $s_t$: explicit agent suggestion;
* $q_t$: question or elicitation type.

Policy provenance is logged separately:

$$
P_t=
\left\langle
\pi_t,
M_t,
\text{policy version}
\right\rangle.
$$

This distinction is important. $C_t$ contains what affected the user's response; $P_t$ records why the agent produced that context and is used to diagnose self-confirmation.

## 4.3 User response model

The primary user model is a finite random-utility model:

$$
Y_t=
\arg\max_{x\in\mathcal X_t}
\left[
\beta\theta^\top\phi(x) +
\delta_\psi(C_t,x) +
\varepsilon_{t,x}
\right],
$$

where:

* $\phi(x)$ contains item attributes;
* $\theta^\top\phi(x)$ is intrinsic user utility;
* $\delta_\psi(C_t,x)$ contains ranking, default, and suggestion effects;
* $\varepsilon_{t,x}$ is decision noise.

Equivalently:

$$
P(Y_t=x\mid \theta,\psi,C_t)
\propto
\exp
\left(
\beta\theta^\top\phi(x) +
\delta_\psi(C_t,x)
\right).
$$

Presentation effects alter choice probability, but not underlying welfare. Intrinsic utility is:

$$
u_\theta(x)=\theta^\top\phi(x).
$$

User regret is evaluated against the complete feasible option pool:

$$
\operatorname{Regret}_t=
\max_{x\in\mathcal X_t^{\text{full}}}
u_\theta(x) -
u_\theta(Y_t).
$$

This prevents a strong default from being mistaken for high user utility.

The implemented user-facing surface is hybrid. The equation above fixes
\(Y_t\) first. A separately frozen, outcome-blind authoring input supplies one
neutral candidate presentation and neutral display names per scenario. The
current 48 visible bases were subsequently project-standardized outcome-blind
onto three source-neutral frames, with each frame balanced across test cells.
Code uses each base for balanced, restricted, and ranking; inserts only the
fixed default or suggestion sentence when required; and states exactly
`I choose {selected_name}.` The author receives neither \(\theta\) nor \(\psi\)
and cannot change the choice. The evaluated profile writer sees natural
dialogue and a semantic attribute codebook rather than numeric feature vectors
or the experiment's target index.

## 4.4 Exact action-aware posterior

The exact reference is explicitly named:

> **Bayes-optimal posterior under the declared user-response model.**

It is:

$$
p_{t+1}^{\text{oracle}}(\theta,\psi)
\propto
p_t^{\text{oracle}}(\theta,\psi)
P(Y_t\mid\theta,\psi,C_t).
$$

The oracle knows the declared model family and simulator coefficients but not the latent user.
Its preference prior is the prospectively supplied experimental belief, and
its susceptibility prior is uniform over the finite support prospectively
assigned to the evaluation split. The split-support declaration is part of the
generator; the realized user's susceptibility, finite-sample frequencies, and
outcomes are not used to set the prior.

For Experiment A, each truth-aligned prior-strength stratum is constructed by
mixing truth mass into each attribute marginal and taking their independent
joint:

$$
q_{s,j}(v)=\frac{1-s}{4}+s\,\mathbf 1[v=\theta_j],
\qquad
q_s(\theta')=\prod_{j=1}^{3}q_{s,j}(\theta'_j),
\qquad 0\le s<1.
$$

The exact updater can therefore reconstruct no more preference-prior
information than the three marginal vectors included in the LLM request.

It is the primary within-generator reference, not a universal normative theory
of human choice.

## 4.5 Fitted action-aware and action-unaware updaters

The fitted action-aware updater learns:

$$
\widehat P_{\text{aware}}(Y\mid\theta,C)
$$

from randomized training trajectories. It estimates population-level presentation effects and, where used, a distribution over susceptibility types.

The capacity-matched action-unaware updater learns:

$$
\widehat P_{\text{unaware}}(Y\mid\theta)
$$

from the same training observations but does not receive the option set, ranking, default, or suggestion.

This comparison answers:

> Is causal-provenance adjustment learnable from the available interaction data even without knowing the true simulator coefficients?

The primary one-step analysis uses the exact action-aware posterior from the
same declared model that generated the simulated users. This removes fitted
response-model misspecification from the controlled attribution estimand. It
estimates model- and mechanism-specific calibration intercepts, slopes,
residual errors, and contrasts; neither the presence nor the sign of
miscalibration is assumed.

The fitted action-aware reference remains valuable as a secondary learnability
and misspecification-robustness analysis. The fitted action-unaware reference,
directional over-update contrast, and action-unaware-proximity contrast are
secondary diagnostics. They are not the primary hypotheses and must not be
used to relabel all contextual errors as “provenance blindness.”

# 5. Two Evaluation Tracks

## Track A: Controlled structured-belief updating

Every updater receives:

* the same prior distribution;
* the user response;
* previous interaction history where applicable; and
* exactly its declared information view: response-only, full visible context,
  or full context plus policy provenance.

It emits:

```json
{
  "beliefs": {
    "attribute_1": {
      "-2": 0.10,
      "-1": 0.20,
      "+1": 0.45,
      "+2": 0.25
    },
    "attribute_2": {
      "-2": 0.30,
      "-1": 0.40,
      "+1": 0.20,
      "+2": 0.10
    },
    "attribute_3": {
      "-2": 0.15,
      "-1": 0.25,
      "+1": 0.35,
      "+2": 0.25
    }
  }
}
```

This track supports:

* Brier score;
* negative log-likelihood;
* posterior divergence;
* update magnitude;
* confidence gain;
* calibration.

LLM temperature parameters are fit only on development users. Experiment A
uses the raw returned vector for every primary update and retains the
temperature-scaled vector only as a secondary forecast-calibration diagnostic;
this prevents calibration from manufacturing an update when a model simply
returns its prior. B/C use the configured active vector for their realized
histories and retain paired raw/calibrated terminal diagnostics. `none` remains
an explicit uncalibrated ablation.

## Track B: Native persistent memory

The current implementation evaluates three native updater families:

* `episodic_memory`;
* `semantic_memory`, including its policy-facing persona projection; and
* `provenance_linked_memory`.

Structured probabilistic updaters remain in Track A rather than being relabeled
as native memory. Every eligible Track B trajectory retains its complete native
state and the common held-out terminal suite.

Native states are evaluated through:

1. two separately executed blinded profile decoders from distinct model
   families;
2. the common terminal behavioral battery;
3. native end-to-end actions produced directly from the retained state.

No main conclusion may depend on only one decoder.
Distinct model families do not imply statistically independent errors,
especially when both executions share a gateway or other infrastructure.

The selected external-decoder protocol uses
`anthropic/claude-sonnet-5` and `google/gemini-3.6-flash` through the shared
OpenRouter gateway. The selected native action adapter is
`cape-loop-openai-native-agent-v1`, backed by OpenAI `gpt-5.6-sol`. These are
versioned evaluation choices, not claims about all models from those
developers. Direct Anthropic and Gemini transports remain optional
first-party-origin replications.

Track B prevents the paper from becoming merely a numerical Bayesian-reasoning benchmark.


# 6. CAPE-Loop Domains

## 6.1 Travel planning

Preference dimensions:

* budget versus premium accommodation;
* central location versus comfort;
* convenience versus flexibility.

Options are hotels, itineraries, or transport plans with controlled attribute vectors.


## 6.2 Writing assistance

Preference dimensions:

* concise versus detailed writing;
* formal versus conversational style;
* British versus American spelling.

Options are matched draft variants or revision strategies.

This domain makes the work distinctly relevant to language interaction rather than only recommendation ranking.


# 7. Provenance Mechanisms

Experiment A uses one baseline and four independently named treatments:

| Condition | Manipulation | Inferential issue |
| --- | --- | --- |
| Balanced | Matched counter-direction options with neutral order | Baseline free-choice evidence |
| Restricted | The peer option has the same target direction | The missing counter-direction cannot be selected |
| Ranking | Both directions remain available but order changes | Selection partly reflects position |
| Default | Both directions remain available but the anchor is preselected | Acceptance partly reflects inertia |
| Suggested | Both directions remain available but the agent recommends the anchor | Acceptance is partly agent-authored |

Positive controls include:

* explicit volunteered preference;
* repeated balanced cross-context choices;
* direct correction.

Negative controls include:

* indifferent responses;
* random choices;
* responses that do not distinguish the target preference.

Restricted choice is a primary one-step attribution treatment in Experiment A.
Hard option filtering is only a stress test for the closed-loop claim: any
strict self-confirmation claim must appear under **soft conditioning**, where
counter-profile alternatives remain available.


# 8. Matched Provenance Construction

## 8.1 Anchor-option design

Each matched set contains an identical anchor option selected in every condition.

| Condition  | Displayed options                           | Default | Suggestion | Selected item |
| ---------- | ------------------------------------------- | ------- | ---------- | ------------- |
| Balanced   | Anchor budget hotel + matched premium hotel | None    | None       | Anchor        |
| Restricted | Anchor budget hotel + second budget hotel   | None    | None       | Anchor        |
| Ranking    | Same balanced pair in the treatment order   | None    | None       | Anchor        |
| Default    | Anchor budget hotel + matched premium hotel | Anchor  | None       | Anchor        |
| Suggested  | Anchor budget hotel + matched premium hotel | None    | Anchor     | Anchor        |

The selected item and surface response are identical. Only the evidential context changes.

Every included response must satisfy:

$$
P_\theta(Y\mid C^{(k)})>\varepsilon
$$

under all matched conditions. This excludes nearly impossible counterfactual responses that would artificially magnify posterior differences.

## 8.2 Two complementary versions

### Controlled identical-response audit

Hold the anchor choice and response fixed.

Purpose:

* isolate functional sensitivity to provenance;
* make the result interpretable;
* compare update strength across contexts.

This is the primary `same_response_provenance` analysis. It is a controlled
functional diagnostic, not an average causal effect. The implementation audits
that the local user sentence is literally invariant across all five matched
conditions; a failed invariant makes the matched set ineligible.
Held-out paraphrase transfer is evaluated in this controlled-anchor track, so
the public controlled-only live configuration still tests wording robustness
without mixing in natural-response variation.

### Naturally sampled evaluation

Sample responses from:

$$
P_\theta(Y\mid C)
$$

independently in each action context.

This is a secondary realism analysis. Its purpose is to:

* measure average proper-score performance;
* verify that the finding persists under realistic response frequencies;
* prevent conclusions from depending on selected identical-response cases.

# 9. Experiment A: How Is the Profile Writer Calibrated to Causal Provenance?

## Objective

Estimate how the applied profile update differs from the action-aware warranted
update across provenance mechanisms, including over-, under-, and
wrong-direction updating.

## Systems

| Updater                           | Purpose                                                 |
| --------------------------------- | ------------------------------------------------------- |
| Prior/no update                   | Conservative lower reference                            |
| Exact action-aware oracle         | Reference under declared simulator                      |
| Fitted action-aware updater       | Learnable causal reference                              |
| Fitted action-unaware updater     | Diagnostic baseline that omits provenance-bearing context |
| Rule-based provenance discounting | Simple non-LLM baseline                                 |
| Response-only LLM                 | Tests surface-response inference                        |
| Full-context LLM                  | Tests whether ordinary dialogue context is sufficient   |
| Provenance-aware LLM              | Diagnostic mitigation with explicit structured metadata |

## Primary estimand and metrics

For anchor direction \(s\) on target attribute \(j\), let
\(p^{*}\) be the exact action-aware posterior under the declared generator.
All Experiment A updaters begin from the same supplied prior
\(p_t^*=q_t=q_0\). Define the warranted and system log-odds updates:

$$
u^* =
\operatorname{logit}p^*_{t+1,j}(s)
-
\operatorname{logit}p^*_{t,j}(s),
$$

$$
\widehat u =
\operatorname{logit}q_{t+1,j}(s)
-
\operatorname{logit}q_{t,j}(s),
\qquad
r=\widehat u-u^*.
$$

The signed residual \(r\) distinguishes over-weighting (\(r>0\)),
under-weighting (\(r<0\)), and wrong-direction behavior when interpreted with
\(u^*\). It is reported by updater and mechanism, not pooled into a universal
“blindness” label.

For each updater \(k\) and mechanism \(m\), estimate:

$$
\widehat u_{k,m}
=
\alpha_{k,m}+\beta_{k,m}u^*+\varepsilon.
$$

The ideal exact-oracle calibration curve has
\((\alpha,\beta)=(0,1)\) and low residual RMSE. Curves with insufficient
within-cell variation are marked not estimable. The primary matched contrast
compares the target writer's signed residual with the balanced condition:

$$
\Delta r_m
=
\mathbb E[
r_{\mathrm{target},m}
-
r_{\mathrm{target},\mathrm{balanced}}
],
$$

As a secondary unsigned magnitude diagnostic, report:

$$
\Delta e_m
=
\mathbb E[
|\widehat u_{\mathrm{target},m}-u^*_{m}|
-
|\widehat u_{\mathrm{target},\mathrm{balanced}}
-u^*_{\mathrm{balanced}}|
].
$$

### Exact action-conditioned update error

For the complete 12-component marginal update vector,

$$
\operatorname{ExactACUE}_t=
\left\|
(q_{t+1}-q_t) -
(p^*_{t+1}-p^*_t)
\right\|_1.
$$

This measures unsigned error in the applied evidence increment. It is a
secondary full-vector magnitude diagnostic for the controlled-anchor track;
the signed calibration residual is the primary mixed-effects outcome.

The legacy field `acue` compares with the fitted action-aware updater and is
retained as a secondary robustness metric. It must remain explicitly labeled
as fitted-reference rather than being pooled with `exact_acue`.

### Posterior divergence

$$
D_{\mathrm{KL}}
\left(
p_{t+1}^*
;|;
q_{t+1}
\right).
$$

Fitted-aware divergence is retained separately as secondary robustness.

### Excess proper-score error

$$
\operatorname{ExcessBrier}=
BS(q,\theta)-BS(p^*,\theta).
$$

This exact-reference quantity is derived from retained beliefs. The legacy
emitted `excess_brier` field uses the fitted-aware reference and remains
secondary.

### Update-direction accuracy

Did each attribute probability move in the correct direction?

### Secondary fitted-reference diagnostics

The fitted action-aware updater is learned only from randomized training
interactions. Repeating the update-error and calibration analyses against this
reference tests whether the exact-oracle result survives a learnable
approximation. Directional over-update relative to that fit and proximity to
the fitted action-unaware updater remain descriptive diagnostics; they are not
the primary hypothesis.

### Evidence-strength ranking

Does the updater distinguish evidence strength in the direction warranted by
the exact declared response model:

$$
\text{volunteered preference}
>
\text{balanced choice}
>
\text{suggested/defaulted acceptance}
>
\text{restricted choice}?
$$

No universal middle ordering is hard-coded. Ranking, default, and suggestion
effects depend on the declared mechanism and susceptibility; fitted-model and
human comparisons are secondary validation.

# 10. Experiment B: Policy-Conditioned Evidential Legibility and Behavioral Feedback

Unlike the same-response attribution track, Experiment B samples each response
from the declared model after the policy chooses the visible context. The
policy may therefore change the realized response and future evidence stream.

The paper keeps two causal arms distinct:

* **fixed-response attribution arm (Experiment A):** hold the selected option
  and user reply fixed while changing option restriction, rank, default,
  suggestion, or other registered provenance. This identifies
  $C_t\rightarrow M_{t+1}$. Ranking is already one of the five required arms.
* **natural-response feedback arm (Experiment B):** allow the displayed action
  to alter the sampled response before the profile is updated. This identifies
  the complete $C_t\rightarrow Y_t\rightarrow M_{t+1}$ path and stratifies
  users prospectively by weak/strong preferences and near-tie, marginal, or
  decisive balanced-choice margins.

An attribution failure with unchanged choices is evidence for the first path,
not for behavioral self-confirmation. Conversely, observed choice divergence
without an updater attribution gap is a policy-side behavioral effect. The two
arms therefore do not share one success criterion.

A four-way source-attribution ablation—full dialogue; user response plus
structured action metadata; assistant action without the response; and
neutralized assistant wording—is reserved for a versioned mechanistic
extension. The current provenance-aware updater changes both metadata and its
normative instruction, so it cannot identify a pure source-label effect.
Silently interpreting that condition as the proposed ablation is prohibited.

## 10.1 Factorial design

Cross:

$$
\text{Initial Profile}
\times
\text{Interaction Policy}
\times
\text{Updater}.
$$

### Initial profiles

* correct;
* incorrect;
* uncertain;
* empty.

The latent population is prospectively stratified by weak
\((|\theta_j|=1)\) and strong \((|\theta_j|=2)\) preferences. Balanced-choice
margin strata are computed before model outcomes are inspected so that
near-indifferent, marginal decisions can be analyzed separately from easy
choices. Random seeds create robustness replicates of a user/trajectory; they
are not counted as additional independent users.

### Interaction policies

* balanced/randomized;
* softly profile-conditioned;
* block-balanced exploratory, which uses current uncertainty to order the
  least-exposed dimensions while covering all three once per complete
  three-turn block.

Hard filtering is a secondary stress condition.

### Core updaters

* fitted action-aware;
* fitted action-unaware;
* full-context LLM;
* provenance-aware LLM;
* recurrent profile consolidator.

The exact oracle is a reference rather than a deployed system.

## 10.2 Soft profile conditioning

The softly conditioned policy does not remove all contradictory options. Instead, it may:

* place the profile-consistent option first;
* assign it as default;
* explicitly recommend it;
* select an option set containing both preference directions but with more profile-consistent framing.

This prevents the main result from being a trivial consequence of forced choice.

The bounded calibration uses six turns only after an outcome-blind planner
admits every balanced/soft trajectory pair. Before any evaluated-model call,
each schedule must contain at least two visibly divergent near-tie/marginal
active turns, one decisive active control, two presentation mechanisms,
retained options in both preference directions, and direction-robust active
susceptibility mass above the frozen threshold. Informative admission uses the
minimum predicted choice-divergence probability across either possible current
profile direction; decisive-control admission uses the maximum. Thus a writer
that crosses zero during the trajectory does not invalidate the ex-ante
susceptibility/control bound, while the treatment still follows its current
profile rather than a stale seed.

The planner may use only the scenario catalog, latent user, initial profile,
declared response model, and semantic seed. Realized choices, updated profiles,
and evaluated-model outputs are forbidden admission inputs. Runtime verifies
the scenario, mechanism, current-profile promotion, visible divergence, option
support, direction-specific prediction, and conservative bound. A 32-seed
simulator-only audit then reports ASM, choice divergence, expected information,
SelectionCost, and coverage without changing plan admission. A local exact
action-aware updater evolves adaptive policy state, but the audit cannot report
target-writer behavioral reinforcement because no evaluated LLM output is
present. Failure
suppresses the trajectory-level mechanism claim; it never licenses dropping
trajectories, rerunning a favorable seed, or changing the rule after outcomes
are inspected. A longer horizon may be frozen only from outcome-blind review or
audit evidence.

## 10.3 Shadow action-aware posterior

For every trajectory generated by updater $U$, maintain an exact
same-history action-aware belief:

$$
p_{t+1}^{\text{shadow}}=
\operatorname{UpdateAware}
\left(
p_t^{\text{shadow}},C_t,Y_t
\right).
$$

The shadow updater uses the declared generating response model and observes
exactly the same actions and responses, but does not control the policy.

This yields a turn-level decomposition.

Evidence-selection cost and cumulative action-aware information gain
characterize policy-induced evidence quality. Evidential-attribution cost
measures additional same-history updater error. The explicitly named
soft-minus-reference attribution-gap contrasts measure policy-conditioned
evidential legibility. DIR and the five-clause predicate are separate stronger
diagnostics, not prerequisites for an evidence-selection result.

### Evidence-selection cost

$$
\operatorname{SelectionCost}=
\operatorname{Err}
\left(
p_T^{\text{shadow, profile-policy}},
\theta
\right) -
\operatorname{Err}
\left(
p_T^{\text{shadow, balanced}},
\theta
\right).
$$

This is the policy-induced difference in realized exact-shadow terminal error
under the paired design. A positive value means the soft history was worse for
the registered terminal loss; it does not by itself prove lower mutual
information or identify which action caused the difference.

### Evidential-attribution cost

$$
\operatorname{AttributionCost}_{U,\pi}=
\operatorname{Err}
\left(
q_T^{U,\pi},
\theta
\right) -
\operatorname{Err}
\left(
p_T^{\text{shadow},\pi},
\theta
\right).
$$

This compares the system and action-aware updater on the same history.

For compact notation, define the policy-specific same-history attribution gap

$$
G_{U,\pi}
=
\operatorname{Err}(q_T^{U,\pi},\theta)
-
\operatorname{Err}(p_T^{\text{shadow},\pi},\theta).
$$

The primary prospectively schedule-matched Experiment B contrast is

$$
\Delta G_{\text{soft-bal}}
=G_{U,\text{soft}}-G_{U,\text{balanced}}
$$

The supporting whole-policy comparator is

$$
\Delta G_{\text{soft-exp}}
=G_{U,\text{soft}}-G_{U,\text{exploratory}}.
$$

These contrasts operationalize policy-conditioned evidential legibility for a
particular updater. A positive contrast means that the soft-policy history was
translated less accurately relative to its own exact diagnostic content. The
exploratory policy intentionally chooses targets and scenarios adaptively, so
its contrast is not a turn-matched causal branch and is outside the primary
matched family. Neither contrast, by itself, establishes that the user response
changed or that a behavioral feedback loop occurred.

The stronger statement that soft histories are **practically noninferior for
exact inference yet less legible to the evaluated writer** is a registered
conjunction, not an interpretation of $\Delta G$ alone. For the balanced
comparison it requires one-sided complete-user evidence for:

1. a positive soft-policy same-history gap $G_{U,\mathrm{soft}}$;
2. a positive $\Delta G_{\mathrm{soft-bal}}$; and
3. exact-shadow terminal-error noninferiority,
   $\operatorname{SelectionCost}<\epsilon_{\mathrm{sel}}$, with the frozen
   marginal-Brier margin $\epsilon_{\mathrm{sel}}=0.02$.

The margin makes this a practical terminal-error criterion, not an equality or
generic equality-of-information claim. Expected preference information,
realized whole-state information, and exact-shadow error improvement are
reported separately to characterize why the criterion holds or fails. The
one-sided complete-user sign-flip decisions control this conjunction;
percentile bootstrap intervals are sensitivity summaries rather than the gate
decision rule.

### Policy-conditioned attribution-gap contrast

For incorrect initial profiles:

$$
\begin{aligned}
\operatorname{SCI}_{\text{wrong}}=&
\left[
\operatorname{Err}
\left(
q_T^{U,\pi_{\text{profile}}},
\theta
\right) -
\operatorname{Err}
\left(
p_T^{*,\pi_{\text{profile}}},
\theta
\right)
\right]\\
&-
\left[
\operatorname{Err}
\left(
q_T^{U,\pi_{\text{balanced}}},
\theta
\right) -
\operatorname{Err}
\left(
p_T^{*,\pi_{\text{balanced}}},
\theta
\right)
\right].
\end{aligned}
$$

A positive value means that the updater's same-history attribution gap is
larger under profile-conditioned evidence collection. The implementation
retains the historical field name `self_confirmation_interaction` as a
compatibility alias for the soft-minus-balanced arithmetic, but the paper must
not call this contrast behavioral self-confirmation. For the same updater under
the two policies, the total paired terminal-error effect obeys the accounting
identity:

$$
\begin{aligned}
\Delta\operatorname{Err}_{U,\text{soft-bal}}
&=
\operatorname{Err}(q_T^{U,\pi_p},\theta)
-
\operatorname{Err}(q_T^{U,\pi_b},\theta)\\
&=
\operatorname{SelectionCost}
+
\left[
\operatorname{AttributionCost}_{U,\pi_p}
-
\operatorname{AttributionCost}_{U,\pi_b}
\right]\\
&=
\operatorname{SelectionCost}
+
\Delta G_{\text{soft-bal}}.
\end{aligned}
$$

The implementation stores this total as
`soft_minus_balanced_terminal_error`, reconstructs it from SelectionCost plus
`soft_minus_balanced_attribution_gap`, and retains the residual and invariant
status. Emitted Experiment B rows must satisfy the identity within the declared
numerical tolerance. This is an arithmetic check, not evidence that the total
or either component is positive.

When the comparator is instead the exact balanced shadow, the related identity
is:

$$
\begin{aligned}
&\operatorname{Err}(q_T^{U,\pi_p},\theta)
-
\operatorname{Err}(p_T^{*,\pi_b},\theta)\\
&\qquad =
\operatorname{SelectionCost}
+
\left[
\operatorname{AttributionCost}_{U,\pi_p}
-
\operatorname{AttributionCost}_{*,\pi_b}
\right],
\end{aligned}
$$

where \(\operatorname{AttributionCost}_{*,\pi_b}=0\) because the balanced
comparator is the exact shadow. This second contrast is not the updater's
soft-minus-balanced total effect and must not be substituted for it.

Correct-profile seeds are an essential negative/control condition. The planned
moderation contrast is

$$
\Delta G_{\text{soft-bal}}^{\text{incorrect}}
-
\Delta G_{\text{soft-bal}}^{\text{correct}}.
$$

It tests whether the policy–updater interaction is specifically larger when
the initial profile is wrong rather than being a generic consequence of the
presentation policy.

## 10.4 Action-level policy characterization

Policy labels are not treated as explanations. Every action records three
descriptive quantities before the natural response is observed:

1. `profile_consistency_score` in `[-1, 1]`, the mean signed alignment of
   option-set composition, first rank, default, and suggestion with the current
   profile direction; its paired balanced-counterfactual difference is also
   retained;
2. exact expected preference information gain,

   $$
   \operatorname{EIG}(C_t)
   =H[p^*(\theta\mid H_t)]
   -\mathbb E_{Y_t}H[p^*(\theta\mid H_t,C_t,Y_t)],
   $$

   distinguished from the realized joint preference–susceptibility entropy
   reduction after one sampled response; and
3. for paired binary actions with identical option IDs, the exact shared-noise
   probability that the action changes the selected option relative to the
   balanced counterfactual. Random utility uses shared Gumbels and the
   rule-based sensitivity model uses a shared inverse-CDF draw. The field is
   null unless both policies expose the identical binary choice set rather
   than receiving an invented probability. Comparable-turn count/rate and the
   distinct choice-set-divergence rate are retained beside its conditional
   trajectory mean.

Together with visible-action and realized-choice divergence, these quantities
separate profile-consistent/informative, profile-consistent/uninformative,
profile-neutral/informative, and profile-neutral/uninformative actions. They
remain separate observables; no unregistered composite policy-quality score is
formed.

## 10.5 False self-confirmation and disconfirmation inversion

Let $W_j$ be the incorrect preference direction encoded by the initial seed for dimension $j$.

Define:

$$
q_t^{\text{wrong},j}=
P_{q_t}
\left(
\theta_j\in W_j
\right).
$$

The system's false confidence gain is:

$$
FCG_j=
\operatorname{logit}
q_{t+1}^{\text{wrong},j} -
\operatorname{logit}
q_t^{\text{wrong},j}.
$$

The **Laundered Confidence Gain** is:

$$
LCG_j=
FCG_j^{\text{system}} -
FCG_j^{\text{shadow-aware}}.
$$

For initially false attributes, policy-specific **cumulative excess
confidence** (CEC) is the mean cumulative LCG in clipped log-odds units. Its
paired policy contrast is

$$
\Delta\operatorname{CEC}_{\text{soft-bal}}
=
\operatorname{CEC}_{\text{soft}}
-
\operatorname{CEC}_{\text{balanced}}.
$$

This contrast is the **relative confidence penalty** used by Gate 2: a positive
value means soft interaction retained more false-direction confidence than
balanced interaction. It is not, by itself, absolute amplification or
reinforcement. The interpretation hierarchy remains explicit:

1. soft-policy CEC above zero is absolute excess confidence relative to the
   exact same-history shadow;
2. soft-policy EAR above one is terminal-error amplification relative to the
   deliberately incorrect seed;
3. a positive partial reinforcement-event rate records treated turns on which
   the false direction was selected and strengthened beyond the shadow, but
   does not require the paired balanced choice to change; and
4. behavioral reinforcement additionally requires a same-turn
   soft-versus-balanced choice change toward the false-profile direction.

An episode counts as a false self-confirming profile only when:

1. the profile remains materially wrong;
2. probability mass on the wrong preference increases;
3. cumulative LCG is positive beyond a declared threshold;
4. the strengthened profile affects subsequent actions;
5. the action-aware shadow does not acquire equivalent confidence.

This excludes inert false memories.

Disconfirmation inversion is reported separately from behavioral
self-confirmation. For every initially false attribute $j$ and turn $t$,
with the registered direction tolerance $\tau$, define

$$
O_{tj}=\mathbb{1}
\left[FCG^{\text{shadow}}_{tj}< -\tau\right]
$$

and

$$
I_{tj}=O_{tj}\,
\mathbb{1}\left[FCG^{\text{system}}_{tj}>\tau\right].
$$

The **Disconfirmation Inversion Rate** is

$$
\operatorname{DIR}=\frac{\sum_{t,j}I_{tj}}{\sum_{t,j}O_{tj}},
$$

and is undefined when the exact shadow has no disconfirmation opportunity.
Opportunity and inversion counts are always reported beside the rate. DIR asks
whether the exact updater reduced confidence in a false sign while the
evaluated writer increased confidence in that same sign. It deliberately does
not require profile influence or changed behavior; adding either condition
would conflate attribution error with the downstream feedback loop. DIR is a
secondary sign-error diagnostic because it discards magnitude; cumulative
excess confidence remains the continuous counterpart.
After the first turn, system and shadow priors can differ. DIR is therefore a
path-dependent trajectory diagnostic, not proof that one current observation
was assigned an opposite one-step likelihood sign from a common prior.

## 10.6 Outcome hierarchy

The sole primary within-model claim is the Gate 3 conjunction of:

* the soft-policy same-history attribution gap $G_{U,\mathrm{soft}}$;
* the prospectively schedule-matched soft-minus-balanced attribution-gap
  contrast; and
* evidence-selection cost under its frozen noninferiority margin.

The multiplicity-controlled secondary claims are Gate 2's behavioral-feedback
conjunction, incorrect-minus-correct seed moderation, and the total
soft-minus-balanced updater terminal-error effect for nested net harm. The
total effect is checked against the exact SelectionCost-plus-$\Delta G$
identity. Soft-minus-exploratory remains a supporting whole-policy comparator.

Supporting continuous outcomes characterize where those pair-level effects
may arise:
policy-specific terminal/initial error-amplification ratio (EAR), absolute
cumulative excess confidence in wrong-direction log odds, exact-shadow information-gain and
disconfirmation-evidence deficits, paired visible-action and realized-choice
divergence, terminal Brier error, exact-posterior divergence, expected and
realized action-aware information gain, action profile consistency, ex-ante
behavioral susceptibility, partial reinforcement-event rate,
preference-dimension coverage, option diversity, intrinsic regret, and
terminal behavioral accuracy. Recovery after correction is included only when
its separate stage-gated protocol is activated.

The paired soft-minus-balanced CEC contrast is reported as a relative
confidence penalty. It controls the continuous-confidence clause of Gate 2 but
does not substitute for absolute soft-policy CEC, EAR, partial or paired
behavioral reinforcement, or the strict five-clause endpoint.

DIR, false-stable, and strict five-clause self-confirming-profile rates are
secondary endpoints. A null strict rate does not erase an updater-side
calibration error, an information deficit, or continuous amplification.

# 11. Experiment C: Is Updater Evaluation Logging-Policy-Dependent?

Experiment C v1 is a secondary evaluation-validity analysis. It can strengthen
the paper if it reveals robust ranking or selection consequences, but the
central causal-provenance claim does not depend on that result.

Experiment C treats performance as
\(\operatorname{Score}(U,\pi_{\mathrm{log}})\), rather than as a
policy-independent property \(\operatorname{Score}(U)\).

## 11.1 Three evidence-collection regimes

### Fixed balanced logging

Every updater receives the same trajectory generated by a randomized balanced policy.

### Fixed mildly biased logging

Every updater receives the same trajectory generated from a fixed reference profile unrelated to the updater being evaluated.

### Endogenous closed loop

Each updater's profile controls future actions through the common policy function.

The first two test sensitivity to the logging distribution. The third tests genuine endogeneity.

The current executable v1 design implements balanced and fixed-bias logging
plus soft endogenous deployment. Generalizing it to arbitrary adaptive
logging-policy × logger-updater pairs, including exploratory and separately
randomized loggers, requires a versioned Experiment C regime and review-schema
migration; it is not implied by the existing three-regime artifacts.

## 11.2 Realistic updater configurations

Ranking analysis should use several practical systems, excluding the Bayesian references:

1. response-only profile writer;
2. full-dialogue profile writer;
3. provenance-aware profile writer;
4. conservative threshold writer;
5. recurrent semantic-profile consolidator;
6. raw episodic memory with query-time profile inference;
7. structured provenance-linked memory.

These may share the same underlying language model so that ranking reflects memory/update design rather than unrelated model capability.

## 11.3 Common terminal diagnostic battery

At the end of every trajectory, all systems receive the same exogenous battery:

* balanced preference decisions;
* matched counterfactual choices;
* cross-context preference applications;
* direct but neutrally worded preference probes;
* choices in which different latent preferences imply different correct decisions.

The battery is independent of the system's own policy and interaction history.

It evaluates:

* decoded profile accuracy;
* calibration;
* behavioral preference accuracy;
* cross-context generalization;
* intrinsic user utility.

## 11.4 Ranking analysis

Report:

* Kendall's $\tau$;
* pairwise ranking-reversal probability;
* bootstrap rank intervals;
* partial orders when systems are statistically tied;
* open-loop optimism by updater.

Do not make a ranking-reversal claim when confidence intervals overlap substantially.
 ---

## 11.5 Evaluation Selection Regret

A more consequential metric is whether static evaluation leads researchers to deploy the wrong system.

Let:

$$
s_{\text{open}}^*=
\arg\min_s
\operatorname{Err}_{\text{open,dev}}(s)
$$

be the updater selected using fixed-history development evaluation.

Let:

$$
s_{\text{closed}}^*=
\arg\min_s
\operatorname{Err}_{\text{closed,dev}}(s).
$$

Define:

$$
\operatorname{ESR}=
\operatorname{Err}_{\text{closed,test}}
\left(
s_{\text{open}}^*
\right) -
\operatorname{Err}_{\text{closed,test}}
\left(
s_{\text{closed}}^*
\right).
$$

**Evaluation Selection Regret** asks:

> How much closed-loop performance is lost because the evaluation protocol selected a system using exogenous fixed histories?

A positive and substantial ESR is stronger than simply reporting low rank correlation.

# 12. Deferred Extension: Human Pragmatic Evidence Validation

Participants or independent annotators see matched interaction contexts and judge:

> How strongly does this interaction support the claim that the user generally prefers $P$?

Conditions include:

* volunteered statement;
* balanced choice;
* restricted choice;
* default acceptance;
* agent-suggestion acceptance.

The study compares:

* human evidence-strength rankings;
* fitted action-aware updates;
* LLM profile updates.

The central analysis is:

> Do humans discount policy-conditioned acceptance more strongly than LLM profile writers?

This is pragmatic validation, not a claim that human judgments are an exact causal oracle.

# 13. Native Memory–Action Validation

The controlled track is scientifically necessary but insufficient for a paper about persistent agents.

The implemented native protocol provides an inspectable
**memory–persona–action loop**:

$$
\text{episodic/semantic memory}
\rightarrow
\text{persona}
\rightarrow
\text{profile-conditioned action}
\rightarrow
\text{user outcome}
\rightarrow
\text{memory refinement}.
$$

This follows the coupling pattern explicitly studied by PersonaAgent, in which memory-derived personas guide personalized actions and action outcomes refine memory. ([ACL Anthology][2])

The repository does not claim a reproduction of that system. Its native
updaters and selected action adapter provide:

* logged episodic and semantic memories;
* an explicit persona state;
* deterministic diagnostic actions and separately collected recorded
  model-mediated actions;
* auditable memory updates;
* a standardized terminal evaluation.

The selected model-mediated adapter,
`cape-loop-openai-native-agent-v1`, sends the complete retained state and
held-out suite to OpenAI `gpt-5.6-sol` and requires one content-bound action per
item. Local deterministic projections are diagnostics and cannot satisfy the
native-action requirement.

The native experiment asks whether causal-provenance miscalibration observed in
Track A also appears in an inspectable persistent state loop. The protocol is
implemented; no eligible paper-scale external-decoder or native-action corpus
is checked in.

# 14. Diagnostic Mitigation, Not a New Method

To make the analysis actionable, test two simple interventions.

## Explicit provenance metadata

Each memory update receives:

```json
{
  "elicitation": {
    "choice_set_balance": "restricted",
    "display_order": ["anchor", "alternative"],
    "default": "anchor",
    "agent_suggestion": "anchor",
    "question_type": "confirmation",
    "policy_profile_snapshot": {
      "prefers_budget": 0.81
    }
  }
}
```

## Provenance-aware instruction

The writer is told:

> Treat a response as evidence conditional on the options and framing that produced it. Do not infer a general preference merely because the user accepted an agent-selected, defaulted, or restricted option.

These are diagnostic mitigations. The paper should not present them as a new state-of-the-art profile-updating method.

A useful result, if supported, is:

> Explicit causal-provenance structure reduces exact-oracle update error
> without suppressing warranted learning from balanced or volunteered
> evidence.

# 15. Stage-Gated Correction Experiment

Run this only after Experiments A–C pass.

After profile-conditioned reinforcement, provide:

> “I do not generally prefer budget hotels. I chose those earlier because of the options and defaults you showed me.”

Deliver the correction:

* before reinforcement;
* after one reinforcing interaction;
* after repeated reinforcing interactions;
* after recurrent profile consolidation.

Then switch all systems to balanced interactions.

Report:

* residual profile error;
* time to recovery;
* recovery area under the error curve;
* textual versus behavioral recovery;
* persistence in derived memories.

Call the result **correction debt**, not preference change.

# 16. User-Response Robustness

The primary model is accompanied by:

1. a mixed-logit or heterogeneous-user response model;
2. a rule-based noisy utility maximizer;
3. broad parameter sweeps over ranking, default, suggestion, and choice noise.

The primary surface uses a frozen outcome-blind bank with one neutral base and
display-name set per scenario. The current provisional bank contains 48
project-standardized visible bases over three source-neutral frames. The
historical OpenRouter record concerns candidate authoring, not provider
authorship of that current text. Authoring is separate from the trials. The
structured decision is sampled first; code then fills the base, adds only a
fixed default/suggestion sentence when applicable, and emits the fixed
`I choose {selected_name}.` reply.

The version 1.5 scenario contract also declares the nuisance coordinate and
direction of every same-direction restricted peer. Across the test bank it
crosses both non-target coordinates with both signs. The calibrated target
half-spans make all 72 scenario-anchor instances numerically distinct, while
the nuisance coordinate/direction marginals remain balanced. This is
prospective mathematical counterbalancing only. All 48 current surfaces remain
provisional until independent human fact-mapping, naturalness, neutrality,
non-dominance, and semantic-strength reviews are completed.

For example:

> **Assistant:** Here are two lower-cost hotel options for your trip. Hotel A
> is a standard room in a mixed-use neighborhood. Hotel B is a standard room
> in a quiet outer neighborhood. Which would you like?
>
> **User:** I choose Hotel A.

The exact local user sentence is reused where the matched design requires an
identical response. Avoid position-dependent replies such as “the first one”
because ranking is itself an experimental treatment.

Generated language is rejected if it adds unsupported claims such as:

> “I generally prefer budget hotels.”

# 17. Simulator-Sensitivity Analysis

Sweep:

* visible profile-conditioning propensity \(\lambda\);
* decision noise;
* default susceptibility;
* ranking susceptibility;
* suggestion acquiescence;
* profile strength;
* prior uncertainty;
* trajectory length.

The direct policy intervention is

$$
P(\text{apply profile treatment at }t)
=
\lambda
\left[
0.15+0.80\,
\operatorname{clip}
\left(
\frac{\operatorname{Conf}_t-0.5}{0.5},
0,1
\right)
\right],
$$

with \(\lambda\in\{0,.33,.67,1\}\), labeled none, weak, moderate,
and full soft-conditioning propensity. This is a multiplier, not an additive
term. At zero, the soft-policy action is visibly neutral and matches the
balanced action under the paired semantic draw. At one, it reproduces the
ordinary adaptive soft policy. Intermediate values share common random numbers,
but assignments are nested only conditional on the same fixed belief; evolving
dose-specific profiles can break nesting across complete trajectories. Applied
ranking, default, and suggestion treatments change the assistant turn the
evaluated LLM sees; assignment rate and actual paired visible-action divergence
are both retained.

`presentation_multiplier` is a separate user-susceptibility robustness axis.
It can change simulated choices and hence the realized conversation, but does
not directly assign a different assistant action. It must not be interpreted
as the policy dose.

The primary sensitivity analysis varies what the evaluated updater can
actually observe: profile-conditioning dose, realized treatment exposure,
and the resulting binary ranking/default/suggestion changes. A positive dose
with zero paired visible-action divergence is a failed manipulation, not a null
behavioral effect. The headline result must hold over a meaningful region,
including settings where users often reject profile-consistent suggestions.

Numeric decision-noise and susceptibility multipliers are response-model
robustness analyses. They are not visible policy interventions unless they
change the rendered interaction or realized response. Hard restriction remains
a stress test and cannot establish the soft-loop headline.
The current version holds recommendation wording fixed. Graded persuasive
wording is deferred until those surfaces receive an independent
strength/naturalness review; this paper version makes no wording-dose claim.
Ranking and default are likewise fixed binary visible treatments. A numeric
`default_multiplier` changes simulated susceptibility, not the visible
strength of a default, and is not labeled as a UI-dose intervention.
Option-set balance is held fixed in the soft-treatment sensitivity analysis;
hard restriction is reported separately as a stress test rather than mislabeled
as a graded visible-dose axis.

The phase diagram should therefore show:

1. whether the visible policy manipulation was active;
2. whether profile-conditioned actions changed information or
   disconfirmation;
3. the same-history exact-shadow attribution gap;
4. continuous profile-error or excess-confidence amplification; and
5. strict self-confirmation only as a secondary overlay.
The null \(\lambda=0\) point is a negative control and is not required to pass
the harmful-region criterion. It is included in grid completion and
phase-boundary inference instead. Loop formation additionally requires a
horizon that revisits an attribute after updating it. Three-turn sensitivity
grids are manipulation/transport smokes. The bounded Experiment B calibration
uses six turns only with prospective admission of two informative active turns,
one decisive active control, two mechanisms, retained counter-profile options,
and sufficient direction-robust ASM in every paired trajectory. A longer
horizon may be frozen after outcome-blind stimulus review or offline audit, but
must not be selected after inspecting evaluated-model outcomes.

# 18. Statistical Plan

## Registered Experiment A mechanism model

$$
\operatorname{CalibrationResidual}
\sim
\operatorname{Mechanism} +
\operatorname{Domain} +
\operatorname{PriorStrength} +
(1+\operatorname{Mechanism}\mid\operatorname{User}) +
(1\mid\operatorname{Scenario}).
$$

This model is fit only to the predeclared target writer's
`response_mode = controlled_anchor` rows in the `same_response_provenance`
track. `CalibrationResidual` is the anchor-direction system log-odds update
minus the exact warranted update. The four primary contrasts compare
restricted, ranking, default, and suggested presentation with balanced.
ExactACUE is secondary unsigned magnitude. Model- and mechanism-specific
log-odds calibration curves are primary descriptive estimands; a parallel
fitted-aware analysis is secondary robustness. Same-seed rerun pooling adds a
run-replicate random intercept. Different seeds are analyzed separately and
cannot increase the independent-user count.

Within each user–domain–target cell, Experiment A reuses one scenario for both
anchor directions and reverses physical anchor position across that pair.
Mechanism, response-mode, prior-strength, and updater comparisons retain the
same scenario/order assignment; the deterministic cycle balances scenario use
within direction across users.

## Registered Experiment B interaction model

The primary conjunctive claim uses the soft-policy exact same-history gap
$G_{U,\mathrm{soft}}$, the prospectively schedule-matched soft-minus-balanced
attribution-gap contrast, and paired exact-shadow SelectionCost defined above.
The incorrect-minus-correct moderation is a prespecified secondary claim.
Soft-minus-exploratory is retained as a supporting whole-policy comparator
because exploratory target/scenario selection remains adaptive. The total
soft-minus-balanced updater terminal
error is retained alongside those components and must equal SelectionCost plus
$\Delta G_{\text{soft-bal}}$ within numerical tolerance. Complete latent users
are the primary inference unit; turns, domains, and trajectory replicates are
repeated observations and never increase the independent sample count.

Experiment B inference v5 (`experiment-b-clustered-randomization-v5`) reduces
each paired estimand to equally weighted complete-user means and uses one-sided
paired sign-flip tests for the registered directional decisions. For $n\leq16$
complete users it enumerates all $2^n$ sign patterns exactly. For larger
samples it uses 16,384 deterministically sampled sign patterns, includes the
observed sign assignment, and applies the plus-one correction. Decisions use
$\alpha=0.05$, require at least eight complete users, and rely on sign
exchangeability of the paired complete-user contrasts around the tested null.
Two-sided percentile bootstrap intervals over complete users are sensitivity
summaries; complete paired-trajectory bootstrap intervals are an additional
sensitivity and never convert repeated trajectories into independent users.

Within each model run, the frozen multiplicity policy is
`experiment-b-within-model-gatekeeping-v1`. Gate 3 is one primary
intersection-union test (IUT): its composite p-value is the maximum of its
three component p-values, so all components must reject at one-sided
$\alpha=0.05$ and no within-conjunction alpha division is needed. Only after
that primary IUT rejects does a fixed Holm family open over three claims: Gate
2's four-component IUT, the incorrect-minus-correct moderation, and nested net
profile harm. An unavailable member remains in that family with p-value one;
the family never shrinks post hoc. All other mechanism, calibration, strict
self-confirmation, and exploratory-comparator endpoints are descriptive or
supporting and cannot authorize standalone discoveries. The hierarchy is
applied separately to each model; it does not pool models or support an
“any-model” claim. Bounded calibration runs remain descriptive even when a
computational decision is positive.

The bounded OpenRouter calibration freezes three full-design primary writers:
Gemini 3.6 Flash, GPT-5.6 Luna, and Mistral Large 3
(`mistralai/mistral-large-2512`). Results remain
model-specific; model identities are not clusters. DeepSeek V4 Flash is a
post-pilot targeted secondary replication restricted to the incorrect-seed
balanced-versus-soft contrast. It is reported separately and never pooled into
the primary trio. Registered decisions are per model; this suite makes no
“any primary model succeeds” or omnibus cross-model claim, so it does not treat
the three model outputs as one multiplicity family. Any later family-level
claim requires a separately frozen estimand and adjustment rule.

The following mixed-effects model is a supporting preliminary model of the
underlying error trajectories:

$$
\begin{aligned}
\operatorname{TerminalError}
\sim&
\operatorname{Updater}
*
\operatorname{Policy}
*
\operatorname{InitialProfile}\\
&+
\operatorname{Domain} +
\operatorname{Turn} +
(1+\operatorname{Policy}\mid\operatorname{User}) +
(1\mid\operatorname{Scenario}) +
(1\mid\operatorname{CRNSet}).
\end{aligned}
$$

The key interaction is specifically evaluated for incorrect initial profiles.
For Experiment B, `Scenario` is the stimulus actually displayed on the
retained turn. `CRNSet` is the common-random-number twin set shared across
counterfactual policy/updater branches. They are distinct random effects:
branches can share a CRN set while endogenous target divergence makes them
display different scenarios.

Experiment A uses the versioned exact-oracle analysis contract; legacy
fitted-reference artifacts remain interpretable only under their original
schema. Experiment B retains this version-1 terminal-error model as supporting
analysis, while the exact same-history policy gaps and continuous decomposition
outcomes are the primary paper-level estimands. A target-versus-fitted-aware
branch contrast must not be relabeled as a same-history shadow contrast.
Experiment C remains secondary.

## Common random numbers

Represent choice as:

$$
Y_t=
\arg\max_x
\left[
\theta^\top\phi(x) +
\delta(C_t,x) +
\varepsilon_{t,x}
\right].
$$

Reuse the same $\varepsilon_{t,x}$ draws across counterfactual policy branches whenever options overlap.

This reduces variance and makes user twins meaningfully paired.

## Inference unit

The primary independent unit is the complete latent user, not the turn or
trajectory replicate. Independent users span both domains and the
prospectively declared weak/strong preference and balanced-choice-margin
strata. Random seeds and trajectory replicates are repeated robustness
observations nested within user; they are not independent users. Complete
paired trajectories form only the declared sensitivity resampling unit.

Report:

* one-sided paired complete-user sign-flip decisions for registered directional
  Experiment B claims;
* complete-user percentile-bootstrap sensitivity intervals and separately
  labeled paired-trajectory sensitivity intervals;
* user-level mixed effects;
* effect sizes;
* raw and calibrated scores;
* simulation-based power analysis;
* multiplicity correction for secondary comparisons;
* held-out profile and template generalization.

Sample size is fixed only after pilot-based power analysis of the:

$$
\text{Updater}
\times
\text{Policy}
\times
\text{InitialProfile}
$$

interaction.

# 19. Data Splits

Splits must be disjoint over:

* complete latent preference profiles;
* susceptibility types;
* option templates;
* dialogue templates;
* scenario families;
* natural-language paraphrase templates.

Fitted response-model parameters are learned from training users only.
Calibration transformations and model-selection diagnostics use development
users only. Neither stage uses test labels.

The common terminal battery uses held-out items and does not reuse training provenance templates.

# 20. Implementation Schema

## User state

```json
{
  "user_id": "user_x",
  "theta": {
    "attribute_1": "+2",
    "attribute_2": "-1",
    "attribute_3": "+1"
  },
  "susceptibility": {
    "ranking": "medium",
    "default": "low",
    "suggestion": "high"
  }
}
```

## Interaction context

```json
{
  "turn_id": "turn_x",
  "options": [
    {
      "option_id": "anchor",
      "features": {}
    },
    {
      "option_id": "alternative",
      "features": {}
    }
  ],
  "ranking": ["anchor", "alternative"],
  "default": "anchor",
  "suggested_option": null,
  "wording_template": "default_confirmation",
  "question_type": "choice"
}
```

## Policy provenance

```json
{
  "policy_id": "soft_profile_conditioned",
  "policy_version": "v1",
  "profile_snapshot": {},
  "random_seed": 0
}
```

## User observation

```json
{
  "selected_option": "anchor",
  "surface_response": "I choose Hotel A.",
  "choice_noise_key": "experiment-a:travel:user-000:turn-0",
  "assistant_message": "Here are two lodging options. Hotel A is [...]. Hotel B is [...]. Which would you like?",
  "surface_id": "scenario:default:anchor>alternative:anchor"
}
```

## Profile update

```json
{
  "belief_before": {},
  "belief_after": {},
  "native_memory_before": [],
  "native_memory_after": [],
  "written_delta": [],
  "updater_id": "full_context_llm"
}
```

Every output should retain the full causal chain from policy state to action context, response, and memory update.

# 21. Required Automated Tests

Before the first large run:

1. Exact posterior sums to one and matches brute-force enumeration.
2. Exact posterior updates reproduce brute-force inference under every
   declared action mechanism.
3. Fitted action-aware inference beats fitted action-unaware inference on
   held-out simulated interactions as a learnability check.
4. Anchor option identity and attributes remain unchanged across matched
   conditions.
5. The local user response is literally identical across balanced,
   restricted, ranking, default, and suggested controlled-anchor rows.
6. Every matched response exceeds the minimum probability threshold.
7. Latent preference remains fixed throughout the trajectory.
8. Presentation effects never enter intrinsic welfare calculations.
9. Action context and internal policy provenance are stored separately.
10. Common random-number pairing is reproducible.
11. Static logging histories are identical across evaluated updaters.
12. Closed-loop actions depend only on the updater's current profile and declared policy.
13. Terminal diagnostics are independent of the evaluated system's policy.
14. Train, development, and test users are disjoint.
15. Probability calibration never uses test labels.
16. Native profile decoders are blinded to system identity and latent truth.
17. Language verbalization cannot introduce unsupported general-preference claims.
18. Every reported self-confirming case satisfies all five definitional conditions.
19. Ranking results are reproduced over complete-user bootstrap samples;
    random seeds remain nested robustness replicates.

# 22. Hypotheses

### H1 — Exact-oracle causal-provenance calibration

In the same-response provenance track, the relationship between the system
update \(\widehat u\) and warranted exact action-aware update \(u^*\) is
estimated separately by updater and mechanism. The primary report gives
\(\alpha_{k,m}\), \(\beta_{k,m}\), residual RMSE, and uncertainty relative to
the ideal \((0,1,0)\). The hypothesis does not prescribe universal
over-weighting or under-weighting.

### H2 — Provenance-specific residual heterogeneity

Holding the selected option and local user response fixed, at least one
policy-conditioned mechanism differs from balanced in the target writer's
signed calibration residual. Calibration curves and secondary unsigned
ExactACUE are estimated by model and mechanism; a pooled sign is not required.

### H3 — Policy-dependent evidence selection

Under natural-response closed-loop interaction, an active soft
profile-conditioned policy changes exact-shadow information,
disconfirmation, or terminal error relative to its paired balanced reference;
the exploratory reference is a supporting whole-policy comparison.

### H4 — Same-history evidential attribution

The primary soft-minus-balanced contrast in $G_{U,\pi}$ is positive
across prospectively schedule-matched natural histories. Within each
$G_{U,\pi}$, the evaluated updater and exact shadow consume the identical
history; histories are not identical across policies. The adaptive exploratory
comparison is supporting rather than turn-matched. This effect is distinct from
evidence selection and may vary in sign by updater and policy; the
incorrect-minus-correct moderation is reported separately.

### H5 — Relative confidence penalty and conditional feedback amplification

For initially wrong profiles, soft profile conditioning produces a positive
paired soft-minus-balanced CEC contrast: the registered **relative confidence
penalty**. This contrast alone does not establish absolute amplification or
reinforcement. Gate 2 additionally requires an active visible treatment,
paired natural-choice divergence, and later action influence. Policy-specific
soft CEC, EAR, partial and paired behavioral reinforcement, and
information/disconfirmation deficits remain separately labeled supporting
outcomes rather than an unadjusted disjunction. Recovery after correction is
reported only if a separately frozen correction protocol is activated; it is
absent, not zero, in the current minimum design. The strict five-clause
self-confirmation rate and DIR are secondary endpoints. The historical SCI
field is only a compatibility alias for the soft-minus-balanced attribution-gap
contrast.

### H6 — Logging-policy-dependent system selection

Fixed-history logging can change system rankings, inferential top tiers, or
held-out closed-loop selection regret. This does not require absolute
closed-loop error to exceed every fixed-history error.

### H7 — Causal provenance is actionable

Explicit provenance metadata or provenance-aware instructions reduce exact
update error or closed-loop attribution/amplification without eliminating
valid learning from balanced choices or volunteered preferences.

### Deferred H8 — Human judgments are more provenance-sensitive

Human evidence-strength judgments distinguish freely elicited and policy-conditioned signals more strongly than ordinary LLM profile writers.

### Secondary H9 — Correction debt

Endogenous reinforcement increases recovery time after an identical explicit correction.

# 23. Planned Main Figures

## Figure 1: Same response, different evidence

For the identical anchor selection, show posterior shifts under:

* balanced choice;
* restricted choice;
* changed ranking;
* default;
* suggestion;

Plot the exact-oracle warranted update and each evaluated updater. Show fitted
aware/unaware references in a secondary panel.

## Figure 2: Selection–attribution causal matrix

Show terminal error for:

$$
\text{balanced/profile-conditioned policy}
\times
\text{aware/LLM updater}.
$$

Include the explicitly named
$\Delta G_{\text{soft-bal}}$ and
$\Delta G_{\text{soft-exp}}$ contrasts. Keep the historical SCI field only as
a compatibility alias, and report DIR with its opportunity count separately.

## Figure 3: Continuous closed-loop trajectories

Plot wrong-profile mass, exact-shadow mass, policy-specific cumulative excess
confidence, the paired soft-minus-balanced relative confidence penalty, and
correction/recovery where applicable for:

* exact aware;
* fitted aware;
* fitted unaware;
* full-context LLM;
* provenance-aware LLM;
* recurrent profile.

## Figure 4: Open-loop versus closed-loop model selection

Show:

* system ranks under both fixed logging policies;
* system ranks closed loop;
* pairwise reversal probabilities;
* Evaluation Selection Regret.

## Figure 5: Sensitivity phase diagram

Show visible intervention dose/exposure/divergence and continuous outcomes.
Use \(\lambda=0\) as a negative control, mark positive-dose zero-divergence
cells as failed manipulations, and show strict self-confirmation only as a
secondary overlay.

# 24. Planned Results Tables

## Table 1: Exact-oracle same-response calibration

| Updater | Balanced | Restricted | Ranking | Default | Suggested | Overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Response-only LLM | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| Full-context LLM | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| Provenance-aware LLM | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| Other updater families | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |

Each cell reports \(n\), \(\alpha\), \(\beta\), residual RMSE, signed residual,
secondary absolute exact update error, and complete-user uncertainty. Exact
and fitted references are not averaged together.

## Table 2: Closed-loop decomposition

| Policy | Updater | Selection cost | Attribution cost | EAR | Absolute CEC | Information/disconfirmation deficit | Strict rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Balanced | Exact shadow | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| Profile-conditioned | Exact shadow | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| Balanced | Full-context LLM | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| Profile-conditioned | Full-context LLM | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| Profile-conditioned | Provenance-aware LLM | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |

Paired companion rows report $\Delta G_{\text{soft-bal}}$, SelectionCost, the
total soft-minus-balanced updater error and its decomposition residual, and
$\Delta\operatorname{CEC}_{\text{soft-bal}}$ under the label **relative
confidence penalty**. Absolute policy-specific CEC and the relative penalty are
not interchangeable.

## Table 3: Evaluation validity

| Updater      | Balanced-log rank | Biased-log rank | Closed-loop rank | Open-loop optimism | Pairwise reversal |
| ------------ | ----------------: | --------------: | ---------------: | -----------------: | ----------------: |
| `[System A]` |           `[TBD]` |         `[TBD]` |          `[TBD]` |            `[TBD]` |           `[TBD]` |
| `[System B]` |           `[TBD]` |         `[TBD]` |          `[TBD]` |            `[TBD]` |           `[TBD]` |
| `[System C]` |           `[TBD]` |         `[TBD]` |          `[TBD]` |            `[TBD]` |           `[TBD]` |

# 25. Original Eight-Week Execution Plan

This table records the proposed sequencing. It is not the current repository
status: the software components are implemented, while the paper evidence
collection and responsible-researcher review remain outstanding.

| Week | Deliverable                                                                                                                          |
| ---- | ------------------------------------------------------------------------------------------------------------------------------------ |
| 1    | Freeze latent profiles, susceptibility types, action-context schema, intrinsic utility, exact posterior, and anchor-option generator |
| 2    | Implement fitted aware/unaware baselines, structured-belief interface, calibration, and simulation-based power analysis              |
| 3    | Complete Experiment A pilot across both domains and all five matched provenance conditions                                           |
| 4    | Freeze go/no-go decision; build closed-loop harness, shadow posterior, common-random-number branches, and native memory states       |
| 5    | Run crossed closed-loop Experiment B and parameter-sensitivity sweeps                                                                |
| 6    | Run two static logging conditions, closed-loop ranking evaluation, common terminal battery, and native memory–action validation      |
| 7    | Optional post-ethics human validation and source-attribution extension; run correction debt only if its stage gates pass              |
| 8    | Statistical analysis, figures, writing, reproducibility audit, and release packaging                                                 |

# 26. Go/No-Go Gates

## Gate 1: Identifiable causal-provenance calibration

Gate 1 is an outcome-neutral identifiability and execution-readiness check. It
must not require an LLM to be wrong. Proceed beyond Experiment A only when:

* the `exact_action_aware` updater reproduces the exact posterior from the same
  declared response model used to generate users, within the numerical
  tolerance;
* the controlled-anchor invariant audit confirms the identical local response
  across balanced, restricted, ranking, default, and suggested conditions;
* every declared domain-by-mechanism cell is present;
* relative to balanced presentation, the exact oracle warrants a nontrivially
  different paired update for at least two non-balanced mechanisms in both
  travel and writing; and
* held-out controlled paraphrases have complete required case/updater coverage
  and preserve the selected option and visible-context binding across surface
  forms.

Fitted-aware versus fitted-unaware performance is a secondary learnability
diagnostic outside the controlling gate. Its failure limits claims about
learnable approximation, but does not invalidate the exact controlled oracle.
Likewise, held-out fitted-aware Brier differences are descriptive diagnostics,
not a Gate 1 condition.

When LLMs closely track the exact oracle within declared practical and
uncertainty bounds, omit the miscalibration claim. A nonzero ACUE alone does
not establish over-update, and no universal sign is required.

## Gate 2: Conditional behavioral feedback amplification

This gate controls only the stronger downstream behavioral-feedback claim; it
does not control the primary attribution-gap analysis. Pass it only when:

* ranking, default, or suggestion visibly changes the action stream while
  counter-profile options remain available;
* the changed action stream changes at least some natural user responses and
  the resulting memory changes a later action;
* the preregistered user-clustered soft-minus-balanced cumulative
  excess-confidence contrast is adequately estimable; and
* the one-sided complete-user sign-flip decision supports a positive contrast
  at $\alpha=0.05$.

The visible-action, natural-choice, later-action, and CEC directional criteria
use the same inference-v5 complete-user decision rule and together form the
Gate 2 IUT. Gate 2 can support a paper claim only after Gate 3's primary IUT
rejects and the Gate 2 composite survives the fixed three-claim secondary Holm
family. The CEC contrast is
labeled the **relative confidence penalty**: it establishes relative
degradation under soft versus balanced interaction, not reinforcement by
itself. Absolute soft-policy CEC, soft-policy EAR, partial and paired behavioral
reinforcement, exact information/disconfirmation deficits, and related
continuous signals are supporting outcomes and do not form an uncorrected “any
endpoint” gate. Percentile bootstrap intervals are sensitivity summaries
rather than the controlling decision.

The strict five-clause self-confirmation predicate is a secondary endpoint and
does not control this gate. An effect found only under fully restricted options
is insufficient.

## Gate 3: Policy-conditioned evidential legibility

For the incorrect-seed soft-versus-balanced comparison, all three conditions
must have at least eight complete users and pass their one-sided paired
complete-user sign-flip decisions:

* $G_{U,\mathrm{soft}}>0$;
* $\Delta G_{\mathrm{soft-bal}}>0$; and
* $\operatorname{SelectionCost}<\epsilon_{\mathrm{sel}}$, where the frozen
  marginal-Brier noninferiority margin is
  $\epsilon_{\mathrm{sel}}=0.02$.

The first comparison is within an identical soft-policy history. The second
and third compare paired policy-specific histories; they do not claim that the
soft and balanced users produced identical responses. Passing the third
criterion establishes practical exact-shadow terminal-error noninferiority, not
zero loss or equality of information. The paired percentile-bootstrap interval
is reported as sensitivity evidence.

These three tests are one intersection-union primary claim, not three chances
to discover an effect. Its valid composite p-value is their maximum; Gate 3
passes only when that value is at most `0.05` and every component is adequate.
No Bonferroni division is applied inside this conjunction. Its rejection opens
the prespecified secondary Holm family; it does not make a secondary claim
significant by itself.

If this gate is not met, retain any supported policy-induced evidence-quality
or updater-attribution result, but omit the conjunctive policy-conditioned
legibility claim.

### Nested Gate 3 report: Net profile harm

The stronger net-harm conclusion is evaluated separately and cannot erase or
replace the narrower legibility result. It first requires Gate 3 to pass, then
requires one-sided complete-user evidence that

$$
\Delta\operatorname{Err}_{U,\text{soft-bal}}
=
\operatorname{SelectionCost}
+
\Delta G_{\text{soft-bal}}
>
\delta_{\mathrm{harm}},
\qquad
\delta_{\mathrm{harm}}=0.02
$$

on the marginal-Brier scale. The corresponding implementation endpoint is
`soft_minus_balanced_terminal_error`, and its arithmetic decomposition must
pass independently of the inferential decision. Failure of this nested report
means that net harm beyond the practical margin was not established; it does
not negate a supported attribution or legibility result.
The net-harm p-value is one of the three fixed post-Gate-3 Holm members, so the
nested report also requires its adjusted decision to reject.

## Gate 4: Native-system validity

At least one inspectable persistent memory–action loop must exhibit the same
causal-provenance miscalibration under the matched, equal-strength contrast. Every
eligible state must also have:

* blind judgments from at least two responsibly reviewed, genuinely distinct
  decoder sources from distinct model families; and
* hash-bound terminal actions emitted by the native system directly from the
  retained native state.

The selected decoder pair shares OpenRouter. It may satisfy the
responsible-researcher-reviewed distinct-source protocol, but it does not
establish distinct transport origins, first-party decoder origin, or
statistically independent errors. The local deterministic decoders and
structured/persona action projections are ineligible substitutes.

Otherwise, restrict the claim to controlled profile updating.

## Gate 5: Evaluation implication

Conceptually, the evaluation implication concerns rank disagreement,
credible reversal, or meaningful selection regret. The executable gate uses
the stronger preregisterable rules below:

* a joint-paired complete-user analysis resolves a system pair in opposite
  directions in fixed-balanced and endogenous closed-loop evaluation and
  resolves the corresponding regime shift; or
* development inferential top tiers select different candidate sets and the
  minimum held-out closed-loop Evaluation Selection Regret, together with its
  conservative paired interval envelope, exceeds the declared practical
  threshold.

Kendall rank agreement, marginal rank intervals, and raw pairwise reversal
probabilities remain descriptive and cannot pass Gate 5 by themselves.

Gate 5 is the principal test of evaluation validity. When rankings remain
stable, remove the system-selection claim while retaining any separately
supported evidence-quality or provenance-calibration results.

## Gate 6: Robustness

The main effect must survive:

* another response model;
* broad simulator parameters;
* both domains;
* multiple LLM families;
* natural-language paraphrases;
* the exact action-aware reference.

The fitted action-aware analysis is always reported as secondary robustness.
Disagreement with the exact analysis narrows claims about learnable
approximation or response-model transport; it does not silently replace or
invalidate the within-generator exact estimand.

Visible-intervention sensitivity must also include the \(\lambda=0\) negative
control and exclude positive-dose cells with zero paired visible-action
divergence from causal policy-effect interpretation. Numeric response-model
multipliers and hard restriction are robustness/stress analyses, not
substitutes for an active soft manipulation.

# 27. Expected Contributions

## 1. Updater–logging-policy pairs as the evaluation object

The paper evaluates an updater jointly with the interaction/logging policy that
generated its evidence, rather than assigning it one policy-independent score.

## 2. A controlled audit of causal-provenance calibration

CAPE-Loop holds the selected item and literal local response fixed across five
matched contexts, then compares model updates with the exact action-aware
posterior under the declared generator. Fitted references provide secondary
learnability and misspecification checks.

## 3. Policy-conditioned evidential legibility and decomposition

The crossed design and shadow posterior separate information loss from
same-history evidential attribution. Policy-specific attribution gaps reveal
histories that remain usable by exact inference but are misread by an evaluated
writer. Disconfirmation inversion and strict behavioral self-confirmation are
reported separately, with the latter only when every registered clause holds.

## 4. A validity test for static personalization evaluation

The paper determines whether fixed histories predict closed-loop profile accuracy, user utility, and deployment-time system selection.

The provenance-aware prompt and metadata conditions are diagnostic mitigations, not a fifth method contribution.

# 28. Relationship to the PhD Agenda

Counterfactual Memory Commitment currently treats a memory operation as an intervention evaluated against later interactions in a user trajectory.

CAPE-Loop establishes that, in a deployed personalized agent:

$$
Q_{>t}=
Q_{>t}^{\pi(M_t)}.
$$

A memory operation may change:

* the questions asked;
* the alternatives shown;
* the default or recommendation;
* the user responses observed;
* the evidence later written into memory.

The division is therefore clean:

| Project                  | Scientific question                                                                                            |
| ------------------------ | -------------------------------------------------------------------------------------------------------------- |
| CAPE-Loop                | Is future user evidence endogenous to memory-conditioned interaction, and can profiles become self-confirming? |
| CMC                      | Which memory operation has positive expected future value?                                                     |
| Memory lifecycle control | How should existing persistent state be updated, suppressed, or forgotten over time?                           |

CAPE-Loop is the empirical diagnosis that motivates eventually making CMC policy-conditional and branch-consistent. It does not duplicate CMC's controller.

# 29. Strong-Accept Narrative

The strongest final paper would establish the following connected results,
while a valid paper need not establish every downstream item:

1. **Identical user responses warrant different action-aware updates under
   some elicitation contexts in the declared model.**
2. **LLM profile writers show model- and mechanism-specific exact-oracle
   provenance miscalibration.**
3. **Natural-response closed loops separate changes in the evidence stream
   from same-history attribution error.**
4. **Continuous amplification and information-deficit outcomes reveal feedback
   effects even when the strict self-confirmation predicate is null; recovery
   is added only if a correction protocol is preregistered.**
5. **The failure appears in an inspectable persistent memory–action loop.**
6. **Optionally, post-ethics human evidence shows greater pragmatic provenance
   sensitivity than ordinary LLM profile writers.**
7. **Conditionally, explicit provenance metadata reduces the loop without
   preventing valid learning.**
8. **Conditionally, Experiment C v1 shows that fixed-history evaluation changes
   system selection or incurs held-out closed-loop regret.**

The stable final conclusion would be:

> **A user-profile updater cannot be evaluated independently of the policy that
> generated its evidence, and current LLM memory systems may not consistently
> calibrate profile updates to that evidence's causal provenance.**

A stronger self-reinforcement conclusion is conditional on the five-clause
trajectory result and is not required for the primary contribution.

The implementation now covers the anchor audit, fitted references, closed-loop
decomposition, structured/native tracks, evaluation-validity analysis,
provider-neutral and live-provider exchanges, immutable reviews, and the
optional confirmatory mixed-effects harness. That software completeness is not
evidence for the narrative above. The next scientific step is a separately
authorized, preregistered collection and analysis wave; until then every result
placeholder and gate claim remains unresolved.

[1]: https://arxiv.org/abs/2602.07442 "Echoes in the Loop: Diagnosing Risks in LLM-Powered Recommender Systems under Feedback Loops"
[2]: https://aclanthology.org/2026.findings-acl.1315/ "PersonaAgent: Bridging Memory and Action for Personalized LLM Agents ..."
