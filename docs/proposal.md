# Final Paper Proposal

## **You Chose What I Showed You: Causal Provenance and Self-Confirming User Profiles in LLM Agents**

### **CAPE-Loop: Causal Attribution of Preference Evidence in Closed-Loop Agents**

**Paper type:** Analysis and evaluation paper
**Target:** ACL main conference
**Primary contribution:** Causal diagnosis of persistent-profile updating
**Not a contribution:** A new recommendation algorithm, memory architecture, user simulator, or personalization controller

## One-sentence pitch

A user's acceptance of an agent-selected option is not independent evidence of preference; CAPE-Loop tests whether persistent LLM profile writers nevertheless treat restricted choices, defaults, rankings, and agent suggestions as if the user had freely revealed an intrinsic preference—and whether this error makes initially false profiles reinforce themselves.

# Abstract

Personalized LLM agents increasingly update persistent user profiles from interaction outcomes such as choices, revisions, acceptances, and implicit feedback. These observations are not exogenous: the agent's current profile influences which alternatives are displayed, how they are ranked, what default is selected, and which option the agent recommends. A profile may therefore shape the evidence subsequently used to update that same profile.

We study **causal-provenance blindness**, in which a persistent profile writer assigns more evidential weight to a policy-conditioned user response than its elicitation context justifies. For example, selecting an inexpensive hotel from a balanced premium-versus-budget choice provides stronger evidence of a general price preference than accepting the same hotel when it was preselected, explicitly recommended, or presented alongside only other inexpensive options. Nevertheless, both interactions may be stored as “the user prefers inexpensive hotels.”

We introduce **CAPE-Loop**, a controlled evaluation of action-conditioned preference inference and closed-loop profile formation. CAPE-Loop maintains a fixed latent user, explicitly logs the option set, ranking, default, recommendation, wording, and policy provenance that generated each response, and provides three inference references: a Bayes-optimal posterior under the declared response model, a fitted action-aware updater that learns the response model from training interactions, and a capacity-matched fitted action-unaware updater. Controlled anchor-option sets hold the selected item and response constant while varying only their causal provenance, while naturally sampled interactions evaluate average proper-score performance under the response distribution.

We then cross initial profile correctness, interaction policy, and memory updater in closed loop. This separates **evidence-selection error**, where profile-conditioned actions collect less diagnostic evidence, from **evidential-attribution error**, where profile writers overinterpret the evidence that was collected. Every trajectory is accompanied by a shadow action-aware posterior updated on the same observations, enabling turn-level measurement of confidence gained beyond what the interaction warrants. Finally, we compare systems under fixed balanced histories, fixed mildly biased histories, and endogenous closed-loop histories using an identical exogenous terminal diagnostic battery.

Across `[UPDATERS]`, `[MODEL FAMILIES]`, `[DOMAINS]`, and `[TRAJECTORIES]`, we find `[TBD]`. Full-context profile writers exhibit `[TBD]` excess action-conditioned inference error relative to fitted action-aware inference; false initial profiles gain `[TBD]` unwarranted confidence under soft profile conditioning; and fixed-history evaluation `[TBD OPEN/CLOSED-LOOP FINDING]`. These results show that reliable persistent personalization requires recording not only what a user selected or accepted, but also how the agent's own behavior caused that evidence to be observed.

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

> **Personalized agents partly determine the evidence from which they subsequently infer the user, and current persistent-profile writers may fail to condition their updates on how that evidence was elicited.**

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

**Do persistent profile writers assign different evidential weight to the same response when it follows balanced exposure, restricted exposure, a default, or an agent-authored suggestion?**

The primary comparison is against a **fitted action-aware updater**, not merely the simulator oracle.

## RQ2: Evidence selection versus evidence attribution

**How much closed-loop profile error arises because the agent collects uninformative evidence, and how much arises because the memory updater overinterprets the evidence that was collected?**

The crossed policy–updater design and shadow posterior provide the decomposition.

## RQ3: False self-confirmation

**Can an initially wrong profile gain confidence from observations generated by actions selected using that same wrong profile, beyond the confidence justified by an action-aware updater?**

A false profile must gain excess confidence and influence later actions; mere persistence is insufficient.

## RQ4: Evaluation validity

**Do fixed-history evaluations rank profile writers differently from closed-loop evaluation, and can an open-loop benchmark select an updater that is materially worse after deployment?**

The strongest result is not merely a score gap. It is a credible model-selection failure.

## RQ5: Pragmatic validity

**Do humans distinguish evidence produced through free choice from acceptance produced through restricted options, defaults, or agent suggestions more strongly than LLM profile writers do?**

Human judgments validate the pragmatic evidential ordering; they are not treated as access to metaphysical “true preferences.”

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
Y_t
=
\arg\max_{x\in\mathcal X_t}
\left[
\beta\theta^\top\phi(x)
+
\delta_\psi(C_t,x)
+
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
\beta\theta^\top\phi(x)
+
\delta_\psi(C_t,x)
\right).
$$

Presentation effects alter choice probability, but not underlying welfare. Intrinsic utility is:

$$
u_\theta(x)=\theta^\top\phi(x).
$$

User regret is evaluated against the complete feasible option pool:

$$
\operatorname{Regret}_t
=
\max_{x\in\mathcal X_t^{\text{full}}}
u_\theta(x)
-
u_\theta(Y_t).
$$

This prevents a strong default from being mistaken for high user utility.

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

It is a diagnostic upper reference, not a universal normative theory of human choice.

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

The central one-step finding should be:

> Full-context LLM profile writers remain closer to fitted action-unaware inference than fitted action-aware inference, despite receiving the complete elicitation context.

That is substantially stronger than “the LLM differs from an omniscient simulator oracle.”

# 5. Two Evaluation Tracks

## Track A: Controlled structured-belief updating

Every updater receives:

* the same prior distribution;
* the complete action context;
* the user response;
* previous interaction history where applicable.

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

LLM probabilities are calibrated only on development users using a declared transformation such as temperature scaling or isotonic regression. Both raw and calibrated results are reported.

## Track B: Native persistent memory

Each system writes its ordinary memory representation:

* free-text profile;
* episodic memory;
* recurrent summary;
* structured profile;
* provenance-linked memory.

Native states are evaluated through:

1. two independent blinded profile decoders;
2. the common terminal behavioral battery;
3. native end-to-end actions.

No main conclusion may depend on only one decoder.

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

The core paper uses three independently controlled mechanisms.

| Mechanism              | Manipulation                                                 | Inferential issue                              |
| ---------------------- | ------------------------------------------------------------ | ---------------------------------------------- |
| Choice-set composition | Balanced alternatives versus profile-consistent alternatives | Unavailable alternatives cannot be selected    |
| Ranking/default        | Both sides remain available, but one is first or preselected | Acceptance partly reflects position or inertia |
| Agent suggestion       | Agent recommends one option before the response              | Acceptance is partly agent-authored            |

Positive controls include:

* explicit volunteered preference;
* repeated balanced cross-context choices;
* direct correction.

Negative controls include:

* indifferent responses;
* random choices;
* responses that do not distinguish the target preference.

Hard option filtering is included only as a stress test. The main self-confirmation result must appear under **soft conditioning**, where counter-profile alternatives remain available.


# 8. Matched Provenance Construction

## 8.1 Anchor-option design

Each matched set contains an identical anchor option selected in every condition.

| Condition  | Displayed options                           | Default | Suggestion | Selected item |
| ---------- | ------------------------------------------- | ------- | ---------- | ------------- |
| Balanced   | Anchor budget hotel + matched premium hotel | None    | None       | Anchor        |
| Restricted | Anchor budget hotel + second budget hotel   | None    | None       | Anchor        |
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

This is a controlled functional diagnostic, not an average causal effect.

### Naturally sampled evaluation

Sample responses from:

$$
P_\theta(Y\mid C)
$$

independently in each action context.

Purpose:

* measure average proper-score performance;
* verify that the finding persists under realistic response frequencies;
* prevent conclusions from depending on selected identical-response cases.

# 9. Experiment A: Does the Profile Writer Understand Causal Provenance?

## Objective

Test whether seeing the full interaction context changes the profile update by the amount justified by action-aware inference.

## Systems

| Updater                           | Purpose                                                 |
| --------------------------------- | ------------------------------------------------------- |
| Prior/no update                   | Conservative lower reference                            |
| Exact action-aware oracle         | Reference under declared simulator                      |
| Fitted action-aware updater       | Learnable causal reference                              |
| Fitted action-unaware updater     | Exact model of provenance blindness                     |
| Rule-based provenance discounting | Simple non-LLM baseline                                 |
| Response-only LLM                 | Tests surface-response inference                        |
| Full-context LLM                  | Tests whether ordinary dialogue context is sufficient   |
| Provenance-aware LLM              | Diagnostic mitigation with explicit structured metadata |

## Primary metrics

### Action-Conditioned Update Error

Let $q_t$ be the system belief and $p_t^A$ the fitted action-aware belief:

$$
\operatorname{ACUE}_t
=
\left|
(q_{t+1}-q_t)
-
(p_{t+1}^A-p_t^A)
\right|_1.
$$

This measures whether the updater applies the correct evidence increment, rather than merely ending at a similar posterior.

### Posterior divergence

$$
D_{\mathrm{KL}}
\left(
p_{t+1}^A
;|;
q_{t+1}
\right).
$$

Exact-oracle divergence is reported as a diagnostic.

### Excess proper-score error

$$
\operatorname{ExcessBrier}
=
BS(q,\theta)-BS(p^A,\theta).
$$

### Update-direction accuracy

Did each attribute probability move in the correct direction?

### Oracle-update slope

Regress system log-odds updates on action-aware updates:

$$
\widehat{\Delta\ell}
=
\alpha
+
\beta\Delta\ell^A
+
\varepsilon.
$$

A provenance-sensitive updater should have:

* $\beta$ near one;
* low mechanism-specific residuals;
* different update magnitude across matched contexts.

### Evidence-strength ranking

Does the updater distinguish, in the direction supported by the fitted response model:

$$
\text{volunteered preference}
>
\text{balanced choice}
>
\text{suggested/defaulted acceptance}
>
\text{restricted choice}?
$$

The exact middle ordering is derived from the fitted model and human judgments rather than assumed universally.

# 10. Experiment B: Can False Profiles Become Self-Confirming?

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

### Interaction policies

* balanced/randomized;
* softly profile-conditioned;
* exploratory.

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

## 10.3 Shadow action-aware posterior

For every trajectory generated by updater $U$, maintain a shadow action-aware belief:

$$
p_{t+1}^{\text{shadow}}
=
\operatorname{UpdateAware}
\left(
p_t^{\text{shadow}},C_t,Y_t
\right).
$$

The shadow updater observes exactly the same actions and responses but does not control the policy.

This yields a turn-level decomposition.

### Evidence-selection cost

$$
\operatorname{SelectionCost}
=
\operatorname{Err}
\left(
p_T^{\text{shadow, profile-policy}},
\theta
\right)
-
\operatorname{Err}
\left(
p_T^{\text{shadow, balanced}},
\theta
\right).
$$

This is the loss caused by collecting less informative evidence.

### Evidential-attribution cost

$$
\operatorname{AttributionCost}_{U,\pi}
=
\operatorname{Err}
\left(
q_T^{U,\pi},
\theta
\right)
-
\operatorname{Err}
\left(
p_T^{\text{shadow},\pi},
\theta
\right).
$$

This compares the system and action-aware updater on the same history.

### Self-Confirmation Interaction

For incorrect initial profiles:

$$
\begin{aligned}
\operatorname{SCI}_{\text{wrong}}
=&
\left[
\operatorname{Err}
\left(
q_T^{U,\pi_{\text{profile}}},
\theta
\right)
-
\operatorname{Err}
\left(
p_T^{A,\pi_{\text{profile}}},
\theta
\right)
\right]\\
&-
\left[
\operatorname{Err}
\left(
q_T^{U,\pi_{\text{balanced}}},
\theta
\right)
-
\operatorname{Err}
\left(
p_T^{A,\pi_{\text{balanced}}},
\theta
\right)
\right].
\end{aligned}
$$

A positive value means that updater blindness amplifies profile-conditioned evidence selection.

## 10.4 False self-confirmation definition

Let $W_j$ be the incorrect preference direction encoded by the initial seed for dimension $j$.

Define:

$$
q_t^{\text{wrong},j}
=
P_{q_t}
\left(
\theta_j\in W_j
\right).
$$

The system's false confidence gain is:

$$
FCG_j
=
\operatorname{logit}
q_{t+1}^{\text{wrong},j}
-
\operatorname{logit}
q_t^{\text{wrong},j}.
$$

The **Laundered Confidence Gain** is:

$$
LCG_j
=
FCG_j^{\text{system}}
-
FCG_j^{\text{shadow-aware}}.
$$

An episode counts as a false self-confirming profile only when:

1. the profile remains materially wrong;
2. probability mass on the wrong preference increases;
3. cumulative LCG is positive beyond a declared threshold;
4. the strengthened profile affects subsequent actions;
5. the action-aware shadow does not acquire equivalent confidence.

This excludes inert false memories.

## 10.5 Additional outcomes

* terminal profile Brier score;
* terminal posterior divergence;
* false stable-profile rate;
* false self-confirming-profile rate;
* cumulative action-aware information gain;
* preference-dimension coverage;
* option diversity;
* intrinsic user regret;
* terminal behavioral accuracy.

# 11. Experiment C: Does Open-Loop Evaluation Select the Wrong Updater?

This is the experiment with the highest strong-accept ceiling.

## 11.1 Three evidence-collection regimes

### Fixed balanced logging

Every updater receives the same trajectory generated by a randomized balanced policy.

### Fixed mildly biased logging

Every updater receives the same trajectory generated from a fixed reference profile unrelated to the updater being evaluated.

### Endogenous closed loop

Each updater's profile controls future actions through the common policy function.

The first two test sensitivity to the logging distribution. The third tests genuine endogeneity.

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
s_{\text{open}}^*
=
\arg\min_s
\operatorname{Err}_{\text{open,dev}}(s)
$$

be the updater selected using fixed-history development evaluation.

Let:

$$
s_{\text{closed}}^*
=
\arg\min_s
\operatorname{Err}_{\text{closed,dev}}(s).
$$

Define:

$$
\operatorname{ESR}
=
\operatorname{Err}_{\text{closed,test}}
\left(
s_{\text{open}}^*
\right)
-
\operatorname{Err}_{\text{closed,test}}
\left(
s_{\text{closed}}^*
\right).
$$

**Evaluation Selection Regret** asks:

> How much closed-loop performance is lost because the evaluation protocol selected a system using exogenous fixed histories?

A positive and substantial ESR is stronger than simply reporting low rank correlation.

# 12. Experiment D: Human Pragmatic Evidence Validation

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

The native validation will therefore include an inspectable **memory–persona–action loop**:

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

The paper need not claim a complete reproduction of that system. It should implement an inspectable adapter with:

* logged episodic and semantic memories;
* an explicit persona state;
* deterministic or recorded policy actions;
* auditable memory updates;
* a standardized terminal evaluation.

The native experiment asks whether causal-provenance blindness observed in Track A also appears in a real persistent state loop.

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

A useful result is:

> Full dialogue alone is insufficient, but explicit causal-provenance structure materially reduces over-updating.

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

An LLM may verbalize structured choices, but it does not decide the latent choice in the main evaluation.

The structured decision is sampled first. The verbalizer may produce only semantically equivalent outputs such as:

* “The first option works.”
* “Let's use that one.”
* “Keep the default.”

Generated language is rejected if it adds unsupported claims such as:

> “I generally prefer budget hotels.”

# 17. Simulator-Sensitivity Analysis

Sweep:

* decision noise;
* default susceptibility;
* ranking susceptibility;
* suggestion acquiescence;
* profile strength;
* prior uncertainty;
* trajectory length.

The main paper should include a phase diagram showing where:

1. profile-conditioned actions reduce information;
2. fitted action-aware inference remains calibrated;
3. LLM profile writers over-update;
4. wrong profiles become self-confirming.

The headline result must hold over a meaningful region, including settings where users often reject profile-consistent suggestions.

# 18. Statistical Plan

## Primary Experiment A model

$$
\operatorname{UpdateError}
\sim
\operatorname{Updater}
*
\operatorname{Mechanism}
+
\operatorname{Domain}
+
\operatorname{PriorStrength}
+
(1+\operatorname{Mechanism}\mid\operatorname{User})
+
(1\mid\operatorname{Scenario}).
$$

## Primary closed-loop model

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
\operatorname{Domain}
+
\operatorname{Turn}
+
(1+\operatorname{Policy}\mid\operatorname{User})
+
(1\mid\operatorname{Scenario}).
\end{aligned}
$$

The key interaction is specifically evaluated for incorrect initial profiles.

## Common random numbers

Represent choice as:

$$
Y_t=
\arg\max_x
\left[
\theta^\top\phi(x)
+
\delta(C_t,x)
+
\varepsilon_{t,x}
\right].
$$

Reuse the same $\varepsilon_{t,x}$ draws across counterfactual policy branches whenever options overlap.

This reduces variance and makes user twins meaningfully paired.

## Inference unit

The independent unit is the complete latent user or trajectory, not the turn.

Report:

* trajectory-level paired bootstrap intervals;
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

Calibration transformations and fitted response-model parameters are learned only on training and development users.

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
  "surface_response": "Keep the first one.",
  "choice_noise_seed": 0
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
2. Fitted action-aware inference beats fitted action-unaware inference on held-out simulated interactions.
3. Anchor option identity and attributes remain unchanged across matched conditions.
4. Every matched response exceeds the minimum probability threshold.
5. Latent preference remains fixed throughout the trajectory.
6. Presentation effects never enter intrinsic welfare calculations.
7. Action context and internal policy provenance are stored separately.
8. Common random-number pairing is reproducible.
9. Static logging histories are identical across evaluated updaters.
10. Closed-loop actions depend only on the updater's current profile and declared policy.
11. Terminal diagnostics are independent of the evaluated system's policy.
12. Train, development, and test users are disjoint.
13. Probability calibration never uses test labels.
14. Native profile decoders are blinded to system identity and latent truth.
15. Language verbalization cannot introduce unsupported general-preference claims.
16. Every reported self-confirming case satisfies all five definitional conditions.
17. Ranking results are reproduced over bootstrap samples and random seeds.

# 22. Hypotheses

### H1 — Causal-provenance blindness

Full-context LLM profile writers assign larger updates than fitted action-aware inference after restricted, defaulted, or suggested choices.

### H2 — Context visibility is insufficient

Full-context LLM writers remain closer to fitted action-unaware inference than fitted action-aware inference on at least two provenance mechanisms.

### H3 — Soft self-confirmation

Under soft profile conditioning, incorrect initial profiles gain excess confidence beyond the shadow action-aware posterior.

### H4 — Selection and attribution are distinct

Profile-conditioned policies reduce information even with an action-aware updater, while LLM updaters add further error on the same trajectories.

### H5 — The channels interact

The attribution gap is larger under profile-conditioned policies than balanced policies for initially wrong profiles:

$$
\operatorname{SCI}_{\text{wrong}}>0.
$$

### H6 — Static evaluation can be optimistic

At least some practical updaters have worse common-terminal performance closed loop than predicted by fixed-history evaluation.

### H7 — Causal provenance is actionable

Explicit provenance metadata or provenance-aware instructions reduce update error and self-confirmation without eliminating valid learning from balanced choices or volunteered preferences.

### H8 — Human judgments are more provenance-sensitive

Human evidence-strength judgments distinguish freely elicited and policy-conditioned signals more strongly than ordinary LLM profile writers.

### Secondary H9 — Correction debt

Endogenous reinforcement increases recovery time after an identical explicit correction.

# 23. Planned Main Figures

## Figure 1: Same response, different evidence

For the identical anchor selection, show posterior shifts under:

* volunteered preference;
* balanced choice;
* default;
* suggestion;
* restricted choice.

Plot exact oracle, fitted aware, fitted unaware, and LLM updaters.

## Figure 2: Selection–attribution causal matrix

Show terminal error for:

$$
\text{balanced/profile-conditioned policy}
\times
\text{aware/LLM updater}.
$$

Include $\operatorname{SCI}_{\text{wrong}}$.

## Figure 3: False-profile confidence trajectories

Plot wrong-profile mass over turns for:

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

Show self-confirmation over user noise and presentation-effect strength.

# 24. Planned Results Tables

## Table 1: One-step provenance inference

| Updater              | Balanced | Restricted | Default | Suggestion | Overall |
| -------------------- | -------: | ---------: | ------: | ---------: | ------: |
| Exact aware          |  `[TBD]` |    `[TBD]` | `[TBD]` |    `[TBD]` | `[TBD]` |
| Fitted aware         |  `[TBD]` |    `[TBD]` | `[TBD]` |    `[TBD]` | `[TBD]` |
| Fitted unaware       |  `[TBD]` |    `[TBD]` | `[TBD]` |    `[TBD]` | `[TBD]` |
| Response-only LLM    |  `[TBD]` |    `[TBD]` | `[TBD]` |    `[TBD]` | `[TBD]` |
| Full-context LLM     |  `[TBD]` |    `[TBD]` | `[TBD]` |    `[TBD]` | `[TBD]` |
| Provenance-aware LLM |  `[TBD]` |    `[TBD]` | `[TBD]` |    `[TBD]` | `[TBD]` |

## Table 2: Closed-loop decomposition

| Policy              | Updater              | Terminal error |     LCG | Information gain | User regret | Self-confirming profiles |
| ------------------- | -------------------- | -------------: | ------: | ---------------: | ----------: | -----------------------: |
| Balanced            | Fitted aware         |        `[TBD]` | `[TBD]` |          `[TBD]` |     `[TBD]` |                  `[TBD]` |
| Profile-conditioned | Fitted aware         |        `[TBD]` | `[TBD]` |          `[TBD]` |     `[TBD]` |                  `[TBD]` |
| Balanced            | Full-context LLM     |        `[TBD]` | `[TBD]` |          `[TBD]` |     `[TBD]` |                  `[TBD]` |
| Profile-conditioned | Full-context LLM     |        `[TBD]` | `[TBD]` |          `[TBD]` |     `[TBD]` |                  `[TBD]` |
| Profile-conditioned | Provenance-aware LLM |        `[TBD]` | `[TBD]` |          `[TBD]` |     `[TBD]` |                  `[TBD]` |

## Table 3: Evaluation validity

| Updater      | Balanced-log rank | Biased-log rank | Closed-loop rank | Open-loop optimism | Pairwise reversal |
| ------------ | ----------------: | --------------: | ---------------: | -----------------: | ----------------: |
| `[System A]` |           `[TBD]` |         `[TBD]` |          `[TBD]` |            `[TBD]` |           `[TBD]` |
| `[System B]` |           `[TBD]` |         `[TBD]` |          `[TBD]` |            `[TBD]` |           `[TBD]` |
| `[System C]` |           `[TBD]` |         `[TBD]` |          `[TBD]` |            `[TBD]` |           `[TBD]` |

# 25. Eight-Week Execution Plan

| Week | Deliverable                                                                                                                          |
| ---- | ------------------------------------------------------------------------------------------------------------------------------------ |
| 1    | Freeze latent profiles, susceptibility types, action-context schema, intrinsic utility, exact posterior, and anchor-option generator |
| 2    | Implement fitted aware/unaware baselines, structured-belief interface, calibration, and simulation-based power analysis              |
| 3    | Complete Experiment A pilot across both domains and all three provenance mechanisms                                                  |
| 4    | Freeze go/no-go decision; build closed-loop harness, shadow posterior, common-random-number branches, and native memory states       |
| 5    | Run crossed closed-loop Experiment B and parameter-sensitivity sweeps                                                                |
| 6    | Run two static logging conditions, closed-loop ranking evaluation, common terminal battery, and native memory–action validation      |
| 7    | Complete human evidence-strength validation and provenance-aware mitigation; run correction debt only if stage gates pass            |
| 8    | Statistical analysis, figures, writing, reproducibility audit, and release packaging                                                 |

# 26. Go/No-Go Gates

## Gate 1: Learnable provenance gap

Proceed beyond Experiment A only when:

* fitted action-aware inference outperforms fitted action-unaware inference;
* full-context LLM writers remain materially worse than fitted aware inference;
* the effect appears for at least two provenance mechanisms;
* the effect transfers across both domains and held-out paraphrases.

When fitted aware and unaware systems perform similarly, the environment is not sufficiently identifying.

When LLMs closely track fitted aware inference, the core failure is weak and the paper should stop or reframe.

## Gate 2: Nontrivial soft self-confirmation

Proceed with the self-confirmation headline only when:

* the effect appears under ranking, default, or suggestion;
* counter-profile options remain available;
* wrong profiles gain positive LCG;
* the strengthened profile changes later actions.

An effect found only under fully restricted options is insufficient.

## Gate 3: Attribution beyond evidence selection

The LLM updater must perform worse than the shadow action-aware updater on the **same profile-conditioned trajectories**.

Otherwise, the paper is primarily about recommendation exposure rather than profile inference.

## Gate 4: Native-system validity

At least one inspectable persistent memory–action loop must exhibit the same causal-provenance failure.

Otherwise, restrict the claim to controlled profile updating.

## Gate 5: Evaluation implication

Use the evaluation-reversal headline only when there is:

* low open/closed rank agreement;
* statistically credible pairwise reversals;
* or substantial Evaluation Selection Regret.

When rankings remain stable, retain the causal-provenance paper but remove the claim that static evaluation selects the wrong system.

## Gate 6: Robustness

The main effect must survive:

* another response model;
* broad simulator parameters;
* both domains;
* multiple LLM families;
* natural-language paraphrases;
* exact and fitted action-aware references.

# 27. Expected Contributions

## 1. Causal provenance as a memory-evaluation object

The paper formalizes the complete elicitation process under which a user response is generated, rather than treating the response alone as independent evidence.

## 2. A controlled audit of evidential weight

CAPE-Loop combines anchor-option provenance pairs, naturally sampled interactions, exact inference, and learned action-aware and action-unaware baselines.

## 3. A decomposition of self-confirming profiles

The crossed design and shadow posterior separate information starvation from excessive evidential attribution and measure their interaction.

## 4. A validity test for static personalization evaluation

The paper determines whether fixed histories predict closed-loop profile accuracy, user utility, and deployment-time system selection.

The provenance-aware prompt and metadata conditions are diagnostic mitigations, not a fifth method contribution.

# 28. Relationship to the PhD Agenda

Counterfactual Memory Commitment currently treats a memory operation as an intervention evaluated against later interactions in a user trajectory.

CAPE-Loop establishes that, in a deployed personalized agent:

$$
Q_{>t}
=
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

The strongest final paper would establish the following connected results:

1. **Identical user responses warrant substantially different updates under different elicitation contexts.**
2. **Full-context LLM profile writers often fail to express those differences and behave more like action-unaware inference.**
3. **Initially false profiles gain excess confidence under soft profile-conditioned interaction.**
4. **Evidence selection and evidential attribution make separable contributions and interact.**
5. **The failure appears in an inspectable persistent memory–action loop.**
6. **Humans show greater pragmatic provenance sensitivity than ordinary LLM profile writers.**
7. **Explicit provenance metadata reduces the loop without preventing valid learning.**
8. **A system selected as best under fixed histories is not the best system closed loop.**

The final conclusion would be:

> **A persistent personalized agent can mistake compliance with its own interaction policy for independent evidence about the user. Once stored, that evidence changes later interactions and can make an initially false profile appear increasingly well supported.**

This is now the version to implement. Further conceptual expansion would weaken it. The immediate priority is the anchor-option Experiment A pilot, the fitted action-aware baseline, and the structured-versus-native two-track interface; the full closed-loop run begins only after Gate 1 passes.

[1]: https://arxiv.org/abs/2602.07442 "Echoes in the Loop: Diagnosing Risks in LLM-Powered Recommender Systems under Feedback Loops"
[2]: https://aclanthology.org/2026.findings-acl.1315/ "PersonaAgent: Bridging Memory and Action for Personalized LLM Agents ..."
