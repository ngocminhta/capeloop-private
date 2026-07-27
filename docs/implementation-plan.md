# CAPE-Loop implementation plan

This document turns the scientific proposal into an executable repository plan. It
is the implementation contract for the reference code; the proposal remains the
source of truth for the paper's claims and study design.

## 1. Scope

The repository will provide:

1. A deterministic, inspectable simulator for fixed latent users in the travel
   and writing domains.
2. Versioned records for the complete causal chain:

   ```text
   profile → policy provenance → visible action context → response → profile update
   ```

3. Exact action-aware Bayesian inference under the declared response model.
4. Fitted action-aware and action-unaware likelihood models trained on randomized
   interactions.
5. Structured profile updaters, native memory adapters, and provider-neutral LLM
   request/response exchange.
6. Matched provenance audits, endogenous closed-loop experiments, static-versus-
   closed-loop evaluation, terminal diagnostics, sensitivity sweeps, and human
   study material generation.
7. Metrics, gate reports, reproducible artifacts, examples, tests, and repository
   documentation.

The implementation does not fabricate paper results, claim that a stage gate
passed, call an LLM by default, or treat presentation utility as user welfare.

## 2. Frozen reference defaults

The proposal intentionally leaves pilot parameters open. The reference
implementation uses explicit, replaceable defaults:

| Decision | Reference default |
| --- | --- |
| Runtime | Python 3.11+, standard-library-only core |
| Preference support | `{-2, -1, +1, +2}³` |
| Susceptibility support | low `0.15`, medium `0.45`, high `0.85` |
| Intrinsic scale (`beta`) | `1.0` |
| Rank/default/suggestion scales | `0.35 / 0.80 / 0.65` |
| Minimum matched-choice probability | `0.05` |
| Primary error | Brier score over attribute marginals |
| Information gain | prior entropy minus posterior entropy |
| Calibration | temperature scaling fitted on development records only |
| Run configuration | schema-versioned TOML |
| Canonical artifacts | JSON and JSON Lines |
| Randomness | semantic-keyed SHA-256 streams |

These are software defaults, not preregistered scientific choices. Paper runs
must freeze their own configuration and retain the resolved configuration and
manifest.

## 3. Architectural boundaries

The evaluator is the only component allowed to access latent truth. Policies see
the current profile and domain option pool. Response models see latent user state
and the visible context. Updaters receive one of three explicit views:

- **response-only:** selected option and its attributes;
- **full-context:** response plus the visible elicitation context;
- **provenance-aware:** full context plus structured policy provenance.

The internal policy reason is never placed in `InteractionContext`. Exact
sequential inference retains a joint posterior over preference and susceptibility;
the public structured belief is its preference marginal.

## 4. Component plan

| Component | Responsibility | Main output |
| --- | --- | --- |
| Schemas | Validate users, options, contexts, provenance, observations, updates, and trajectories | Versioned JSON records |
| Domain/builders | Define attributes and option pools in `DomainSpec`; construct anchor contexts, terminal diagnostics, and surface identifiers in experiment builders | Domain and context records |
| Deterministic RNG | Pair counterfactual branches by semantic random keys | Reproducible uniform/Gumbel draws |
| Response models | Softmax random utility and a rule-based noisy-choice robustness component | Choice probabilities/observations |
| Beliefs and inference | Enumerate latent states, normalize posteriors, expose marginals | `PreferenceBelief` |
| Fitted likelihoods | Learn aware/unaware conditional-choice coefficients without external ML libraries | Serializable fitted model |
| Updaters | Prior, exact/fitted, response-only, blind full-context proxy, provenance-discount, conservative and memory variants | Auditable profile update |
| Policies | Balanced, soft profile-conditioned, exploratory, fixed-bias, and hard-filter stress policies | Context + separate provenance |
| Elicitation | Construct invariant matched anchor contexts and eligibility checks | Matched provenance set |
| Trajectory runner | Run/replay histories and maintain a same-history action-aware shadow | Immutable trajectory |
| Native memory | Episodic, semantic/persona, and provenance-linked state with blinded decoder views | Native update records |
| Evaluation | Proper scores, ACUE, KL, confidence, selection/attribution costs, SCI, LCG, regret, ranking, ESR | Metric records |
| Experiments | Implement A–C, sensitivity, and Experiment D material generation; keep correction debt stage-gated | Run artifacts |
| LLM exchange | Export blinded JSONL requests and validate imported structured responses | Replayable model records |
| Reporting | Aggregate JSON/CSV tables, SVG figures, run/gate summaries | Human-readable artifacts |

## 5. Delivery milestones

### M1 — Scientific core and Experiment A

- Freeze schemas, domains, response model, semantic RNG, matched anchors, exact
  posterior, fitted models, structured updater protocol, and one-step metrics.
- Run a deterministic two-domain smoke audit in controlled and naturally sampled
  modes.
- Emit a machine-readable Gate 1 report without asserting passage.

### M2 — Closed-loop Experiment B

- Add initial-profile seeds, balanced/soft/exploratory policies, shadow posterior,
  common-random-number branches, and the five-clause self-confirmation
  predicate.
- Report evidence-selection cost, attribution cost, SCI, LCG, intrinsic regret,
  coverage, and information gain, with deterministic paired user-clustered
  intervals and an explicitly non-GLMM trajectory-cluster sensitivity view.

### M3 — Evaluation validity and native memory

- Add fixed balanced, fixed mildly biased, and endogenous logging regimes.
- Add a common exogenous terminal diagnostic battery, practical memory/update
  variants, two blinded decoder views, rank comparisons, and Evaluation
  Selection Regret.

### M4 — External evaluation and robustness

- Add LLM JSONL exchange, development-only calibration, prompt hashes, sensitivity
  grids, alternative response models, human-study packets, and correction-debt
  support guarded by stage gates.

The software scope of this milestone is implemented: it includes hash-bound
offline replay, explicitly authorized live turn-by-turn provider orchestration,
development-only LLM calibration, held-out paraphrase transfer, alternative
response-model sweeps, blinded human-study generation and analysis, and
stage-gated correction-debt diagnostics. Empirical execution still requires
external inputs such as authorized provider responses, genuinely distinct
decoder sources, recorded native-system actions, and eligible de-identified
human ratings. The exact software/evidence boundary is maintained in
[Implementation status](implementation-status.md).

### M5 — Release readiness

- Complete examples, schema exports, docs, contributor files, CI, packaging, and
  deterministic artifact verification.

## 6. Verification contract

The automated suite maps directly to the proposal's 17 requirements:

1. posterior normalization and brute-force agreement;
2. identifying fixture where fitted aware beats unaware;
3. anchor identity invariance;
4. matched-response probability threshold;
5. fixed latent preference;
6. welfare/presentation separation;
7. context/provenance separation;
8. common-random-number reproducibility;
9. identical static histories;
10. declared closed-loop policy inputs only;
11. exogenous terminal diagnostics;
12. disjoint latent-user groups and a group-disjoint template manifest;
13. no test-label calibration;
14. blinded native decoders;
15. constrained verbalization;
16. complete five-part self-confirmation predicate;
17. reproducible bootstrap ranks within each run plus immutable comparison of
    point rankings, inferential orders, Gate 5, and ESR selections across
    distinct compatible random-seed runs.

Additional tests cover serialization, invalid record rejection, likelihood
stability, CLI smoke runs, artifact checksums, and train/dev/test leakage.

## 7. Completion criteria

The executable core-infrastructure milestone is complete when:

- `PYTHONPATH=src python -m unittest discover -s tests` passes without network
  access;
- `PYTHONPATH=src python -m cape_loop doctor` reports a valid environment;
- the smoke suite creates a self-describing run directory and verifies its
  checksums;
- every implemented component and experiment has a documented input, output,
  latent-truth access rule, and extension point;
- planned or externally dependent conditions are labeled honestly;
- no generated table or narrative is presented as an empirical paper result
  unless backed by retained run artifacts.

Completing this milestone does not mean that every stage-gated study in the
proposal has run. External and missing conditions remain incomplete until the
status page records an implemented, tested path.
