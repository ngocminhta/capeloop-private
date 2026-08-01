# Configuration

CAPE-Loop accepts strict TOML through:

```bash
PYTHONPATH=src python -m cape_loop config validate CONFIG
```

Validation prints the fully resolved configuration as JSON. Every unknown root
table or section key is an error, and the selected experiment kind imposes an
additional contract for factor fields whose inactive values would otherwise
look like executed experiment cells. Some shared tables remain present in every
resolved config even when a run kind does not consume them; those boundaries
are called out below.

## Checked-in configurations

CAPE-Loop keeps user-facing presets in three places:

```text
configs/smoke.toml     quick offline validation
configs/offline/       deterministic study and source-packet designs
configs/live/          bounded provider-specific pilots
```

The 18 public presets are intentionally split by responsibility:

| File | Kind | Purpose |
| --- | --- | --- |
| `configs/smoke.toml` | A | Fast offline implementation check |
| `configs/offline/experiment_a.toml` | A | Synthetic Experiment A dataset candidate |
| `configs/offline/experiment_b.toml` | B | Full offline closed-loop reference |
| `configs/offline/gate4_source.toml` | B | Gate 4 source: 640 decoder requests/source and 80 native actions |
| `configs/offline/experiment_c.toml` | C | Full offline evaluation-validity reference, seed 1729 |
| `configs/offline/experiment_c_seed_271828.toml` | C | Matched robustness seed 271828 |
| `configs/offline/experiment_c_seed_314159.toml` | C | Matched robustness seed 314159 |
| `configs/offline/experiment_c_rescore_source.toml` | C | External-rescore source: 360 decoder requests/source |
| `configs/offline/sensitivity.toml` | sensitivity | Baseline-first 22-point OAT robustness design |
| `configs/live/experiment_a_openai.toml` | A | Direct-OpenAI primary pilot |
| `configs/live/experiment_a_openai_replication.toml` | A | Matched GPT-5.6 model-variant replication |
| `configs/live/experiment_a_openrouter.toml` | A | Matched OpenRouter/Gemini pilot |
| `configs/live/experiment_b_openai.toml` | B | Six-turn, one-model-at-a-time direct-OpenAI pilot |
| `configs/live/experiment_b_openrouter.toml` | B | Matched six-turn OpenRouter pilot |
| `configs/live/experiment_c_openai.toml` | C | Bounded direct-OpenAI pilot |
| `configs/live/experiment_c_openrouter.toml` | C | Matched OpenRouter pilot |
| `configs/live/sensitivity_openai.toml` | sensitivity | Direct-OpenAI Gate 6 OAT pilot |
| `configs/live/sensitivity_openrouter.toml` | sensitivity | Matched OpenRouter Gate 6 OAT pilot |

Transport-only smoke presets and the superseded 81-point Cartesian diagnostic
are deliberately not public configs. Provider behavior is covered by offline
tests and explicit bounded commands instead of additional near-duplicate
TOMLs. Local model/route variants belong under ignored `configs/local/`; copy a
reviewed live preset there rather than modifying the checked declaration. The
optional R job owns its synthetic fixture at
`analysis/confirmatory-mixed-effects/fixtures/confirmatory_ci.toml`.

These files are executable designs, not preregistrations or paper results.
Every live pilot declares the approved hard ceiling of 900 physical HTTP
attempts and 6,000,000 conservatively accounted tokens per provider ledger.
They set `max_retries = 0`, so the complete retry-expanded plan remains inside
those ceilings.

The live pilot counts are whole-design bounds, not expected call counts:

| Pilot | Experiment calls | Calibration | A paraphrases | Physical bound | Maximum output allocation |
| --- | ---: | ---: | ---: | ---: | ---: |
| A primary/replication/OpenRouter | 480 | 60 | 40 | 580 | 1,187,840 |
| B pair | 576 | 60 | 0 | 636 | 1,302,528 |
| C pair | 768 | 60 | 0 | 828 | 1,695,744 |
| Gate 6 OAT pair | 720 | 0 | 0 | 720 | 1,474,560 |

The A row is the current controlled-only design: four users × two domains ×
three attributes × two anchor directions × two prior strengths × five
mechanisms × one LLM updater gives 480 experiment requests. Its one-user,
five-mechanism development probe adds 60 requests, and the controlled held-out
paraphrase suite adds 40. All three presets disable retries, so 580 logical
requests are also 580 physical-attempt bounds.

The offline Gate 4 and Experiment C external-rescore source configs do not call
models. They are sized so their later selected OpenRouter or optional direct
external-model collections fit the same approved per-source ceilings.

The live-only diagnostic commands are deliberately not additional TOML presets
or configuration schemas:

- [`demo one-scenario`](getting-started.md#run-one-understandable-live-scenario)
  makes exactly one OpenRouter update; and
- [`demo experiment-b-case`](getting-started.md#run-one-multi-turn-experiment-b-case)
  makes two logical updates per selected turn and at most that many physical
  provider calls for one matched B user case. It accepts complete
  three-attribute cycles of 3, 6, 9, or 12 turns.

Both use zero retries and hard request/token guardrails. The one-scenario
command uses the official OpenRouter endpoint with no fallback. The B
diagnostic uses the same constrained OpenRouter path by default and can instead
select the official direct OpenAI path with `--provider openai`. Their input,
model, routing, and execution controls are command-line options. Use their
`--help` output for the small command surfaces, and use a checked-in or reviewed
local TOML configuration for any pilot or paper experiment.

For Experiment B, three turns are a transport check only: each of the three
attributes normally appears once, leaving no later same-attribute action for
the stored update to influence. The public live B TOMLs therefore use six turns
and one evaluated LLM arm at a time. They select the `correct` and `incorrect`
seeds and cross balanced, soft, and exploratory policies. This revisits every
attribute, preserves eight independent user clusters across two domains,
supports the disconfirmation-deficit comparison, and stays below the approved
request ceiling.

The checked-in
`data/model-suites/experiment-b-bounded-calibration-v1.json` freezes the
multi-model use of that base design. Its primary panel is Gemini 3.6 Flash,
GPT-5.6 Luna, and Mistral Large 3 (`mistralai/mistral-large-2512`), each run over
the complete B design in an isolated output subtree. DeepSeek V4 Flash is a
post-pilot targeted secondary replication containing only the incorrect-seed
balanced-versus-soft contrast. Every model has a separate analysis; no model
outputs are pooled, and DeepSeek is outside the primary analysis set.

Plan all four arms, including their resolved conditions, output paths, and hard
budgets, without loading a credential or calling a model:

```bash
PYTHONPATH=src python -m cape_loop experiment-b model-suite \
  configs/live/experiment_b_openrouter.toml \
  --output-root runs/experiment-b-suite
```

Only the additional `--execute-live` flag authorizes the command to execute the
four paid OpenRouter runs sequentially. Planning is the default; a successful
plan is not model evidence.

## Root schema

The only accepted root keys are:

```toml
schema_version = 1

[run]
[scenarios]
[population]
[experiment]
[response_model]
[inference]
[thresholds]
[manipulation]
[sensitivity]
[llm]
[artifacts]
```

`schema_version = 1` is required. Omitted tables use the defaults listed below.
The smallest valid file is:

```toml
schema_version = 1
```

That minimal file resolves to the default one-step provenance audit; it is not a
paper configuration.

Validation follows TOML types strictly. Integer fields reject Booleans; numeric
scientific fields reject Booleans, nonnumeric values, NaN, and infinities; and
`run.deterministic` plus all `[artifacts]` flags require actual Booleans. Factor
arrays, `susceptibility_levels`, and every sensitivity-coordinate array reject
duplicate entries.

## `[run]`

| Key | Type | Default | Validation and behavior |
| --- | --- | --- | --- |
| `name` | string | `"cape-loop-smoke"` | Nonempty; `/`, `\`, and NUL are forbidden |
| `seed` | integer | `1729` | Nonnegative semantic-randomness root |
| `output_root` | string | `"runs"` | Nonempty default output parent |
| `deterministic` | Boolean | `true` | Recorded in the manifest |

The simulation implementation is semantic-key deterministic regardless of
array traversal. `run.deterministic` is a recorded whole-run declaration, not
an alternate execution engine. It must be `false` when an `llm_*` updater uses
live OpenAI or OpenRouter generation, because seeded simulator state does not
make an external model response deterministic. Loading a frozen
`scenarios.conversation_file` does not change determinism: no authoring call is
made during the run.

The run directory is:

```text
<output-root>/<run.name>-<first-12-hex-of-resolved-config-SHA256>
```

`run CONFIG --output-root DIR` changes the parent directory but not the
configuration digest. `--allow-existing` reuses only a checksum-valid completed
directory whose source digest matches the current source tree. For an LLM run,
the retained input manifest must also exactly match the current replay corpus or
declared live-model configuration.

Live OpenAI and OpenRouter modes additionally require
`run CONFIG --execute-live`. If a live attempt fails,
`--resume-failed-live` may be combined with `--execute-live` to preserve the
failed artifact under
`<output-root>/.failed-runs/<run-id>-attempt-NNN/`, recreate the deterministic
run path, and resume its external provider journal. It accepts only a failed
artifact with the same resolved configuration; it cannot overwrite or resume a
completed run.

## `[scenarios]`

| Key | Type | Default | Validation and behavior |
| --- | --- | --- | --- |
| `catalog_file` | string | `""` | Path to the strict versioned JSON catalog; empty enables only the legacy/programmatic generated-surface fallback |
| `catalog_sha256` | string | `""` | Lowercase 64-character SHA-256 of the exact catalog bytes |
| `selection_policy` | string | `"deterministic-stratified-v1"` | Only accepted selection policy |
| `conversation_file` | string | `""` | Frozen per-scenario natural-language template bank; requires `catalog_file` |

`catalog_file` and `catalog_sha256` must be both empty or both nonempty. Every
checked-in configuration opts in to
`data/scenarios/scenario-catalog-v1.json`; the empty default exists so legacy
and direct programmatic configurations remain interpretable. The configured
digest is verified before provider construction or run-artifact creation, and
reuse also requires the retained catalog input manifest to match.

Selection filters by domain, split, and target attribute, then uses the run
seed and an experiment-owned semantic pairing key. Longitudinal histories
consume a deterministic per-cell permutation without replacement until the
pool is exhausted. Selection does not use latent truth, current belief,
observed response, updater result, or sensitivity-grid point.

Experiment A assigns one scenario to each user–domain–target pair and reuses it
for both anchor directions; their physical orders are opposite, and that
scenario/order pair is reused across mechanisms, response modes, prior
strengths, and updaters. Experiment B keys each per-target occurrence schedule
to the common trajectory-pair key. Its `exploratory` v2 policy keeps target
counts within one by choosing among the least-exposed attributes and using
current marginal entropy only to order those eligible targets.

Official hybrid configurations set:

```toml
[scenarios]
catalog_file = "data/scenarios/scenario-catalog-v1.json"
catalog_sha256 = "<catalog digest>"
selection_policy = "deterministic-stratified-v1"
conversation_file = "data/scenarios/conversation-templates-v1.json"
```

`conversation_file` does not configure a runtime LLM. The mathematical
response model selects an option first; the frozen bank then renders a natural
assistant/user exchange for the visible context and selected option. Empty
preserves legacy and small programmatic fixtures that do not exercise the
hybrid surface. A configured bank must cover every catalog scenario and all
four option IDs for each one.

The authoring workflow can use OpenRouter to produce one neutral
`base_template` and four `display_names` per scenario. That is a candidate
workflow: the current 48 visible bases were subsequently project-standardized
outcome-blind onto three source-neutral frames. Each frame appears 16 times
overall and twice in every six-scenario test domain×target cell, and every
record remains unreviewed. The historical OpenRouter log records candidate
calls rather than provider authorship of the current visible text. Code expands
each base into the five stored presentation forms:
balanced/restricted/ranking share neutral wording, while default and suggested
receive only their fixed treatment sentence. Code also fixes
`choice_template = "I choose {selected_name}."`; neither treatment language nor
the reply is model-authored. Runtime assigns A/B by visible position and maps
control-plane option IDs to `presented_option_N` before constructing an
evaluated-model prompt.

Audit the exact catalog, bank, coefficients, susceptibility levels, configured
domains and policies, split, matched-probability floor, and planned horizon
bound by a configuration:

```bash
cape-loop scenarios audit CONFIG OUTPUT_DIR --split test --turns 16
```

Omit `--turns` to use `experiment.turns`. This command is local and outcome
free; it writes a machine report and an exhaustive human-review packet but does
not change any catalog status.

Author or refresh the bank separately:

```bash
cape-loop conversations generate-openrouter \
  data/scenarios/scenario-catalog-v1.json \
  data/scenarios/conversation-templates-v1.json \
  --model anthropic/claude-sonnet-5 --execute-live
```

The command writes
`data/scenarios/conversation-templates-v1.generation.jsonl` beside the bank.
It is an authoring record, not an experiment response file. Review and freeze
the result before referencing it from a run.

The catalog is a frozen scientific input. Any byte edit—including wording,
metadata, or formatting—requires a new catalog version/freeze and an updated
`catalog_sha256` in every configuration that consumes it. Do not change only
the digest to bless an unreviewed edit, and never modify the retained copy in a
completed run. Catalog structure, current eligibility, and retained artifacts
are documented in
[Data model](data-model.md#scenario-catalog-input).

## `[population]`

| Key | Type | Default | Validation and behavior |
| --- | --- | --- | --- |
| `theta_policy` | string | `"legacy-hash-v1"` | `"legacy-hash-v1"` or `"orthogonal-balanced-v2"` |
| `susceptibility_policy` | string | `"legacy-hash-v1"` | `"legacy-hash-v1"` or `"orthogonal-balanced-v2"` |

The table is opt-in so an old config with no `[population]` table keeps its
exact historical split, user sequence, resolved configuration, and run
identity. All checked-in presets explicitly select `orthogonal-balanced-v2`
for both fields.

For theta, v2 partitions the 64 complete three-coordinate profiles into
strength-two orthogonal arrays: train receives 32 profiles and development and
test receive 16 each. Every coordinate level and every coordinate pair is
balanced within each split. The deterministic allocation order is also
marginally balanced in four-user blocks.

For presentation susceptibility, v2 partitions the 27 complete profiles into
nine profiles per split. Within each split, every level of ranking, default,
and suggestion susceptibility appears three times, and every pair of
coordinate levels appears once. The deterministic allocation order is
marginally balanced in three-user blocks. Counts not divisible by these block
sizes differ by at most one user per coordinate level. The policies balance a
designed synthetic population; they do not estimate a distribution of real
people.

When both v2 policies are selected, the runner does not pair their two balanced
orders by an arbitrary shared index. A cached, seed-stable combinatorial search
chooses legal four-user theta blocks and three-user susceptibility blocks to
reduce cross-coordinate contingency imbalance and linear association at the
official \(N=\{4,8,10,16,20,24,32\}\) horizons. The search is outcome blind and
preserves every split, support, profile-count, and marginal-prefix guarantee.
It reduces finite-sample composition dependence; it does not claim statistical
independence, especially at \(N=4\), where the cross-tables are necessarily
sparse.

## `[experiment]`

| Key | Type | Default |
| --- | --- | --- |
| `kind` | string | `"provenance_audit"` |
| `domains` | nonempty string array | `["travel", "writing"]` |
| `mechanisms` | nonempty string array | `["balanced", "restricted", "ranking", "default", "suggested"]` |
| `response_modes` | nonempty string array | `["controlled_anchor", "naturally_sampled"]` |
| `prior_strengths` | nonempty numeric array | `[0.0]` |
| `initial_profile_conditions` | nonempty string array | `["correct", "incorrect", "uncertain", "empty"]` |
| `policies` | nonempty string array | `["balanced"]` |
| `updaters` | nonempty string array | baseline set shown below |
| `users` | integer | `8` (test user/cluster count) |
| `trajectories_per_cell` | integer | `1` |
| `turns` | integer | `1` |
| `bootstrap_replicates` | integer | `0` |

`users`, `trajectories_per_cell`, and `turns` must be positive.
`bootstrap_replicates` must be nonnegative.

For ordinary A–C preparation, \(N=\texttt{users}\) is the test-user count.
The runner separately creates
\(\max(24,\min(128,4N))\) training users and \(\max(8,N)\) development
users. When both domains are selected, the same shared user ID and latent state
produce one retained population row per domain; those rows are not additional
independent users.

For `closed_loop`, `bootstrap_replicates = 0` retains point estimates but marks
the Experiment B clustered intervals and one-sided directional tests as
`not_computed`; Gates 2 and 3 therefore cannot pass. Any positive value enables
both the complete-user bootstrap intervals and the one-sided paired
complete-user sign-flip decisions. The value is the bootstrap resample count;
it does **not** set the number of sign patterns. The latter is determined from
the number of complete user clusters, using exact enumeration for small samples
and a code-bounded Monte Carlo reference distribution for larger samples.

The same positive value is also requested for the Experiment B pilot-power
simulation, which is separately bounded to 200–10,000 simulations. A zero value
still produces the inexpensive 200-simulation planning fallback. The candidate
user counts (16, 32, 64, 128), alpha (0.05), target power (0.80), factor
contrast, and lower-Wilson-bound decision rule are frozen in code rather than
exposed as quietly mutable paper settings. The resulting candidate is advisory
and never commits the final sample size automatically.

`prior_strengths` is Experiment A's executable prior-concentration factor.
Every value must be finite, unique, and in `[0, 1)`. For each latent user, a
level `s` mixes `(1-s)` uniform joint mass with `s` mass on that user's true
theta. This truth-aligned construction is balanced by the latent population;
all updaters and provenance mechanisms within a matched stratum receive the
same prior. The default `[0.0]` preserves the original uniform-prior pilot.
Use, for example, `[0.0, 0.35, 0.70]` for a crossed confirmatory run and account
for the proportional increase in LLM requests. B, C, and sensitivity reject
any value other than `[0.0]`.

For Experiment A, `controlled_anchor` selects the primary same-response track:
the selected anchor and local user reply are identical across all requested
mechanisms and the runner verifies that invariant. `naturally_sampled` is an
optional secondary A robustness track. The checked-in offline A preset selects
both; all public live A presets select only `controlled_anchor`. The primary
controlled analysis uses the exact action-aware generating-model reference;
the fitted aware reference remains secondary.

`initial_profile_conditions` is an Experiment B factor. It accepts a unique
nonempty subset of `correct`, `incorrect`, `uncertain`, and `empty`; other run
kinds require all four defaults because they do not consume this field. The
offline reference keeps the full crossing. The bounded live B presets keep
`correct` as a control and `incorrect` as the primary false-seed condition,
omitting `uncertain` and `empty` to fund the exploratory policy without reducing
the independent-user count.

The two suggestion labels are deliberately scoped: `suggested` is Experiment
A's matched-context condition name, while `suggestion` is the causal
presentation-channel identifier declared by Experiments B, C, and sensitivity.
Strict per-experiment contracts prevent using one in the other's slot.

Accepted experiment kinds:

```text
provenance_audit
closed_loop
evaluation_validity
sensitivity
```

Accepted domains:

```text
travel
writing
```

Globally accepted mechanism labels:

```text
balanced
restricted
ranking
default
suggested
suggestion
```

Experiment A (`provenance_audit`) accepts only `balanced`, `restricted`,
`ranking`, `default`, and `suggested`. Its ranking condition reverses the
balanced display order while holding the anchor and option set fixed.
Experiments B, C, and sensitivity require the set `ranking`, `default`, and
`suggestion`; their policies produce the actual presentation context.

Accepted response modes:

```text
controlled_anchor
naturally_sampled
```

Accepted policies:

```text
balanced
soft_profile_conditioned
exploratory
fixed_bias
hard_filter
```

`exploratory` means `v3-balanced-coverage-shared-neutral-ranking`, not unconstrained entropy
maximization. Each complete three-turn block covers all three attributes, while
the within-block order may adapt to current marginal entropy. The scenario
capacity checker consequently uses \(\lceil T/3\rceil\) scenarios per cell for
this policy. Unknown or custom adaptive policies remain conservatively budgeted
at \(T\) scenarios per cell.

Accepted updaters:

```text
no_update
exact_action_aware
fitted_action_aware
fitted_action_unaware
response_only
full_context_blind
provenance_discount
provenance_aware
conservative
episodic_memory
semantic_memory
provenance_linked_memory
llm_response_only
llm_full_context
llm_provenance_aware
```

The default updater tuple is:

```text
no_update
exact_action_aware
fitted_action_aware
fitted_action_unaware
provenance_discount
```

### Strict experiment contracts

After field-level validation, these rules are enforced:

| Kind | Required contract |
| --- | --- |
| `provenance_audit` | `policies = ["balanced"]`; `trajectories_per_cell = 1`; `turns = 1`; prior strengths are unique values in `[0, 1)`; non-negative `bootstrap_replicates` (zero selects the 200-replicate smoke fallback) |
| `closed_loop` | mechanisms are exactly ranking/default/suggestion; response modes are exactly naturally sampled; `prior_strengths = [0.0]`; nonnegative `bootstrap_replicates` (`0` is smoke-only and cannot satisfy Gates 2/3) |
| `evaluation_validity` | mechanisms are exactly ranking/default/suggestion; response modes are exactly naturally sampled; policies are exactly balanced/fixed-bias/soft-profile-conditioned; `prior_strengths = [0.0]` |
| `sensitivity` | mechanisms are exactly ranking/default/suggestion; response modes are exactly naturally sampled; policies are exactly balanced/soft-profile-conditioned; `prior_strengths = [0.0]`; `turns = 1`; `bootstrap_replicates = 0`; an `llm_*` updater additionally requires `llm.calibration = "none"`, `artifacts.retain_prompts = true`, and `artifacts.retain_events = true` |

For set-based requirements, ordering is not significant. For the one-element
provenance-audit policy and response-mode tuples, the exact listed tuple is
required.

Sensitivity passes one shared, content-addressed completion provider through
all grid points. This supports replay, direct OpenAI, and OpenRouter without
mistaking point-specific prompts for a fixed corpus. Point-specific LLM
temperature fitting is deliberately unsupported: use raw vectors with
`llm.calibration = "none"`. Full prompts and trajectory events are mandatory
so every consumed response remains reconstructable and linked to its adaptive
history. The generic `experiment.turns` must be `1`; executed lengths come
exclusively from `sensitivity.trajectory_lengths`.

Experiment A may use a reviewed subset of mechanisms or response modes, but a
subset may make a gate criterion incomplete. The other runners require the
fixed mechanism/response-mode declarations above. In Experiments B and C,
presentation is produced by the selected policies; the required mechanism list
records the supported presentation channels rather than directly choosing each
turn.

## `[response_model]`

| Key | Default | Validation |
| --- | ---: | --- |
| `beta` | `1.0` | positive |
| `decision_noise` | `1.0` | positive |
| `rank_scale` | `0.35` | nonnegative |
| `default_scale` | `0.80` | nonnegative |
| `suggestion_scale` | `0.65` | nonnegative |
| `susceptibility_levels` | `[0.15, 0.45, 0.85]` | nonempty, finite, nonnegative, and unique |
| `minimum_matched_probability` | `0.05` | strictly between `0` and `0.5` |

The runner divides intrinsic and presentation coefficients by
`decision_noise`. Presentation coefficients affect simulated choices, never
intrinsic welfare or regret.

The generic default remains `default_scale = 0.80` for backward-compatible
programmatic configs. Every checked-in study preset explicitly uses the
prospectively calibrated value `0.75`. At the declared susceptibility support,
this keeps both binary responses strictly above the configured 0.05 matched
probability floor in every audited mechanism and display order; the stronger
0.80 value made the nondefault response slightly too rare in the most
susceptible state.

`minimum_matched_probability` is used by Experiment A's matched-anchor
eligibility check and by the prospective audit's symmetric, per-display-order
binary-response guardrail.

## `[inference]`

| Key | Default | Validation |
| --- | ---: | --- |
| `training_interactions` | `512` | positive |
| `fit_steps` | `600` | positive |
| `learning_rate` | `0.04` | positive |
| `l2` | `0.001` | nonnegative |
| `calibration` | `"temperature"` | `"temperature"` or `"none"` |

Aware and unaware likelihood models are fitted on the same generated randomized
training records. The runner then generates a distinct development set.

With `calibration = "temperature"`, it fits separate aware and unaware
temperatures on development outcomes and divides each fitted coefficient vector
by its temperature. It retains:

- raw bundle: `models/raw-fitted-likelihoods.json`;
- active bundle used by updaters: `models/fitted-likelihoods.json`;
- transformation: `models/calibration.json`; and
- raw and active development diagnostics:
  `models/held-out-response-diagnostics.json`.

The aware diagnostic is option-identity NLL and the unaware diagnostic is
semantic-direction NLL. Their outcome spaces differ, so their numeric NLLs are
not a like-for-like model comparison. Preference-posterior scores are used for
the provenance audit.

With `calibration = "none"`, raw and active bundles are equal and the
calibration artifact records `kind = "none"`.

This `[inference]` calibration applies only to fitted simulator likelihood
references. LLM belief-vector calibration is configured independently under
`[llm]`. It also does not enable or disable Experiment B's clustered inference:
that switch is `experiment.bootstrap_replicates`, as described above.

## `[thresholds]`

| Key | Default | Validation/use |
| --- | ---: | --- |
| `materially_wrong_mass` | `0.50` | in `[0, 1]`; self-confirmation assessment |
| `laundered_confidence_gain` | `0.25` | nonnegative; self-confirmation assessment |
| `shadow_equivalence_tolerance` | `0.05` | in `[0, 1]`; maximum terminal wrong-mass gap still treated as equivalent shadow confidence |
| `false_stability_tolerance` | `0.02` | in `[0, 1]`; maximum longitudinal excursion from seeded wrong mass for false-stable classification |
| `direction_tolerance` | `1e-9` | nonnegative; update-direction comparisons |
| `ranking_tie_tolerance` | `1e-6` | nonnegative; Experiment C ranking ties |
| `selection_noninferiority_margin` | `0.02` | in `[0, 2]` on the marginal-Brier error scale; Experiment B SelectionCost practical noninferiority margin |
| `net_harm_margin` | `0.02` | in `[0, 2]` on the marginal-Brier error scale; Experiment B soft-minus-balanced updater-error harm margin |
| `decomposition_tolerance` | `1e-12` | positive; numerical equality tolerance for the Experiment B error decomposition |

The two `0.02` margins have different scientific roles. The legibility gate
tests the incorrect-initial-profile soft-versus-balanced comparison:
SelectionCost must be below the frozen noninferiority margin while the two
attribution-gap contrasts are positive. The nested net-profile-harm gate uses
that same stratum and additionally tests whether soft-minus-balanced updater
terminal error is above the separate harm margin. These are one-sided paired
complete-user randomization decisions when Experiment B inference is enabled,
not decisions from the sign of a point estimate alone; clustered intervals
remain reported as sensitivity evidence. `decomposition_tolerance` is only an
accounting invariant for

```text
soft-minus-balanced updater error
= SelectionCost + soft-minus-balanced same-history attribution gap
```

and is not a scientific effect-size margin. Freeze all scientific thresholds
before analyzing a paper-intended test set.

## `[manipulation]`

| Key | Default | Validation/use |
| --- | ---: | --- |
| `planning_mode` | `"disabled"` | `"disabled"` or `"required"`; required planning is available only for Experiment B |
| `minimum_informative_active_turns` | `2` | positive integer; minimum susceptible active turns in every paired trajectory |
| `minimum_active_mechanisms` | `2` | positive integer, currently at most `2` because the planner actively schedules default and suggestion |
| `minimum_decisive_active_controls` | `1` | positive integer; minimum active control turns expected to resist the presentation change |
| `minimum_informative_choice_divergence_probability` | `0.02` | in `[0, 1]`; minimum conservative shared-noise choice-divergence probability for an informative active turn, across either current-profile direction |
| `maximum_decisive_choice_divergence_probability` | `0.05` | in `[0, 1]`; maximum predicted divergence probability for a decisive active control |
| `minimum_active_susceptibility_mass` | `0.05` | nonnegative; minimum per-trajectory sum of conservative predicted divergence probabilities over active turns |
| `require_counter_profile_options` | `true` | Boolean; retain both preference directions rather than removing all counter-profile evidence |
| `offline_response_seeds` | `32` | positive integer; simulator draws in the offline stress audit; use a smaller explicit CLI override for smoke checks and reserve 64 for a one-time final audit when runtime permits |

With `planning_mode = "required"`, Experiment B must include both `balanced`
and `soft_profile_conditioned`, a scenario catalog, and enough turns for the
declared informative and decisive roles. The runner fixes one scenario, role,
mechanism, neutral ranking, and response-noise schedule per
domain-user-replicate group and reuses it across correct and incorrect initial
profiles. Active promotion follows the current profile direction; if that
direction is exactly neutral, it uses and logs the condition-specific initial
direction frozen in the plan. Realized choices, updated profiles, and
evaluated-model outputs are forbidden admission inputs. A plan that misses any
requirement fails before an evaluated-model call.

Run the same planning logic and its multi-seed simulator stress audit before
spending API budget:

```bash
PYTHONPATH=src python -m cape_loop experiment-b manipulation-audit \
  configs/live/experiment_b_openrouter.toml \
  artifacts/experiment-b-manipulation-audit
```

This command has no live-execution flag and makes zero LLM calls. It writes the
complete JSON schedule, a readable Markdown plan, and a JSON audit covering
active-treatment execution, neutral-direction fallback use, susceptibility,
choice divergence, expected information, SelectionCost, condition/domain
strata, prospective roles, and a descriptive active-turn cross-tab over role,
mechanism, effective/planned direction, target, and domain. Cross-tab counts
reconcile to the pooled active total but do not create new admission gates.
`--response-seeds N`
overrides only the offline simulator draw count. Passing this audit establishes
coverage under the declared simulator, not an LLM effect or paper eligibility.

## `[sensitivity]`

| Key | Default | Validation |
| --- | --- | --- |
| `design` | `"cartesian"` | `"cartesian"` or `"one_at_a_time"` |
| `decision_noise_values` | `[0.6, 1.0, 1.6]` | nonempty; finite, positive, and unique |
| `presentation_multipliers` | `[0.5, 1.0, 1.5]` | nonempty; finite, nonnegative, and unique |
| `profile_conditioning_strength_values` | `[1.0]` | nonempty; finite, unique policy-propensity doses in `[0, 1]`; `1` preserves ordinary soft conditioning and `0` is the balanced-action negative control |
| `rank_multipliers` | `[1.0]` | nonempty; finite, nonnegative, and unique |
| `default_multipliers` | `[1.0]` | nonempty; finite, nonnegative, and unique |
| `suggestion_multipliers` | `[1.0]` | nonempty; finite, nonnegative, and unique |
| `profile_strength_values` | `[0.65, 0.80, 0.90]` | nonempty; finite, unique, and each in `[0.5, 1)` |
| `prior_uncertainty_values` | `[0.0]` | nonempty; finite, unique, and each in `[0, 1)` |
| `trajectory_lengths` | `[4, 8, 12]` | nonempty; unique positive integers |
| `response_model_families` | `["random_utility"]` | unique values from `random_utility`, `rule_based` |
| `rule_noise_values` | `[0.15]` | nonempty; finite, unique values in `[0, 1]`; used only for rule-based points |
| `phase_min_selection_cost` | `0.0` | finite; phase selection-cost threshold |
| `phase_max_aware_ece` | `0.10` | finite value in `[0, 1]` |
| `phase_min_attribution_gap` | `0.0` | finite; phase soft-minus-balanced same-history attribution-gap threshold |
| `phase_min_self_confirming_rate` | `0.0` | finite value in `[0, 1]`; reference threshold for the secondary strict endpoint, not an operational-joint-region criterion |
| `phase_min_suggestion_rejection_rate` | `0.20` | finite value in `[0, 1]`; frozen “often rejects” criterion |

These fields are consumed only by `kind = "sensitivity"`. Under
`one_at_a_time`, the first value of every numeric axis and the first
response-model family define the baseline; each later value is varied
separately. Alternate response families are evaluated at the numeric baseline,
with every declared rule-noise value for a rule-based alternative.

The canonical `configs/offline/sensitivity.toml` declaration is a 22-point OAT
design. It covers every declared axis efficiently but cannot estimate
interactions among axes; artifacts record
`interaction_effects_estimable = false`. The two live Gate 6 OAT pilots contain
14 points whose trajectory lengths sum to 45. The first value on the policy
dose axis is `1.0` so the baseline point is byte-for-byte compatible with the
ordinary soft-policy propensity; `0.0`, `0.33`, and `0.67` are separate OAT
departures. The dose participates in grid completion, manipulation summaries,
and phase-boundary inference but does not silently change the version-1
Gate 6 broad-simulator-parameter clause. In particular, the null point is a
negative control rather than a level at which harm should persist. At the null,
treatment exposure and visible-action divergence must both be zero. A positive
dose passes its manipulation check only when both exposure and visible-action
divergence are positive; zero divergence is a failed manipulation, not a null
effect estimate. The strict self-confirming-profile rate remains a secondary
endpoint and does not control the operational joint region. The suggestion
rejection rate counts only profile-conditioned suggestions where the
counter-profile option remains displayed; selecting that alternative is a
rejection.

## `[llm]`

This section configures the evaluated profile writer. It does not configure the
separate OpenRouter conversation-authoring command. Runtime dialogue rendering
is offline once `[scenarios] conversation_file` is frozen.

| Key | Type/default | Meaning |
| --- | --- | --- |
| `mode` | string / `"replay"` | `"replay"` for retained JSONL, `"openai"` for direct OpenAI Responses API execution, or `"openrouter"` for OpenRouter Chat Completions |
| `responses_file` | string / `""` | Input JSONL in replay mode; required when an `llm_` updater uses replay |
| `calibration` | string / `"temperature"` | `"temperature"` fits development-only LLM probability calibration; `"none"` uses raw vectors |
| `calibration_users` | integer / `1` | Positive number of declared development users used by the fixed calibration probe |
| `model_role` | string / `"primary"` | `primary`, `replication`, or `decoder` default model declaration |
| `model` | string / `""` | OpenAI override, empty to resolve from `model_role`; OpenRouter requires one exact canonical `author/model` slug |
| `reasoning_effort` | string / `""` | Explicit `none`, `low`, `medium`, `high`, `xhigh`, or `max`; OpenRouter also accepts `minimal`, and empty omits its optional reasoning control |
| `api_key_env` | string / `"OPENAI_API_KEY"` | Name of the environment variable read immediately before a live request |
| `base_url` | string / `"https://api.openai.com"` | Provider HTTPS origin/path; official origin required unless the separate opt-in below is true |
| `allow_custom_base_url` | Boolean / `false` | Explicitly permit sending the configured credential to the reviewed non-official HTTPS endpoint |
| `timeout_seconds` | number / `180.0` | Positive timeout for each HTTP attempt |
| `max_retries` | integer / `4` | Nonnegative retry count for failures the selected adapter explicitly admits; OpenRouter transport ambiguity stops immediately |
| `max_output_tokens` | integer / `4096` | Positive per-response output ceiling |
| `max_requests` | integer / `100` | Positive hard request ceiling for one provider ledger |
| `max_total_tokens` | integer / `500000` | Positive hard conservative-token ceiling for one provider ledger |
| `journal_dir` | string / `""` | Optional live recovery-journal root; empty uses `<output-root>/.llm-journals` |
| `openrouter_upstream_provider` | string / `""` | Optional OpenRouter provider slug placed in both `provider.order` and `provider.only`; use a full endpoint-variant slug when region/variant identity matters |
| `openrouter_allow_fallbacks` | Boolean / `false` | Permit another endpoint for the same model after a failed route |
| `openrouter_require_parameters` | Boolean / `true` | Exclude endpoints that do not advertise every requested parameter, including structured output |
| `openrouter_data_collection` | string / `"deny"` | OpenRouter provider-data filter: `"deny"` or `"allow"` |
| `openrouter_zdr` | Boolean / `false` | When true, require a zero-data-retention endpoint |
| `openrouter_http_referer` | string / `""` | Optional absolute HTTP(S) app-attribution URL sent as `HTTP-Referer` |
| `openrouter_app_title` | string / `"CAPE-Loop"` | Optional app-attribution title sent as `X-OpenRouter-Title` |

When `mode = "openrouter"`, omitted mode-specific common fields resolve to
`api_key_env = "OPENROUTER_API_KEY"`,
`base_url = "https://openrouter.ai/api"`, and `max_retries = 2`. The endpoint
constructed from that base path is
`https://openrouter.ai/api/v1/chat/completions`. `model_role` still names the
journal role, but it does not select an OpenRouter model; `model` must be
explicit. The validator rejects aliases beginning with `~`, colon-suffixed
route variants, `-latest` labels, and `openrouter/auto`.

The default role resolution is:

| Role | Model | Reasoning effort |
| --- | --- | --- |
| `primary` | `gpt-5.6-sol` | `medium` |
| `replication` | `gpt-5.6-terra` | `medium` |
| `decoder` | `gpt-5.6-luna` | `low` |

The `replication` role is a GPT-5.6 model-variant/tier replication. It does not
establish robustness to a distinct model family or provider. The
`llm evaluation-suite` command enforces the checked primary/replication
declarations and matched design, then retains each role's separate config hash,
run ID, journal path, and request/token ceilings in one combined index.

For within-model causal comparisons, keep the resolved model and reasoning
effort fixed across `llm_response_only`, `llm_full_context`, and
`llm_provenance_aware`. The two checked-in OpenAI pilot configs do this. A model
override is valid software configuration but must be treated as a reported
protocol change.

All evaluated views receive a semantic codebook that names each domain
attribute and explains the four signed levels. Model-facing option records
contain readable descriptions, not the simulator's numeric feature vectors.
Full-context and provenance-aware views also receive the exact rendered
assistant/user dialogue; neither receives the internal target index.
Response-only deliberately omits the unselected options and assistant turn.

### LLM probability calibration

For any Experiment A, B, or C run with an `llm_` updater and
`calibration = "temperature"`, the runner selects the first
`calibration_users` users from the declared development population and executes
a fixed matched provenance probe. The probe spans balanced, restricted,
ranking, default, and suggested contexts with naturally sampled responses. Runtime
validation fails if `calibration_users` exceeds the available development
population.

Using development labels only, the runner fits one scalar temperature per LLM
updater ID. The response-only, full-context, and provenance-aware views
therefore do not share a fitted calibrator. A calibrated provider wrapper
constructs the corresponding test/runtime diagnostic vectors. It does not
change prompts, information views, model identity, or reasoning effort.

Experiment A is intentionally different from B/C: A's primary updater rows,
gates, exact-oracle residuals, and held-out paraphrase checks consume the raw
returned vector. Its temperature-scaled vector is secondary calibration
diagnostics only. B/C consume the configured active calibrated vector for their
realized histories. This boundary prevents temperature scaling from creating an
apparent A update when the raw response exactly repeats the supplied prior.

The following artifact boundary is always explicit:

- `models/llm-calibration.json` records `per-updater-temperature`, the
  development split, each calibrator, and `test_labels_used = false`; with
  `calibration = "none"` it records `kind = "none"`;
- `llm/development-raw-responses.jsonl` retains uncalibrated development-probe
  model outputs when temperature fitting is active;
- `metrics/llm-development-calibration.jsonl` retains raw and calibrated
  development Brier scores;
- `llm/test-raw-responses.jsonl` retains uncalibrated outputs underlying
  test/runtime updates;
- `llm/test-calibrated-responses.jsonl` retains their temperature-scaled
  counterparts; and
- `llm/responses.jsonl` contains the responses consumed by the primary
  experiment path: raw for A, configured-active for B/C.

`calibration = "none"` skips the development probe and uses raw responses
directly. `calibration_users` remains a positive declared field but is not
consumed in that mode.

### Replay mode

If any updater ID begins with `llm_` and `mode = "replay"`,
`responses_file` must be nonempty. A relative path is resolved from the
invoking process's working directory. With temperature calibration, the corpus
must cover both the development probe and the test/runtime requests. Every
consumed response must match the locally reconstructed `request_id` and
`prompt_sha256`.

Before creating the run directory, the runner parses and fingerprints the
entire configured response file. It retains the configured path, byte SHA-256,
record count, and distinct model IDs in `llm/input-manifest.json`.

### OpenAI mode

With `mode = "openai"`, an experiment containing an `llm_` updater fails
closed unless the CLI also supplies `--execute-live`. The API key itself is
never a configuration value: only the environment-variable **name** is stored.
No credential, Authorization header, or secret is retained in journals or run
artifacts. [`.env.example`](../.env.example) documents the expected name, but
the CLI does not automatically load that file or a local `.env`. Load an
ignored `.env` explicitly into the invoking shell, for example with
`set -a; source .env; set +a`, and never commit it.

With the default `allow_custom_base_url = false`, `base_url` must be exactly the
official `https://api.openai.com` origin (an optional trailing slash is
accepted): a different host, explicit port, or path is rejected. Setting
`allow_custom_base_url = true` permits another reviewed HTTPS origin or path,
to which the executor appends `/v1/responses`. A custom host also requires a
dedicated `api_key_env` name other than `OPENAI_API_KEY`, preventing an
ordinary OpenAI credential from being routed to a proxy by configuration
accident. This opt-in causes authorized live requests to send that dedicated
credential to the reviewed endpoint. Treat it as a credential-routing security
decision and do not enable it for an untrusted proxy or service. The declared
Boolean is retained in the live input manifest.

`max_requests` and `max_total_tokens` are enforced before a request is sent.
The token ledger reserves a deliberately conservative byte-based maximum,
including `max_output_tokens`, and commits provider-reported usage when
available. Resumed requests restore their usage into the same ledger, so a
restart does not reset a ceiling. These are safety bounds, not a currency
budget; review current pricing independently. The same ledger covers the
development calibration probe and all subsequent test/runtime calls.

### Universal live request preflight

Every adaptive configuration containing an `llm_*` updater is passed through
the same credential-free preflight:

```text
A = users × domains × 3 attributes × 2 anchor directions × prior strengths
    × mechanisms × response modes × LLM updater count
    + temperature calibration + eligible held-out paraphrases
B = users × domains × configured initial profiles × repeats × policies × turns
    × LLM updater count + calibration
C = (max(8, users) development users + users test users) × domains
    × repeats × three regimes × turns × LLM updater count + calibration
sensitivity = domains × users × repeats × policies × LLM updater count
              × sum(point trajectory lengths)
```

The logical completion bound is multiplied by `max_retries + 1` to obtain the
maximum physical HTTP attempts. That bound must not exceed `max_requests`, and
`physical_attempts × max_output_tokens` must not by itself exceed
`max_total_tokens`. Configuration validation, suite planning, and live runner
startup all fail closed when either condition is false. Live startup performs
this check before provider construction, credential access, journal creation,
or run-directory creation.

Adaptive input bytes depend on earlier model outputs, so no tool claims an
exact whole-run prompt-token total. The provider ledger reserves and enforces
the cumulative token ceiling before every attempt. Every run with an LLM
updater records the calculation in `llm/request-preflight.json`; sensitivity
also retains its compatibility view in
`llm/sensitivity-request-preflight.json`.

The runner writes successful audit records before replay-compatible response
records. By default the durable journal paths are:

```text
<output-root>/.llm-journals/<run-id>/<model-role>/provider-audit.jsonl
<output-root>/.llm-journals/<run-id>/<model-role>/responses.jsonl
```

On a successful run, used audit records and a credential-free provider
manifest are copied into the checksummed artifact. The external journal remains
available for `--resume-failed-live`. See
[Live execution](live-execution.md#transport-audits-locks-and-recovery) for
planning, static execution, recovery, and decoder commands.

### OpenRouter mode

`mode = "openrouter"` is a first-class gateway mode. It does not reinterpret
OpenRouter as an OpenAI custom base URL. The checked-in example is:

```toml
[llm]
mode = "openrouter"
# Change this model slug. If the replacement is not served by the pinned
# endpoint below, also change or clear openrouter_upstream_provider.
model = "google/gemini-3.6-flash"
api_key_env = "OPENROUTER_API_KEY"
base_url = "https://openrouter.ai/api"
allow_custom_base_url = false

openrouter_upstream_provider = "google-vertex/global"
openrouter_allow_fallbacks = false
openrouter_require_parameters = true
openrouter_data_collection = "deny"
openrouter_zdr = false
openrouter_http_referer = ""
openrouter_app_title = "CAPE-Loop"
```

The standalone OpenRouter CLI defaults to no upstream-provider pin, so one
`--model author/model` argument switches its model. The checked-in adaptive
pilot pins the Google Vertex global endpoint for route reproducibility. When
changing that pilot to a model not served there, also replace the provider
slug or set
`openrouter_upstream_provider = ""`; an unpinned route is accepted only with
its selected upstream identity retained for analysis.

Preparing a request sends `stream = false`, `max_tokens`, and a
strict `response_format.type = "json_schema"`. The provider preferences contain
the declared fallback, parameter-support, data-collection, ZDR, and optional
provider constraints. The transport always sends
`X-OpenRouter-Metadata: enabled` and `X-OpenRouter-Cache: false`; attribution
headers are separate and optional. A successful response must report the exact
requested model, a `direct` routing strategy, exactly one selected upstream
endpoint whose model is the canonical model or one of its dated snapshots, no
disallowed fallback, no cache hit, and no material router pipeline
transformation. A configured endpoint slug is enforced in the request with
both `provider.only` and `provider.order`; OpenRouter returns a display provider
name rather than that exact slug, so the display field is retained but is not
treated as exact-slug attestation. Structured message content is parsed and
validated locally before it can become an `LLMResponse`.

The API key is read from `OPENROUTER_API_KEY` only after `--execute-live` and is
never retained. The official base path is required by default. As with direct
OpenAI execution, a different HTTPS endpoint requires
`allow_custom_base_url = true` and a dedicated credential-variable name; that
choice authorizes sending the dedicated credential to the reviewed endpoint.

OpenRouter journals use:

```text
<output-root>/.llm-journals/<run-id>/openrouter/<model-role>/provider-audit.jsonl
<output-root>/.llm-journals/<run-id>/openrouter/<model-role>/responses.jsonl
<output-root>/.llm-journals/<run-id>/openrouter/<model-role>/transport-attempts.jsonl
```

The checksummed run receives `llm/provider-audit.jsonl` and
`llm/transport-attempts.jsonl` plus `llm/provider-manifest.json`. Audit rows
separate the gateway from the selected upstream route through `gateway`,
`model_requested`, `model_returned`, `upstream_provider`, `upstream_model`,
`routing_strategy`, `routing_attempt`, and the full additive
`routing_metadata`; they also retain the submitted upstream constraint and
provider preferences, their request-constraint evidence label, the explicit
display-identity interpretation boundary, provider response ID,
`X-Generation-Id` when present, cache status, usage, timing, body/prompt
hashes, redacted raw response, and provider-neutral replay response. The
implementation records `first_party_origin_claimed = false` and does not
automatically call OpenRouter's generation-lookup endpoint.

OpenRouter's official documentation defines the
[Chat Completions request](https://openrouter.ai/docs/api/api-reference/chat/send-chat-completion-request),
[structured-output contract](https://openrouter.ai/docs/guides/features/structured-outputs),
[provider preferences](https://openrouter.ai/docs/guides/routing/provider-selection),
[router metadata](https://openrouter.ai/docs/guides/features/router-metadata),
[response-cache controls](https://openrouter.ai/docs/guides/features/response-caching),
and [app-attribution headers](https://openrouter.ai/docs/app-attribution).
Recheck endpoint support and the model catalog before each collection wave.

This gateway mode is valid for profile-writer studies and multi-model decoder
collection, but it is not direct first-party provenance. Multiple models or
reported upstream providers behind the same gateway do not establish
statistically independent errors. Gate 4's selected decoder pair is the exact
`anthropic/claude-sonnet-5` and `google/gemini-3.6-flash` collection, with
Claude `low` and Gemini `minimal` reasoning recorded separately. The decoder
CLI supports repeatable
`--model-reasoning-effort MODEL=EFFORT` overrides for alternate/generic
collections; the selected validator rejects a changed effort until the
versioned protocol is updated. The selected defaults do not use one global
effort for both families.

OpenRouter is also the selected one-time conversation-authoring path. That
command uses its explicit `--model` argument to obtain neutral base wording and
display names. Code expands the frozen bank and writes the readable
`.generation.jsonl`; the command does not use `[llm]`, fit a profile, author
treatments, choose a response, or run once per experimental trial.

Unlike the standalone profile-writer CLI defaults, the decoder-study
`plan-openrouter` and `execute-openrouter` commands with no model/budget
overrides load the selected two-model suite with zero retries, 1,024 output
tokens, 900 attempts, and 6,000,000 conservative tokens per model. Supplying
`--model` switches to an explicit alternate collection; `--additional-model`
then adds more exact slugs.

The complete pair is admitted under
`selected_openrouter_gateway_collection`, with
`first_party_origin_claimed = false` and responsible-researcher review. Direct
Anthropic/Gemini adapters remain an optional origin-replication mode. See
[Live execution](live-execution.md#gate-4-collection-and-admission).

## `[artifacts]`

| Key | Default | Behavior |
| --- | --- | --- |
| `retain_events` | `true` | Write configured full training and experiment audit-event files |
| `retain_prompts` | `false` | Write consumed evaluated-profile-writer requests when LLM replay is used |
| `checksum_manifest` | `true` | Write `SHA256SUMS` on success or captured failure |

Some scientific records are written unconditionally because they are metrics or
required evaluation definitions. `retain_events = false` is therefore not a
promise that every `events/` file is absent; for example, Experiment C retains
its terminal-battery definitions.

When full events are retained, each hybrid observation includes
`assistant_message`, user `surface_response`, `selected_option`, and
`surface_id`. The frozen source bank is copied to
`inputs/conversation-templates.json` regardless of `retain_prompts`.
Conversation-authoring requests remain in the sibling `.generation.jsonl` and
are not controlled by this section.

Full event retention can be the dominant storage cost in multi-turn B/C and
sensitivity runs because complete posterior, shadow, and native state is
repeated for reconstruction. Newly generated A–C runs always write their narrow
runner-native `analysis/*.jsonl` projection from the same evaluated records;
`retain_events` does not control those files. The projection creates no new
synthetic users, interactions, or observations. Use it for routine analysis,
but retain the full events for a release that promises forensic
reconstruction. Sensitivity's existing aggregated metrics and CSV already
serve as its compact projection.

A–C and sensitivity runs also write an exhaustive deduplicated
`conversations/<experiment>.jsonl` and a matching Markdown preview. These
outputs are derived from the same evaluated records and are not controlled by
`retain_events` or `retain_prompts`. The JSONL contains every logical
conversation while grouping evaluations that share it. A run without a
configured natural surface uses nullable dialogue fields and
`surface_available = false`; it does not fabricate text. The Markdown uses a
deterministic diverse selection capped at 100 trace records by default; its
header still reports exact complete record, turn, and outcome counts and
provides readable metric labels and interpretation guidance. The cap limits
only the readable preview, not data collection or the exhaustive JSONL.

The run summary reports
`conversation_log_artifact`, `conversation_log_markdown_artifact`,
`conversation_record_count`, `conversation_turn_count`,
`conversation_outcome_count`, and `conversation_markdown_preview_count`.
These counts describe the complete trace except for the explicitly bounded
Markdown preview count.

Changing `retain_events` changes the resolved configuration and therefore the
content-addressed run ID. It is not a command for stripping an existing
completed run. Completed runs remain immutable; use the separately
checksum-bound `artifact compact` workflow when a small projection of a
historical verified run is needed.

If checksums are disabled, `verify RUN_DIR` reports that `SHA256SUMS` is
missing.

## Python API

```python
from cape_loop import load_config

config = load_config("configs/smoke.toml")
print(config.experiment.kind)
print(config.canonical_json())
```

`AppConfig` and its section objects are frozen dataclasses. Experiment runners
call `config.validated()` before execution, reparsing the full canonical object
through field-level and experiment-contract checks. Custom programmatic configs
therefore cannot bypass the TOML-equivalent validation path.

## What validation does not prove

Configuration validation does not establish:

- adequate power or sample size;
- successful convergence or identifiability in a new setting;
- real held-out paraphrase transfer;
- fitted-aware superiority on a paper test set;
- universal causal-provenance miscalibration or self-confirmation;
- a ranking reversal;
- passage of any stage gate; or
- reproducibility of an external provider call.

For checked-in configurations, the runner consumes the split manifest through
the checksum-bound catalog's atlas/beacon/cedar option, dialogue, and scenario
families and through the content-addressed paraphrase suite. It writes a
concrete binding/overlap audit; see
[Data model](data-model.md#splits-and-leakage-controls). Gate 1 still remains
incomplete whenever its required held-out updater/case pairs are absent.
