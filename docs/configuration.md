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
| `configs/offline/sensitivity.toml` | sensitivity | Baseline-first 19-point OAT robustness design |
| `configs/live/experiment_a_openai.toml` | A | Direct-OpenAI primary pilot |
| `configs/live/experiment_a_openai_replication.toml` | A | Matched GPT-5.6 model-variant replication |
| `configs/live/experiment_a_openrouter.toml` | A | Matched OpenRouter/Gemini pilot |
| `configs/live/experiment_b_openai.toml` | B | Bounded direct-OpenAI pilot |
| `configs/live/experiment_b_openrouter.toml` | B | Matched OpenRouter pilot |
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
| A primary/replication/OpenRouter | 768 | 48 | 32 | 848 | 1,736,704 |
| B pair | 768 | 96 | 0 | 864 | 1,769,472 |
| C pair | 768 | 48 | 0 | 816 | 1,671,168 |
| Gate 6 OAT pair | 576 | 0 | 0 | 576 | 1,179,648 |

The offline Gate 4 and Experiment C external-rescore source configs do not call
models. They are sized so their later selected OpenRouter or optional direct
external-model collections fit the same approved per-source ceilings.

## Root schema

The only accepted root keys are:

```toml
schema_version = 1

[run]
[experiment]
[response_model]
[inference]
[thresholds]
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
make an external model response deterministic.

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

## `[experiment]`

| Key | Type | Default |
| --- | --- | --- |
| `kind` | string | `"provenance_audit"` |
| `domains` | nonempty string array | `["travel", "writing"]` |
| `mechanisms` | nonempty string array | `["balanced", "restricted", "default", "suggested"]` |
| `response_modes` | nonempty string array | `["controlled_anchor", "naturally_sampled"]` |
| `prior_strengths` | nonempty numeric array | `[0.0]` |
| `policies` | nonempty string array | `["balanced"]` |
| `updaters` | nonempty string array | baseline set shown below |
| `users` | integer | `8` |
| `trajectories_per_cell` | integer | `1` |
| `turns` | integer | `1` |
| `bootstrap_replicates` | integer | `0` |

`users`, `trajectories_per_cell`, and `turns` must be positive.
`bootstrap_replicates` must be nonnegative.

For `closed_loop`, the same `bootstrap_replicates` value requests both the
clustered inferential bootstrap and the Experiment B pilot-power simulation
count. Power simulation alone is bounded to the inclusive range 200–10,000:
zero therefore retains the inexpensive 200-replicate planning smoke fallback,
and requests above 10,000 are capped in the power artifact. The candidate user
counts (16, 32, 64, 128), alpha (0.05), target power (0.80), factor contrast,
and lower-Wilson-bound decision rule are frozen in code rather than exposed as
quietly mutable paper settings. The resulting candidate is advisory and never
commits the final sample size automatically.

`prior_strengths` is Experiment A's executable prior-concentration factor.
Every value must be finite, unique, and in `[0, 1)`. For each latent user, a
level `s` mixes `(1-s)` uniform joint mass with `s` mass on that user's true
theta. This truth-aligned construction is balanced by the latent population;
all updaters and provenance mechanisms within a matched stratum receive the
same prior. The default `[0.0]` preserves the original uniform-prior pilot.
Use, for example, `[0.0, 0.35, 0.70]` for a crossed confirmatory run and account
for the proportional increase in LLM requests. B, C, and sensitivity reject
any value other than `[0.0]`.

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
`default`, and `suggested`. Experiments B, C, and sensitivity require the set
`ranking`, `default`, and `suggestion`; their policies produce the actual
presentation context.

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

`minimum_matched_probability` is used by Experiment A's matched-anchor
eligibility check.

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
`[llm]`.

## `[thresholds]`

| Key | Default | Validation/use |
| --- | ---: | --- |
| `materially_wrong_mass` | `0.50` | in `[0, 1]`; self-confirmation assessment |
| `laundered_confidence_gain` | `0.25` | nonnegative; self-confirmation assessment |
| `shadow_equivalence_tolerance` | `0.05` | in `[0, 1]`; maximum terminal wrong-mass gap still treated as equivalent shadow confidence |
| `false_stability_tolerance` | `0.02` | in `[0, 1]`; maximum longitudinal excursion from seeded wrong mass for false-stable classification |
| `direction_tolerance` | `1e-9` | nonnegative; update-direction comparisons |
| `ranking_tie_tolerance` | `1e-6` | nonnegative; Experiment C ranking ties |

These thresholds affect diagnostic labels. Freeze them before analyzing a
paper-intended test set.

## `[sensitivity]`

| Key | Default | Validation |
| --- | --- | --- |
| `design` | `"cartesian"` | `"cartesian"` or `"one_at_a_time"` |
| `decision_noise_values` | `[0.6, 1.0, 1.6]` | nonempty; finite, positive, and unique |
| `presentation_multipliers` | `[0.5, 1.0, 1.5]` | nonempty; finite, nonnegative, and unique |
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
| `phase_min_attribution_cost` | `0.0` | finite; phase attribution-cost threshold |
| `phase_min_self_confirming_rate` | `0.0` | finite value in `[0, 1]` |
| `phase_min_suggestion_rejection_rate` | `0.20` | finite value in `[0, 1]`; frozen “often rejects” criterion |

These fields are consumed only by `kind = "sensitivity"`. Under
`one_at_a_time`, the first value of every numeric axis and the first
response-model family define the baseline; each later value is varied
separately. Alternate response families are evaluated at the numeric baseline,
with every declared rule-noise value for a rule-based alternative.

The canonical `configs/offline/sensitivity.toml` declaration is a 19-point OAT
design. It covers every declared axis efficiently but cannot estimate
interactions among axes; artifacts record
`interaction_effects_estimable = false`. The two live Gate 6 OAT pilots contain
11 points whose trajectory lengths sum to 36. The suggestion rejection rate
counts only profile-conditioned suggestions where the counter-profile option
remains displayed; selecting that alternative is a rejection.

## `[llm]`

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

### LLM probability calibration

For any Experiment A, B, or C run with an `llm_` updater and
`calibration = "temperature"`, the runner selects the first
`calibration_users` users from the declared development population and executes
a fixed matched provenance probe. The probe spans balanced, restricted,
default, and suggested contexts with naturally sampled responses. Runtime
validation fails if `calibration_users` exceeds the available development
population.

Using development labels only, the runner fits one scalar temperature per LLM
updater ID. The response-only, full-context, and provenance-aware views
therefore do not share a fitted calibrator. A calibrated provider wrapper then
transforms all subsequent test/runtime probability vectors. It does not change
prompts, information views, model identity, or reasoning effort.

The following artifact boundary is always explicit:

- `models/llm-calibration.json` records `per-updater-temperature`, the
  development split, each calibrator, and `test_labels_used = false`; with
  `calibration = "none"` it records `kind = "none"`;
- `llm/development-raw-responses.jsonl` retains uncalibrated development-probe
  model outputs when temperature fitting is active;
- `metrics/llm-development-calibration.jsonl` retains raw and calibrated
  development Brier scores;
- `llm/test-raw-responses.jsonl` retains uncalibrated outputs underlying
  calibrated test/runtime updates; and
- `llm/responses.jsonl` contains the active calibrated responses consumed by
  experiment updaters.

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
A = matched experiment cells + temperature calibration + held-out paraphrases
B = users × domains × four initial profiles × repeats × policies × turns
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
| `retain_events` | `true` | Write configured training and experiment event files |
| `retain_prompts` | `false` | Write consumed LLM requests when LLM replay is used |
| `checksum_manifest` | `true` | Write `SHA256SUMS` on success or captured failure |

Some scientific records are written unconditionally because they are metrics or
required evaluation definitions. `retain_events = false` is therefore not a
promise that every `events/` file is absent; for example, Experiment C retains
its terminal-battery definitions.

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
- causal-provenance blindness or self-confirmation;
- a ranking reversal;
- passage of any stage gate; or
- reproducibility of an external provider call.

The runner consumes the split manifest through feature-matched
atlas/beacon/cedar option, dialogue, and scenario families and through the
content-addressed paraphrase suite. It writes a concrete binding/overlap audit;
see [Data model](data-model.md#splits-and-leakage-controls). Gate 1 still
remains incomplete whenever its
required held-out updater/case pairs are absent.
