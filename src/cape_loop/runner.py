"""Validated-config experiment orchestration and artifact materialization."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence
import json
import math

from .artifacts import (
    RunArtifacts,
    config_digest,
    source_tree_digest,
    verify_run,
)
from .beliefs import PreferenceBelief
from .calibration import (
    CalibrationExample,
    TemperatureCalibration,
    fit_temperature,
)
from .config import AppConfig
from .domains import (
    DATA_SPLITS,
    DomainSpec,
    dialogue_template_id,
    domain_for_split,
    get_domain,
    option_template_id,
    scenario_family_id,
)
from .decoder_study import (
    DecoderTruthLabel,
    build_blinded_native_decoder_request,
)
from .experiments import (
    analyze_experiment_b_inference,
    build_terminal_battery,
    evaluate_native_decoders,
    evaluate_terminal_battery,
    run_experiment_b,
    run_experiment_c,
    run_provenance_audit,
    summarize_terminal_calibration,
)
from .experiments.provenance import (
    ExperimentAConfirmatoryResult,
    build_experiment_a_control_battery,
    compare_experiment_a_raw_calibrated,
    experiment_a_mechanism_contrasts,
    experiment_a_updater_mechanism_interaction,
    fit_experiment_a_marginal_ols,
)
from .gates import (
    GateCriterion,
    GateReport,
    gate_1_from_rows,
    gate_2_and_3_from_trajectories,
    incomplete_gate,
)
from .heldout import (
    ParaphraseEvaluationRecord,
    ParaphraseSource,
    TerminalAction,
    build_default_paraphrase_suite,
    build_heldout_terminal_suite,
    evaluate_gate1_paraphrase_transfer,
    generate_paraphrase_cases,
    score_heldout_terminal_actions,
)
from .metrics import marginal_brier, mean_or_nan
from .llm_exchange import (
    CompletionProvider,
    ReplayProvider,
    TemperatureCalibratedProvider,
    read_responses,
)
from .llm_outcomes import (
    CachedTerminalCalibrationOutcome,
    cached_outcome_manifest,
    score_cached_raw_calibrated_terminal,
)
from .native import NativeMemoryState
from .openai_provider import (
    DEFAULT_OPENAI_MODEL_ROLES,
    OpenAIProviderConfig,
    OpenAIResponsesProvider,
    ResumableCompletionProvider,
    ResumableOpenAICompletionProvider,
)
from .openrouter_provider import (
    OpenRouterChatProvider,
    OpenRouterProviderConfig,
    ResumableOpenRouterCompletionProvider,
)
from .policies import build_policy
from .population import (
    generate_users,
    susceptibility_grid,
    user_state_record,
)
from .reporting import grouped_mean, write_csv, write_line_svg
from .response import RandomUtilityModel
from .sensitivity import (
    PhaseCriterion,
    classify_phase_point,
    infer_axis_boundaries,
    response_model_at,
    sensitivity_grid,
)
from .schemas import LatentUser, Observation, THETA_VALUES
from .splits import SplitManifest, build_split_manifest
from .statistics import holm_bonferroni, simulate_paired_cluster_power
from .training import (
    FittedModelBundle,
    fit_model_bundle,
    generate_training_examples,
    held_out_aware_reliability,
    held_out_response_scores,
    temperature_calibrate_model_bundle,
)
from .updaters import (
    ExactActionAwareUpdater,
    FittedActionAwareUpdater,
    LLMReplayUpdater,
    ProfileUpdater,
    build_updater_registry,
    make_update_view,
    updater_views,
)


@dataclass(frozen=True, slots=True)
class PreparedStudy:
    domains: tuple[DomainSpec, ...]
    development_domains: tuple[DomainSpec, ...]
    manifest: SplitManifest
    response_model: RandomUtilityModel
    raw_fitted_models: FittedModelBundle
    fitted_models: FittedModelBundle
    calibration: Mapping[str, Any]
    training_users: tuple[Any, ...]
    development_users: tuple[Any, ...]
    test_users: tuple[Any, ...]
    training_records: tuple[Any, ...]
    development_records: tuple[Any, ...]
    held_out_diagnostics: Mapping[str, Any]
    split_leakage_audit: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class PreparedLLMExecution:
    raw_provider: CompletionProvider | None
    active_provider: CompletionProvider | None
    development_registry: Mapping[str, ProfileUpdater]
    calibrations: Mapping[str, TemperatureCalibration]
    development_metrics: tuple[Mapping[str, Any], ...]


def _response_model(config: AppConfig) -> RandomUtilityModel:
    response = config.response_model
    inverse_noise = 1.0 / response.decision_noise
    return RandomUtilityModel(
        beta=response.beta * inverse_noise,
        ranking_scale=response.rank_scale * inverse_noise,
        default_scale=response.default_scale * inverse_noise,
        suggestion_scale=response.suggestion_scale * inverse_noise,
    )


def _selected_domains(config: AppConfig) -> tuple[DomainSpec, ...]:
    return tuple(
        domain_for_split(get_domain(domain_id), "test")
        for domain_id in config.experiment.domains
    )


def _study_split_manifest(
    config: AppConfig,
    domains: Sequence[DomainSpec],
) -> SplitManifest:
    domain_ids = tuple(domain.domain_id for domain in domains)
    option_by_split = {
        split: tuple(
            option_template_id(domain_id, split)
            for domain_id in domain_ids
        )
        for split in DATA_SPLITS
    }
    dialogue_by_split = {
        split: tuple(
            dialogue_template_id(domain_id, split)
            for domain_id in domain_ids
        )
        for split in DATA_SPLITS
    }
    scenario_by_split = {
        split: tuple(
            scenario_family_id(domain_id, split)
            for domain_id in domain_ids
        )
        for split in DATA_SPLITS
    }
    paraphrase_suite = build_default_paraphrase_suite()
    paraphrase_by_split = {
        split: tuple(
            template.template_id
            for template in paraphrase_suite.for_split(split)
        )
        for split in DATA_SPLITS
    }
    terminal_option = "heldout-terminal-v2-options"
    terminal_dialogue = "heldout-terminal-v2-wording"
    terminal_scenario = "heldout-terminal-v2-scenarios"
    return build_split_manifest(
        seed=config.run.seed,
        susceptibility_levels=config.response_model.susceptibility_levels,
        option_templates=(
            *option_by_split["train"],
            *option_by_split["development"],
            *option_by_split["test"],
            terminal_option,
        ),
        dialogue_templates=(
            *dialogue_by_split["train"],
            *dialogue_by_split["development"],
            *dialogue_by_split["test"],
            terminal_dialogue,
        ),
        scenario_families=(
            *scenario_by_split["train"],
            *scenario_by_split["development"],
            *scenario_by_split["test"],
            terminal_scenario,
        ),
        paraphrase_templates=tuple(
            template.template_id for template in paraphrase_suite.templates
        ),
        train_option_templates=option_by_split["train"],
        development_option_templates=option_by_split["development"],
        test_option_templates=(*option_by_split["test"], terminal_option),
        train_dialogue_templates=dialogue_by_split["train"],
        development_dialogue_templates=dialogue_by_split["development"],
        test_dialogue_templates=(
            *dialogue_by_split["test"],
            terminal_dialogue,
        ),
        train_scenario_families=scenario_by_split["train"],
        development_scenario_families=scenario_by_split["development"],
        test_scenario_families=(
            *scenario_by_split["test"],
            terminal_scenario,
        ),
        train_paraphrase_templates=paraphrase_by_split["train"],
        development_paraphrase_templates=paraphrase_by_split["development"],
        test_paraphrase_templates=paraphrase_by_split["test"],
    )


def _allocate(total: int, groups: int) -> tuple[int, ...]:
    base, remainder = divmod(total, groups)
    return tuple(base + (1 if index < remainder else 0) for index in range(groups))


def _training_record(example: Any) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "theta": list(example.theta),
        "context": example.context.to_dict(),
        "observation": example.observation.to_dict(),
        "weight": example.weight,
    }


def _split_leakage_audit(
    *,
    manifest: SplitManifest,
    training_records: Sequence[Any],
    development_records: Sequence[Any],
    test_domains: Sequence[DomainSpec],
) -> dict[str, Any]:
    """Validate the concrete assets consumed by fitting and evaluation."""

    records_by_split = {
        "train": tuple(training_records),
        "development": tuple(development_records),
    }
    option_ids = {
        split: {
            option.option_id
            for record in records
            for option in record.context.options
        }
        for split, records in records_by_split.items()
    }
    option_ids["test"] = {
        option.option_id
        for domain in test_domains
        for option in (*domain.option_pool, *domain.isolated_options)
    }
    wording_templates = {
        split: {record.context.wording_template for record in records}
        for split, records in records_by_split.items()
    }
    wording_templates["test"] = {
        dialogue_template_id(domain.domain_id, "test")
        for domain in test_domains
    }
    scenario_families = {
        split: {
            scenario_family_id(record.context.domain, split)
            for record in records
        }
        for split, records in records_by_split.items()
    }
    scenario_families["test"] = {
        scenario_family_id(domain.domain_id, "test")
        for domain in test_domains
    }

    def overlaps(groups: Mapping[str, set[str]]) -> dict[str, list[str]]:
        result = {}
        for first_index, first in enumerate(DATA_SPLITS):
            for second in DATA_SPLITS[first_index + 1 :]:
                shared = sorted(groups[first] & groups[second])
                if shared:
                    result[f"{first}:{second}"] = shared
        return result

    overlap_report = {
        "option_ids": overlaps(option_ids),
        "dialogue_templates": overlaps(wording_templates),
        "scenario_families": overlaps(scenario_families),
    }
    if any(overlap_report.values()):
        raise ValueError(
            "concrete train/development/test assets overlap: "
            + json.dumps(overlap_report, sort_keys=True)
        )

    manifest_checks = []
    for split in DATA_SPLITS:
        for domain in test_domains:
            expected = (
                (
                    "option_templates",
                    option_template_id(domain.domain_id, split),
                ),
                (
                    "dialogue_templates",
                    dialogue_template_id(domain.domain_id, split),
                ),
                (
                    "scenario_families",
                    scenario_family_id(domain.domain_id, split),
                ),
            )
            for group_name, template_id in expected:
                observed_split = manifest.group_maps()[group_name].get(
                    template_id
                )
                if observed_split != split:
                    raise ValueError(
                        f"{group_name}[{template_id!r}] is assigned to "
                        f"{observed_split!r}, expected {split!r}"
                    )
                manifest_checks.append(
                    {
                        "group": group_name,
                        "template_id": template_id,
                        "split": split,
                    }
                )

    suite = build_default_paraphrase_suite()
    fitted_surface_templates = (
        *suite.for_split("train"),
        *suite.for_split("development"),
    )
    suite.assert_no_test_leakage(
        fitted_template_ids=(
            template.template_id
            for template in fitted_surface_templates
        ),
        fitted_template_sha256=(
            template.template_sha256
            for template in fitted_surface_templates
        ),
        fitted_surface_patterns=(
            template.pattern for template in fitted_surface_templates
        ),
    )
    return {
        "schema_version": 1,
        "status": "passed",
        "independent_split_axes": [
            "complete_latent_preference_profile",
            "susceptibility_type",
            "option_template",
            "dialogue_template",
            "scenario_family",
            "natural_language_paraphrase_template",
        ],
        "concrete_asset_counts": {
            split: {
                "option_ids": len(option_ids[split]),
                "dialogue_templates": len(wording_templates[split]),
                "scenario_families": len(scenario_families[split]),
                "paraphrase_templates": len(suite.for_split(split)),
            }
            for split in DATA_SPLITS
        },
        "overlaps": overlap_report,
        "manifest_bindings_checked": manifest_checks,
        "paraphrase_suite_sha256": suite.suite_sha256,
        "paraphrase_test_leakage_check": "passed",
    }


def _prepare_study(
    config: AppConfig,
    *,
    response_model: RandomUtilityModel | None = None,
    seed_namespace: int = 0,
) -> PreparedStudy:
    domains = _selected_domains(config)
    training_domains = tuple(
        domain_for_split(get_domain(domain.domain_id), "train")
        for domain in domains
    )
    development_domains = tuple(
        domain_for_split(get_domain(domain.domain_id), "development")
        for domain in domains
    )
    manifest = _study_split_manifest(config, domains)
    training_count = max(24, min(128, config.experiment.users * 4))
    development_count = max(8, config.experiment.users)
    training_users = generate_users(
        domain_id="shared",
        count=training_count,
        split="train",
        manifest=manifest,
        susceptibility_levels=config.response_model.susceptibility_levels,
        seed=config.run.seed,
    )
    development_users = generate_users(
        domain_id="shared",
        count=development_count,
        split="development",
        manifest=manifest,
        susceptibility_levels=config.response_model.susceptibility_levels,
        seed=config.run.seed,
    )
    test_users = generate_users(
        domain_id="shared",
        count=config.experiment.users,
        split="test",
        manifest=manifest,
        susceptibility_levels=config.response_model.susceptibility_levels,
        seed=config.run.seed,
    )
    declared_response = response_model or _response_model(config)
    training_records = []
    counts = _allocate(config.inference.training_interactions, len(domains))
    for domain, count in zip(training_domains, counts):
        training_records.extend(
            generate_training_examples(
                domain,
                training_users,
                declared_response,
                count=count,
                seed=config.run.seed + seed_namespace,
                split="train",
            )
        )
    bundle = fit_model_bundle(
        training_records,
        seed=config.run.seed + seed_namespace,
        fit_steps=config.inference.fit_steps,
        learning_rate=config.inference.learning_rate,
        l2=config.inference.l2,
    )
    held_out = []
    held_out_count = max(16, min(128, config.inference.training_interactions // 4))
    for domain, count in zip(
        development_domains,
        _allocate(held_out_count, len(domains)),
    ):
        held_out.extend(
            generate_training_examples(
                domain,
                development_users,
                declared_response,
                count=count,
                seed=config.run.seed + seed_namespace + 1_000_003,
                split="development",
            )
        )
    raw_diagnostics = held_out_response_scores(bundle, held_out)
    raw_reliability = held_out_aware_reliability(bundle, held_out)
    if config.inference.calibration == "temperature":
        active_bundle, calibration = temperature_calibrate_model_bundle(
            bundle,
            held_out,
        )
    else:
        active_bundle = bundle
        calibration = {
            "schema_version": 1,
            "kind": "none",
            "fitted_splits": [],
            "example_count": 0,
        }
    active_diagnostics = held_out_response_scores(active_bundle, held_out)
    active_reliability = held_out_aware_reliability(
        active_bundle,
        held_out,
    )
    diagnostics = {
        **active_diagnostics,
        **active_reliability,
        "calibration": calibration["kind"],
        "raw_aware_option_nll": raw_diagnostics["aware_option_nll"],
        "raw_aware_option_ece": raw_reliability["aware_option_ece"],
        "raw_aware_reliability_bins": raw_reliability[
            "aware_reliability_bins"
        ],
        "raw_unaware_semantic_nll": raw_diagnostics[
            "unaware_semantic_nll"
        ],
    }
    split_audit = _split_leakage_audit(
        manifest=manifest,
        training_records=training_records,
        development_records=held_out,
        test_domains=domains,
    )
    return PreparedStudy(
        domains=domains,
        development_domains=development_domains,
        manifest=manifest,
        response_model=declared_response,
        raw_fitted_models=bundle,
        fitted_models=active_bundle,
        calibration=calibration,
        training_users=training_users,
        development_users=development_users,
        test_users=test_users,
        training_records=tuple(training_records),
        development_records=tuple(held_out),
        held_out_diagnostics=diagnostics,
        split_leakage_audit=split_audit,
    )


def _registry(
    config: AppConfig,
    prepared: PreparedStudy,
    *,
    completion_provider: CompletionProvider | None = None,
) -> dict[str, ProfileUpdater]:
    support = susceptibility_grid(config.response_model.susceptibility_levels)
    uses_llm = any(
        updater_id.startswith("llm_")
        for updater_id in config.experiment.updaters
    )
    replay_provider: CompletionProvider | None = None
    if uses_llm:
        if completion_provider is not None:
            replay_provider = completion_provider
        elif config.llm.mode == "replay":
            if not config.llm.responses_file:
                raise ValueError(
                    "LLM replay updaters require llm.responses_file"
                )
            replay_provider = ReplayProvider(
                read_responses(config.llm.responses_file)
            )
        else:
            raise ValueError(
                "live LLM updaters require an explicitly authorized "
                "completion provider"
            )
    return build_updater_registry(
        list(config.experiment.updaters),
        response_model=prepared.response_model,
        aware_model=prepared.fitted_models.aware,
        unaware_model=prepared.fitted_models.unaware,
        susceptibilities=support,
        replay_provider=replay_provider,
    )


def _llm_input_manifest(config: AppConfig) -> dict[str, Any] | None:
    """Fingerprint the replay corpus or declared live model configuration."""

    if not any(
        updater_id.startswith("llm_")
        for updater_id in config.experiment.updaters
    ):
        return None
    if config.llm.mode == "openai":
        role = DEFAULT_OPENAI_MODEL_ROLES[config.llm.model_role]
        return {
            "schema_version": 1,
            "mode": "openai",
            "provider": "openai",
            "model_role": config.llm.model_role,
            "model": config.llm.model or role.model,
            "reasoning_effort": (
                config.llm.reasoning_effort or role.reasoning_effort
            ),
            "calibration": config.llm.calibration,
            "calibration_users": config.llm.calibration_users,
            "api_key_env": config.llm.api_key_env,
            "base_url": config.llm.base_url,
            "allow_custom_base_url": (
                config.llm.allow_custom_base_url
            ),
            "max_output_tokens": config.llm.max_output_tokens,
            "max_requests": config.llm.max_requests,
            "max_total_tokens": config.llm.max_total_tokens,
            "credential_retained": False,
        }
    if config.llm.mode == "openrouter":
        provider = OpenRouterProviderConfig(
            model=config.llm.model,
            reasoning_effort=config.llm.reasoning_effort,
            api_key_env=config.llm.api_key_env,
            base_url=config.llm.base_url,
            allow_custom_base_url=config.llm.allow_custom_base_url,
            upstream_provider=(
                config.llm.openrouter_upstream_provider
            ),
            allow_fallbacks=(
                config.llm.openrouter_allow_fallbacks
            ),
            require_parameters=(
                config.llm.openrouter_require_parameters
            ),
            data_collection=(
                config.llm.openrouter_data_collection
            ),
            zdr=config.llm.openrouter_zdr,
            http_referer=config.llm.openrouter_http_referer,
            app_title=config.llm.openrouter_app_title,
            timeout_seconds=config.llm.timeout_seconds,
            max_retries=config.llm.max_retries,
            max_output_tokens=config.llm.max_output_tokens,
            max_requests=config.llm.max_requests,
            max_total_tokens=config.llm.max_total_tokens,
            live_execution=False,
        )
        return {
            "schema_version": 1,
            "mode": "openrouter",
            "provider": "openrouter",
            "gateway": "openrouter",
            "model_role": config.llm.model_role,
            "model": provider.model,
            "reasoning_effort": provider.reasoning_effort or None,
            "calibration": config.llm.calibration,
            "calibration_users": config.llm.calibration_users,
            "api_key_env": provider.api_key_env,
            "base_url": provider.base_url,
            "endpoint": provider.endpoint,
            "allow_custom_base_url": provider.allow_custom_base_url,
            "upstream_provider_constraint": (
                provider.upstream_provider or None
            ),
            "provider_preferences": provider.provider_preferences(),
            "response_cache_enabled": False,
            "router_metadata_requested": True,
            "router_transforms_accepted": False,
            "max_output_tokens": provider.max_output_tokens,
            "max_requests": provider.max_requests,
            "request_budget_unit": "physical_http_attempt",
            "max_retries_per_logical_request": provider.max_retries,
            "max_total_tokens": provider.max_total_tokens,
            "credential_retained": False,
            "first_party_origin_claimed": False,
        }
    response_path = Path(config.llm.responses_file)
    try:
        payload = response_path.read_bytes()
    except OSError as exc:
        raise ValueError(
            f"cannot read LLM replay responses {response_path}: {exc}"
        ) from exc
    responses = read_responses(response_path)
    # Constructing a provider also enforces unique request IDs before any
    # artifact directory is created.
    ReplayProvider(responses)
    return {
        "schema_version": 1,
        "mode": "replay",
        "calibration": config.llm.calibration,
        "calibration_users": config.llm.calibration_users,
        "configured_path": config.llm.responses_file,
        "sha256": sha256(payload).hexdigest(),
        "response_count": len(responses),
        "models": sorted({response.model_id for response in responses}),
    }


def _live_completion_provider(
    config: AppConfig,
    *,
    destination: Path,
    execute_live: bool,
) -> ResumableCompletionProvider | None:
    """Build the adaptive provider only for an explicitly authorized run."""

    uses_llm = any(
        updater_id.startswith("llm_")
        for updater_id in config.experiment.updaters
    )
    if not uses_llm or config.llm.mode not in {"openai", "openrouter"}:
        return None
    if not execute_live:
        raise ValueError(
            f"this configuration requests live {config.llm.mode} execution; "
            "rerun with --execute-live only after reviewing the model, "
            "route, and hard budgets"
        )
    if config.llm.mode == "openai":
        role = DEFAULT_OPENAI_MODEL_ROLES[config.llm.model_role]
        provider = OpenAIResponsesProvider(
            OpenAIProviderConfig(
                model=config.llm.model or role.model,
                reasoning_effort=(
                    config.llm.reasoning_effort or role.reasoning_effort
                ),
                api_key_env=config.llm.api_key_env,
                base_url=config.llm.base_url,
                allow_custom_base_url=config.llm.allow_custom_base_url,
                timeout_seconds=config.llm.timeout_seconds,
                max_retries=config.llm.max_retries,
                max_output_tokens=config.llm.max_output_tokens,
                max_requests=config.llm.max_requests,
                max_total_tokens=config.llm.max_total_tokens,
                live_execution=True,
            )
        )
    else:
        provider = OpenRouterChatProvider(
            OpenRouterProviderConfig(
                model=config.llm.model,
                reasoning_effort=config.llm.reasoning_effort,
                api_key_env=config.llm.api_key_env,
                base_url=config.llm.base_url,
                allow_custom_base_url=config.llm.allow_custom_base_url,
                upstream_provider=(
                    config.llm.openrouter_upstream_provider
                ),
                allow_fallbacks=(
                    config.llm.openrouter_allow_fallbacks
                ),
                require_parameters=(
                    config.llm.openrouter_require_parameters
                ),
                data_collection=(
                    config.llm.openrouter_data_collection
                ),
                zdr=config.llm.openrouter_zdr,
                http_referer=config.llm.openrouter_http_referer,
                app_title=config.llm.openrouter_app_title,
                timeout_seconds=config.llm.timeout_seconds,
                max_retries=config.llm.max_retries,
                max_output_tokens=config.llm.max_output_tokens,
                max_requests=config.llm.max_requests,
                max_total_tokens=config.llm.max_total_tokens,
                live_execution=True,
            )
        )
    journal_root = (
        Path(config.llm.journal_dir)
        if config.llm.journal_dir
        else destination.parent / ".llm-journals"
    )
    journal = journal_root / destination.name
    if config.llm.mode == "openrouter":
        journal = journal / "openrouter"
    journal = journal / config.llm.model_role
    adapter_type = (
        ResumableOpenAICompletionProvider
        if config.llm.mode == "openai"
        else ResumableOpenRouterCompletionProvider
    )
    return adapter_type(
        provider,
        responses_path=journal / "responses.jsonl",
        audit_path=journal / "provider-audit.jsonl",
    )


def _calibrated_belief(
    belief: PreferenceBelief,
    calibration: TemperatureCalibration,
) -> PreferenceBelief:
    marginals = belief.marginals()
    rows = tuple(
        calibration.apply(marginals.marginal(attribute))
        for attribute in range(3)
    )
    from .beliefs import MarginalPreferenceBelief

    return PreferenceBelief.from_marginals(
        MarginalPreferenceBelief(rows)  # type: ignore[arg-type]
    )


def _prepare_llm_execution(
    config: AppConfig,
    prepared: PreparedStudy,
    *,
    raw_provider: CompletionProvider | None,
) -> PreparedLLMExecution:
    """Fit declared LLM probability calibration on development users only."""

    llm_updater_ids = tuple(
        updater_id
        for updater_id in config.experiment.updaters
        if updater_id.startswith("llm_")
    )
    if not llm_updater_ids:
        return PreparedLLMExecution(None, None, {}, {}, ())
    if raw_provider is None:
        raise ValueError("LLM execution requires a raw completion provider")
    if config.llm.calibration == "none":
        return PreparedLLMExecution(
            raw_provider,
            raw_provider,
            {},
            {},
            (),
        )

    development_registry = build_updater_registry(
        list(llm_updater_ids),
        response_model=prepared.response_model,
        aware_model=prepared.fitted_models.aware,
        unaware_model=prepared.fitted_models.unaware,
        susceptibilities=susceptibility_grid(
            config.response_model.susceptibility_levels
        ),
        replay_provider=raw_provider,
    )
    calibration_users = prepared.development_users[
        : config.llm.calibration_users
    ]
    if len(calibration_users) < config.llm.calibration_users:
        raise ValueError(
            "llm.calibration_users exceeds the development population"
        )
    development = run_provenance_audit(
        users=calibration_users,
        domains=prepared.development_domains,
        updaters=development_registry,
        response_model=prepared.response_model,
        fitted_aware_model=prepared.fitted_models.aware,
        mechanisms=("balanced", "restricted", "default", "suggested"),
        response_modes=("naturally_sampled",),
        minimum_probability=config.response_model.minimum_matched_probability,
        direction_tolerance=config.thresholds.direction_tolerance,
        seed=config.run.seed,
        data_split="development",
    )
    truth_by_user = {user.user_id: user.theta for user in calibration_users}
    examples: dict[str, list[CalibrationExample]] = {
        updater_id: [] for updater_id in llm_updater_ids
    }
    for row in development.rows:
        truth = truth_by_user[row.user_id]
        marginals = row.posterior.marginals()
        for attribute in range(3):
            examples[row.updater_id].append(
                CalibrationExample(
                    probabilities=marginals.marginal(attribute),
                    true_index=THETA_VALUES.index(truth[attribute]),
                    split="development",
                )
            )
    calibrations = {
        updater_id: fit_temperature(updater_examples)
        for updater_id, updater_examples in examples.items()
    }
    metrics = []
    for row in development.rows:
        calibration = calibrations[row.updater_id]
        calibrated = _calibrated_belief(row.posterior, calibration)
        truth = truth_by_user[row.user_id]
        metrics.append(
            {
                "schema_version": 1,
                "split": "development",
                "trial_id": row.trial_id,
                "user_id": row.user_id,
                "domain_id": row.domain_id,
                "mechanism": row.mechanism,
                "updater_id": row.updater_id,
                "raw_brier": marginal_brier(row.posterior, truth),
                "calibrated_brier": marginal_brier(calibrated, truth),
            }
        )
    active_provider = TemperatureCalibratedProvider(
        raw_provider,
        calibrations,
    )
    return PreparedLLMExecution(
        raw_provider=raw_provider,
        active_provider=active_provider,
        development_registry=development_registry,
        calibrations=calibrations,
        development_metrics=tuple(metrics),
    )


def _write_llm_calibration(
    run: RunArtifacts,
    execution: PreparedLLMExecution,
) -> None:
    if execution.raw_provider is None:
        return
    run.write_json(
        "models/llm-calibration.json",
        {
            "schema_version": 1,
            "kind": (
                "none"
                if not execution.calibrations
                else "per-updater-temperature"
            ),
            "fitted_split": (
                None if not execution.calibrations else "development"
            ),
            "calibrators": {
                updater_id: calibration.to_dict()
                for updater_id, calibration in sorted(
                    execution.calibrations.items()
                )
            },
            "test_labels_used": False,
        },
    )
    if not execution.development_registry:
        return
    adapters = tuple(
        updater
        for updater in execution.development_registry.values()
        if isinstance(updater, LLMReplayUpdater)
    )
    development_requests = tuple(
        request for updater in adapters for request in updater.requests
    )
    development_responses = tuple(
        response for updater in adapters for response in updater.responses
    )
    if run.config.artifacts.retain_prompts:
        run.write_jsonl(
            "llm/development-requests.jsonl",
            (request.to_dict() for request in development_requests),
        )
    run.write_jsonl(
        "llm/development-raw-responses.jsonl",
        (response.to_dict() for response in development_responses),
    )
    run.write_jsonl(
        "metrics/llm-development-calibration.jsonl",
        execution.development_metrics,
    )


def _write_prepared(run: RunArtifacts, prepared: PreparedStudy) -> None:
    run.write_json("splits.json", prepared.manifest.to_dict())
    run.write_json(
        "metrics/split-leakage-audit.json",
        dict(prepared.split_leakage_audit),
    )
    run.write_json("models/fitted-likelihoods.json", prepared.fitted_models.to_dict())
    run.write_json(
        "models/raw-fitted-likelihoods.json",
        prepared.raw_fitted_models.to_dict(),
    )
    run.write_json("models/calibration.json", dict(prepared.calibration))
    run.write_json(
        "models/held-out-response-diagnostics.json",
        dict(prepared.held_out_diagnostics),
    )
    population_rows = []
    for split, users in (
        ("train", prepared.training_users),
        ("development", prepared.development_users),
        ("test", prepared.test_users),
    ):
        for domain in prepared.domains:
            population_rows.extend(
                user_state_record(user, domain_id=domain.domain_id, split=split)
                for user in users
            )
    run.write_jsonl("population/users.jsonl", population_rows)
    if run.config.artifacts.retain_events:
        run.write_jsonl(
            "events/fitted-model-training.jsonl",
            (_training_record(example) for example in prepared.training_records),
        )
        run.write_jsonl(
            "events/fitted-model-development.jsonl",
            (
                _training_record(example)
                for example in prepared.development_records
            ),
        )


def _write_llm_exchange(
    run: RunArtifacts,
    registry: Mapping[str, ProfileUpdater],
    *,
    additional_registries: Sequence[Mapping[str, ProfileUpdater]] = (),
    live_provider: ResumableCompletionProvider | None = None,
    calibrated_provider: TemperatureCalibratedProvider | None = None,
) -> None:
    adapters = tuple(
        updater
        for current_registry in (registry, *additional_registries)
        for updater in current_registry.values()
        if isinstance(updater, LLMReplayUpdater)
    )
    if not adapters:
        return
    raw_requests = tuple(
        request for adapter in adapters for request in adapter.requests
    )
    raw_responses = tuple(
        response for adapter in adapters for response in adapter.responses
    )
    request_by_id = {}
    for request in raw_requests:
        existing = request_by_id.get(request.request_id)
        if existing is not None and existing != request:
            raise ValueError(
                "duplicate LLM request ID has different prompt material"
            )
        request_by_id[request.request_id] = request
    response_by_id = {}
    for response in raw_responses:
        existing = response_by_id.get(response.request_id)
        if existing is not None and existing != response:
            raise ValueError(
                "duplicate LLM response ID has different response material"
            )
        response_by_id[response.request_id] = response
    requests = tuple(request_by_id.values())
    responses = tuple(response_by_id.values())
    if run.config.artifacts.retain_prompts:
        run.write_jsonl(
            "llm/requests.jsonl",
            (request.to_dict() for request in requests),
        )
    run.write_jsonl(
        "llm/responses.jsonl",
        (response.to_dict() for response in responses),
    )
    run.write_json(
        "llm/exchange-manifest.json",
        {
            "schema_version": 1,
            "prompts_retained": run.config.artifacts.retain_prompts,
            "requests": [
                {
                    "request_id": request.request_id,
                    "updater_id": request.updater_id,
                    "view": request.view,
                    "prompt_sha256": request.prompt_sha256,
                }
                for request in requests
            ],
            "models": sorted(
                {response.model_id for response in responses}
            ),
            "execution_mode": run.config.llm.mode,
            "probability_calibration": run.config.llm.calibration,
        },
    )
    if calibrated_provider is not None:
        run.write_jsonl(
            "llm/test-raw-responses.jsonl",
            (
                response.to_dict()
                for response in calibrated_provider.raw_responses
            ),
        )
    if live_provider is not None:
        run.write_jsonl(
            "llm/provider-audit.jsonl",
            live_provider.used_audit_records,
        )
        provider_manifest = live_provider.to_manifest()
        provider_manifest.pop("responses_journal", None)
        provider_manifest.pop("audit_journal", None)
        provider_manifest["external_recovery_journal_retained"] = True
        provider_manifest["credentials_retained"] = False
        run.write_json("llm/provider-manifest.json", provider_manifest)


def _all_gates(primary: GateReport, *additional: GateReport) -> dict[str, Any]:
    reports = {report.gate_id: report for report in (primary,) + additional}
    titles = {
        1: "Learnable provenance gap",
        2: "Nontrivial soft self-confirmation",
        3: "Attribution beyond evidence selection",
        4: "Native-system validity",
        5: "Evaluation implication",
        6: "Robustness",
    }
    for gate_id, title in titles.items():
        key = f"gate-{gate_id}"
        reports.setdefault(
            key,
            incomplete_gate(
                gate_id,
                title,
                "This experiment configuration does not evaluate the gate.",
            ),
        )
    return {
        "schema_version": 1,
        "claim_status": "not_claimed",
        "gates": [
            reports[f"gate-{gate_id}"].to_dict()
            for gate_id in range(1, 7)
        ],
    }


def _experiment_a_power_differences(
    rows: Sequence[Any],
    *,
    target_updater_id: str,
    reference_updater_id: str = "fitted_action_aware",
) -> tuple[float, ...]:
    """Reduce the primary updater×mechanism interaction to complete users."""

    selected = tuple(
        row for row in rows if row.response_mode == "controlled_anchor"
    )
    cells = {
        (
            row.user_id,
            row.domain_id,
            row.target_attribute,
            row.anchor_direction,
            row.updater_id,
            row.mechanism,
        ): row
        for row in selected
    }
    mechanisms = sorted(
        {
            row.mechanism
            for row in selected
            if row.mechanism != "balanced"
        }
    )
    by_user: dict[str, list[float]] = {}
    base_cells = sorted(
        {
            (
                row.user_id,
                row.domain_id,
                row.target_attribute,
                row.anchor_direction,
            )
            for row in selected
        }
    )
    for user_id, domain_id, attribute, direction in base_cells:
        prefix = (user_id, domain_id, attribute, direction)
        for mechanism in mechanisms:
            required = (
                (*prefix, target_updater_id, mechanism),
                (*prefix, target_updater_id, "balanced"),
                (*prefix, reference_updater_id, mechanism),
                (*prefix, reference_updater_id, "balanced"),
            )
            if not all(key in cells for key in required):
                continue
            target_difference = (
                cells[required[0]].acue - cells[required[1]].acue
            )
            reference_difference = (
                cells[required[2]].acue - cells[required[3]].acue
            )
            by_user.setdefault(user_id, []).append(
                target_difference - reference_difference
            )
    return tuple(
        mean(by_user[user_id])
        for user_id in sorted(by_user)
        if by_user[user_id]
    )


def _visible_context_payload(row: Any) -> dict[str, Any]:
    """Return the same non-audit context fields visible to LLM updaters."""

    context = row.context
    return {
        "domain": context.domain,
        "options": [option.to_dict() for option in context.options],
        "ranking": list(context.ranking),
        "default": context.default_option_id,
        "suggested_option": context.suggested_option_id,
        "wording_template": context.wording_template,
        "question_type": context.question_type,
        "target_attribute": context.target_attribute,
    }


def _heldout_paraphrase_evaluation(
    *,
    rows: Sequence[Any],
    registry: Mapping[str, ProfileUpdater],
    prepared: PreparedStudy,
    config: AppConfig,
) -> tuple[Any, tuple[Any, ...], tuple[ParaphraseEvaluationRecord, ...], Any]:
    """Run the fixed test-split surface families through bound updater views."""

    suite = build_default_paraphrase_suite()
    unique_trials: dict[str, Any] = {}
    for row in sorted(
        rows,
        key=lambda item: (
            item.domain_id,
            item.mechanism,
            item.trial_id,
            item.updater_id,
        ),
    ):
        if row.response_mode != "naturally_sampled":
            continue
        unique_trials.setdefault(row.trial_id, row)

    selected_rows: list[Any] = []
    cell_counts: dict[tuple[str, str], int] = {}
    for row in unique_trials.values():
        cell = (row.domain_id, row.mechanism)
        if cell_counts.get(cell, 0) >= 2:
            continue
        cell_counts[cell] = cell_counts.get(cell, 0) + 1
        selected_rows.append(row)

    ordinals = ("first", "second", "third", "fourth", "fifth")
    sources: list[ParaphraseSource] = []
    source_rows: dict[str, Any] = {}
    for row in selected_rows:
        selected = row.context.option(row.selected_option_id)
        displayed_index = next(
            index
            for index, option in enumerate(row.context.options)
            if option.option_id == row.selected_option_id
        )
        source = ParaphraseSource.build(
            source_trial_id=row.trial_id,
            domain_id=row.domain_id,
            mechanism=row.mechanism,
            selected_option_id=row.selected_option_id,
            selected_label=selected.label or selected.option_id,
            selected_ordinal=(
                ordinals[displayed_index]
                if displayed_index < len(ordinals)
                else f"option {displayed_index + 1}"
            ),
            visible_context=_visible_context_payload(row),
        )
        sources.append(source)
        source_rows[source.source_trial_id] = row
    cases = (
        generate_paraphrase_cases(sources, suite, split="test")
        if sources
        else ()
    )

    aware = FittedActionAwareUpdater(prepared.fitted_models.aware)
    evaluated_updaters: list[ProfileUpdater] = [aware]
    if "llm_full_context" in registry:
        evaluated_updaters.append(registry["llm_full_context"])
    theta_by_user = {user.user_id: user.theta for user in prepared.test_users}
    records: list[ParaphraseEvaluationRecord] = []
    for case in cases:
        source_row = source_rows[case.source_trial_id]
        observation = Observation(
            selected_option_id=source_row.observation.selected_option_id,
            surface_response=case.surface_response,
            choice_noise_key=source_row.observation.choice_noise_key,
        )
        for updater in evaluated_updaters:
            state = updater.initial_state(source_row.prior)
            view = make_update_view(
                updater.view_kind,
                source_row.context,
                observation,
                source_row.provenance,
                event_id=f"{case.case_id}:{updater.updater_id}",
            )
            posterior = updater.update(state, view).state.belief
            records.append(
                ParaphraseEvaluationRecord.from_case(
                    case,
                    updater_id=updater.updater_id,
                    brier=marginal_brier(
                        posterior,
                        theta_by_user[source_row.user_id],
                    ),
                    belief_payload=posterior.to_dict(),
                )
            )
    nonbalanced = {
        mechanism
        for mechanism in config.experiment.mechanisms
        if mechanism != "balanced"
    }
    criterion = evaluate_gate1_paraphrase_transfer(
        cases,
        records,
        suite=suite,
        required_mechanisms=max(1, min(2, len(nonbalanced))),
        required_domains=config.experiment.domains,
    )
    return suite, tuple(cases), tuple(records), criterion


def _run_a(
    config: AppConfig,
    run: RunArtifacts,
    prepared: PreparedStudy,
    *,
    completion_provider: CompletionProvider | None = None,
    raw_completion_provider: CompletionProvider | None = None,
    live_provider: ResumableCompletionProvider | None = None,
    calibrated_provider: TemperatureCalibratedProvider | None = None,
) -> dict[str, Any]:
    registry = _registry(
        config,
        prepared,
        completion_provider=completion_provider,
    )
    result = run_provenance_audit(
        users=prepared.test_users,
        domains=prepared.domains,
        updaters=registry,
        response_model=prepared.response_model,
        fitted_aware_model=prepared.fitted_models.aware,
        prior_strengths=config.experiment.prior_strengths,
        mechanisms=config.experiment.mechanisms,
        response_modes=config.experiment.response_modes,
        minimum_probability=config.response_model.minimum_matched_probability,
        direction_tolerance=config.thresholds.direction_tolerance,
        seed=config.run.seed,
    )
    raw_prepared = PreparedStudy(
        domains=prepared.domains,
        development_domains=prepared.development_domains,
        manifest=prepared.manifest,
        response_model=prepared.response_model,
        raw_fitted_models=prepared.raw_fitted_models,
        fitted_models=prepared.raw_fitted_models,
        calibration={
            "schema_version": 1,
            "kind": "none",
            "fitted_splits": [],
            "example_count": 0,
        },
        training_users=prepared.training_users,
        development_users=prepared.development_users,
        test_users=prepared.test_users,
        training_records=prepared.training_records,
        development_records=prepared.development_records,
        held_out_diagnostics=prepared.held_out_diagnostics,
        split_leakage_audit=prepared.split_leakage_audit,
    )
    raw_registry = _registry(
        config,
        raw_prepared,
        completion_provider=(
            raw_completion_provider or completion_provider
        ),
    )
    raw_result = run_provenance_audit(
        users=prepared.test_users,
        domains=prepared.domains,
        updaters=raw_registry,
        response_model=prepared.response_model,
        fitted_aware_model=prepared.raw_fitted_models.aware,
        prior_strengths=config.experiment.prior_strengths,
        mechanisms=config.experiment.mechanisms,
        response_modes=config.experiment.response_modes,
        minimum_probability=config.response_model.minimum_matched_probability,
        direction_tolerance=config.thresholds.direction_tolerance,
        seed=config.run.seed,
    )
    if config.artifacts.retain_events:
        run.write_jsonl(
            "events/experiment-a.jsonl",
            (
                {
                    "schema_version": 1,
                    **row.to_dict(include_joint_states=False),
                }
                for row in result.rows
            ),
        )
        exact_references: dict[str, dict[str, Any]] = {}
        for row in result.rows:
            if row.trial_id not in exact_references:
                exact_references[row.trial_id] = {
                    "schema_version": 1,
                    "exact_reference_id": row.trial_id,
                    "exact_posterior": row.exact_posterior.to_dict(),
                    "exact_theta_psi": (
                        None
                        if row.exact_theta_psi is None
                        else row.exact_theta_psi.to_dict()
                    ),
                }
        run.write_jsonl(
            "events/experiment-a-exact-references.jsonl",
            exact_references.values(),
        )
        run.write_jsonl(
            "events/experiment-a-exclusions.jsonl",
            (
                {"schema_version": 1, **item.to_dict()}
                for item in result.excluded
            ),
        )
    analysis_replicates = (
        config.experiment.bootstrap_replicates
        if config.experiment.bootstrap_replicates > 0
        else 200
    )
    raw_calibrated = compare_experiment_a_raw_calibrated(
        raw_result.rows,
        result.rows,
        true_theta_by_user={
            user.user_id: user.theta for user in prepared.test_users
        },
    )
    oracle_slopes = result.oracle_update_slopes(
        replicates=analysis_replicates,
        seed=config.run.seed,
    )
    evidence_strength = result.evidence_strength_analysis()
    control_battery = build_experiment_a_control_battery()
    run.write_json(
        "models/experiment-a-control-battery.json",
        control_battery.to_dict(),
    )
    mechanism_contrasts = tuple(
        contrast
        for mechanism in config.experiment.mechanisms
        if mechanism != "balanced"
        for contrast in experiment_a_mechanism_contrasts(
            result.rows,
            first_mechanism=mechanism,
            second_mechanism="balanced",
            metric="acue",
            replicates=analysis_replicates,
            seed=config.run.seed,
        )
    ) if "balanced" in config.experiment.mechanisms else ()
    primary_profile_writer = next(
        (
            updater_id
            for updater_id in (
                "llm_full_context",
                "full_context_blind",
            )
            if updater_id in registry
        ),
        None,
    )
    interaction_rows = []
    if (
        primary_profile_writer is not None
        and "fitted_action_aware" in registry
        and "balanced" in config.experiment.mechanisms
    ):
        for mechanism in config.experiment.mechanisms:
            if mechanism == "balanced":
                continue
            try:
                interaction_rows.append(
                    experiment_a_updater_mechanism_interaction(
                        result.rows,
                        first_updater=primary_profile_writer,
                        second_updater="fitted_action_aware",
                        treated_mechanism=mechanism,
                        reference_mechanism="balanced",
                        metric="acue",
                        replicates=analysis_replicates,
                        seed=config.run.seed,
                    )
                )
            except ValueError:
                continue
    analysis_notes = [
        (
            "The dependency-free regression is a marginal OLS robustness "
            "analysis with user-clustered CR1 covariance, not the proposal's "
            "user-random-slope/scenario-random-intercept mixed-effects model."
        ),
        (
            "The fixed positive/negative control battery is a protocol "
            "artifact only. Its outcomes require the declared dedicated "
            "executors and were not invented from one-step anchor rows."
        ),
        (
            f"Bootstrap analyses used {analysis_replicates} replicates; a "
            "zero configured value selects the documented smoke fallback of 200."
        ),
    ]
    try:
        marginal_regression = fit_experiment_a_marginal_ols(
            result.rows,
            outcome="acue",
            response_mode="naturally_sampled",
        )
    except ValueError as exc:
        marginal_regression = None
        analysis_notes.append(
            f"Marginal regression unavailable for this configuration: {exc}"
        )
    confirmatory = ExperimentAConfirmatoryResult(
        oracle_update_slopes=oracle_slopes,
        evidence_strength=evidence_strength,
        mechanism_contrasts=mechanism_contrasts,
        updater_mechanism_interactions=tuple(interaction_rows),
        marginal_regression=marginal_regression,
        raw_calibrated_comparison=raw_calibrated,
        bootstrap_replicates=analysis_replicates,
        notes=tuple(analysis_notes),
    )
    run.write_json(
        "metrics/experiment-a-confirmatory.json",
        confirmatory.to_dict(),
    )
    run.write_jsonl(
        "metrics/experiment-a-oracle-slopes.jsonl",
        (row.to_dict() for row in oracle_slopes),
    )
    run.write_json(
        "metrics/experiment-a-evidence-strength.json",
        evidence_strength.to_dict(),
    )
    run.write_jsonl(
        "metrics/experiment-a-raw-calibrated-scores.jsonl",
        (row.to_dict() for row in raw_calibrated.scores),
    )
    run.write_jsonl(
        "metrics/experiment-a-reliability.jsonl",
        (row.to_dict() for row in raw_calibrated.reliability_bins),
    )
    write_csv(
        run.path / "tables/experiment-a-raw-calibrated-scores.csv",
        [row.to_dict() for row in raw_calibrated.scores],
    )
    write_csv(
        run.path / "tables/experiment-a-reliability.csv",
        [row.to_dict() for row in raw_calibrated.reliability_bins],
    )
    if marginal_regression is not None:
        p_values = {
            coefficient.name: coefficient.p_value
            for coefficient in marginal_regression.coefficients
            if coefficient.name != "intercept"
            and coefficient.p_value is not None
        }
    else:
        p_values = {}
    multiplicity = (
        holm_bonferroni(p_values).to_dict()
        if p_values
        else {
            "status": "not_estimable",
            "reason": "no non-intercept CR1 coefficient p-values",
        }
    )
    run.write_json(
        "metrics/experiment-a-multiplicity.json",
        {
            "schema_version": 1,
            "family": "Experiment A CR1 non-intercept coefficients",
            "result": multiplicity,
        },
    )
    power_differences = (
        _experiment_a_power_differences(
            result.rows,
            target_updater_id=primary_profile_writer,
        )
        if primary_profile_writer is not None
        and "fitted_action_aware" in registry
        else ()
    )
    if len(power_differences) >= 2:
        power_payload: dict[str, Any] = {
            "schema_version": 1,
            "status": "estimated_from_configured_pilot",
            **simulate_paired_cluster_power(
                power_differences,
                (16, 32, 64, 128),
                estimand=(
                    "Experiment A profile-writer versus fitted-aware "
                    "updater-by-mechanism ACUE interaction"
                ),
                simulations=max(200, analysis_replicates),
                seed=config.run.seed,
            ).to_dict(),
        }
    else:
        power_payload = {
            "schema_version": 1,
            "status": "not_estimable",
            "reason": (
                "power simulation requires at least two complete independent "
                "user-level pilot interaction differences"
            ),
            "pilot_cluster_count": len(power_differences),
        }
    run.write_json("metrics/experiment-a-power.json", power_payload)
    metric_rows = [
        {
            "trial_id": row.trial_id,
            "user_id": row.user_id,
            "domain": row.domain_id,
            "target_attribute": row.target_attribute,
            "anchor_direction": row.anchor_direction,
            "prior_stratum": row.prior_stratum,
            "prior_strength": row.prior_strength,
            "mechanism": row.mechanism,
            "response_mode": row.response_mode,
            "updater_id": row.updater_id,
            "brier": row.brier,
            "fitted_aware_brier": row.fitted_aware_brier,
            "excess_brier": row.excess_brier,
            "acue": row.acue,
            "marginal_kl": row.fitted_aware_kl,
            "update_direction_accuracy": row.update_direction_accuracy,
            "update_direction_evaluated_components": (
                row.update_direction_evaluated_components
            ),
            "update_direction_excluded_components": (
                row.update_direction_excluded_components
            ),
            "update_magnitude": row.update_magnitude,
            "evidence_weight": row.evidence_weight,
            "log_odds_update": row.log_odds_update,
            "fitted_aware_log_odds_update": (
                row.fitted_aware_log_odds_update
            ),
            "fitted_evidence_strength": row.fitted_evidence_strength,
        }
        for row in result.rows
    ]
    run.write_jsonl("metrics/experiment-a.jsonl", metric_rows)
    aggregate = grouped_mean(
        metric_rows,
        by=(
            "response_mode",
            "domain",
            "prior_stratum",
            "prior_strength",
            "mechanism",
            "updater_id",
        ),
        metric="brier",
    )
    write_csv(run.path / "tables/experiment-a-brier.csv", aggregate)
    controlled = [
        row for row in metric_rows if row["response_mode"] == "controlled_anchor"
    ]
    plot_rows = grouped_mean(
        controlled,
        by=("updater_id", "mechanism"),
        metric="update_magnitude",
    )
    mechanism_x = {
        mechanism: index
        for index, mechanism in enumerate(config.experiment.mechanisms)
    }
    series: dict[str, list[tuple[float, float]]] = {}
    for row in plot_rows:
        series.setdefault(str(row["updater_id"]), []).append(
            (
                float(mechanism_x[str(row["mechanism"])]),
                float(row["update_magnitude"]),
            )
        )
    if series:
        write_line_svg(
            run.path / "figures/experiment-a-update-magnitude.svg",
            series,
            title="Controlled identical-response update magnitude",
            x_label="Provenance mechanism index (see table)",
            y_label="Mean marginal L1 update",
        )
    (
        paraphrase_suite,
        paraphrase_cases,
        paraphrase_records,
        paraphrase_criterion,
    ) = _heldout_paraphrase_evaluation(
        rows=result.rows,
        registry=registry,
        prepared=prepared,
        config=config,
    )
    run.write_json(
        "models/held-out-paraphrase-suite.json",
        paraphrase_suite.to_dict(),
    )
    if config.artifacts.retain_events:
        run.write_jsonl(
            "events/experiment-a-held-out-paraphrases.jsonl",
            (case.to_dict() for case in paraphrase_cases),
        )
    run.write_jsonl(
        "metrics/experiment-a-held-out-paraphrase-scores.jsonl",
        (record.to_dict() for record in paraphrase_records),
    )
    run.write_json(
        "metrics/experiment-a-held-out-paraphrase-transfer.json",
        paraphrase_criterion.to_dict(),
    )
    gate_1 = gate_1_from_rows(
        metric_rows,
        full_context_updater_id="llm_full_context",
        held_out_paraphrase_verified=paraphrase_criterion.verified,
    )
    _write_llm_exchange(
        run,
        registry,
        live_provider=live_provider,
        calibrated_provider=calibrated_provider,
    )
    gate_report = _all_gates(gate_1)
    run.write_json("metrics/gate-report.json", gate_report)
    summary = {
        "experiment": "A",
        "scientific_claim_status": "not_claimed",
        "result_scope": "implementation_smoke" if "smoke" in config.run.name else "configured_run",
        **result.summary(),
        "updater_views": updater_views(registry),
        "held_out_response_diagnostics": dict(prepared.held_out_diagnostics),
        "oracle_update_slope_rows": len(oracle_slopes),
        "evidence_strength_volunteered_control_status": (
            evidence_strength.volunteered_control_status
        ),
        "prior_strengths": list(config.experiment.prior_strengths),
        "control_battery_id": control_battery.battery_id,
        "control_battery_status": (
            "fixed_protocol_not_scored_by_one_step_choice_runner"
        ),
        "raw_calibrated_forecasts": len(raw_calibrated.scores),
        "marginal_cr1_model_available": marginal_regression is not None,
        "mixed_effects_model_status": "external_confirmatory_stage_required",
        "power_analysis_status": power_payload["status"],
        "held_out_paraphrase_cases": len(paraphrase_cases),
        "held_out_paraphrase_complete": paraphrase_criterion.complete,
        "held_out_paraphrase_verified": paraphrase_criterion.verified,
        "gate_1_computed_status": gate_1.computed_status,
    }
    return summary


def _counter_profile_available(trajectory: Any) -> bool:
    for interaction in trajectory.audit_record.interactions:
        target = interaction.context.target_attribute
        if target is None:
            continue
        directions = {
            1 if option.features[target] > 0 else -1
            for option in interaction.context.options
            if option.features[target] != 0
        }
        if directions != {-1, 1}:
            return False
    return True


def _observed_profile_mechanisms(trajectory: Any) -> list[str]:
    """Recover visible profile-aligned channels without policy-labeled text."""

    mechanisms: set[str] = set()
    for interaction in trajectory.audit_record.interactions:
        provenance = interaction.provenance
        if provenance.profile_conditioned and (
            provenance.presentation_mechanism
            in {"ranking", "default", "suggestion", "restriction"}
        ):
            mechanisms.add(provenance.presentation_mechanism)
    return sorted(mechanisms)


def _gate_4_for_b(
    result: Any,
    deterministic_decoder_rows: Sequence[Mapping[str, Any]],
    terminal_rows: Sequence[Mapping[str, Any]],
    heldout_action_rows: Sequence[Mapping[str, Any]],
    *,
    events_retained: bool,
    external_decoder_evidence: Mapping[str, Any] | None = None,
) -> GateReport:
    native = {"episodic_memory", "semantic_memory", "provenance_linked_memory"}
    eligible_native = tuple(
        trajectory
        for trajectory in result.trajectories
        if trajectory.updater_id in native
        and trajectory.policy_id == "soft_profile_conditioned"
        and trajectory.initial_profile_condition == "incorrect"
        and _counter_profile_available(trajectory)
    )
    native_trajectory_ids = {
        trajectory.trajectory_id for trajectory in eligible_native
    }
    assessment_by_key = {
        (assessment.trajectory_id, assessment.attribute): assessment
        for assessment in result.self_confirmation_assessments
    }
    trajectory_by_crn_updater = {
        (trajectory.crn_key, trajectory.updater_id): trajectory
        for trajectory in eligible_native
    }
    matched_failure_cases = []
    for trajectory in eligible_native:
        # Semantic and provenance-linked memory share the same consolidation
        # strength and differ in causal-provenance access/discounting. Episodic
        # memory is retained as a practical system but is not a matched causal
        # control because its transition strength differs.
        if trajectory.updater_id != "semantic_memory":
            continue
        control = trajectory_by_crn_updater.get(
            (trajectory.crn_key, "provenance_linked_memory")
        )
        if control is None:
            continue
        for attribute in range(3):
            candidate = assessment_by_key.get(
                (trajectory.trajectory_id, attribute)
            )
            control_assessment = assessment_by_key.get(
                (control.trajectory_id, attribute)
            )
            if (
                candidate is not None
                and control_assessment is not None
                and candidate.reportable
                and (
                    not control_assessment.reportable
                    or control_assessment.evidence.cumulative_lcg
                    < candidate.evidence.cumulative_lcg
                )
            ):
                matched_failure_cases.append(
                    {
                        "blind_trajectory_id": trajectory.trajectory_id,
                        "control_trajectory_id": control.trajectory_id,
                        "attribute": attribute,
                    }
                )
    proxy_decoder_trajectory_ids = {
        str(row["trajectory_id"]) for row in deterministic_decoder_rows
    }
    proxy_decoder_counts = {
        trajectory_id: sum(
            row["trajectory_id"] == trajectory_id
            for row in deterministic_decoder_rows
        )
        for trajectory_id in native_trajectory_ids
    }
    required_native_keys = {
        "memory_kind",
        "base_belief",
        "episodes",
        "claims",
        "persona_belief",
        "persona_text",
        "state_id",
    }
    complete_state_trajectory_ids = set()
    for trajectory in result.trajectories:
        if trajectory.trajectory_id not in native_trajectory_ids:
            continue
        state = trajectory.terminal_opaque_state
        to_dict = getattr(state, "to_dict", None)
        payload = to_dict() if callable(to_dict) else None
        if (
            isinstance(payload, Mapping)
            and required_native_keys <= set(payload)
        ):
            complete_state_trajectory_ids.add(trajectory.trajectory_id)
    projected_terminal_trajectory_ids = {
        str(row["trajectory_id"]) for row in terminal_rows
    }
    external_evidence = (
        None
        if external_decoder_evidence is None
        else dict(external_decoder_evidence)
    )
    externally_decoded_trajectory_ids = (
        set()
        if external_evidence is None
        else {
            str(item)
            for item in external_evidence.get(
                "eligible_trajectory_ids",
                (),
            )
        }
    )
    external_decoder_ready = (
        None
        if external_evidence is None
        else (
            external_evidence.get("import_status") == "import_validated"
            and external_evidence.get("complete_coverage") is True
            and external_evidence.get("source_design_eligible") is True
            and external_evidence.get("blind_to_system_identity") is True
            and external_evidence.get("blind_to_latent_truth") is True
            and external_evidence.get("independent_source_reviewed") is True
            and native_trajectory_ids <= externally_decoded_trajectory_ids
        )
    )
    genuine_native_action_rows = tuple(
        row
        for row in heldout_action_rows
        if row.get("adapter_kind") == "native_end_to_end_recorded"
        and row.get("evidence_origin") == "imported_native_system"
    )
    genuine_action_trajectory_ids = {
        str(row["trajectory_id"])
        for row in genuine_native_action_rows
        if "trajectory_id" in row
    }
    genuine_action_ready = (
        (
            native_trajectory_ids <= genuine_action_trajectory_ids
            and all(
                row.get("suite_binding_validated") is True
                and row.get("action_execution_mode")
                in {"recorded_live", "recorded_replay"}
                and isinstance(row.get("native_state_id"), str)
                and bool(str(row.get("native_state_id", "")).strip())
                for row in genuine_native_action_rows
            )
        )
        if genuine_native_action_rows and native_trajectory_ids
        else None
    )
    return GateReport(
        gate_id="gate-4",
        title="Native-system validity",
        criteria=(
            GateCriterion(
                "native-loop-present",
                "At least one inspectable native memory-action loop was evaluated.",
                bool(native_trajectory_ids),
                {"native_trajectories": len(native_trajectory_ids)},
                "native_trajectories > 0",
            ),
            GateCriterion(
                "native-state-retained",
                "Complete terminal native states and transition events are retained.",
                (
                    events_retained
                    and complete_state_trajectory_ids
                    == native_trajectory_ids
                    if native_trajectory_ids
                    else None
                ),
                {
                    "complete_terminal_state_trajectory_ids": sorted(
                        complete_state_trajectory_ids
                    ),
                    "events_retained": events_retained,
                },
                "complete retained state and events for every native trajectory",
            ),
            GateCriterion(
                "independent-blinded-decoder-judgments",
                (
                    "Imported blinded judgments from at least two genuinely "
                    "distinct, independently reviewed decoder sources cover "
                    "every eligible native trajectory."
                ),
                external_decoder_ready if native_trajectory_ids else None,
                {
                    "external_evidence_imported": (
                        external_evidence is not None
                    ),
                    "externally_decoded_trajectory_ids": sorted(
                        externally_decoded_trajectory_ids
                    ),
                    "deterministic_proxy_decoder_trajectory_ids": sorted(
                        proxy_decoder_trajectory_ids
                    ),
                    "deterministic_proxy_decoder_counts": (
                        proxy_decoder_counts
                    ),
                    "deterministic_projections_count_as_external": False,
                },
                (
                    "validated imported coverage; blind to system/truth; "
                    "source-design eligible; genuinely distinct-source review; "
                    "deterministic native projections are ineligible"
                ),
            ),
            GateCriterion(
                "native-end-to-end-terminal-actions",
                (
                    "The native system itself produces recorded end-to-end "
                    "actions on the hash-bound held-out terminal suite."
                ),
                genuine_action_ready,
                {
                    "genuine_native_action_trajectory_ids": sorted(
                        genuine_action_trajectory_ids
                    ),
                    "projected_terminal_trajectory_ids": sorted(
                        projected_terminal_trajectory_ids
                    ),
                    "reference_action_rows": sum(
                        str(row.get("adapter_kind", "")).endswith("_reference")
                        for row in heldout_action_rows
                    ),
                    "persona_or_structured_references_count_as_native_actions": (
                        False
                    ),
                },
                (
                    "imported native_end_to_end_recorded actions with exact "
                    "suite binding and recorded live/replay execution for "
                    "every eligible native trajectory"
                ),
            ),
            GateCriterion(
                "native-failure-observed",
                (
                    "A provenance-blind native soft loop has a five-clause "
                    "failure relative to a matched provenance-linked control."
                ),
                bool(matched_failure_cases)
                if native_trajectory_ids
                else None,
                {"matched_native_failure_cases": matched_failure_cases},
                "at least one matched blind-versus-provenance-linked case",
            ),
        ),
    )


def _heldout_actions_from_belief(
    belief: PreferenceBelief,
    suite: Any,
) -> tuple[TerminalAction, ...]:
    """Reference action interface for structured or native persona beliefs."""

    expected = belief.expected_theta()
    actions: list[TerminalAction] = []
    for item in suite.items:
        if item.question_type == "direct_preference_probe":
            assert item.target_attribute is not None
            actions.append(
                TerminalAction(
                    item_id=item.item_id,
                    item_sha256=item.item_sha256,
                    wording_template_id=item.wording_template_id,
                    question_type=item.question_type,
                    declared_direction=(
                        1 if expected[item.target_attribute] >= 0.0 else -1
                    ),
                )
            )
            continue
        selected = max(
            item.options,
            key=lambda option: (
                sum(
                    coefficient * feature
                    for coefficient, feature in zip(
                        expected,
                        option.features,
                    )
                ),
                option.option_id,
            ),
        )
        actions.append(
            TerminalAction(
                item_id=item.item_id,
                item_sha256=item.item_sha256,
                wording_template_id=item.wording_template_id,
                question_type=item.question_type,
                selected_option_id=selected.option_id,
            )
        )
    return tuple(actions)


def _external_decoder_material(
    trajectories: Sequence[Any],
    *,
    evaluation_split: str,
) -> tuple[tuple[Any, ...], tuple[DecoderTruthLabel, ...], tuple[dict[str, Any], ...]]:
    """Create blinded requests, separately retained truth, and a codebook."""

    requests = []
    labels = []
    codebook = []
    for trajectory in trajectories:
        state = trajectory.terminal_opaque_state
        if not isinstance(state, NativeMemoryState):
            continue
        nonce = sha256(
            (
                "cape-loop-decoder-assignment-v1\n"
                + trajectory.trajectory_id
            ).encode("utf-8")
        ).hexdigest()
        request = build_blinded_native_decoder_request(
            state,
            evaluation_split=evaluation_split,
            assignment_nonce=nonce,
        )
        requests.append(request)
        labels.append(
            DecoderTruthLabel(
                pseudonymous_state_id=request.pseudonymous_state_id,
                theta=trajectory.theta,
                evaluation_split=evaluation_split,
            )
        )
        codebook.append(
            {
                "schema_version": 1,
                "request_id": request.request_id,
                "pseudonymous_state_id": request.pseudonymous_state_id,
                "trajectory_id": trajectory.trajectory_id,
                "updater_id": trajectory.updater_id,
                "domain_id": trajectory.domain_id,
                "evaluation_split": evaluation_split,
                "native_state_id": state.state_id,
            }
        )
    return tuple(requests), tuple(labels), tuple(codebook)


def _write_cached_calibration_outcomes(
    run: RunArtifacts,
    *,
    experiment: str,
    rows: Sequence[CachedTerminalCalibrationOutcome],
    calibration_configured: bool,
) -> None:
    """Write paired cached-vector outcomes and their estimand boundary."""

    material = tuple(rows)
    stem = f"experiment-{experiment.lower()}-llm-raw-calibrated-terminal"
    run.write_jsonl(
        f"metrics/{stem}.jsonl",
        (row.to_dict() for row in material),
    )
    write_csv(
        run.path / f"tables/{stem}.csv",
        [row.to_dict() for row in material],
    )
    manifest = cached_outcome_manifest(experiment, material)
    if not material:
        manifest["status"] = (
            "not_applicable_no_temperature_calibration"
            if not calibration_configured
            else "not_applicable_no_llm_terminal_rows"
        )
    run.write_json(
        f"metrics/{stem}-manifest.json",
        manifest,
    )


def _run_b(
    config: AppConfig,
    run: RunArtifacts,
    prepared: PreparedStudy,
    *,
    completion_provider: CompletionProvider | None = None,
    live_provider: ResumableCompletionProvider | None = None,
    calibrated_provider: TemperatureCalibratedProvider | None = None,
) -> dict[str, Any]:
    registry = _registry(
        config,
        prepared,
        completion_provider=completion_provider,
    )
    policies = {
        policy_id: build_policy(policy_id)
        for policy_id in config.experiment.policies
    }
    shadow = ExactActionAwareUpdater(
        prepared.response_model,
        susceptibility_grid(config.response_model.susceptibility_levels),
    )
    result = run_experiment_b(
        users=prepared.test_users,
        domains=prepared.domains,
        updaters=registry,
        policies=policies,
        turns=config.experiment.turns,
        trajectories_per_cell=config.experiment.trajectories_per_cell,
        response_model=prepared.response_model,
        shadow_updater=shadow,
        seed=config.run.seed,
        materially_wrong_mass=config.thresholds.materially_wrong_mass,
        lcg_threshold=config.thresholds.laundered_confidence_gain,
        shadow_equivalence_tolerance=(
            config.thresholds.shadow_equivalence_tolerance
        ),
        false_stability_tolerance=(
            config.thresholds.false_stability_tolerance
        ),
        direction_tolerance=config.thresholds.direction_tolerance,
    )
    b_inference = analyze_experiment_b_inference(
        result,
        bootstrap_replicates=config.experiment.bootstrap_replicates,
        seed=config.run.seed,
    )
    native_updater_ids = tuple(
        updater_id
        for updater_id in config.experiment.updaters
        if updater_id
        in {
            "episodic_memory",
            "semantic_memory",
            "provenance_linked_memory",
        }
    )
    development_native_trajectories: tuple[Any, ...] = ()
    if native_updater_ids:
        development_registry = build_updater_registry(
            list(native_updater_ids),
            response_model=prepared.response_model,
            aware_model=prepared.fitted_models.aware,
            unaware_model=prepared.fitted_models.unaware,
            susceptibilities=susceptibility_grid(
                config.response_model.susceptibility_levels
            ),
        )
        development_result = run_experiment_b(
            users=prepared.development_users,
            domains=prepared.domains,
            updaters=development_registry,
            policies=policies,
            turns=config.experiment.turns,
            trajectories_per_cell=config.experiment.trajectories_per_cell,
            response_model=prepared.response_model,
            shadow_updater=shadow,
            seed=config.run.seed,
            materially_wrong_mass=config.thresholds.materially_wrong_mass,
            lcg_threshold=config.thresholds.laundered_confidence_gain,
            shadow_equivalence_tolerance=(
                config.thresholds.shadow_equivalence_tolerance
            ),
            false_stability_tolerance=(
                config.thresholds.false_stability_tolerance
            ),
            direction_tolerance=config.thresholds.direction_tolerance,
        )
        development_native_trajectories = (
            development_result.trajectories
        )
    development_requests, development_labels, development_codebook = (
        _external_decoder_material(
            development_native_trajectories,
            evaluation_split="development",
        )
    )
    test_requests, test_labels, test_codebook = _external_decoder_material(
        result.trajectories,
        evaluation_split="test",
    )
    decoder_requests = development_requests + test_requests
    decoder_labels = development_labels + test_labels
    decoder_codebook = development_codebook + test_codebook
    run.write_jsonl(
        "decoder/external-requests.jsonl",
        (request.to_dict() for request in decoder_requests),
    )
    run.write_jsonl(
        "decoder/truth-labels.researcher-only.jsonl",
        (label.to_dict() for label in decoder_labels),
    )
    run.write_jsonl(
        "decoder/researcher-codebook.jsonl",
        decoder_codebook,
    )
    run.write_json(
        "decoder/design-manifest.json",
        {
            "schema_version": 1,
            "status": (
                "ready_for_external_judgments"
                if decoder_requests
                else "not_applicable_no_native_updaters"
            ),
            "request_count": len(decoder_requests),
            "development_request_count": len(development_requests),
            "test_request_count": len(test_requests),
            "minimum_external_sources_per_request": 2,
            "distinct_decoder_families_required": True,
            "requests_blind_to_system_identity_and_truth": True,
            "truth_file_must_not_be_shared_with_decoders": (
                "decoder/truth-labels.researcher-only.jsonl"
            ),
            "independence_claimed": False,
        },
    )
    batteries = {
        domain.domain_id: build_terminal_battery(domain)
        for domain in prepared.domains
    }
    heldout_suites = {
        domain.domain_id: build_heldout_terminal_suite(domain)
        for domain in prepared.domains
    }
    training_option_ids = {
        option.option_id
        for example in prepared.training_records
        for option in example.context.options
    }
    training_feature_vectors = {
        tuple(option.features)
        for example in prepared.training_records
        for option in example.context.options
    }
    training_wording_ids = {
        example.context.wording_template
        for example in prepared.training_records
    }
    training_scenario_ids = {
        example.context.scenario_id
        for example in prepared.training_records
        if example.context.scenario_id
    }
    for suite in heldout_suites.values():
        suite.assert_genuinely_held_out(
            training_option_ids=training_option_ids,
            training_feature_vectors=training_feature_vectors,
            training_wording_template_ids=training_wording_ids,
            training_scenario_family_ids=training_scenario_ids,
        )
    terminal_rows: list[dict[str, Any]] = []
    native_decoder_rows: list[dict[str, Any]] = []
    heldout_action_rows: list[dict[str, Any]] = []
    cached_calibration_rows: list[
        CachedTerminalCalibrationOutcome
    ] = []
    cached_raw_responses = (
        {
            response.request_id: response
            for response in calibrated_provider.raw_responses
        }
        if calibrated_provider is not None
        else {}
    )
    cached_calibrated_responses = (
        {
            response.request_id: response
            for response in calibrated_provider.calibrated_responses
        }
        if calibrated_provider is not None
        else {}
    )
    terminal_calibration_groups: dict[
        tuple[str, str, str | None], list[Any]
    ] = {}
    assessments_by_trajectory: dict[str, list[Any]] = {}
    for assessment in result.self_confirmation_assessments:
        assessments_by_trajectory.setdefault(
            assessment.trajectory_id,
            [],
        ).append(assessment)
    for trajectory in result.trajectories:
        user = LatentUser(
            trajectory.user_id,
            trajectory.theta,
            trajectory.susceptibility,
        )
        battery = batteries[trajectory.domain_id]
        score = evaluate_terminal_battery(
            trajectory.terminal_belief,
            user,
            battery,
        )
        trajectory_assessments = assessments_by_trajectory.get(
            trajectory.trajectory_id,
            [],
        )
        terminal_rows.append(
            {
                "schema_version": 1,
                "trajectory_id": trajectory.trajectory_id,
                "domain_id": trajectory.domain_id,
                "updater_id": trajectory.updater_id,
                "battery_id": battery.battery_id,
                "battery_digest": battery.battery_digest,
                "terminal_shadow_to_system_marginal_kl": (
                    trajectory.terminal_shadow_to_system_marginal_kl
                ),
                "preference_dimension_coverage": (
                    trajectory.preference_dimension_coverage
                ),
                "turns_to_full_preference_coverage": (
                    trajectory.turns_to_full_preference_coverage
                ),
                "displayed_option_diversity": (
                    trajectory.displayed_option_diversity
                ),
                "selected_option_count": trajectory.selected_option_count,
                "profile_conditioned_exposure_rate": (
                    trajectory.profile_conditioned_exposure_rate
                ),
                "presentation_mechanism_count": (
                    trajectory.presentation_mechanism_count
                ),
                "presentation_mechanism_evenness": (
                    trajectory.presentation_mechanism_evenness
                ),
                "cumulative_action_aware_information_gain": (
                    trajectory.cumulative_information_gain
                ),
                "total_intrinsic_regret": trajectory.total_regret,
                "false_stable_attribute_rate": (
                    sum(
                        assessment.false_stable
                        for assessment in trajectory_assessments
                    )
                    / len(trajectory_assessments)
                    if trajectory_assessments
                    else None
                ),
                "false_stable_profile": (
                    any(
                        assessment.false_stable
                        for assessment in trajectory_assessments
                    )
                    if trajectory_assessments
                    else None
                ),
                **score.to_dict(),
            }
        )
        if (
            calibrated_provider is not None
            and trajectory.updater_id.startswith("llm_")
        ):
            cached_calibration_rows.extend(
                score_cached_raw_calibrated_terminal(
                    experiment="B",
                    pairing_id=trajectory.trajectory_id,
                    split="test",
                    regime=(
                        f"closed_loop/{trajectory.policy_id}/"
                        f"{trajectory.initial_profile_condition}"
                    ),
                    updater_id=trajectory.updater_id,
                    active_terminal_belief=trajectory.terminal_belief,
                    audit_record=trajectory.audit_record,
                    user=user,
                    battery=battery,
                    raw_responses=cached_raw_responses,
                    calibrated_responses=cached_calibrated_responses,
                )
            )
        terminal_calibration_groups.setdefault(
            ("system_projection", trajectory.updater_id, None),
            [],
        ).append(score)
        heldout_suite = heldout_suites[trajectory.domain_id]
        profile_action_score = score_heldout_terminal_actions(
            heldout_suite,
            _heldout_actions_from_belief(
                trajectory.terminal_belief,
                heldout_suite,
            ),
            trajectory.theta,
        )
        heldout_action_rows.append(
            {
                "schema_version": 1,
                "trajectory_id": trajectory.trajectory_id,
                "domain_id": trajectory.domain_id,
                "updater_id": trajectory.updater_id,
                "adapter_kind": "structured_profile_action_reference",
                "suite_id": heldout_suite.suite_id,
                "suite_sha256": heldout_suite.suite_sha256,
                **profile_action_score.to_dict(),
            }
        )
        if isinstance(trajectory.terminal_opaque_state, NativeMemoryState):
            native_action_score = score_heldout_terminal_actions(
                heldout_suite,
                _heldout_actions_from_belief(
                    trajectory.terminal_opaque_state.policy_belief,
                    heldout_suite,
                ),
                trajectory.theta,
            )
            heldout_action_rows.append(
                {
                    "schema_version": 1,
                    "trajectory_id": trajectory.trajectory_id,
                    "domain_id": trajectory.domain_id,
                    "updater_id": trajectory.updater_id,
                    "adapter_kind": "native_persona_action_reference",
                    "native_state_id": (
                        trajectory.terminal_opaque_state.state_id
                    ),
                    "suite_id": heldout_suite.suite_id,
                    "suite_sha256": heldout_suite.suite_sha256,
                    **native_action_score.to_dict(),
                }
            )
        native_evaluations = evaluate_native_decoders(
            trajectory.terminal_opaque_state,
            user,
            battery,
        )
        for evaluation in native_evaluations:
            terminal_calibration_groups.setdefault(
                (
                    "deterministic_native_projection",
                    trajectory.updater_id,
                    evaluation.decoder_id,
                ),
                [],
            ).append(evaluation.score)
        native_decoder_rows.extend(
            {
                "schema_version": 1,
                "trajectory_id": trajectory.trajectory_id,
                "domain_id": trajectory.domain_id,
                "updater_id": trajectory.updater_id,
                "battery_id": battery.battery_id,
                "battery_digest": battery.battery_digest,
                **evaluation.to_dict(),
            }
            for evaluation in native_evaluations
        )
    run.write_jsonl(
        "metrics/experiment-b-terminal.jsonl",
        terminal_rows,
    )
    run.write_jsonl(
        "metrics/experiment-b-native-decoders.jsonl",
        native_decoder_rows,
    )
    run.write_jsonl(
        "metrics/experiment-b-held-out-actions.jsonl",
        heldout_action_rows,
    )
    _write_cached_calibration_outcomes(
        run,
        experiment="B",
        rows=cached_calibration_rows,
        calibration_configured=calibrated_provider is not None,
    )
    run.write_json(
        "metrics/experiment-b-terminal-calibration.json",
        {
            "schema_version": 1,
            "experiment": "B",
            "grouping_fields": [
                "score_projection",
                "updater_id",
                "decoder_id",
            ],
            "groups": [
                {
                    "score_projection": score_projection,
                    "updater_id": updater_id,
                    "decoder_id": decoder_id,
                    **summarize_terminal_calibration(scores),
                }
                for (
                    score_projection,
                    updater_id,
                    decoder_id,
                ), scores in sorted(
                    terminal_calibration_groups.items(),
                    key=lambda item: (
                        item[0][0],
                        item[0][1],
                        "" if item[0][2] is None else item[0][2],
                    ),
                )
            ],
        },
    )
    if config.artifacts.retain_events:
        run.write_jsonl(
            "events/experiment-b-terminal-batteries.jsonl",
            (
                {"schema_version": 1, **battery.to_dict()}
                for battery in batteries.values()
            ),
        )
        run.write_jsonl(
            "events/experiment-b-held-out-terminal-suites.jsonl",
            (
                {"schema_version": 1, **suite.to_dict()}
                for suite in heldout_suites.values()
            ),
        )
    if config.artifacts.retain_events:
        run.write_jsonl(
            "events/experiment-b-trajectories.jsonl",
            (
                {"schema_version": 1, **trajectory.to_dict(include_truth=True)}
                for trajectory in result.trajectories
            ),
        )
    run.write_jsonl(
        "metrics/experiment-b-decomposition.jsonl",
        (
            {"schema_version": 1, **row.to_dict()}
            for row in result.decompositions
        ),
    )
    run.write_jsonl(
        "metrics/experiment-b-self-confirmation.jsonl",
        (
            {"schema_version": 1, **assessment.to_dict()}
            for assessment in result.self_confirmation_assessments
        ),
    )
    run.write_json(
        "metrics/experiment-b-inference.json",
        b_inference.to_dict(),
    )
    decomposition_by_profile = {
        row.profile_trajectory_id: row for row in result.decompositions
    }
    assessment_by_trajectory: dict[str, list[Any]] = {}
    for assessment in result.self_confirmation_assessments:
        assessment_by_trajectory.setdefault(assessment.trajectory_id, []).append(
            assessment
        )
    gate_rows = []
    for trajectory in result.trajectories:
        assessments = assessment_by_trajectory.get(trajectory.trajectory_id, [])
        decomposition = decomposition_by_profile.get(trajectory.trajectory_id)
        gate_rows.append(
            {
                "policy_id": trajectory.policy_id,
                "initial_profile": trajectory.initial_profile_condition,
                "updater_id": trajectory.updater_id,
                "mechanisms": _observed_profile_mechanisms(trajectory),
                "counter_profile_available": _counter_profile_available(trajectory),
                "cumulative_lcg": max(
                    (item.evidence.cumulative_lcg for item in assessments),
                    default=0.0,
                ),
                "profile_changed_later_action": any(
                    item.evidence.profile_changed_later_action
                    for item in assessments
                ),
                "is_self_confirming": any(item.reportable for item in assessments),
                "attribution_cost": (
                    trajectory.terminal_error - trajectory.terminal_shadow_error
                    if decomposition is not None
                    else None
                ),
            }
        )
    # The central paper condition is the ordinary full-context writer.
    # Response-only and provenance-aware variants remain controls and must not
    # be pooled into the gate decision.
    llm_targets = (
        ("llm_full_context",)
        if "llm_full_context" in registry
        else ()
    )
    gate_inference = (
        b_inference.gate_evidence(llm_targets[0])
        if llm_targets
        else None
    )
    gate_2, gate_3 = gate_2_and_3_from_trajectories(
        gate_rows,
        target_updater_ids=llm_targets,
        inferential_evidence=gate_inference,
    )
    gate_4 = _gate_4_for_b(
        result,
        native_decoder_rows,
        terminal_rows,
        heldout_action_rows,
        events_retained=config.artifacts.retain_events,
    )
    gate_report = _all_gates(
        incomplete_gate(1, "Learnable provenance gap", "Run Experiment A."),
        gate_2,
        gate_3,
        gate_4,
    )
    run.write_json("metrics/gate-report.json", gate_report)
    table_rows = [row.to_dict() for row in result.decompositions]
    write_csv(run.path / "tables/experiment-b-decomposition.csv", table_rows)
    _write_llm_exchange(
        run,
        registry,
        live_provider=live_provider,
        calibrated_provider=calibrated_provider,
    )
    summary = {
        "experiment": "B",
        "scientific_claim_status": "not_claimed",
        "trajectories": len(result.trajectories),
        "decomposition_rows": len(result.decompositions),
        "bootstrap_replicates": config.experiment.bootstrap_replicates,
        "experiment_b_inference_status": (
            b_inference.to_dict()["analysis_status"]
        ),
        "self_confirmation_assessments": len(
            result.self_confirmation_assessments
        ),
        "terminal_evaluations": len(terminal_rows),
        "terminal_calibration_artifact": (
            "metrics/experiment-b-terminal-calibration.json"
        ),
        "terminal_calibration_sample_unit": (
            "preference_attribute_forecast"
        ),
        "terminal_calibration_groups": len(terminal_calibration_groups),
        "native_decoder_evaluations": len(native_decoder_rows),
        "held_out_action_evaluations": len(heldout_action_rows),
        "llm_raw_calibrated_terminal_rows": len(
            cached_calibration_rows
        ),
        "llm_raw_calibrated_terminal_pairs": (
            len(cached_calibration_rows) // 2
        ),
        "external_decoder_requests": len(decoder_requests),
        "external_decoder_development_requests": len(
            development_requests
        ),
        "external_decoder_test_requests": len(test_requests),
        "reportable_self_confirming_attribute_count": len(
            result.reportable_self_confirming
        ),
        "reportable_self_confirming_profile_count": len(
            {
                assessment.trajectory_id
                for assessment in result.reportable_self_confirming
            }
        ),
        "mean_terminal_error": mean_or_nan(
            trajectory.terminal_error for trajectory in result.trajectories
        ),
        "mean_shadow_error": mean_or_nan(
            trajectory.terminal_shadow_error for trajectory in result.trajectories
        ),
        "updater_views": updater_views(registry),
        "gate_2_computed_status": gate_2.computed_status,
        "gate_3_computed_status": gate_3.computed_status,
        "gate_4_computed_status": gate_4.computed_status,
    }
    return summary


def _gate_5_for_c(result: Any) -> GateReport:
    ranking = result.rankings
    esr_payload = dict(ranking.evaluation_selection_regret)
    esr = esr_payload.get(
        "evaluation_selection_regret"
    )
    esr_min = esr_payload.get("evaluation_selection_regret_min")
    esr_interval_lower = esr_payload.get(
        "evaluation_selection_regret_interval_envelope_lower"
    )
    reversal = max(
        dict(ranking.pairwise_reversal_probabilities).values(),
        default=0.0,
    )
    tau = ranking.open_closed_kendall_tau
    credible_reversals = list(ranking.credible_pairwise_reversals)
    finite_tau = isinstance(tau, (int, float)) and math.isfinite(float(tau))
    low_rank_agreement_descriptive = finite_tau and float(tau) < 0.50
    substantial_esr = (
        isinstance(esr_min, (int, float))
        and math.isfinite(float(esr_min))
        and float(esr_min) > 0.01
        and isinstance(esr_interval_lower, (int, float))
        and math.isfinite(float(esr_interval_lower))
        and float(esr_interval_lower) > 0.01
    )
    evidence_available = (
        ranking.development_cluster_count > 0
        and ranking.test_cluster_count > 0
        and bool(ranking.pairwise_open_closed_shift_intervals)
    )
    return GateReport(
        gate_id="gate-5",
        title="Evaluation implication",
        criteria=(
            GateCriterion(
                "evaluation-implication-disjunction",
                (
                    "A joint-paired credible pairwise reversal or substantial "
                    "inferential-top-tier evaluation selection regret is present."
                ),
                (
                    (
                        bool(credible_reversals)
                        or substantial_esr
                    )
                    if evidence_available
                    else None
                ),
                {
                    "kendall_tau_b": tau,
                    "low_rank_agreement_threshold": 0.50,
                    "low_rank_agreement_descriptive": (
                        low_rank_agreement_descriptive
                    ),
                    "kendall_tau_is_gate_sufficient_without_interval": False,
                    "credible_reversed_pairs": credible_reversals,
                    "credible_reversal_method": (
                        "joint paired complete-user open/closed error-"
                        "difference and difference-of-differences intervals"
                    ),
                    "maximum_pairwise_reversal_probability": reversal,
                    "reversal_probability_is_gate_sufficient": False,
                    "evaluation_selection_regret": esr,
                    "evaluation_selection_regret_min": esr_min,
                    "evaluation_selection_regret_interval_envelope_lower": (
                        esr_interval_lower
                    ),
                    "evaluation_selection_basis": esr_payload.get(
                        "selection_basis"
                    ),
                    "selection_regret_threshold": 0.01,
                },
                (
                    "joint-paired credible reversal OR every inferential-top-"
                    "tier ESR estimate and the paired test interval envelope "
                    "clear 0.01"
                ),
            ),
        ),
    )


def _run_c(
    config: AppConfig,
    run: RunArtifacts,
    prepared: PreparedStudy,
    *,
    completion_provider: CompletionProvider | None = None,
    live_provider: ResumableCompletionProvider | None = None,
    calibrated_provider: TemperatureCalibratedProvider | None = None,
) -> dict[str, Any]:
    registry = _registry(
        config,
        prepared,
        completion_provider=completion_provider,
    )
    policies = {
        policy_id: build_policy(policy_id)
        for policy_id in config.experiment.policies
    }
    result = run_experiment_c(
        development_users=prepared.development_users[: config.experiment.users],
        test_users=prepared.test_users,
        domains=prepared.domains,
        updaters=registry,
        policies=policies,
        turns=config.experiment.turns,
        trajectories_per_cell=config.experiment.trajectories_per_cell,
        response_model=prepared.response_model,
        seed=config.run.seed,
        bootstrap_replicates=config.experiment.bootstrap_replicates,
        tie_tolerance=config.thresholds.ranking_tie_tolerance,
    )
    cached_calibration_rows: list[
        CachedTerminalCalibrationOutcome
    ] = []
    if calibrated_provider is not None:
        cached_raw_responses = {
            response.request_id: response
            for response in calibrated_provider.raw_responses
        }
        cached_calibrated_responses = {
            response.request_id: response
            for response in calibrated_provider.calibrated_responses
        }
        users_by_id = {
            user.user_id: user
            for user in (
                *prepared.development_users,
                *prepared.test_users,
            )
        }
        split_by_user = {
            **{
                user.user_id: "development"
                for user in prepared.development_users
            },
            **{
                user.user_id: "test"
                for user in prepared.test_users
            },
        }
        batteries_by_domain = {
            battery.domain_id: battery
            for battery in result.terminal_batteries
        }
        histories_by_digest = {
            history.history_digest: history
            for history in result.fixed_histories
        }
        for replay in result.replay_results:
            if not replay.updater_id.startswith("llm_"):
                continue
            history = histories_by_digest[replay.history_digest]
            regime = {
                "balanced": "fixed_balanced",
                "fixed_bias": "fixed_biased",
            }[history.logger_policy_id]
            cached_calibration_rows.extend(
                score_cached_raw_calibrated_terminal(
                    experiment="C",
                    pairing_id=replay.audit_record.trajectory_id,
                    split=split_by_user[history.user_id],
                    regime=regime,
                    updater_id=replay.updater_id,
                    active_terminal_belief=replay.terminal_belief,
                    audit_record=replay.audit_record,
                    user=users_by_id[history.user_id],
                    battery=batteries_by_domain[history.domain_id],
                    raw_responses=cached_raw_responses,
                    calibrated_responses=cached_calibrated_responses,
                )
            )
        for trajectory in result.endogenous_trajectories:
            if not trajectory.updater_id.startswith("llm_"):
                continue
            cached_calibration_rows.extend(
                score_cached_raw_calibrated_terminal(
                    experiment="C",
                    pairing_id=trajectory.trajectory_id,
                    split=split_by_user[trajectory.user_id],
                    regime="endogenous_closed_loop",
                    updater_id=trajectory.updater_id,
                    active_terminal_belief=trajectory.terminal_belief,
                    audit_record=trajectory.audit_record,
                    user=users_by_id[trajectory.user_id],
                    battery=batteries_by_domain[trajectory.domain_id],
                    raw_responses=cached_raw_responses,
                    calibrated_responses=cached_calibrated_responses,
                )
            )
    _write_cached_calibration_outcomes(
        run,
        experiment="C",
        rows=cached_calibration_rows,
        calibration_configured=calibrated_provider is not None,
    )
    if config.artifacts.retain_events:
        run.write_jsonl(
            "events/experiment-c-fixed-histories.jsonl",
            (
                {"schema_version": 1, **history.to_dict()}
                for history in result.fixed_histories
            ),
        )
        run.write_jsonl(
            "events/experiment-c-replays.jsonl",
            (
                {"schema_version": 1, **replay.to_dict()}
                for replay in result.replay_results
            ),
        )
        run.write_jsonl(
            "events/experiment-c-endogenous.jsonl",
            (
                {"schema_version": 1, **trajectory.to_dict(include_truth=True)}
                for trajectory in result.endogenous_trajectories
            ),
        )
    metric_rows = [
        {"schema_version": 1, **row.to_dict()} for row in result.rows
    ]
    run.write_jsonl("metrics/experiment-c.jsonl", metric_rows)
    terminal_calibration_groups: dict[
        tuple[str, str, str, str, str | None], list[Any]
    ] = {}
    for row in result.rows:
        ranking_score = (
            row.system_projection_score
            if row.ranking_score is None
            else row.ranking_score
        )
        terminal_calibration_groups.setdefault(
            (
                row.split,
                row.regime,
                row.updater_id,
                f"ranking:{row.score_basis}",
                None,
            ),
            [],
        ).append(ranking_score)
        terminal_calibration_groups.setdefault(
            (
                row.split,
                row.regime,
                row.updater_id,
                "system_projection",
                None,
            ),
            [],
        ).append(row.system_projection_score)
        for evaluation in row.native_decoder_evaluations:
            terminal_calibration_groups.setdefault(
                (
                    row.split,
                    row.regime,
                    row.updater_id,
                    "deterministic_native_projection",
                    evaluation.decoder_id,
                ),
                [],
            ).append(evaluation.score)
    run.write_json(
        "metrics/experiment-c-terminal-calibration.json",
        {
            "schema_version": 1,
            "experiment": "C",
            "grouping_fields": [
                "split",
                "regime",
                "updater_id",
                "score_projection",
                "decoder_id",
            ],
            "groups": [
                {
                    "split": split,
                    "regime": regime,
                    "updater_id": updater_id,
                    "score_projection": score_projection,
                    "decoder_id": decoder_id,
                    **summarize_terminal_calibration(scores),
                }
                for (
                    split,
                    regime,
                    updater_id,
                    score_projection,
                    decoder_id,
                ), scores in sorted(
                    terminal_calibration_groups.items(),
                    key=lambda item: (
                        item[0][0],
                        item[0][1],
                        item[0][2],
                        item[0][3],
                        "" if item[0][4] is None else item[0][4],
                    ),
                )
            ],
        },
    )
    write_csv(
        run.path / "tables/experiment-c-ranks.csv",
        [
            {
                "updater_id": updater,
                "open_rank": dict(result.rankings.open_ranks)[updater],
                "biased_rank": dict(result.rankings.biased_ranks)[updater],
                "closed_rank": dict(result.rankings.closed_ranks)[updater],
                "open_loop_optimism": dict(
                    result.rankings.open_loop_optimism
                )[updater],
            }
            for updater in sorted(registry)
        ],
    )
    run.write_json("metrics/experiment-c-rankings.json", result.rankings.to_dict())
    run.write_jsonl(
        "events/terminal-batteries.jsonl",
        (
            {"schema_version": 1, **battery.to_dict()}
            for battery in result.terminal_batteries
        ),
    )
    gate_5 = _gate_5_for_c(result)
    gate_report = _all_gates(
        incomplete_gate(1, "Learnable provenance gap", "Run Experiment A."),
        incomplete_gate(2, "Nontrivial soft self-confirmation", "Run Experiment B."),
        incomplete_gate(3, "Attribution beyond evidence selection", "Run Experiment B."),
        incomplete_gate(4, "Native-system validity", "Review native Experiment B/C conditions."),
        gate_5,
    )
    run.write_json("metrics/gate-report.json", gate_report)
    esr = dict(result.rankings.evaluation_selection_regret)
    _write_llm_exchange(
        run,
        registry,
        live_provider=live_provider,
        calibrated_provider=calibrated_provider,
    )
    return {
        "experiment": "C",
        "scientific_claim_status": "not_claimed",
        "fixed_histories": len(result.fixed_histories),
        "replays": len(result.replay_results),
        "endogenous_trajectories": len(result.endogenous_trajectories),
        "evaluation_rows": len(result.rows),
        "terminal_calibration_artifact": (
            "metrics/experiment-c-terminal-calibration.json"
        ),
        "terminal_calibration_sample_unit": (
            "preference_attribute_forecast"
        ),
        "terminal_calibration_groups": len(terminal_calibration_groups),
        "native_decoder_evaluations": sum(
            len(row.native_decoder_evaluations)
            for row in result.rows
        ),
        "llm_raw_calibrated_terminal_rows": len(
            cached_calibration_rows
        ),
        "llm_raw_calibrated_terminal_pairs": (
            len(cached_calibration_rows) // 2
        ),
        "open_closed_kendall_tau_b": result.rankings.open_closed_kendall_tau,
        "evaluation_selection_regret": esr.get("evaluation_selection_regret"),
        "updater_views": updater_views(registry),
        "gate_5_computed_status": gate_5.computed_status,
    }


def _run_sensitivity(
    config: AppConfig,
    run: RunArtifacts,
) -> dict[str, Any]:
    points = sensitivity_grid(
        decision_noise_values=config.sensitivity.decision_noise_values,
        presentation_multipliers=config.sensitivity.presentation_multipliers,
        rank_multipliers=config.sensitivity.rank_multipliers,
        default_multipliers=config.sensitivity.default_multipliers,
        suggestion_multipliers=(
            config.sensitivity.suggestion_multipliers
        ),
        profile_strength_values=config.sensitivity.profile_strength_values,
        prior_uncertainty_values=(
            config.sensitivity.prior_uncertainty_values
        ),
        trajectory_lengths=config.sensitivity.trajectory_lengths,
        response_model_families=(
            config.sensitivity.response_model_families
        ),
        rule_noise_values=config.sensitivity.rule_noise_values,
    )
    stratified_rows = []
    grand_rows = []
    decomposition_rows = []
    model_rows = []
    retained_trajectories = []
    for point in points:
        model = response_model_at(
            point,
            beta=config.response_model.beta,
            rank_scale=config.response_model.rank_scale,
            default_scale=config.response_model.default_scale,
            suggestion_scale=config.response_model.suggestion_scale,
        )
        prepared = _prepare_study(
            config,
            response_model=model,
            # Reuse the same population, contexts, split identities, and
            # semantic choice-noise draws at every grid point. Parameters can
            # change outcomes, but grid enumeration order cannot.
            seed_namespace=0,
        )
        registry = _registry(config, prepared)
        policy_ids = tuple(
            policy_id
            for policy_id in config.experiment.policies
            if policy_id in {"balanced", "soft_profile_conditioned"}
        )
        policies = {policy_id: build_policy(policy_id) for policy_id in policy_ids}
        shadow = ExactActionAwareUpdater(
            model,
            susceptibility_grid(config.response_model.susceptibility_levels),
        )
        result = run_experiment_b(
            users=prepared.test_users,
            domains=prepared.domains,
            updaters=registry,
            policies=policies,
            initial_profile_conditions=("incorrect",),
            turns=point.trajectory_length,
            trajectories_per_cell=config.experiment.trajectories_per_cell,
            response_model=model,
            shadow_updater=shadow,
            seed=config.run.seed,
            materially_wrong_mass=config.thresholds.materially_wrong_mass,
            lcg_threshold=config.thresholds.laundered_confidence_gain,
            shadow_equivalence_tolerance=(
                config.thresholds.shadow_equivalence_tolerance
            ),
            false_stability_tolerance=(
                config.thresholds.false_stability_tolerance
            ),
            direction_tolerance=config.thresholds.direction_tolerance,
            profile_strength=point.profile_strength,
            prior_uncertainty=point.prior_uncertainty,
        )
        eligible_profile_ids = {
            assessment.trajectory_id
            for assessment in result.self_confirmation_assessments
        }
        self_confirming_profile_ids = {
            assessment.trajectory_id
            for assessment in result.reportable_self_confirming
        }
        false_stable_profile_ids = {
            assessment.trajectory_id
            for assessment in result.self_confirmation_assessments
            if assessment.false_stable
        }
        phase_target_id = next(
            (
                updater_id
                for updater_id in (
                    "llm_full_context",
                    "full_context_blind",
                )
                if updater_id in registry
            ),
            next(iter(registry)),
        )
        phase_target_trajectory_ids = {
            trajectory.trajectory_id
            for trajectory in result.trajectories
            if trajectory.updater_id == phase_target_id
        }
        phase_target_assessments = tuple(
            assessment
            for assessment in result.self_confirmation_assessments
            if assessment.trajectory_id in phase_target_trajectory_ids
        )
        phase_target_eligible_ids = {
            assessment.trajectory_id
            for assessment in phase_target_assessments
        }
        phase_target_reportable_ids = {
            assessment.trajectory_id
            for assessment in phase_target_assessments
            if assessment.reportable
        }
        phase_target_decompositions = tuple(
            row
            for row in result.decompositions
            if row.updater_id == phase_target_id
        )
        aware_decompositions = tuple(
            row
            for row in result.decompositions
            if row.updater_id == "fitted_action_aware"
        )
        model_rows.append(
            {
                "schema_version": 1,
                **point.to_dict(),
                "raw_fitted_models": prepared.raw_fitted_models.to_dict(),
                "fitted_models": prepared.fitted_models.to_dict(),
                "calibration": dict(prepared.calibration),
                "held_out_response_diagnostics": dict(
                    prepared.held_out_diagnostics
                ),
            }
        )
        grand_rows.append(
            {
                "schema_version": 1,
                **point.to_dict(),
                "mean_terminal_error": mean_or_nan(
                    trajectory.terminal_error for trajectory in result.trajectories
                ),
                "mean_shadow_error": mean_or_nan(
                    trajectory.terminal_shadow_error
                    for trajectory in result.trajectories
                ),
                "aware_option_ece": prepared.held_out_diagnostics[
                    "aware_option_ece"
                ],
                "raw_aware_option_ece": prepared.held_out_diagnostics[
                    "raw_aware_option_ece"
                ],
                "phase_target_updater_id": phase_target_id,
                "phase_target_is_live_llm": (
                    phase_target_id == "llm_full_context"
                ),
                "mean_information_gain": mean_or_nan(
                    trajectory.cumulative_information_gain
                    for trajectory in result.trajectories
                ),
                "mean_information_gain_per_turn": mean_or_nan(
                    trajectory.cumulative_information_gain
                    / len(trajectory.turns)
                    for trajectory in result.trajectories
                ),
                "mean_regret": mean_or_nan(
                    trajectory.total_regret for trajectory in result.trajectories
                ),
                "mean_regret_per_turn": mean_or_nan(
                    trajectory.total_regret / len(trajectory.turns)
                    for trajectory in result.trajectories
                ),
                "self_confirming_attribute_rate": (
                    len(result.reportable_self_confirming)
                    / max(len(result.self_confirmation_assessments), 1)
                ),
                "self_confirming_profile_rate": (
                    len(self_confirming_profile_ids)
                    / len(eligible_profile_ids)
                ),
                "false_stable_profile_rate": (
                    len(false_stable_profile_ids)
                    / len(eligible_profile_ids)
                ),
                "mean_selection_cost": mean_or_nan(
                    row.evidence_selection_cost for row in result.decompositions
                ),
                "mean_attribution_cost": mean_or_nan(
                    row.profile_attribution_cost for row in result.decompositions
                ),
                "phase_selection_cost": mean_or_nan(
                    row.evidence_selection_cost
                    for row in (
                        aware_decompositions
                        if aware_decompositions
                        else result.decompositions
                    )
                ),
                "phase_attribution_cost": mean_or_nan(
                    row.profile_attribution_cost
                    for row in phase_target_decompositions
                ),
                "phase_self_confirming_profile_rate": (
                    len(phase_target_reportable_ids)
                    / len(phase_target_eligible_ids)
                    if phase_target_eligible_ids
                    else None
                ),
            }
        )
        for domain in prepared.domains:
            for policy_id in policy_ids:
                for updater_id in registry:
                    group = tuple(
                        trajectory
                        for trajectory in result.trajectories
                        if trajectory.domain_id == domain.domain_id
                        and trajectory.policy_id == policy_id
                        and trajectory.updater_id == updater_id
                    )
                    if not group:
                        continue
                    group_ids = {
                        trajectory.trajectory_id for trajectory in group
                    }
                    assessments = tuple(
                        assessment
                        for assessment in result.self_confirmation_assessments
                        if assessment.trajectory_id in group_ids
                    )
                    eligible_ids = {
                        assessment.trajectory_id
                        for assessment in assessments
                    }
                    reportable_ids = {
                        assessment.trajectory_id
                        for assessment in assessments
                        if assessment.reportable
                    }
                    false_stable_ids = {
                        assessment.trajectory_id
                        for assessment in assessments
                        if assessment.false_stable
                    }
                    stratified_rows.append(
                        {
                            "schema_version": 1,
                            **point.to_dict(),
                            "domain_id": domain.domain_id,
                            "policy_id": policy_id,
                            "updater_id": updater_id,
                            "trajectories": len(group),
                            "mean_terminal_error": mean_or_nan(
                                trajectory.terminal_error
                                for trajectory in group
                            ),
                            "mean_shadow_error": mean_or_nan(
                                trajectory.terminal_shadow_error
                                for trajectory in group
                            ),
                            "mean_information_gain": mean_or_nan(
                                trajectory.cumulative_information_gain
                                for trajectory in group
                            ),
                            "mean_information_gain_per_turn": mean_or_nan(
                                trajectory.cumulative_information_gain
                                / len(trajectory.turns)
                                for trajectory in group
                            ),
                            "mean_regret": mean_or_nan(
                                trajectory.total_regret
                                for trajectory in group
                            ),
                            "mean_regret_per_turn": mean_or_nan(
                                trajectory.total_regret
                                / len(trajectory.turns)
                                for trajectory in group
                            ),
                            "self_confirming_attribute_rate": (
                                sum(
                                    assessment.reportable
                                    for assessment in assessments
                                )
                                / len(assessments)
                                if assessments
                                else None
                            ),
                            "self_confirming_profile_rate": (
                                len(reportable_ids) / len(eligible_ids)
                                if eligible_ids
                                else None
                            ),
                            "false_stable_profile_rate": (
                                len(false_stable_ids) / len(eligible_ids)
                                if eligible_ids
                                else None
                            ),
                            "mean_attribution_cost": mean_or_nan(
                                trajectory.terminal_error
                                - trajectory.terminal_shadow_error
                                for trajectory in group
                            ),
                        }
                    )
        for domain in prepared.domains:
            for updater_id in registry:
                paired = tuple(
                    row
                    for row in result.decompositions
                    if row.domain_id == domain.domain_id
                    and row.updater_id == updater_id
                )
                if not paired:
                    continue
                decomposition_rows.append(
                    {
                        "schema_version": 1,
                        **point.to_dict(),
                        "domain_id": domain.domain_id,
                        "updater_id": updater_id,
                        "paired_trajectories": len(paired),
                        "mean_selection_cost": mean_or_nan(
                            row.evidence_selection_cost for row in paired
                        ),
                        "mean_profile_attribution_cost": mean_or_nan(
                            row.profile_attribution_cost for row in paired
                        ),
                        "mean_balanced_attribution_cost": mean_or_nan(
                            row.balanced_attribution_cost for row in paired
                        ),
                        "mean_self_confirmation_interaction": mean_or_nan(
                            row.self_confirmation_interaction for row in paired
                        ),
                    }
                )
        if config.artifacts.retain_events:
            retained_trajectories.extend(
                {
                    "schema_version": 1,
                    "sensitivity_point_id": point.point_id,
                    **trajectory.to_dict(include_truth=True),
                }
                for trajectory in result.trajectories
            )
    phase_criteria = (
        PhaseCriterion(
            "profile-conditioned-actions-reduce-information",
            "phase_selection_cost",
            "gt",
            config.sensitivity.phase_min_selection_cost,
        ),
        PhaseCriterion(
            "fitted-aware-calibration",
            "aware_option_ece",
            "le",
            config.sensitivity.phase_max_aware_ece,
        ),
        PhaseCriterion(
            "profile-writer-over-update",
            "phase_attribution_cost",
            "gt",
            config.sensitivity.phase_min_attribution_cost,
        ),
        PhaseCriterion(
            "wrong-profile-self-confirmation",
            "phase_self_confirming_profile_rate",
            "gt",
            config.sensitivity.phase_min_self_confirming_rate,
        ),
    )
    phase_rows = []
    for row in grand_rows:
        classified = classify_phase_point(row, phase_criteria)
        operational = classified["joint_region"]
        phase_rows.append(
            {
                "schema_version": 1,
                **{
                    key: row[key]
                    for key in (
                        "point_id",
                        "decision_noise",
                        "presentation_multiplier",
                        "rank_multiplier",
                        "default_multiplier",
                        "suggestion_multiplier",
                        "profile_strength",
                        "prior_uncertainty",
                        "trajectory_length",
                        "response_model_family",
                        "rule_noise",
                        "phase_target_updater_id",
                        "phase_target_is_live_llm",
                    )
                },
                "criteria": classified["criteria"],
                "criteria_complete": classified["criteria_complete"],
                "operational_joint_region": operational,
                "confirmatory_llm_joint_region": (
                    operational
                    if row["phase_target_is_live_llm"]
                    else None
                ),
                "interpretation": (
                    "confirmatory_llm"
                    if row["phase_target_is_live_llm"]
                    else "deterministic_profile_writer_proxy"
                ),
            }
        )
    boundary_axes = tuple(
        axis
        for axis in (
            "decision_noise",
            "presentation_multiplier",
            "rank_multiplier",
            "default_multiplier",
            "suggestion_multiplier",
            "profile_strength",
            "prior_uncertainty",
            "trajectory_length",
        )
        if len({row[axis] for row in grand_rows}) > 1
    )
    phase_boundaries = tuple(
        boundary
        for axis in boundary_axes
        for boundary in infer_axis_boundaries(
            grand_rows,
            phase_criteria,
            axis=axis,
        )
    )
    run.write_jsonl("models/sensitivity-fits.jsonl", model_rows)
    run.write_jsonl("metrics/sensitivity.jsonl", stratified_rows)
    run.write_jsonl("metrics/sensitivity-grand.jsonl", grand_rows)
    run.write_jsonl("metrics/sensitivity-phase-points.jsonl", phase_rows)
    run.write_jsonl(
        "metrics/sensitivity-phase-boundaries.jsonl",
        phase_boundaries,
    )
    run.write_json(
        "metrics/sensitivity-phase-specification.json",
        {
            "schema_version": 1,
            "criteria": [
                criterion.to_dict() for criterion in phase_criteria
            ],
            "boundary_kind": "observed_grid_interval",
            "boundary_axes": list(boundary_axes),
            "confirmatory_requires_live_llm_target": True,
        },
    )
    run.write_jsonl(
        "metrics/sensitivity-decomposition.jsonl",
        decomposition_rows,
    )
    write_csv(run.path / "tables/sensitivity.csv", stratified_rows)
    if config.artifacts.retain_events:
        run.write_jsonl(
            "events/sensitivity-trajectories.jsonl",
            retained_trajectories,
        )
    robustness_gate = GateReport(
        gate_id="gate-6",
        title="Robustness",
        criteria=(
            GateCriterion(
                "grid-complete",
                "Every declared simulator grid point completed.",
                len(grand_rows) == len(points),
                {"completed": len(grand_rows), "declared": len(points)},
                "completed == declared",
            ),
            GateCriterion(
                "claim-review-required",
                "Scientific effect robustness requires a preregistered contrast.",
                None,
                None,
                "researcher-reviewed effect criterion",
            ),
        ),
    )
    run.write_json(
        "metrics/gate-report.json",
        _all_gates(
            incomplete_gate(1, "Learnable provenance gap", "Run Experiment A by grid point."),
            incomplete_gate(2, "Nontrivial soft self-confirmation", "Inspect sensitivity contrasts."),
            incomplete_gate(3, "Attribution beyond evidence selection", "Inspect sensitivity contrasts."),
            incomplete_gate(4, "Native-system validity", "Requires native conditions."),
            incomplete_gate(5, "Evaluation implication", "Run Experiment C."),
            robustness_gate,
        ),
    )
    return {
        "experiment": "sensitivity",
        "scientific_claim_status": "not_claimed",
        "declared_points": len(points),
        "completed_points": len(grand_rows),
        "stratified_rows": len(stratified_rows),
        "decomposition_rows": len(decomposition_rows),
        "phase_rows": len(phase_rows),
        "phase_boundary_rows": len(phase_boundaries),
        "retained_trajectories": len(retained_trajectories),
        "gate_6_computed_status": robustness_gate.computed_status,
    }


def _existing_run(
    config: AppConfig,
    *,
    output_root: str | Path | None,
) -> Path:
    root = Path(output_root or config.run.output_root)
    return root / f"{config.run.name}-{config_digest(config)[:12]}"


def _archive_failed_live_attempt(
    destination: Path,
    config: AppConfig,
) -> Path:
    """Preserve a failed artifact before resuming from external journals."""

    manifest_path = destination / "manifest.json"
    resolved_path = destination / "config.resolved.json"
    if not manifest_path.is_file() or not resolved_path.is_file():
        raise ValueError(
            "cannot resume: existing destination lacks a failed-run manifest "
            "or resolved configuration"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "failed":
        raise ValueError(
            "cannot resume: existing destination is not marked failed"
        )
    retained_config = AppConfig.parse(
        json.loads(resolved_path.read_text(encoding="utf-8"))
    )
    if config_digest(retained_config) != config_digest(config):
        raise ValueError(
            "cannot resume: failed destination belongs to a different config"
        )
    recovery_root = destination.parent / ".failed-runs"
    recovery_root.mkdir(parents=True, exist_ok=True)
    attempt = 1
    while True:
        archived = recovery_root / f"{destination.name}-attempt-{attempt:03d}"
        if not archived.exists():
            break
        attempt += 1
    destination.rename(archived)
    return archived


def run_experiment(
    config: AppConfig,
    *,
    output_root: str | Path | None = None,
    allow_existing: bool = False,
    source_config: str | Path | None = None,
    execute_live: bool = False,
    resume_failed_live: bool = False,
) -> dict[str, Any]:
    """Run a validated experiment and return its completed artifact identity."""

    config = config.validated()
    llm_input = _llm_input_manifest(config)
    destination = _existing_run(config, output_root=output_root)
    archived_failed_run: Path | None = None
    if resume_failed_live:
        if config.llm.mode not in {"openai", "openrouter"} or not execute_live:
            raise ValueError(
                "--resume-failed-live requires an OpenAI or OpenRouter "
                "configuration and --execute-live"
            )
        if not destination.exists():
            raise FileNotFoundError(
                f"no failed run exists to resume: {destination}"
            )
        archived_failed_run = _archive_failed_live_attempt(
            destination,
            config,
        )
    if destination.exists():
        if allow_existing:
            valid, errors = verify_run(destination)
            summary_path = destination / "metrics" / "summary.json"
            manifest_path = destination / "manifest.json"
            source_root = Path(__file__).resolve().parents[2]
            manifest = (
                json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest_path.is_file()
                else {}
            )
            source_matches = manifest.get("source_sha256") == source_tree_digest(
                source_root
            )
            retained_llm_input_path = destination / "llm" / "input-manifest.json"
            if llm_input is None:
                llm_input_matches = not retained_llm_input_path.exists()
            else:
                retained_llm_input = (
                    json.loads(
                        retained_llm_input_path.read_text(encoding="utf-8")
                    )
                    if retained_llm_input_path.is_file()
                    else None
                )
                llm_input_matches = retained_llm_input == llm_input
            if (
                valid
                and summary_path.is_file()
                and source_matches
                and llm_input_matches
            ):
                return {
                    "run_dir": str(destination),
                    "reused": True,
                    "summary": json.loads(summary_path.read_text(encoding="utf-8")),
                }
            raise FileExistsError(
                "existing run is not a complete verified artifact for the "
                f"current source tree: checksum_errors={errors}, "
                f"source_matches={source_matches}, "
                f"llm_input_matches={llm_input_matches}"
            )
        raise FileExistsError(
            f"run directory already exists: {destination}; "
            "use --allow-existing only for a verified completed run"
        )

    live_provider = _live_completion_provider(
        config,
        destination=destination,
        execute_live=execute_live,
    )
    uses_llm = any(
        updater_id.startswith("llm_")
        for updater_id in config.experiment.updaters
    )
    raw_completion_provider: CompletionProvider | None = live_provider
    if (
        uses_llm
        and raw_completion_provider is None
        and config.llm.mode == "replay"
    ):
        raw_completion_provider = ReplayProvider(
            read_responses(config.llm.responses_file)
        )
    source_material: bytes | None = None
    if source_config is None:
        config_origin: Mapping[str, Any] = {
            "kind": "programmatic",
            "descriptor": (
                "AppConfig supplied directly to "
                "cape_loop.runner.run_experiment"
            ),
            "config_sha256": config_digest(config),
        }
    else:
        source = Path(source_config)
        source_material = source.read_bytes()
        config_origin = {
            "kind": "toml_file",
            "retained_file": "config.source.toml",
            "source_filename": source.name,
            "source_sha256": sha256(source_material).hexdigest(),
            "config_sha256": config_digest(config),
        }
    run = RunArtifacts.create(
        config,
        root=output_root,
        config_origin=config_origin,
    )
    if llm_input is not None:
        run.write_json("llm/input-manifest.json", llm_input)
    try:
        if source_material is not None:
            run.write_bytes(
                "config.source.toml",
                source_material,
            )
        if config.experiment.kind == "sensitivity":
            summary = _run_sensitivity(config, run)
        else:
            prepared = _prepare_study(config)
            _write_prepared(run, prepared)
            llm_execution = _prepare_llm_execution(
                config,
                prepared,
                raw_provider=raw_completion_provider,
            )
            _write_llm_calibration(run, llm_execution)
            completion_provider = llm_execution.active_provider
            calibrated_provider = (
                completion_provider
                if isinstance(
                    completion_provider,
                    TemperatureCalibratedProvider,
                )
                else None
            )
            if config.experiment.kind == "provenance_audit":
                summary = _run_a(
                    config,
                    run,
                    prepared,
                    completion_provider=completion_provider,
                    raw_completion_provider=(
                        llm_execution.raw_provider
                    ),
                    live_provider=live_provider,
                    calibrated_provider=calibrated_provider,
                )
            elif config.experiment.kind == "closed_loop":
                summary = _run_b(
                    config,
                    run,
                    prepared,
                    completion_provider=completion_provider,
                    live_provider=live_provider,
                    calibrated_provider=calibrated_provider,
                )
            elif config.experiment.kind == "evaluation_validity":
                summary = _run_c(
                    config,
                    run,
                    prepared,
                    completion_provider=completion_provider,
                    live_provider=live_provider,
                    calibrated_provider=calibrated_provider,
                )
            else:
                raise ValueError(
                    f"unsupported experiment kind: {config.experiment.kind}"
                )
        run.finalize(summary)
    except Exception as exc:
        manifest_path = run.path / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["status"] = "failed"
        run.write_json("manifest.json", manifest)
        run.write_json(
            "failure.json",
            {
                "schema_version": 1,
                "exception_type": type(exc).__name__,
                "message": str(exc),
            },
        )
        if config.artifacts.checksum_manifest:
            run.write_checksums()
        raise
    return {
        "run_dir": str(run.path),
        "reused": False,
        "archived_failed_run": (
            None
            if archived_failed_run is None
            else str(archived_failed_run)
        ),
        "summary": summary,
    }
