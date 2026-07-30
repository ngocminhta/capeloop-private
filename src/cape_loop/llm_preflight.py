"""Credential-free request-budget preflight for adaptive LLM experiments."""

from __future__ import annotations

from typing import Any

from .config import AppConfig
from .experiments.evaluation import ALL_REGIMES
from .heldout import build_default_paraphrase_suite
from .sensitivity import sensitivity_grid


_ATTRIBUTE_COUNT = 3
_ANCHOR_DIRECTIONS = 2
_CALIBRATION_MECHANISMS = 4


def _llm_updater_ids(config: AppConfig) -> tuple[str, ...]:
    return tuple(
        updater_id
        for updater_id in config.experiment.updaters
        if updater_id.startswith("llm_")
    )


def _calibration_requests(
    config: AppConfig,
    llm_updater_ids: tuple[str, ...],
) -> int:
    if config.llm.calibration != "temperature":
        return 0
    return (
        config.llm.calibration_users
        * len(config.experiment.domains)
        * _ATTRIBUTE_COUNT
        * _ANCHOR_DIRECTIONS
        * _CALIBRATION_MECHANISMS
        * len(llm_updater_ids)
    )


def _provenance_audit_requests(
    config: AppConfig,
    llm_updater_ids: tuple[str, ...],
) -> tuple[int, int]:
    experiment_requests = (
        config.experiment.users
        * len(config.experiment.domains)
        * _ATTRIBUTE_COUNT
        * _ANCHOR_DIRECTIONS
        * len(config.experiment.prior_strengths)
        * len(config.experiment.mechanisms)
        * len(config.experiment.response_modes)
        * len(llm_updater_ids)
    )
    paraphrase_requests = 0
    if (
        "llm_full_context" in llm_updater_ids
        and "naturally_sampled" in config.experiment.response_modes
    ):
        paraphrase_requests = (
            len(config.experiment.domains)
            * len(config.experiment.mechanisms)
            * _ANCHOR_DIRECTIONS
            * len(build_default_paraphrase_suite().for_split("test"))
        )
    return experiment_requests, paraphrase_requests


def _closed_loop_requests(
    config: AppConfig,
    llm_updater_ids: tuple[str, ...],
) -> int:
    return (
        config.experiment.users
        * len(config.experiment.domains)
        * len(config.experiment.initial_profile_conditions)
        * config.experiment.trajectories_per_cell
        * len(config.experiment.policies)
        * config.experiment.turns
        * len(llm_updater_ids)
    )


def _evaluation_requests(
    config: AppConfig,
    llm_updater_ids: tuple[str, ...],
) -> int:
    development_users = max(8, config.experiment.users)
    return (
        (development_users + config.experiment.users)
        * len(config.experiment.domains)
        * config.experiment.trajectories_per_cell
        * len(ALL_REGIMES)
        * config.experiment.turns
        * len(llm_updater_ids)
    )


def _sensitivity_requests(
    config: AppConfig,
    llm_updater_ids: tuple[str, ...],
) -> tuple[int, int, int]:
    points = sensitivity_grid(
        design=config.sensitivity.design,
        decision_noise_values=config.sensitivity.decision_noise_values,
        presentation_multipliers=(
            config.sensitivity.presentation_multipliers
        ),
        profile_conditioning_strength_values=(
            config.sensitivity.profile_conditioning_strength_values
        ),
        rank_multipliers=config.sensitivity.rank_multipliers,
        default_multipliers=config.sensitivity.default_multipliers,
        suggestion_multipliers=(
            config.sensitivity.suggestion_multipliers
        ),
        profile_strength_values=(
            config.sensitivity.profile_strength_values
        ),
        prior_uncertainty_values=(
            config.sensitivity.prior_uncertainty_values
        ),
        trajectory_lengths=config.sensitivity.trajectory_lengths,
        response_model_families=(
            config.sensitivity.response_model_families
        ),
        rule_noise_values=config.sensitivity.rule_noise_values,
    )
    trajectory_turns = sum(point.trajectory_length for point in points)
    requests = (
        len(config.experiment.domains)
        * config.experiment.users
        * config.experiment.trajectories_per_cell
        * len(config.experiment.policies)
        * len(llm_updater_ids)
        * trajectory_turns
    )
    return requests, len(points), trajectory_turns


def build_llm_request_preflight(
    config: AppConfig,
) -> dict[str, Any] | None:
    """Return an exact logical and worst-case physical request-count bound.

    Adaptive prompts depend on earlier provider responses, so their complete
    byte-based input-token reservation cannot be known before execution. The
    provider ledger still reserves and enforces that hard cumulative token
    ceiling before every physical attempt. This preflight additionally proves
    that the declared request ceiling and maximum possible output allocation
    can cover the complete retry-expanded request plan.
    """

    llm_updater_ids = _llm_updater_ids(config)
    if not llm_updater_ids:
        return None

    experiment_requests = 0
    paraphrase_requests = 0
    grid_points: int | None = None
    trajectory_turns: int | None = None
    if config.experiment.kind == "provenance_audit":
        experiment_requests, paraphrase_requests = (
            _provenance_audit_requests(config, llm_updater_ids)
        )
    elif config.experiment.kind == "closed_loop":
        experiment_requests = _closed_loop_requests(
            config,
            llm_updater_ids,
        )
    elif config.experiment.kind == "evaluation_validity":
        experiment_requests = _evaluation_requests(
            config,
            llm_updater_ids,
        )
    elif config.experiment.kind == "sensitivity":
        (
            experiment_requests,
            grid_points,
            trajectory_turns,
        ) = _sensitivity_requests(config, llm_updater_ids)
    else:  # guarded by AppConfig, retained as a fail-closed boundary
        raise ValueError(
            f"unsupported experiment kind: {config.experiment.kind}"
        )

    calibration_requests = _calibration_requests(
        config,
        llm_updater_ids,
    )
    logical_requests = (
        experiment_requests
        + calibration_requests
        + paraphrase_requests
    )
    live = config.llm.mode in {"openai", "openrouter"}
    retry_expansion_factor = config.llm.max_retries + 1 if live else None
    physical_requests = (
        logical_requests * retry_expansion_factor
        if retry_expansion_factor is not None
        else None
    )
    output_token_upper_bound = (
        physical_requests * config.llm.max_output_tokens
        if physical_requests is not None
        else None
    )
    within_request_ceiling = (
        physical_requests <= config.llm.max_requests
        if physical_requests is not None
        else None
    )
    within_output_token_ceiling = (
        output_token_upper_bound <= config.llm.max_total_tokens
        if output_token_upper_bound is not None
        else None
    )
    return {
        "schema_version": 1,
        "kind": "adaptive-llm-request-preflight",
        "experiment_kind": config.experiment.kind,
        "execution_mode": config.llm.mode,
        "calibration": config.llm.calibration,
        "llm_updater_ids": list(llm_updater_ids),
        "experiment_request_upper_bound": experiment_requests,
        "calibration_request_count": calibration_requests,
        "heldout_paraphrase_request_upper_bound": paraphrase_requests,
        "logical_completion_upper_bound": logical_requests,
        "live_transport": live,
        "retry_expansion_factor": retry_expansion_factor,
        "physical_http_attempt_upper_bound": physical_requests,
        "configured_max_requests": (
            config.llm.max_requests if live else None
        ),
        "request_headroom": (
            config.llm.max_requests - physical_requests
            if physical_requests is not None
            else None
        ),
        "within_request_ceiling": within_request_ceiling,
        "max_output_tokens_per_attempt": config.llm.max_output_tokens,
        "maximum_output_token_allocation": output_token_upper_bound,
        "configured_max_total_tokens": (
            config.llm.max_total_tokens if live else None
        ),
        "output_token_headroom_before_input": (
            config.llm.max_total_tokens - output_token_upper_bound
            if output_token_upper_bound is not None
            else None
        ),
        "within_output_token_ceiling": within_output_token_ceiling,
        "within_declared_retry_expanded_bounds": (
            within_request_ceiling is True
            and within_output_token_ceiling is True
            if live
            else None
        ),
        "adaptive_input_token_preflight": (
            "runtime_enforced_before_each_attempt"
            if live
            else "not_applicable"
        ),
        "budget_accounting_unit": "physical_http_attempt",
        "grid_design": (
            config.sensitivity.design
            if config.experiment.kind == "sensitivity"
            else None
        ),
        "grid_points": grid_points,
        "sum_trajectory_lengths_over_points": trajectory_turns,
    }


def require_live_llm_budget(config: AppConfig) -> dict[str, Any] | None:
    """Reject a live configuration whose full retry-expanded plan cannot fit."""

    preflight = build_llm_request_preflight(config)
    if preflight is None or not preflight["live_transport"]:
        return preflight
    if preflight["within_request_ceiling"] is not True:
        raise ValueError(
            f"live LLM {config.experiment.kind} can require up to "
            f"{preflight['physical_http_attempt_upper_bound']} physical "
            "HTTP attempts after retry expansion, exceeding "
            f"llm.max_requests = {config.llm.max_requests}; reduce the "
            "design or retries to stay within the reviewed hard ceiling"
        )
    if preflight["within_output_token_ceiling"] is not True:
        raise ValueError(
            f"live LLM {config.experiment.kind} can allocate up to "
            f"{preflight['maximum_output_token_allocation']} output tokens "
            "after retry expansion before counting adaptive prompt input, "
            f"exceeding llm.max_total_tokens = "
            f"{config.llm.max_total_tokens}; reduce max_output_tokens, the "
            "design, or retries to stay within the reviewed hard ceiling"
        )
    return preflight
