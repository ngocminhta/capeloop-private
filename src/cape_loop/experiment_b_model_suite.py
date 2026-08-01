"""Frozen, budget-bounded OpenRouter model panel for Experiment B."""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any, Mapping
import json

from .artifacts import config_digest, read_control_bytes
from .config import AppConfig, load_config
from .llm_preflight import require_live_llm_budget


SUITE_ID = "cape-loop-experiment-b-bounded-calibration-v1"
DEFAULT_SUITE_PATH = (
    Path(__file__).resolve().parents[2]
    / "data/model-suites/experiment-b-bounded-calibration-v1.json"
)
_FULL_SEEDS = ("correct", "incorrect")
_FULL_POLICIES = ("balanced", "soft_profile_conditioned", "exploratory")
_TARGETED_SEEDS = ("incorrect",)
_TARGETED_POLICIES = ("balanced", "soft_profile_conditioned")


@dataclass(frozen=True, slots=True)
class _ArmContract:
    role: str
    model: str
    effort: str
    upstream: str
    profile: str
    seeds: tuple[str, ...]
    policies: tuple[str, ...]
    timing: str
    primary: bool
    max_requests: int
    max_tokens: int


_FROZEN_ARMS = {
    "primary-gemini-3-6-flash": _ArmContract(
        "primary_panel", "google/gemini-3.6-flash", "minimal",
        "google-vertex/global", "full_base", _FULL_SEEDS, _FULL_POLICIES,
        "frozen_before_bounded_multi_user_calibration", True, 900, 6_000_000,
    ),
    "primary-gpt-5-6-luna": _ArmContract(
        "primary_panel", "openai/gpt-5.6-luna", "low", "", "full_base",
        _FULL_SEEDS, _FULL_POLICIES,
        "frozen_before_bounded_multi_user_calibration", True, 900, 6_000_000,
    ),
    "primary-mistral-large-3": _ArmContract(
        "primary_panel", "mistralai/mistral-large-2512", "", "", "full_base",
        _FULL_SEEDS, _FULL_POLICIES,
        "frozen_before_bounded_multi_user_calibration", True, 900, 6_000_000,
    ),
    "secondary-deepseek-v4-flash": _ArmContract(
        "targeted_secondary_replication", "deepseek/deepseek-v4-flash", "", "",
        "incorrect_seed_balanced_soft_only", _TARGETED_SEEDS,
        _TARGETED_POLICIES, "post_pilot_targeted_secondary", False, 300,
        2_000_000,
    ),
}
_ARM_ORDER = tuple(_FROZEN_ARMS)
_PRIMARY_IDS = _ARM_ORDER[:3]
_SECONDARY_IDS = _ARM_ORDER[3:]
_ANALYSIS_POLICY = {
    "primary_arm_ids": list(_PRIMARY_IDS),
    "secondary_arm_ids": list(_SECONDARY_IDS),
    "primary_results_are_model_specific": True,
    "claim_unit": "per_model",
    "cross_model_multiplicity_policy": (
        "no_any-model_or_omnibus_claim_in_this_suite"
    ),
    "models_are_not_user_clusters": True,
    "secondary_selected_after_pilot": True,
    "secondary_may_be_pooled_with_primary": False,
}
_TOP_KEYS = {
    "schema_version", "suite_id", "resolved_on", "status", "transport",
    "project_generated", "source_status", "license", "expected_consumer",
    "base_design_id", "analysis_policy", "arms",
}
_ARM_KEYS = {
    "arm_id", "role", "model", "reasoning_effort",
    "openrouter_upstream_provider", "condition_profile",
    "initial_profile_conditions", "policies", "run_name", "output_subdir",
    "max_requests", "max_total_tokens", "selection_timing",
    "primary_analysis_eligible",
}


@dataclass(frozen=True, slots=True)
class ExperimentBModelArm:
    arm_id: str
    role: str
    model: str
    reasoning_effort: str
    openrouter_upstream_provider: str
    condition_profile: str
    initial_profile_conditions: tuple[str, ...]
    policies: tuple[str, ...]
    run_name: str
    output_subdir: str
    max_requests: int
    max_total_tokens: int
    selection_timing: str
    primary_analysis_eligible: bool

    def frozen_contract(self) -> _ArmContract:
        return _ArmContract(
            self.role, self.model, self.reasoning_effort,
            self.openrouter_upstream_provider, self.condition_profile,
            self.initial_profile_conditions, self.policies,
            self.selection_timing, self.primary_analysis_eligible,
            self.max_requests, self.max_total_tokens,
        )


@dataclass(frozen=True, slots=True)
class ExperimentBModelSuite:
    schema_version: int
    suite_id: str
    resolved_on: str
    status: str
    project_generated: bool
    source_status: str
    license: str
    expected_consumer: str
    transport: str
    base_design_id: str
    analysis_policy: Mapping[str, Any]
    arms: tuple[ExperimentBModelArm, ...]
    source_path: Path
    source_sha256: str


def _without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant is not allowed: {value}")


def _exact_keys(raw: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(raw) != expected:
        raise ValueError(
            f"{label} fields differ from the frozen contract; "
            f"missing={sorted(expected - set(raw))}, "
            f"unknown={sorted(set(raw) - expected)}"
        )


def _strings(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ValueError(f"{label} must be a nonempty string array")
    result = tuple(value)
    if len(result) != len(set(result)):
        raise ValueError(f"{label} must not contain duplicates")
    return result


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _subdir(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{label} must be a relative POSIX path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != value
    ):
        raise ValueError(f"{label} must remain inside the suite output root")
    return value


def _arm(raw: Any, index: int) -> ExperimentBModelArm:
    if not isinstance(raw, Mapping):
        raise ValueError(f"arms[{index}] must be an object")
    _exact_keys(raw, _ARM_KEYS, f"arms[{index}]")
    string_fields = (
        "arm_id", "role", "model", "reasoning_effort",
        "openrouter_upstream_provider", "condition_profile", "run_name",
        "selection_timing",
    )
    if not all(isinstance(raw[field], str) for field in string_fields):
        raise ValueError(f"arms[{index}] string fields must be strings")
    if not isinstance(raw["primary_analysis_eligible"], bool):
        raise ValueError(f"arms[{index}].primary_analysis_eligible must be Boolean")
    arm = ExperimentBModelArm(
        arm_id=raw["arm_id"], role=raw["role"], model=raw["model"],
        reasoning_effort=raw["reasoning_effort"],
        openrouter_upstream_provider=raw["openrouter_upstream_provider"],
        condition_profile=raw["condition_profile"],
        initial_profile_conditions=_strings(
            raw["initial_profile_conditions"],
            f"arms[{index}].initial_profile_conditions",
        ),
        policies=_strings(raw["policies"], f"arms[{index}].policies"),
        run_name=raw["run_name"],
        output_subdir=_subdir(raw["output_subdir"], f"arms[{index}].output_subdir"),
        max_requests=_positive_int(raw["max_requests"], f"arms[{index}].max_requests"),
        max_total_tokens=_positive_int(
            raw["max_total_tokens"], f"arms[{index}].max_total_tokens"
        ),
        selection_timing=raw["selection_timing"],
        primary_analysis_eligible=raw["primary_analysis_eligible"],
    )
    expected = _FROZEN_ARMS.get(arm.arm_id)
    if expected is None or arm.frozen_contract() != expected:
        raise ValueError(f"arm {arm.arm_id!r} differs from its frozen contract")
    if not arm.run_name or "/" in arm.run_name or "\\" in arm.run_name:
        raise ValueError(f"arm {arm.arm_id!r} has an invalid run_name")
    expected_prefix = "primary/" if arm.primary_analysis_eligible else "secondary/"
    if not arm.output_subdir.startswith(expected_prefix):
        raise ValueError(f"arm {arm.arm_id!r} uses the wrong output subtree")
    return arm


def load_experiment_b_model_suite(
    path: str | Path = DEFAULT_SUITE_PATH,
) -> ExperimentBModelSuite:
    """Strictly load the canonical model and primary/secondary declaration."""

    source = Path(path).resolve()
    payload = read_control_bytes(source, label="Experiment B model suite")
    try:
        raw = json.loads(
            payload.decode("utf-8"), object_pairs_hook=_without_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid Experiment B model suite JSON: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("Experiment B model suite must be a JSON object")
    _exact_keys(raw, _TOP_KEYS, "Experiment B model suite")
    metadata = (
        raw["schema_version"], raw["suite_id"], raw["resolved_on"], raw["status"],
        raw["project_generated"], raw["source_status"], raw["license"],
        raw["expected_consumer"],
        raw["transport"], raw["base_design_id"],
    )
    expected_metadata = (
        1, SUITE_ID, "2026-07-31", "frozen_for_bounded_multi_user_calibration",
        True, "project-authored-frozen-protocol", "Apache-2.0",
        "cape-loop experiment-b model-suite",
        "openrouter", "experiment-b-openrouter-six-turn-full-v1",
    )
    if metadata != expected_metadata:
        raise ValueError("Experiment B model suite metadata changed")
    if raw["analysis_policy"] != _ANALYSIS_POLICY:
        raise ValueError("analysis_policy must retain the primary/secondary boundary")
    if not isinstance(raw["arms"], list):
        raise ValueError("arms must be an array")
    arms = tuple(_arm(value, index) for index, value in enumerate(raw["arms"]))
    if tuple(arm.arm_id for arm in arms) != _ARM_ORDER:
        raise ValueError("arms must retain the frozen primary-then-secondary order")
    for field in ("model", "run_name", "output_subdir"):
        if len({getattr(arm, field) for arm in arms}) != len(arms):
            raise ValueError(f"suite arm {field} values must be unique")
    return ExperimentBModelSuite(
        1, SUITE_ID, raw["resolved_on"], raw["status"],
        raw["project_generated"], raw["source_status"], raw["license"],
        raw["expected_consumer"], raw["transport"], raw["base_design_id"],
        dict(raw["analysis_policy"]), arms, source, sha256(payload).hexdigest(),
    )


def _base(path: str | Path) -> tuple[Path, AppConfig, str]:
    source = Path(path).resolve()
    payload = read_control_bytes(source, label="Experiment B base config")
    config = load_config(source)
    experiment = config.experiment
    if config.llm.mode != "openrouter" or experiment.kind != "closed_loop":
        raise ValueError("suite requires an OpenRouter closed_loop base")
    if config.run.deterministic:
        raise ValueError("live Experiment B base must be nondeterministic")
    design = (
        experiment.domains, experiment.mechanisms, experiment.response_modes,
        experiment.initial_profile_conditions, experiment.policies,
        experiment.users, experiment.trajectories_per_cell, experiment.turns,
    )
    expected_design = (
        ("travel", "writing"), ("ranking", "default", "suggestion"),
        ("naturally_sampled",), _FULL_SEEDS, _FULL_POLICIES, 8, 1, 6,
    )
    if design != expected_design:
        raise ValueError("base must retain the full eight-user, six-turn B design")
    llm_updaters = tuple(
        updater for updater in experiment.updaters if updater.startswith("llm_")
    )
    transport_contract = (
        config.llm.calibration, config.llm.calibration_users,
        config.llm.max_retries, config.llm.max_output_tokens,
        config.llm.openrouter_allow_fallbacks,
        config.llm.openrouter_require_parameters,
    )
    if llm_updaters != ("llm_full_context",) or transport_contract != (
        "temperature", 1, 0, 2048, False, True,
    ):
        raise ValueError("base must retain the strict calibrated OpenRouter contract")
    return source, config, sha256(payload).hexdigest()


def _config(base: AppConfig, arm: ExperimentBModelArm) -> AppConfig:
    experiment = base.experiment
    if arm.condition_profile != "full_base":
        experiment = replace(
            experiment, initial_profile_conditions=arm.initial_profile_conditions,
            policies=arm.policies,
        )
    return replace(
        base,
        run=replace(
            base.run, name=arm.run_name,
            output_root=(Path(base.run.output_root) / arm.output_subdir).as_posix(),
        ),
        experiment=experiment,
        llm=replace(
            base.llm, model=arm.model, reasoning_effort=arm.reasoning_effort,
            openrouter_upstream_provider=arm.openrouter_upstream_provider,
            max_requests=arm.max_requests, max_total_tokens=arm.max_total_tokens,
        ),
    ).validated()


def build_experiment_b_model_suite_plan(
    base_config: str | Path,
    *,
    suite_path: str | Path = DEFAULT_SUITE_PATH,
    output_root: str | Path | None = None,
) -> tuple[dict[str, Any], tuple[AppConfig, ...]]:
    """Return a credential-free plan and four isolated resolved configs."""

    suite = load_experiment_b_model_suite(suite_path)
    base_path, base, base_sha256 = _base(base_config)
    root = Path(output_root or base.run.output_root).resolve()
    configs: list[AppConfig] = []
    records: list[dict[str, Any]] = []
    for arm in suite.arms:
        config = _config(base, arm)
        preflight = require_live_llm_budget(config)
        if preflight is None:
            raise AssertionError("validated model-suite arm lost its LLM updater")
        expected_attempts = 636 if arm.primary_analysis_eligible else 252
        attempts = preflight["physical_http_attempt_upper_bound"]
        if attempts != expected_attempts:
            raise ValueError(
                f"arm {arm.arm_id!r} must preflight to {expected_attempts} attempts"
            )
        digest = config_digest(config)
        arm_root = (root / arm.output_subdir).resolve()
        run_id = f"{config.run.name}-{digest[:12]}"
        records.append({
            "arm_id": arm.arm_id,
            "role": arm.role,
            "analysis_set": "primary" if arm.primary_analysis_eligible else "secondary",
            "model": arm.model,
            "reasoning_effort": arm.reasoning_effort or None,
            "transport": config.llm.mode,
            "openrouter_upstream_provider": arm.openrouter_upstream_provider or None,
            "condition_profile": arm.condition_profile,
            "conditions": {
                "domains": list(config.experiment.domains),
                "initial_profile_conditions": list(
                    config.experiment.initial_profile_conditions
                ),
                "policies": list(config.experiment.policies),
                "users": config.experiment.users,
                "turns": config.experiment.turns,
            },
            "selection_timing": arm.selection_timing,
            "primary_analysis_eligible": arm.primary_analysis_eligible,
            "secondary_never_pooled_with_primary": not arm.primary_analysis_eligible,
            "config_sha256": digest,
            "run_name": config.run.name,
            "run_id": run_id,
            "output_root": str(arm_root),
            "run_directory": str((arm_root / run_id).resolve()),
            "physical_http_attempt_upper_bound": attempts,
            "experiment_request_upper_bound": preflight["experiment_request_upper_bound"],
            "calibration_request_count": preflight["calibration_request_count"],
            "maximum_output_token_allocation": preflight[
                "maximum_output_token_allocation"
            ],
            "max_requests": config.llm.max_requests,
            "max_total_tokens": config.llm.max_total_tokens,
            "request_headroom": preflight["request_headroom"],
            "within_declared_retry_expanded_bounds": preflight[
                "within_declared_retry_expanded_bounds"
            ],
            "execution_status": "planned",
            "result": None,
        })
        configs.append(config)
    if len({record["run_directory"] for record in records}) != len(records):
        raise ValueError("suite arms resolve to a shared run directory")
    plan = {
        "schema_version": 1,
        "suite_id": suite.suite_id,
        "status": "planned",
        "live_execution": False,
        "credential_read": False,
        "suite_source": str(suite.source_path),
        "suite_source_sha256": suite.source_sha256,
        "base_design_id": suite.base_design_id,
        "base_config_source": str(base_path),
        "base_config_source_sha256": base_sha256,
        "output_root": str(root),
        "execution_order": list(_ARM_ORDER),
        "analysis_policy": dict(suite.analysis_policy),
        "arms": records,
        "primary_physical_attempt_upper_bound": 3 * 636,
        "secondary_physical_attempt_upper_bound": 252,
        "total_physical_attempt_upper_bound": 3 * 636 + 252,
        "execution_policy": (
            "Sequential isolated runs with one provider ledger and output subtree "
            "per arm; the targeted secondary arm is never pooled into primary."
        ),
    }
    return plan, tuple(configs)


def orchestrate_experiment_b_model_suite(
    base_config: str | Path,
    *,
    execute_live: bool,
    suite_path: str | Path = DEFAULT_SUITE_PATH,
    output_root: str | Path | None = None,
    allow_existing: bool = False,
) -> dict[str, Any]:
    """Plan or explicitly execute all arms sequentially."""

    if not isinstance(execute_live, bool):
        raise TypeError("execute_live must be an explicit Boolean")
    plan, configs = build_experiment_b_model_suite_plan(
        base_config, suite_path=suite_path, output_root=output_root,
    )
    if not execute_live:
        return plan
    from .runner import run_experiment

    plan["status"] = "executing"
    plan["live_execution"] = True
    for record, config in zip(plan["arms"], configs):
        record["execution_status"] = "executing"
        result = run_experiment(
            config, output_root=record["output_root"],
            allow_existing=allow_existing, source_config=None, execute_live=True,
        )
        if Path(result["run_dir"]).resolve() != Path(record["run_directory"]).resolve():
            raise ValueError(f"arm {record['arm_id']!r} returned the wrong run path")
        record["execution_status"] = "complete"
        record["result"] = result
    plan["status"] = "complete"
    return plan


__all__ = [
    "DEFAULT_SUITE_PATH",
    "ExperimentBModelArm",
    "ExperimentBModelSuite",
    "SUITE_ID",
    "build_experiment_b_model_suite_plan",
    "load_experiment_b_model_suite",
    "orchestrate_experiment_b_model_suite",
]
