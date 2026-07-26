"""Schema-versioned, dependency-free TOML configuration.

The configuration layer deliberately rejects unknown keys. A typo in a paper run
must fail before simulation instead of silently selecting a default.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit
import json
import math
import tomllib


SCHEMA_VERSION = 1
KNOWN_DOMAINS = frozenset({"travel", "writing"})
KNOWN_EXPERIMENTS = frozenset(
    {"provenance_audit", "closed_loop", "evaluation_validity", "sensitivity"}
)
KNOWN_MECHANISMS = frozenset(
    {
        "balanced",
        "restricted",
        "ranking",
        "default",
        "suggested",
        "suggestion",
    }
)
KNOWN_RESPONSE_MODES = frozenset({"controlled_anchor", "naturally_sampled"})
KNOWN_RESPONSE_MODEL_FAMILIES = frozenset(
    {"random_utility", "rule_based"}
)
KNOWN_LLM_MODES = frozenset({"replay", "openai", "openrouter"})
KNOWN_LLM_MODEL_ROLES = frozenset({"primary", "replication", "decoder"})
KNOWN_REASONING_EFFORTS = frozenset(
    {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
)
KNOWN_POLICIES = frozenset(
    {"balanced", "soft_profile_conditioned", "exploratory", "fixed_bias", "hard_filter"}
)
KNOWN_UPDATERS = frozenset(
    {
        "no_update",
        "exact_action_aware",
        "fitted_action_aware",
        "fitted_action_unaware",
        "response_only",
        "full_context_blind",
        "provenance_discount",
        "provenance_aware",
        "conservative",
        "episodic_memory",
        "semantic_memory",
        "provenance_linked_memory",
        "llm_response_only",
        "llm_full_context",
        "llm_provenance_aware",
    }
)


class ConfigError(ValueError):
    """Raised when a run configuration is invalid."""


def _only_keys(section: str, value: Mapping[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ConfigError(f"unknown key(s) in [{section}]: {', '.join(unknown)}")


def _tuple_of_strings(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(isinstance(x, str) for x in value):
        raise ConfigError(f"{field_name} must be a non-empty array of strings")
    result = tuple(value)
    if len(result) != len(set(result)):
        raise ConfigError(f"{field_name} must not contain duplicates")
    return result


def _require_integer(value: Any, field_name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        comparison = "non-negative" if minimum == 0 else f">= {minimum}"
        raise ConfigError(f"{field_name} must be an integer that is {comparison}")
    return value


def _require_finite_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{field_name} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ConfigError(f"{field_name} must be finite")
    return numeric


def _reject_duplicates(values: tuple[Any, ...], field_name: str) -> None:
    if len(values) != len(set(values)):
        raise ConfigError(f"{field_name} must not contain duplicates")


@dataclass(frozen=True, slots=True)
class RunSection:
    name: str = "cape-loop-smoke"
    seed: int = 1729
    output_root: str = "runs"
    deterministic: bool = True

    @classmethod
    def parse(cls, raw: Mapping[str, Any]) -> "RunSection":
        _only_keys("run", raw, {"name", "seed", "output_root", "deterministic"})
        result = cls(**raw)
        if (
            not isinstance(result.name, str)
            or not result.name
            or any(c in result.name for c in "/\\\0")
        ):
            raise ConfigError("run.name must be a non-empty filesystem-safe name")
        _require_integer(result.seed, "run.seed", minimum=0)
        if not isinstance(result.output_root, str) or not result.output_root:
            raise ConfigError("run.output_root must not be empty")
        if not isinstance(result.deterministic, bool):
            raise ConfigError("run.deterministic must be a Boolean")
        return result


@dataclass(frozen=True, slots=True)
class ExperimentSection:
    kind: str = "provenance_audit"
    domains: tuple[str, ...] = ("travel", "writing")
    mechanisms: tuple[str, ...] = ("balanced", "restricted", "default", "suggested")
    response_modes: tuple[str, ...] = ("controlled_anchor", "naturally_sampled")
    prior_strengths: tuple[float, ...] = (0.0,)
    policies: tuple[str, ...] = ("balanced",)
    updaters: tuple[str, ...] = (
        "no_update",
        "exact_action_aware",
        "fitted_action_aware",
        "fitted_action_unaware",
        "provenance_discount",
    )
    users: int = 8
    trajectories_per_cell: int = 1
    turns: int = 1
    bootstrap_replicates: int = 0

    @classmethod
    def parse(cls, raw: Mapping[str, Any]) -> "ExperimentSection":
        allowed = {
            "kind",
            "domains",
            "mechanisms",
            "response_modes",
            "prior_strengths",
            "policies",
            "updaters",
            "users",
            "trajectories_per_cell",
            "turns",
            "bootstrap_replicates",
        }
        _only_keys("experiment", raw, allowed)
        prepared = dict(raw)
        for name in ("domains", "mechanisms", "response_modes", "policies", "updaters"):
            if name in prepared:
                prepared[name] = _tuple_of_strings(prepared[name], f"experiment.{name}")
        if "prior_strengths" in prepared:
            values = prepared["prior_strengths"]
            if not isinstance(values, list) or not values:
                raise ConfigError(
                    "experiment.prior_strengths must be a non-empty numeric array"
                )
            prepared["prior_strengths"] = tuple(
                _require_finite_number(
                    value,
                    "experiment.prior_strengths",
                )
                for value in values
            )
        result = cls(**prepared)
        if not isinstance(result.kind, str):
            raise ConfigError("experiment.kind must be a string")
        if result.kind not in KNOWN_EXPERIMENTS:
            raise ConfigError(
                f"experiment.kind must be one of {sorted(KNOWN_EXPERIMENTS)}"
            )
        checks = (
            ("domains", result.domains, KNOWN_DOMAINS),
            ("mechanisms", result.mechanisms, KNOWN_MECHANISMS),
            ("response_modes", result.response_modes, KNOWN_RESPONSE_MODES),
            ("policies", result.policies, KNOWN_POLICIES),
            ("updaters", result.updaters, KNOWN_UPDATERS),
        )
        for name, values, known in checks:
            unknown = sorted(set(values) - known)
            if unknown:
                raise ConfigError(
                    f"unknown experiment.{name}: {', '.join(unknown)}"
                )
        for name in ("users", "trajectories_per_cell", "turns"):
            _require_integer(
                getattr(result, name),
                f"experiment.{name}",
                minimum=1,
            )
        _require_integer(
            result.bootstrap_replicates,
            "experiment.bootstrap_replicates",
            minimum=0,
        )
        if any(
            not 0.0 <= value < 1.0
            for value in result.prior_strengths
        ):
            raise ConfigError(
                "experiment.prior_strengths must lie in [0, 1)"
            )
        _reject_duplicates(
            result.prior_strengths,
            "experiment.prior_strengths",
        )
        return result


@dataclass(frozen=True, slots=True)
class ResponseModelSection:
    beta: float = 1.0
    decision_noise: float = 1.0
    rank_scale: float = 0.35
    default_scale: float = 0.80
    suggestion_scale: float = 0.65
    susceptibility_levels: tuple[float, ...] = (0.15, 0.45, 0.85)
    minimum_matched_probability: float = 0.05

    @classmethod
    def parse(cls, raw: Mapping[str, Any]) -> "ResponseModelSection":
        allowed = {
            "beta",
            "decision_noise",
            "rank_scale",
            "default_scale",
            "suggestion_scale",
            "susceptibility_levels",
            "minimum_matched_probability",
        }
        _only_keys("response_model", raw, allowed)
        prepared = dict(raw)
        if "susceptibility_levels" in prepared:
            levels = prepared["susceptibility_levels"]
            if not isinstance(levels, list) or not levels:
                raise ConfigError(
                    "response_model.susceptibility_levels must be a non-empty array"
                )
            prepared["susceptibility_levels"] = tuple(
                _require_finite_number(
                    value,
                    "response_model.susceptibility_levels",
                )
                for value in levels
            )
        result = cls(**prepared)
        beta = _require_finite_number(result.beta, "response_model.beta")
        noise = _require_finite_number(
            result.decision_noise,
            "response_model.decision_noise",
        )
        if beta <= 0 or noise <= 0:
            raise ConfigError("beta and decision_noise must be positive")
        for name in ("rank_scale", "default_scale", "suggestion_scale"):
            if _require_finite_number(
                getattr(result, name),
                f"response_model.{name}",
            ) < 0:
                raise ConfigError(f"response_model.{name} must be non-negative")
        if not result.susceptibility_levels or any(
            x < 0 for x in result.susceptibility_levels
        ):
            raise ConfigError("susceptibility levels must be non-negative")
        _reject_duplicates(
            result.susceptibility_levels,
            "response_model.susceptibility_levels",
        )
        minimum = _require_finite_number(
            result.minimum_matched_probability,
            "response_model.minimum_matched_probability",
        )
        if not 0 < minimum < 0.5:
            raise ConfigError(
                "minimum_matched_probability must lie strictly between 0 and 0.5"
            )
        return result


@dataclass(frozen=True, slots=True)
class InferenceSection:
    training_interactions: int = 512
    fit_steps: int = 600
    learning_rate: float = 0.04
    l2: float = 0.001
    calibration: str = "temperature"

    @classmethod
    def parse(cls, raw: Mapping[str, Any]) -> "InferenceSection":
        allowed = {
            "training_interactions",
            "fit_steps",
            "learning_rate",
            "l2",
            "calibration",
        }
        _only_keys("inference", raw, allowed)
        result = cls(**raw)
        _require_integer(
            result.training_interactions,
            "inference.training_interactions",
            minimum=1,
        )
        _require_integer(
            result.fit_steps,
            "inference.fit_steps",
            minimum=1,
        )
        learning_rate = _require_finite_number(
            result.learning_rate,
            "inference.learning_rate",
        )
        l2 = _require_finite_number(result.l2, "inference.l2")
        if learning_rate <= 0 or l2 < 0:
            raise ConfigError("learning_rate must be positive and l2 non-negative")
        if not isinstance(result.calibration, str):
            raise ConfigError("inference.calibration must be a string")
        if result.calibration not in {"none", "temperature"}:
            raise ConfigError("inference.calibration must be 'none' or 'temperature'")
        return result


@dataclass(frozen=True, slots=True)
class ThresholdSection:
    materially_wrong_mass: float = 0.50
    laundered_confidence_gain: float = 0.25
    shadow_equivalence_tolerance: float = 0.05
    false_stability_tolerance: float = 0.02
    direction_tolerance: float = 1e-9
    ranking_tie_tolerance: float = 1e-6

    @classmethod
    def parse(cls, raw: Mapping[str, Any]) -> "ThresholdSection":
        allowed = {
            "materially_wrong_mass",
            "laundered_confidence_gain",
            "shadow_equivalence_tolerance",
            "false_stability_tolerance",
            "direction_tolerance",
            "ranking_tie_tolerance",
        }
        _only_keys("thresholds", raw, allowed)
        result = cls(**raw)
        wrong_mass = _require_finite_number(
            result.materially_wrong_mass,
            "thresholds.materially_wrong_mass",
        )
        lcg = _require_finite_number(
            result.laundered_confidence_gain,
            "thresholds.laundered_confidence_gain",
        )
        shadow_equivalence = _require_finite_number(
            result.shadow_equivalence_tolerance,
            "thresholds.shadow_equivalence_tolerance",
        )
        false_stability = _require_finite_number(
            result.false_stability_tolerance,
            "thresholds.false_stability_tolerance",
        )
        direction = _require_finite_number(
            result.direction_tolerance,
            "thresholds.direction_tolerance",
        )
        ranking = _require_finite_number(
            result.ranking_tie_tolerance,
            "thresholds.ranking_tie_tolerance",
        )
        if not 0 <= wrong_mass <= 1:
            raise ConfigError("materially_wrong_mass must lie in [0, 1]")
        if lcg < 0:
            raise ConfigError("laundered_confidence_gain must be non-negative")
        if not 0 <= shadow_equivalence <= 1:
            raise ConfigError(
                "shadow_equivalence_tolerance must lie in [0, 1]"
            )
        if not 0 <= false_stability <= 1:
            raise ConfigError(
                "false_stability_tolerance must lie in [0, 1]"
            )
        if direction < 0 or ranking < 0:
            raise ConfigError("tolerances must be non-negative")
        return result


@dataclass(frozen=True, slots=True)
class SensitivitySection:
    decision_noise_values: tuple[float, ...] = (0.6, 1.0, 1.6)
    presentation_multipliers: tuple[float, ...] = (0.5, 1.0, 1.5)
    rank_multipliers: tuple[float, ...] = (1.0,)
    default_multipliers: tuple[float, ...] = (1.0,)
    suggestion_multipliers: tuple[float, ...] = (1.0,)
    profile_strength_values: tuple[float, ...] = (0.65, 0.80, 0.90)
    prior_uncertainty_values: tuple[float, ...] = (0.0,)
    trajectory_lengths: tuple[int, ...] = (4, 8, 12)
    response_model_families: tuple[str, ...] = ("random_utility",)
    rule_noise_values: tuple[float, ...] = (0.15,)
    phase_min_selection_cost: float = 0.0
    phase_max_aware_ece: float = 0.10
    phase_min_attribution_cost: float = 0.0
    phase_min_self_confirming_rate: float = 0.0

    @classmethod
    def parse(cls, raw: Mapping[str, Any]) -> "SensitivitySection":
        allowed = {
            "decision_noise_values",
            "presentation_multipliers",
            "rank_multipliers",
            "default_multipliers",
            "suggestion_multipliers",
            "profile_strength_values",
            "prior_uncertainty_values",
            "trajectory_lengths",
            "response_model_families",
            "rule_noise_values",
            "phase_min_selection_cost",
            "phase_max_aware_ece",
            "phase_min_attribution_cost",
            "phase_min_self_confirming_rate",
        }
        _only_keys("sensitivity", raw, allowed)
        prepared = dict(raw)
        for name in (
            "decision_noise_values",
            "presentation_multipliers",
            "rank_multipliers",
            "default_multipliers",
            "suggestion_multipliers",
            "profile_strength_values",
            "prior_uncertainty_values",
            "rule_noise_values",
        ):
            if name in prepared:
                values = prepared[name]
                if not isinstance(values, list) or not values:
                    raise ConfigError(f"sensitivity.{name} must be a non-empty array")
                prepared[name] = tuple(
                    _require_finite_number(
                        value,
                        f"sensitivity.{name}",
                    )
                    for value in values
                )
        if "response_model_families" in prepared:
            prepared["response_model_families"] = _tuple_of_strings(
                prepared["response_model_families"],
                "sensitivity.response_model_families",
            )
        if "trajectory_lengths" in prepared:
            lengths = prepared["trajectory_lengths"]
            if not isinstance(lengths, list) or not lengths:
                raise ConfigError(
                    "sensitivity.trajectory_lengths must be a non-empty array"
                )
            prepared["trajectory_lengths"] = tuple(
                _require_integer(
                    value,
                    "sensitivity.trajectory_lengths",
                    minimum=1,
                )
                for value in lengths
            )
        result = cls(**prepared)
        if any(value <= 0 for value in result.decision_noise_values):
            raise ConfigError("decision noise values must be positive")
        if any(value < 0 for value in result.presentation_multipliers):
            raise ConfigError("presentation multipliers must be non-negative")
        for name in (
            "rank_multipliers",
            "default_multipliers",
            "suggestion_multipliers",
        ):
            if any(value < 0 for value in getattr(result, name)):
                raise ConfigError(
                    f"{name.replace('_', ' ')} must be non-negative"
                )
        if any(not 0.5 <= value < 1 for value in result.profile_strength_values):
            raise ConfigError("profile strength values must lie in [0.5, 1)")
        if any(
            not 0.0 <= value < 1.0
            for value in result.prior_uncertainty_values
        ):
            raise ConfigError("prior uncertainty values must lie in [0, 1)")
        if any(value <= 0 for value in result.trajectory_lengths):
            raise ConfigError("trajectory lengths must be positive")
        if any(
            not 0.0 <= value <= 1.0
            for value in result.rule_noise_values
        ):
            raise ConfigError("rule noise values must lie in [0, 1]")
        for name in (
            "phase_min_selection_cost",
            "phase_max_aware_ece",
            "phase_min_attribution_cost",
            "phase_min_self_confirming_rate",
        ):
            value = _require_finite_number(
                getattr(result, name),
                f"sensitivity.{name}",
            )
            if name in {
                "phase_max_aware_ece",
                "phase_min_self_confirming_rate",
            } and not 0.0 <= value <= 1.0:
                raise ConfigError(f"sensitivity.{name} must lie in [0, 1]")
        unknown_families = sorted(
            set(result.response_model_families)
            - KNOWN_RESPONSE_MODEL_FAMILIES
        )
        if unknown_families:
            raise ConfigError(
                "unknown sensitivity response model families: "
                + ", ".join(unknown_families)
            )
        for name in (
            "decision_noise_values",
            "presentation_multipliers",
            "rank_multipliers",
            "default_multipliers",
            "suggestion_multipliers",
            "profile_strength_values",
            "prior_uncertainty_values",
            "trajectory_lengths",
            "response_model_families",
            "rule_noise_values",
        ):
            _reject_duplicates(getattr(result, name), f"sensitivity.{name}")
        return result


@dataclass(frozen=True, slots=True)
class ArtifactSection:
    retain_events: bool = True
    retain_prompts: bool = False
    checksum_manifest: bool = True

    @classmethod
    def parse(cls, raw: Mapping[str, Any]) -> "ArtifactSection":
        _only_keys(
            "artifacts",
            raw,
            {"retain_events", "retain_prompts", "checksum_manifest"},
        )
        result = cls(**raw)
        for name in ("retain_events", "retain_prompts", "checksum_manifest"):
            if not isinstance(getattr(result, name), bool):
                raise ConfigError(f"artifacts.{name} must be a Boolean")
        return result


@dataclass(frozen=True, slots=True)
class LLMSection:
    mode: str = "replay"
    responses_file: str = ""
    calibration: str = "temperature"
    calibration_users: int = 1
    model_role: str = "primary"
    model: str = ""
    reasoning_effort: str = ""
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str = "https://api.openai.com"
    allow_custom_base_url: bool = False
    timeout_seconds: float = 180.0
    max_retries: int = 4
    max_output_tokens: int = 4096
    max_requests: int = 100
    max_total_tokens: int = 500_000
    journal_dir: str = ""
    openrouter_upstream_provider: str = ""
    openrouter_allow_fallbacks: bool = False
    openrouter_require_parameters: bool = True
    openrouter_data_collection: str = "deny"
    openrouter_zdr: bool = False
    openrouter_http_referer: str = ""
    openrouter_app_title: str = "CAPE-Loop"

    @classmethod
    def parse(cls, raw: Mapping[str, Any]) -> "LLMSection":
        _only_keys(
            "llm",
            raw,
            {
                "mode",
                "responses_file",
                "calibration",
                "calibration_users",
                "model_role",
                "model",
                "reasoning_effort",
                "api_key_env",
                "base_url",
                "allow_custom_base_url",
                "timeout_seconds",
                "max_retries",
                "max_output_tokens",
                "max_requests",
                "max_total_tokens",
                "journal_dir",
                "openrouter_upstream_provider",
                "openrouter_allow_fallbacks",
                "openrouter_require_parameters",
                "openrouter_data_collection",
                "openrouter_zdr",
                "openrouter_http_referer",
                "openrouter_app_title",
            },
        )
        prepared = dict(raw)
        if prepared.get("mode") == "openrouter":
            prepared.setdefault("api_key_env", "OPENROUTER_API_KEY")
            prepared.setdefault("base_url", "https://openrouter.ai/api")
            prepared.setdefault("max_retries", 2)
        result = cls(**prepared)
        for name in (
            "mode",
            "responses_file",
            "calibration",
            "model_role",
            "model",
            "reasoning_effort",
            "api_key_env",
            "base_url",
            "journal_dir",
            "openrouter_upstream_provider",
            "openrouter_data_collection",
            "openrouter_http_referer",
            "openrouter_app_title",
        ):
            if not isinstance(getattr(result, name), str):
                raise ConfigError(f"llm.{name} must be a string")
        if result.mode not in KNOWN_LLM_MODES:
            raise ConfigError(
                f"llm.mode must be one of {sorted(KNOWN_LLM_MODES)}"
            )
        if result.calibration not in {"none", "temperature"}:
            raise ConfigError(
                "llm.calibration must be 'none' or 'temperature'"
            )
        _require_integer(
            result.calibration_users,
            "llm.calibration_users",
            minimum=1,
        )
        if result.model_role not in KNOWN_LLM_MODEL_ROLES:
            raise ConfigError(
                "llm.model_role must be one of "
                f"{sorted(KNOWN_LLM_MODEL_ROLES)}"
            )
        if (
            result.reasoning_effort
            and result.reasoning_effort not in KNOWN_REASONING_EFFORTS
        ):
            raise ConfigError(
                "llm.reasoning_effort must be empty or one of "
                f"{sorted(KNOWN_REASONING_EFFORTS)}"
            )
        if (
            result.reasoning_effort == "minimal"
            and result.mode != "openrouter"
        ):
            raise ConfigError(
                "llm.reasoning_effort = 'minimal' is supported only in "
                "OpenRouter mode"
            )
        if result.mode == "openrouter" and (
            not result.model
            or result.model != result.model.strip()
            or "/" not in result.model
            or result.model.startswith(("~", "/"))
            or result.model.endswith("/")
            or any(character.isspace() for character in result.model)
            or ":" in result.model
            or result.model.lower().endswith("-latest")
            or result.model.lower() == "openrouter/auto"
        ):
            raise ConfigError(
                "OpenRouter mode requires an explicit author/model slug; "
                "aliases, route variants, and openrouter/auto are not "
                "reproducible"
            )
        if not result.api_key_env or not (
            result.api_key_env[0].isascii()
            and result.api_key_env[0].isalpha()
            or result.api_key_env[0] == "_"
        ) or not all(
            character.isascii()
            and (character.isalnum() or character == "_")
            for character in result.api_key_env
        ):
            raise ConfigError(
                "llm.api_key_env must be a valid environment-variable name"
            )
        if (
            result.mode == "openrouter"
            and result.api_key_env
            in {
                "OPENAI_API_KEY",
                "ANTHROPIC_API_KEY",
                "GEMINI_API_KEY",
            }
        ):
            raise ConfigError(
                "OpenRouter mode requires a dedicated credential variable; "
                "a first-party provider key must never be sent to OpenRouter"
            )
        if (
            result.mode == "openai"
            and result.api_key_env == "OPENROUTER_API_KEY"
        ):
            raise ConfigError(
                "OpenAI mode requires a dedicated credential variable; "
                "OPENROUTER_API_KEY must never be sent to OpenAI"
            )
        parsed_base_url = urlsplit(result.base_url)
        if (
            parsed_base_url.scheme != "https"
            or not parsed_base_url.netloc
            or parsed_base_url.username is not None
            or parsed_base_url.password is not None
            or parsed_base_url.query
            or parsed_base_url.fragment
        ):
            raise ConfigError(
                "llm.base_url must be an HTTPS origin or HTTPS path"
            )
        if not isinstance(result.allow_custom_base_url, bool):
            raise ConfigError(
                "llm.allow_custom_base_url must be a Boolean"
            )
        if not result.allow_custom_base_url:
            if result.mode == "openrouter":
                official_origin = (
                    parsed_base_url.hostname == "openrouter.ai"
                    and parsed_base_url.port is None
                    and parsed_base_url.path.rstrip("/") == "/api"
                )
                if not official_origin:
                    raise ConfigError(
                        "OpenRouter mode requires the official "
                        "https://openrouter.ai/api path unless "
                        "llm.allow_custom_base_url = true"
                    )
            elif (
                parsed_base_url.hostname != "api.openai.com"
                or parsed_base_url.port is not None
                or parsed_base_url.path not in {"", "/"}
            ):
                raise ConfigError(
                    "llm.base_url must be the official "
                    "https://api.openai.com origin unless "
                    "llm.allow_custom_base_url = true"
                )
        default_credential = (
            "OPENROUTER_API_KEY"
            if result.mode == "openrouter"
            else "OPENAI_API_KEY"
        )
        official_hostname = (
            "openrouter.ai"
            if result.mode == "openrouter"
            else "api.openai.com"
        )
        if (
            result.allow_custom_base_url
            and parsed_base_url.hostname != official_hostname
            and result.api_key_env == default_credential
        ):
            raise ConfigError(
                "a custom llm.base_url requires a dedicated credential "
                "environment variable; set llm.api_key_env to a name other "
                f"than {default_credential}"
            )
        if _require_finite_number(
            result.timeout_seconds, "llm.timeout_seconds"
        ) <= 0:
            raise ConfigError("llm.timeout_seconds must be positive")
        _require_integer(result.max_retries, "llm.max_retries", minimum=0)
        for name in ("max_output_tokens", "max_requests", "max_total_tokens"):
            _require_integer(getattr(result, name), f"llm.{name}", minimum=1)
        for name in (
            "openrouter_allow_fallbacks",
            "openrouter_require_parameters",
            "openrouter_zdr",
        ):
            if not isinstance(getattr(result, name), bool):
                raise ConfigError(f"llm.{name} must be a Boolean")
        if result.openrouter_data_collection not in {"allow", "deny"}:
            raise ConfigError(
                "llm.openrouter_data_collection must be 'allow' or 'deny'"
            )
        provider_slug = result.openrouter_upstream_provider
        if provider_slug and (
            not provider_slug[0].isalnum()
            or any(
                not (
                    character.isalnum()
                    or character in "._/-"
                )
                for character in provider_slug
            )
            or "//" in provider_slug
            or ".." in provider_slug
        ):
            raise ConfigError(
                "llm.openrouter_upstream_provider must be one exact "
                "OpenRouter provider slug"
            )
        if "\r" in result.openrouter_app_title or "\n" in (
            result.openrouter_app_title
        ):
            raise ConfigError(
                "llm.openrouter_app_title must not contain newlines"
            )
        if len(result.openrouter_app_title) > 200:
            raise ConfigError(
                "llm.openrouter_app_title must contain at most 200 characters"
            )
        if result.openrouter_http_referer:
            referer = urlsplit(result.openrouter_http_referer)
            if (
                referer.scheme not in {"http", "https"}
                or not referer.netloc
                or referer.username is not None
                or referer.password is not None
            ):
                raise ConfigError(
                    "llm.openrouter_http_referer must be an absolute "
                    "HTTP(S) URL"
                )
            if "\r" in result.openrouter_http_referer or "\n" in (
                result.openrouter_http_referer
            ):
                raise ConfigError(
                    "llm.openrouter_http_referer must not contain newlines"
                )
        return result


@dataclass(frozen=True, slots=True)
class AppConfig:
    schema_version: int = SCHEMA_VERSION
    run: RunSection = field(default_factory=RunSection)
    experiment: ExperimentSection = field(default_factory=ExperimentSection)
    response_model: ResponseModelSection = field(default_factory=ResponseModelSection)
    inference: InferenceSection = field(default_factory=InferenceSection)
    thresholds: ThresholdSection = field(default_factory=ThresholdSection)
    sensitivity: SensitivitySection = field(default_factory=SensitivitySection)
    llm: LLMSection = field(default_factory=LLMSection)
    artifacts: ArtifactSection = field(default_factory=ArtifactSection)

    @classmethod
    def parse(cls, raw: Mapping[str, Any]) -> "AppConfig":
        _only_keys(
            "root",
            raw,
            {
                "schema_version",
                "run",
                "experiment",
                "response_model",
                "inference",
                "thresholds",
                "sensitivity",
                "llm",
                "artifacts",
            },
        )
        version = raw.get("schema_version")
        if (
            isinstance(version, bool)
            or not isinstance(version, int)
            or version != SCHEMA_VERSION
        ):
            raise ConfigError(
                f"schema_version must be {SCHEMA_VERSION}; received {version!r}"
            )
        sections = {}
        parsers = {
            "run": RunSection,
            "experiment": ExperimentSection,
            "response_model": ResponseModelSection,
            "inference": InferenceSection,
            "thresholds": ThresholdSection,
            "sensitivity": SensitivitySection,
            "llm": LLMSection,
            "artifacts": ArtifactSection,
        }
        for name, parser in parsers.items():
            value = raw.get(name, {})
            if not isinstance(value, Mapping):
                raise ConfigError(f"[{name}] must be a TOML table")
            sections[name] = parser.parse(value)
        result = cls(schema_version=SCHEMA_VERSION, **sections)
        uses_llm = any(
            updater_id.startswith("llm_")
            for updater_id in result.experiment.updaters
        )
        if (
            uses_llm
            and result.llm.mode == "replay"
            and not result.llm.responses_file
        ):
            raise ConfigError(
                "LLM replay updaters require llm.responses_file"
            )
        result.validate_experiment_contract()
        return result

    def validate_experiment_contract(self) -> None:
        """Reject generic TOML fields that a selected runner would ignore."""

        experiment = self.experiment
        if experiment.kind == "provenance_audit":
            allowed_mechanisms = {"balanced", "restricted", "default", "suggested"}
            if not set(experiment.mechanisms) <= allowed_mechanisms:
                raise ConfigError(
                    "provenance_audit mechanisms must be drawn from "
                    "balanced, restricted, default, and suggested"
                )
            if experiment.policies != ("balanced",):
                raise ConfigError(
                    "provenance_audit requires policies = ['balanced']; "
                    "policy variation is represented by mechanisms"
                )
            if (
                experiment.trajectories_per_cell != 1
                or experiment.turns != 1
            ):
                raise ConfigError(
                    "provenance_audit is one-step and requires "
                    "trajectories_per_cell = 1 and turns = 1"
                )
            return

        if experiment.prior_strengths != (0.0,):
            raise ConfigError(
                "experiment.prior_strengths is an Experiment A factor; "
                "other experiment kinds require [0.0]"
            )

        expected_mechanisms = {"ranking", "default", "suggestion"}
        if set(experiment.mechanisms) != expected_mechanisms:
            raise ConfigError(
                f"{experiment.kind} requires mechanisms = "
                "['ranking', 'default', 'suggestion']"
            )
        if experiment.response_modes != ("naturally_sampled",):
            raise ConfigError(
                f"{experiment.kind} requires "
                "response_modes = ['naturally_sampled']"
            )
        if experiment.kind == "closed_loop":
            return

        if experiment.kind == "evaluation_validity":
            if set(experiment.policies) != {
                "balanced",
                "fixed_bias",
                "soft_profile_conditioned",
            }:
                raise ConfigError(
                    "evaluation_validity requires balanced, fixed_bias, and "
                    "soft_profile_conditioned policies"
                )
            if len(experiment.updaters) < 2:
                raise ConfigError(
                    "evaluation_validity requires at least two updaters "
                    "for ranking comparisons"
                )
            if experiment.bootstrap_replicates <= 0:
                raise ConfigError(
                    "evaluation_validity requires positive "
                    "bootstrap_replicates"
                )
            return

        if experiment.kind == "sensitivity":
            if any(
                updater_id.startswith("llm_")
                for updater_id in experiment.updaters
            ):
                raise ConfigError(
                    "sensitivity does not support adaptive LLM execution; "
                    "use deterministic/native updaters for the grid"
                )
            if set(experiment.policies) != {
                "balanced",
                "soft_profile_conditioned",
            }:
                raise ConfigError(
                    "sensitivity requires balanced and "
                    "soft_profile_conditioned policies"
                )
            if experiment.turns != 1:
                raise ConfigError(
                    "sensitivity experiment.turns must be 1; trajectory lengths "
                    "are declared exclusively by "
                    "[sensitivity].trajectory_lengths"
                )
            if experiment.bootstrap_replicates != 0:
                raise ConfigError(
                    "sensitivity does not bootstrap each grid point; "
                    "set bootstrap_replicates = 0"
                )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def validated(self) -> "AppConfig":
        """Re-parse a programmatically constructed config through all checks."""

        return AppConfig.parse(json.loads(self.canonical_json()))


def load_config(path: str | Path) -> AppConfig:
    """Load and validate an application configuration."""

    config_path = Path(path)
    try:
        with config_path.open("rb") as handle:
            raw = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"cannot load {config_path}: {exc}") from exc
    return AppConfig.parse(raw)
