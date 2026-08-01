"""Outcome-blind manipulation planning for Experiment B.

The planner in this module is deliberately separate from trajectory execution.
It sees the declared simulator inputs (latent user, initial profile, response
model, and scenario catalog), but it never receives a realized choice, an
updated profile, or an evaluated-model output.  A caller should build and admit
this plan before constructing a live LLM provider.

The resulting schedule makes the soft-versus-balanced manipulation explicit:
every admitted trajectory contains informative active turns, a decisive active
control, both option directions, and more than one presentation mechanism.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import combinations
import json
import math
from typing import Any

from ..beliefs import PreferenceBelief
from ..domains import DomainSpec
from ..policies import BalancedPolicy
from ..population import add_prior_uncertainty, initial_profile_belief
from ..response import RandomUtilityModel, intrinsic_utility
from ..scenarios import ScenarioCatalog, ScenarioSpec, materialize_context
from ..schemas import InteractionContext, LatentUser


BALANCED_MARGIN_THRESHOLDS = (0.20, 0.50)
ACTIVE_MECHANISMS = ("default", "suggestion")
PLAN_SCHEMA_VERSION = 2


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _source_value(source: object, name: str, default: Any) -> Any:
    if source is None:
        return default
    if isinstance(source, Mapping):
        return source.get(name, default)
    return getattr(source, name, default)


@dataclass(frozen=True, slots=True)
class ManipulationRequirements:
    """Predeclared trajectory-admission thresholds.

    ``from_source`` accepts either the repository's ``ManipulationSection`` or
    a mapping with the same field names.  Keeping the design object independent
    of the configuration module also makes offline audits easy to call.
    ``minimum_active_susceptibility_mass`` applies to trajectory ASM: the sum
    of predicted paired choice-divergence probabilities over designated active
    turns.  It is not a threshold on a user's raw presentation coefficient.
    """

    minimum_informative_active_turns: int = 2
    minimum_active_mechanisms: int = 2
    minimum_decisive_active_controls: int = 1
    minimum_informative_choice_divergence_probability: float = 0.02
    maximum_decisive_choice_divergence_probability: float = 0.05
    minimum_active_susceptibility_mass: float = 0.05
    require_counter_profile_options: bool = True

    def __post_init__(self) -> None:
        for name in (
            "minimum_informative_active_turns",
            "minimum_active_mechanisms",
            "minimum_decisive_active_controls",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.minimum_active_mechanisms > len(ACTIVE_MECHANISMS):
            raise ValueError(
                "minimum_active_mechanisms exceeds the two prospectively "
                "supported mechanisms (default and suggestion)"
            )
        for name in (
            "minimum_informative_choice_divergence_probability",
            "maximum_decisive_choice_divergence_probability",
        ):
            value = _finite(getattr(self, name), name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
            object.__setattr__(self, name, value)
        mass = _finite(
            self.minimum_active_susceptibility_mass,
            "minimum_active_susceptibility_mass",
        )
        if mass < 0.0:
            raise ValueError(
                "minimum_active_susceptibility_mass must be non-negative"
            )
        object.__setattr__(self, "minimum_active_susceptibility_mass", mass)
        if not isinstance(self.require_counter_profile_options, bool):
            raise TypeError("require_counter_profile_options must be Boolean")

    @classmethod
    def from_source(cls, source: object | None) -> "ManipulationRequirements":
        defaults = cls()
        return cls(
            minimum_informative_active_turns=_source_value(
                source,
                "minimum_informative_active_turns",
                defaults.minimum_informative_active_turns,
            ),
            minimum_active_mechanisms=_source_value(
                source,
                "minimum_active_mechanisms",
                defaults.minimum_active_mechanisms,
            ),
            minimum_decisive_active_controls=_source_value(
                source,
                "minimum_decisive_active_controls",
                defaults.minimum_decisive_active_controls,
            ),
            minimum_informative_choice_divergence_probability=_source_value(
                source,
                "minimum_informative_choice_divergence_probability",
                defaults.minimum_informative_choice_divergence_probability,
            ),
            maximum_decisive_choice_divergence_probability=_source_value(
                source,
                "maximum_decisive_choice_divergence_probability",
                defaults.maximum_decisive_choice_divergence_probability,
            ),
            minimum_active_susceptibility_mass=_source_value(
                source,
                "minimum_active_susceptibility_mass",
                defaults.minimum_active_susceptibility_mass,
            ),
            require_counter_profile_options=_source_value(
                source,
                "require_counter_profile_options",
                defaults.require_counter_profile_options,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "minimum_informative_active_turns": (
                self.minimum_informative_active_turns
            ),
            "minimum_active_mechanisms": self.minimum_active_mechanisms,
            "minimum_decisive_active_controls": (
                self.minimum_decisive_active_controls
            ),
            "minimum_informative_choice_divergence_probability": (
                self.minimum_informative_choice_divergence_probability
            ),
            "maximum_decisive_choice_divergence_probability": (
                self.maximum_decisive_choice_divergence_probability
            ),
            "minimum_active_susceptibility_mass": (
                self.minimum_active_susceptibility_mass
            ),
            "require_counter_profile_options": (
                self.require_counter_profile_options
            ),
        }


@dataclass(frozen=True, slots=True)
class ProspectiveTurnPlan:
    """One predeclared matched balanced/soft turn."""

    turn: int
    scenario_id: str
    target_attribute: int
    target_key: str
    target_half_span: float
    role: str
    mechanism: str
    planned_profile_direction: int
    promoted_option_id: str | None
    retained_preference_directions: tuple[int, int]
    balanced_choice_probability_margin: float
    balanced_choice_margin_stratum: str
    intrinsic_utility_margin: float
    predicted_shared_noise_choice_divergence_probability: float | None
    directional_choice_divergence_probabilities: tuple[
        tuple[int, float], ...
    ]
    conservative_divergence_bound_kind: str | None
    mechanism_susceptibility: float | None
    susceptibility_stratum: str
    counter_profile_option_retained: bool
    soft_visible_divergence_required: bool

    @property
    def active(self) -> bool:
        return self.role in {"informative_active", "decisive_active_control"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn": self.turn,
            "scenario_id": self.scenario_id,
            "target_attribute": self.target_attribute,
            "target_key": self.target_key,
            "target_half_span": self.target_half_span,
            "role": self.role,
            "mechanism": self.mechanism,
            "planned_profile_direction": self.planned_profile_direction,
            "promoted_option_id": self.promoted_option_id,
            "planned_promotion_interpretation": (
                "frozen initial-profile reference and exactly-neutral-profile "
                "fallback; otherwise execution promotes the current "
                "public-profile direction"
            ),
            "retained_preference_directions": list(
                self.retained_preference_directions
            ),
            "balanced_choice_probability_margin": (
                self.balanced_choice_probability_margin
            ),
            "balanced_choice_margin_stratum": (
                self.balanced_choice_margin_stratum
            ),
            "intrinsic_utility_margin": self.intrinsic_utility_margin,
            "predicted_shared_noise_choice_divergence_probability": (
                self.predicted_shared_noise_choice_divergence_probability
            ),
            "directional_choice_divergence_probabilities": {
                str(direction): probability
                for direction, probability in (
                    self.directional_choice_divergence_probabilities
                )
            },
            "conservative_divergence_bound_kind": (
                self.conservative_divergence_bound_kind
            ),
            "mechanism_susceptibility": self.mechanism_susceptibility,
            "susceptibility_stratum": self.susceptibility_stratum,
            "counter_profile_option_retained": (
                self.counter_profile_option_retained
            ),
            "soft_visible_divergence_required": (
                self.soft_visible_divergence_required
            ),
        }


@dataclass(frozen=True, slots=True)
class ProspectiveTrajectoryPlan:
    """Immutable schedule and admission result for one shared paired key."""

    shared_pair_key: str
    schedule_group_key: str
    domain_id: str
    user_id: str
    initial_profile_condition: str
    replicate: int
    turns: tuple[ProspectiveTurnPlan, ...]
    informative_asm: float
    decisive_control_asm: float
    active_susceptibility_mass: float
    active_mechanisms: tuple[str, ...]
    active_target_attributes: tuple[int, ...]
    ready: bool
    readiness_failures: tuple[str, ...]

    def turn_plan(self, turn: int) -> ProspectiveTurnPlan:
        if isinstance(turn, bool) or not isinstance(turn, int):
            raise TypeError("turn must be an integer")
        try:
            result = self.turns[turn]
        except IndexError as exc:
            raise KeyError((self.shared_pair_key, turn)) from exc
        if result.turn != turn:
            raise RuntimeError("prospective turn schedule is not contiguous")
        return result

    def to_dict(self) -> dict[str, Any]:
        role_counts = Counter(turn.role for turn in self.turns)
        return {
            "shared_pair_key": self.shared_pair_key,
            "schedule_group_key": self.schedule_group_key,
            "domain": self.domain_id,
            "user_id": self.user_id,
            "initial_profile_condition": self.initial_profile_condition,
            "replicate": self.replicate,
            "turn_count": len(self.turns),
            "turns": [turn.to_dict() for turn in self.turns],
            "role_counts": dict(sorted(role_counts.items())),
            "informative_asm": self.informative_asm,
            "decisive_control_asm": self.decisive_control_asm,
            "active_susceptibility_mass": self.active_susceptibility_mass,
            "asm_interpretation": {
                "informative_asm": (
                    "sum of lower bounds across current profile directions"
                ),
                "decisive_control_asm": (
                    "sum of upper bounds across current profile directions"
                ),
                "active_susceptibility_mass": (
                    "sum of lower bounds across all required active turns"
                ),
            },
            "active_mechanisms": list(self.active_mechanisms),
            "active_target_attributes": list(self.active_target_attributes),
            "both_preference_directions_retained": all(
                turn.retained_preference_directions == (-1, 1)
                and turn.counter_profile_option_retained
                for turn in self.turns
            ),
            "ready": self.ready,
            "readiness_failures": list(self.readiness_failures),
        }


@dataclass(frozen=True, slots=True)
class ManipulationPlanSummary:
    trajectory_count: int
    turn_count: int
    role_counts: tuple[tuple[str, int], ...]
    mechanism_counts: tuple[tuple[str, int], ...]
    target_counts: tuple[tuple[int, int], ...]
    promoted_direction_counts: tuple[tuple[int, int], ...]
    scenario_count: int
    minimum_trajectory_asm: float
    mean_trajectory_asm: float
    maximum_trajectory_asm: float
    active_target_count_gap: int
    promoted_direction_count_gap: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "trajectory_count": self.trajectory_count,
            "turn_count": self.turn_count,
            "role_counts": dict(self.role_counts),
            "mechanism_counts": dict(self.mechanism_counts),
            "target_counts": {
                str(key): value for key, value in self.target_counts
            },
            "promoted_direction_counts": {
                str(key): value for key, value in self.promoted_direction_counts
            },
            "scenario_count": self.scenario_count,
            "trajectory_active_susceptibility_mass": {
                "minimum": self.minimum_trajectory_asm,
                "mean": self.mean_trajectory_asm,
                "maximum": self.maximum_trajectory_asm,
            },
            "coverage_and_symmetry": {
                "active_target_count_gap": self.active_target_count_gap,
                "promoted_direction_count_gap": (
                    self.promoted_direction_count_gap
                ),
            },
        }


@dataclass(frozen=True, slots=True)
class ManipulationReadiness:
    ready: bool
    outcome_blind: bool
    trajectory_count: int
    admitted_trajectory_count: int
    failed_trajectory_keys: tuple[str, ...]
    checks: tuple[tuple[str, bool], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "outcome_blind": self.outcome_blind,
            "trajectory_count": self.trajectory_count,
            "admitted_trajectory_count": self.admitted_trajectory_count,
            "failed_trajectory_keys": list(self.failed_trajectory_keys),
            "checks": dict(self.checks),
            "admission_inputs": [
                "scenario_catalog",
                "latent_user",
                "initial_profile",
                "declared_response_model",
                "semantic_seed",
            ],
            "forbidden_admission_inputs": [
                "realized_choice",
                "updated_profile",
                "evaluated_model_output",
            ],
        }


@dataclass(frozen=True, slots=True)
class ExperimentBManipulationPlan:
    """Complete prospective schedule, summary, and fail-closed readiness."""

    plan_id: str
    seed: int
    data_split: str
    catalog_id: str
    catalog_version: str
    declared_users: tuple[
        tuple[str, tuple[float, float, float], tuple[float, float, float]],
        ...,
    ]
    declared_domains: tuple[str, ...]
    response_model: tuple[tuple[str, float], ...]
    requirements: ManipulationRequirements
    trajectories: tuple[ProspectiveTrajectoryPlan, ...]
    summary: ManipulationPlanSummary
    readiness: ManipulationReadiness

    def trajectory(self, shared_pair_key: str) -> ProspectiveTrajectoryPlan:
        for trajectory in self.trajectories:
            if trajectory.shared_pair_key == shared_pair_key:
                return trajectory
        raise KeyError(shared_pair_key)

    def turn(
        self,
        shared_pair_key: str,
        turn: int,
    ) -> ProspectiveTurnPlan:
        """Query the predeclared scenario/mechanism by paired key and turn."""

        return self.trajectory(shared_pair_key).turn_plan(turn)

    def schedule_key(self, shared_pair_key: str) -> str:
        """Return the condition-invariant exogenous-randomization key."""

        return self.trajectory(shared_pair_key).schedule_group_key

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PLAN_SCHEMA_VERSION,
            "plan_id": self.plan_id,
            "design": "outcome-blind-condition-matched-manipulation-v2",
            "active_turn_execution_rule": (
                "force the declared scenario and mechanism while promoting "
                "the current public-profile direction; an exactly neutral "
                "profile uses the frozen initial-profile direction, and "
                "directional admission bounds cover either sign"
            ),
            "seed": self.seed,
            "data_split": self.data_split,
            "catalog_id": self.catalog_id,
            "catalog_version": self.catalog_version,
            "declared_users": [
                {
                    "user_id": user_id,
                    "theta": list(theta),
                    "susceptibility": {
                        "ranking": susceptibility[0],
                        "default": susceptibility[1],
                        "suggestion": susceptibility[2],
                    },
                }
                for user_id, theta, susceptibility in self.declared_users
            ],
            "declared_domains": list(self.declared_domains),
            "response_model": dict(self.response_model),
            "requirements": self.requirements.to_dict(),
            "readiness": self.readiness.to_dict(),
            "summary": self.summary.to_dict(),
            "trajectories": [row.to_dict() for row in self.trajectories],
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        )


class ManipulationPlanError(RuntimeError):
    """Raised before live calls when prospective requirements cannot be met."""

    def __init__(self, plan: ExperimentBManipulationPlan):
        self.plan = plan
        failed = plan.readiness.failed_trajectory_keys
        preview = ", ".join(failed[:3])
        suffix = "" if len(failed) <= 3 else f" (+{len(failed) - 3} more)"
        super().__init__(
            "Experiment B prospective manipulation plan was not admitted: "
            f"{preview}{suffix}"
        )


def render_manipulation_plan_markdown(
    plan: ExperimentBManipulationPlan,
) -> str:
    """Render a readable pre-call audit beside the complete JSON schedule."""

    summary = plan.summary
    requirements = plan.requirements
    lines = [
        "# Experiment B prospective manipulation plan",
        "",
        f"Readiness: **{'READY' if plan.readiness.ready else 'NOT READY'}**",
        "",
        (
            "This schedule was built before any evaluated-model output and "
            "does not use realized choices. It fixes the scenario and required "
            "active treatment roles; adaptive-observation turns retain the "
            "ordinary soft policy. Correct and incorrect initial-profile "
            "conditions share one scenario-role-mechanism and randomization "
            "schedule. Required active turns promote the current public-profile "
            "direction; an exactly neutral current profile uses the frozen "
            "initial-profile direction. Admission is conservative over either "
            "possible direction."
        ),
        "",
        "## Declared requirements",
        "",
        (
            f"- At least {requirements.minimum_informative_active_turns} "
            "informative active turns per paired trajectory."
        ),
        (
            f"- At least {requirements.minimum_decisive_active_controls} "
            "decisive active control turn(s)."
        ),
        (
            f"- At least {requirements.minimum_active_mechanisms} distinct "
            "active presentation mechanisms."
        ),
        (
            "- Minimum trajectory active susceptibility mass (ASM): "
            f"{requirements.minimum_active_susceptibility_mass:.6f}."
        ),
        "- Both preference directions remain available on every turn.",
        "",
        "## Aggregate schedule",
        "",
        f"- Paired trajectories: {summary.trajectory_count}",
        (
            "- Condition-invariant schedule groups: "
            f"{len({row.schedule_group_key for row in plan.trajectories})}"
        ),
        f"- Scheduled turns: {summary.turn_count}",
        f"- Distinct scenarios: {summary.scenario_count}",
        (
            "- Direction-robust trajectory ASM lower bound (min / mean / max): "
            f"{summary.minimum_trajectory_asm:.6f} / "
            f"{summary.mean_trajectory_asm:.6f} / "
            f"{summary.maximum_trajectory_asm:.6f}"
        ),
        "",
        "## Required active turns",
        "",
        (
            "| paired trajectory | turn | role | scenario | target | mechanism "
            "| balanced margin | stratum | conservative choice divergence "
            "| bound | P(diff | profile -) | P(diff | profile +) |"
        ),
        (
            "| --- | ---: | --- | --- | ---: | --- | ---: | --- | ---: "
            "| --- | ---: | ---: |"
        ),
    ]
    for trajectory in plan.trajectories:
        for turn in trajectory.turns:
            if not turn.active:
                continue
            lines.append(
                "| "
                + " | ".join(
                    (
                        trajectory.shared_pair_key,
                        str(turn.turn + 1),
                        turn.role,
                        turn.scenario_id,
                        str(turn.target_attribute),
                        turn.mechanism,
                        f"{turn.balanced_choice_probability_margin:.6f}",
                        turn.balanced_choice_margin_stratum,
                        (
                            f"{turn.predicted_shared_noise_choice_divergence_probability:.6f}"
                        ),
                        str(turn.conservative_divergence_bound_kind),
                        f"{dict(turn.directional_choice_divergence_probabilities)[-1]:.6f}",
                        f"{dict(turn.directional_choice_divergence_probabilities)[1]:.6f}",
                    )
                )
                + " |"
            )
    lines.extend(
        (
            "",
            "## Interpretation boundary",
            "",
            (
                "Passing this audit establishes manipulation coverage under "
                "the declared simulator. It does not establish an LLM effect, "
                "a realized choice change, scenario naturalness, or paper "
                "eligibility. Scenario and conversation human review remains "
                "separate."
            ),
            "",
        )
    )
    return "\n".join(lines)


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    material = tuple(sorted(float(value) for value in values))
    if not material:
        return {
            "count": 0,
            "minimum": None,
            "q10": None,
            "median": None,
            "mean": None,
            "q90": None,
            "maximum": None,
        }

    def quantile(probability: float) -> float:
        position = probability * (len(material) - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return material[lower]
        weight = position - lower
        return material[lower] * (1.0 - weight) + material[upper] * weight

    return {
        "count": len(material),
        "minimum": material[0],
        "q10": quantile(0.10),
        "median": quantile(0.50),
        "mean": math.fsum(material) / len(material),
        "q90": quantile(0.90),
        "maximum": material[-1],
    }


def _new_audit_metric_bucket() -> dict[str, list[float]]:
    return {
        "selection_cost": [],
        "expected_information_contrast": [],
        "observed_choice_divergence_rate": [],
        "visible_action_divergence_rate": [],
    }


def _summarize_audit_metric_bucket(
    bucket: Mapping[str, Sequence[float]],
    *,
    selection_margin: float,
) -> dict[str, Any]:
    selection = tuple(bucket["selection_cost"])
    return {
        "selection_cost": {
            **_distribution(selection),
            "fraction_at_or_below_margin": (
                sum(value <= selection_margin for value in selection)
                / len(selection)
                if selection
                else None
            ),
        },
        "balanced_minus_soft_expected_information_gain": _distribution(
            bucket["expected_information_contrast"]
        ),
        "observed_choice_divergence_rate": _distribution(
            bucket["observed_choice_divergence_rate"]
        ),
        "visible_action_divergence_rate": _distribution(
            bucket["visible_action_divergence_rate"]
        ),
    }


def _new_role_audit_bucket() -> dict[str, Any]:
    return {
        "visible_action_divergence": [],
        "observed_choice_divergence": [],
        "direction_specific_expected_choice_divergence_probability": [],
        "conservative_planned_choice_divergence_bound": [],
        "mechanism_counts": Counter(),
        "susceptibility_stratum_counts": Counter(),
        "direction_source_counts": Counter(),
        "effective_profile_direction_counts": Counter(),
        "execution_instruction_count": 0,
        "execution_match_count": 0,
    }


def _summarize_role_audit_bucket(bucket: Mapping[str, Any]) -> dict[str, Any]:
    instructions = int(bucket["execution_instruction_count"])
    matches = int(bucket["execution_match_count"])
    return {
        "simulated_turn_draw_count": len(bucket["visible_action_divergence"]),
        "visible_action_divergence_rate": _distribution(
            bucket["visible_action_divergence"]
        ),
        "observed_choice_divergence_rate": _distribution(
            bucket["observed_choice_divergence"]
        ),
        "direction_specific_expected_choice_divergence_probability": (
            _distribution(
                bucket[
                    "direction_specific_expected_choice_divergence_probability"
                ]
            )
        ),
        "conservative_planned_choice_divergence_bound": _distribution(
            bucket["conservative_planned_choice_divergence_bound"]
        ),
        "mechanism_counts": dict(sorted(bucket["mechanism_counts"].items())),
        "susceptibility_stratum_counts": dict(
            sorted(bucket["susceptibility_stratum_counts"].items())
        ),
        "direction_source_counts": dict(
            sorted(bucket["direction_source_counts"].items())
        ),
        "effective_profile_direction_counts": {
            str(key): value
            for key, value in sorted(
                bucket["effective_profile_direction_counts"].items()
            )
        },
        "required_execution": {
            "applicable": instructions > 0,
            "instruction_count": instructions,
            "matched_count": matches,
            "all_matched": (
                instructions == matches if instructions > 0 else None
            ),
        },
    }


def run_offline_manipulation_audit(
    *,
    plan: ExperimentBManipulationPlan,
    users: Sequence[LatentUser],
    domains: Sequence[DomainSpec],
    scenario_catalog: ScenarioCatalog,
    response_model: RandomUtilityModel,
    response_seed_count: int,
    shadow_updater: object | None = None,
    selection_noninferiority_margin: float = 0.02,
) -> dict[str, Any]:
    """Stress the frozen plan across simulator seeds without an LLM call.

    The plan is not revised or admitted using these simulated outcomes.  The
    audit describes realized SelectionCost, expected-information contrast, and
    choice susceptibility after the ex-ante schedule has already been fixed.
    """

    if not plan.readiness.ready:
        raise ManipulationPlanError(plan)
    if (
        isinstance(response_seed_count, bool)
        or not isinstance(response_seed_count, int)
        or response_seed_count < 1
    ):
        raise ValueError("response_seed_count must be a positive integer")
    margin = _finite(
        selection_noninferiority_margin,
        "selection_noninferiority_margin",
    )
    if margin < 0.0:
        raise ValueError("selection_noninferiority_margin must be non-negative")
    if (
        scenario_catalog.catalog_id != plan.catalog_id
        or scenario_catalog.catalog_version != plan.catalog_version
    ):
        raise ValueError(
            "offline manipulation audit catalog differs from the frozen plan"
        )
    declared_response = dict(plan.response_model)
    observed_response = {
        "beta": response_model.beta,
        "ranking_scale": response_model.ranking_scale,
        "default_scale": response_model.default_scale,
        "suggestion_scale": response_model.suggestion_scale,
    }
    if observed_response != declared_response:
        raise ValueError(
            "offline manipulation audit response model differs from the "
            "frozen plan"
        )

    population = tuple(users)
    domain_specs = tuple(domains)
    if len({user.user_id for user in population}) != len(population):
        raise ValueError("offline manipulation audit user IDs must be unique")
    if len({domain.domain_id for domain in domain_specs}) != len(domain_specs):
        raise ValueError("offline manipulation audit domain IDs must be unique")
    supplied_user_design = tuple(
        (
            user.user_id,
            tuple(float(value) for value in user.theta),
            (
                float(user.susceptibility.ranking),
                float(user.susceptibility.default),
                float(user.susceptibility.suggestion),
            ),
        )
        for user in population
    )
    if supplied_user_design != plan.declared_users:
        raise ValueError(
            "offline manipulation audit latent users differ from the frozen "
            "plan"
        )
    if tuple(domain.domain_id for domain in domain_specs) != plan.declared_domains:
        raise ValueError(
            "offline manipulation audit domains differ from the frozen plan"
        )
    for trajectory in plan.trajectories:
        for turn in trajectory.turns:
            scenario = scenario_catalog.scenario(turn.scenario_id)
            if (
                scenario.domain != trajectory.domain_id
                or scenario.split != plan.data_split
                or scenario.target_attribute != turn.target_attribute
                or scenario.target_key != turn.target_key
                or scenario.target_half_span != turn.target_half_span
            ):
                raise ValueError(
                    "offline manipulation audit scenario contract differs "
                    f"from the frozen plan: {turn.scenario_id}"
                )

    # Local imports keep the prospective planner independent of trajectory
    # execution during ordinary import and avoid a closed-loop module cycle.
    from .closed_loop import run_experiment_b
    from ..policies import BalancedPolicy, SoftProfileConditionedPolicy
    from ..rng import semantic_seed
    from ..updaters import ExactActionAwareUpdater

    conditions = tuple(
        dict.fromkeys(
            row.initial_profile_condition for row in plan.trajectories
        )
    )
    replicates = max(row.replicate for row in plan.trajectories) + 1
    horizons = {len(row.turns) for row in plan.trajectories}
    if len(horizons) != 1:
        raise ValueError("prospective audit requires one common turn horizon")
    turns = next(iter(horizons))
    expected_cells = {
        (
            row.domain_id,
            row.user_id,
            row.initial_profile_condition,
            row.replicate,
        )
        for row in plan.trajectories
    }
    supplied_cells = {
        (domain.domain_id, user.user_id, condition, replicate)
        for domain in domain_specs
        for user in population
        for condition in conditions
        for replicate in range(replicates)
    }
    if supplied_cells != expected_cells:
        raise ValueError(
            "offline manipulation audit users/domains do not reproduce every "
            "frozen plan cell exactly"
        )
    policies = {
        "balanced": BalancedPolicy(prospective_plan=plan),
        "soft_profile_conditioned": SoftProfileConditionedPolicy(
            prospective_plan=plan
        ),
    }
    if shadow_updater is None:
        observed_support = tuple(
            dict.fromkeys(user.susceptibility for user in population)
        )
        audit_profile_updater = ExactActionAwareUpdater(
            response_model,
            observed_support,
        )
    elif isinstance(shadow_updater, ExactActionAwareUpdater):
        audit_profile_updater = shadow_updater
    else:
        raise TypeError(
            "offline manipulation audit requires an ExactActionAwareUpdater "
            "profile-state driver"
        )
    if audit_profile_updater.response_model != response_model:
        raise ValueError(
            "offline manipulation audit exact updater response model differs "
            "from the frozen plan"
        )
    support = set(audit_profile_updater.susceptibilities)
    missing_support = tuple(
        user.user_id
        for user in population
        if user.susceptibility not in support
    )
    if missing_support:
        raise ValueError(
            "offline manipulation audit exact updater susceptibility support "
            "does not contain every declared user: "
            + ", ".join(missing_support)
        )
    selection_costs: list[float] = []
    expected_information_contrasts: list[float] = []
    choice_divergence_rates: list[float] = []
    visible_divergence_rates: list[float] = []
    condition_buckets: defaultdict[str, dict[str, list[float]]] = defaultdict(
        _new_audit_metric_bucket
    )
    domain_buckets: defaultdict[str, dict[str, list[float]]] = defaultdict(
        _new_audit_metric_bucket
    )
    condition_domain_buckets: defaultdict[
        tuple[str, str], dict[str, list[float]]
    ] = defaultdict(_new_audit_metric_bucket)
    role_buckets: defaultdict[str, dict[str, Any]] = defaultdict(
        _new_role_audit_bucket
    )
    active_turn_crosstab: Counter[
        tuple[str, str, int, int, int, str]
    ] = Counter()
    seed_summaries: list[dict[str, Any]] = []
    execution_match_count = 0
    execution_instruction_count = 0
    for response_seed_index in range(response_seed_count):
        response_seed = semantic_seed(
            plan.seed,
            "experiment-b-offline-manipulation-audit",
            response_seed_index,
        )
        result = run_experiment_b(
            users=population,
            domains=domain_specs,
            updaters={"exact_action_aware": audit_profile_updater},  # type: ignore[dict-item]
            policies=policies,
            initial_profile_conditions=conditions,
            turns=turns,
            trajectories_per_cell=replicates,
            response_model=response_model,
            shadow_updater=audit_profile_updater,
            seed=plan.seed,
            response_seed=response_seed,
            scenario_catalog=scenario_catalog,
            data_split=plan.data_split,
        )
        seed_selection = []
        seed_choice = []
        trajectory_by_id = {
            trajectory.trajectory_id: trajectory
            for trajectory in result.trajectories
        }
        for row in result.decompositions:
            selection_costs.append(row.evidence_selection_cost)
            seed_selection.append(row.evidence_selection_cost)
            information_contrast = (
                row.balanced_expected_preference_information_gain_deficit
            )
            if information_contrast is not None:
                expected_information_contrasts.append(
                    information_contrast
                )
            choice_divergence_rates.append(row.observed_choice_divergence_rate)
            seed_choice.append(row.observed_choice_divergence_rate)
            visible_divergence_rates.append(row.visible_action_divergence_rate)
            metric_buckets = (
                condition_buckets[row.initial_profile_condition],
                domain_buckets[row.domain_id],
                condition_domain_buckets[
                    (row.initial_profile_condition, row.domain_id)
                ],
            )
            for bucket in metric_buckets:
                bucket["selection_cost"].append(row.evidence_selection_cost)
                if information_contrast is not None:
                    bucket["expected_information_contrast"].append(
                        information_contrast
                    )
                bucket["observed_choice_divergence_rate"].append(
                    row.observed_choice_divergence_rate
                )
                bucket["visible_action_divergence_rate"].append(
                    row.visible_action_divergence_rate
                )

            profile = trajectory_by_id[row.profile_trajectory_id]
            balanced = trajectory_by_id[row.balanced_trajectory_id]
            for profile_turn, balanced_turn in zip(
                profile.turns,
                balanced.turns,
                strict=True,
            ):
                role = profile_turn.prospective_manipulation_role
                if role is None:
                    continue
                planned = plan.turn(profile.crn_key, profile_turn.turn)
                role_bucket = role_buckets[role]
                role_bucket["visible_action_divergence"].append(
                    float(
                        profile_turn.action_signature
                        != balanced_turn.action_signature
                    )
                )
                role_bucket["observed_choice_divergence"].append(
                    float(
                        profile_turn.selected_option_id
                        != balanced_turn.selected_option_id
                    )
                )
                expected_divergence = (
                    profile_turn.ex_ante_balanced_choice_divergence_probability
                )
                if expected_divergence is not None:
                    role_bucket[
                        "direction_specific_expected_choice_divergence_probability"
                    ].append(expected_divergence)
                planned_bound = (
                    profile_turn
                    .prospective_predicted_choice_divergence_probability
                )
                if planned_bound is not None:
                    role_bucket[
                        "conservative_planned_choice_divergence_bound"
                    ].append(planned_bound)
                role_bucket["mechanism_counts"][planned.mechanism] += 1
                role_bucket["susceptibility_stratum_counts"][
                    planned.susceptibility_stratum
                ] += 1
                direction_source = profile_turn.prospective_direction_source
                if direction_source is not None:
                    role_bucket["direction_source_counts"][direction_source] += 1
                effective_direction = (
                    profile_turn.prospective_effective_profile_direction
                )
                if effective_direction is not None:
                    role_bucket["effective_profile_direction_counts"][
                        effective_direction
                    ] += 1
                if role in {
                    "informative_active",
                    "decisive_active_control",
                }:
                    if effective_direction not in {-1, 1}:
                        raise RuntimeError(
                            "a matched prospective active turn must record its "
                            "effective profile direction"
                        )
                    active_turn_crosstab[
                        (
                            role,
                            planned.mechanism,
                            int(effective_direction),
                            planned.planned_profile_direction,
                            planned.target_attribute,
                            profile.domain_id,
                        )
                    ] += 1
                    role_bucket["execution_instruction_count"] += 1
                    role_bucket["execution_match_count"] += int(
                        profile_turn.prospective_execution_matched is True
                    )
                    execution_instruction_count += 1
                    execution_match_count += int(
                        profile_turn.prospective_execution_matched is True
                    )
        seed_summaries.append(
            {
                "response_seed_index": response_seed_index,
                "paired_trajectory_count": len(result.decompositions),
                "mean_selection_cost": (
                    math.fsum(seed_selection) / len(seed_selection)
                ),
                "mean_choice_divergence_rate": (
                    math.fsum(seed_choice) / len(seed_choice)
                ),
            }
        )
    selection_distribution = _distribution(selection_costs)
    role_summaries = {
        role: _summarize_role_audit_bucket(bucket)
        for role, bucket in sorted(role_buckets.items())
    }
    zero_direction_fallback_count = sum(
        int(
            summary["direction_source_counts"].get(
                "frozen_initial_profile_fallback",
                0,
            )
        )
        for summary in role_summaries.values()
    )
    active_crosstab_rows = [
        {
            "role": role,
            "mechanism": mechanism,
            "effective_profile_direction": effective_direction,
            "planned_initial_profile_direction": planned_direction,
            "target_attribute": target_attribute,
            "domain": domain_id,
            "simulated_turn_draw_count": count,
        }
        for (
            role,
            mechanism,
            effective_direction,
            planned_direction,
            target_attribute,
            domain_id,
        ), count in sorted(active_turn_crosstab.items())
    ]
    active_crosstab_count = sum(
        row["simulated_turn_draw_count"] for row in active_crosstab_rows
    )
    active_role_summary_count = sum(
        int(role_summaries[role]["simulated_turn_draw_count"])
        for role in ("informative_active", "decisive_active_control")
        if role in role_summaries
    )
    by_condition_and_domain: dict[str, dict[str, Any]] = {}
    for (condition, domain_id), bucket in sorted(
        condition_domain_buckets.items()
    ):
        by_condition_and_domain.setdefault(condition, {})[domain_id] = (
            _summarize_audit_metric_bucket(
                bucket,
                selection_margin=margin,
            )
        )
    return {
        "schema_version": 2,
        "audit_id": "experiment-b-offline-manipulation-audit-v2",
        "status": "completed",
        "llm_calls": 0,
        "evaluated_model_outputs_used": False,
        "simulated_choices_used": True,
        "adaptive_policy_state_driver": (
            "local exact action-aware updater; not an evaluated LLM"
        ),
        "exact_profile_state_driver": {
            "updater_id": audit_profile_updater.updater_id,
            "response_model": observed_response,
            "susceptibility_support": [
                {
                    "ranking": item.ranking,
                    "default": item.default,
                    "suggestion": item.suggestion,
                }
                for item in audit_profile_updater.susceptibilities
            ],
            "susceptibility_weights": (
                None
                if audit_profile_updater.susceptibility_weights is None
                else list(audit_profile_updater.susceptibility_weights)
            ),
        },
        "plan_reselected_from_simulated_outcomes": False,
        "admission_effect": "none; descriptive stress audit after plan freeze",
        "plan_id": plan.plan_id,
        "response_seed_count": response_seed_count,
        "paired_trajectory_draw_count": len(selection_costs),
        "selection_noninferiority_margin": margin,
        "selection_cost": {
            **selection_distribution,
            "fraction_at_or_below_margin": (
                sum(value <= margin for value in selection_costs)
                / len(selection_costs)
            ),
        },
        "balanced_minus_soft_expected_information_gain": _distribution(
            expected_information_contrasts
        ),
        "observed_choice_divergence_rate": _distribution(
            choice_divergence_rates
        ),
        "visible_action_divergence_rate": _distribution(
            visible_divergence_rates
        ),
        "behavioral_reinforcement": {
            "status": "not_evaluated",
            "reason": (
                "requires an evaluated target-updater output; the local exact "
                "updater only evolves the adaptive policy state, and its "
                "structural agreement with the exact shadow is not behavioral "
                "evidence"
            ),
        },
        "required_active_execution": {
            "instruction_count": execution_instruction_count,
            "matched_count": execution_match_count,
            "zero_direction_fallback_count": (
                zero_direction_fallback_count
            ),
            "all_matched": (
                execution_instruction_count > 0
                and execution_match_count == execution_instruction_count
            ),
        },
        "prospective_active_turn_crosstab": {
            "status": "descriptive_transparency_audit",
            "admission_effect": "none",
            "dimensions": [
                "role",
                "mechanism",
                "effective_profile_direction",
                "planned_initial_profile_direction",
                "target_attribute",
                "domain",
            ],
            "effective_direction_interpretation": (
                "runtime current-profile direction, or the logged frozen "
                "initial-profile fallback when exactly neutral"
            ),
            "simulated_turn_draw_count": active_crosstab_count,
            "pooled_required_active_instruction_count": (
                execution_instruction_count
            ),
            "active_role_summary_turn_draw_count": active_role_summary_count,
            "counts_reconcile": (
                active_crosstab_count
                == execution_instruction_count
                == active_role_summary_count
            ),
            "rows": active_crosstab_rows,
        },
        "coverage_and_symmetry": plan.summary.to_dict()[
            "coverage_and_symmetry"
        ],
        "plan_summary": plan.summary.to_dict(),
        "by_initial_profile_condition": {
            condition: _summarize_audit_metric_bucket(
                bucket,
                selection_margin=margin,
            )
            for condition, bucket in sorted(condition_buckets.items())
        },
        "by_domain": {
            domain_id: _summarize_audit_metric_bucket(
                bucket,
                selection_margin=margin,
            )
            for domain_id, bucket in sorted(domain_buckets.items())
        },
        "by_condition_and_domain": by_condition_and_domain,
        "by_prospective_role": role_summaries,
        "per_response_seed": seed_summaries,
    }


@dataclass(frozen=True, slots=True)
class _Candidate:
    turn: int
    scenario: ScenarioSpec
    mechanism: str
    profile_direction: int
    promoted_option_id: str | None
    balanced_probability_margin: float
    margin_stratum: str
    intrinsic_margin: float
    directional_divergence_probabilities: tuple[tuple[int, float], ...]
    initial_direction_divergence_probability: float
    minimum_divergence_probability: float
    maximum_divergence_probability: float
    mechanism_susceptibility: float
    susceptibility_stratum: str
    counter_profile_option_retained: bool


def _margin_stratum(margin: float) -> str:
    near_tie, decisive = BALANCED_MARGIN_THRESHOLDS
    if margin < near_tie:
        return "near_tie"
    if margin < decisive:
        return "marginal"
    return "decisive"


def _susceptibility_stratum(value: float) -> str:
    if value <= 0.0:
        return "inactive"
    if value < 0.30:
        return "low"
    if value < 0.65:
        return "medium"
    return "high"


def _profile_direction(belief: PreferenceBelief, target: int) -> int:
    expected = belief.expected_theta()[target]
    if expected < -1e-12:
        return -1
    if expected > 1e-12:
        return 1
    return 0


def _promoted_option_id(context: InteractionContext, direction: int) -> str | None:
    if direction == 0:
        return None
    target = context.target_attribute
    if target is None:
        return None
    matches = tuple(
        option.option_id
        for option in context.options
        if option.features[target] * direction > 0.0
    )
    return matches[0] if len(matches) == 1 else None


def _retains_both_directions(context: InteractionContext) -> bool:
    target = context.target_attribute
    if target is None:
        return False
    directions = {
        -1 if option.features[target] < 0.0 else 1
        for option in context.options
        if option.features[target] != 0.0
    }
    return directions == {-1, 1}


def _candidate(
    *,
    user: LatentUser,
    domain: DomainSpec,
    initial: PreferenceBelief,
    scenario: ScenarioSpec,
    turn: int,
    schedule_group_key: str,
    mechanism: str,
    response_model: RandomUtilityModel,
    seed: int,
) -> _Candidate:
    balanced_action = BalancedPolicy().action(
        domain,
        initial,
        turn=turn,
        master_seed=seed,
        trajectory_id=schedule_group_key,
    )
    balanced = materialize_context(balanced_action.context, scenario)
    direction = _profile_direction(initial, scenario.target_attribute)
    promoted = _promoted_option_id(balanced, direction)
    mechanism_susceptibility = (
        float(getattr(user.susceptibility, mechanism))
        if mechanism in ACTIVE_MECHANISMS
        else 0.0
    )
    balanced_probabilities = response_model.probabilities(
        user.theta,
        user.susceptibility,
        balanced,
    )
    first, second = sorted(balanced_probabilities, reverse=True)[:2]
    probability_margin = first - second
    probability_by_id = dict(zip(balanced.option_ids, balanced_probabilities))
    reference_id = sorted(probability_by_id)[0]
    directional_rows: list[tuple[int, float]] = []
    for profile_direction in (-1, 1):
        promoted_for_direction = _promoted_option_id(
            balanced,
            profile_direction,
        )
        if promoted_for_direction is None:
            raise RuntimeError(
                "a binary candidate must retain one option in each direction"
            )
        treated = replace(
            balanced,
            context_id=(
                f"{balanced.context_id}:planned-{mechanism}:"
                f"direction-{profile_direction}"
            ),
            default_option_id=(
                promoted_for_direction if mechanism == "default" else None
            ),
            suggested_option_id=(
                promoted_for_direction if mechanism == "suggestion" else None
            ),
        )
        treated_probabilities = response_model.probability_map(
            user.theta,
            user.susceptibility,
            treated,
        )
        directional_rows.append(
            (
                profile_direction,
                abs(
                    probability_by_id[reference_id]
                    - treated_probabilities[reference_id]
                ),
            )
        )
    directional_divergences = tuple(directional_rows)
    divergence_by_direction = dict(directional_divergences)
    utilities = tuple(
        intrinsic_utility(user.theta, option) for option in balanced.options
    )
    return _Candidate(
        turn=turn,
        scenario=scenario,
        mechanism=mechanism,
        profile_direction=direction,
        promoted_option_id=promoted,
        balanced_probability_margin=probability_margin,
        margin_stratum=_margin_stratum(probability_margin),
        intrinsic_margin=max(utilities) - min(utilities),
        directional_divergence_probabilities=directional_divergences,
        initial_direction_divergence_probability=(
            0.0 if direction == 0 else divergence_by_direction[direction]
        ),
        minimum_divergence_probability=min(divergence_by_direction.values()),
        maximum_divergence_probability=max(divergence_by_direction.values()),
        mechanism_susceptibility=mechanism_susceptibility,
        susceptibility_stratum=_susceptibility_stratum(
            mechanism_susceptibility
        ),
        counter_profile_option_retained=(
            promoted is not None and _retains_both_directions(balanced)
        ),
    )


def _turn_from_candidate(candidate: _Candidate, role: str) -> ProspectiveTurnPlan:
    active = role in {"informative_active", "decisive_active_control"}
    conservative_probability = (
        candidate.minimum_divergence_probability
        if role == "informative_active"
        else candidate.maximum_divergence_probability
        if role == "decisive_active_control"
        else None
    )
    bound_kind = (
        "lower_bound_across_current_profile_directions"
        if role == "informative_active"
        else "upper_bound_across_current_profile_directions"
        if role == "decisive_active_control"
        else None
    )
    return ProspectiveTurnPlan(
        turn=candidate.turn,
        scenario_id=candidate.scenario.scenario_id,
        target_attribute=candidate.scenario.target_attribute,
        target_key=candidate.scenario.target_key,
        target_half_span=candidate.scenario.target_half_span,
        role=role,
        mechanism=candidate.mechanism if active else "adaptive",
        planned_profile_direction=candidate.profile_direction,
        promoted_option_id=(candidate.promoted_option_id if active else None),
        retained_preference_directions=(-1, 1),
        balanced_choice_probability_margin=(
            candidate.balanced_probability_margin
        ),
        balanced_choice_margin_stratum=candidate.margin_stratum,
        intrinsic_utility_margin=candidate.intrinsic_margin,
        predicted_shared_noise_choice_divergence_probability=(
            conservative_probability
        ),
        directional_choice_divergence_probabilities=(
            candidate.directional_divergence_probabilities
        ),
        conservative_divergence_bound_kind=bound_kind,
        mechanism_susceptibility=(
            candidate.mechanism_susceptibility if active else None
        ),
        susceptibility_stratum=(
            candidate.susceptibility_stratum if active else "adaptive"
        ),
        counter_profile_option_retained=(
            candidate.counter_profile_option_retained
        ),
        soft_visible_divergence_required=active,
    )


def _eligible_informative(
    candidate: _Candidate,
    requirements: ManipulationRequirements,
) -> bool:
    return (
        candidate.margin_stratum in {"near_tie", "marginal"}
        and candidate.minimum_divergence_probability
        >= requirements.minimum_informative_choice_divergence_probability
        and candidate.promoted_option_id is not None
        and (
            not requirements.require_counter_profile_options
            or candidate.counter_profile_option_retained
        )
    )


def _eligible_control(
    candidate: _Candidate,
    requirements: ManipulationRequirements,
) -> bool:
    return (
        candidate.margin_stratum == "decisive"
        and candidate.maximum_divergence_probability
        <= requirements.maximum_decisive_choice_divergence_probability
        and candidate.promoted_option_id is not None
        and (
            not requirements.require_counter_profile_options
            or candidate.counter_profile_option_retained
        )
    )


def _selection_score(
    informative: Sequence[_Candidate],
    controls: Sequence[_Candidate],
) -> tuple[float, ...]:
    active = (*informative, *controls)
    informative_divergences = tuple(
        item.minimum_divergence_probability for item in informative
    )
    target_counts = Counter(item.scenario.target_attribute for item in active)
    directions = Counter(item.profile_direction for item in active)
    return (
        float(len({item.scenario.target_attribute for item in active})),
        float(len({item.scenario.target_attribute for item in informative})),
        min(informative_divergences),
        math.fsum(informative_divergences),
        -math.fsum(item.maximum_divergence_probability for item in controls),
        math.fsum(item.balanced_probability_margin for item in controls),
        -float(max(target_counts.values()) - min(target_counts.values())),
        -float(abs(directions.get(-1, 0) - directions.get(1, 0))),
    )


def _select_active_candidates(
    candidates: Sequence[_Candidate],
    requirements: ManipulationRequirements,
) -> tuple[tuple[_Candidate, ...], tuple[_Candidate, ...]] | None:
    informative_pool = tuple(
        item for item in candidates if _eligible_informative(item, requirements)
    )
    control_pool = tuple(
        item for item in candidates if _eligible_control(item, requirements)
    )
    informative_count = requirements.minimum_informative_active_turns
    control_count = requirements.minimum_decisive_active_controls
    best: tuple[tuple[_Candidate, ...], tuple[_Candidate, ...]] | None = None
    best_score: tuple[float, ...] | None = None
    best_key: tuple[tuple[int, str, str], ...] | None = None
    for informative in combinations(informative_pool, informative_count):
        informative_turns = {item.turn for item in informative}
        informative_scenarios = {item.scenario.scenario_id for item in informative}
        if (
            len(informative_turns) != informative_count
            or len(informative_scenarios) != informative_count
        ):
            continue
        remaining_controls = tuple(
            item
            for item in control_pool
            if item.turn not in informative_turns
            and item.scenario.scenario_id not in informative_scenarios
        )
        for controls in combinations(remaining_controls, control_count):
            active = (*informative, *controls)
            if len({item.turn for item in active}) != len(active):
                continue
            if len({item.scenario.scenario_id for item in active}) != len(active):
                continue
            if (
                len({item.mechanism for item in active})
                < requirements.minimum_active_mechanisms
            ):
                continue
            active_susceptibility_mass = math.fsum(
                item.minimum_divergence_probability for item in active
            )
            if (
                active_susceptibility_mass
                < requirements.minimum_active_susceptibility_mass
            ):
                continue
            score = _selection_score(informative, controls)
            tie_key = tuple(
                sorted(
                    (
                        item.turn,
                        item.mechanism,
                        item.scenario.scenario_id,
                    )
                    for item in active
                )
            )
            if (
                best is None
                or score > best_score  # type: ignore[operator]
                or (score == best_score and tie_key < best_key)  # type: ignore[operator]
            ):
                best = (tuple(informative), tuple(controls))
                best_score = score
                best_key = tie_key
    return best


def _adaptive_candidate(
    *,
    all_candidates: Sequence[_Candidate],
    turn: int,
    used_scenarios: set[str],
) -> _Candidate:
    pool = tuple(item for item in all_candidates if item.turn == turn)
    if not pool:
        raise RuntimeError(f"no catalog scenario candidate for turn {turn}")
    # Each mechanism shares the same balanced action; use the default row as a
    # canonical carrier and prefer an unused scenario with a middle span.
    canonical = {
        item.scenario.scenario_id: item
        for item in pool
        if item.mechanism == "default"
    }
    ordered = sorted(
        canonical.values(),
        key=lambda item: (
            item.scenario.scenario_id in used_scenarios,
            abs(item.scenario.target_half_span - 0.34),
            item.scenario.scenario_id,
        ),
    )
    return ordered[0]


def _trajectory_plan(
    *,
    user: LatentUser,
    domain: DomainSpec,
    condition: str,
    replicate: int,
    schedule_group_key: str,
    turns: int,
    catalog: ScenarioCatalog,
    response_model: RandomUtilityModel,
    requirements: ManipulationRequirements,
    profile_strength: float,
    prior_uncertainty: float,
    seed: int,
    data_split: str,
) -> ProspectiveTrajectoryPlan:
    paired_key = (
        f"experiment-b:{domain.domain_id}:{user.user_id}:{condition}:"
        f"replicate-{replicate}"
    )
    initial = add_prior_uncertainty(
        initial_profile_belief(
            user.theta,
            condition,
            profile_strength=profile_strength,
        ),
        prior_uncertainty,
    )
    failures: list[str] = []
    if requirements.minimum_informative_active_turns + (
        requirements.minimum_decisive_active_controls
    ) > turns:
        failures.append(
            "turn budget is smaller than the required informative and "
            "decisive active roles"
        )
    all_candidates: list[_Candidate] = []
    for turn in range(turns):
        target = turn % 3
        scenarios = tuple(
            sorted(
                catalog.eligible(domain.domain_id, data_split, target),
                key=lambda item: (item.target_half_span, item.scenario_id),
            )
        )
        if not scenarios:
            failures.append(
                f"catalog has no {data_split} scenario for target {target}"
            )
            continue
        for scenario in scenarios:
            for mechanism in ACTIVE_MECHANISMS:
                all_candidates.append(
                    _candidate(
                        user=user,
                        domain=domain,
                        initial=initial,
                        scenario=scenario,
                        turn=turn,
                        schedule_group_key=schedule_group_key,
                        mechanism=mechanism,
                        response_model=response_model,
                        seed=seed,
                    )
                )
    selected = (
        None
        if failures
        else _select_active_candidates(all_candidates, requirements)
    )
    if selected is None:
        failures.append(
            "no outcome-blind assignment satisfies the informative-turn, "
            "decisive-control, mechanism, divergence, and option-retention "
            "thresholds"
        )
        informative: tuple[_Candidate, ...] = ()
        controls: tuple[_Candidate, ...] = ()
    else:
        informative, controls = selected
    assigned: dict[int, tuple[_Candidate, str]] = {
        item.turn: (item, "informative_active") for item in informative
    }
    assigned.update(
        {
            item.turn: (item, "decisive_active_control")
            for item in controls
        }
    )
    used_scenarios = {item.scenario.scenario_id for item in (*informative, *controls)}
    for turn in range(turns):
        if turn in assigned:
            continue
        if not any(item.turn == turn for item in all_candidates):
            continue
        adaptive = _adaptive_candidate(
            all_candidates=all_candidates,
            turn=turn,
            used_scenarios=used_scenarios,
        )
        assigned[turn] = (adaptive, "adaptive_observation")
        used_scenarios.add(adaptive.scenario.scenario_id)
    planned_turns = tuple(
        _turn_from_candidate(*assigned[index])
        for index in range(turns)
        if index in assigned
    )
    if len(planned_turns) != turns:
        failures.append("prospective schedule does not cover every turn")
    active_turns = tuple(turn for turn in planned_turns if turn.active)
    informative_turns = tuple(
        turn for turn in planned_turns if turn.role == "informative_active"
    )
    control_turns = tuple(
        turn for turn in planned_turns if turn.role == "decisive_active_control"
    )
    active_mechanisms = tuple(sorted({turn.mechanism for turn in active_turns}))
    if len(informative_turns) < requirements.minimum_informative_active_turns:
        failures.append("too few admitted informative active turns")
    if len(control_turns) < requirements.minimum_decisive_active_controls:
        failures.append("too few admitted decisive active controls")
    if len(active_mechanisms) < requirements.minimum_active_mechanisms:
        failures.append("too few distinct active presentation mechanisms")
    if any(not turn.counter_profile_option_retained for turn in planned_turns):
        failures.append("one or more turns do not retain both preference directions")
    informative_asm = math.fsum(
        turn.predicted_shared_noise_choice_divergence_probability or 0.0
        for turn in informative_turns
    )
    control_asm = math.fsum(
        turn.predicted_shared_noise_choice_divergence_probability or 0.0
        for turn in control_turns
    )
    active_susceptibility_mass = math.fsum(
        min(dict(turn.directional_choice_divergence_probabilities).values())
        for turn in active_turns
    )
    if (
        active_susceptibility_mass
        < requirements.minimum_active_susceptibility_mass
    ):
        failures.append(
            "trajectory active susceptibility mass is below the declared "
            "minimum"
        )
    return ProspectiveTrajectoryPlan(
        shared_pair_key=paired_key,
        schedule_group_key=schedule_group_key,
        domain_id=domain.domain_id,
        user_id=user.user_id,
        initial_profile_condition=condition,
        replicate=replicate,
        turns=planned_turns,
        informative_asm=informative_asm,
        decisive_control_asm=control_asm,
        active_susceptibility_mass=active_susceptibility_mass,
        active_mechanisms=active_mechanisms,
        active_target_attributes=tuple(
            sorted({turn.target_attribute for turn in active_turns})
        ),
        ready=not failures,
        readiness_failures=tuple(dict.fromkeys(failures)),
    )


def _summary(
    trajectories: Sequence[ProspectiveTrajectoryPlan],
) -> ManipulationPlanSummary:
    all_turns = tuple(turn for row in trajectories for turn in row.turns)
    active = tuple(turn for turn in all_turns if turn.active)
    role_counts = Counter(turn.role for turn in all_turns)
    mechanism_counts = Counter(turn.mechanism for turn in active)
    target_counts = Counter(turn.target_attribute for turn in active)
    direction_counts = Counter(
        turn.planned_profile_direction for turn in active
    )
    masses = tuple(row.active_susceptibility_mass for row in trajectories)

    def count_gap(counter: Counter[int]) -> int:
        values = tuple(counter.values())
        return 0 if not values else max(values) - min(values)

    return ManipulationPlanSummary(
        trajectory_count=len(trajectories),
        turn_count=len(all_turns),
        role_counts=tuple(sorted(role_counts.items())),
        mechanism_counts=tuple(sorted(mechanism_counts.items())),
        target_counts=tuple(sorted(target_counts.items())),
        promoted_direction_counts=tuple(sorted(direction_counts.items())),
        scenario_count=len({turn.scenario_id for turn in all_turns}),
        minimum_trajectory_asm=min(masses, default=0.0),
        mean_trajectory_asm=(
            math.fsum(masses) / len(masses) if masses else 0.0
        ),
        maximum_trajectory_asm=max(masses, default=0.0),
        active_target_count_gap=count_gap(target_counts),
        promoted_direction_count_gap=count_gap(direction_counts),
    )


def build_experiment_b_manipulation_plan(
    *,
    users: Sequence[LatentUser],
    domains: Sequence[DomainSpec],
    scenario_catalog: ScenarioCatalog,
    response_model: RandomUtilityModel,
    initial_profile_conditions: Sequence[str],
    turns: int,
    trajectories_per_cell: int = 1,
    requirements: object | None = None,
    profile_strength: float = 0.80,
    prior_uncertainty: float = 0.0,
    seed: int = 1729,
    data_split: str = "test",
    fail_closed: bool = True,
) -> ExperimentBManipulationPlan:
    """Build and admit the complete Experiment B schedule without outcomes.

    The shared-pair key exactly matches :func:`run_experiment_b`, so execution
    can retrieve a declaration with ``plan.turn(crn_key, turn)``.  With the
    default ``fail_closed=True``, any unsatisfied trajectory raises before a
    provider call can be made; the exception retains the complete audit plan.
    """

    population = tuple(users)
    domain_specs = tuple(domains)
    conditions = tuple(initial_profile_conditions)
    declared_requirements = ManipulationRequirements.from_source(requirements)
    if not population:
        raise ValueError("prospective Experiment B planning requires users")
    if not domain_specs:
        raise ValueError("prospective Experiment B planning requires domains")
    if not conditions:
        raise ValueError("initial_profile_conditions cannot be empty")
    if isinstance(turns, bool) or not isinstance(turns, int) or turns <= 0:
        raise ValueError("turns must be a positive integer")
    if (
        isinstance(trajectories_per_cell, bool)
        or not isinstance(trajectories_per_cell, int)
        or trajectories_per_cell <= 0
    ):
        raise ValueError("trajectories_per_cell must be a positive integer")
    strength = _finite(profile_strength, "profile_strength")
    uncertainty = _finite(prior_uncertainty, "prior_uncertainty")
    if not 0.5 <= strength < 1.0:
        raise ValueError("profile_strength must lie in [0.5, 1)")
    if not 0.0 <= uncertainty < 1.0:
        raise ValueError("prior_uncertainty must lie in [0, 1)")
    trajectory_plans = tuple(
        _trajectory_plan(
            user=user,
            domain=domain,
            condition=condition,
            replicate=replicate,
            schedule_group_key=(
                f"experiment-b:{domain.domain_id}:{user.user_id}:"
                f"condition-invariant:replicate-{replicate}"
            ),
            turns=turns,
            catalog=scenario_catalog,
            response_model=response_model,
            requirements=declared_requirements,
            profile_strength=strength,
            prior_uncertainty=uncertainty,
            seed=seed,
            data_split=data_split,
        )
        for domain in domain_specs
        for user in population
        for condition in conditions
        for replicate in range(trajectories_per_cell)
    )
    schedule_signatures: dict[str, tuple[tuple[object, ...], ...]] = {}
    mismatched_schedule_groups: set[str] = set()
    for row in trajectory_plans:
        signature = tuple(
            (
                turn.turn,
                turn.scenario_id,
                turn.target_attribute,
                turn.role,
                turn.mechanism,
            )
            for turn in row.turns
        )
        existing = schedule_signatures.setdefault(
            row.schedule_group_key,
            signature,
        )
        if existing != signature:
            mismatched_schedule_groups.add(row.schedule_group_key)
    failed = tuple(
        row.shared_pair_key
        for row in trajectory_plans
        if not row.ready
        or row.schedule_group_key in mismatched_schedule_groups
    )
    readiness = ManipulationReadiness(
        ready=not failed,
        outcome_blind=True,
        trajectory_count=len(trajectory_plans),
        admitted_trajectory_count=len(trajectory_plans) - len(failed),
        failed_trajectory_keys=failed,
        checks=(
            ("built_without_realized_choices", True),
            ("built_without_updated_profiles", True),
            ("built_without_evaluated_model_outputs", True),
            (
                "condition_invariant_scenario_role_mechanism_schedule",
                not mismatched_schedule_groups,
            ),
            (
                "every_trajectory_satisfies_declared_thresholds",
                all(row.ready for row in trajectory_plans),
            ),
        ),
    )
    plan = ExperimentBManipulationPlan(
        plan_id="experiment-b-prospective-manipulation-v2",
        seed=seed,
        data_split=data_split,
        catalog_id=scenario_catalog.catalog_id,
        catalog_version=scenario_catalog.catalog_version,
        declared_users=tuple(
            (
                user.user_id,
                tuple(float(value) for value in user.theta),
                (
                    float(user.susceptibility.ranking),
                    float(user.susceptibility.default),
                    float(user.susceptibility.suggestion),
                ),
            )
            for user in population
        ),
        declared_domains=tuple(domain.domain_id for domain in domain_specs),
        response_model=(
            ("beta", response_model.beta),
            ("ranking_scale", response_model.ranking_scale),
            ("default_scale", response_model.default_scale),
            ("suggestion_scale", response_model.suggestion_scale),
        ),
        requirements=declared_requirements,
        trajectories=trajectory_plans,
        summary=_summary(trajectory_plans),
        readiness=readiness,
    )
    if fail_closed and not readiness.ready:
        raise ManipulationPlanError(plan)
    return plan


__all__ = [
    "ACTIVE_MECHANISMS",
    "BALANCED_MARGIN_THRESHOLDS",
    "ExperimentBManipulationPlan",
    "ManipulationPlanError",
    "ManipulationPlanSummary",
    "ManipulationReadiness",
    "ManipulationRequirements",
    "ProspectiveTrajectoryPlan",
    "ProspectiveTurnPlan",
    "build_experiment_b_manipulation_plan",
    "render_manipulation_plan_markdown",
    "run_offline_manipulation_audit",
]
