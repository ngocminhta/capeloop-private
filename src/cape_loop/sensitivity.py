"""Declared simulator-sensitivity grids and phase-diagram records."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
from typing import Any, Callable, Iterable, Mapping, Sequence

from .response import RandomUtilityModel, RuleBasedResponseModel


@dataclass(frozen=True, slots=True)
class SensitivityPoint:
    decision_noise: float
    presentation_multiplier: float
    profile_strength: float
    trajectory_length: int
    profile_conditioning_strength: float = 1.0
    rank_multiplier: float = 1.0
    default_multiplier: float = 1.0
    suggestion_multiplier: float = 1.0
    prior_uncertainty: float = 0.0
    response_model_family: str = "random_utility"
    rule_noise: float | None = None

    def __post_init__(self) -> None:
        if self.decision_noise <= 0:
            raise ValueError("decision_noise must be positive")
        if self.presentation_multiplier < 0:
            raise ValueError("presentation_multiplier must be non-negative")
        if not 0.0 <= self.profile_conditioning_strength <= 1.0:
            raise ValueError(
                "profile_conditioning_strength must lie in [0, 1]"
            )
        if any(
            value < 0
            for value in (
                self.rank_multiplier,
                self.default_multiplier,
                self.suggestion_multiplier,
            )
        ):
            raise ValueError("presentation-channel multipliers must be non-negative")
        if not 0.5 <= self.profile_strength < 1:
            raise ValueError("profile_strength must lie in [0.5, 1)")
        if not 0.0 <= self.prior_uncertainty < 1.0:
            raise ValueError("prior_uncertainty must lie in [0, 1)")
        if self.trajectory_length <= 0:
            raise ValueError("trajectory_length must be positive")
        if self.response_model_family not in {
            "random_utility",
            "rule_based",
        }:
            raise ValueError("unknown response_model_family")
        if self.response_model_family == "rule_based":
            if self.rule_noise is None or not 0.0 <= self.rule_noise <= 1.0:
                raise ValueError(
                    "rule_based points require rule_noise in [0, 1]"
                )
        elif self.rule_noise is not None:
            raise ValueError(
                "random_utility points must not declare rule_noise"
            )

    @property
    def point_id(self) -> str:
        return (
            f"family={self.response_model_family};"
            f"noise={self.decision_noise:.8g};"
            f"presentation={self.presentation_multiplier:.8g};"
            f"conditioning={self.profile_conditioning_strength:.8g};"
            f"rank={self.rank_multiplier:.8g};"
            f"default={self.default_multiplier:.8g};"
            f"suggestion={self.suggestion_multiplier:.8g};"
            f"profile={self.profile_strength:.8g};"
            f"prior_uncertainty={self.prior_uncertainty:.8g};"
            f"turns={self.trajectory_length}"
            + (
                ""
                if self.rule_noise is None
                else f";rule_noise={self.rule_noise:.8g}"
            )
        )

    def to_dict(self) -> dict[str, float | int | str | None]:
        return {
            "point_id": self.point_id,
            "decision_noise": self.decision_noise,
            "presentation_multiplier": self.presentation_multiplier,
            "profile_conditioning_strength": (
                self.profile_conditioning_strength
            ),
            "rank_multiplier": self.rank_multiplier,
            "default_multiplier": self.default_multiplier,
            "suggestion_multiplier": self.suggestion_multiplier,
            "profile_strength": self.profile_strength,
            "prior_uncertainty": self.prior_uncertainty,
            "trajectory_length": self.trajectory_length,
            "response_model_family": self.response_model_family,
            "rule_noise": self.rule_noise,
        }


def sensitivity_grid(
    *,
    design: str = "cartesian",
    decision_noise_values: Iterable[float],
    presentation_multipliers: Iterable[float],
    profile_strength_values: Iterable[float],
    trajectory_lengths: Iterable[int],
    profile_conditioning_strength_values: Iterable[float] = (1.0,),
    rank_multipliers: Iterable[float] = (1.0,),
    default_multipliers: Iterable[float] = (1.0,),
    suggestion_multipliers: Iterable[float] = (1.0,),
    prior_uncertainty_values: Iterable[float] = (0.0,),
    response_model_families: Iterable[str] = ("random_utility",),
    rule_noise_values: Iterable[float] = (0.15,),
) -> tuple[SensitivityPoint, ...]:
    """Build a deterministic Cartesian or baseline-first OAT grid.

    ``one_at_a_time`` treats the first value of every numeric axis and the
    first response-model family as the baseline. Each remaining numeric value
    is varied separately under that baseline family. Remaining response-model
    families are evaluated at the numeric baseline; every declared rule-noise
    value is evaluated when the alternate family is ``rule_based``. The design
    measures broad marginal perturbations but does not identify interactions
    among sensitivity axes.
    """

    if design not in {"cartesian", "one_at_a_time"}:
        raise ValueError(
            "sensitivity design must be 'cartesian' or 'one_at_a_time'"
        )
    axes = (
        tuple(decision_noise_values),
        tuple(presentation_multipliers),
        tuple(profile_conditioning_strength_values),
        tuple(rank_multipliers),
        tuple(default_multipliers),
        tuple(suggestion_multipliers),
        tuple(profile_strength_values),
        tuple(prior_uncertainty_values),
        tuple(trajectory_lengths),
    )
    if any(not values for values in axes):
        raise ValueError("sensitivity axes must be non-empty")
    if design == "cartesian":
        bases = tuple(product(*axes))
    else:
        baseline = tuple(values[0] for values in axes)
        one_at_a_time = [baseline]
        for axis_index, values in enumerate(axes):
            for value in values[1:]:
                point = list(baseline)
                point[axis_index] = value
                one_at_a_time.append(tuple(point))
        bases = tuple(one_at_a_time)
    families = tuple(response_model_families)
    rule_noises = tuple(rule_noise_values)
    if not families:
        raise ValueError("response_model_families must be non-empty")
    if "rule_based" in families and not rule_noises:
        raise ValueError(
            "rule_noise_values must be non-empty for rule_based points"
        )
    points: list[SensitivityPoint] = []
    baseline_family = families[0]
    for (
        noise,
        presentation,
        conditioning,
        rank,
        default,
        suggestion,
        profile,
        prior_uncertainty,
        turns,
    ) in bases:
        point_families = (
            families if design == "cartesian" else (baseline_family,)
        )
        for family in point_families:
            if family == "rule_based":
                family_rule_noises: tuple[float | None, ...] = (
                    rule_noises
                    if design == "cartesian"
                    else (rule_noises[0],)
                )
            else:
                family_rule_noises = (None,)
            for rule_noise in family_rule_noises:
                points.append(
                    SensitivityPoint(
                        decision_noise=noise,
                        presentation_multiplier=presentation,
                        profile_conditioning_strength=conditioning,
                        profile_strength=profile,
                        trajectory_length=turns,
                        rank_multiplier=rank,
                        default_multiplier=default,
                        suggestion_multiplier=suggestion,
                        prior_uncertainty=prior_uncertainty,
                        response_model_family=family,
                        rule_noise=rule_noise,
                    )
                )
    if design == "one_at_a_time":
        baseline = bases[0]
        (
            noise,
            presentation,
            conditioning,
            rank,
            default,
            suggestion,
            profile,
            prior_uncertainty,
            turns,
        ) = baseline
        for family in families:
            if family == baseline_family:
                family_rule_noises = (
                    rule_noises[1:]
                    if family == "rule_based"
                    else ()
                )
            else:
                family_rule_noises = (
                    rule_noises if family == "rule_based" else (None,)
                )
            for rule_noise in family_rule_noises:
                points.append(
                    SensitivityPoint(
                        decision_noise=noise,
                        presentation_multiplier=presentation,
                        profile_conditioning_strength=conditioning,
                        profile_strength=profile,
                        trajectory_length=turns,
                        rank_multiplier=rank,
                        default_multiplier=default,
                        suggestion_multiplier=suggestion,
                        prior_uncertainty=prior_uncertainty,
                        response_model_family=family,
                        rule_noise=rule_noise,
                    )
                )
    return tuple(points)


def response_model_at(
    point: SensitivityPoint,
    *,
    beta: float,
    rank_scale: float,
    default_scale: float,
    suggestion_scale: float,
) -> RandomUtilityModel | RuleBasedResponseModel:
    """Convert Gumbel decision-noise scale into equivalent normalized logits."""

    inverse_noise = 1.0 / point.decision_noise
    presentation = point.presentation_multiplier * inverse_noise
    parameters = {
        "beta": beta * inverse_noise,
        "ranking_scale": (
            rank_scale * presentation * point.rank_multiplier
        ),
        "default_scale": (
            default_scale * presentation * point.default_multiplier
        ),
        "suggestion_scale": (
            suggestion_scale * presentation * point.suggestion_multiplier
        ),
    }
    if point.response_model_family == "random_utility":
        return RandomUtilityModel(**parameters)
    return RuleBasedResponseModel(
        decision_noise=float(point.rule_noise),
        **parameters,
    )


def sensitivity_breadth_coverage(
    points: Sequence[SensitivityPoint],
    passing_rows: Sequence[Mapping[str, Any]],
) -> tuple[
    dict[str, list[float | int]],
    dict[str, dict[str, bool]],
    bool,
]:
    """Check meaningful-region survival at every declared sensitivity level.

    Rule noise is a conditional axis: only rule-based points declare it, and
    only passing rule-based rows can cover one of its levels. Policy dose is a
    visible manipulation/boundary axis but is deliberately not folded into the
    version-1 broad-simulator-parameter gate. Keeping this logic beside grid
    construction prevents the run-level Gate 6 report and the immutable
    cross-run reviewer from drifting apart.
    """

    axes = (
        "decision_noise",
        "presentation_multiplier",
        "rank_multiplier",
        "default_multiplier",
        "suggestion_multiplier",
        "profile_strength",
        "prior_uncertainty",
        "trajectory_length",
    )
    levels: dict[str, list[float | int]] = {
        axis: sorted({getattr(point, axis) for point in points})
        for axis in axes
    }
    levels["rule_noise"] = sorted(
        {
            float(point.rule_noise)
            for point in points
            if point.response_model_family == "rule_based"
            and point.rule_noise is not None
        }
    )
    survival = {
        axis: {
            str(level): any(
                row.get(axis) == level
                and (
                    axis != "rule_noise"
                    or row.get("response_model_family") == "rule_based"
                )
                for row in passing_rows
            )
            for level in axis_levels
        }
        for axis, axis_levels in levels.items()
    }
    passed = (
        all(len(axis_levels) >= 2 for axis_levels in levels.values())
        and all(
            all(axis_survival.values())
            for axis_survival in survival.values()
        )
    )
    return levels, survival, passed


@dataclass(frozen=True, slots=True)
class PhaseCriterion:
    """One preregisterable operational criterion for a sensitivity point."""

    criterion_id: str
    metric: str
    relation: str
    threshold: float

    def __post_init__(self) -> None:
        if not self.criterion_id or not self.metric:
            raise ValueError("phase criterion identifiers cannot be empty")
        if self.relation not in {"gt", "ge", "lt", "le"}:
            raise ValueError("phase criterion relation is invalid")
        if not math.isfinite(self.threshold):
            raise ValueError("phase criterion threshold must be finite")

    def evaluate(self, row: Mapping[str, Any]) -> bool | None:
        value = row.get(self.metric)
        if (
            value is None
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            return None
        numeric = float(value)
        if self.relation == "gt":
            return numeric > self.threshold
        if self.relation == "ge":
            return numeric >= self.threshold
        if self.relation == "lt":
            return numeric < self.threshold
        return numeric <= self.threshold

    def to_dict(self) -> dict[str, Any]:
        return {
            "criterion_id": self.criterion_id,
            "metric": self.metric,
            "relation": self.relation,
            "threshold": self.threshold,
        }


def classify_phase_point(
    row: Mapping[str, Any],
    criteria: Sequence[PhaseCriterion],
) -> dict[str, Any]:
    """Evaluate declared phase criteria without converting missing data to false."""

    material = tuple(criteria)
    if not material:
        raise ValueError("at least one phase criterion is required")
    evaluations = {
        criterion.criterion_id: criterion.evaluate(row)
        for criterion in material
    }
    complete = all(value is not None for value in evaluations.values())
    return {
        "point_id": row.get("point_id"),
        "criteria": evaluations,
        "criteria_complete": complete,
        "joint_region": (
            all(bool(value) for value in evaluations.values())
            if complete
            else None
        ),
    }


def infer_axis_boundaries(
    rows: Sequence[Mapping[str, Any]],
    criteria: Sequence[PhaseCriterion],
    *,
    axis: str,
) -> tuple[dict[str, Any], ...]:
    """Report observed pass intervals along one axis for each other-axis slice.

    This is an operational grid boundary, not a smooth fitted phase transition.
    Missing/non-numeric axis coordinates are rejected so the output cannot
    silently mix incomparable slices.
    """

    if not axis:
        raise ValueError("axis cannot be empty")
    classified = []
    coordinate_fields: set[str] = set()
    for row in rows:
        value = row.get(axis)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"row has invalid phase axis {axis!r}")
        phase = classify_phase_point(row, criteria)
        classified.append((row, phase))
        coordinate_fields.update(
            key
            for key in row
            if key in {
                "decision_noise",
                "presentation_multiplier",
                "profile_conditioning_strength",
                "rank_multiplier",
                "default_multiplier",
                "suggestion_multiplier",
                "profile_strength",
                "prior_uncertainty",
                "trajectory_length",
                "response_model_family",
                "rule_noise",
            }
        )
    slice_fields = tuple(sorted(coordinate_fields - {axis}))
    grouped: dict[tuple[Any, ...], list[tuple[float, bool | None]]] = {}
    for row, phase in classified:
        slice_key = tuple(row.get(field) for field in slice_fields)
        grouped.setdefault(slice_key, []).append(
            (float(row[axis]), phase["joint_region"])
        )
    output = []
    for slice_key, values in sorted(grouped.items(), key=lambda item: repr(item[0])):
        passing = sorted(
            coordinate
            for coordinate, result in values
            if result is True
        )
        output.append(
            {
                "schema_version": 1,
                "axis": axis,
                "slice": dict(zip(slice_fields, slice_key)),
                "evaluated_coordinates": sorted(
                    coordinate for coordinate, _ in values
                ),
                "passing_coordinates": passing,
                "observed_pass_min": passing[0] if passing else None,
                "observed_pass_max": passing[-1] if passing else None,
                "boundary_kind": "observed_grid_interval",
            }
        )
    return tuple(output)


def evaluate_grid(
    points: Iterable[SensitivityPoint],
    evaluator: Callable[[SensitivityPoint], dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Evaluate points in stable input order and retain all grid coordinates."""

    rows = []
    for point in points:
        outcome = evaluator(point)
        overlap = set(point.to_dict()) & set(outcome)
        if overlap:
            raise ValueError(
                "sensitivity evaluator overwrote coordinate fields: "
                + ", ".join(sorted(overlap))
            )
        rows.append({**point.to_dict(), **outcome})
    return tuple(rows)
