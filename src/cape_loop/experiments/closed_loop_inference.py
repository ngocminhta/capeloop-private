"""Clustered confirmatory summaries for Experiment B.

The estimands in this module are already paired at the complete-trajectory
level: profile-conditioned versus balanced trajectories, or system versus
same-history shadow.  Resampling then happens over complete latent users for
the primary interval and over complete trajectory pairs as a sensitivity
interval.  This is a deterministic nonparametric analysis, not a GLMM or
mixed-effects model.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Any, Sequence

from ..statistics import clustered_bootstrap_mean
from .closed_loop import ExperimentBResult


DEFAULT_MINIMUM_USER_CLUSTERS = 8
CLUSTER_UNITS = ("latent_user", "paired_trajectory")


@dataclass(frozen=True, slots=True)
class ExperimentBInterval:
    """One clustered interval for a declared Experiment B estimand."""

    metric_id: str
    updater_id: str
    initial_profile_condition: str
    cluster_unit: str
    estimand: str
    observation_unit: str
    observation_count: int
    cluster_count: int
    minimum_clusters: int
    bootstrap_replicates: int
    estimate: float
    lower: float | None
    upper: float | None
    confidence_level: float
    adequacy_status: str

    @property
    def adequate(self) -> bool:
        return self.adequacy_status == "adequate"

    @property
    def interval_available(self) -> bool:
        return self.lower is not None and self.upper is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "updater_id": self.updater_id,
            "initial_profile_condition": self.initial_profile_condition,
            "cluster_unit": self.cluster_unit,
            "estimand": self.estimand,
            "observation_unit": self.observation_unit,
            "observation_count": self.observation_count,
            "cluster_count": self.cluster_count,
            "minimum_clusters": self.minimum_clusters,
            "bootstrap_replicates": self.bootstrap_replicates,
            "estimate": self.estimate,
            "lower": self.lower,
            "upper": self.upper,
            "confidence_level": self.confidence_level,
            "adequacy_status": self.adequacy_status,
            "adequate": self.adequate,
            "method": (
                "percentile bootstrap over equally weighted complete clusters"
                if self.interval_available
                else "point estimate only; bootstrap disabled"
            ),
        }


@dataclass(frozen=True, slots=True)
class ExperimentBInference:
    """Deterministic collection of Experiment B clustered intervals."""

    intervals: tuple[ExperimentBInterval, ...]
    bootstrap_replicates: int
    minimum_user_clusters: int
    confidence_level: float

    def find(
        self,
        metric_id: str,
        updater_id: str,
        *,
        initial_profile_condition: str = "incorrect",
        cluster_unit: str = "latent_user",
    ) -> ExperimentBInterval | None:
        return next(
            (
                item
                for item in self.intervals
                if item.metric_id == metric_id
                and item.updater_id == updater_id
                and item.initial_profile_condition
                == initial_profile_condition
                and item.cluster_unit == cluster_unit
            ),
            None,
        )

    def gate_evidence(self, updater_id: str) -> dict[str, Any]:
        """Return only the predeclared primary user-clustered gate estimands."""

        metrics = {}
        for metric_id in (
            "mean_cumulative_lcg",
            "self_confirming_profile_rate",
            "profile_attribution_cost",
        ):
            interval = self.find(metric_id, updater_id)
            metrics[metric_id] = (
                None if interval is None else interval.to_dict()
            )
        return {
            "analysis": "experiment-b-clustered-bootstrap-v1",
            "target_updater_id": updater_id,
            "primary_cluster_unit": "latent_user",
            "bootstrap_replicates": self.bootstrap_replicates,
            "minimum_user_clusters": self.minimum_user_clusters,
            "metrics": metrics,
        }

    def to_dict(self) -> dict[str, Any]:
        statuses = {item.adequacy_status for item in self.intervals}
        if not self.intervals:
            status = "no_estimable_groups"
        elif statuses == {"adequate"}:
            status = "adequate"
        elif "not_computed" in statuses:
            status = "not_computed"
        else:
            status = "contains_inadequate_clusters"
        return {
            "schema_version": 1,
            "analysis_id": "experiment-b-clustered-bootstrap-v1",
            "analysis_status": status,
            "scientific_claim_status": "not_claimed",
            "primary_cluster_unit": "latent_user",
            "sensitivity_cluster_unit": "paired_trajectory",
            "bootstrap_replicates": self.bootstrap_replicates,
            "minimum_user_clusters": self.minimum_user_clusters,
            "confidence_level": self.confidence_level,
            "method": (
                "paired complete-trajectory estimands with deterministic "
                "percentile resampling of equally weighted clusters"
            ),
            "limitations": [
                "The primary interval clusters by latent user.",
                (
                    "The paired-trajectory interval is a sensitivity analysis "
                    "and must not be used to treat repeated trajectories as "
                    "independent users."
                ),
                "This analysis is not a GLMM or mixed-effects model.",
            ],
            "intervals": [item.to_dict() for item in self.intervals],
        }


def _cluster_mean(values: Sequence[float], cluster_ids: Sequence[str]) -> float:
    grouped: dict[str, list[float]] = {}
    for value, cluster_id in zip(values, cluster_ids):
        grouped.setdefault(cluster_id, []).append(float(value))
    return mean(mean(grouped[key]) for key in sorted(grouped))


def _interval(
    *,
    metric_id: str,
    updater_id: str,
    initial_profile_condition: str,
    cluster_unit: str,
    estimand: str,
    observation_unit: str,
    values: Sequence[float],
    cluster_ids: Sequence[str],
    bootstrap_replicates: int,
    minimum_clusters: int,
    confidence_level: float,
    seed: int,
) -> ExperimentBInterval:
    cluster_count = len(set(cluster_ids))
    estimate = _cluster_mean(values, cluster_ids)
    lower = None
    upper = None
    if bootstrap_replicates > 0:
        calculated = clustered_bootstrap_mean(
            values,
            cluster_ids,
            replicates=bootstrap_replicates,
            confidence_level=confidence_level,
            seed=seed,
            namespace=(
                "experiment-b:"
                f"{metric_id}:{updater_id}:{initial_profile_condition}:"
                f"{cluster_unit}"
            ),
        )
        estimate = calculated.estimate
        lower = calculated.lower
        upper = calculated.upper
    if bootstrap_replicates <= 0:
        adequacy_status = "not_computed"
    elif cluster_count < minimum_clusters:
        adequacy_status = "insufficient_clusters"
    else:
        adequacy_status = "adequate"
    return ExperimentBInterval(
        metric_id=metric_id,
        updater_id=updater_id,
        initial_profile_condition=initial_profile_condition,
        cluster_unit=cluster_unit,
        estimand=estimand,
        observation_unit=observation_unit,
        observation_count=len(values),
        cluster_count=cluster_count,
        minimum_clusters=minimum_clusters,
        bootstrap_replicates=bootstrap_replicates,
        estimate=estimate,
        lower=lower,
        upper=upper,
        confidence_level=confidence_level,
        adequacy_status=adequacy_status,
    )


def analyze_experiment_b_inference(
    result: ExperimentBResult,
    *,
    bootstrap_replicates: int,
    seed: int = 1729,
    minimum_user_clusters: int = DEFAULT_MINIMUM_USER_CLUSTERS,
    confidence_level: float = 0.95,
) -> ExperimentBInference:
    """Analyze paired Experiment B estimands with complete-cluster bootstrap.

    ``bootstrap_replicates=0`` intentionally produces point estimates marked
    ``not_computed``.  This supports inexpensive smoke runs while ensuring
    their stage gates cannot computationally pass.
    """

    if bootstrap_replicates < 0:
        raise ValueError("bootstrap_replicates must be non-negative")
    if minimum_user_clusters < 2:
        raise ValueError("minimum_user_clusters must be at least two")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0, 1)")

    intervals: list[ExperimentBInterval] = []
    decomposition_metrics = (
        (
            "evidence_selection_cost",
            "profile-policy shadow error - balanced-policy shadow error",
        ),
        (
            "profile_attribution_cost",
            "profile-policy system error - same-history shadow error",
        ),
        (
            "balanced_attribution_cost",
            "balanced-policy system error - same-history shadow error",
        ),
        (
            "self_confirmation_interaction",
            "profile attribution cost - balanced attribution cost",
        ),
    )
    decomposition_groups: dict[tuple[str, str], list[Any]] = {}
    for row in result.decompositions:
        decomposition_groups.setdefault(
            (row.updater_id, row.initial_profile_condition),
            [],
        ).append(row)
    for updater_id, condition in sorted(decomposition_groups):
        rows = decomposition_groups[(updater_id, condition)]
        for metric_id, estimand in decomposition_metrics:
            values = tuple(float(getattr(row, metric_id)) for row in rows)
            for cluster_unit in CLUSTER_UNITS:
                cluster_ids = (
                    tuple(row.user_id for row in rows)
                    if cluster_unit == "latent_user"
                    else tuple(
                        f"{row.profile_trajectory_id}|{row.balanced_trajectory_id}"
                        for row in rows
                    )
                )
                intervals.append(
                    _interval(
                        metric_id=metric_id,
                        updater_id=updater_id,
                        initial_profile_condition=condition,
                        cluster_unit=cluster_unit,
                        estimand=estimand,
                        observation_unit="paired_complete_trajectory",
                        values=values,
                        cluster_ids=cluster_ids,
                        bootstrap_replicates=bootstrap_replicates,
                        minimum_clusters=minimum_user_clusters,
                        confidence_level=confidence_level,
                        seed=seed,
                    )
                )

    trajectories = {
        trajectory.trajectory_id: trajectory
        for trajectory in result.trajectories
    }
    assessments: dict[str, list[Any]] = {}
    for assessment in result.self_confirmation_assessments:
        assessments.setdefault(assessment.trajectory_id, []).append(assessment)
    lcg_groups: dict[str, list[tuple[Any, float, float, float]]] = {}
    for trajectory_id, items in assessments.items():
        trajectory = trajectories[trajectory_id]
        if trajectory.policy_id != "soft_profile_conditioned":
            continue
        lcg_groups.setdefault(trajectory.updater_id, []).append(
            (
                trajectory,
                mean(item.evidence.cumulative_lcg for item in items),
                float(any(item.reportable for item in items)),
                float(
                    any(
                        item.evidence.profile_changed_later_action
                        for item in items
                    )
                ),
            )
        )
    trajectory_metrics = (
        (
            "mean_cumulative_lcg",
            1,
            "mean attribute-level system-minus-shadow log-odds gain",
        ),
        (
            "self_confirming_profile_rate",
            2,
            "fraction of profiles with at least one five-clause case",
        ),
        (
            "later_action_influence_rate",
            3,
            "fraction of profiles with strengthened-memory action influence",
        ),
    )
    for updater_id in sorted(lcg_groups):
        rows = lcg_groups[updater_id]
        for metric_id, value_index, estimand in trajectory_metrics:
            values = tuple(float(row[value_index]) for row in rows)
            for cluster_unit in CLUSTER_UNITS:
                cluster_ids = (
                    tuple(row[0].user_id for row in rows)
                    if cluster_unit == "latent_user"
                    else tuple(row[0].trajectory_id for row in rows)
                )
                intervals.append(
                    _interval(
                        metric_id=metric_id,
                        updater_id=updater_id,
                        initial_profile_condition="incorrect",
                        cluster_unit=cluster_unit,
                        estimand=estimand,
                        observation_unit=(
                            "complete_soft_profile_conditioned_trajectory"
                        ),
                        values=values,
                        cluster_ids=cluster_ids,
                        bootstrap_replicates=bootstrap_replicates,
                        minimum_clusters=minimum_user_clusters,
                        confidence_level=confidence_level,
                        seed=seed,
                    )
                )

    return ExperimentBInference(
        intervals=tuple(intervals),
        bootstrap_replicates=bootstrap_replicates,
        minimum_user_clusters=minimum_user_clusters,
        confidence_level=confidence_level,
    )
