"""Version-1 diagnostics retained beside the exact-oracle primary analysis.

The directional over-update and action-unaware-proximity criteria are no longer
the paper's primary hypotheses. They remain explicit, versioned diagnostics so
older pilot results stay interpretable. H7's update-error component uses the
exact generating-model reference.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

from ..beliefs import PreferenceBelief
from ..metrics import action_conditioned_update_error
from ..statistics import (
    IntervalEstimate,
    PairedContrast,
    clustered_bootstrap_mean,
    paired_cluster_contrast,
)
from .closed_loop import ExperimentBResult
from .provenance import ExperimentARow


HYPOTHESIS_ESTIMAND_SCHEMA_VERSION = 1
POLICY_CONDITIONED_MECHANISMS = ("restricted", "default", "suggested")
H2_REQUIRED_MECHANISMS = 2
H7_REQUIRED_SUPERIORITY_MECHANISMS = 2
H7_VALID_LEARNING_RETENTION_FRACTION = 0.80
MINIMUM_CLUSTER_COUNT = 2


def _computed_status(value: bool | None) -> str:
    if value is None:
        return "incomplete"
    return "criterion_met" if value else "criterion_not_met"


def _validate_analysis_settings(
    *,
    replicates: int,
    confidence_level: float,
) -> None:
    if replicates <= 0:
        raise ValueError("hypothesis analysis requires positive bootstrap replicates")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("hypothesis confidence level must lie in (0, 1)")


def _clipped_logit(probability: float, *, clip: float = 1e-6) -> float:
    bounded = min(max(float(probability), clip), 1.0 - clip)
    return math.log(bounded / (1.0 - bounded))


def _directional_log_odds_update(
    prior: PreferenceBelief,
    posterior: PreferenceBelief,
    attribute: int,
    direction: int,
) -> float:
    """Return the update toward the matched response's anchor direction."""

    return _clipped_logit(
        posterior.sign_mass(attribute, direction)
    ) - _clipped_logit(prior.sign_mass(attribute, direction))


def _row_index(
    rows: Sequence[ExperimentARow],
    *,
    response_mode: str,
) -> dict[tuple[str, str], ExperimentARow]:
    result: dict[tuple[str, str], ExperimentARow] = {}
    for row in rows:
        if row.response_mode != response_mode:
            continue
        key = (row.trial_id, row.updater_id)
        if key in result:
            raise ValueError(f"duplicate Experiment A row key {key}")
        result[key] = row
    return result


def _validate_pair(first: ExperimentARow, second: ExperimentARow) -> None:
    if (
        first.trial_id != second.trial_id
        or first.user_id != second.user_id
        or first.mechanism != second.mechanism
        or first.context != second.context
        or first.observation != second.observation
        or first.prior != second.prior
        or first.fitted_aware_posterior
        != second.fitted_aware_posterior
    ):
        raise ValueError(
            "hypothesis estimands require identical matched histories across "
            f"updaters ({first.trial_id!r}, {second.trial_id!r})"
        )


def _paired_rows(
    index: dict[tuple[str, str], ExperimentARow],
    *,
    first_updater_id: str,
    second_updater_id: str,
    mechanism: str,
) -> tuple[tuple[ExperimentARow, ExperimentARow], ...]:
    pairs = []
    for (trial_id, updater_id), first in sorted(index.items()):
        if updater_id != first_updater_id or first.mechanism != mechanism:
            continue
        second = index.get((trial_id, second_updater_id))
        if second is None:
            continue
        _validate_pair(first, second)
        pairs.append((first, second))
    return tuple(pairs)


@dataclass(frozen=True, slots=True)
class H1MechanismEstimand:
    """Full-context minus fitted-aware update contrasts for one mechanism."""

    mechanism: str
    directional_update: PairedContrast
    update_strength: PairedContrast
    minimum_cluster_count: int
    criterion_met: bool | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mechanism": self.mechanism,
            "directional_update": self.directional_update.to_dict(),
            "update_strength": self.update_strength.to_dict(),
            "minimum_cluster_count": self.minimum_cluster_count,
            "criterion": (
                "both one-sided evidence conditions hold: the 95% "
                "cluster-bootstrap lower bound for full-context minus "
                "fitted-aware directional update is > 0, and the lower bound "
                "for the corresponding absolute update-strength contrast is > 0"
            ),
            "criterion_met": self.criterion_met,
            "computed_status": _computed_status(self.criterion_met),
        }


@dataclass(frozen=True, slots=True)
class H1Analysis:
    target_updater_id: str
    reference_updater_id: str
    response_mode: str
    required_mechanisms: tuple[str, ...]
    estimands: tuple[H1MechanismEstimand, ...]
    missing_mechanisms: tuple[str, ...]
    inadequate_cluster_mechanisms: tuple[str, ...]
    criterion_met: bool | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": "H1",
            "analysis_role": "secondary_diagnostic",
            "name": "Directional causal-provenance over-update",
            "target_updater_id": self.target_updater_id,
            "reference_updater_id": self.reference_updater_id,
            "response_mode": self.response_mode,
            "required_mechanisms": list(self.required_mechanisms),
            "missing_mechanisms": list(self.missing_mechanisms),
            "inadequate_cluster_mechanisms": list(
                self.inadequate_cluster_mechanisms
            ),
            "estimand": {
                "directional_update": (
                    "Delta_log_odds_anchor(full_context) - "
                    "Delta_log_odds_anchor(fitted_action_aware)"
                ),
                "update_strength": (
                    "abs(Delta_log_odds_anchor(full_context)) - "
                    "abs(Delta_log_odds_anchor(fitted_action_aware))"
                ),
                "aggregation": (
                    "mean within complete latent user, followed by a "
                    "complete-user percentile cluster bootstrap"
                ),
            },
            "mechanisms": [item.to_dict() for item in self.estimands],
            "complete": (
                not self.missing_mechanisms
                and not self.inadequate_cluster_mechanisms
            ),
            "criterion": (
                "every registered policy-conditioned mechanism must satisfy "
                "both mechanism-wise one-sided contrasts"
            ),
            "criterion_met": self.criterion_met,
            "computed_status": _computed_status(self.criterion_met),
            "claim_status": "not_claimed",
        }


def analyze_h1(
    rows: Sequence[ExperimentARow],
    *,
    target_updater_id: str = "llm_full_context",
    reference_updater_id: str = "fitted_action_aware",
    response_mode: str = "controlled_anchor",
    mechanisms: Sequence[str] = POLICY_CONDITIONED_MECHANISMS,
    replicates: int = 2000,
    confidence_level: float = 0.95,
    minimum_cluster_count: int = MINIMUM_CLUSTER_COUNT,
    seed: int = 1729,
) -> H1Analysis:
    """Estimate H1 directly on anchor-aligned update direction and strength.

    The fitted-aware posterior retained on each Experiment A row is the
    reference.  Therefore H1 remains estimable even when the configured output
    rows omit a redundant ``fitted_action_aware`` updater row.
    """

    required = tuple(mechanisms)
    if target_updater_id != "llm_full_context":
        raise ValueError("H1 target is frozen to llm_full_context")
    if reference_updater_id != "fitted_action_aware":
        raise ValueError("H1 reference is frozen to fitted_action_aware")
    if response_mode != "controlled_anchor":
        raise ValueError("H1 response mode is frozen to controlled_anchor")
    if not required or len(required) != len(set(required)):
        raise ValueError("H1 mechanisms must be non-empty and distinct")
    if minimum_cluster_count < 2:
        raise ValueError("H1 requires at least two independent user clusters")
    _validate_analysis_settings(
        replicates=replicates,
        confidence_level=confidence_level,
    )
    index = _row_index(rows, response_mode=response_mode)
    estimates = []
    missing = []
    inadequate = []
    for mechanism in required:
        target_rows = tuple(
            row
            for (trial_id, updater_id), row in sorted(index.items())
            if updater_id == target_updater_id and row.mechanism == mechanism
        )
        if not target_rows:
            missing.append(mechanism)
            continue
        target_direction = [
            _directional_log_odds_update(
                row.prior,
                row.posterior,
                row.target_attribute,
                row.anchor_direction,
            )
            for row in target_rows
        ]
        aware_direction = [
            _directional_log_odds_update(
                row.prior,
                row.fitted_aware_posterior,
                row.target_attribute,
                row.anchor_direction,
            )
            for row in target_rows
        ]
        cluster_ids = [row.user_id for row in target_rows]
        directional = paired_cluster_contrast(
            target_direction,
            aware_direction,
            cluster_ids,
            contrast_id=f"H1:{mechanism}:directional-update",
            first_label=target_updater_id,
            second_label=reference_updater_id,
            replicates=replicates,
            confidence_level=confidence_level,
            seed=seed,
        )
        strength = paired_cluster_contrast(
            [abs(value) for value in target_direction],
            [abs(value) for value in aware_direction],
            cluster_ids,
            contrast_id=f"H1:{mechanism}:update-strength",
            first_label=f"abs({target_updater_id})",
            second_label=f"abs({reference_updater_id})",
            replicates=replicates,
            confidence_level=confidence_level,
            seed=seed,
        )
        enough_clusters = (
            directional.interval.cluster_count >= minimum_cluster_count
            and strength.interval.cluster_count >= minimum_cluster_count
        )
        if not enough_clusters:
            inadequate.append(mechanism)
        estimates.append(
            H1MechanismEstimand(
                mechanism=mechanism,
                directional_update=directional,
                update_strength=strength,
                minimum_cluster_count=minimum_cluster_count,
                criterion_met=(
                    directional.interval.lower > 0.0
                    and strength.interval.lower > 0.0
                    if enough_clusters
                    else None
                ),
            )
        )
    complete = (
        not missing
        and not inadequate
        and len(estimates) == len(required)
    )
    return H1Analysis(
        target_updater_id=target_updater_id,
        reference_updater_id=reference_updater_id,
        response_mode=response_mode,
        required_mechanisms=required,
        estimands=tuple(estimates),
        missing_mechanisms=tuple(missing),
        inadequate_cluster_mechanisms=tuple(inadequate),
        criterion_met=(
            all(item.criterion_met is True for item in estimates)
            if complete
            else None
        ),
    )


@dataclass(frozen=True, slots=True)
class H2MechanismEstimand:
    """Relative update-vector proximity for one provenance mechanism."""

    mechanism: str
    distance_to_aware: IntervalEstimate
    distance_to_unaware: IntervalEstimate
    aware_minus_unaware_distance: PairedContrast
    minimum_cluster_count: int
    criterion_met: bool | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mechanism": self.mechanism,
            "distance_to_aware": self.distance_to_aware.to_dict(),
            "distance_to_unaware": self.distance_to_unaware.to_dict(),
            "aware_minus_unaware_distance": (
                self.aware_minus_unaware_distance.to_dict()
            ),
            "minimum_cluster_count": self.minimum_cluster_count,
            "criterion": (
                "the 95% complete-user cluster-bootstrap lower bound for "
                "distance(full, aware) - distance(full, unaware) is > 0"
            ),
            "criterion_met": self.criterion_met,
            "computed_status": _computed_status(self.criterion_met),
        }


@dataclass(frozen=True, slots=True)
class H2Analysis:
    target_updater_id: str
    aware_updater_id: str
    unaware_updater_id: str
    response_mode: str
    required_qualifying_mechanisms: int
    estimands: tuple[H2MechanismEstimand, ...]
    missing_mechanisms: tuple[str, ...]
    inadequate_cluster_mechanisms: tuple[str, ...]
    qualifying_mechanisms: tuple[str, ...]
    criterion_met: bool | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": "H2",
            "analysis_role": "secondary_diagnostic",
            "name": "Context visibility is insufficient",
            "target_updater_id": self.target_updater_id,
            "aware_updater_id": self.aware_updater_id,
            "unaware_updater_id": self.unaware_updater_id,
            "response_mode": self.response_mode,
            "distance": (
                "L1 distance between the updater's and reference updater's "
                "12 marginal-probability increments"
            ),
            "proximity_advantage": (
                "distance(full_context, fitted_action_aware) - "
                "distance(full_context, fitted_action_unaware); positive "
                "values mean closer to action-unaware"
            ),
            "required_qualifying_mechanisms": (
                self.required_qualifying_mechanisms
            ),
            "missing_mechanisms": list(self.missing_mechanisms),
            "inadequate_cluster_mechanisms": list(
                self.inadequate_cluster_mechanisms
            ),
            "qualifying_mechanisms": list(self.qualifying_mechanisms),
            "mechanisms": [item.to_dict() for item in self.estimands],
            "complete": (
                sum(
                    item.criterion_met is not None
                    for item in self.estimands
                )
                >= self.required_qualifying_mechanisms
            ),
            "full_mechanism_coverage": not self.missing_mechanisms,
            "criterion": (
                "the action-unaware proximity criterion must hold on at least "
                f"{self.required_qualifying_mechanisms} mechanisms"
            ),
            "criterion_met": self.criterion_met,
            "computed_status": _computed_status(self.criterion_met),
            "claim_status": "not_claimed",
        }


def analyze_h2(
    rows: Sequence[ExperimentARow],
    *,
    target_updater_id: str = "llm_full_context",
    aware_updater_id: str = "fitted_action_aware",
    unaware_updater_id: str = "fitted_action_unaware",
    response_mode: str = "controlled_anchor",
    mechanisms: Sequence[str] = POLICY_CONDITIONED_MECHANISMS,
    required_qualifying_mechanisms: int = H2_REQUIRED_MECHANISMS,
    replicates: int = 2000,
    confidence_level: float = 0.95,
    minimum_cluster_count: int = MINIMUM_CLUSTER_COUNT,
    seed: int = 1729,
) -> H2Analysis:
    """Test whether full-context updates are closer to unaware than aware."""

    required = tuple(mechanisms)
    if target_updater_id != "llm_full_context":
        raise ValueError("H2 target is frozen to llm_full_context")
    if aware_updater_id != "fitted_action_aware":
        raise ValueError("H2 aware reference is frozen to fitted_action_aware")
    if unaware_updater_id != "fitted_action_unaware":
        raise ValueError(
            "H2 unaware reference is frozen to fitted_action_unaware"
        )
    if response_mode != "controlled_anchor":
        raise ValueError("H2 response mode is frozen to controlled_anchor")
    if not required or len(required) != len(set(required)):
        raise ValueError("H2 mechanisms must be non-empty and distinct")
    if not 1 <= required_qualifying_mechanisms <= len(required):
        raise ValueError(
            "required H2 mechanisms must lie between one and mechanism count"
        )
    if minimum_cluster_count < 2:
        raise ValueError("H2 requires at least two independent user clusters")
    _validate_analysis_settings(
        replicates=replicates,
        confidence_level=confidence_level,
    )
    index = _row_index(rows, response_mode=response_mode)
    estimates = []
    missing = []
    inadequate = []
    for mechanism in required:
        pairs = _paired_rows(
            index,
            first_updater_id=target_updater_id,
            second_updater_id=unaware_updater_id,
            mechanism=mechanism,
        )
        if not pairs:
            missing.append(mechanism)
            continue
        distance_to_aware = [
            action_conditioned_update_error(
                target.prior,
                target.posterior,
                target.prior,
                target.fitted_aware_posterior,
            )
            for target, _ in pairs
        ]
        distance_to_unaware = [
            action_conditioned_update_error(
                target.prior,
                target.posterior,
                unaware.prior,
                unaware.posterior,
            )
            for target, unaware in pairs
        ]
        cluster_ids = [target.user_id for target, _ in pairs]
        aware_interval = clustered_bootstrap_mean(
            distance_to_aware,
            cluster_ids,
            replicates=replicates,
            confidence_level=confidence_level,
            seed=seed,
            namespace=f"H2:{mechanism}:distance-aware",
        )
        unaware_interval = clustered_bootstrap_mean(
            distance_to_unaware,
            cluster_ids,
            replicates=replicates,
            confidence_level=confidence_level,
            seed=seed,
            namespace=f"H2:{mechanism}:distance-unaware",
        )
        advantage = paired_cluster_contrast(
            distance_to_aware,
            distance_to_unaware,
            cluster_ids,
            contrast_id=f"H2:{mechanism}:aware-minus-unaware-distance",
            first_label=f"distance({target_updater_id}, {aware_updater_id})",
            second_label=f"distance({target_updater_id}, {unaware_updater_id})",
            replicates=replicates,
            confidence_level=confidence_level,
            seed=seed,
        )
        enough_clusters = (
            advantage.interval.cluster_count >= minimum_cluster_count
        )
        if not enough_clusters:
            inadequate.append(mechanism)
        estimates.append(
            H2MechanismEstimand(
                mechanism=mechanism,
                distance_to_aware=aware_interval,
                distance_to_unaware=unaware_interval,
                aware_minus_unaware_distance=advantage,
                minimum_cluster_count=minimum_cluster_count,
                criterion_met=(
                    advantage.interval.lower > 0.0
                    if enough_clusters
                    else None
                ),
            )
        )
    qualifying = tuple(
        item.mechanism for item in estimates if item.criterion_met is True
    )
    evaluable = (
        sum(item.criterion_met is not None for item in estimates)
        >= required_qualifying_mechanisms
    )
    return H2Analysis(
        target_updater_id=target_updater_id,
        aware_updater_id=aware_updater_id,
        unaware_updater_id=unaware_updater_id,
        response_mode=response_mode,
        required_qualifying_mechanisms=required_qualifying_mechanisms,
        estimands=tuple(estimates),
        missing_mechanisms=tuple(missing),
        inadequate_cluster_mechanisms=tuple(inadequate),
        qualifying_mechanisms=qualifying,
        criterion_met=(
            len(qualifying) >= required_qualifying_mechanisms
            if evaluable
            else None
        ),
    )


@dataclass(frozen=True, slots=True)
class VolunteeredPreferenceUpdate:
    """Imported direct-statement outcome for H7's volunteered positive control."""

    case_id: str
    user_id: str
    updater_id: str
    directional_log_odds_update: float

    def __post_init__(self) -> None:
        if not self.case_id or not self.user_id or not self.updater_id:
            raise ValueError("volunteered update IDs must be non-empty")
        if not math.isfinite(self.directional_log_odds_update):
            raise ValueError("volunteered directional update must be finite")

    def to_dict(self) -> dict[str, float | str]:
        return {
            "case_id": self.case_id,
            "user_id": self.user_id,
            "updater_id": self.updater_id,
            "directional_log_odds_update": (
                self.directional_log_odds_update
            ),
        }


def analyze_h7_volunteered_updates(
    volunteered_updates: Sequence[VolunteeredPreferenceUpdate],
    *,
    baseline_updater_id: str = "llm_full_context",
    mitigation_updater_id: str = "llm_provenance_aware",
    retention_fraction: float = H7_VALID_LEARNING_RETENTION_FRACTION,
    replicates: int = 2000,
    confidence_level: float = 0.95,
    minimum_cluster_count: int = MINIMUM_CLUSTER_COUNT,
    seed: int = 1729,
) -> H7ValidLearning:
    """Analyze explicit, paired direct-statement outcomes for H7.

    This public boundary is intentionally narrower than
    :func:`analyze_h7_experiment_a`: it accepts only genuinely supplied
    direct-statement updates.  It never derives or imputes a volunteered
    outcome from a balanced-choice row.
    """

    if baseline_updater_id != "llm_full_context":
        raise ValueError("H7 baseline is frozen to llm_full_context")
    if mitigation_updater_id != "llm_provenance_aware":
        raise ValueError("H7 mitigation is frozen to llm_provenance_aware")
    if not 0.0 < retention_fraction <= 1.0:
        raise ValueError("H7 retention fraction must lie in (0, 1]")
    if minimum_cluster_count < 2:
        raise ValueError("H7 requires at least two independent user clusters")
    _validate_analysis_settings(
        replicates=replicates,
        confidence_level=confidence_level,
    )

    volunteer_index: dict[
        tuple[str, str], VolunteeredPreferenceUpdate
    ] = {}
    for record in volunteered_updates:
        if not isinstance(record, VolunteeredPreferenceUpdate):
            raise TypeError(
                "volunteered_updates must contain "
                "VolunteeredPreferenceUpdate records"
            )
        key = (record.case_id, record.updater_id)
        if key in volunteer_index:
            raise ValueError(f"duplicate volunteered update key {key}")
        volunteer_index[key] = record

    permitted_updaters = {
        baseline_updater_id,
        mitigation_updater_id,
    }
    unexpected = sorted(
        {
            record.updater_id
            for record in volunteered_updates
            if record.updater_id not in permitted_updaters
        }
    )
    if unexpected:
        raise ValueError(
            "volunteered updates contain unexpected updater IDs: "
            + ", ".join(unexpected)
        )

    baseline_case_ids = {
        case_id
        for case_id, updater_id in volunteer_index
        if updater_id == baseline_updater_id
    }
    mitigation_case_ids = {
        case_id
        for case_id, updater_id in volunteer_index
        if updater_id == mitigation_updater_id
    }
    if baseline_case_ids != mitigation_case_ids:
        raise ValueError(
            "volunteered direct-statement coverage must be exactly paired; "
            f"missing_baseline={sorted(mitigation_case_ids - baseline_case_ids)}, "
            f"missing_mitigation={sorted(baseline_case_ids - mitigation_case_ids)}"
        )

    volunteer_pairs: list[
        tuple[VolunteeredPreferenceUpdate, VolunteeredPreferenceUpdate]
    ] = []
    for case_id in sorted(baseline_case_ids):
        baseline_record = volunteer_index[
            (case_id, baseline_updater_id)
        ]
        mitigation_record = volunteer_index[
            (case_id, mitigation_updater_id)
        ]
        if baseline_record.user_id != mitigation_record.user_id:
            raise ValueError(
                "paired volunteered updates must share a user cluster"
            )
        volunteer_pairs.append((baseline_record, mitigation_record))

    return _valid_learning_result(
        condition="volunteered",
        baseline=[
            baseline.directional_log_odds_update
            for baseline, _ in volunteer_pairs
        ],
        mitigation=[
            mitigation.directional_log_odds_update
            for _, mitigation in volunteer_pairs
        ],
        cluster_ids=[
            baseline.user_id for baseline, _ in volunteer_pairs
        ],
        baseline_updater_id=baseline_updater_id,
        mitigation_updater_id=mitigation_updater_id,
        retention_fraction=retention_fraction,
        minimum_cluster_count=minimum_cluster_count,
        replicates=replicates,
        confidence_level=confidence_level,
        seed=seed,
        missing_reason=(
            "no paired volunteered direct-statement outcomes were supplied; "
            "the one-step choice runner must not invent this positive control"
        ),
    )


@dataclass(frozen=True, slots=True)
class H7MechanismSuperiority:
    mechanism: str
    acue_reduction: PairedContrast
    minimum_cluster_count: int
    criterion_met: bool | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mechanism": self.mechanism,
            "reference_basis": "exact_action_aware",
            "acue_reduction": self.acue_reduction.to_dict(),
            "minimum_cluster_count": self.minimum_cluster_count,
            "criterion": (
                "the 95% complete-user cluster-bootstrap lower bound for "
                "ACUE(full_context) - ACUE(provenance_aware) is > 0"
            ),
            "criterion_met": self.criterion_met,
            "computed_status": _computed_status(self.criterion_met),
        }


@dataclass(frozen=True, slots=True)
class H7ValidLearning:
    condition: str
    pair_count: int
    baseline_directional_update: IntervalEstimate | None
    mitigation_directional_update: IntervalEstimate | None
    retention_contrast: PairedContrast | None
    retention_fraction: float
    minimum_cluster_count: int
    criterion_met: bool | None
    missing_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "pair_count": self.pair_count,
            "baseline_directional_update": (
                None
                if self.baseline_directional_update is None
                else self.baseline_directional_update.to_dict()
            ),
            "mitigation_directional_update": (
                None
                if self.mitigation_directional_update is None
                else self.mitigation_directional_update.to_dict()
            ),
            "retention_contrast": (
                None
                if self.retention_contrast is None
                else self.retention_contrast.to_dict()
            ),
            "retention_fraction": self.retention_fraction,
            "minimum_cluster_count": self.minimum_cluster_count,
            "criterion": (
                "the unmitigated full-context positive-control update has a "
                "95% cluster-bootstrap lower bound > 0 and the lower bound for "
                "provenance-aware update - retention_fraction * full-context "
                "update is >= 0"
            ),
            "criterion_met": self.criterion_met,
            "computed_status": _computed_status(self.criterion_met),
            "missing_reason": self.missing_reason,
        }


def _valid_learning_result(
    *,
    condition: str,
    baseline: Sequence[float],
    mitigation: Sequence[float],
    cluster_ids: Sequence[str],
    baseline_updater_id: str,
    mitigation_updater_id: str,
    retention_fraction: float,
    minimum_cluster_count: int,
    replicates: int,
    confidence_level: float,
    seed: int,
    missing_reason: str,
) -> H7ValidLearning:
    if len(baseline) != len(mitigation) or len(baseline) != len(cluster_ids):
        raise ValueError(
            "valid-learning baseline, mitigation, and cluster inputs must align"
        )
    if not baseline:
        return H7ValidLearning(
            condition=condition,
            pair_count=0,
            baseline_directional_update=None,
            mitigation_directional_update=None,
            retention_contrast=None,
            retention_fraction=retention_fraction,
            minimum_cluster_count=minimum_cluster_count,
            criterion_met=None,
            missing_reason=missing_reason,
        )
    baseline_interval = clustered_bootstrap_mean(
        baseline,
        cluster_ids,
        replicates=replicates,
        confidence_level=confidence_level,
        seed=seed,
        namespace=f"H7:{condition}:baseline-valid-update",
    )
    mitigation_interval = clustered_bootstrap_mean(
        mitigation,
        cluster_ids,
        replicates=replicates,
        confidence_level=confidence_level,
        seed=seed,
        namespace=f"H7:{condition}:mitigation-valid-update",
    )
    retained_values = tuple(
        mitigation_value - retention_fraction * baseline_value
        for baseline_value, mitigation_value in zip(baseline, mitigation)
    )
    retained_interval = clustered_bootstrap_mean(
        retained_values,
        cluster_ids,
        replicates=replicates,
        confidence_level=confidence_level,
        seed=seed,
        namespace=f"H7:{condition}:retention-noninferiority",
    )
    contrast = PairedContrast(
        contrast_id=f"H7:{condition}:valid-learning-retention",
        expression=(
            f"{mitigation_updater_id} - {retention_fraction:.2f} * "
            f"{baseline_updater_id}"
        ),
        pair_count=len(retained_values),
        interval=retained_interval,
    )
    enough_clusters = (
        baseline_interval.cluster_count >= minimum_cluster_count
        and mitigation_interval.cluster_count >= minimum_cluster_count
        and retained_interval.cluster_count >= minimum_cluster_count
    )
    return H7ValidLearning(
        condition=condition,
        pair_count=len(baseline),
        baseline_directional_update=baseline_interval,
        mitigation_directional_update=mitigation_interval,
        retention_contrast=contrast,
        retention_fraction=retention_fraction,
        minimum_cluster_count=minimum_cluster_count,
        criterion_met=(
            baseline_interval.lower > 0.0
            and retained_interval.lower >= 0.0
            if enough_clusters
            else None
        ),
        missing_reason=None,
    )


@dataclass(frozen=True, slots=True)
class H7ExperimentAAnalysis:
    baseline_updater_id: str
    mitigation_updater_id: str
    response_mode: str
    required_superiority_mechanisms: int
    superiority_estimands: tuple[H7MechanismSuperiority, ...]
    qualifying_superiority_mechanisms: tuple[str, ...]
    missing_superiority_mechanisms: tuple[str, ...]
    inadequate_cluster_mechanisms: tuple[str, ...]
    balanced_valid_learning: H7ValidLearning
    volunteered_valid_learning: H7ValidLearning
    criterion_met: bool | None

    def to_dict(self) -> dict[str, Any]:
        superiority_evaluable = (
            sum(
                item.criterion_met is not None
                for item in self.superiority_estimands
            )
            >= self.required_superiority_mechanisms
        )
        return {
            "hypothesis_id": "H7",
            "name": "Causal provenance is actionable",
            "component": "experiment_a_update_error_and_valid_learning",
            "baseline_updater_id": self.baseline_updater_id,
            "mitigation_updater_id": self.mitigation_updater_id,
            "response_mode": self.response_mode,
            "required_superiority_mechanisms": (
                self.required_superiority_mechanisms
            ),
            "qualifying_superiority_mechanisms": list(
                self.qualifying_superiority_mechanisms
            ),
            "missing_superiority_mechanisms": list(
                self.missing_superiority_mechanisms
            ),
            "inadequate_cluster_mechanisms": list(
                self.inadequate_cluster_mechanisms
            ),
            "superiority_estimands": [
                item.to_dict() for item in self.superiority_estimands
            ],
            "balanced_valid_learning": (
                self.balanced_valid_learning.to_dict()
            ),
            "volunteered_valid_learning": (
                self.volunteered_valid_learning.to_dict()
            ),
            "retention_noninferiority_margin": {
                "retention_fraction": (
                    self.balanced_valid_learning.retention_fraction
                ),
                "maximum_relative_loss": (
                    1.0
                    - self.balanced_valid_learning.retention_fraction
                ),
            },
            "complete": (
                superiority_evaluable
                and self.balanced_valid_learning.criterion_met is not None
                and self.volunteered_valid_learning.criterion_met is not None
            ),
            "criterion": (
                "ACUE superiority on at least the required number of "
                "policy-conditioned mechanisms, plus directional valid-learning "
                "noninferiority for both balanced and volunteered evidence"
            ),
            "criterion_met": self.criterion_met,
            "computed_status": _computed_status(self.criterion_met),
            "claim_status": "not_claimed",
            "scope_note": (
                "This is H7's Experiment A update-error component. The "
                "closed-loop self-confirmation component is emitted separately "
                "by Experiment B and is also required for the full H7 claim."
            ),
        }


def analyze_h7_experiment_a(
    rows: Sequence[ExperimentARow],
    *,
    volunteered_updates: Sequence[VolunteeredPreferenceUpdate] = (),
    baseline_updater_id: str = "llm_full_context",
    mitigation_updater_id: str = "llm_provenance_aware",
    response_mode: str = "controlled_anchor",
    mechanisms: Sequence[str] = POLICY_CONDITIONED_MECHANISMS,
    required_superiority_mechanisms: int = (
        H7_REQUIRED_SUPERIORITY_MECHANISMS
    ),
    retention_fraction: float = H7_VALID_LEARNING_RETENTION_FRACTION,
    replicates: int = 2000,
    confidence_level: float = 0.95,
    minimum_cluster_count: int = MINIMUM_CLUSTER_COUNT,
    seed: int = 1729,
) -> H7ExperimentAAnalysis:
    """Estimate H7 update-error superiority and valid-learning retention."""

    required = tuple(mechanisms)
    if baseline_updater_id != "llm_full_context":
        raise ValueError("H7 baseline is frozen to llm_full_context")
    if mitigation_updater_id != "llm_provenance_aware":
        raise ValueError("H7 mitigation is frozen to llm_provenance_aware")
    if response_mode != "controlled_anchor":
        raise ValueError(
            "H7 Experiment A response mode is frozen to controlled_anchor"
        )
    if not required or len(required) != len(set(required)):
        raise ValueError("H7 mechanisms must be non-empty and distinct")
    if not 1 <= required_superiority_mechanisms <= len(required):
        raise ValueError(
            "required H7 superiority mechanisms must lie within mechanism count"
        )
    if not 0.0 < retention_fraction <= 1.0:
        raise ValueError("H7 retention fraction must lie in (0, 1]")
    if minimum_cluster_count < 2:
        raise ValueError("H7 requires at least two independent user clusters")
    _validate_analysis_settings(
        replicates=replicates,
        confidence_level=confidence_level,
    )
    index = _row_index(rows, response_mode=response_mode)
    superiority = []
    missing = []
    inadequate = []
    for mechanism in required:
        pairs = _paired_rows(
            index,
            first_updater_id=baseline_updater_id,
            second_updater_id=mitigation_updater_id,
            mechanism=mechanism,
        )
        if not pairs:
            missing.append(mechanism)
            continue
        reduction = paired_cluster_contrast(
            [baseline.exact_acue for baseline, _ in pairs],
            [mitigation.exact_acue for _, mitigation in pairs],
            [baseline.user_id for baseline, _ in pairs],
            contrast_id=f"H7:{mechanism}:exact-acue-reduction",
            first_label=baseline_updater_id,
            second_label=mitigation_updater_id,
            replicates=replicates,
            confidence_level=confidence_level,
            seed=seed,
        )
        enough_clusters = (
            reduction.interval.cluster_count >= minimum_cluster_count
        )
        if not enough_clusters:
            inadequate.append(mechanism)
        superiority.append(
            H7MechanismSuperiority(
                mechanism=mechanism,
                acue_reduction=reduction,
                minimum_cluster_count=minimum_cluster_count,
                criterion_met=(
                    reduction.interval.lower > 0.0
                    if enough_clusters
                    else None
                ),
            )
        )
    qualifying = tuple(
        item.mechanism
        for item in superiority
        if item.criterion_met is True
    )

    balanced_pairs = _paired_rows(
        index,
        first_updater_id=baseline_updater_id,
        second_updater_id=mitigation_updater_id,
        mechanism="balanced",
    )
    balanced_baseline = [
        _directional_log_odds_update(
            baseline.prior,
            baseline.posterior,
            baseline.target_attribute,
            baseline.anchor_direction,
        )
        for baseline, _ in balanced_pairs
    ]
    balanced_mitigation = [
        _directional_log_odds_update(
            mitigation.prior,
            mitigation.posterior,
            mitigation.target_attribute,
            mitigation.anchor_direction,
        )
        for _, mitigation in balanced_pairs
    ]
    balanced = _valid_learning_result(
        condition="balanced",
        baseline=balanced_baseline,
        mitigation=balanced_mitigation,
        cluster_ids=[
            baseline.user_id for baseline, _ in balanced_pairs
        ],
        baseline_updater_id=baseline_updater_id,
        mitigation_updater_id=mitigation_updater_id,
        retention_fraction=retention_fraction,
        minimum_cluster_count=minimum_cluster_count,
        replicates=replicates,
        confidence_level=confidence_level,
        seed=seed,
        missing_reason=(
            "no paired balanced full-context/provenance-aware rows"
        ),
    )

    volunteered = analyze_h7_volunteered_updates(
        volunteered_updates,
        baseline_updater_id=baseline_updater_id,
        mitigation_updater_id=mitigation_updater_id,
        retention_fraction=retention_fraction,
        replicates=replicates,
        confidence_level=confidence_level,
        minimum_cluster_count=minimum_cluster_count,
        seed=seed,
    )
    superiority_evaluable = (
        sum(item.criterion_met is not None for item in superiority)
        >= required_superiority_mechanisms
    )
    complete = (
        superiority_evaluable
        and balanced.criterion_met is not None
        and volunteered.criterion_met is not None
    )
    return H7ExperimentAAnalysis(
        baseline_updater_id=baseline_updater_id,
        mitigation_updater_id=mitigation_updater_id,
        response_mode=response_mode,
        required_superiority_mechanisms=required_superiority_mechanisms,
        superiority_estimands=tuple(superiority),
        qualifying_superiority_mechanisms=qualifying,
        missing_superiority_mechanisms=tuple(missing),
        inadequate_cluster_mechanisms=tuple(inadequate),
        balanced_valid_learning=balanced,
        volunteered_valid_learning=volunteered,
        criterion_met=(
            len(qualifying) >= required_superiority_mechanisms
            and balanced.criterion_met is True
            and volunteered.criterion_met is True
            if complete
            else None
        ),
    )


@dataclass(frozen=True, slots=True)
class ExperimentAHypothesisEstimands:
    h1: H1Analysis
    h2: H2Analysis
    h7: H7ExperimentAAnalysis
    bootstrap_replicates: int
    confidence_level: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": HYPOTHESIS_ESTIMAND_SCHEMA_VERSION,
            "analysis": "experiment_a_hypothesis_estimands",
            "analysis_role": (
                "secondary_directional_diagnostics_plus_mitigation_checks"
            ),
            "primary_analysis_artifact": (
                "metrics/experiment-a-exact-calibration.json"
            ),
            "independent_unit": "complete latent user",
            "bootstrap_replicates": self.bootstrap_replicates,
            "confidence_level": self.confidence_level,
            "frozen_decision_constants": {
                "policy_conditioned_mechanisms": list(
                    POLICY_CONDITIONED_MECHANISMS
                ),
                "minimum_cluster_count": MINIMUM_CLUSTER_COUNT,
                "h2_required_mechanisms": H2_REQUIRED_MECHANISMS,
                "h7_required_superiority_mechanisms": (
                    H7_REQUIRED_SUPERIORITY_MECHANISMS
                ),
                "h7_valid_learning_retention_fraction": (
                    H7_VALID_LEARNING_RETENTION_FRACTION
                ),
            },
            "hypotheses": {
                "H1": self.h1.to_dict(),
                "H2": self.h2.to_dict(),
                "H7": self.h7.to_dict(),
            },
            "claim_status": "not_claimed",
            "interpretation": (
                "H1/H2 here are retained version-1 directional diagnostics, "
                "not the primary provenance claim. The primary Experiment A "
                "estimand is exact-oracle mechanism-specific calibration. A "
                "computed criterion is not an empirical paper claim; "
                "preregistration, sample adequacy, multiplicity review, and "
                "the other H7 component remain required."
            ),
        }


def analyze_experiment_a_hypotheses(
    rows: Sequence[ExperimentARow],
    *,
    volunteered_updates: Sequence[VolunteeredPreferenceUpdate] = (),
    replicates: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 1729,
) -> ExperimentAHypothesisEstimands:
    """Build the versioned H1/H2/H7 Experiment A estimand artifact."""

    return ExperimentAHypothesisEstimands(
        h1=analyze_h1(
            rows,
            replicates=replicates,
            confidence_level=confidence_level,
            seed=seed,
        ),
        h2=analyze_h2(
            rows,
            replicates=replicates,
            confidence_level=confidence_level,
            seed=seed,
        ),
        h7=analyze_h7_experiment_a(
            rows,
            volunteered_updates=volunteered_updates,
            replicates=replicates,
            confidence_level=confidence_level,
            seed=seed,
        ),
        bootstrap_replicates=replicates,
        confidence_level=confidence_level,
    )


@dataclass(frozen=True, slots=True)
class H7ClosedLoopAnalysis:
    baseline_updater_id: str
    mitigation_updater_id: str
    policy_id: str
    initial_profile_condition: str
    attribution_error_reduction: PairedContrast | None
    self_confirming_profile_rate_reduction: PairedContrast | None
    minimum_cluster_count: int
    criterion_met: bool | None
    missing_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": HYPOTHESIS_ESTIMAND_SCHEMA_VERSION,
            "hypothesis_id": "H7",
            "name": "Causal provenance is actionable",
            "component": "experiment_b_self_confirmation_mitigation",
            "baseline_updater_id": self.baseline_updater_id,
            "mitigation_updater_id": self.mitigation_updater_id,
            "policy_id": self.policy_id,
            "initial_profile_condition": self.initial_profile_condition,
            "attribution_error_reduction": (
                None
                if self.attribution_error_reduction is None
                else self.attribution_error_reduction.to_dict()
            ),
            "self_confirming_profile_rate_reduction": (
                None
                if self.self_confirming_profile_rate_reduction is None
                else self.self_confirming_profile_rate_reduction.to_dict()
            ),
            "minimum_cluster_count": self.minimum_cluster_count,
            "criterion": (
                "under soft profile conditioning with incorrect initial "
                "profiles, both 95% complete-user cluster-bootstrap lower "
                "bounds for full-context minus provenance-aware attribution "
                "error and self-confirming-profile rate are > 0"
            ),
            "criterion_met": self.criterion_met,
            "computed_status": _computed_status(self.criterion_met),
            "missing_reason": self.missing_reason,
            "claim_status": "not_claimed",
            "scope_note": (
                "The full H7 criterion also requires Experiment A update-error "
                "superiority and balanced/volunteered valid-learning retention."
            ),
        }


def analyze_h7_closed_loop(
    result: ExperimentBResult,
    *,
    baseline_updater_id: str = "llm_full_context",
    mitigation_updater_id: str = "llm_provenance_aware",
    policy_id: str = "soft_profile_conditioned",
    initial_profile_condition: str = "incorrect",
    replicates: int = 2000,
    confidence_level: float = 0.95,
    minimum_cluster_count: int = MINIMUM_CLUSTER_COUNT,
    seed: int = 1729,
) -> H7ClosedLoopAnalysis:
    """Estimate H7's matched closed-loop self-confirmation reduction."""

    if minimum_cluster_count < 2:
        raise ValueError("H7 requires at least two independent user clusters")
    if baseline_updater_id != "llm_full_context":
        raise ValueError("H7 baseline is frozen to llm_full_context")
    if mitigation_updater_id != "llm_provenance_aware":
        raise ValueError("H7 mitigation is frozen to llm_provenance_aware")
    if policy_id != "soft_profile_conditioned":
        raise ValueError(
            "H7 closed-loop policy is frozen to soft_profile_conditioned"
        )
    if initial_profile_condition != "incorrect":
        raise ValueError(
            "H7 closed-loop initial profile is frozen to incorrect"
        )
    _validate_analysis_settings(
        replicates=replicates,
        confidence_level=confidence_level,
    )
    assessments: dict[str, list[bool]] = {}
    for item in result.self_confirmation_assessments:
        assessments.setdefault(item.trajectory_id, []).append(item.reportable)
    trajectories = {}
    for trajectory in result.trajectories:
        if (
            trajectory.policy_id != policy_id
            or trajectory.initial_profile_condition
            != initial_profile_condition
        ):
            continue
        key = (trajectory.crn_key, trajectory.updater_id)
        if key in trajectories:
            raise ValueError(f"duplicate closed-loop H7 trajectory key {key}")
        trajectories[key] = trajectory
    pairs = []
    for (crn_key, updater_id), baseline in sorted(trajectories.items()):
        if updater_id != baseline_updater_id:
            continue
        mitigation = trajectories.get((crn_key, mitigation_updater_id))
        if mitigation is None:
            continue
        if (
            baseline.user_id != mitigation.user_id
            or baseline.domain_id != mitigation.domain_id
            or baseline.policy_id != mitigation.policy_id
            or baseline.initial_profile_condition
            != mitigation.initial_profile_condition
        ):
            raise ValueError("closed-loop H7 pairs are not factorially matched")
        pairs.append((baseline, mitigation))
    if not pairs:
        return H7ClosedLoopAnalysis(
            baseline_updater_id=baseline_updater_id,
            mitigation_updater_id=mitigation_updater_id,
            policy_id=policy_id,
            initial_profile_condition=initial_profile_condition,
            attribution_error_reduction=None,
            self_confirming_profile_rate_reduction=None,
            minimum_cluster_count=minimum_cluster_count,
            criterion_met=None,
            missing_reason=(
                "no matched full-context/provenance-aware closed-loop "
                "trajectories for the declared policy and initial profile"
            ),
        )
    cluster_ids = [baseline.user_id for baseline, _ in pairs]
    baseline_attribution = [
        baseline.terminal_error - baseline.terminal_shadow_error
        for baseline, _ in pairs
    ]
    mitigation_attribution = [
        mitigation.terminal_error - mitigation.terminal_shadow_error
        for _, mitigation in pairs
    ]
    attribution_reduction = paired_cluster_contrast(
        baseline_attribution,
        mitigation_attribution,
        cluster_ids,
        contrast_id="H7:closed-loop:attribution-error-reduction",
        first_label=baseline_updater_id,
        second_label=mitigation_updater_id,
        replicates=replicates,
        confidence_level=confidence_level,
        seed=seed,
    )
    baseline_profiles = [
        1.0 if any(assessments.get(baseline.trajectory_id, ())) else 0.0
        for baseline, _ in pairs
    ]
    mitigation_profiles = [
        1.0 if any(assessments.get(mitigation.trajectory_id, ())) else 0.0
        for _, mitigation in pairs
    ]
    profile_reduction = paired_cluster_contrast(
        baseline_profiles,
        mitigation_profiles,
        cluster_ids,
        contrast_id="H7:closed-loop:self-confirming-profile-rate-reduction",
        first_label=baseline_updater_id,
        second_label=mitigation_updater_id,
        replicates=replicates,
        confidence_level=confidence_level,
        seed=seed,
    )
    enough_clusters = (
        attribution_reduction.interval.cluster_count >= minimum_cluster_count
        and profile_reduction.interval.cluster_count >= minimum_cluster_count
    )
    return H7ClosedLoopAnalysis(
        baseline_updater_id=baseline_updater_id,
        mitigation_updater_id=mitigation_updater_id,
        policy_id=policy_id,
        initial_profile_condition=initial_profile_condition,
        attribution_error_reduction=attribution_reduction,
        self_confirming_profile_rate_reduction=profile_reduction,
        minimum_cluster_count=minimum_cluster_count,
        criterion_met=(
            attribution_reduction.interval.lower > 0.0
            and profile_reduction.interval.lower > 0.0
            if enough_clusters
            else None
        ),
        missing_reason=(
            None
            if enough_clusters
            else (
                "matched closed-loop rows contain fewer than the frozen "
                f"{minimum_cluster_count} independent user clusters"
            )
        ),
    )
