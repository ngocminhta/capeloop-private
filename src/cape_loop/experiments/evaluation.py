"""Experiment C fixed-history replay and open/closed-loop ranking analysis."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from statistics import mean
from typing import Any, Mapping, Sequence

from ..beliefs import PreferenceBelief
from ..domains import DOMAINS, DomainSpec
from ..heldout import build_heldout_terminal_suite
from ..metrics import marginal_brier
from ..native import NativeMemoryState, decode_native_state
from ..policies import (
    BalancedPolicy,
    FixedBiasPolicy,
    InteractionPolicy,
    SoftProfileConditionedPolicy,
)
from ..response import RandomUtilityModel, intrinsic_utility
from ..schemas import (
    InteractionContext,
    InteractionRecord,
    LatentUser,
    Observation,
    Option,
    PolicyProvenance,
    THETA_VALUES,
    TrajectoryRecord,
)
from ..statistics import (
    BootstrapRankSummary,
    PairwiseDifferenceInterval,
    PairwiseRegimeShiftInterval,
    bootstrap_ranks,
    inferential_partial_order,
    inferential_tier_evaluation_selection_regret,
    kendall_tau_b,
    paired_system_difference_intervals,
    paired_system_regime_shift_intervals,
    pairwise_reversal_and_tie_probability,
    ranks_from_errors,
)
from ..updaters import (
    ProfileUpdater,
    UpdaterState,
    build_updater_registry,
    make_update_view,
)
from .closed_loop import run_trajectory
from .closed_loop import ClosedLoopTrajectory


STATIC_REGIMES = ("fixed_balanced", "fixed_biased")
ALL_REGIMES = STATIC_REGIMES + ("endogenous_closed_loop",)


def _digest(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class LoggedEvent:
    event_id: str
    context: InteractionContext
    provenance: PolicyProvenance
    observation: Observation

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "context": self.context.to_dict(),
            "policy_provenance": self.provenance.to_dict(),
            "observation": self.observation.to_dict(),
        }

    def signature(self) -> str:
        return _digest(self.to_dict())


@dataclass(frozen=True, slots=True)
class FixedHistory:
    """Canonical logger output generated once and replayed to all updaters."""

    history_id: str
    user_id: str
    domain_id: str
    logger_policy_id: str
    events: tuple[LoggedEvent, ...]
    history_digest: str = ""

    def __post_init__(self) -> None:
        if not self.events:
            raise ValueError("fixed history requires at least one event")
        if len({event.event_id for event in self.events}) != len(self.events):
            raise ValueError("fixed history event IDs must be unique")
        payload = {
            "history_id": self.history_id,
            "user_id": self.user_id,
            "domain_id": self.domain_id,
            "logger_policy_id": self.logger_policy_id,
            "events": [event.to_dict() for event in self.events],
        }
        expected = _digest(payload)
        if self.history_digest and self.history_digest != expected:
            raise ValueError("fixed history digest does not match its events")
        object.__setattr__(self, "history_digest", expected)

    def event_signatures(self) -> tuple[str, ...]:
        return tuple(event.signature() for event in self.events)

    def to_dict(self) -> dict[str, Any]:
        return {
            "history_id": self.history_id,
            "user_id": self.user_id,
            "domain_id": self.domain_id,
            "logger_policy_id": self.logger_policy_id,
            "history_digest": self.history_digest,
            "events": [event.to_dict() for event in self.events],
        }


def generate_fixed_history(
    *,
    user: LatentUser,
    domain: DomainSpec,
    policy: InteractionPolicy,
    turns: int,
    seed: int,
    response_model: RandomUtilityModel | None = None,
    reference_belief: PreferenceBelief | None = None,
    history_id: str | None = None,
    crn_key: str | None = None,
) -> FixedHistory:
    """Generate a logger history without consulting any evaluated updater."""

    if turns <= 0:
        raise ValueError("turns must be positive")
    declared_response = response_model or RandomUtilityModel()
    fixed_profile = (
        PreferenceBelief.uniform()
        if reference_belief is None
        else reference_belief
    )
    identifier = history_id or (
        f"fixed:{domain.domain_id}:{user.user_id}:{policy.policy_id}"
    )
    common_key = crn_key or identifier
    events: list[LoggedEvent] = []
    for turn in range(turns):
        action = policy.action(
            domain,
            fixed_profile,
            turn=turn,
            master_seed=seed,
            trajectory_id=identifier,
        )
        observation = declared_response.sample(
            user.theta,
            user.susceptibility,
            action.context,
            seed,
            noise_key=("fixed-history-crn", common_key, turn),
        )
        events.append(
            LoggedEvent(
                event_id=f"{identifier}:turn-{turn}",
                context=action.context,
                provenance=action.provenance,
                observation=observation,
            )
        )
    return FixedHistory(
        history_id=identifier,
        user_id=user.user_id,
        domain_id=domain.domain_id,
        logger_policy_id=policy.policy_id,
        events=tuple(events),
    )


@dataclass(frozen=True, slots=True)
class ReplayResult:
    updater_id: str
    history_id: str
    history_digest: str
    event_signatures: tuple[str, ...]
    initial_belief: PreferenceBelief
    terminal_state: UpdaterState
    audit_record: TrajectoryRecord

    @property
    def terminal_belief(self) -> PreferenceBelief:
        return self.terminal_state.belief

    def to_dict(self) -> dict[str, Any]:
        result = {
            "updater_id": self.updater_id,
            "history_id": self.history_id,
            "history_digest": self.history_digest,
            "event_signatures": list(self.event_signatures),
            "initial_belief": self.initial_belief.to_dict(),
            "terminal_belief": self.terminal_belief.to_dict(),
            "terminal_state": self.terminal_state.to_dict(),
            "audit_record": self.audit_record.to_dict(),
        }
        opaque = self.terminal_state.opaque_state
        to_dict = getattr(opaque, "to_dict", None)
        if callable(to_dict):
            result["terminal_native_state"] = to_dict()
        return result


def replay_history(
    history: FixedHistory,
    updater: ProfileUpdater,
    *,
    prior: PreferenceBelief | None = None,
    replay_id: str | None = None,
) -> ReplayResult:
    """Replay the exact retained event objects; no sampling occurs here."""

    initial = PreferenceBelief.uniform() if prior is None else prior
    state = updater.initial_state(initial)
    interactions: list[InteractionRecord] = []
    for event in history.events:
        view = make_update_view(
            updater.view_kind,
            event.context,
            event.observation,
            event.provenance,
            event_id=event.event_id,
        )
        result = updater.update(state, view)
        state = result.state
        interactions.append(
            InteractionRecord(
                record_id=event.event_id,
                context=event.context,
                provenance=event.provenance,
                observation=event.observation,
                profile_update=result.profile_update,
            )
        )
    identifier = replay_id or f"{history.history_id}:replay:{updater.updater_id}"
    audit = TrajectoryRecord(
        trajectory_id=identifier,
        user_id=history.user_id,
        domain=history.domain_id,
        interactions=tuple(interactions),
    )
    return ReplayResult(
        updater_id=updater.updater_id,
        history_id=history.history_id,
        history_digest=history.history_digest,
        event_signatures=history.event_signatures(),
        initial_belief=initial,
        terminal_state=state,
        audit_record=audit,
    )


@dataclass(frozen=True, slots=True)
class TerminalBatteryItem:
    item_id: str
    item_kind: str
    context: InteractionContext

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "item_kind": self.item_kind,
            "context": self.context.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class TerminalBattery:
    battery_id: str
    domain_id: str
    items: tuple[TerminalBatteryItem, ...]
    battery_digest: str = ""

    def __post_init__(self) -> None:
        if not self.items:
            raise ValueError("terminal battery cannot be empty")
        if len({item.item_id for item in self.items}) != len(self.items):
            raise ValueError("terminal battery item IDs must be unique")
        for item in self.items:
            context = item.context
            if context.default_option_id is not None:
                raise ValueError("terminal diagnostics cannot contain defaults")
            if context.suggested_option_id is not None:
                raise ValueError("terminal diagnostics cannot contain suggestions")
            if context.domain != self.domain_id:
                raise ValueError("terminal diagnostic has the wrong domain")
        payload = {
            "battery_id": self.battery_id,
            "domain_id": self.domain_id,
            "items": [item.to_dict() for item in self.items],
        }
        expected = _digest(payload)
        if self.battery_digest and self.battery_digest != expected:
            raise ValueError("terminal battery digest does not match its items")
        object.__setattr__(self, "battery_digest", expected)

    def to_dict(self) -> dict[str, Any]:
        return {
            "battery_id": self.battery_id,
            "domain_id": self.domain_id,
            "battery_digest": self.battery_digest,
            "items": [item.to_dict() for item in self.items],
        }


def build_terminal_battery(
    domain: DomainSpec,
    *,
    version: str = "heldout-terminal-v2",
) -> TerminalBattery:
    """Build the common battery from genuinely novel v2 terminal material."""

    suite = build_heldout_terminal_suite(domain, version=version)
    isolated_options = {
        item.target_attribute: item.options
        for item in suite.items
        if item.question_type == "forced_choice"
    }
    kind_by_question = {
        "forced_choice": "balanced_preference",
        "counterfactual_choice": "matched_counterfactual",
        "direct_preference_probe": "neutral_direct_probe",
        "cross_context_choice": "cross_context_application",
    }
    items: list[TerminalBatteryItem] = []
    for item in suite.items:
        raw_options = (
            isolated_options[item.target_attribute]
            if item.question_type == "direct_preference_probe"
            else item.options
        )
        options = tuple(
            Option(
                option_id=option.option_id,
                features=option.features,
                label=option.label,
                domain=domain.domain_id,
            )
            for option in raw_options
        )
        items.append(
            TerminalBatteryItem(
                item_id=item.item_id,
                item_kind=kind_by_question[item.question_type],
                context=InteractionContext(
                    context_id=item.item_id,
                    options=options,
                    ranking=tuple(
                        option.option_id for option in options
                    ),
                    domain=domain.domain_id,
                    scenario_id=(
                        f"{item.scenario_family_id}:{item.item_id}"
                    ),
                    turn_id=f"terminal:{item.item_id}",
                    wording_template=item.wording_template_id,
                    question_type=item.question_type,
                    target_attribute=item.target_attribute,
                ),
            )
        )
    return TerminalBattery(
        battery_id=suite.suite_id,
        domain_id=domain.domain_id,
        items=tuple(items),
    )


@dataclass(frozen=True, slots=True)
class TerminalReliabilityBin:
    """Fixed-width multiclass confidence bin for terminal profile forecasts."""

    bin_index: int
    lower: float
    upper: float
    prediction_count: int
    mean_confidence: float | None
    empirical_accuracy: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "bin_index": self.bin_index,
            "lower": self.lower,
            "upper": self.upper,
            "prediction_count": self.prediction_count,
            "mean_confidence": self.mean_confidence,
            "empirical_accuracy": self.empirical_accuracy,
        }


def _terminal_profile_reliability(
    belief: PreferenceBelief,
    user: LatentUser,
    *,
    bin_count: int = 10,
) -> tuple[float, tuple[TerminalReliabilityBin, ...], int]:
    """Return top-label ECE over the three terminal attribute forecasts."""

    if bin_count <= 1:
        raise ValueError("terminal reliability requires at least two bins")
    buckets: list[list[tuple[float, float]]] = [
        [] for _ in range(bin_count)
    ]
    for attribute in range(len(user.theta)):
        probabilities = belief.marginal(attribute)
        predicted = max(
            range(len(probabilities)),
            key=lambda index: (probabilities[index], -index),
        )
        confidence = probabilities[predicted]
        truth = THETA_VALUES.index(user.theta[attribute])
        bin_index = min(int(confidence * bin_count), bin_count - 1)
        buckets[bin_index].append(
            (confidence, 1.0 if predicted == truth else 0.0)
        )
    prediction_count = sum(len(bucket) for bucket in buckets)
    rows = []
    ece = 0.0
    for bin_index, bucket in enumerate(buckets):
        if bucket:
            mean_confidence = mean(item[0] for item in bucket)
            empirical_accuracy = mean(item[1] for item in bucket)
            ece += (
                len(bucket)
                / prediction_count
                * abs(mean_confidence - empirical_accuracy)
            )
        else:
            mean_confidence = None
            empirical_accuracy = None
        rows.append(
            TerminalReliabilityBin(
                bin_index=bin_index,
                lower=bin_index / bin_count,
                upper=(bin_index + 1) / bin_count,
                prediction_count=len(bucket),
                mean_confidence=mean_confidence,
                empirical_accuracy=empirical_accuracy,
            )
        )
    return ece, tuple(rows), prediction_count


@dataclass(frozen=True, slots=True)
class TerminalBatteryScore:
    profile_brier: float
    behavioral_accuracy: float
    tie_excluded_behavioral_accuracy: float | None
    fractional_behavioral_accuracy: float
    cross_context_accuracy: float | None
    mean_intrinsic_regret: float
    predicted_option_ids: tuple[str, ...]
    predicted_utility_tie_count: float
    intrinsic_utility_tie_count: float
    evaluated_item_count: int
    profile_ece: float | None = None
    profile_reliability_bins: tuple[TerminalReliabilityBin, ...] = ()
    profile_calibration_prediction_count: int = 0
    profile_calibration_sample_unit: str = "preference_attribute_forecast"

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_brier": self.profile_brier,
            "behavioral_accuracy": self.behavioral_accuracy,
            "tie_excluded_behavioral_accuracy": (
                self.tie_excluded_behavioral_accuracy
            ),
            "fractional_behavioral_accuracy": (
                self.fractional_behavioral_accuracy
            ),
            "cross_context_accuracy": self.cross_context_accuracy,
            "mean_intrinsic_regret": self.mean_intrinsic_regret,
            "predicted_option_ids": list(self.predicted_option_ids),
            "predicted_utility_tie_count": (
                self.predicted_utility_tie_count
            ),
            "intrinsic_utility_tie_count": (
                self.intrinsic_utility_tie_count
            ),
            "evaluated_item_count": self.evaluated_item_count,
            "profile_ece": self.profile_ece,
            "profile_calibration_sample_unit": (
                self.profile_calibration_sample_unit
            ),
            "profile_calibration_prediction_count": (
                self.profile_calibration_prediction_count
            ),
            "profile_reliability_bins": [
                row.to_dict() for row in self.profile_reliability_bins
            ],
            "profile_calibration_interpretation": (
                "Descriptive top-label multiclass calibration. Each forecast "
                "unit is one preference attribute; trajectory/user clustering "
                "must be retained for inferential uncertainty."
            ),
        }


def evaluate_terminal_battery(
    belief: PreferenceBelief,
    user: LatentUser,
    battery: TerminalBattery,
    *,
    tie_tolerance: float = 1e-12,
) -> TerminalBatteryScore:
    """Evaluate a profile on a fixed battery without updating the system."""

    if tie_tolerance < 0:
        raise ValueError("tie_tolerance must be non-negative")
    expected = belief.expected_theta()
    predictions: list[str] = []
    correct: list[float] = []
    tie_excluded_correct: list[float] = []
    fractional_correct: list[float] = []
    cross_correct: list[float] = []
    regrets: list[float] = []
    predicted_ties = 0
    intrinsic_ties = 0
    for item in battery.items:
        context = item.context
        projected_utilities = {
            option.option_id: math.fsum(
                coefficient * feature
                for coefficient, feature in zip(expected, option.features)
            )
            for option in context.options
        }
        projected_optimum = max(projected_utilities.values())
        projected_maximizers = tuple(
            option
            for option in context.options
            if math.isclose(
                projected_utilities[option.option_id],
                projected_optimum,
                rel_tol=0.0,
                abs_tol=tie_tolerance,
            )
        )
        predicted_ties += len(projected_maximizers) > 1
        predicted = max(
            projected_maximizers,
            key=lambda option: option.option_id,
        )
        true_utilities = {
            option.option_id: intrinsic_utility(user.theta, option)
            for option in context.options
        }
        optimum = max(true_utilities.values())
        intrinsic_maximizer_count = sum(
            math.isclose(
                value,
                optimum,
                rel_tol=0.0,
                abs_tol=tie_tolerance,
            )
            for value in true_utilities.values()
        )
        intrinsic_ties += intrinsic_maximizer_count > 1
        is_correct = math.isclose(
            true_utilities[predicted.option_id],
            optimum,
            rel_tol=0.0,
            abs_tol=tie_tolerance,
        )
        predictions.append(predicted.option_id)
        correct.append(1.0 if is_correct else 0.0)
        fractional_correct.append(
            (1.0 / intrinsic_maximizer_count) if is_correct else 0.0
        )
        if intrinsic_maximizer_count == 1:
            tie_excluded_correct.append(1.0 if is_correct else 0.0)
        if item.item_kind == "cross_context_application":
            cross_correct.append(1.0 if is_correct else 0.0)
        regrets.append(optimum - true_utilities[predicted.option_id])
    profile_ece, profile_reliability, profile_prediction_count = (
        _terminal_profile_reliability(belief, user)
    )
    return TerminalBatteryScore(
        profile_brier=marginal_brier(belief, user.theta),
        behavioral_accuracy=math.fsum(correct) / len(correct),
        tie_excluded_behavioral_accuracy=(
            None
            if not tie_excluded_correct
            else math.fsum(tie_excluded_correct)
            / len(tie_excluded_correct)
        ),
        fractional_behavioral_accuracy=(
            math.fsum(fractional_correct) / len(fractional_correct)
        ),
        cross_context_accuracy=(
            None
            if not cross_correct
            else math.fsum(cross_correct) / len(cross_correct)
        ),
        mean_intrinsic_regret=math.fsum(regrets) / len(regrets),
        predicted_option_ids=tuple(predictions),
        predicted_utility_tie_count=float(predicted_ties),
        intrinsic_utility_tie_count=float(intrinsic_ties),
        evaluated_item_count=len(battery.items),
        profile_ece=profile_ece,
        profile_reliability_bins=profile_reliability,
        profile_calibration_prediction_count=profile_prediction_count,
    )


@dataclass(frozen=True, slots=True)
class NativeDecoderEvaluation:
    decoder_id: str
    pseudonymous_state_id: str
    score: TerminalBatteryScore

    def to_dict(self) -> dict[str, Any]:
        return {
            "decoder_id": self.decoder_id,
            "pseudonymous_state_id": self.pseudonymous_state_id,
            **self.score.to_dict(),
        }


def evaluate_native_decoders(
    state: object | None,
    user: LatentUser,
    battery: TerminalBattery,
) -> tuple[NativeDecoderEvaluation, ...]:
    """Evaluate both blinded native decoders, or return no rows for structured state."""

    if not isinstance(state, NativeMemoryState):
        return ()
    return tuple(
        NativeDecoderEvaluation(
            decoder_id=result.decoder_id,
            pseudonymous_state_id=result.pseudonymous_state_id,
            score=evaluate_terminal_battery(result.belief, user, battery),
        )
        for result in decode_native_state(state)
    )


@dataclass(frozen=True, slots=True)
class EvaluationRow:
    split: str
    regime: str
    replicate: int
    user_id: str
    domain_id: str
    updater_id: str
    profile_error: float
    behavioral_accuracy: float
    cross_context_accuracy: float | None
    intrinsic_regret: float
    history_digest: str
    event_signatures: tuple[str, ...]
    battery_id: str
    battery_digest: str
    predicted_option_ids: tuple[str, ...]
    score_basis: str
    system_projection_score: TerminalBatteryScore
    native_decoder_evaluations: tuple[NativeDecoderEvaluation, ...] = ()
    ranking_score: TerminalBatteryScore | None = None

    def to_dict(self) -> dict[str, Any]:
        active_ranking_score = (
            self.system_projection_score
            if self.ranking_score is None
            else self.ranking_score
        )
        return {
            "split": self.split,
            "regime": self.regime,
            "replicate": self.replicate,
            "user_id": self.user_id,
            "domain_id": self.domain_id,
            "updater_id": self.updater_id,
            "profile_error": self.profile_error,
            "behavioral_accuracy": self.behavioral_accuracy,
            "cross_context_accuracy": self.cross_context_accuracy,
            "intrinsic_regret": self.intrinsic_regret,
            "history_digest": self.history_digest,
            "event_signatures": list(self.event_signatures),
            "battery_id": self.battery_id,
            "battery_digest": self.battery_digest,
            "predicted_option_ids": list(self.predicted_option_ids),
            "score_basis": self.score_basis,
            "ranking_score": active_ranking_score.to_dict(),
            "system_projection_score": self.system_projection_score.to_dict(),
            "native_decoder_evaluations": [
                evaluation.to_dict()
                for evaluation in self.native_decoder_evaluations
            ],
        }


@dataclass(frozen=True, slots=True)
class ClusteredRankingSamples:
    """System errors aligned and reduced to complete latent-user clusters.

    ``member_keys`` records every trajectory component retained in each user
    cluster.  The corresponding entry in each system sample is the mean over
    that complete component set, so downstream paired bootstraps resample one
    complete latent user at a time rather than treating domains or trajectory
    replicates as independent observations.
    """

    split: str
    regime: str
    cluster_ids: tuple[str, ...]
    member_keys: tuple[tuple[tuple[str, str, int], ...], ...]
    system_samples: tuple[tuple[str, tuple[float, ...]], ...]

    @property
    def errors_by_system(self) -> dict[str, tuple[float, ...]]:
        return dict(self.system_samples)

    @property
    def component_layout(self) -> tuple[tuple[str, int], ...]:
        return tuple(
            (domain_id, replicate)
            for _, domain_id, replicate in self.member_keys[0]
        )


def build_clustered_ranking_samples(
    rows: Sequence[EvaluationRow],
    *,
    split: str,
    regime: str,
    updater_ids: Sequence[str],
) -> ClusteredRankingSamples:
    """Align systems by stable trajectory keys and form user clusters.

    The stable row key is ``(user_id, domain_id, replicate)`` within a declared
    split/regime cell.  Every system must have exactly one row for every key,
    and every user must contain the same complete domain/replicate layout.
    """

    systems = tuple(updater_ids)
    if not systems or len(set(systems)) != len(systems):
        raise ValueError("clustered ranking samples need distinct updater IDs")
    keyed: dict[str, dict[tuple[str, str, int], float]] = {
        system: {} for system in systems
    }
    for row in rows:
        if (
            row.split != split
            or row.regime != regime
            or row.updater_id not in keyed
        ):
            continue
        key = (row.user_id, row.domain_id, row.replicate)
        if key in keyed[row.updater_id]:
            raise ValueError(
                "duplicate ranking row for "
                f"{split}/{regime}/{row.updater_id}/{key}"
            )
        if not math.isfinite(row.profile_error):
            raise ValueError("ranking profile errors must be finite")
        keyed[row.updater_id][key] = row.profile_error

    reference_system = systems[0]
    reference_keys = set(keyed[reference_system])
    if not reference_keys:
        raise ValueError(
            f"missing {split}/{regime} rows for {reference_system}"
        )
    for system in systems[1:]:
        observed_keys = set(keyed[system])
        if observed_keys != reference_keys:
            missing = sorted(reference_keys - observed_keys)
            extra = sorted(observed_keys - reference_keys)
            raise ValueError(
                "ranking rows are not aligned by stable trajectory key for "
                f"{split}/{regime}/{system}; missing={missing}, extra={extra}"
            )

    cluster_ids = tuple(sorted({key[0] for key in reference_keys}))
    member_keys = tuple(
        tuple(sorted(key for key in reference_keys if key[0] == cluster_id))
        for cluster_id in cluster_ids
    )
    reference_layout = tuple(
        (domain_id, replicate)
        for _, domain_id, replicate in member_keys[0]
    )
    if not reference_layout:
        raise ValueError("ranking user clusters cannot be empty")
    expected_layout = tuple(
        (domain_id, replicate)
        for domain_id in sorted({key[1] for key in reference_keys})
        for replicate in sorted({key[2] for key in reference_keys})
    )
    if reference_layout != expected_layout:
        raise ValueError(
            "ranking user clusters must contain the complete observed "
            f"domain × replicate product; expected={expected_layout}, "
            f"observed={reference_layout}"
        )
    for cluster_id, members in zip(cluster_ids, member_keys):
        layout = tuple(
            (domain_id, replicate)
            for _, domain_id, replicate in members
        )
        if layout != reference_layout:
            raise ValueError(
                "ranking user clusters do not contain the same complete "
                f"domain/replicate layout; cluster={cluster_id}, "
                f"expected={reference_layout}, observed={layout}"
            )

    system_samples = tuple(
        (
            system,
            tuple(
                mean(keyed[system][key] for key in members)
                for members in member_keys
            ),
        )
        for system in systems
    )
    return ClusteredRankingSamples(
        split=split,
        regime=regime,
        cluster_ids=cluster_ids,
        member_keys=member_keys,
        system_samples=system_samples,
    )


def summarize_terminal_calibration(
    scores: Sequence[TerminalBatteryScore],
) -> dict[str, Any]:
    """Pool terminal reliability bins without treating attributes as users."""

    material = tuple(scores)
    if not material:
        raise ValueError("terminal calibration summary requires scores")
    if any(
        score.profile_ece is None
        or not score.profile_reliability_bins
        or score.profile_calibration_prediction_count <= 0
        for score in material
    ):
        raise ValueError("terminal scores lack profile calibration records")
    layouts = {
        tuple(
            (row.bin_index, row.lower, row.upper)
            for row in score.profile_reliability_bins
        )
        for score in material
    }
    if len(layouts) != 1:
        raise ValueError("terminal calibration bin layouts differ")
    rows = []
    total_predictions = sum(
        score.profile_calibration_prediction_count for score in material
    )
    weighted_ece = 0.0
    for bin_index in range(len(material[0].profile_reliability_bins)):
        source_rows = tuple(
            score.profile_reliability_bins[bin_index] for score in material
        )
        prediction_count = sum(row.prediction_count for row in source_rows)
        populated = tuple(
            row for row in source_rows if row.prediction_count > 0
        )
        if populated:
            mean_confidence = (
                math.fsum(
                    float(row.mean_confidence) * row.prediction_count
                    for row in populated
                )
                / prediction_count
            )
            empirical_accuracy = (
                math.fsum(
                    float(row.empirical_accuracy) * row.prediction_count
                    for row in populated
                )
                / prediction_count
            )
            weighted_ece += (
                prediction_count
                / total_predictions
                * abs(mean_confidence - empirical_accuracy)
            )
        else:
            mean_confidence = None
            empirical_accuracy = None
        exemplar = source_rows[0]
        rows.append(
            TerminalReliabilityBin(
                bin_index=exemplar.bin_index,
                lower=exemplar.lower,
                upper=exemplar.upper,
                prediction_count=prediction_count,
                mean_confidence=mean_confidence,
                empirical_accuracy=empirical_accuracy,
            )
        )
    return {
        "profile_ece": weighted_ece,
        "profile_reliability_bins": [row.to_dict() for row in rows],
        "score_count": len(material),
        "profile_calibration_prediction_count": total_predictions,
        "profile_calibration_sample_unit": "preference_attribute_forecast",
        "dependence_unit": (
            "trajectory/user; this pooled ECE is descriptive and does not "
            "treat the attribute forecasts as independent inferential units"
        ),
    }


def _pooled_terminal_calibration(
    scores: Sequence[TerminalBatteryScore],
) -> tuple[
    float | None,
    tuple[TerminalReliabilityBin, ...],
    int,
]:
    """Pool score-level bins for an averaged multi-decoder score."""

    if any(
        score.profile_ece is None
        or not score.profile_reliability_bins
        for score in scores
    ):
        return None, (), 0
    summary = summarize_terminal_calibration(scores)
    rows = tuple(
        TerminalReliabilityBin(
            bin_index=int(row["bin_index"]),
            lower=float(row["lower"]),
            upper=float(row["upper"]),
            prediction_count=int(row["prediction_count"]),
            mean_confidence=(
                None
                if row["mean_confidence"] is None
                else float(row["mean_confidence"])
            ),
            empirical_accuracy=(
                None
                if row["empirical_accuracy"] is None
                else float(row["empirical_accuracy"])
            ),
        )
        for row in summary["profile_reliability_bins"]
    )
    return (
        float(summary["profile_ece"]),
        rows,
        int(summary["profile_calibration_prediction_count"]),
    )


def _ranking_score(
    system_projection: TerminalBatteryScore,
    decoder_evaluations: tuple[NativeDecoderEvaluation, ...],
) -> tuple[TerminalBatteryScore, str]:
    """Use both blinded decoders for native ranking, never a single view."""

    if not decoder_evaluations:
        return system_projection, "system_structured_projection"
    if len(decoder_evaluations) != 2:
        raise ValueError(
            "native ranking requires exactly two blinded decoder evaluations"
        )
    scores = tuple(item.score for item in decoder_evaluations)
    profile_ece, profile_reliability, profile_prediction_count = (
        _pooled_terminal_calibration(scores)
    )
    return (
        TerminalBatteryScore(
            profile_brier=math.fsum(
                score.profile_brier for score in scores
            )
            / len(scores),
            behavioral_accuracy=math.fsum(
                score.behavioral_accuracy for score in scores
            )
            / len(scores),
            tie_excluded_behavioral_accuracy=(
                None
                if any(
                    score.tie_excluded_behavioral_accuracy is None
                    for score in scores
                )
                else math.fsum(
                    float(score.tie_excluded_behavioral_accuracy)
                    for score in scores
                )
                / len(scores)
            ),
            fractional_behavioral_accuracy=math.fsum(
                score.fractional_behavioral_accuracy for score in scores
            )
            / len(scores),
            cross_context_accuracy=(
                None
                if any(
                    score.cross_context_accuracy is None
                    for score in scores
                )
                else math.fsum(
                    float(score.cross_context_accuracy)
                    for score in scores
                )
                / len(scores)
            ),
            mean_intrinsic_regret=math.fsum(
                score.mean_intrinsic_regret for score in scores
            )
            / len(scores),
            # A mean score has no single predicted action sequence. Individual
            # sequences remain in native_decoder_evaluations.
            predicted_option_ids=(),
            predicted_utility_tie_count=math.fsum(
                score.predicted_utility_tie_count for score in scores
            )
            / len(scores),
            intrinsic_utility_tie_count=math.fsum(
                score.intrinsic_utility_tie_count for score in scores
            )
            / len(scores),
            evaluated_item_count=scores[0].evaluated_item_count,
            profile_ece=profile_ece,
            profile_reliability_bins=profile_reliability,
            profile_calibration_prediction_count=profile_prediction_count,
        ),
        "mean_of_two_blinded_native_decoders",
    )


@dataclass(frozen=True, slots=True)
class RankingAnalysis:
    inference_unit: str
    alignment_key: tuple[str, ...]
    development_cluster_count: int
    test_cluster_count: int
    cluster_component_layout: tuple[tuple[str, int], ...]
    open_mean_errors: tuple[tuple[str, float], ...]
    biased_mean_errors: tuple[tuple[str, float], ...]
    closed_development_mean_errors: tuple[tuple[str, float], ...]
    closed_test_mean_errors: tuple[tuple[str, float], ...]
    open_ranks: tuple[tuple[str, float], ...]
    biased_ranks: tuple[tuple[str, float], ...]
    closed_ranks: tuple[tuple[str, float], ...]
    open_closed_kendall_tau: float | None
    biased_closed_kendall_tau: float | None
    open_bootstrap_ranks: tuple[BootstrapRankSummary, ...]
    closed_bootstrap_ranks: tuple[BootstrapRankSummary, ...]
    pairwise_reversal_probabilities: tuple[tuple[str, float], ...]
    pairwise_tie_probabilities: tuple[tuple[str, float], ...]
    pairwise_open_difference_intervals: tuple[
        PairwiseDifferenceInterval, ...
    ]
    pairwise_closed_difference_intervals: tuple[
        PairwiseDifferenceInterval, ...
    ]
    pairwise_open_closed_shift_intervals: tuple[
        PairwiseRegimeShiftInterval, ...
    ]
    credible_pairwise_reversals: tuple[str, ...]
    open_partial_order: tuple[tuple[str, ...], ...]
    closed_partial_order: tuple[tuple[str, ...], ...]
    partial_order: tuple[tuple[str, ...], ...]
    open_loop_optimism: tuple[tuple[str, float], ...]
    evaluation_selection_regret: tuple[tuple[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "inference_unit": self.inference_unit,
            "alignment_key": list(self.alignment_key),
            "development_cluster_count": self.development_cluster_count,
            "test_cluster_count": self.test_cluster_count,
            "cluster_component_layout": [
                {
                    "domain_id": domain_id,
                    "replicate": replicate,
                }
                for domain_id, replicate in self.cluster_component_layout
            ],
            "bootstrap_method": (
                "paired percentile bootstrap over complete latent-user "
                "clusters; all domains and trajectory replicates remain "
                "grouped within each resampled unit"
            ),
            "open_mean_errors": dict(self.open_mean_errors),
            "biased_mean_errors": dict(self.biased_mean_errors),
            "closed_development_mean_errors": dict(
                self.closed_development_mean_errors
            ),
            "closed_test_mean_errors": dict(self.closed_test_mean_errors),
            "open_ranks": dict(self.open_ranks),
            "biased_ranks": dict(self.biased_ranks),
            "closed_ranks": dict(self.closed_ranks),
            "open_closed_kendall_tau": self.open_closed_kendall_tau,
            "biased_closed_kendall_tau": self.biased_closed_kendall_tau,
            "open_bootstrap_ranks": [
                {
                    "system_id": item.system_id,
                    "mean_rank": item.mean_rank,
                    "lower": item.lower,
                    "upper": item.upper,
                }
                for item in self.open_bootstrap_ranks
            ],
            "closed_bootstrap_ranks": [
                {
                    "system_id": item.system_id,
                    "mean_rank": item.mean_rank,
                    "lower": item.lower,
                    "upper": item.upper,
                }
                for item in self.closed_bootstrap_ranks
            ],
            "pairwise_reversal_probabilities": dict(
                self.pairwise_reversal_probabilities
            ),
            "pairwise_tie_probabilities": dict(
                self.pairwise_tie_probabilities
            ),
            "pairwise_open_difference_intervals": [
                interval.to_dict()
                for interval in self.pairwise_open_difference_intervals
            ],
            "pairwise_closed_difference_intervals": [
                interval.to_dict()
                for interval in self.pairwise_closed_difference_intervals
            ],
            "pairwise_open_closed_shift_intervals": [
                interval.to_dict()
                for interval in self.pairwise_open_closed_shift_intervals
            ],
            "credible_pairwise_reversals": list(
                self.credible_pairwise_reversals
            ),
            "credible_reversal_basis": (
                "joint paired open/closed complete-user error-difference "
                "intervals clear the tie region in opposite directions and "
                "their difference-of-differences interval clears it too"
            ),
            "open_partial_order": [
                list(group) for group in self.open_partial_order
            ],
            "closed_partial_order": [
                list(group) for group in self.closed_partial_order
            ],
            "partial_order": [list(group) for group in self.partial_order],
            "partial_order_basis": (
                "paired development error-difference intervals by regime; "
                "partial_order is the closed-development alias, and same-tier "
                "systems are not separated by interval-supported dominance"
            ),
            "open_loop_optimism": dict(self.open_loop_optimism),
            "evaluation_selection_regret": dict(
                self.evaluation_selection_regret
            ),
        }


def _assert_same_cluster_members(
    first: ClusteredRankingSamples,
    second: ClusteredRankingSamples,
) -> None:
    if (
        first.cluster_ids != second.cluster_ids
        or first.member_keys != second.member_keys
    ):
        raise ValueError(
            "ranking regimes are not paired on identical complete "
            f"latent-user clusters: {first.split}/{first.regime} versus "
            f"{second.split}/{second.regime}"
        )


def _means(samples: Mapping[str, Sequence[float]]) -> dict[str, float]:
    return {system: mean(values) for system, values in samples.items()}


def _partial_order(
    errors: Mapping[str, float],
    *,
    tie_tolerance: float,
) -> tuple[tuple[str, ...], ...]:
    ranks = ranks_from_errors(errors, tie_tolerance=tie_tolerance)
    grouped: dict[float, list[str]] = {}
    for system, rank in ranks.items():
        grouped.setdefault(rank, []).append(system)
    return tuple(
        tuple(sorted(grouped[rank]))
        for rank in sorted(grouped)
    )


def analyze_rankings(
    rows: Sequence[EvaluationRow],
    *,
    updater_ids: Sequence[str],
    bootstrap_replicates: int = 1000,
    seed: int = 1729,
    tie_tolerance: float = 1e-6,
) -> RankingAnalysis:
    """Compute all ranking and deployment-selection quantities in the proposal."""

    systems = tuple(updater_ids)
    if len(systems) < 2 or len(set(systems)) != len(systems):
        raise ValueError(
            "ranking analysis requires at least two distinct updater IDs"
        )
    if bootstrap_replicates <= 0:
        raise ValueError("ranking analysis requires positive bootstrap replicates")
    if tie_tolerance < 0:
        raise ValueError("tie_tolerance must be non-negative")
    open_clusters = build_clustered_ranking_samples(
        rows,
        split="development",
        regime="fixed_balanced",
        updater_ids=systems,
    )
    biased_clusters = build_clustered_ranking_samples(
        rows,
        split="development",
        regime="fixed_biased",
        updater_ids=systems,
    )
    closed_development_clusters = build_clustered_ranking_samples(
        rows,
        split="development",
        regime="endogenous_closed_loop",
        updater_ids=systems,
    )
    closed_test_clusters = build_clustered_ranking_samples(
        rows,
        split="test",
        regime="endogenous_closed_loop",
        updater_ids=systems,
    )
    _assert_same_cluster_members(open_clusters, biased_clusters)
    _assert_same_cluster_members(open_clusters, closed_development_clusters)
    if (
        open_clusters.component_layout
        != closed_test_clusters.component_layout
    ):
        raise ValueError(
            "development and test ranking clusters must retain the same "
            "domain/replicate layout"
        )
    open_samples = open_clusters.errors_by_system
    biased_samples = biased_clusters.errors_by_system
    closed_development_samples = (
        closed_development_clusters.errors_by_system
    )
    closed_test_samples = closed_test_clusters.errors_by_system
    open_errors = _means(open_samples)
    biased_errors = _means(biased_samples)
    closed_dev_errors = _means(closed_development_samples)
    closed_test_errors = _means(closed_test_samples)
    open_ranks = ranks_from_errors(
        open_errors,
        tie_tolerance=tie_tolerance,
    )
    biased_ranks = ranks_from_errors(
        biased_errors,
        tie_tolerance=tie_tolerance,
    )
    closed_ranks = ranks_from_errors(
        closed_dev_errors,
        tie_tolerance=tie_tolerance,
    )
    open_tau = kendall_tau_b(
        open_errors,
        closed_dev_errors,
        tie_tolerance=tie_tolerance,
    )
    biased_tau = kendall_tau_b(
        biased_errors,
        closed_dev_errors,
        tie_tolerance=tie_tolerance,
    )
    reversal_probabilities, tie_probabilities = (
        pairwise_reversal_and_tie_probability(
            open_samples,
            closed_development_samples,
            replicates=bootstrap_replicates,
            seed=seed,
            tie_tolerance=tie_tolerance,
        )
    )
    open_difference_intervals = paired_system_difference_intervals(
        open_samples,
        replicates=bootstrap_replicates,
        seed=seed,
        tie_tolerance=tie_tolerance,
    )
    closed_difference_intervals = paired_system_difference_intervals(
        closed_development_samples,
        replicates=bootstrap_replicates,
        seed=seed,
        tie_tolerance=tie_tolerance,
    )
    closed_test_difference_intervals = paired_system_difference_intervals(
        closed_test_samples,
        replicates=bootstrap_replicates,
        seed=seed,
        tie_tolerance=tie_tolerance,
    )
    regime_shift_intervals = paired_system_regime_shift_intervals(
        open_samples,
        closed_development_samples,
        replicates=bootstrap_replicates,
        seed=seed,
        tie_tolerance=tie_tolerance,
    )
    open_partial_order = inferential_partial_order(
        systems,
        open_difference_intervals,
    )
    closed_partial_order = inferential_partial_order(
        systems,
        closed_difference_intervals,
    )
    esr = inferential_tier_evaluation_selection_regret(
        open_partial_order[0],
        closed_partial_order[0],
        closed_test_errors,
        closed_test_difference_intervals,
    )
    credible_reversals = tuple(
        f"{interval.first_system}|{interval.second_system}"
        for interval in regime_shift_intervals
        if interval.credible_reversal
    )
    return RankingAnalysis(
        inference_unit="complete_latent_user_cluster",
        alignment_key=(
            "split",
            "regime",
            "user_id",
            "domain_id",
            "replicate",
        ),
        development_cluster_count=len(open_clusters.cluster_ids),
        test_cluster_count=len(closed_test_clusters.cluster_ids),
        cluster_component_layout=open_clusters.component_layout,
        open_mean_errors=tuple(sorted(open_errors.items())),
        biased_mean_errors=tuple(sorted(biased_errors.items())),
        closed_development_mean_errors=tuple(
            sorted(closed_dev_errors.items())
        ),
        closed_test_mean_errors=tuple(sorted(closed_test_errors.items())),
        open_ranks=tuple(sorted(open_ranks.items())),
        biased_ranks=tuple(sorted(biased_ranks.items())),
        closed_ranks=tuple(sorted(closed_ranks.items())),
        open_closed_kendall_tau=(
            open_tau if math.isfinite(open_tau) else None
        ),
        biased_closed_kendall_tau=(
            biased_tau if math.isfinite(biased_tau) else None
        ),
        open_bootstrap_ranks=bootstrap_ranks(
            open_samples,
            replicates=bootstrap_replicates,
            seed=seed,
            tie_tolerance=tie_tolerance,
        ),
        closed_bootstrap_ranks=bootstrap_ranks(
            closed_development_samples,
            replicates=bootstrap_replicates,
            seed=seed,
            tie_tolerance=tie_tolerance,
        ),
        pairwise_reversal_probabilities=tuple(
            sorted(reversal_probabilities.items())
        ),
        pairwise_tie_probabilities=tuple(sorted(tie_probabilities.items())),
        pairwise_open_difference_intervals=(
            open_difference_intervals
        ),
        pairwise_closed_difference_intervals=(
            closed_difference_intervals
        ),
        pairwise_open_closed_shift_intervals=regime_shift_intervals,
        credible_pairwise_reversals=credible_reversals,
        open_partial_order=open_partial_order,
        closed_partial_order=closed_partial_order,
        partial_order=closed_partial_order,
        open_loop_optimism=tuple(
            sorted(
                (
                    system,
                    closed_dev_errors[system] - open_errors[system],
                )
                for system in systems
            )
        ),
        evaluation_selection_regret=tuple(sorted(esr.items())),
    )


@dataclass(frozen=True, slots=True)
class ExperimentCResult:
    fixed_histories: tuple[FixedHistory, ...]
    terminal_batteries: tuple[TerminalBattery, ...]
    replay_results: tuple[ReplayResult, ...]
    endogenous_trajectories: tuple[ClosedLoopTrajectory, ...]
    rows: tuple[EvaluationRow, ...]
    rankings: RankingAnalysis

    def assert_static_replay_identity(self) -> None:
        """Fail if any updater received a different nominally fixed history."""

        expected = {
            history.history_digest: history.event_signatures()
            for history in self.fixed_histories
        }
        grouped: dict[tuple[str, str, str], set[tuple[str, tuple[str, ...]]]] = {}
        for row in self.rows:
            if row.regime not in STATIC_REGIMES:
                continue
            grouped.setdefault(
                (row.regime, row.domain_id, row.user_id, row.replicate),
                set(),
            ).add((row.history_digest, row.event_signatures))
            if row.history_digest not in expected:
                raise AssertionError("row references an unknown fixed history")
            if expected[row.history_digest] != row.event_signatures:
                raise AssertionError("fixed history event signatures changed")
        if any(len(signatures) != 1 for signatures in grouped.values()):
            raise AssertionError("static logging histories differ across updaters")

    def assert_terminal_battery_identity(self) -> None:
        by_domain = {
            battery.domain_id: (
                battery.battery_id,
                battery.battery_digest,
            )
            for battery in self.terminal_batteries
        }
        for row in self.rows:
            if (row.battery_id, row.battery_digest) != by_domain[row.domain_id]:
                raise AssertionError(
                    "terminal diagnostic battery depends on the evaluated system"
                )

    def to_dict(self) -> dict[str, Any]:
        self.assert_static_replay_identity()
        self.assert_terminal_battery_identity()
        return {
            "experiment": "C",
            "fixed_histories": [
                history.to_dict() for history in self.fixed_histories
            ],
            "terminal_batteries": [
                battery.to_dict() for battery in self.terminal_batteries
            ],
            "replay_results": [
                replay.to_dict() for replay in self.replay_results
            ],
            "endogenous_trajectories": [
                trajectory.to_dict()
                for trajectory in self.endogenous_trajectories
            ],
            "rows": [row.to_dict() for row in self.rows],
            "rankings": self.rankings.to_dict(),
        }


def _default_practical_updaters() -> dict[str, ProfileUpdater]:
    return build_updater_registry(
        (
            "response_only",
            "full_context_blind",
            "provenance_aware",
            "conservative",
            "semantic_memory",
            "episodic_memory",
            "provenance_linked_memory",
        )
    )


def _audit_digest(record: TrajectoryRecord) -> str:
    return _digest(
        [
            {
                "event_id": event.record_id,
                "context": event.context.to_dict(),
                "policy_provenance": event.provenance.to_dict(),
                "observation": event.observation.to_dict(),
            }
            for event in record.interactions
        ]
    )


def _audit_event_signatures(record: TrajectoryRecord) -> tuple[str, ...]:
    return tuple(
        _digest(
            {
                "event_id": event.record_id,
                "context": event.context.to_dict(),
                "policy_provenance": event.provenance.to_dict(),
                "observation": event.observation.to_dict(),
            }
        )
        for event in record.interactions
    )


def run_experiment_c(
    *,
    development_users: Sequence[LatentUser],
    test_users: Sequence[LatentUser],
    domains: Sequence[DomainSpec] = DOMAINS,
    updaters: Mapping[str, ProfileUpdater] | None = None,
    policies: Mapping[str, InteractionPolicy] | None = None,
    turns: int = 16,
    trajectories_per_cell: int = 1,
    response_model: RandomUtilityModel | None = None,
    seed: int = 1729,
    bootstrap_replicates: int = 1000,
    tie_tolerance: float = 1e-6,
) -> ExperimentCResult:
    """Run fixed balanced, fixed biased, and endogenous evaluation regimes."""

    split_populations = {
        "development": tuple(development_users),
        "test": tuple(test_users),
    }
    if not split_populations["development"] or not split_populations["test"]:
        raise ValueError("Experiment C requires non-empty development and test users")
    if trajectories_per_cell <= 0:
        raise ValueError("trajectories_per_cell must be positive")
    all_ids = [
        user.user_id
        for population in split_populations.values()
        for user in population
    ]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("development and test user IDs must be disjoint")
    domain_specs = tuple(domains)
    if not domain_specs:
        raise ValueError("Experiment C requires at least one domain")
    updater_registry = dict(
        _default_practical_updaters() if updaters is None else updaters
    )
    if not updater_registry:
        raise ValueError("Experiment C requires at least one updater")
    if len(updater_registry) < 2:
        raise ValueError("Experiment C requires at least two updaters")
    if bootstrap_replicates <= 0:
        raise ValueError(
            "Experiment C requires positive bootstrap_replicates"
        )
    if tie_tolerance < 0:
        raise ValueError("tie_tolerance must be non-negative")
    for key, updater in updater_registry.items():
        if key != updater.updater_id:
            raise ValueError("updater registry keys must equal updater IDs")
    required_policy_ids = {
        "balanced",
        "fixed_bias",
        "soft_profile_conditioned",
    }
    policy_registry = (
        {
            "balanced": BalancedPolicy(),
            "fixed_bias": FixedBiasPolicy(),
            "soft_profile_conditioned": SoftProfileConditionedPolicy(),
        }
        if policies is None
        else dict(policies)
    )
    if set(policy_registry) != required_policy_ids:
        raise ValueError(
            "Experiment C requires exactly balanced, fixed_bias, and "
            "soft_profile_conditioned policies"
        )
    for key, policy in policy_registry.items():
        if key != policy.policy_id:
            raise ValueError("policy registry keys must equal policy IDs")
    declared_response = response_model or RandomUtilityModel()
    batteries = {
        domain.domain_id: build_terminal_battery(domain)
        for domain in domain_specs
    }
    histories: list[FixedHistory] = []
    replay_results: list[ReplayResult] = []
    endogenous_trajectories: list[ClosedLoopTrajectory] = []
    rows: list[EvaluationRow] = []

    for split, population in split_populations.items():
        for domain in domain_specs:
            battery = batteries[domain.domain_id]
            for user in population:
                for replicate in range(trajectories_per_cell):
                    paired_key = (
                        f"experiment-c:{split}:{domain.domain_id}:{user.user_id}:"
                        f"replicate-{replicate}"
                    )
                    balanced_history = generate_fixed_history(
                        user=user,
                        domain=domain,
                        policy=policy_registry["balanced"],
                        turns=turns,
                        seed=seed,
                        response_model=declared_response,
                        history_id=f"{paired_key}:fixed-balanced",
                        crn_key=paired_key,
                    )
                    biased_history = generate_fixed_history(
                        user=user,
                        domain=domain,
                        policy=policy_registry["fixed_bias"],
                        turns=turns,
                        seed=seed,
                        response_model=declared_response,
                        history_id=f"{paired_key}:fixed-biased",
                        crn_key=paired_key,
                    )
                    histories.extend((balanced_history, biased_history))

                    for updater_id, updater in updater_registry.items():
                        for regime, history in (
                            ("fixed_balanced", balanced_history),
                            ("fixed_biased", biased_history),
                        ):
                            replay = replay_history(
                                history,
                                updater,
                                replay_id=(
                                    f"{history.history_id}:replay:{updater_id}"
                                ),
                            )
                            replay_results.append(replay)
                            system_projection_score = evaluate_terminal_battery(
                                replay.terminal_belief,
                                user,
                                battery,
                            )
                            native_decoder_evaluations = (
                                evaluate_native_decoders(
                                    replay.terminal_state.opaque_state,
                                    user,
                                    battery,
                                )
                            )
                            score, score_basis = _ranking_score(
                                system_projection_score,
                                native_decoder_evaluations,
                            )
                            rows.append(
                                EvaluationRow(
                                    split=split,
                                    regime=regime,
                                    replicate=replicate,
                                    user_id=user.user_id,
                                    domain_id=domain.domain_id,
                                    updater_id=updater_id,
                                    profile_error=score.profile_brier,
                                    behavioral_accuracy=score.behavioral_accuracy,
                                    cross_context_accuracy=(
                                        score.cross_context_accuracy
                                    ),
                                    intrinsic_regret=score.mean_intrinsic_regret,
                                    history_digest=replay.history_digest,
                                    event_signatures=replay.event_signatures,
                                    battery_id=battery.battery_id,
                                    battery_digest=battery.battery_digest,
                                    predicted_option_ids=(
                                        score.predicted_option_ids
                                    ),
                                    score_basis=score_basis,
                                    system_projection_score=(
                                        system_projection_score
                                    ),
                                    native_decoder_evaluations=(
                                        native_decoder_evaluations
                                    ),
                                    ranking_score=score,
                                )
                            )

                        closed = run_trajectory(
                            user=user,
                            domain=domain,
                            policy=policy_registry["soft_profile_conditioned"],
                            updater=updater,
                            turns=turns,
                            seed=seed,
                            initial_profile_condition="empty",
                            response_model=declared_response,
                            trajectory_id=(
                                f"{paired_key}:closed:{updater_id}"
                            ),
                            crn_key=f"{paired_key}:closed",
                        )
                        endogenous_trajectories.append(closed)
                        system_projection_score = evaluate_terminal_battery(
                            closed.terminal_belief,
                            user,
                            battery,
                        )
                        native_decoder_evaluations = (
                            evaluate_native_decoders(
                                closed.terminal_opaque_state,
                                user,
                                battery,
                            )
                        )
                        closed_score, score_basis = _ranking_score(
                            system_projection_score,
                            native_decoder_evaluations,
                        )
                        rows.append(
                            EvaluationRow(
                                split=split,
                                regime="endogenous_closed_loop",
                                replicate=replicate,
                                user_id=user.user_id,
                                domain_id=domain.domain_id,
                                updater_id=updater_id,
                                profile_error=closed_score.profile_brier,
                                behavioral_accuracy=(
                                    closed_score.behavioral_accuracy
                                ),
                                cross_context_accuracy=(
                                    closed_score.cross_context_accuracy
                                ),
                                intrinsic_regret=(
                                    closed_score.mean_intrinsic_regret
                                ),
                                history_digest=_audit_digest(
                                    closed.audit_record
                                ),
                                event_signatures=_audit_event_signatures(
                                    closed.audit_record
                                ),
                                battery_id=battery.battery_id,
                                battery_digest=battery.battery_digest,
                                predicted_option_ids=(
                                    closed_score.predicted_option_ids
                                ),
                                score_basis=score_basis,
                                system_projection_score=(
                                    system_projection_score
                                ),
                                native_decoder_evaluations=(
                                    native_decoder_evaluations
                                ),
                                ranking_score=closed_score,
                            )
                        )

    ranking = analyze_rankings(
        rows,
        updater_ids=tuple(updater_registry),
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
        tie_tolerance=tie_tolerance,
    )
    result = ExperimentCResult(
        fixed_histories=tuple(histories),
        terminal_batteries=tuple(batteries.values()),
        replay_results=tuple(replay_results),
        endogenous_trajectories=tuple(endogenous_trajectories),
        rows=tuple(rows),
        rankings=ranking,
    )
    result.assert_static_replay_identity()
    result.assert_terminal_battery_identity()
    return result


# Descriptive integration alias.
run_evaluation_validity_experiment = run_experiment_c
