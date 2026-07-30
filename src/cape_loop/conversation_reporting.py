"""Compact natural-language experiment traces with adjacent metrics.

The canonical event artifacts intentionally retain enough state for forensic
reconstruction.  They are not pleasant to read and can repeat the same visible
history once per evaluated updater.  This module builds a smaller reporting
view: each logical conversation is stored once, while updater outcomes are
grouped beside it.

Only visible dialogue, experimental condition labels, and derived metrics are
included.  Latent users, option feature vectors, beliefs, native memory, and
provider prompts/responses remain in their purpose-specific artifacts.
"""

from __future__ import annotations

from collections import defaultdict, deque
from hashlib import sha256
from typing import Any, Iterable, Mapping, Sequence
import json
import math

from .conversation_surfaces import ConversationTemplateBank
from .metrics import marginal_brier


SCHEMA_VERSION = 1
DEFAULT_MARKDOWN_PREVIEW_LIMIT = 100

_METRIC_LABELS = {
    "acue": "ACUE — excess update divergence (lower is better)",
    "exact_acue": "Exact-oracle ACUE (lower is better)",
    "brier": "Profile Brier error (lower is better)",
    "excess_brier": "Excess Brier error vs fitted aware reference",
    "fitted_aware_kl": "Divergence from fitted aware reference",
    "exact_kl": "Divergence from exact action-aware reference",
    "update_direction_accuracy": "Update-direction accuracy (higher is better)",
    "update_magnitude": "Update magnitude",
    "evidence_weight": "Evidence weight",
    "profile_error_after_turn": "Profile error after this turn (lower is better)",
    "shadow_profile_error_after_turn": (
        "Action-aware shadow error after this turn (lower is better)"
    ),
    "action_aware_information_gain": (
        "Action-aware information gain (higher is better)"
    ),
    "intrinsic_regret": "Intrinsic regret on this turn (lower is better)",
    "laundered_confidence_gain": "Laundered confidence gain by attribute",
    "profile_influenced_action": "Did the stored profile change the action?",
    "visible_action_diverged_from_balanced": (
        "Did the visible action differ from the paired balanced action?"
    ),
    "profile_aligned_treatment": (
        "Did the visible treatment promote the initially false profile?"
    ),
    "reinforcement_event": (
        "Did this turn satisfy all four partial-reinforcement clauses?"
    ),
    "initial_error": "Initial profile error",
    "terminal_error": "Terminal profile error (lower is better)",
    "error_amplification_ratio": (
        "Error amplification ratio, terminal divided by initial"
    ),
    "terminal_shadow_error": (
        "Terminal action-aware shadow error (lower is better)"
    ),
    "terminal_shadow_to_system_marginal_kl": (
        "Terminal divergence from the same-history shadow"
    ),
    "cumulative_information_gain": "Cumulative information gain",
    "cumulative_lcg": (
        "Cumulative excess confidence (CEC/LCG) by attribute"
    ),
    "mean_cumulative_excess_confidence_log_odds": (
        "Mean cumulative excess confidence on initially false attributes"
    ),
    "action_aware_disconfirmation_gain_log_odds": (
        "Action-aware evidence against initially false attributes"
    ),
    "profile_aligned_treatment_opportunities": (
        "Profile-aligned visible-treatment turns"
    ),
    "reinforcement_event_count": "Partial-reinforcement event count",
    "reinforcement_event_rate": (
        "Partial-reinforcement events divided by all turns"
    ),
    "total_regret": "Total intrinsic regret (lower is better)",
    "same_history_shadow": "Did the shadow consume the identical history?",
    "profile_error": "Terminal profile error (lower is better)",
    "behavioral_accuracy": "Held-out behavioral accuracy (higher is better)",
    "cross_context_accuracy": (
        "Held-out cross-context accuracy (higher is better)"
    ),
    "terminal_battery_mean_intrinsic_regret": (
        "Held-out mean intrinsic regret (lower is better)"
    ),
    "profile_ece": "Profile calibration error (lower is better)",
    "reportable_self_confirmation": (
        "Meets every declared self-confirmation clause?"
    ),
    "false_stable": "Incorrect profile remained stably wrong?",
    "evidence_selection_cost": "Evidence-selection cost",
    "profile_attribution_cost": "Profile-attribution cost",
    "balanced_attribution_cost": "Balanced-history attribution cost",
    "self_confirmation_interaction": "Self-confirmation interaction",
    "visible_action_divergence_rate": (
        "Profile-conditioned vs balanced visible-action divergence"
    ),
    "observed_choice_divergence_rate": (
        "Profile-conditioned vs balanced simulated-choice divergence"
    ),
    "action_aware_information_gain_deficit": (
        "Exploratory minus profile-conditioned information gain"
    ),
    "disconfirmation_evidence_deficit_log_odds": (
        "Exploratory minus profile-conditioned disconfirming evidence"
    ),
}


def updater_model_ids(
    registry: Mapping[str, Any],
) -> dict[str, tuple[str, ...]]:
    """Return the actual response model IDs observed by each updater."""

    result: dict[str, tuple[str, ...]] = {}
    for updater_id, updater in registry.items():
        responses = getattr(updater, "responses", ())
        result[updater_id] = tuple(
            sorted(
                {
                    model_id
                    for response in responses
                    if isinstance(
                        model_id := getattr(response, "model_id", None),
                        str,
                    )
                    and model_id
                }
            )
        )
    return result


def _updater_outcome(
    updater_id: str,
    metrics: Mapping[str, Any],
    *,
    updater_views: Mapping[str, str] | None,
    model_ids: Mapping[str, Sequence[str]] | None,
) -> dict[str, Any]:
    return {
        "updater_id": updater_id,
        "updater_view": (
            None if updater_views is None else updater_views.get(updater_id)
        ),
        "model_ids": list(
            () if model_ids is None else model_ids.get(updater_id, ())
        ),
        "metrics": dict(metrics),
    }


def _surface(
    *,
    context: Any,
    provenance: Any,
    observation: Any,
    conversation_bank: ConversationTemplateBank | None,
) -> tuple[str | None, str | None]:
    """Return the selected display label and frozen authoring source."""

    assistant = getattr(observation, "assistant_message", None)
    user = getattr(observation, "surface_response", None)
    if conversation_bank is None or not assistant or not user:
        return None, None
    rendered = conversation_bank.render(
        context,
        provenance,
        observation.selected_option_id,
    )
    if (
        rendered.assistant_message != assistant
        or rendered.user_message != user
        or rendered.surface_id != observation.surface_id
    ):
        raise ValueError(
            "retained observation differs from the configured frozen "
            "conversation surface"
        )
    return (
        rendered.display_names[observation.selected_option_id],
        rendered.source,
    )


def _dialogue_turn(
    *,
    turn: int,
    event_id: str,
    context: Any,
    provenance: Any,
    observation: Any,
    conversation_bank: ConversationTemplateBank | None,
    turn_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    selected_label, surface_source = _surface(
        context=context,
        provenance=provenance,
        observation=observation,
        conversation_bank=conversation_bank,
    )
    assistant = getattr(observation, "assistant_message", None)
    user = getattr(observation, "surface_response", None)
    return {
        "turn": turn,
        "event_id": event_id,
        "scenario_id": getattr(context, "scenario_id", None) or None,
        "surface_id": getattr(observation, "surface_id", None) or None,
        "surface_available": bool(assistant and user),
        "assistant": assistant,
        "user": user,
        "selected_option_id": observation.selected_option_id,
        "selected_option_label": selected_label,
        "presentation_mechanism": provenance.presentation_mechanism,
        "choice_source": "mathematical_user_simulator",
        "surface_source": surface_source,
        "turn_metrics": dict(turn_metrics or {}),
    }


def _record(
    *,
    experiment: str,
    conversation_id: str,
    conversation_kind: str,
    source_id: str,
    user_id: str,
    domain_id: str,
    conditions: Mapping[str, Any],
    dialogue: Sequence[Mapping[str, Any]],
    outcomes: Sequence[Mapping[str, Any]],
    assessments: Sequence[Mapping[str, Any]] = (),
    comparisons: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "record_kind": "conversation_trace",
        "experiment": experiment,
        "conversation_id": conversation_id,
        "conversation_kind": conversation_kind,
        "source_id": source_id,
        "user_id": user_id,
        "domain_id": domain_id,
        "conditions": dict(conditions),
        "dialogue": [dict(turn) for turn in dialogue],
        "outcomes": [dict(outcome) for outcome in outcomes],
        "assessments": [dict(item) for item in assessments],
        "comparisons": [dict(item) for item in comparisons],
    }


def build_experiment_a_records(
    rows: Sequence[Any],
    *,
    conversation_bank: ConversationTemplateBank | None = None,
    updater_views: Mapping[str, str] | None = None,
    model_ids: Mapping[str, Sequence[str]] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Group Experiment A's repeated updater rows by one visible trial."""

    grouped: dict[str, list[Any]] = {}
    for row in rows:
        grouped.setdefault(row.trial_id, []).append(row)
    records: list[dict[str, Any]] = []
    for trial_id, group in grouped.items():
        first = group[0]
        for row in group[1:]:
            if (
                row.user_id != first.user_id
                or row.domain_id != first.domain_id
                or row.context != first.context
                or row.provenance != first.provenance
                or row.observation != first.observation
                or row.mechanism != first.mechanism
                or row.response_mode != first.response_mode
                or row.prior_stratum != first.prior_stratum
                or row.prior_strength != first.prior_strength
            ):
                raise ValueError(
                    f"Experiment A trial {trial_id!r} does not share one "
                    "canonical conversation"
                )
        dialogue = (
            _dialogue_turn(
                turn=1,
                event_id=trial_id,
                context=first.context,
                provenance=first.provenance,
                observation=first.observation,
                conversation_bank=conversation_bank,
            ),
        )
        outcomes = tuple(
            _updater_outcome(
                row.updater_id,
                {
                    "acue": row.acue,
                    "exact_acue": row.exact_acue,
                    "brier": row.brier,
                    "excess_brier": row.excess_brier,
                    "fitted_aware_kl": row.fitted_aware_kl,
                    "exact_kl": row.exact_kl,
                    "update_direction_accuracy": (
                        row.update_direction_accuracy
                    ),
                    "update_magnitude": row.update_magnitude,
                    "evidence_weight": row.evidence_weight,
                },
                updater_views=updater_views,
                model_ids=model_ids,
            )
            for row in group
        )
        records.append(
            _record(
                experiment="A",
                conversation_id=trial_id,
                conversation_kind="single_turn",
                source_id=trial_id,
                user_id=first.user_id,
                domain_id=first.domain_id,
                conditions={
                    "split": "test",
                    "mechanism": first.mechanism,
                    "response_mode": first.response_mode,
                    "prior_stratum": first.prior_stratum,
                    "prior_strength": first.prior_strength,
                },
                dialogue=dialogue,
                outcomes=outcomes,
            )
        )
    return tuple(records)


def _assessment_view(assessment: Any) -> dict[str, Any]:
    return {
        "attribute": assessment.attribute,
        "metrics": {
            "cumulative_lcg": assessment.evidence.cumulative_lcg,
            "false_stable": assessment.false_stable,
            "reportable_self_confirmation": assessment.reportable,
        },
        "clause_results": dict(assessment.evidence.clauses()),
    }


def _comparison_view(comparison: Any) -> dict[str, Any]:
    return {
        "comparison_id": comparison.balanced_trajectory_id,
        "metrics": {
            "evidence_selection_cost": comparison.evidence_selection_cost,
            "profile_attribution_cost": (
                comparison.profile_attribution_cost
            ),
            "balanced_attribution_cost": (
                comparison.balanced_attribution_cost
            ),
            "self_confirmation_interaction": (
                comparison.self_confirmation_interaction
            ),
            "visible_action_divergence_rate": (
                comparison.visible_action_divergence_rate
            ),
            "observed_choice_divergence_rate": (
                comparison.observed_choice_divergence_rate
            ),
            "action_aware_information_gain_deficit": (
                comparison.action_aware_information_gain_deficit
            ),
            "disconfirmation_evidence_deficit_log_odds": (
                comparison.disconfirmation_evidence_deficit_log_odds
            ),
        },
    }


def build_closed_loop_records(
    trajectories: Sequence[Any],
    *,
    experiment: str,
    conversation_bank: ConversationTemplateBank | None = None,
    updater_views: Mapping[str, str] | None = None,
    model_ids: Mapping[str, Sequence[str]] | None = None,
    assessments: Sequence[Any] = (),
    comparisons: Sequence[Any] = (),
    split_by_user: Mapping[str, str] | None = None,
    extra_conditions: Mapping[str, Any] | None = None,
    conversation_id_prefix: str | None = None,
    outcome_metrics_by_source_id: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Build B-style traces, including C endogenous and sensitivity cases."""

    assessment_groups: dict[str, list[Any]] = defaultdict(list)
    for assessment in assessments:
        assessment_groups[assessment.trajectory_id].append(assessment)
    comparison_groups: dict[str, list[Any]] = defaultdict(list)
    for comparison in comparisons:
        comparison_groups[comparison.profile_trajectory_id].append(comparison)
    records: list[dict[str, Any]] = []
    for trajectory in trajectories:
        if len(trajectory.turns) != len(
            trajectory.audit_record.interactions
        ):
            raise ValueError(
                f"trajectory {trajectory.trajectory_id!r} turn/event mismatch"
            )
        profile_aligned_treatments = (
            trajectory.profile_aligned_treatment_flags()
        )
        reinforcement_events = trajectory.reinforcement_event_flags()
        dialogue = tuple(
            _dialogue_turn(
                turn=turn.turn + 1,
                event_id=turn.event_id,
                context=interaction.context,
                provenance=interaction.provenance,
                observation=interaction.observation,
                conversation_bank=conversation_bank,
                turn_metrics={
                    "profile_error_after_turn": marginal_brier(
                        turn.belief_after,
                        trajectory.theta,
                    ),
                    "shadow_profile_error_after_turn": marginal_brier(
                        turn.shadow_after,
                        trajectory.theta,
                    ),
                    "action_aware_information_gain": (
                        turn.action_aware_information_gain
                    ),
                    "intrinsic_regret": turn.intrinsic_regret,
                    "laundered_confidence_gain": list(
                        turn.laundered_confidence_gain
                    ),
                    "profile_influenced_action": (
                        turn.profile_influenced_action
                    ),
                    "visible_action_diverged_from_balanced": (
                        turn.action_signature
                        != turn.balanced_action_signature
                    ),
                    "profile_aligned_treatment": (
                        profile_aligned_treatment
                    ),
                    "reinforcement_event": reinforcement_event,
                },
            )
            for (
                turn,
                interaction,
                profile_aligned_treatment,
                reinforcement_event,
            ) in zip(
                trajectory.turns,
                trajectory.audit_record.interactions,
                profile_aligned_treatments,
                reinforcement_events,
            )
        )
        terminal_metrics: dict[str, Any] = {
            "initial_error": trajectory.initial_error,
            "terminal_error": trajectory.terminal_error,
            "error_amplification_ratio": (
                trajectory.error_amplification_ratio
            ),
            "terminal_shadow_error": trajectory.terminal_shadow_error,
            "terminal_shadow_to_system_marginal_kl": (
                trajectory.terminal_shadow_to_system_marginal_kl
            ),
            "cumulative_information_gain": (
                trajectory.cumulative_information_gain
            ),
            "cumulative_lcg": list(trajectory.cumulative_lcg),
            "mean_cumulative_excess_confidence_log_odds": (
                trajectory.mean_cumulative_excess_confidence_log_odds
            ),
            "action_aware_disconfirmation_gain_log_odds": (
                trajectory.action_aware_disconfirmation_gain_log_odds
            ),
            "profile_aligned_treatment_opportunities": (
                trajectory.profile_aligned_treatment_opportunities
            ),
            "reinforcement_event_count": (
                trajectory.reinforcement_event_count
            ),
            "reinforcement_event_rate": trajectory.reinforcement_event_rate,
            "total_regret": trajectory.total_regret,
            "same_history_shadow": trajectory.same_history_shadow,
        }
        if (
            outcome_metrics_by_source_id is not None
            and trajectory.trajectory_id in outcome_metrics_by_source_id
        ):
            terminal_metrics.update(
                outcome_metrics_by_source_id[trajectory.trajectory_id]
            )
        conditions = {
            "split": (
                None
                if split_by_user is None
                else split_by_user.get(trajectory.user_id)
            ),
            "policy_id": trajectory.policy_id,
            "initial_profile_condition": (
                trajectory.initial_profile_condition
            ),
        }
        if extra_conditions:
            conditions.update(extra_conditions)
        conversation_id = trajectory.trajectory_id
        if conversation_id_prefix:
            conversation_id = (
                f"{conversation_id_prefix}:{trajectory.trajectory_id}"
            )
        records.append(
            _record(
                experiment=experiment,
                conversation_id=conversation_id,
                conversation_kind="closed_loop",
                source_id=trajectory.trajectory_id,
                user_id=trajectory.user_id,
                domain_id=trajectory.domain_id,
                conditions=conditions,
                dialogue=dialogue,
                outcomes=(
                    _updater_outcome(
                        trajectory.updater_id,
                        terminal_metrics,
                        updater_views=updater_views,
                        model_ids=model_ids,
                    ),
                ),
                assessments=tuple(
                    _assessment_view(item)
                    for item in assessment_groups.get(
                        trajectory.trajectory_id,
                        (),
                    )
                ),
                comparisons=tuple(
                    _comparison_view(item)
                    for item in comparison_groups.get(
                        trajectory.trajectory_id,
                        (),
                    )
                ),
            )
        )
    return tuple(records)


def _event_signature(event: Any) -> str:
    payload = {
        "event_id": event.record_id,
        "context": event.context.to_dict(),
        "policy_provenance": event.provenance.to_dict(),
        "observation": event.observation.to_dict(),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return sha256(encoded.encode("utf-8")).hexdigest()


def _c_outcome_metrics(row: Any) -> dict[str, Any]:
    score = (
        row.system_projection_score
        if row.ranking_score is None
        else row.ranking_score
    )
    return {
        "profile_error": row.profile_error,
        "behavioral_accuracy": row.behavioral_accuracy,
        "cross_context_accuracy": row.cross_context_accuracy,
        "terminal_battery_mean_intrinsic_regret": row.intrinsic_regret,
        "profile_ece": score.profile_ece,
    }


def build_experiment_c_records(
    fixed_histories: Sequence[Any],
    endogenous_trajectories: Sequence[Any],
    evaluation_rows: Sequence[Any],
    *,
    conversation_bank: ConversationTemplateBank | None = None,
    updater_views: Mapping[str, str] | None = None,
    model_ids: Mapping[str, Sequence[str]] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Group fixed replays and match every endogenous trajectory evaluation."""

    rows_by_digest: dict[str, list[Any]] = defaultdict(list)
    rows_by_signatures: dict[tuple[str, ...], list[Any]] = defaultdict(list)
    for row in evaluation_rows:
        if row.regime in {"fixed_balanced", "fixed_biased"}:
            rows_by_digest[row.history_digest].append(row)
        elif row.regime == "endogenous_closed_loop":
            rows_by_signatures[tuple(row.event_signatures)].append(row)

    records: list[dict[str, Any]] = []
    consumed_rows: set[int] = set()
    for history in fixed_histories:
        grouped_rows = rows_by_digest.get(history.history_digest, [])
        if not grouped_rows:
            raise ValueError(
                f"fixed history {history.history_id!r} has no evaluation rows"
            )
        first = grouped_rows[0]
        if any(
            row.split != first.split
            or row.regime != first.regime
            or row.user_id != history.user_id
            or row.domain_id != history.domain_id
            for row in grouped_rows
        ):
            raise ValueError(
                f"fixed history {history.history_id!r} evaluation mismatch"
            )
        dialogue = tuple(
            _dialogue_turn(
                turn=index,
                event_id=event.event_id,
                context=event.context,
                provenance=event.provenance,
                observation=event.observation,
                conversation_bank=conversation_bank,
            )
            for index, event in enumerate(history.events, start=1)
        )
        outcomes = tuple(
            _updater_outcome(
                row.updater_id,
                _c_outcome_metrics(row),
                updater_views=updater_views,
                model_ids=model_ids,
            )
            for row in grouped_rows
        )
        consumed_rows.update(id(row) for row in grouped_rows)
        records.append(
            _record(
                experiment="C",
                conversation_id=history.history_id,
                conversation_kind="fixed_history",
                source_id=history.history_id,
                user_id=history.user_id,
                domain_id=history.domain_id,
                conditions={
                    "split": first.split,
                    "regime": first.regime,
                    "replicate": first.replicate,
                    "logger_policy_id": history.logger_policy_id,
                },
                dialogue=dialogue,
                outcomes=outcomes,
            )
        )

    endogenous_metric_rows: dict[str, Mapping[str, Any]] = {}
    split_by_user: dict[str, str] = {}
    conditions_by_id: dict[str, dict[str, Any]] = {}
    for trajectory in endogenous_trajectories:
        signatures = tuple(
            _event_signature(event)
            for event in trajectory.audit_record.interactions
        )
        candidates = [
            row
            for row in rows_by_signatures.get(signatures, ())
            if row.user_id == trajectory.user_id
            and row.domain_id == trajectory.domain_id
            and row.updater_id == trajectory.updater_id
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"endogenous trajectory {trajectory.trajectory_id!r} "
                f"matched {len(candidates)} evaluation rows"
            )
        row = candidates[0]
        consumed_rows.add(id(row))
        endogenous_metric_rows[trajectory.trajectory_id] = (
            _c_outcome_metrics(row)
        )
        split_by_user[trajectory.user_id] = row.split
        conditions_by_id[trajectory.trajectory_id] = {
            "regime": row.regime,
            "replicate": row.replicate,
        }

    for trajectory in endogenous_trajectories:
        built = build_closed_loop_records(
            (trajectory,),
            experiment="C",
            conversation_bank=conversation_bank,
            updater_views=updater_views,
            model_ids=model_ids,
            split_by_user=split_by_user,
            extra_conditions=conditions_by_id[trajectory.trajectory_id],
            outcome_metrics_by_source_id=endogenous_metric_rows,
        )
        records.extend(built)

    if len(consumed_rows) != len(evaluation_rows):
        raise ValueError(
            "Experiment C conversation reporting did not consume every "
            "evaluation row exactly once"
        )
    return tuple(records)


def conversation_stats(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, int]:
    record_count = 0
    turn_count = 0
    outcome_count = 0
    for record in records:
        record_count += 1
        turn_count += len(record["dialogue"])
        outcome_count += len(record["outcomes"])
    return {
        "record_count": record_count,
        "turn_count": turn_count,
        "outcome_count": outcome_count,
    }


def _preview_group(record: Mapping[str, Any]) -> tuple[str, ...]:
    conditions = record.get("conditions", {})
    outcomes = record.get("outcomes", ())
    first_updater = (
        str(outcomes[0].get("updater_id"))
        if outcomes
        else "no-updater"
    )
    return (
        str(record.get("experiment")),
        str(record.get("domain_id")),
        str(record.get("conversation_kind")),
        str(conditions.get("sensitivity_point_id")),
        str(conditions.get("regime")),
        str(conditions.get("mechanism")),
        str(conditions.get("policy_id")),
        first_updater,
    )


def select_diverse_records(
    records: Sequence[Mapping[str, Any]],
    *,
    limit: int = DEFAULT_MARKDOWN_PREVIEW_LIMIT,
) -> tuple[Mapping[str, Any], ...]:
    """Deterministically round-robin across experimental condition groups."""

    if limit < 0:
        raise ValueError("preview limit cannot be negative")
    if limit == 0 or not records:
        return ()
    groups: dict[tuple[str, ...], deque[Mapping[str, Any]]] = {}
    for record in records:
        groups.setdefault(_preview_group(record), deque()).append(record)
    ordered = sorted(groups.items(), key=lambda item: item[0])
    selected: list[Mapping[str, Any]] = []
    while ordered and len(selected) < limit:
        next_round: list[
            tuple[tuple[str, ...], deque[Mapping[str, Any]]]
        ] = []
        for key, queue in ordered:
            if len(selected) >= limit:
                break
            selected.append(queue.popleft())
            if queue:
                next_round.append((key, queue))
        ordered = next_round
    return tuple(selected)


def _format_value(value: Any) -> str:
    if value is None:
        return "not available"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        return f"{value:.6g}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_format_value(item) for item in value) + "]"
    return str(value)


def _metric_lines(
    metrics: Mapping[str, Any],
    *,
    indent: str,
) -> list[str]:
    return [
        (
            f"{indent}- {_METRIC_LABELS.get(key, key.replace('_', ' '))}: "
            f"{_format_value(value)}"
        )
        for key, value in metrics.items()
    ]


def _blockquote(value: str | None) -> list[str]:
    if value is None:
        return ["> _Natural-language surface was not configured for this run._"]
    return [f"> {line}" if line else ">" for line in value.splitlines()]


def render_markdown(
    records: Sequence[Mapping[str, Any]],
    *,
    experiment: str,
    complete_stats: Mapping[str, int] | None = None,
    complete_jsonl_path: str,
    preview_limit: int = DEFAULT_MARKDOWN_PREVIEW_LIMIT,
    records_are_preselected: bool = False,
) -> str:
    """Render a bounded, human-readable preview with exact complete totals."""

    stats = (
        conversation_stats(records)
        if complete_stats is None
        else {
            "record_count": int(complete_stats["record_count"]),
            "turn_count": int(complete_stats["turn_count"]),
            "outcome_count": int(complete_stats["outcome_count"]),
        }
    )
    preview = (
        tuple(records[:preview_limit])
        if records_are_preselected
        else select_diverse_records(records, limit=preview_limit)
    )
    lines = [
        f"# Experiment {experiment} conversation and metric log",
        "",
        (
            f"This is a readable preview of **{len(preview)}** of "
            f"**{stats['record_count']}** complete conversation records "
            f"({stats['turn_count']} dialogue turns and "
            f"{stats['outcome_count']} evaluated updater outcomes)."
        ),
        (
            f"The exhaustive deduplicated data is in "
            f"`{complete_jsonl_path}`."
        ),
        "",
        "## How to read this log",
        "",
        (
            "- **Scenario presenter (assistant)** is frozen scenario wording; "
            "it is not the model being evaluated."
        ),
        (
            "- **Simulated user** is a natural-language rendering of the "
            "option already selected by the mathematical response model."
        ),
        (
            "- **Evaluated profile updater** is the system/model whose metrics "
            "appear after the dialogue. Its `updater_view` states what it "
            "actually received; a response-only updater did not see the full "
            "assistant turn even though the audit log preserves it."
        ),
        (
            "- For an external updater, `model_ids` identifies the model used. "
            "The exact provider requests, responses, and transport audit remain "
            "in the sibling `llm/` directory instead of being duplicated here."
        ),
        (
            "- Lower error, regret, divergence, ACUE, and calibration values "
            "are better. Higher accuracy and information gain values are "
            "better. Exact values are retained in JSONL; this preview rounds "
            "numbers for readability."
        ),
        "",
    ]
    if not preview:
        lines.extend(["_No conversation records were produced._", ""])
        return "\n".join(lines)
    for index, record in enumerate(preview, start=1):
        lines.extend(
            [
                f"## {index}. Conversation `{record['conversation_id']}`",
                "",
                (
                    f"- Experiment: `{record['experiment']}`; kind: "
                    f"`{record['conversation_kind']}`"
                ),
                (
                    f"- User: `{record['user_id']}`; domain: "
                    f"`{record['domain_id']}`"
                ),
            ]
        )
        conditions = record.get("conditions", {})
        if conditions:
            lines.append(
                "- Conditions: "
                + "; ".join(
                    f"{key}={_format_value(value)}"
                    for key, value in conditions.items()
                    if value is not None
                )
            )
        lines.append("")
        for turn in record["dialogue"]:
            lines.extend(
                [
                    f"### Turn {turn['turn']}",
                    "",
                    "**Scenario presenter (assistant):**",
                    "",
                    *_blockquote(turn.get("assistant")),
                    "",
                    "**Simulated user:**",
                    "",
                    *_blockquote(turn.get("user")),
                    "",
                    (
                        "**Recorded choice:** "
                        + (
                            str(turn["selected_option_label"])
                            if turn.get("selected_option_label")
                            else f"`{turn['selected_option_id']}`"
                        )
                        + f"; presentation mechanism: "
                        f"`{turn['presentation_mechanism']}`."
                    ),
                    "",
                ]
            )
            if turn.get("turn_metrics"):
                lines.extend(
                    [
                        "**Metrics after this turn:**",
                        "",
                        *_metric_lines(
                            turn["turn_metrics"],
                            indent="",
                        ),
                        "",
                    ]
                )
        lines.extend(
            [
                "### Evaluated profile updater outcomes",
                "",
            ]
        )
        for outcome in record["outcomes"]:
            models = outcome.get("model_ids") or []
            model_text = ", ".join(f"`{item}`" for item in models)
            if not model_text:
                model_text = "none (deterministic updater)"
            lines.extend(
                [
                    (
                        f"- **`{outcome['updater_id']}`**; view: "
                        f"`{outcome.get('updater_view') or 'not recorded'}`; "
                        f"model(s): {model_text}"
                    ),
                    *_metric_lines(
                        outcome.get("metrics", {}),
                        indent="  ",
                    ),
                ]
            )
        if record.get("assessments"):
            lines.extend(["", "### Self-confirmation assessments", ""])
            for item in record["assessments"]:
                lines.append(f"- Attribute {item['attribute'] + 1}")
                lines.extend(
                    _metric_lines(item["metrics"], indent="  ")
                )
                clauses = item.get("clause_results", {})
                if clauses:
                    lines.append(
                        "  - Clause results: "
                        + ", ".join(
                            f"{key}={_format_value(value)}"
                            for key, value in clauses.items()
                        )
                    )
        if record.get("comparisons"):
            lines.extend(["", "### Matched trajectory comparisons", ""])
            for item in record["comparisons"]:
                lines.append(
                    f"- Matched case: `{item['comparison_id']}`"
                )
                lines.extend(
                    _metric_lines(item["metrics"], indent="  ")
                )
        lines.extend(["", "---", ""])
    return "\n".join(lines)
