"""Raw/calibrated LLM outcome scoring on already realized histories.

Calibration may alter later profile-conditioned actions and the prior embedded
in later prompts. The functions here therefore score both terminal forecasts
for the *same final prompt* from the active calibrated run. They never relabel
that paired diagnostic as a recursively raw counterfactual trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping

from .beliefs import MarginalPreferenceBelief, PreferenceBelief
from .experiments.evaluation import (
    TerminalBattery,
    TerminalBatteryScore,
    evaluate_terminal_battery,
)
from .llm_exchange import ATTRIBUTES, VALUES, LLMResponse
from .schemas import LatentUser, TrajectoryRecord


def belief_from_llm_response(response: LLMResponse) -> PreferenceBelief:
    """Convert one validated exchange response to the canonical joint belief."""

    rows = tuple(
        tuple(
            float(response.beliefs[attribute][value])
            for value in VALUES
        )
        for attribute in ATTRIBUTES
    )
    return PreferenceBelief.from_marginals(
        MarginalPreferenceBelief(rows)  # type: ignore[arg-type]
    )


def terminal_llm_request_id(
    audit_record: TrajectoryRecord,
    *,
    updater_id: str,
) -> str:
    """Recover the content-addressed terminal request from its audit delta."""

    if not audit_record.interactions:
        raise ValueError("LLM terminal scoring requires a non-empty history")
    update = audit_record.interactions[-1].profile_update
    if update is None or update.updater_id != updater_id:
        raise ValueError(
            "terminal audit interaction lacks the requested updater record"
        )
    prefix = f"{updater_id}:"
    candidates = []
    for delta in update.written_delta:
        if not delta.startswith("external ") or " response " not in delta:
            continue
        request_id = delta.rsplit(" ", 1)[-1]
        if request_id.startswith(prefix):
            candidates.append(request_id)
    if len(candidates) != 1:
        raise ValueError(
            "terminal LLM audit record must identify exactly one request"
        )
    return candidates[0]


def _response_index(
    responses: Iterable[LLMResponse] | Mapping[str, LLMResponse],
    *,
    label: str,
) -> dict[str, LLMResponse]:
    if isinstance(responses, Mapping):
        result = dict(responses)
        if any(
            request_id != response.request_id
            for request_id, response in result.items()
        ):
            raise ValueError(f"{label} response index has a mismatched key")
        return result
    result: dict[str, LLMResponse] = {}
    for response in responses:
        if response.request_id in result:
            raise ValueError(f"duplicate {label} response {response.request_id}")
        result[response.request_id] = response
    return result


def _same_marginals(
    first: PreferenceBelief,
    second: PreferenceBelief,
    *,
    tolerance: float = 1e-10,
) -> bool:
    return all(
        math.isclose(
            first_probability,
            second_probability,
            rel_tol=0.0,
            abs_tol=tolerance,
        )
        for attribute in range(3)
        for first_probability, second_probability in zip(
            first.marginal(attribute),
            second.marginal(attribute),
        )
    )


@dataclass(frozen=True, slots=True)
class CachedTerminalCalibrationOutcome:
    """One raw or calibrated score tied to an active-run terminal prompt."""

    experiment: str
    pairing_id: str
    split: str
    regime: str
    user_id: str
    domain_id: str
    updater_id: str
    calibration_variant: str
    request_id: str
    prompt_sha256: str
    model_id: str
    history_turns: int
    battery_id: str
    battery_digest: str
    score: TerminalBatteryScore

    @property
    def full_counterfactual_rerun_required(self) -> bool:
        return self.history_turns > 1

    @property
    def estimand_scope(self) -> str:
        if self.full_counterfactual_rerun_required:
            return "same-realized-history-terminal-forecast"
        return "complete-one-step-counterfactual"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "experiment": self.experiment,
            "pairing_id": self.pairing_id,
            "split": self.split,
            "regime": self.regime,
            "user_id": self.user_id,
            "domain_id": self.domain_id,
            "updater_id": self.updater_id,
            "calibration_variant": self.calibration_variant,
            "request_id": self.request_id,
            "prompt_sha256": self.prompt_sha256,
            "model_id": self.model_id,
            "history_turns": self.history_turns,
            "battery_id": self.battery_id,
            "battery_digest": self.battery_digest,
            "trajectory_pairing_preserved": True,
            "provider_calls_added": 0,
            "active_history_variant": "calibrated",
            "estimand_scope": self.estimand_scope,
            "full_counterfactual_rerun_required": (
                self.full_counterfactual_rerun_required
            ),
            "counterfactual_limitation": (
                (
                    "The raw row scores the uncalibrated forecast for the "
                    "terminal prompt generated on the calibrated active "
                    "history. A recursive raw run may produce different later "
                    "priors, prompts, responses, and endogenous actions."
                )
                if self.full_counterfactual_rerun_required
                else (
                    "The one-step raw and calibrated rows share the same "
                    "pre-update prior, prompt, observation, and action."
                )
            ),
            **self.score.to_dict(),
        }


def score_cached_raw_calibrated_terminal(
    *,
    experiment: str,
    pairing_id: str,
    split: str,
    regime: str,
    updater_id: str,
    active_terminal_belief: PreferenceBelief,
    audit_record: TrajectoryRecord,
    user: LatentUser,
    battery: TerminalBattery,
    raw_responses: Iterable[LLMResponse] | Mapping[str, LLMResponse],
    calibrated_responses: (
        Iterable[LLMResponse] | Mapping[str, LLMResponse]
    ),
) -> tuple[
    CachedTerminalCalibrationOutcome,
    CachedTerminalCalibrationOutcome,
]:
    """Score cached terminal vectors without calling a completion provider."""

    request_id = terminal_llm_request_id(
        audit_record,
        updater_id=updater_id,
    )
    raw_by_id = _response_index(raw_responses, label="raw")
    calibrated_by_id = _response_index(
        calibrated_responses,
        label="calibrated",
    )
    try:
        raw = raw_by_id[request_id]
        calibrated = calibrated_by_id[request_id]
    except KeyError as exc:
        raise ValueError(
            f"cached raw/calibrated response pair is missing {request_id}"
        ) from exc
    if (
        raw.prompt_sha256 != calibrated.prompt_sha256
        or raw.model_id != calibrated.model_id
    ):
        raise ValueError(
            "raw/calibrated terminal responses must share prompt and model"
        )
    calibrated_belief = belief_from_llm_response(calibrated)
    if not _same_marginals(
        active_terminal_belief,
        calibrated_belief,
    ):
        raise ValueError(
            "cached calibrated terminal response differs from active trajectory"
        )
    raw_belief = belief_from_llm_response(raw)
    common = {
        "experiment": experiment,
        "pairing_id": pairing_id,
        "split": split,
        "regime": regime,
        "user_id": user.user_id,
        "domain_id": battery.domain_id,
        "updater_id": updater_id,
        "request_id": request_id,
        "prompt_sha256": raw.prompt_sha256,
        "model_id": raw.model_id,
        "history_turns": len(audit_record.interactions),
        "battery_id": battery.battery_id,
        "battery_digest": battery.battery_digest,
    }
    return (
        CachedTerminalCalibrationOutcome(
            calibration_variant="raw",
            score=evaluate_terminal_battery(raw_belief, user, battery),
            **common,
        ),
        CachedTerminalCalibrationOutcome(
            calibration_variant="calibrated",
            score=evaluate_terminal_battery(
                calibrated_belief,
                user,
                battery,
            ),
            **common,
        ),
    )


def cached_outcome_manifest(
    experiment: str,
    rows: Iterable[CachedTerminalCalibrationOutcome],
) -> dict[str, Any]:
    material = tuple(rows)
    pair_ids = {row.pairing_id for row in material}
    variants_by_pair: dict[str, set[str]] = {}
    for row in material:
        variants_by_pair.setdefault(row.pairing_id, set()).add(
            row.calibration_variant
        )
    if any(
        variants != {"raw", "calibrated"}
        for variants in variants_by_pair.values()
    ):
        raise ValueError("every cached outcome pair needs raw and calibrated rows")
    return {
        "schema_version": 1,
        "experiment": experiment,
        "status": "complete" if material else "not_applicable",
        "row_count": len(material),
        "pair_count": len(pair_ids),
        "provider_calls_added": 0,
        "active_history_variant": "calibrated",
        "trajectory_pairing_preserved": True,
        "full_counterfactual_rerun_required_pair_count": len(
            {
                row.pairing_id
                for row in material
                if row.full_counterfactual_rerun_required
            }
        ),
        "ranking_or_gate_inputs_replaced": False,
        "interpretation": (
            "Rows compare raw and development-calibrated terminal forecasts "
            "for identical cached prompts on the realized active history. "
            "Multi-turn rows are not recursively raw trajectories and do not "
            "replace confirmatory rankings or gates."
        ),
    }
