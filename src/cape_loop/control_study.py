"""Executable, content-addressed Experiment A positive/negative controls.

The one-step choice schema used by the main provenance audit cannot faithfully
represent direct statements, explicit indifference, or a registered
randomization device.  This module therefore gives the six proposal controls a
separate exchange boundary:

* :func:`build_experiment_a_control_plan` materializes fixed stimuli and binds
  them to ``experiment-a-controls-v1``;
* :func:`run_diagnostic_control_executions` runs an inspectable reference and a
  no-update baseline without presenting either as empirical evidence;
* :func:`build_control_llm_exchange` creates provider-neutral
  :class:`~cape_loop.llm_exchange.LLMRequest` records only for information views
  that can faithfully carry a control; and
* :func:`execute_control_llm_exchange` validates and scores replayed or live
  provider responses while retaining request, prompt, model, and response
  bindings.

Every report keeps ``claim_status = "not_claimed"``.  Passing a diagnostic
criterion is a software/protocol result, not a paper finding.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import json
import math

from .beliefs import MarginalPreferenceBelief
from .correction_debt import (
    CorrectionProtocol,
    ReferenceLogOddsCorrectionAdapter,
)
from .experiments.provenance import (
    ExperimentAControlBattery,
    ExperimentAControlCase,
    build_experiment_a_control_battery,
)
from .llm_exchange import (
    ATTRIBUTES,
    VALUES,
    CompletionProvider,
    LLMRequest,
    LLMResponse,
    ReplayProvider,
    VIEWS,
    write_requests,
)
from .schemas import NUM_ATTRIBUTES, Option, THETA_VALUES


CONTROL_PLAN_VERSION = "experiment-a-control-execution-v1"
CONTROL_EXCHANGE_VERSION = "experiment-a-control-llm-exchange-v1"
CONTROL_SCHEMA_VERSION = 1
CLAIM_STATUS = "not_claimed"

CONTROL_IDS = (
    "positive-volunteered-preference",
    "positive-repeated-balanced-cross-context",
    "positive-direct-correction",
    "negative-indifferent-response",
    "negative-random-choice",
    "negative-nondistinguishing-response",
)

CONTROL_VALID_VIEWS: Mapping[str, frozenset[str]] = {
    "positive-volunteered-preference": frozenset(VIEWS),
    "positive-repeated-balanced-cross-context": frozenset(
        {"full_context", "provenance_aware"}
    ),
    "positive-direct-correction": frozenset(VIEWS),
    "negative-indifferent-response": frozenset(VIEWS),
    # A choice is identifiable as random only when its registered generation
    # provenance is visible.  Sending it to a response-only/full-context writer
    # would silently change the negative-control estimand.
    "negative-random-choice": frozenset({"provenance_aware"}),
    "negative-nondistinguishing-response": frozenset(
        {"full_context", "provenance_aware"}
    ),
}

_POLARITIES = frozenset({"positive", "negative"})
_RESPONSE_KINDS = frozenset(
    {"volunteered_statement", "choice", "direct_correction", "indifference"}
)
_RESPONSE_SOURCES = frozenset(
    {"user", "registered_randomization_device"}
)
_EXECUTION_MODES = frozenset(
    {
        "deterministic_reference",
        "deterministic_no_update_baseline",
        "provider_replay",
        "provider_live",
    }
)
_EVIDENCE_CLASSES = {
    "deterministic_reference": "diagnostic_reference",
    "deterministic_no_update_baseline": "diagnostic_baseline",
    "provider_replay": "external_model_response",
    "provider_live": "external_model_response",
}

ProbabilityRow = tuple[float, float, float, float]
ProbabilityRows = tuple[ProbabilityRow, ProbabilityRow, ProbabilityRow]


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _digest(value: Any) -> str:
    return sha256(_canonical(value).encode("utf-8")).hexdigest()


def _require_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _validate_digest(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _probability_rows(value: object, *, name: str) -> ProbabilityRows:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != NUM_ATTRIBUTES
    ):
        raise ValueError(f"{name} must contain exactly three rows")
    rows: list[ProbabilityRow] = []
    for attribute, raw in enumerate(value):
        if (
            not isinstance(raw, Sequence)
            or isinstance(raw, (str, bytes))
            or len(raw) != len(THETA_VALUES)
        ):
            raise ValueError(
                f"{name}[{attribute}] must contain four probabilities"
            )
        row: list[float] = []
        for index, item in enumerate(raw):
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise TypeError(
                    f"{name}[{attribute}][{index}] must be numeric"
                )
            parsed = float(item)
            if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
                raise ValueError(
                    f"{name}[{attribute}][{index}] must lie in [0, 1]"
                )
            row.append(parsed)
        if not math.isclose(
            math.fsum(row),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(f"{name}[{attribute}] must sum to one")
        rows.append((row[0], row[1], row[2], row[3]))
    return (rows[0], rows[1], rows[2])


def _rows_to_dict(rows: ProbabilityRows) -> dict[str, dict[str, float]]:
    return {
        attribute: {
            value: probability
            for value, probability in zip(VALUES, rows[index])
        }
        for index, attribute in enumerate(ATTRIBUTES)
    }


def _rows_from_response(response: LLMResponse) -> ProbabilityRows:
    return _probability_rows(
        tuple(
            tuple(
                float(response.beliefs[attribute][value])
                for value in VALUES
            )
            for attribute in ATTRIBUTES
        ),
        name="response beliefs",
    )


def _uniform_rows() -> ProbabilityRows:
    return MarginalPreferenceBelief.uniform().probabilities


def _sign_mass(
    rows: ProbabilityRows,
    attribute: int,
    direction: int,
) -> float:
    if not 0 <= attribute < NUM_ATTRIBUTES:
        raise ValueError("attribute must be in [0, 3)")
    if direction not in (-1, 1):
        raise ValueError("direction must be -1 or +1")
    indexes = (0, 1) if direction < 0 else (2, 3)
    return rows[attribute][indexes[0]] + rows[attribute][indexes[1]]


def _with_sign_mass(
    rows: ProbabilityRows,
    *,
    attribute: int,
    direction: int,
    direction_mass: float,
) -> ProbabilityRows:
    if not 0.0 <= direction_mass <= 1.0:
        raise ValueError("direction_mass must lie in [0, 1]")
    current = rows[attribute]
    direction_indexes = (0, 1) if direction < 0 else (2, 3)
    opposite_indexes = (2, 3) if direction < 0 else (0, 1)

    def redistributed(indexes: tuple[int, int], total: float) -> tuple[float, float]:
        existing = current[indexes[0]] + current[indexes[1]]
        if existing <= 0.0:
            return (total / 2.0, total / 2.0)
        return (
            total * current[indexes[0]] / existing,
            total * current[indexes[1]] / existing,
        )

    toward = redistributed(direction_indexes, direction_mass)
    away = redistributed(opposite_indexes, 1.0 - direction_mass)
    updated = [0.0, 0.0, 0.0, 0.0]
    updated[direction_indexes[0]], updated[direction_indexes[1]] = toward
    updated[opposite_indexes[0]], updated[opposite_indexes[1]] = away
    material = list(rows)
    material[attribute] = (
        updated[0],
        updated[1],
        updated[2],
        updated[3],
    )
    return (material[0], material[1], material[2])


def _bayes_sign_update(
    rows: ProbabilityRows,
    *,
    attribute: int,
    direction: int,
    reliability: float,
) -> ProbabilityRows:
    if not 0.5 < reliability < 1.0:
        raise ValueError("reliability must lie strictly in (0.5, 1)")
    before = _sign_mass(rows, attribute, direction)
    numerator = reliability * before
    denominator = numerator + (1.0 - reliability) * (1.0 - before)
    return _with_sign_mass(
        rows,
        attribute=attribute,
        direction=direction,
        direction_mass=numerator / denominator,
    )


def _option_from_dict(raw: Mapping[str, Any]) -> Option:
    allowed = {"option_id", "features", "label", "domain"}
    if set(raw) != allowed:
        raise ValueError("control option fields do not match the Option contract")
    features = raw["features"]
    if not isinstance(features, Sequence) or isinstance(features, (str, bytes)):
        raise ValueError("control option features must be an array")
    return Option(
        option_id=raw["option_id"],
        features=tuple(features),  # type: ignore[arg-type]
        label=raw["label"],
        domain=raw["domain"],
    )


@dataclass(frozen=True, slots=True)
class ControlRandomizationRegistration:
    """Precommitted deterministic device for the random-choice control."""

    registration_id: str
    algorithm: str
    seed_material: str
    draw_count: int
    registration_sha256: str = ""

    def __post_init__(self) -> None:
        _require_text(self.registration_id, "registration_id")
        if self.algorithm != "sha256-modulo-option-count-v1":
            raise ValueError("unsupported randomization algorithm")
        _require_text(self.seed_material, "seed_material")
        if (
            isinstance(self.draw_count, bool)
            or not isinstance(self.draw_count, int)
            or self.draw_count <= 0
        ):
            raise ValueError("draw_count must be a positive integer")
        expected = _digest(self._binding_payload())
        if self.registration_sha256 and self.registration_sha256 != expected:
            raise ValueError(
                "randomization registration digest does not bind its content"
            )
        object.__setattr__(self, "registration_sha256", expected)

    def _binding_payload(self) -> dict[str, Any]:
        return {
            "registration_id": self.registration_id,
            "algorithm": self.algorithm,
            "seed_material": self.seed_material,
            "draw_count": self.draw_count,
        }

    def draw(self, *, event_id: str, option_count: int) -> tuple[int, str]:
        _require_text(event_id, "event_id")
        if (
            isinstance(option_count, bool)
            or not isinstance(option_count, int)
            or option_count < 2
        ):
            raise ValueError("option_count must be an integer of at least two")
        draw_sha256 = sha256(
            (
                self.registration_sha256
                + "\n"
                + event_id
                + "\n"
                + str(option_count)
            ).encode("utf-8")
        ).hexdigest()
        return int(draw_sha256, 16) % option_count, draw_sha256

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CONTROL_SCHEMA_VERSION,
            **self._binding_payload(),
            "registration_sha256": self.registration_sha256,
        }


@dataclass(frozen=True, slots=True)
class ExperimentAControlEvent:
    """One concrete direct statement, response, or choice in a control."""

    event_id: str
    turn_index: int
    response_kind: str
    surface_response: str
    scenario_id: str
    wording_template_id: str
    options: tuple[Option, ...] = ()
    ranking: tuple[str, ...] = ()
    selected_option_id: str | None = None
    response_source: str = "user"
    elicitation_provenance: str = "none"
    randomization_registration_sha256: str | None = None
    randomization_draw_sha256: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "event_id",
            "surface_response",
            "scenario_id",
            "wording_template_id",
            "elicitation_provenance",
        ):
            _require_text(getattr(self, name), name)
        if (
            isinstance(self.turn_index, bool)
            or not isinstance(self.turn_index, int)
            or self.turn_index < 0
        ):
            raise ValueError("turn_index must be a non-negative integer")
        if self.response_kind not in _RESPONSE_KINDS:
            raise ValueError(f"response_kind must be one of {sorted(_RESPONSE_KINDS)}")
        if self.response_source not in _RESPONSE_SOURCES:
            raise ValueError(
                f"response_source must be one of {sorted(_RESPONSE_SOURCES)}"
            )
        options = tuple(self.options)
        if not all(isinstance(option, Option) for option in options):
            raise TypeError("control event options must contain only Option objects")
        identifiers = tuple(option.option_id for option in options)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("control event option IDs must be unique")
        ranking = tuple(self.ranking)
        if set(ranking) != set(identifiers) or len(ranking) != len(identifiers):
            raise ValueError("control event ranking must exactly cover its options")
        if self.selected_option_id is not None and self.selected_option_id not in identifiers:
            raise ValueError("selected_option_id must name a displayed option")
        if self.response_kind == "choice" and self.selected_option_id is None:
            raise ValueError("choice controls require a selected option")
        if self.response_kind != "choice" and self.selected_option_id is not None:
            raise ValueError("only a choice control may select an option")
        if self.response_source == "registered_randomization_device":
            if self.response_kind != "choice":
                raise ValueError("the randomization device can emit only choices")
            _validate_digest(
                self.randomization_registration_sha256,
                "randomization_registration_sha256",
            )
            _validate_digest(
                self.randomization_draw_sha256,
                "randomization_draw_sha256",
            )
        elif (
            self.randomization_registration_sha256 is not None
            or self.randomization_draw_sha256 is not None
        ):
            raise ValueError(
                "user responses cannot carry randomization-device bindings"
            )
        object.__setattr__(self, "options", options)
        object.__setattr__(self, "ranking", ranking)

    @property
    def selected_option(self) -> Option | None:
        if self.selected_option_id is None:
            return None
        return next(
            option
            for option in self.options
            if option.option_id == self.selected_option_id
        )

    def observation_payload(self) -> dict[str, Any]:
        return {
            "response_kind": self.response_kind,
            "surface_response": self.surface_response,
            "selected_option": (
                None
                if self.selected_option is None
                else self.selected_option.to_dict()
            ),
        }

    def context_payload(self, *, target_attribute: int) -> dict[str, Any]:
        return {
            "domain": (
                self.options[0].domain if self.options else "direct_statement"
            ),
            "scenario_id": self.scenario_id,
            "wording_template_id": self.wording_template_id,
            "options": [option.to_dict() for option in self.options],
            "ranking": list(self.ranking),
            "question_type": self.response_kind,
            "target_attribute": target_attribute,
        }

    def provenance_payload(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "response_source": self.response_source,
            "elicitation_provenance": self.elicitation_provenance,
        }
        if self.response_source == "registered_randomization_device":
            result["registered_randomization"] = {
                "registration_sha256": (
                    self.randomization_registration_sha256
                ),
                "draw_sha256": self.randomization_draw_sha256,
                "preference_independent": True,
            }
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "turn_index": self.turn_index,
            "response_kind": self.response_kind,
            "surface_response": self.surface_response,
            "scenario_id": self.scenario_id,
            "wording_template_id": self.wording_template_id,
            "options": [option.to_dict() for option in self.options],
            "ranking": list(self.ranking),
            "selected_option_id": self.selected_option_id,
            "response_source": self.response_source,
            "elicitation_provenance": self.elicitation_provenance,
            "randomization_registration_sha256": (
                self.randomization_registration_sha256
            ),
            "randomization_draw_sha256": self.randomization_draw_sha256,
        }

    @classmethod
    def parse(cls, raw: Mapping[str, Any]) -> "ExperimentAControlEvent":
        allowed = {
            "event_id",
            "turn_index",
            "response_kind",
            "surface_response",
            "scenario_id",
            "wording_template_id",
            "options",
            "ranking",
            "selected_option_id",
            "response_source",
            "elicitation_provenance",
            "randomization_registration_sha256",
            "randomization_draw_sha256",
        }
        if set(raw) != allowed:
            raise ValueError("unknown or missing control event fields")
        raw_options = raw["options"]
        if not isinstance(raw_options, Sequence) or isinstance(
            raw_options, (str, bytes)
        ):
            raise ValueError("control event options must be an array")
        options = tuple(
            _option_from_dict(option)
            for option in raw_options
            if isinstance(option, Mapping)
        )
        if len(options) != len(raw_options):
            raise ValueError("every control event option must be an object")
        raw_ranking = raw["ranking"]
        if not isinstance(raw_ranking, Sequence) or isinstance(
            raw_ranking, (str, bytes)
        ):
            raise ValueError("control event ranking must be an array")
        return cls(
            event_id=raw["event_id"],
            turn_index=raw["turn_index"],
            response_kind=raw["response_kind"],
            surface_response=raw["surface_response"],
            scenario_id=raw["scenario_id"],
            wording_template_id=raw["wording_template_id"],
            options=options,
            ranking=tuple(raw_ranking),
            selected_option_id=raw["selected_option_id"],
            response_source=raw["response_source"],
            elicitation_provenance=raw["elicitation_provenance"],
            randomization_registration_sha256=raw[
                "randomization_registration_sha256"
            ],
            randomization_draw_sha256=raw["randomization_draw_sha256"],
        )


@dataclass(frozen=True, slots=True)
class ExperimentAControlStimulus:
    """A fully executable stimulus bound to one fixed battery case."""

    control_id: str
    case_sha256: str
    polarity: str
    signal_kind: str
    target_attribute: int
    target_direction: int
    prior_probabilities: ProbabilityRows
    events: tuple[ExperimentAControlEvent, ...]
    minimum_directional_mass_delta: float = 0.05
    maximum_negative_control_abs_delta: float = 1e-9
    stimulus_sha256: str = ""

    def __post_init__(self) -> None:
        _require_text(self.control_id, "control_id")
        _validate_digest(self.case_sha256, "case_sha256")
        if self.polarity not in _POLARITIES:
            raise ValueError(f"polarity must be one of {sorted(_POLARITIES)}")
        _require_text(self.signal_kind, "signal_kind")
        if (
            isinstance(self.target_attribute, bool)
            or not isinstance(self.target_attribute, int)
            or not 0 <= self.target_attribute < NUM_ATTRIBUTES
        ):
            raise ValueError("target_attribute must be in [0, 3)")
        if self.target_direction not in (-1, 1):
            raise ValueError("target_direction must be -1 or +1")
        rows = _probability_rows(
            self.prior_probabilities,
            name="prior_probabilities",
        )
        events = tuple(self.events)
        if not events:
            raise ValueError("a control stimulus requires at least one event")
        if len({event.event_id for event in events}) != len(events):
            raise ValueError("control event IDs must be unique")
        if tuple(event.turn_index for event in events) != tuple(range(len(events))):
            raise ValueError("control turn indexes must be contiguous from zero")
        for name in (
            "minimum_directional_mass_delta",
            "maximum_negative_control_abs_delta",
        ):
            raw = getattr(self, name)
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise TypeError(f"{name} must be numeric")
            value = float(raw)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "prior_probabilities", rows)
        object.__setattr__(self, "events", events)
        self._validate_control_semantics()
        expected = _digest(self._binding_payload())
        if self.stimulus_sha256 and self.stimulus_sha256 != expected:
            raise ValueError("stimulus_sha256 does not bind the control content")
        object.__setattr__(self, "stimulus_sha256", expected)

    def _validate_control_semantics(self) -> None:
        if self.control_id == "positive-volunteered-preference":
            if (
                len(self.events) != 1
                or self.events[0].response_kind != "volunteered_statement"
                or self.events[0].options
                or self.events[0].response_source != "user"
            ):
                raise ValueError("volunteered control must be one user statement")
        elif self.control_id == "positive-repeated-balanced-cross-context":
            if len(self.events) != 3:
                raise ValueError("repeated balanced control requires three turns")
            if len({event.scenario_id for event in self.events}) != 3 or len(
                {event.wording_template_id for event in self.events}
            ) != 3:
                raise ValueError(
                    "repeated balanced turns require disjoint scenarios and wording"
                )
            for event in self.events:
                selected = event.selected_option
                if (
                    event.response_kind != "choice"
                    or event.response_source != "user"
                    or selected is None
                    or selected.features[self.target_attribute]
                    * self.target_direction
                    <= 0.0
                    or len(event.options) != 2
                    or math.prod(
                        option.features[self.target_attribute]
                        for option in event.options
                    )
                    >= 0.0
                ):
                    raise ValueError(
                        "each repeated turn must select the target direction "
                        "from a balanced opposing pair"
                    )
        elif self.control_id == "positive-direct-correction":
            if (
                len(self.events) != 1
                or self.events[0].response_kind != "direct_correction"
                or self.events[0].options
            ):
                raise ValueError("direct correction must be one option-free statement")
        elif self.control_id == "negative-indifferent-response":
            if (
                len(self.events) != 1
                or self.events[0].response_kind != "indifference"
                or len(self.events[0].options) != 2
            ):
                raise ValueError("indifference control requires one two-option response")
        elif self.control_id == "negative-random-choice":
            if len(self.events) != 3 or any(
                event.response_source != "registered_randomization_device"
                for event in self.events
            ):
                raise ValueError(
                    "random-choice control requires three registered device draws"
                )
        elif self.control_id == "negative-nondistinguishing-response":
            if len(self.events) != 1 or self.events[0].response_kind != "choice":
                raise ValueError(
                    "nondistinguishing control requires one selected choice"
                )
            target_values = {
                option.features[self.target_attribute]
                for option in self.events[0].options
            }
            if len(target_values) != 1:
                raise ValueError(
                    "nondistinguishing options must share the target feature"
                )
        else:
            raise ValueError(f"unknown Experiment A control {self.control_id!r}")

    def _binding_payload(self) -> dict[str, Any]:
        return {
            "control_id": self.control_id,
            "case_sha256": self.case_sha256,
            "polarity": self.polarity,
            "signal_kind": self.signal_kind,
            "target_attribute": self.target_attribute,
            "target_direction": self.target_direction,
            "prior_probabilities": _rows_to_dict(self.prior_probabilities),
            "events": [event.to_dict() for event in self.events],
            "minimum_directional_mass_delta": (
                self.minimum_directional_mass_delta
            ),
            "maximum_negative_control_abs_delta": (
                self.maximum_negative_control_abs_delta
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CONTROL_SCHEMA_VERSION,
            **self._binding_payload(),
            "stimulus_sha256": self.stimulus_sha256,
        }

    def model_payload(self, view: str) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
        if view not in VIEWS:
            raise ValueError(f"unknown LLM view: {view}")
        observations = [
            event.observation_payload()
            for event in self.events
        ]
        observation: dict[str, Any] = {
            "event_sequence": observations,
            "target_attribute": self.target_attribute,
        }
        context = (
            None
            if view == "response_only"
            else {
                "event_sequence": [
                    event.context_payload(
                        target_attribute=self.target_attribute
                    )
                    for event in self.events
                ]
            }
        )
        provenance = (
            None
            if view != "provenance_aware"
            else {
                "event_sequence": [
                    event.provenance_payload()
                    for event in self.events
                ]
            }
        )
        return observation, context, provenance


@dataclass(frozen=True, slots=True)
class ExperimentAControlPlan:
    """The complete six-control protocol and its content binding."""

    battery_id: str
    battery_sha256: str
    protocol_version: str
    evaluation_split: str
    randomization_registration: ControlRandomizationRegistration
    stimuli: tuple[ExperimentAControlStimulus, ...]
    plan_sha256: str = ""

    def __post_init__(self) -> None:
        _require_text(self.battery_id, "battery_id")
        _validate_digest(self.battery_sha256, "battery_sha256")
        if self.protocol_version != CONTROL_PLAN_VERSION:
            raise ValueError(
                f"protocol_version must be {CONTROL_PLAN_VERSION!r}"
            )
        if self.evaluation_split != "test":
            raise ValueError("Experiment A controls are fixed to the test split")
        if not isinstance(
            self.randomization_registration,
            ControlRandomizationRegistration,
        ):
            raise TypeError(
                "randomization_registration must be a "
                "ControlRandomizationRegistration"
            )
        stimuli = tuple(self.stimuli)
        if tuple(stimulus.control_id for stimulus in stimuli) != CONTROL_IDS:
            raise ValueError(
                "control plan must contain the six controls in fixed order"
            )
        object.__setattr__(self, "stimuli", stimuli)
        expected = _digest(self._binding_payload())
        if self.plan_sha256 and self.plan_sha256 != expected:
            raise ValueError("plan_sha256 does not bind the control plan")
        object.__setattr__(self, "plan_sha256", expected)

    def _binding_payload(self) -> dict[str, Any]:
        return {
            "battery_id": self.battery_id,
            "battery_sha256": self.battery_sha256,
            "protocol_version": self.protocol_version,
            "evaluation_split": self.evaluation_split,
            "randomization_registration": (
                self.randomization_registration.to_dict()
            ),
            "stimuli": [stimulus.to_dict() for stimulus in self.stimuli],
        }

    def stimulus(self, control_id: str) -> ExperimentAControlStimulus:
        try:
            return next(
                stimulus
                for stimulus in self.stimuli
                if stimulus.control_id == control_id
            )
        except StopIteration as exc:
            raise KeyError(control_id) from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CONTROL_SCHEMA_VERSION,
            **self._binding_payload(),
            "plan_sha256": self.plan_sha256,
            "claim_status": CLAIM_STATUS,
            "interpretation": (
                "This plan makes the six protocol controls executable. "
                "Reference and baseline executions are diagnostics; provider "
                "responses remain non-claiming evidence until the full paper "
                "analysis is frozen and reviewed."
            ),
        }


def _case_digest(case: ExperimentAControlCase) -> str:
    return _digest(case.to_dict())


def _clone_option(
    *,
    option_id: str,
    features: tuple[float, float, float],
    label: str,
) -> Option:
    return Option(
        option_id=option_id,
        features=features,
        label=label,
        domain="travel",
    )


def _direct_correction_prior(
    *,
    target_attribute: int,
    target_direction: int,
    protocol: CorrectionProtocol,
) -> ProbabilityRows:
    adapter = ReferenceLogOddsCorrectionAdapter()
    state = adapter.initialize(
        pair_id="experiment-a-positive-direct-correction",
        truth_direction=target_direction,
        seed_condition="false",
        protocol=protocol,
    )
    before = adapter.measure(state, truth_direction=target_direction)
    return _with_sign_mass(
        _uniform_rows(),
        attribute=target_attribute,
        direction=target_direction,
        direction_mass=1.0 - before.wrong_profile_mass,
    )


def _build_random_events(
    registration: ControlRandomizationRegistration,
) -> tuple[ExperimentAControlEvent, ...]:
    events: list[ExperimentAControlEvent] = []
    for turn in range(registration.draw_count):
        negative = _clone_option(
            option_id=f"control-random-{turn}-negative",
            features=(-0.5, 0.0, 0.0),
            label=f"budget random alternative {turn + 1}",
        )
        positive = _clone_option(
            option_id=f"control-random-{turn}-positive",
            features=(0.5, 0.0, 0.0),
            label=f"premium random alternative {turn + 1}",
        )
        options = (negative, positive)
        event_id = f"control-random-turn-{turn}"
        index, draw_sha256 = registration.draw(
            event_id=event_id,
            option_count=len(options),
        )
        selected = options[index]
        events.append(
            ExperimentAControlEvent(
                event_id=event_id,
                turn_index=turn,
                response_kind="choice",
                surface_response=(
                    "The registered randomization device selected "
                    f"{selected.label}."
                ),
                scenario_id=f"control-random-scenario-{turn}",
                wording_template_id=f"control-random-wording-{turn}",
                options=options,
                ranking=tuple(option.option_id for option in options),
                selected_option_id=selected.option_id,
                response_source="registered_randomization_device",
                elicitation_provenance="registered_random_choice",
                randomization_registration_sha256=(
                    registration.registration_sha256
                ),
                randomization_draw_sha256=draw_sha256,
            )
        )
    return tuple(events)


def build_experiment_a_control_plan(
    battery: ExperimentAControlBattery | None = None,
) -> ExperimentAControlPlan:
    """Materialize the six proposal controls as deterministic test stimuli."""

    fixed = battery or build_experiment_a_control_battery()
    cases = {case.control_id: case for case in fixed.cases}
    if tuple(case.control_id for case in fixed.cases) != CONTROL_IDS:
        raise ValueError("control battery IDs differ from the execution contract")

    target_attribute = 0
    target_direction = 1
    uniform = _uniform_rows()
    registration = ControlRandomizationRegistration(
        registration_id="experiment-a-negative-random-choice-v1",
        algorithm="sha256-modulo-option-count-v1",
        seed_material=(
            "CAPE-Loop fixed public control randomization; "
            "independent of latent user state and presentation"
        ),
        draw_count=3,
    )

    volunteered = ExperimentAControlStimulus(
        control_id=CONTROL_IDS[0],
        case_sha256=_case_digest(cases[CONTROL_IDS[0]]),
        polarity="positive",
        signal_kind=cases[CONTROL_IDS[0]].signal_kind,
        target_attribute=target_attribute,
        target_direction=target_direction,
        prior_probabilities=uniform,
        events=(
            ExperimentAControlEvent(
                event_id="control-volunteered-turn-0",
                turn_index=0,
                response_kind="volunteered_statement",
                surface_response=(
                    "I generally prefer premium travel options."
                ),
                scenario_id="control-volunteered-scenario",
                wording_template_id="volunteered-user-statement-v1",
                response_source="user",
                elicitation_provenance="user_originated_unprompted",
            ),
        ),
    )

    repeated_events: list[ExperimentAControlEvent] = []
    for turn, nuisance in enumerate((0.0, 0.15, -0.15)):
        negative = _clone_option(
            option_id=f"control-balanced-{turn}-negative",
            features=(-0.5, nuisance, 0.0),
            label=f"budget cross-context option {turn + 1}",
        )
        positive = _clone_option(
            option_id=f"control-balanced-{turn}-positive",
            features=(0.5, nuisance, 0.0),
            label=f"premium cross-context option {turn + 1}",
        )
        repeated_events.append(
            ExperimentAControlEvent(
                event_id=f"control-balanced-turn-{turn}",
                turn_index=turn,
                response_kind="choice",
                surface_response=f"I choose {positive.label}.",
                scenario_id=f"control-balanced-scenario-{turn}",
                wording_template_id=f"balanced-cross-context-wording-{turn}",
                options=(negative, positive),
                ranking=(negative.option_id, positive.option_id),
                selected_option_id=positive.option_id,
                response_source="user",
                elicitation_provenance="balanced_exposure",
            )
        )
    repeated = ExperimentAControlStimulus(
        control_id=CONTROL_IDS[1],
        case_sha256=_case_digest(cases[CONTROL_IDS[1]]),
        polarity="positive",
        signal_kind=cases[CONTROL_IDS[1]].signal_kind,
        target_attribute=target_attribute,
        target_direction=target_direction,
        prior_probabilities=uniform,
        events=tuple(repeated_events),
    )

    correction_protocol = CorrectionProtocol()
    correction = ExperimentAControlStimulus(
        control_id=CONTROL_IDS[2],
        case_sha256=_case_digest(cases[CONTROL_IDS[2]]),
        polarity="positive",
        signal_kind=cases[CONTROL_IDS[2]].signal_kind,
        target_attribute=target_attribute,
        target_direction=target_direction,
        prior_probabilities=_direct_correction_prior(
            target_attribute=target_attribute,
            target_direction=target_direction,
            protocol=correction_protocol,
        ),
        events=(
            ExperimentAControlEvent(
                event_id="control-direct-correction-turn-0",
                turn_index=0,
                response_kind="direct_correction",
                surface_response=(
                    "Correction: I prefer premium travel options, not budget "
                    "travel options."
                ),
                scenario_id="control-direct-correction-scenario",
                wording_template_id="neutral-direct-correction-v1",
                response_source="user",
                elicitation_provenance="explicit_user_correction",
            ),
        ),
    )

    indifferent_negative = _clone_option(
        option_id="control-indifferent-negative",
        features=(-0.5, 0.0, 0.0),
        label="budget indifferent option",
    )
    indifferent_positive = _clone_option(
        option_id="control-indifferent-positive",
        features=(0.5, 0.0, 0.0),
        label="premium indifferent option",
    )
    indifferent = ExperimentAControlStimulus(
        control_id=CONTROL_IDS[3],
        case_sha256=_case_digest(cases[CONTROL_IDS[3]]),
        polarity="negative",
        signal_kind=cases[CONTROL_IDS[3]].signal_kind,
        target_attribute=target_attribute,
        target_direction=target_direction,
        prior_probabilities=uniform,
        events=(
            ExperimentAControlEvent(
                event_id="control-indifferent-turn-0",
                turn_index=0,
                response_kind="indifference",
                surface_response=(
                    "I am indifferent between these two travel options."
                ),
                scenario_id="control-indifferent-scenario",
                wording_template_id="explicit-indifference-v1",
                options=(indifferent_negative, indifferent_positive),
                ranking=(
                    indifferent_negative.option_id,
                    indifferent_positive.option_id,
                ),
                response_source="user",
                elicitation_provenance="balanced_exposure",
            ),
        ),
    )

    randomized = ExperimentAControlStimulus(
        control_id=CONTROL_IDS[4],
        case_sha256=_case_digest(cases[CONTROL_IDS[4]]),
        polarity="negative",
        signal_kind=cases[CONTROL_IDS[4]].signal_kind,
        target_attribute=target_attribute,
        target_direction=target_direction,
        prior_probabilities=uniform,
        events=_build_random_events(registration),
    )

    same_target_first = _clone_option(
        option_id="control-nondistinguishing-first",
        features=(0.5, -0.5, 0.0),
        label="premium central option",
    )
    same_target_second = _clone_option(
        option_id="control-nondistinguishing-second",
        features=(0.5, 0.5, 0.0),
        label="premium comfort option",
    )
    nondistinguishing = ExperimentAControlStimulus(
        control_id=CONTROL_IDS[5],
        case_sha256=_case_digest(cases[CONTROL_IDS[5]]),
        polarity="negative",
        signal_kind=cases[CONTROL_IDS[5]].signal_kind,
        target_attribute=target_attribute,
        target_direction=target_direction,
        prior_probabilities=uniform,
        events=(
            ExperimentAControlEvent(
                event_id="control-nondistinguishing-turn-0",
                turn_index=0,
                response_kind="choice",
                surface_response=f"I choose {same_target_second.label}.",
                scenario_id="control-nondistinguishing-scenario",
                wording_template_id="target-invariant-choice-v1",
                options=(same_target_first, same_target_second),
                ranking=(
                    same_target_first.option_id,
                    same_target_second.option_id,
                ),
                selected_option_id=same_target_second.option_id,
                response_source="user",
                elicitation_provenance="balanced_non_target_contrast",
            ),
        ),
    )

    return ExperimentAControlPlan(
        battery_id=fixed.battery_id,
        battery_sha256=fixed.battery_sha256,
        protocol_version=CONTROL_PLAN_VERSION,
        evaluation_split="test",
        randomization_registration=registration,
        stimuli=(
            volunteered,
            repeated,
            correction,
            indifferent,
            randomized,
            nondistinguishing,
        ),
    )


@dataclass(frozen=True, slots=True)
class ControlExecutionOutcome:
    """One control outcome with evidence-source and content bindings."""

    execution_id: str
    control_id: str
    plan_sha256: str
    stimulus_sha256: str
    executor_id: str
    execution_mode: str
    source_descriptor: str
    prior_probabilities: ProbabilityRows
    posterior_probabilities: ProbabilityRows
    target_attribute: int
    target_direction: int
    polarity: str
    directional_mass_before: float
    directional_mass_after: float
    directional_mass_delta: float
    criterion_met: bool
    request_id: str | None = None
    prompt_sha256: str | None = None
    model_id: str | None = None
    response_sha256: str | None = None
    reference_binding: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        for name in (
            "execution_id",
            "control_id",
            "executor_id",
            "source_descriptor",
        ):
            _require_text(getattr(self, name), name)
        _validate_digest(self.plan_sha256, "plan_sha256")
        _validate_digest(self.stimulus_sha256, "stimulus_sha256")
        if self.execution_mode not in _EXECUTION_MODES:
            raise ValueError(
                f"execution_mode must be one of {sorted(_EXECUTION_MODES)}"
            )
        if self.polarity not in _POLARITIES:
            raise ValueError(f"polarity must be one of {sorted(_POLARITIES)}")
        before = _probability_rows(
            self.prior_probabilities,
            name="prior_probabilities",
        )
        after = _probability_rows(
            self.posterior_probabilities,
            name="posterior_probabilities",
        )
        object.__setattr__(self, "prior_probabilities", before)
        object.__setattr__(self, "posterior_probabilities", after)
        expected_before = _sign_mass(
            before,
            self.target_attribute,
            self.target_direction,
        )
        expected_after = _sign_mass(
            after,
            self.target_attribute,
            self.target_direction,
        )
        if not math.isclose(
            self.directional_mass_before,
            expected_before,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("directional_mass_before differs from prior")
        if not math.isclose(
            self.directional_mass_after,
            expected_after,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("directional_mass_after differs from posterior")
        if not math.isclose(
            self.directional_mass_delta,
            expected_after - expected_before,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("directional_mass_delta is inconsistent")
        provider_fields = (
            self.request_id,
            self.prompt_sha256,
            self.model_id,
            self.response_sha256,
        )
        is_provider = self.execution_mode.startswith("provider_")
        if is_provider and any(field is None for field in provider_fields):
            raise ValueError("provider outcomes require all response bindings")
        if not is_provider and any(field is not None for field in provider_fields):
            raise ValueError(
                "diagnostic outcomes cannot carry provider response bindings"
            )
        for name in ("prompt_sha256", "response_sha256"):
            value = getattr(self, name)
            if value is not None:
                _validate_digest(value, name)

    @property
    def evidence_class(self) -> str:
        return _EVIDENCE_CLASSES[self.execution_mode]

    @property
    def is_live_evidence(self) -> bool:
        return self.execution_mode == "provider_live"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CONTROL_SCHEMA_VERSION,
            "execution_id": self.execution_id,
            "control_id": self.control_id,
            "plan_sha256": self.plan_sha256,
            "stimulus_sha256": self.stimulus_sha256,
            "executor_id": self.executor_id,
            "execution_mode": self.execution_mode,
            "evidence_class": self.evidence_class,
            "is_live_evidence": self.is_live_evidence,
            "source_descriptor": self.source_descriptor,
            "prior_probabilities": _rows_to_dict(self.prior_probabilities),
            "posterior_probabilities": _rows_to_dict(
                self.posterior_probabilities
            ),
            "target_attribute": self.target_attribute,
            "target_direction": self.target_direction,
            "polarity": self.polarity,
            "directional_mass_before": self.directional_mass_before,
            "directional_mass_after": self.directional_mass_after,
            "directional_mass_delta": self.directional_mass_delta,
            "criterion_met": self.criterion_met,
            "request_id": self.request_id,
            "prompt_sha256": self.prompt_sha256,
            "model_id": self.model_id,
            "response_sha256": self.response_sha256,
            "reference_binding": (
                None
                if self.reference_binding is None
                else dict(self.reference_binding)
            ),
            "claim_status": CLAIM_STATUS,
        }


@dataclass(frozen=True, slots=True)
class ControlExecutionReport:
    """Exact-coverage report for one control executor."""

    report_id: str
    plan_sha256: str
    battery_sha256: str
    executor_id: str
    execution_mode: str
    source_descriptor: str
    outcomes: tuple[ControlExecutionOutcome, ...]
    required_control_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("report_id", "executor_id", "source_descriptor"):
            _require_text(getattr(self, name), name)
        _validate_digest(self.plan_sha256, "plan_sha256")
        _validate_digest(self.battery_sha256, "battery_sha256")
        if self.execution_mode not in _EXECUTION_MODES:
            raise ValueError("unknown control execution mode")
        outcomes = tuple(self.outcomes)
        required = tuple(self.required_control_ids)
        observed = tuple(outcome.control_id for outcome in outcomes)
        if len(set(observed)) != len(observed):
            raise ValueError("control report contains duplicate outcomes")
        if set(observed) != set(required):
            raise ValueError(
                "control report coverage mismatch; "
                f"missing={sorted(set(required) - set(observed))}, "
                f"unexpected={sorted(set(observed) - set(required))}"
            )
        if any(
            outcome.plan_sha256 != self.plan_sha256
            or outcome.executor_id != self.executor_id
            or outcome.execution_mode != self.execution_mode
            for outcome in outcomes
        ):
            raise ValueError("control outcomes disagree with report bindings")
        object.__setattr__(self, "outcomes", outcomes)
        object.__setattr__(self, "required_control_ids", required)

    @property
    def criterion_pass_count(self) -> int:
        return sum(outcome.criterion_met for outcome in self.outcomes)

    @property
    def live_evidence_count(self) -> int:
        return sum(outcome.is_live_evidence for outcome in self.outcomes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CONTROL_SCHEMA_VERSION,
            "report_id": self.report_id,
            "plan_sha256": self.plan_sha256,
            "battery_sha256": self.battery_sha256,
            "executor_id": self.executor_id,
            "execution_mode": self.execution_mode,
            "evidence_class": _EVIDENCE_CLASSES[self.execution_mode],
            "source_descriptor": self.source_descriptor,
            "coverage": {
                "required_control_ids": list(self.required_control_ids),
                "observed_control_ids": [
                    outcome.control_id for outcome in self.outcomes
                ],
                "missing_control_ids": [],
                "unexpected_control_ids": [],
                "complete": True,
            },
            "criterion_pass_count": self.criterion_pass_count,
            "criterion_fail_count": (
                len(self.outcomes) - self.criterion_pass_count
            ),
            "live_evidence_count": self.live_evidence_count,
            "outcomes": [outcome.to_dict() for outcome in self.outcomes],
            "claim_status": CLAIM_STATUS,
            "interpretation": (
                "Coverage and criteria describe this control executor only. "
                "They do not promote a software diagnostic or provider "
                "response to a paper claim."
            ),
        }


def _criterion(
    stimulus: ExperimentAControlStimulus,
    delta: float,
) -> bool:
    if stimulus.polarity == "positive":
        return delta >= stimulus.minimum_directional_mass_delta
    return (
        abs(delta)
        <= stimulus.maximum_negative_control_abs_delta
    )


def _outcome(
    *,
    plan: ExperimentAControlPlan,
    stimulus: ExperimentAControlStimulus,
    executor_id: str,
    execution_mode: str,
    source_descriptor: str,
    posterior: ProbabilityRows,
    request: LLMRequest | None = None,
    response: LLMResponse | None = None,
    reference_binding: Mapping[str, Any] | None = None,
) -> ControlExecutionOutcome:
    before = _sign_mass(
        stimulus.prior_probabilities,
        stimulus.target_attribute,
        stimulus.target_direction,
    )
    after = _sign_mass(
        posterior,
        stimulus.target_attribute,
        stimulus.target_direction,
    )
    delta = after - before
    response_sha256 = (
        None if response is None else _digest(response.to_dict())
    )
    execution_id = (
        f"{executor_id}:{stimulus.control_id}:"
        f"{_digest({'plan': plan.plan_sha256, 'stimulus': stimulus.stimulus_sha256, 'mode': execution_mode, 'request': None if request is None else request.prompt_sha256, 'response': response_sha256})}"
    )
    return ControlExecutionOutcome(
        execution_id=execution_id,
        control_id=stimulus.control_id,
        plan_sha256=plan.plan_sha256,
        stimulus_sha256=stimulus.stimulus_sha256,
        executor_id=executor_id,
        execution_mode=execution_mode,
        source_descriptor=source_descriptor,
        prior_probabilities=stimulus.prior_probabilities,
        posterior_probabilities=posterior,
        target_attribute=stimulus.target_attribute,
        target_direction=stimulus.target_direction,
        polarity=stimulus.polarity,
        directional_mass_before=before,
        directional_mass_after=after,
        directional_mass_delta=delta,
        criterion_met=_criterion(stimulus, delta),
        request_id=None if request is None else request.request_id,
        prompt_sha256=None if request is None else request.prompt_sha256,
        model_id=None if response is None else response.model_id,
        response_sha256=response_sha256,
        reference_binding=reference_binding,
    )


def _reference_posterior(
    stimulus: ExperimentAControlStimulus,
) -> tuple[ProbabilityRows, Mapping[str, Any]]:
    rows = stimulus.prior_probabilities
    if stimulus.control_id == "positive-volunteered-preference":
        return (
            _bayes_sign_update(
                rows,
                attribute=stimulus.target_attribute,
                direction=stimulus.target_direction,
                reliability=0.90,
            ),
            {
                "reference_family": "declared_sign_likelihood",
                "per_event_reliability": 0.90,
                "event_count": 1,
            },
        )
    if stimulus.control_id == "positive-repeated-balanced-cross-context":
        for _ in stimulus.events:
            rows = _bayes_sign_update(
                rows,
                attribute=stimulus.target_attribute,
                direction=stimulus.target_direction,
                reliability=0.75,
            )
        return (
            rows,
            {
                "reference_family": "sequential_declared_sign_likelihood",
                "per_event_reliability": 0.75,
                "event_count": len(stimulus.events),
            },
        )
    if stimulus.control_id == "positive-direct-correction":
        protocol = CorrectionProtocol()
        adapter = ReferenceLogOddsCorrectionAdapter()
        state = adapter.initialize(
            pair_id="experiment-a-positive-direct-correction",
            truth_direction=stimulus.target_direction,
            seed_condition="false",
            protocol=protocol,
        )
        before = adapter.measure(
            state,
            truth_direction=stimulus.target_direction,
        )
        state = adapter.apply_explicit_correction(
            state,
            truth_direction=stimulus.target_direction,
            evidence_strength=protocol.explicit_correction_evidence,
            correction_text=stimulus.events[0].surface_response,
            event_key=stimulus.events[0].event_id,
        )
        after = adapter.measure(
            state,
            truth_direction=stimulus.target_direction,
        )
        expected_prior_mass = 1.0 - before.wrong_profile_mass
        actual_prior_mass = _sign_mass(
            rows,
            stimulus.target_attribute,
            stimulus.target_direction,
        )
        if not math.isclose(
            actual_prior_mass,
            expected_prior_mass,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "direct-correction prior is not bound to its reference adapter"
            )
        posterior = _with_sign_mass(
            rows,
            attribute=stimulus.target_attribute,
            direction=stimulus.target_direction,
            direction_mass=1.0 - after.wrong_profile_mass,
        )
        return (
            posterior,
            {
                "reference_family": "correction_debt_reference_adapter",
                "adapter_id": adapter.adapter_id,
                "protocol_version": protocol.protocol_version,
                "protocol_sha256": protocol.protocol_sha256,
                "state_before_sha256": before.state_sha256,
                "state_after_sha256": after.state_sha256,
                "wrong_profile_mass_before": before.wrong_profile_mass,
                "wrong_profile_mass_after": after.wrong_profile_mass,
            },
        )
    # The three negative controls are explicitly non-evidential for the target
    # dimension.  Other dimensions are also left untouched by this narrow
    # diagnostic reference.
    return (
        rows,
        {
            "reference_family": "target_no_update",
            "reason": {
                "negative-indifferent-response": "explicit_indifference",
                "negative-random-choice": "registered_preference_independent_draw",
                "negative-nondistinguishing-response": (
                    "displayed_options_share_target_feature"
                ),
            }[stimulus.control_id],
        },
    )


def _report(
    *,
    plan: ExperimentAControlPlan,
    executor_id: str,
    execution_mode: str,
    source_descriptor: str,
    outcomes: Sequence[ControlExecutionOutcome],
    required_control_ids: Sequence[str],
) -> ControlExecutionReport:
    material = tuple(outcomes)
    report_id = (
        f"{executor_id}:"
        f"{_digest([outcome.to_dict() for outcome in material])}"
    )
    return ControlExecutionReport(
        report_id=report_id,
        plan_sha256=plan.plan_sha256,
        battery_sha256=plan.battery_sha256,
        executor_id=executor_id,
        execution_mode=execution_mode,
        source_descriptor=source_descriptor,
        outcomes=material,
        required_control_ids=tuple(required_control_ids),
    )


def run_diagnostic_control_executions(
    plan: ExperimentAControlPlan | None = None,
) -> tuple[ControlExecutionReport, ControlExecutionReport]:
    """Run transparent reference and no-update baseline over all six controls."""

    material = plan or build_experiment_a_control_plan()
    reference_id = "experiment_a_control_reference_v1"
    reference_mode = "deterministic_reference"
    reference_outcomes: list[ControlExecutionOutcome] = []
    for stimulus in material.stimuli:
        posterior, binding = _reference_posterior(stimulus)
        reference_outcomes.append(
            _outcome(
                plan=material,
                stimulus=stimulus,
                executor_id=reference_id,
                execution_mode=reference_mode,
                source_descriptor=(
                    "inspectable project-authored diagnostic reference; "
                    "not external or empirical evidence"
                ),
                posterior=posterior,
                reference_binding=binding,
            )
        )
    reference = _report(
        plan=material,
        executor_id=reference_id,
        execution_mode=reference_mode,
        source_descriptor=(
            "inspectable project-authored diagnostic reference; "
            "not external or empirical evidence"
        ),
        outcomes=reference_outcomes,
        required_control_ids=CONTROL_IDS,
    )

    baseline_id = "experiment_a_control_no_update_baseline_v1"
    baseline_mode = "deterministic_no_update_baseline"
    baseline_outcomes = tuple(
        _outcome(
            plan=material,
            stimulus=stimulus,
            executor_id=baseline_id,
            execution_mode=baseline_mode,
            source_descriptor=(
                "project-authored no-update diagnostic baseline; "
                "not external or empirical evidence"
            ),
            posterior=stimulus.prior_probabilities,
            reference_binding={
                "reference_family": "identity_no_update",
                "expected_role": (
                    "nonresponsive comparator for positive controls and "
                    "null-behavior comparator for negative controls"
                ),
            },
        )
        for stimulus in material.stimuli
    )
    baseline = _report(
        plan=material,
        executor_id=baseline_id,
        execution_mode=baseline_mode,
        source_descriptor=(
            "project-authored no-update diagnostic baseline; "
            "not external or empirical evidence"
        ),
        outcomes=baseline_outcomes,
        required_control_ids=CONTROL_IDS,
    )
    return reference, baseline


@dataclass(frozen=True, slots=True)
class ControlLLMRequest:
    """An LLM request plus the control-plan binding withheld from its prompt."""

    control_id: str
    plan_sha256: str
    battery_sha256: str
    stimulus_sha256: str
    llm_request: LLMRequest
    binding_sha256: str = ""

    def __post_init__(self) -> None:
        _require_text(self.control_id, "control_id")
        for name in ("plan_sha256", "battery_sha256", "stimulus_sha256"):
            _validate_digest(getattr(self, name), name)
        if not isinstance(self.llm_request, LLMRequest):
            raise TypeError("llm_request must be an LLMRequest")
        expected = _digest(self._binding_payload())
        if self.binding_sha256 and self.binding_sha256 != expected:
            raise ValueError(
                "binding_sha256 does not bind the control LLM request"
            )
        object.__setattr__(self, "binding_sha256", expected)

    def _binding_payload(self) -> dict[str, Any]:
        return {
            "control_id": self.control_id,
            "plan_sha256": self.plan_sha256,
            "battery_sha256": self.battery_sha256,
            "stimulus_sha256": self.stimulus_sha256,
            "llm_request": self.llm_request.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CONTROL_SCHEMA_VERSION,
            **self._binding_payload(),
            "binding_sha256": self.binding_sha256,
        }

    @classmethod
    def parse(cls, raw: Mapping[str, Any]) -> "ControlLLMRequest":
        allowed = {
            "schema_version",
            "control_id",
            "plan_sha256",
            "battery_sha256",
            "stimulus_sha256",
            "llm_request",
            "binding_sha256",
        }
        if set(raw) != allowed:
            raise ValueError("unknown or missing control LLM request fields")
        if raw["schema_version"] != CONTROL_SCHEMA_VERSION:
            raise ValueError("control LLM request schema_version must be 1")
        llm_raw = raw["llm_request"]
        if not isinstance(llm_raw, Mapping):
            raise ValueError("llm_request must be an object")
        llm_request = _parse_llm_request(llm_raw)
        return cls(
            control_id=raw["control_id"],
            plan_sha256=raw["plan_sha256"],
            battery_sha256=raw["battery_sha256"],
            stimulus_sha256=raw["stimulus_sha256"],
            llm_request=llm_request,
            binding_sha256=raw["binding_sha256"],
        )


def _parse_llm_request(raw: Mapping[str, Any]) -> LLMRequest:
    allowed = {
        "schema_version",
        "request_id",
        "updater_id",
        "view",
        "system_instruction",
        "payload",
        "prompt_sha256",
    }
    if set(raw) != allowed:
        raise ValueError("LLM request fields do not match schema version 1")
    if raw["schema_version"] != 1:
        raise ValueError("LLM request schema_version must be 1")
    for name in ("request_id", "updater_id", "system_instruction"):
        _require_text(raw[name], name)
    if raw["view"] not in VIEWS:
        raise ValueError(f"unknown LLM view: {raw['view']}")
    if not isinstance(raw["payload"], Mapping):
        raise ValueError("LLM request payload must be an object")
    expected = sha256(
        (
            raw["system_instruction"]
            + "\n"
            + _canonical(raw["payload"])
        ).encode("utf-8")
    ).hexdigest()
    if raw["prompt_sha256"] != expected:
        raise ValueError("prompt_sha256 does not bind the LLM request")
    return LLMRequest(
        request_id=raw["request_id"],
        updater_id=raw["updater_id"],
        view=raw["view"],
        payload=dict(raw["payload"]),
        system_instruction=raw["system_instruction"],
        prompt_sha256=expected,
    )


@dataclass(frozen=True, slots=True)
class ControlLLMExchange:
    """Provider request packet with explicit semantic omissions and coverage."""

    exchange_id: str
    plan_sha256: str
    battery_sha256: str
    updater_id: str
    view: str
    requests: tuple[ControlLLMRequest, ...]
    omitted_controls: tuple[tuple[str, str], ...]
    exchange_sha256: str = ""

    def __post_init__(self) -> None:
        _require_text(self.exchange_id, "exchange_id")
        _require_text(self.updater_id, "updater_id")
        _validate_digest(self.plan_sha256, "plan_sha256")
        _validate_digest(self.battery_sha256, "battery_sha256")
        if self.view not in VIEWS:
            raise ValueError(f"unknown LLM view: {self.view}")
        requests = tuple(self.requests)
        omitted = tuple(self.omitted_controls)
        request_controls = tuple(request.control_id for request in requests)
        omitted_controls = tuple(control_id for control_id, _ in omitted)
        if len(set(request_controls + omitted_controls)) != len(
            request_controls + omitted_controls
        ):
            raise ValueError("control exchange has duplicate controls")
        if set(request_controls + omitted_controls) != set(CONTROL_IDS):
            raise ValueError("control exchange must classify every control")
        if len(
            {
                request.llm_request.request_id
                for request in requests
            }
        ) != len(requests):
            raise ValueError("control exchange request IDs must be unique")
        if any(
            request.plan_sha256 != self.plan_sha256
            or request.battery_sha256 != self.battery_sha256
            or request.llm_request.updater_id != self.updater_id
            or request.llm_request.view != self.view
            for request in requests
        ):
            raise ValueError("request bindings disagree with the exchange")
        object.__setattr__(self, "requests", requests)
        object.__setattr__(self, "omitted_controls", omitted)
        expected = _digest(self._binding_payload())
        if self.exchange_sha256 and self.exchange_sha256 != expected:
            raise ValueError("exchange_sha256 does not bind the request packet")
        object.__setattr__(self, "exchange_sha256", expected)

    def _binding_payload(self) -> dict[str, Any]:
        return {
            "exchange_id": self.exchange_id,
            "plan_sha256": self.plan_sha256,
            "battery_sha256": self.battery_sha256,
            "updater_id": self.updater_id,
            "view": self.view,
            "requests": [request.to_dict() for request in self.requests],
            "omitted_controls": [
                {"control_id": control_id, "reason": reason}
                for control_id, reason in self.omitted_controls
            ],
        }

    @property
    def requested_control_ids(self) -> tuple[str, ...]:
        return tuple(request.control_id for request in self.requests)

    @property
    def llm_requests(self) -> tuple[LLMRequest, ...]:
        return tuple(request.llm_request for request in self.requests)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CONTROL_SCHEMA_VERSION,
            "exchange_version": CONTROL_EXCHANGE_VERSION,
            **self._binding_payload(),
            "exchange_sha256": self.exchange_sha256,
            "coverage": {
                "all_control_ids": list(CONTROL_IDS),
                "requested_control_ids": list(self.requested_control_ids),
                "omitted_control_ids": [
                    control_id
                    for control_id, _ in self.omitted_controls
                ],
                "complete_for_semantically_valid_controls": True,
                "complete_for_all_six_controls": not self.omitted_controls,
            },
            "claim_status": CLAIM_STATUS,
        }


def build_control_llm_exchange(
    plan: ExperimentAControlPlan | None = None,
    *,
    updater_id: str = "llm_control_provenance_aware",
    view: str = "provenance_aware",
) -> ControlLLMExchange:
    """Build requests only where ``view`` carries the required control facts."""

    material = plan or build_experiment_a_control_plan()
    _require_text(updater_id, "updater_id")
    if view not in VIEWS:
        raise ValueError(f"unknown LLM view: {view}")
    requests: list[ControlLLMRequest] = []
    omitted: list[tuple[str, str]] = []
    for stimulus in material.stimuli:
        if view not in CONTROL_VALID_VIEWS[stimulus.control_id]:
            omitted.append(
                (
                    stimulus.control_id,
                    (
                        f"{view} does not expose the context/provenance needed "
                        "to preserve this control's estimand"
                    ),
                )
            )
            continue
        observation, context, provenance = stimulus.model_payload(view)
        draft = LLMRequest.build(
            request_id="content-addressed-control-request",
            updater_id=updater_id,
            view=view,
            prior=_rows_to_dict(stimulus.prior_probabilities),
            observation=observation,
            context=context,
            provenance=provenance,
        )
        request = LLMRequest(
            request_id=f"{updater_id}:{draft.prompt_sha256}",
            updater_id=updater_id,
            view=view,
            payload=draft.payload,
            system_instruction=draft.system_instruction,
            prompt_sha256=draft.prompt_sha256,
        )
        requests.append(
            ControlLLMRequest(
                control_id=stimulus.control_id,
                plan_sha256=material.plan_sha256,
                battery_sha256=material.battery_sha256,
                stimulus_sha256=stimulus.stimulus_sha256,
                llm_request=request,
            )
        )
    return ControlLLMExchange(
        exchange_id=f"{CONTROL_EXCHANGE_VERSION}:{updater_id}:{view}",
        plan_sha256=material.plan_sha256,
        battery_sha256=material.battery_sha256,
        updater_id=updater_id,
        view=view,
        requests=tuple(requests),
        omitted_controls=tuple(omitted),
    )


def validate_control_response_coverage(
    exchange: ControlLLMExchange,
    responses: Iterable[LLMResponse],
) -> tuple[LLMResponse, ...]:
    """Require exact IDs and prompt hashes before any control is scored."""

    material = tuple(responses)
    if len({response.request_id for response in material}) != len(material):
        raise ValueError("control responses contain duplicate request IDs")
    by_id = {response.request_id: response for response in material}
    expected = {
        request.llm_request.request_id: request
        for request in exchange.requests
    }
    missing = sorted(set(expected) - set(by_id))
    unexpected = sorted(set(by_id) - set(expected))
    if missing or unexpected:
        raise ValueError(
            "control response coverage mismatch; "
            f"missing={missing}, unexpected={unexpected}"
        )
    ordered: list[LLMResponse] = []
    for wrapped in exchange.requests:
        response = by_id[wrapped.llm_request.request_id]
        if response.prompt_sha256 != wrapped.llm_request.prompt_sha256:
            raise ValueError(
                f"prompt hash mismatch for {response.request_id}"
            )
        ordered.append(response)
    return tuple(ordered)


def execute_control_llm_exchange(
    plan: ExperimentAControlPlan,
    exchange: ControlLLMExchange,
    provider: CompletionProvider,
    *,
    execution_mode: str,
    source_descriptor: str,
) -> ControlExecutionReport:
    """Execute and score a replay or explicitly authorized live provider.

    This function contains no live-authorization switch and does not load API
    keys.  A caller must pass an already configured provider.  Live-provider
    implementations retain their own transport audit and budgets.
    """

    if execution_mode not in {"provider_replay", "provider_live"}:
        raise ValueError(
            "provider execution_mode must be provider_replay or provider_live"
        )
    if (
        isinstance(provider, ReplayProvider)
        and execution_mode != "provider_replay"
    ):
        raise ValueError(
            "ReplayProvider executions must be labeled provider_replay, "
            "not live evidence"
        )
    _require_text(source_descriptor, "source_descriptor")
    if exchange.plan_sha256 != plan.plan_sha256:
        raise ValueError("control exchange is bound to a different plan")
    responses = tuple(
        provider.complete(wrapped.llm_request)
        for wrapped in exchange.requests
    )
    ordered = validate_control_response_coverage(exchange, responses)
    outcomes: list[ControlExecutionOutcome] = []
    for wrapped, response in zip(exchange.requests, ordered):
        stimulus = plan.stimulus(wrapped.control_id)
        if wrapped.stimulus_sha256 != stimulus.stimulus_sha256:
            raise ValueError(
                f"stimulus binding mismatch for {wrapped.control_id}"
            )
        outcomes.append(
            _outcome(
                plan=plan,
                stimulus=stimulus,
                executor_id=exchange.updater_id,
                execution_mode=execution_mode,
                source_descriptor=source_descriptor,
                posterior=_rows_from_response(response),
                request=wrapped.llm_request,
                response=response,
            )
        )
    return _report(
        plan=plan,
        executor_id=exchange.updater_id,
        execution_mode=execution_mode,
        source_descriptor=source_descriptor,
        outcomes=outcomes,
        required_control_ids=exchange.requested_control_ids,
    )


def write_control_plan(
    path: str | Path,
    plan: ExperimentAControlPlan,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        _canonical(plan.to_dict()) + "\n",
        encoding="utf-8",
    )
    return destination


def write_control_request_bindings(
    path: str | Path,
    exchange: ControlLLMExchange,
) -> int:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for request in exchange.requests:
            handle.write(_canonical(request.to_dict()) + "\n")
    return len(exchange.requests)


def write_control_provider_requests(
    path: str | Path,
    exchange: ControlLLMExchange,
) -> int:
    """Write generic request JSONL consumable by existing provider commands."""

    return write_requests(path, exchange.llm_requests)


def read_control_request_bindings(
    path: str | Path | bytes,
) -> tuple[ControlLLMRequest, ...]:
    """Read binding JSONL from a path or one immutable byte snapshot."""

    if isinstance(path, bytes):
        source_label = "<control-binding-bytes>"
        try:
            lines = path.decode("utf-8").splitlines()
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"{source_label}: input must be valid UTF-8"
            ) from exc
    else:
        source = Path(path)
        source_label = str(source)
        lines = source.read_text(encoding="utf-8").splitlines()
    material: list[ControlLLMRequest] = []
    seen: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            if not isinstance(raw, Mapping):
                raise ValueError("control request line must be an object")
            request = ControlLLMRequest.parse(raw)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{source_label}:{line_number}: {exc}"
            ) from exc
        request_id = request.llm_request.request_id
        if request_id in seen:
            raise ValueError(f"duplicate control request_id: {request_id}")
        seen.add(request_id)
        material.append(request)
    return tuple(material)


def write_control_report(
    path: str | Path,
    report: ControlExecutionReport,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        _canonical(report.to_dict()) + "\n",
        encoding="utf-8",
    )
    return destination
