"""Provider-neutral, replayable JSONL exchange for LLM profile writers."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol
import json
import math

from .calibration import TemperatureCalibration

VALUES = ("-2", "-1", "+1", "+2")
VIEWS = frozenset({"response_only", "full_context", "provenance_aware"})
ATTRIBUTES = ("attribute_1", "attribute_2", "attribute_3")

BASE_INSTRUCTION = """\
Return JSON only. Infer a persistent preference profile from the supplied prior
and interaction. For each attribute, return probabilities for -2, -1, +1, +2.
Do not use information that is absent from the request.
"""

PROVENANCE_INSTRUCTION = """\
Treat the response as evidence conditional on the options and framing that
produced it. Do not infer a general preference merely because the user accepted
an agent-selected, defaulted, suggested, or restricted option.
"""


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass(frozen=True, slots=True)
class LLMRequest:
    request_id: str
    updater_id: str
    view: str
    payload: Mapping[str, Any]
    system_instruction: str
    prompt_sha256: str

    @classmethod
    def build(
        cls,
        *,
        request_id: str,
        updater_id: str,
        view: str,
        prior: Mapping[str, Any],
        observation: Mapping[str, Any],
        context: Mapping[str, Any] | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> "LLMRequest":
        if view not in VIEWS:
            raise ValueError(f"unknown LLM view: {view}")
        if not request_id or not updater_id:
            raise ValueError("request_id and updater_id are required")
        payload: dict[str, Any] = {
            "prior": prior,
            "observation": observation,
        }
        if view in {"full_context", "provenance_aware"}:
            if context is None:
                raise ValueError(f"{view} requests require context")
            payload["context"] = context
        if view == "provenance_aware":
            if provenance is None:
                raise ValueError("provenance-aware requests require provenance")
            payload["provenance"] = provenance
        instruction = BASE_INSTRUCTION
        if view == "provenance_aware":
            instruction += "\n" + PROVENANCE_INSTRUCTION
        digest = sha256(
            (instruction + "\n" + _canonical(payload)).encode("utf-8")
        ).hexdigest()
        return cls(
            request_id=request_id,
            updater_id=updater_id,
            view=view,
            payload=payload,
            system_instruction=instruction,
            prompt_sha256=digest,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "request_id": self.request_id,
            "updater_id": self.updater_id,
            "view": self.view,
            "system_instruction": self.system_instruction,
            "payload": self.payload,
            "prompt_sha256": self.prompt_sha256,
        }


@dataclass(frozen=True, slots=True)
class LLMResponse:
    request_id: str
    prompt_sha256: str
    model_id: str
    beliefs: Mapping[str, Mapping[str, float]]
    raw_response_sha256: str | None = None

    @classmethod
    def parse(cls, raw: Mapping[str, Any]) -> "LLMResponse":
        allowed = {
            "schema_version",
            "request_id",
            "prompt_sha256",
            "model_id",
            "beliefs",
            "raw_response_sha256",
        }
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"unknown response fields: {sorted(unknown)}")
        if raw.get("schema_version") != 1:
            raise ValueError("LLM response schema_version must be 1")
        request_id = raw.get("request_id")
        prompt_sha256 = raw.get("prompt_sha256")
        model_id = raw.get("model_id")
        beliefs = raw.get("beliefs")
        if not isinstance(request_id, str) or not request_id.strip():
            raise ValueError("request_id is required")
        if (
            not isinstance(prompt_sha256, str)
            or len(prompt_sha256) != 64
            or any(character not in "0123456789abcdef" for character in prompt_sha256)
        ):
            raise ValueError("prompt_sha256 must be a lowercase SHA-256 digest")
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id is required")
        if not isinstance(beliefs, Mapping) or not beliefs:
            raise ValueError("beliefs must be a non-empty object")
        if set(beliefs) != set(ATTRIBUTES):
            raise ValueError(
                "beliefs must contain exactly " + ", ".join(ATTRIBUTES)
            )
        validated: dict[str, dict[str, float]] = {}
        for attribute, vector in beliefs.items():
            if not isinstance(attribute, str) or not isinstance(vector, Mapping):
                raise ValueError("each belief must be a probability object")
            if set(vector) != set(VALUES):
                raise ValueError(
                    f"{attribute} must contain exactly {', '.join(VALUES)}"
                )
            if any(
                isinstance(vector[value], bool)
                or not isinstance(vector[value], (int, float))
                for value in VALUES
            ):
                raise ValueError(f"{attribute} probabilities must be numeric")
            parsed = {value: float(vector[value]) for value in VALUES}
            if any(
                not math.isfinite(p) or p < 0 or p > 1
                for p in parsed.values()
            ):
                raise ValueError(f"{attribute} contains probability outside [0, 1]")
            if abs(sum(parsed.values()) - 1.0) > 1e-6:
                raise ValueError(f"{attribute} probabilities do not sum to one")
            validated[attribute] = parsed
        raw_response_sha256 = raw.get("raw_response_sha256")
        if (
            raw_response_sha256 is not None
            and (
                not isinstance(raw_response_sha256, str)
                or len(raw_response_sha256) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in raw_response_sha256
                )
            )
        ):
            raise ValueError(
                "raw_response_sha256 must be null or a lowercase SHA-256 digest"
            )
        return cls(
            request_id=request_id,
            prompt_sha256=prompt_sha256,
            model_id=model_id,
            beliefs=validated,
            raw_response_sha256=raw_response_sha256,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "request_id": self.request_id,
            "prompt_sha256": self.prompt_sha256,
            "model_id": self.model_id,
            "beliefs": self.beliefs,
            "raw_response_sha256": self.raw_response_sha256,
        }


class CompletionProvider(Protocol):
    """Minimal provider boundary shared by replay and live execution."""

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Return one response bound to the supplied content-addressed request."""


class TemperatureCalibratedProvider:
    """Apply development-fitted temperatures to provider belief vectors."""

    def __init__(
        self,
        provider: CompletionProvider,
        calibrations: Mapping[str, TemperatureCalibration],
    ) -> None:
        if not calibrations:
            raise ValueError("at least one updater calibration is required")
        for updater_id, calibration in calibrations.items():
            if not isinstance(updater_id, str) or not updater_id:
                raise ValueError("calibration updater IDs must be nonempty")
            if calibration.fitted_splits != ("development",):
                raise ValueError(
                    "LLM calibration must be fitted on development only"
                )
        self.provider = provider
        self.calibrations = dict(calibrations)
        self._raw: dict[str, LLMResponse] = {}
        self._calibrated: dict[str, LLMResponse] = {}

    def complete(self, request: LLMRequest) -> LLMResponse:
        try:
            calibration = self.calibrations[request.updater_id]
        except KeyError as exc:
            raise KeyError(
                f"no LLM calibration for updater {request.updater_id!r}"
            ) from exc
        raw = self.provider.complete(request)
        beliefs = {
            attribute: {
                value: probability
                for value, probability in zip(
                    VALUES,
                    calibration.apply(
                        tuple(
                            float(raw.beliefs[attribute][value])
                            for value in VALUES
                        )
                    ),
                )
            }
            for attribute in ATTRIBUTES
        }
        calibrated = LLMResponse.parse(
            {
                "schema_version": 1,
                "request_id": raw.request_id,
                "prompt_sha256": raw.prompt_sha256,
                "model_id": raw.model_id,
                "beliefs": beliefs,
                "raw_response_sha256": raw.raw_response_sha256,
            }
        )
        self._raw[request.request_id] = raw
        self._calibrated[request.request_id] = calibrated
        return calibrated

    @property
    def raw_responses(self) -> tuple[LLMResponse, ...]:
        return tuple(self._raw.values())

    @property
    def calibrated_responses(self) -> tuple[LLMResponse, ...]:
        return tuple(self._calibrated.values())


def write_requests(path: str | Path, requests: Iterable[LLMRequest]) -> int:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for request in requests:
            handle.write(_canonical(request.to_dict()) + "\n")
            count += 1
    return count


def read_responses(path: str | Path) -> tuple[LLMResponse, ...]:
    source = Path(path)
    responses: list[LLMResponse] = []
    seen: set[str] = set()
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
                response = LLMResponse.parse(parsed)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"{source}:{line_number}: {exc}") from exc
            if response.request_id in seen:
                raise ValueError(f"duplicate request_id: {response.request_id}")
            seen.add(response.request_id)
            responses.append(response)
    return tuple(responses)


class ReplayProvider:
    """Content-addressed response lookup; never performs a network call."""

    def __init__(self, responses: Iterable[LLMResponse]) -> None:
        material = tuple(responses)
        if len({response.request_id for response in material}) != len(material):
            raise ValueError("replay responses contain duplicate request IDs")
        self._responses = {
            response.request_id: response for response in material
        }

    def complete(self, request: LLMRequest) -> LLMResponse:
        try:
            response = self._responses[request.request_id]
        except KeyError as exc:
            raise KeyError(f"no replay response for {request.request_id}") from exc
        if response.prompt_sha256 != request.prompt_sha256:
            raise ValueError(
                f"prompt hash mismatch for {request.request_id}: "
                "response is not bound to this request"
            )
        return response

    def validate_coverage(self, requests: Iterable[LLMRequest]) -> None:
        """Require an exact request/response ID set and matching prompt hashes."""

        material = tuple(requests)
        request_ids = {request.request_id for request in material}
        response_ids = set(self._responses)
        missing = sorted(request_ids - response_ids)
        unexpected = sorted(response_ids - request_ids)
        if missing or unexpected:
            raise ValueError(
                f"replay coverage mismatch; missing={missing}, unexpected={unexpected}"
            )
        for request in material:
            self.complete(request)
