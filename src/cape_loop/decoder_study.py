"""External decoder and human-judgment collection infrastructure.

The deterministic native projections in :mod:`cape_loop.native` are useful
representation checks, but they are not independent judgments.  This module
defines a separate, content-addressed exchange for externally supplied model or
human decoder judgments and records enough source metadata to audit a proposed
two-decoder design.  Metadata eligibility is never presented as proof of
statistical independence.

The human-study records are likewise collection and analysis contracts only.
They enforce consent and blinding metadata but make no claim about ethics or
IRB approval.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable, Mapping, Sequence
import json
import math

from .calibration import (
    CalibrationExample,
    TemperatureCalibration,
    fit_temperature,
)
from .human_study import CONDITIONS
from .llm_exchange import ATTRIBUTES, VALUES, LLMRequest, LLMResponse
from .native import NativeMemoryState
from .schemas import THETA_VALUES, Theta, validate_theta
from .statistics import paired_bootstrap_mean_difference


SPLITS = ("train", "development", "test")
VALUE_LABELS = ("-2", "-1", "+1", "+2")
JUDGMENT_ORIGINS = ("external_model", "human_annotator")
_FORBIDDEN_DECODER_KEYS = frozenset(
    {
        "system",
        "system_id",
        "updater",
        "updater_id",
        "memory_kind",
        "latent_truth",
        "truth",
        "user_id",
        "initial_profile",
    }
)

DECODER_INSTRUCTION = """\
Infer the user's persistent preference distribution from only the blinded state
payload. Return one probability distribution over -2, -1, +1, +2 for each of
the three declared attributes. Do not infer system identity or latent truth.
"""


def _require_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


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


def _validate_digest(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _assert_blinded(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key).lower() in _FORBIDDEN_DECODER_KEYS:
                raise ValueError(f"decoder payload leaks forbidden field {key!r}")
            _assert_blinded(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _assert_blinded(nested)


def _probability_rows(
    value: object,
    *,
    name: str = "probabilities",
) -> tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a three-row array")
    if len(value) != 3:
        raise ValueError(f"{name} must contain exactly three attributes")
    rows: list[tuple[float, float, float, float]] = []
    for attribute, raw in enumerate(value):
        if not isinstance(raw, Mapping) or set(raw) != set(VALUE_LABELS):
            raise ValueError(
                f"{name}[{attribute}] must contain exactly {VALUE_LABELS}"
            )
        parsed = tuple(
            _finite(raw[label], f"{name}[{attribute}][{label}]")
            for label in VALUE_LABELS
        )
        if any(item < 0.0 or item > 1.0 for item in parsed):
            raise ValueError(f"{name}[{attribute}] has a value outside [0, 1]")
        if not math.isclose(
            math.fsum(parsed), 1.0, rel_tol=0.0, abs_tol=1e-6
        ):
            raise ValueError(f"{name}[{attribute}] does not sum to one")
        rows.append(parsed)  # type: ignore[arg-type]
    return (rows[0], rows[1], rows[2])


def _rows_to_dict(
    rows: Sequence[Sequence[float]],
) -> list[dict[str, float]]:
    return [
        {label: float(value) for label, value in zip(VALUE_LABELS, row)}
        for row in rows
    ]


@dataclass(frozen=True, slots=True)
class ExternalDecoderRequest:
    """A blinded decoder request with a content-addressed rubric and payload."""

    request_id: str
    pseudonymous_state_id: str
    representation_id: str
    evaluation_split: str
    rubric_version: str
    payload: Mapping[str, Any]
    instruction: str
    request_sha256: str = ""

    def __post_init__(self) -> None:
        for name in (
            "request_id",
            "pseudonymous_state_id",
            "representation_id",
            "rubric_version",
            "instruction",
        ):
            _require_text(getattr(self, name), name)
        if self.evaluation_split not in SPLITS:
            raise ValueError(f"evaluation_split must be one of {SPLITS}")
        if not isinstance(self.payload, Mapping) or not self.payload:
            raise ValueError("decoder payload must be a nonempty object")
        _assert_blinded(self.payload)
        expected = _digest(self._binding_payload())
        if self.request_sha256 and self.request_sha256 != expected:
            raise ValueError("decoder request digest does not match its content")
        object.__setattr__(self, "request_sha256", expected)

    @classmethod
    def build(
        cls,
        *,
        request_id: str,
        pseudonymous_state_id: str,
        representation_id: str,
        evaluation_split: str,
        payload: Mapping[str, Any],
        rubric_version: str = "native-profile-decoder-v1",
        instruction: str = DECODER_INSTRUCTION,
    ) -> "ExternalDecoderRequest":
        return cls(
            request_id=request_id,
            pseudonymous_state_id=pseudonymous_state_id,
            representation_id=representation_id,
            evaluation_split=evaluation_split,
            rubric_version=rubric_version,
            payload=dict(payload),
            instruction=instruction,
        )

    @classmethod
    def parse(cls, raw: Mapping[str, Any]) -> "ExternalDecoderRequest":
        allowed = {
            "schema_version",
            "request_id",
            "pseudonymous_state_id",
            "representation_id",
            "evaluation_split",
            "rubric_version",
            "payload",
            "instruction",
            "request_sha256",
        }
        if set(raw) != allowed:
            raise ValueError(
                "decoder request has missing or unknown fields: "
                + _canonical(
                    {
                        "missing": sorted(allowed - set(raw)),
                        "unknown": sorted(set(raw) - allowed),
                    }
                )
            )
        if raw["schema_version"] != 1:
            raise ValueError("decoder request schema_version must be 1")
        payload = raw["payload"]
        if not isinstance(payload, Mapping):
            raise ValueError("decoder request payload must be an object")
        return cls(
            request_id=raw["request_id"],
            pseudonymous_state_id=raw["pseudonymous_state_id"],
            representation_id=raw["representation_id"],
            evaluation_split=raw["evaluation_split"],
            rubric_version=raw["rubric_version"],
            payload=dict(payload),
            instruction=raw["instruction"],
            request_sha256=raw["request_sha256"],
        )

    def _binding_payload(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "pseudonymous_state_id": self.pseudonymous_state_id,
            "representation_id": self.representation_id,
            "evaluation_split": self.evaluation_split,
            "rubric_version": self.rubric_version,
            "instruction": self.instruction,
            "payload": self.payload,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            **self._binding_payload(),
            "request_sha256": self.request_sha256,
        }


def read_external_decoder_requests(
    path: str | Path,
) -> tuple[ExternalDecoderRequest, ...]:
    source = Path(path)
    result: list[ExternalDecoderRequest] = []
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                decoded = json.loads(line)
                if not isinstance(decoded, Mapping):
                    raise ValueError("record must be a JSON object")
                result.append(ExternalDecoderRequest.parse(decoded))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"{source}:{line_number}: {exc}") from exc
    if len({item.request_id for item in result}) != len(result):
        raise ValueError(f"{source}: duplicate decoder request IDs")
    return tuple(result)


def build_blinded_native_decoder_request(
    state: NativeMemoryState,
    *,
    evaluation_split: str,
    assignment_nonce: str = "",
) -> ExternalDecoderRequest:
    """Project a native state without system, user, policy, or truth labels."""

    payload = {
        "representation_version": "blinded-native-content-v1",
        "episodes": [
            {
                "target_attribute": episode.target_attribute,
                "selected_direction": episode.selected_direction,
                "visible_mechanisms": list(episode.visible_mechanisms),
                "displayed_directions": list(episode.displayed_directions),
                "evidence_weight": episode.evidence_weight,
                "surface_response": episode.surface_response,
            }
            for episode in state.episodes
        ],
        "semantic_claims": [
            {
                "attribute": claim.attribute,
                "direction": claim.direction,
                "confidence": claim.confidence,
                "cumulative_evidence_weight": (
                    claim.cumulative_evidence_weight
                ),
                "support_count": len(claim.source_event_ids),
            }
            for claim in state.claims
        ],
        "persona_text": state.persona_text,
    }
    if not isinstance(assignment_nonce, str):
        raise TypeError("assignment_nonce must be a string")
    pseudonym = _digest(
        {
            "payload": payload,
            "assignment_nonce": assignment_nonce,
        }
    )
    return ExternalDecoderRequest.build(
        request_id=f"native-decoder:{pseudonym}",
        pseudonymous_state_id=pseudonym,
        representation_id="blinded-native-content-v1",
        evaluation_split=evaluation_split,
        payload=payload,
    )


def external_decoder_llm_request(
    request: ExternalDecoderRequest,
    *,
    decoder_instance_id: str,
) -> LLMRequest:
    """Adapt a blinded decoder item to the shared structured-output contract."""

    _require_text(decoder_instance_id, "decoder_instance_id")
    payload = {
        "decoder_request": {
            "representation_id": request.representation_id,
            "rubric_version": request.rubric_version,
            "payload": request.payload,
        }
    }
    instruction = request.instruction
    prompt_sha256 = sha256(
        (instruction + "\n" + _canonical(payload)).encode("utf-8")
    ).hexdigest()
    return LLMRequest(
        request_id=(
            f"external-decoder:{decoder_instance_id}:"
            f"{request.request_sha256}"
        ),
        updater_id="external_decoder",
        view="response_only",
        payload=payload,
        system_instruction=instruction,
        prompt_sha256=prompt_sha256,
    )


def external_decoder_judgment_from_response(
    request: ExternalDecoderRequest,
    response: LLMResponse,
    *,
    decoder_instance_id: str,
    decoder_family_id: str,
    source_descriptor: str,
) -> ExternalDecoderJudgment:
    """Bind one validated provider response back to its blinded request."""

    expected = external_decoder_llm_request(
        request,
        decoder_instance_id=decoder_instance_id,
    )
    if (
        response.request_id != expected.request_id
        or response.prompt_sha256 != expected.prompt_sha256
    ):
        raise ValueError("provider response is not bound to decoder request")
    probabilities = tuple(
        tuple(
            float(response.beliefs[attribute][value])
            for value in VALUES
        )
        for attribute in ATTRIBUTES
    )
    return ExternalDecoderJudgment(
        request_id=request.request_id,
        request_sha256=request.request_sha256,
        decoder_instance_id=decoder_instance_id,
        decoder_family_id=decoder_family_id,
        judgment_origin="external_model",
        source_descriptor=source_descriptor,
        blind_to_system_identity=True,
        blind_to_latent_truth=True,
        probabilities=probabilities,  # type: ignore[arg-type]
    )


@dataclass(frozen=True, slots=True)
class ExternalDecoderJudgment:
    """One externally produced judgment; deterministic views are disallowed."""

    request_id: str
    request_sha256: str
    decoder_instance_id: str
    decoder_family_id: str
    judgment_origin: str
    source_descriptor: str
    blind_to_system_identity: bool
    blind_to_latent_truth: bool
    probabilities: tuple[
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
    ]

    def __post_init__(self) -> None:
        for name in (
            "request_id",
            "decoder_instance_id",
            "decoder_family_id",
            "source_descriptor",
        ):
            _require_text(getattr(self, name), name)
        _validate_digest(self.request_sha256, "request_sha256")
        if self.judgment_origin not in JUDGMENT_ORIGINS:
            raise ValueError(
                "judgment_origin must be external_model or human_annotator; "
                "deterministic native views are not external judgments"
            )
        if not isinstance(self.blind_to_system_identity, bool):
            raise TypeError("blind_to_system_identity must be Boolean")
        if not isinstance(self.blind_to_latent_truth, bool):
            raise TypeError("blind_to_latent_truth must be Boolean")
        object.__setattr__(
            self,
            "probabilities",
            _probability_rows(
                _rows_to_dict(self.probabilities),
                name="probabilities",
            ),
        )

    @classmethod
    def parse(cls, raw: Mapping[str, Any]) -> "ExternalDecoderJudgment":
        allowed = {
            "schema_version",
            "request_id",
            "request_sha256",
            "decoder_instance_id",
            "decoder_family_id",
            "judgment_origin",
            "source_descriptor",
            "blind_to_system_identity",
            "blind_to_latent_truth",
            "probabilities",
        }
        if set(raw) != allowed:
            raise ValueError(
                "decoder judgment has missing or unknown fields: "
                + _canonical(
                    {
                        "missing": sorted(allowed - set(raw)),
                        "unknown": sorted(set(raw) - allowed),
                    }
                )
            )
        if raw["schema_version"] != 1:
            raise ValueError("decoder judgment schema_version must be 1")
        return cls(
            request_id=raw["request_id"],
            request_sha256=raw["request_sha256"],
            decoder_instance_id=raw["decoder_instance_id"],
            decoder_family_id=raw["decoder_family_id"],
            judgment_origin=raw["judgment_origin"],
            source_descriptor=raw["source_descriptor"],
            blind_to_system_identity=raw["blind_to_system_identity"],
            blind_to_latent_truth=raw["blind_to_latent_truth"],
            probabilities=_probability_rows(raw["probabilities"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "request_id": self.request_id,
            "request_sha256": self.request_sha256,
            "decoder_instance_id": self.decoder_instance_id,
            "decoder_family_id": self.decoder_family_id,
            "judgment_origin": self.judgment_origin,
            "source_descriptor": self.source_descriptor,
            "blind_to_system_identity": self.blind_to_system_identity,
            "blind_to_latent_truth": self.blind_to_latent_truth,
            "probabilities": _rows_to_dict(self.probabilities),
        }


def read_external_decoder_judgments(
    path: str | Path,
) -> tuple[ExternalDecoderJudgment, ...]:
    source = Path(path)
    result: list[ExternalDecoderJudgment] = []
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                decoded = json.loads(line)
                if not isinstance(decoded, Mapping):
                    raise ValueError("record must be a JSON object")
                result.append(ExternalDecoderJudgment.parse(decoded))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"{source}:{line_number}: {exc}") from exc
    return tuple(result)


@dataclass(frozen=True, slots=True)
class DecoderImportAudit:
    request_count: int
    judgment_count: int
    minimum_sources_per_request: int
    complete_coverage: bool
    source_design_eligible: bool
    counts_by_request: tuple[tuple[str, int], ...]
    family_counts_by_request: tuple[tuple[str, int], ...]
    source_descriptor_counts_by_request: tuple[tuple[str, int], ...]
    missing_request_ids: tuple[str, ...]
    caveat: str = (
        "Distinct external source metadata is a design check, not proof of "
        "statistical independence."
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "request_count": self.request_count,
            "judgment_count": self.judgment_count,
            "minimum_sources_per_request": self.minimum_sources_per_request,
            "complete_coverage": self.complete_coverage,
            "source_design_eligible": self.source_design_eligible,
            "counts_by_request": dict(self.counts_by_request),
            "family_counts_by_request": dict(
                self.family_counts_by_request
            ),
            "source_descriptor_counts_by_request": dict(
                self.source_descriptor_counts_by_request
            ),
            "missing_request_ids": list(self.missing_request_ids),
            "caveat": self.caveat,
        }


def validate_external_decoder_import(
    requests: Iterable[ExternalDecoderRequest],
    judgments: Iterable[ExternalDecoderJudgment],
    *,
    minimum_sources_per_request: int = 2,
    require_distinct_families: bool = True,
) -> DecoderImportAudit:
    """Validate hashes, blinding attestations, coverage, and source diversity."""

    if (
        isinstance(minimum_sources_per_request, bool)
        or not isinstance(minimum_sources_per_request, int)
        or minimum_sources_per_request <= 0
    ):
        raise ValueError("minimum_sources_per_request must be positive")
    request_rows = tuple(requests)
    judgment_rows = tuple(judgments)
    if len({item.request_id for item in request_rows}) != len(request_rows):
        raise ValueError("decoder request IDs must be unique")
    request_by_id = {item.request_id: item for item in request_rows}
    pair_keys = [
        (item.request_id, item.decoder_instance_id) for item in judgment_rows
    ]
    if len(set(pair_keys)) != len(pair_keys):
        raise ValueError("duplicate decoder instance judgment for a request")

    instance_metadata: dict[str, set[tuple[str, str, str]]] = {}
    for judgment in judgment_rows:
        try:
            request = request_by_id[judgment.request_id]
        except KeyError as exc:
            raise ValueError(
                f"judgment references unknown request {judgment.request_id!r}"
            ) from exc
        if judgment.request_sha256 != request.request_sha256:
            raise ValueError(
                f"request hash mismatch for {judgment.request_id}"
            )
        if (
            not judgment.blind_to_system_identity
            or not judgment.blind_to_latent_truth
        ):
            raise ValueError(
                f"decoder source {judgment.decoder_instance_id!r} is not "
                "declared blind to both protected fields"
            )
        instance_metadata.setdefault(
            judgment.decoder_instance_id, set()
        ).add(
            (
                judgment.decoder_family_id,
                judgment.source_descriptor,
                judgment.judgment_origin,
            )
        )
    inconsistent = sorted(
        instance
        for instance, metadata in instance_metadata.items()
        if len(metadata) != 1
    )
    if inconsistent:
        raise ValueError(
            "decoder instance changes family across judgments: "
            + ", ".join(inconsistent)
        )

    counts: list[tuple[str, int]] = []
    family_counts: list[tuple[str, int]] = []
    descriptor_counts: list[tuple[str, int]] = []
    missing: list[str] = []
    eligible = True
    for request in sorted(request_rows, key=lambda item: item.request_id):
        matching = [
            row for row in judgment_rows if row.request_id == request.request_id
        ]
        distinct_instances = {
            row.decoder_instance_id for row in matching
        }
        distinct_families = {row.decoder_family_id for row in matching}
        distinct_descriptors = {
            row.source_descriptor for row in matching
        }
        counts.append((request.request_id, len(distinct_instances)))
        family_counts.append((request.request_id, len(distinct_families)))
        descriptor_counts.append(
            (request.request_id, len(distinct_descriptors))
        )
        enough = (
            len(distinct_instances) >= minimum_sources_per_request
            and len(distinct_descriptors) >= minimum_sources_per_request
        )
        if require_distinct_families:
            enough = (
                enough
                and len(distinct_families) >= minimum_sources_per_request
            )
        if not enough:
            missing.append(request.request_id)
            eligible = False
    return DecoderImportAudit(
        request_count=len(request_rows),
        judgment_count=len(judgment_rows),
        minimum_sources_per_request=minimum_sources_per_request,
        complete_coverage=not missing,
        source_design_eligible=eligible,
        counts_by_request=tuple(counts),
        family_counts_by_request=tuple(family_counts),
        source_descriptor_counts_by_request=tuple(descriptor_counts),
        missing_request_ids=tuple(missing),
    )


@dataclass(frozen=True, slots=True)
class DecoderTruthLabel:
    """Truth retained outside decoder payloads and joined by pseudonym later."""

    pseudonymous_state_id: str
    theta: Theta
    evaluation_split: str

    def __post_init__(self) -> None:
        _require_text(self.pseudonymous_state_id, "pseudonymous_state_id")
        object.__setattr__(self, "theta", validate_theta(self.theta))
        if self.evaluation_split not in SPLITS:
            raise ValueError(f"evaluation_split must be one of {SPLITS}")

    @classmethod
    def parse(cls, raw: Mapping[str, Any]) -> "DecoderTruthLabel":
        allowed = {
            "schema_version",
            "pseudonymous_state_id",
            "theta",
            "evaluation_split",
        }
        if set(raw) != allowed:
            raise ValueError(
                "decoder truth label has missing or unknown fields: "
                + _canonical(
                    {
                        "missing": sorted(allowed - set(raw)),
                        "unknown": sorted(set(raw) - allowed),
                    }
                )
            )
        if raw["schema_version"] != 1:
            raise ValueError("decoder truth schema_version must be 1")
        return cls(
            pseudonymous_state_id=raw["pseudonymous_state_id"],
            theta=raw["theta"],
            evaluation_split=raw["evaluation_split"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "pseudonymous_state_id": self.pseudonymous_state_id,
            "theta": list(self.theta),
            "evaluation_split": self.evaluation_split,
        }


def read_decoder_truth_labels(
    path: str | Path,
) -> tuple[DecoderTruthLabel, ...]:
    source = Path(path)
    result: list[DecoderTruthLabel] = []
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                decoded = json.loads(line)
                if not isinstance(decoded, Mapping):
                    raise ValueError("record must be a JSON object")
                result.append(DecoderTruthLabel.parse(decoded))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"{source}:{line_number}: {exc}") from exc
    if len({item.pseudonymous_state_id for item in result}) != len(result):
        raise ValueError(f"{source}: duplicate decoder truth state IDs")
    return tuple(result)


@dataclass(frozen=True, slots=True)
class DecoderCalibrationBundle:
    calibrators: tuple[tuple[str, TemperatureCalibration], ...]
    fitted_split: str = "development"

    def __post_init__(self) -> None:
        if self.fitted_split != "development":
            raise ValueError("decoder calibration must be development-only")
        families = [family for family, _ in self.calibrators]
        if len(set(families)) != len(families):
            raise ValueError("decoder calibration families must be unique")
        if any(
            calibrator.fitted_splits != ("development",)
            for _, calibrator in self.calibrators
        ):
            raise ValueError("decoder calibrator used a non-development split")

    def for_family(self, family_id: str) -> TemperatureCalibration:
        for family, calibrator in self.calibrators:
            if family == family_id:
                return calibrator
        raise KeyError(f"no decoder calibrator for family {family_id!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "per-decoder-family-temperature",
            "fitted_split": self.fitted_split,
            "calibrators": {
                family: calibrator.to_dict()
                for family, calibrator in self.calibrators
            },
        }


def _bound_decoder_rows(
    requests: Sequence[ExternalDecoderRequest],
    judgments: Sequence[ExternalDecoderJudgment],
    labels: Sequence[DecoderTruthLabel],
) -> tuple[
    tuple[ExternalDecoderRequest, ExternalDecoderJudgment, DecoderTruthLabel],
    ...,
]:
    request_by_id = {item.request_id: item for item in requests}
    if len(request_by_id) != len(requests):
        raise ValueError("decoder request IDs must be unique")
    label_by_state = {item.pseudonymous_state_id: item for item in labels}
    if len(label_by_state) != len(labels):
        raise ValueError("decoder truth labels must have unique state IDs")
    result = []
    for judgment in judgments:
        try:
            request = request_by_id[judgment.request_id]
            label = label_by_state[request.pseudonymous_state_id]
        except KeyError as exc:
            raise ValueError("decoder row lacks a bound request or truth label") from exc
        if request.evaluation_split != label.evaluation_split:
            raise ValueError("request and truth label splits disagree")
        if judgment.request_sha256 != request.request_sha256:
            raise ValueError("decoder judgment has a mismatched request hash")
        result.append((request, judgment, label))
    return tuple(result)


def fit_decoder_calibration(
    requests: Iterable[ExternalDecoderRequest],
    judgments: Iterable[ExternalDecoderJudgment],
    labels: Iterable[DecoderTruthLabel],
) -> DecoderCalibrationBundle:
    """Fit one temperature per decoder family using development labels only."""

    request_rows = tuple(requests)
    judgment_rows = tuple(judgments)
    label_rows = tuple(labels)
    bound = _bound_decoder_rows(request_rows, judgment_rows, label_rows)
    examples_by_family: dict[str, list[CalibrationExample]] = {}
    for request, judgment, label in bound:
        if request.evaluation_split != "development":
            continue
        for attribute, probabilities in enumerate(judgment.probabilities):
            true_index = THETA_VALUES.index(label.theta[attribute])
            examples_by_family.setdefault(
                judgment.decoder_family_id, []
            ).append(
                CalibrationExample(
                    probabilities=tuple(probabilities),
                    true_index=true_index,
                    split="development",
                )
            )
    if not examples_by_family:
        raise ValueError("no development decoder judgments for calibration")
    calibrators = tuple(
        (
            family,
            fit_temperature(
                examples,
                allowed_splits=("development",),
            ),
        )
        for family, examples in sorted(examples_by_family.items())
    )
    return DecoderCalibrationBundle(calibrators=calibrators)


@dataclass(frozen=True, slots=True)
class ReliabilityBin:
    lower: float
    upper: float
    count: int
    mean_confidence: float | None
    empirical_accuracy: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "lower": self.lower,
            "upper": self.upper,
            "count": self.count,
            "mean_confidence": self.mean_confidence,
            "empirical_accuracy": self.empirical_accuracy,
        }


@dataclass(frozen=True, slots=True)
class DecoderFamilyMetrics:
    decoder_family_id: str
    judgment_count: int
    attribute_prediction_count: int
    raw_brier: float
    calibrated_brier: float
    raw_nll: float
    calibrated_nll: float
    raw_accuracy: float
    calibrated_accuracy: float
    raw_ece: float
    calibrated_ece: float
    raw_reliability: tuple[ReliabilityBin, ...]
    calibrated_reliability: tuple[ReliabilityBin, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "decoder_family_id": self.decoder_family_id,
            "judgment_count": self.judgment_count,
            "attribute_prediction_count": self.attribute_prediction_count,
            "raw_brier": self.raw_brier,
            "calibrated_brier": self.calibrated_brier,
            "raw_nll": self.raw_nll,
            "calibrated_nll": self.calibrated_nll,
            "raw_accuracy": self.raw_accuracy,
            "calibrated_accuracy": self.calibrated_accuracy,
            "raw_ece": self.raw_ece,
            "calibrated_ece": self.calibrated_ece,
            "raw_reliability": [
                item.to_dict() for item in self.raw_reliability
            ],
            "calibrated_reliability": [
                item.to_dict() for item in self.calibrated_reliability
            ],
        }


@dataclass(frozen=True, slots=True)
class DecoderAgreement:
    source_pair: str
    shared_request_count: int
    attribute_comparison_count: int
    argmax_agreement: float
    mean_total_variation: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_pair": self.source_pair,
            "shared_request_count": self.shared_request_count,
            "attribute_comparison_count": self.attribute_comparison_count,
            "argmax_agreement": self.argmax_agreement,
            "mean_total_variation": self.mean_total_variation,
        }


@dataclass(frozen=True, slots=True)
class DecoderAnalysis:
    evaluation_splits: tuple[str, ...]
    calibration: DecoderCalibrationBundle
    family_metrics: tuple[DecoderFamilyMetrics, ...]
    agreement: tuple[DecoderAgreement, ...]
    source_design_audit: DecoderImportAudit
    interpretation_boundary: str = (
        "Agreement and source-diversity metadata do not establish statistically "
        "independent errors; deterministic native projections are excluded."
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "evaluation_splits": list(self.evaluation_splits),
            "calibration": self.calibration.to_dict(),
            "family_metrics": [
                item.to_dict() for item in self.family_metrics
            ],
            "agreement": [item.to_dict() for item in self.agreement],
            "source_design_audit": self.source_design_audit.to_dict(),
            "interpretation_boundary": self.interpretation_boundary,
        }


def _argmax(probabilities: Sequence[float]) -> int:
    return max(range(len(probabilities)), key=lambda index: (probabilities[index], -index))


def _reliability(
    predictions: Sequence[tuple[Sequence[float], int]],
    *,
    bins: int,
) -> tuple[tuple[ReliabilityBin, ...], float]:
    bucketed: list[list[tuple[float, float]]] = [[] for _ in range(bins)]
    for probabilities, truth_index in predictions:
        predicted = _argmax(probabilities)
        confidence = float(probabilities[predicted])
        index = min(int(confidence * bins), bins - 1)
        bucketed[index].append(
            (confidence, 1.0 if predicted == truth_index else 0.0)
        )
    result: list[ReliabilityBin] = []
    ece = 0.0
    total = len(predictions)
    for index, values in enumerate(bucketed):
        lower, upper = index / bins, (index + 1) / bins
        if not values:
            result.append(ReliabilityBin(lower, upper, 0, None, None))
            continue
        confidence = mean(item[0] for item in values)
        accuracy = mean(item[1] for item in values)
        ece += len(values) / total * abs(confidence - accuracy)
        result.append(
            ReliabilityBin(lower, upper, len(values), confidence, accuracy)
        )
    return tuple(result), ece


def _prediction_metrics(
    predictions: Sequence[tuple[Sequence[float], int]],
    *,
    bins: int,
) -> tuple[float, float, float, tuple[ReliabilityBin, ...], float]:
    if not predictions:
        raise ValueError("decoder metric group is empty")
    brier = mean(
        math.fsum(
            (float(probability) - (1.0 if index == truth else 0.0)) ** 2
            for index, probability in enumerate(probabilities)
        )
        for probabilities, truth in predictions
    )
    nll = mean(
        -math.log(max(float(probabilities[truth]), 1e-15))
        for probabilities, truth in predictions
    )
    accuracy = mean(
        1.0 if _argmax(probabilities) == truth else 0.0
        for probabilities, truth in predictions
    )
    reliability, ece = _reliability(predictions, bins=bins)
    return brier, nll, accuracy, reliability, ece


def analyze_external_decoders(
    requests: Iterable[ExternalDecoderRequest],
    judgments: Iterable[ExternalDecoderJudgment],
    labels: Iterable[DecoderTruthLabel],
    *,
    calibration: DecoderCalibrationBundle | None = None,
    evaluation_splits: Sequence[str] = ("test",),
    reliability_bins: int = 10,
) -> DecoderAnalysis:
    """Report raw/calibrated performance, reliability, and source agreement."""

    splits = tuple(sorted(set(evaluation_splits)))
    if not splits or set(splits) - set(SPLITS):
        raise ValueError(f"evaluation_splits must be drawn from {SPLITS}")
    if (
        isinstance(reliability_bins, bool)
        or not isinstance(reliability_bins, int)
        or reliability_bins <= 1
    ):
        raise ValueError("reliability_bins must be an integer greater than one")
    request_rows = tuple(requests)
    judgment_rows = tuple(judgments)
    label_rows = tuple(labels)
    audit = validate_external_decoder_import(request_rows, judgment_rows)
    fitted = calibration or fit_decoder_calibration(
        request_rows, judgment_rows, label_rows
    )
    bound = tuple(
        row
        for row in _bound_decoder_rows(
            request_rows, judgment_rows, label_rows
        )
        if row[0].evaluation_split in splits
    )
    if not bound:
        raise ValueError("no decoder judgments in requested evaluation splits")

    raw_by_family: dict[str, list[tuple[Sequence[float], int]]] = {}
    calibrated_by_family: dict[
        str, list[tuple[Sequence[float], int]]
    ] = {}
    judgment_counts: dict[str, int] = {}
    for _, judgment, label in bound:
        calibrator = fitted.for_family(judgment.decoder_family_id)
        judgment_counts[judgment.decoder_family_id] = (
            judgment_counts.get(judgment.decoder_family_id, 0) + 1
        )
        for attribute, probabilities in enumerate(judgment.probabilities):
            truth_index = THETA_VALUES.index(label.theta[attribute])
            raw_by_family.setdefault(
                judgment.decoder_family_id, []
            ).append((probabilities, truth_index))
            calibrated_by_family.setdefault(
                judgment.decoder_family_id, []
            ).append((calibrator.apply(probabilities), truth_index))

    family_metrics: list[DecoderFamilyMetrics] = []
    for family in sorted(raw_by_family):
        raw_metrics = _prediction_metrics(
            raw_by_family[family], bins=reliability_bins
        )
        calibrated_metrics = _prediction_metrics(
            calibrated_by_family[family], bins=reliability_bins
        )
        family_metrics.append(
            DecoderFamilyMetrics(
                decoder_family_id=family,
                judgment_count=judgment_counts[family],
                attribute_prediction_count=len(raw_by_family[family]),
                raw_brier=raw_metrics[0],
                calibrated_brier=calibrated_metrics[0],
                raw_nll=raw_metrics[1],
                calibrated_nll=calibrated_metrics[1],
                raw_accuracy=raw_metrics[2],
                calibrated_accuracy=calibrated_metrics[2],
                raw_ece=raw_metrics[4],
                calibrated_ece=calibrated_metrics[4],
                raw_reliability=raw_metrics[3],
                calibrated_reliability=calibrated_metrics[3],
            )
        )

    judgments_by_request: dict[str, list[ExternalDecoderJudgment]] = {}
    selected_request_ids = {request.request_id for request, _, _ in bound}
    for judgment in judgment_rows:
        if judgment.request_id in selected_request_ids:
            judgments_by_request.setdefault(
                judgment.request_id, []
            ).append(judgment)
    agreement_values: dict[str, list[tuple[float, float]]] = {}
    shared_requests: dict[str, set[str]] = {}
    for request_id, source_rows in judgments_by_request.items():
        ordered = sorted(
            source_rows,
            key=lambda item: (
                item.decoder_family_id,
                item.decoder_instance_id,
            ),
        )
        for left_index, left in enumerate(ordered):
            for right in ordered[left_index + 1 :]:
                if left.decoder_family_id == right.decoder_family_id:
                    continue
                pair = "|".join(
                    sorted(
                        (left.decoder_family_id, right.decoder_family_id)
                    )
                )
                shared_requests.setdefault(pair, set()).add(request_id)
                for left_row, right_row in zip(
                    left.probabilities, right.probabilities
                ):
                    same = 1.0 if _argmax(left_row) == _argmax(right_row) else 0.0
                    tv = 0.5 * math.fsum(
                        abs(first - second)
                        for first, second in zip(left_row, right_row)
                    )
                    agreement_values.setdefault(pair, []).append((same, tv))
    agreements = tuple(
        DecoderAgreement(
            source_pair=pair,
            shared_request_count=len(shared_requests[pair]),
            attribute_comparison_count=len(values),
            argmax_agreement=mean(item[0] for item in values),
            mean_total_variation=mean(item[1] for item in values),
        )
        for pair, values in sorted(agreement_values.items())
    )
    return DecoderAnalysis(
        evaluation_splits=splits,
        calibration=fitted,
        family_metrics=tuple(family_metrics),
        agreement=agreements,
        source_design_audit=audit,
    )


@dataclass(frozen=True, slots=True)
class HumanCollectionRecord:
    """A de-identified response row captured while item conditions are hidden."""

    participant_code: str
    assignment_id: str
    assignment_protocol_id: str
    display_id: str
    rating: int
    response_time_ms: int
    consent_version: str
    consented: bool
    blinding_version: str
    comprehension_check_id: str
    comprehension_passed: bool

    def __post_init__(self) -> None:
        for name in (
            "participant_code",
            "assignment_id",
            "assignment_protocol_id",
            "display_id",
            "consent_version",
            "blinding_version",
            "comprehension_check_id",
        ):
            _require_text(getattr(self, name), name)
        if (
            isinstance(self.rating, bool)
            or not isinstance(self.rating, int)
            or not 1 <= self.rating <= 7
        ):
            raise ValueError("rating must be an integer from 1 to 7")
        if (
            isinstance(self.response_time_ms, bool)
            or not isinstance(self.response_time_ms, int)
            or self.response_time_ms < 0
        ):
            raise ValueError("response_time_ms must be a non-negative integer")
        if not isinstance(self.consented, bool):
            raise TypeError("consented must be Boolean")
        if not isinstance(self.comprehension_passed, bool):
            raise TypeError("comprehension_passed must be Boolean")

    @classmethod
    def parse(cls, raw: Mapping[str, Any]) -> "HumanCollectionRecord":
        allowed = {
            "schema_version",
            "participant_code",
            "assignment_id",
            "assignment_protocol_id",
            "display_id",
            "rating",
            "response_time_ms",
            "consent_version",
            "consented",
            "blinding_version",
            "comprehension_check_id",
            "comprehension_passed",
        }
        if set(raw) != allowed:
            raise ValueError(
                "human response has missing or unknown fields: "
                + _canonical(
                    {
                        "missing": sorted(allowed - set(raw)),
                        "unknown": sorted(set(raw) - allowed),
                    }
                )
            )
        if raw["schema_version"] != 1:
            raise ValueError("human response schema_version must be 1")
        return cls(
            participant_code=raw["participant_code"],
            assignment_id=raw["assignment_id"],
            assignment_protocol_id=raw["assignment_protocol_id"],
            display_id=raw["display_id"],
            rating=raw["rating"],
            response_time_ms=raw["response_time_ms"],
            consent_version=raw["consent_version"],
            consented=raw["consented"],
            blinding_version=raw["blinding_version"],
            comprehension_check_id=raw["comprehension_check_id"],
            comprehension_passed=raw["comprehension_passed"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "participant_code": self.participant_code,
            "assignment_id": self.assignment_id,
            "assignment_protocol_id": self.assignment_protocol_id,
            "display_id": self.display_id,
            "rating": self.rating,
            "response_time_ms": self.response_time_ms,
            "consent_version": self.consent_version,
            "consented": self.consented,
            "blinding_version": self.blinding_version,
            "comprehension_check_id": self.comprehension_check_id,
            "comprehension_passed": self.comprehension_passed,
        }


def read_human_collection(
    path: str | Path | bytes,
) -> tuple[HumanCollectionRecord, ...]:
    if isinstance(path, bytes):
        source_label = "<human-collection-bytes>"
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
    result: list[HumanCollectionRecord] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            decoded = json.loads(line)
            if not isinstance(decoded, Mapping):
                raise ValueError("record must be a JSON object")
            result.append(HumanCollectionRecord.parse(decoded))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{source_label}:{line_number}: {exc}"
            ) from exc
    return tuple(result)


@dataclass(frozen=True, slots=True)
class HumanImportAudit:
    record_count: int
    participant_count: int
    consent_eligible_count: int
    comprehension_eligible_count: int
    analysis_eligible_count: int
    excluded_no_consent: int
    excluded_comprehension: int
    ethics_boundary: str = (
        "Consent-version and comprehension fields were validated; this "
        "software does not assert ethics-review or IRB approval."
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "record_count": self.record_count,
            "participant_count": self.participant_count,
            "consent_eligible_count": self.consent_eligible_count,
            "comprehension_eligible_count": self.comprehension_eligible_count,
            "analysis_eligible_count": self.analysis_eligible_count,
            "excluded_no_consent": self.excluded_no_consent,
            "excluded_comprehension": self.excluded_comprehension,
            "ethics_boundary": self.ethics_boundary,
        }


Codebook = Mapping[str, Mapping[str, Mapping[str, str]]]


def validate_human_collection(
    records: Iterable[HumanCollectionRecord],
    *,
    assignment_codebooks: Codebook,
    expected_assignment_protocol_id: str,
    expected_consent_version: str,
    expected_blinding_version: str,
) -> HumanImportAudit:
    """Validate de-identification, assignment binding, and eligibility fields."""

    for value, name in (
        (expected_assignment_protocol_id, "expected_assignment_protocol_id"),
        (expected_consent_version, "expected_consent_version"),
        (expected_blinding_version, "expected_blinding_version"),
    ):
        _require_text(value, name)
    material = tuple(records)
    response_keys = [
        (row.participant_code, row.assignment_id, row.display_id)
        for row in material
    ]
    if len(set(response_keys)) != len(response_keys):
        raise ValueError("duplicate human response for participant/item")
    participant_metadata: dict[
        str, set[tuple[str, bool, str, bool]]
    ] = {}
    for row in material:
        if row.assignment_protocol_id != expected_assignment_protocol_id:
            raise ValueError("human response uses the wrong assignment protocol")
        if row.consent_version != expected_consent_version:
            raise ValueError("human response uses the wrong consent version")
        if row.blinding_version != expected_blinding_version:
            raise ValueError("human response uses the wrong blinding version")
        try:
            entry = assignment_codebooks[row.assignment_id][row.display_id]
        except KeyError as exc:
            raise ValueError(
                "human response has an unknown assignment/display binding"
            ) from exc
        if set(entry) != {"item_id", "scenario_id", "condition"}:
            raise ValueError("assignment codebook entry is malformed")
        if entry["condition"] not in CONDITIONS:
            raise ValueError("assignment codebook has an unknown condition")
        participant_metadata.setdefault(
            row.participant_code, set()
        ).add(
            (
                row.assignment_id,
                row.consented,
                row.comprehension_check_id,
                row.comprehension_passed,
            )
        )
    inconsistent_participants = sorted(
        participant
        for participant, metadata in participant_metadata.items()
        if len(metadata) != 1
    )
    if inconsistent_participants:
        raise ValueError(
            "participant assignment/eligibility metadata changes across rows: "
            + ", ".join(inconsistent_participants)
        )
    eligible_participants = {
        participant
        for participant, metadata in participant_metadata.items()
        if next(iter(metadata))[1] and next(iter(metadata))[3]
    }
    consented_participants = {
        participant
        for participant, metadata in participant_metadata.items()
        if next(iter(metadata))[1]
    }
    comprehension_participants = {
        participant
        for participant, metadata in participant_metadata.items()
        if next(iter(metadata))[3]
    }
    consent_count = sum(
        row.participant_code in consented_participants for row in material
    )
    comprehension_count = sum(
        row.participant_code in comprehension_participants
        for row in material
    )
    eligible = sum(
        row.participant_code in eligible_participants for row in material
    )
    return HumanImportAudit(
        record_count=len(material),
        participant_count=len({row.participant_code for row in material}),
        consent_eligible_count=consent_count,
        comprehension_eligible_count=comprehension_count,
        analysis_eligible_count=eligible,
        excluded_no_consent=sum(
            row.participant_code not in consented_participants
            for row in material
        ),
        excluded_comprehension=sum(
            row.participant_code in consented_participants
            and row.participant_code not in comprehension_participants
            for row in material
        ),
    )


@dataclass(frozen=True, slots=True)
class HumanConditionSummary:
    condition: str
    response_count: int
    participant_count: int
    mean_rating: float
    standard_deviation: float | None
    mean_response_time_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "response_count": self.response_count,
            "participant_count": self.participant_count,
            "mean_rating": self.mean_rating,
            "standard_deviation": self.standard_deviation,
            "mean_response_time_ms": self.mean_response_time_ms,
        }


@dataclass(frozen=True, slots=True)
class HumanPairedContrast:
    contrast_id: str
    first_condition: str
    second_condition: str
    paired_participant_count: int
    mean_difference: float | None
    bootstrap_lower: float | None
    bootstrap_upper: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "contrast_id": self.contrast_id,
            "first_condition": self.first_condition,
            "second_condition": self.second_condition,
            "paired_participant_count": self.paired_participant_count,
            "mean_difference": self.mean_difference,
            "bootstrap_lower": self.bootstrap_lower,
            "bootstrap_upper": self.bootstrap_upper,
        }


@dataclass(frozen=True, slots=True)
class HumanEvidenceAnalysis:
    import_audit: HumanImportAudit
    condition_summaries: tuple[HumanConditionSummary, ...]
    evidence_strength_ranking: tuple[str, ...]
    paired_contrasts: tuple[HumanPairedContrast, ...]
    interpretation_boundary: str = (
        "These are pragmatic judgments, not an exact causal oracle; no "
        "ethics-review approval is inferred by this analysis."
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "import_audit": self.import_audit.to_dict(),
            "condition_summaries": [
                item.to_dict() for item in self.condition_summaries
            ],
            "evidence_strength_ranking": list(
                self.evidence_strength_ranking
            ),
            "paired_contrasts": [
                item.to_dict() for item in self.paired_contrasts
            ],
            "interpretation_boundary": self.interpretation_boundary,
        }


def analyze_human_evidence_strength(
    records: Iterable[HumanCollectionRecord],
    *,
    assignment_codebooks: Codebook,
    expected_assignment_protocol_id: str,
    expected_consent_version: str,
    expected_blinding_version: str,
    bootstrap_replicates: int = 1000,
    seed: int = 1729,
) -> HumanEvidenceAnalysis:
    """Analyze only consented, comprehension-passing blinded responses."""

    if (
        isinstance(bootstrap_replicates, bool)
        or not isinstance(bootstrap_replicates, int)
        or bootstrap_replicates <= 0
    ):
        raise ValueError("bootstrap_replicates must be positive")
    material = tuple(records)
    audit = validate_human_collection(
        material,
        assignment_codebooks=assignment_codebooks,
        expected_assignment_protocol_id=expected_assignment_protocol_id,
        expected_consent_version=expected_consent_version,
        expected_blinding_version=expected_blinding_version,
    )
    eligible: list[tuple[HumanCollectionRecord, Mapping[str, str]]] = []
    for row in material:
        if not row.consented or not row.comprehension_passed:
            continue
        eligible.append(
            (
                row,
                assignment_codebooks[row.assignment_id][row.display_id],
            )
        )
    if not eligible:
        raise ValueError("no consented, comprehension-passing human responses")

    summaries: list[HumanConditionSummary] = []
    for condition in CONDITIONS:
        rows = [
            row for row, entry in eligible if entry["condition"] == condition
        ]
        if not rows:
            continue
        ratings = [row.rating for row in rows]
        summaries.append(
            HumanConditionSummary(
                condition=condition,
                response_count=len(rows),
                participant_count=len(
                    {row.participant_code for row in rows}
                ),
                mean_rating=mean(ratings),
                standard_deviation=(
                    None if len(ratings) < 2 else stdev(ratings)
                ),
                mean_response_time_ms=mean(
                    row.response_time_ms for row in rows
                ),
            )
        )
    ranking = tuple(
        item.condition
        for item in sorted(
            summaries,
            key=lambda item: (-item.mean_rating, item.condition),
        )
    )

    participant_condition: dict[tuple[str, str], list[float]] = {}
    for row, entry in eligible:
        participant_condition.setdefault(
            (row.participant_code, entry["condition"]), []
        ).append(float(row.rating))
    contrasts = (
        ("volunteered", "balanced"),
        ("balanced", "restricted"),
        ("balanced", "default"),
        ("balanced", "suggested"),
    )
    paired_results: list[HumanPairedContrast] = []
    participants = sorted({row.participant_code for row, _ in eligible})
    for first, second in contrasts:
        pairs = [
            (
                mean(participant_condition[(participant, first)]),
                mean(participant_condition[(participant, second)]),
            )
            for participant in participants
            if (participant, first) in participant_condition
            and (participant, second) in participant_condition
        ]
        if pairs:
            estimate, lower, upper = paired_bootstrap_mean_difference(
                [pair[0] for pair in pairs],
                [pair[1] for pair in pairs],
                replicates=bootstrap_replicates,
                seed=seed,
            )
        else:
            estimate = lower = upper = None
        paired_results.append(
            HumanPairedContrast(
                contrast_id=f"{first}-minus-{second}",
                first_condition=first,
                second_condition=second,
                paired_participant_count=len(pairs),
                mean_difference=estimate,
                bootstrap_lower=lower,
                bootstrap_upper=upper,
            )
        )
    return HumanEvidenceAnalysis(
        import_audit=audit,
        condition_summaries=tuple(summaries),
        evidence_strength_ranking=ranking,
        paired_contrasts=tuple(paired_results),
    )
