"""Development-only probability calibration."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence


@dataclass(frozen=True, slots=True)
class CalibrationExample:
    probabilities: tuple[float, ...]
    true_index: int
    split: str

    def __post_init__(self) -> None:
        if not self.probabilities or any(p < 0 for p in self.probabilities):
            raise ValueError("probabilities must be non-empty and non-negative")
        if abs(sum(self.probabilities) - 1.0) > 1e-8:
            raise ValueError("probabilities must sum to one")
        if not 0 <= self.true_index < len(self.probabilities):
            raise ValueError("true_index is outside the probability vector")
        if self.split not in {"train", "development", "test"}:
            raise ValueError("split must be train, development, or test")


@dataclass(frozen=True, slots=True)
class TemperatureCalibration:
    temperature: float
    fitted_splits: tuple[str, ...]
    example_count: int

    def apply(self, probabilities: Sequence[float]) -> tuple[float, ...]:
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if not probabilities or any(p < 0 for p in probabilities):
            raise ValueError("invalid probability vector")
        power = 1.0 / self.temperature
        adjusted = [max(float(p), 1e-15) ** power for p in probabilities]
        total = sum(adjusted)
        return tuple(p / total for p in adjusted)

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "temperature",
            "temperature": self.temperature,
            "fitted_splits": list(self.fitted_splits),
            "example_count": self.example_count,
        }


def _nll(examples: Sequence[CalibrationExample], temperature: float) -> float:
    calibrator = TemperatureCalibration(
        temperature=temperature,
        fitted_splits=(),
        example_count=len(examples),
    )
    return sum(
        -math.log(max(calibrator.apply(item.probabilities)[item.true_index], 1e-15))
        for item in examples
    ) / len(examples)


def fit_temperature(
    examples: Iterable[CalibrationExample],
    *,
    allowed_splits: tuple[str, ...] = ("development",),
) -> TemperatureCalibration:
    """Fit temperature by deterministic log-space golden-section search.

    Test records are rejected even if the caller mistakenly includes ``"test"``
    in ``allowed_splits``. This makes the paper's leakage rule executable.
    """

    if "test" in allowed_splits:
        raise ValueError("calibration may not use test labels")
    selected = tuple(item for item in examples if item.split in allowed_splits)
    if not selected:
        raise ValueError("no eligible calibration examples")
    if any(item.split == "test" for item in selected):
        raise ValueError("calibration may not use test labels")

    left, right = math.log(0.05), math.log(20.0)
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    x1 = right - ratio * (right - left)
    x2 = left + ratio * (right - left)
    f1, f2 = _nll(selected, math.exp(x1)), _nll(selected, math.exp(x2))
    for _ in range(80):
        if f1 <= f2:
            right, x2, f2 = x2, x1, f1
            x1 = right - ratio * (right - left)
            f1 = _nll(selected, math.exp(x1))
        else:
            left, x1, f1 = x1, x2, f2
            x2 = left + ratio * (right - left)
            f2 = _nll(selected, math.exp(x2))
    temperature = math.exp((left + right) / 2.0)
    return TemperatureCalibration(
        temperature=temperature,
        fitted_splits=tuple(sorted(set(allowed_splits))),
        example_count=len(selected),
    )

