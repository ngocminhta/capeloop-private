"""Simulation-based pilot power and multiplicity helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import NormalDist, mean, stdev
from typing import Mapping, Sequence

from .rng import weighted_index


@dataclass(frozen=True, slots=True)
class PowerEstimate:
    sample_size: int
    simulations: int
    alpha: float
    estimated_power: float
    pilot_effect: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "sample_size": self.sample_size,
            "simulations": self.simulations,
            "alpha": self.alpha,
            "estimated_power": self.estimated_power,
            "pilot_effect": self.pilot_effect,
        }


def paired_pilot_power(
    paired_differences: Sequence[float],
    sample_sizes: Sequence[int],
    *,
    simulations: int = 2000,
    alpha: float = 0.05,
    seed: int = 1729,
) -> tuple[PowerEstimate, ...]:
    """Bootstrap paired trajectory effects and use a two-sided normal test."""

    pilot = tuple(float(value) for value in paired_differences)
    if len(pilot) < 2:
        raise ValueError("pilot power requires at least two paired trajectories")
    if simulations <= 0:
        raise ValueError("simulations must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    if any(size < 2 for size in sample_sizes):
        raise ValueError("sample sizes must be at least two")
    threshold = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    weights = [1.0] * len(pilot)
    estimates = []
    for sample_size in sample_sizes:
        rejections = 0
        for simulation in range(simulations):
            sample = [
                pilot[
                    weighted_index(
                        weights,
                        seed,
                        "power",
                        sample_size,
                        simulation,
                        draw,
                    )
                ]
                for draw in range(sample_size)
            ]
            standard_error = stdev(sample) / math.sqrt(sample_size)
            if standard_error == 0:
                reject = mean(sample) != 0
            else:
                reject = abs(mean(sample) / standard_error) >= threshold
            rejections += int(reject)
        estimates.append(
            PowerEstimate(
                sample_size=sample_size,
                simulations=simulations,
                alpha=alpha,
                estimated_power=rejections / simulations,
                pilot_effect=mean(pilot),
            )
        )
    return tuple(estimates)


def benjamini_hochberg(
    p_values: Mapping[str, float],
    *,
    alpha: float = 0.05,
) -> dict[str, bool]:
    """Return false-discovery-rate rejection decisions for secondary tests."""

    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    if any(not 0 <= value <= 1 for value in p_values.values()):
        raise ValueError("p-values must lie in [0, 1]")
    ordered = sorted(p_values, key=lambda key: (p_values[key], key))
    largest = -1
    count = len(ordered)
    for index, key in enumerate(ordered, start=1):
        if p_values[key] <= alpha * index / max(count, 1):
            largest = index
    return {
        key: index <= largest
        for index, key in enumerate(ordered, start=1)
    }

