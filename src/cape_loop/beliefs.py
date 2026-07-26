"""Canonical normalized beliefs over CAPE-Loop latent state spaces."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
from typing import Any

from .schemas import (
    NUM_ATTRIBUTES,
    THETA_VALUES,
    Susceptibility,
    Theta,
    validate_theta,
)


THETA_STATES: tuple[Theta, ...] = tuple(
    (values[0], values[1], values[2])
    for values in product(THETA_VALUES, repeat=NUM_ATTRIBUTES)
)
THETA_INDEX: dict[Theta, int] = {
    theta: index for index, theta in enumerate(THETA_STATES)
}


def normalize_weights(weights: tuple[float, ...] | list[float]) -> tuple[float, ...]:
    """Normalize non-negative finite weights into a probability vector."""

    if not weights:
        raise ValueError("weights cannot be empty")
    numeric: list[float] = []
    for index, weight in enumerate(weights):
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise TypeError(f"weights[{index}] must be numeric")
        value = float(weight)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("weights must be finite and non-negative")
        numeric.append(value)
    total = math.fsum(numeric)
    if total <= 0.0:
        raise ValueError("at least one weight must be positive")
    return tuple(value / total for value in numeric)


def _validate_probability_vector(
    probabilities: tuple[float, ...],
    expected_length: int,
    name: str,
) -> tuple[float, ...]:
    values = tuple(probabilities)
    if len(values) != expected_length:
        raise ValueError(f"{name} must contain {expected_length} probabilities")
    normalize_weights(list(values))
    if not math.isclose(
        math.fsum(float(value) for value in values),
        1.0,
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        raise ValueError(f"{name} must sum to one")
    # Preserve exact input semantics while canonicalizing ints to floats.
    return tuple(float(value) for value in values)


def _entropy(probabilities: tuple[float, ...]) -> float:
    return -math.fsum(
        probability * math.log(probability)
        for probability in probabilities
        if probability > 0.0
    )


@dataclass(frozen=True, slots=True)
class MarginalPreferenceBelief:
    """Three independently represented four-class preference marginals."""

    probabilities: tuple[
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
    ]

    def __post_init__(self) -> None:
        rows = tuple(tuple(row) for row in self.probabilities)
        if len(rows) != NUM_ATTRIBUTES:
            raise ValueError(f"expected {NUM_ATTRIBUTES} marginal rows")
        validated = tuple(
            _validate_probability_vector(
                row,
                len(THETA_VALUES),
                f"probabilities[{attribute}]",
            )
            for attribute, row in enumerate(rows)
        )
        object.__setattr__(self, "probabilities", validated)

    @classmethod
    def uniform(cls) -> MarginalPreferenceBelief:
        row = tuple(1.0 / len(THETA_VALUES) for _ in THETA_VALUES)
        return cls((row, row, row))

    @classmethod
    def from_weights(
        cls,
        rows: tuple[
            tuple[float, ...],
            tuple[float, ...],
            tuple[float, ...],
        ],
    ) -> MarginalPreferenceBelief:
        normalized = tuple(normalize_weights(list(row)) for row in rows)
        return cls(normalized)  # type: ignore[arg-type]

    def marginal(self, attribute: int) -> tuple[float, float, float, float]:
        if not 0 <= attribute < NUM_ATTRIBUTES:
            raise ValueError(f"attribute must be in [0, {NUM_ATTRIBUTES})")
        return self.probabilities[attribute]

    def sign_mass(self, attribute: int, direction: int) -> float:
        if direction not in (-1, 1):
            raise ValueError("direction must be -1 or +1")
        row = self.marginal(attribute)
        indexes = (0, 1) if direction < 0 else (2, 3)
        return row[indexes[0]] + row[indexes[1]]

    def expected_theta(self) -> tuple[float, float, float]:
        values = tuple(
            math.fsum(
                probability * theta_value
                for probability, theta_value in zip(row, THETA_VALUES)
            )
            for row in self.probabilities
        )
        return (values[0], values[1], values[2])

    def entropy(self) -> float:
        return math.fsum(_entropy(row) for row in self.probabilities)

    def independent_joint(self) -> PreferenceBelief:
        weights = tuple(
            math.prod(
                self.probabilities[attribute][THETA_VALUES.index(theta[attribute])]
                for attribute in range(NUM_ATTRIBUTES)
            )
            for theta in THETA_STATES
        )
        return PreferenceBelief(weights)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "marginal",
            "values": list(THETA_VALUES),
            "probabilities": [list(row) for row in self.probabilities],
        }


# Short public name.
MarginalBelief = MarginalPreferenceBelief


@dataclass(frozen=True, slots=True)
class PreferenceBelief:
    """A full joint distribution over the 64 theta states."""

    probabilities: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "probabilities",
            _validate_probability_vector(
                tuple(self.probabilities),
                len(THETA_STATES),
                "probabilities",
            ),
        )

    @classmethod
    def uniform(cls) -> PreferenceBelief:
        probability = 1.0 / len(THETA_STATES)
        return cls(tuple(probability for _ in THETA_STATES))

    @classmethod
    def from_weights(
        cls,
        weights: tuple[float, ...] | list[float],
    ) -> PreferenceBelief:
        if len(weights) != len(THETA_STATES):
            raise ValueError(f"weights must contain {len(THETA_STATES)} entries")
        return cls(normalize_weights(weights))

    @classmethod
    def point_mass(cls, theta: Theta) -> PreferenceBelief:
        canonical = validate_theta(theta)
        weights = [0.0 for _ in THETA_STATES]
        weights[THETA_INDEX[canonical]] = 1.0
        return cls(tuple(weights))

    @classmethod
    def from_marginals(
        cls,
        marginals: MarginalPreferenceBelief,
    ) -> PreferenceBelief:
        return marginals.independent_joint()

    def probability(self, theta: Theta) -> float:
        return self.probabilities[THETA_INDEX[validate_theta(theta)]]

    def marginal(self, attribute: int) -> tuple[float, float, float, float]:
        if not 0 <= attribute < NUM_ATTRIBUTES:
            raise ValueError(f"attribute must be in [0, {NUM_ATTRIBUTES})")
        row = []
        for value in THETA_VALUES:
            row.append(
                math.fsum(
                    probability
                    for theta, probability in zip(
                        THETA_STATES, self.probabilities
                    )
                    if theta[attribute] == value
                )
            )
        return (row[0], row[1], row[2], row[3])

    def marginals(self) -> MarginalPreferenceBelief:
        rows = tuple(self.marginal(attribute) for attribute in range(NUM_ATTRIBUTES))
        return MarginalPreferenceBelief(rows)  # type: ignore[arg-type]

    def sign_mass(self, attribute: int, direction: int) -> float:
        return self.marginals().sign_mass(attribute, direction)

    def expected_theta(self) -> tuple[float, float, float]:
        values = tuple(
            math.fsum(
                probability * theta[attribute]
                for theta, probability in zip(THETA_STATES, self.probabilities)
            )
            for attribute in range(NUM_ATTRIBUTES)
        )
        return (values[0], values[1], values[2])

    def entropy(self) -> float:
        return _entropy(self.probabilities)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "theta_joint",
            "theta_states": [list(theta) for theta in THETA_STATES],
            "probabilities": list(self.probabilities),
            "marginals": self.marginals().to_dict()["probabilities"],
        }


ThetaBelief = PreferenceBelief


@dataclass(frozen=True, slots=True)
class JointThetaPsiBelief:
    """A theta-major joint posterior over theta and susceptibility support."""

    susceptibilities: tuple[Susceptibility, ...]
    probabilities: tuple[float, ...]

    def __post_init__(self) -> None:
        susceptibilities = tuple(self.susceptibilities)
        if not susceptibilities:
            raise ValueError("susceptibility support cannot be empty")
        if not all(
            isinstance(susceptibility, Susceptibility)
            for susceptibility in susceptibilities
        ):
            raise TypeError("susceptibilities must contain Susceptibility objects")
        if len(set(susceptibilities)) != len(susceptibilities):
            raise ValueError("susceptibility support must be unique")
        object.__setattr__(self, "susceptibilities", susceptibilities)

        expected = len(THETA_STATES) * len(susceptibilities)
        object.__setattr__(
            self,
            "probabilities",
            _validate_probability_vector(
                tuple(self.probabilities),
                expected,
                "probabilities",
            ),
        )

    @classmethod
    def uniform(
        cls,
        susceptibilities: tuple[Susceptibility, ...],
    ) -> JointThetaPsiBelief:
        support_size = len(THETA_STATES) * len(susceptibilities)
        if support_size == 0:
            raise ValueError("susceptibility support cannot be empty")
        return cls(
            susceptibilities,
            tuple(1.0 / support_size for _ in range(support_size)),
        )

    @classmethod
    def from_independent(
        cls,
        theta_belief: PreferenceBelief,
        susceptibilities: tuple[Susceptibility, ...],
        susceptibility_weights: tuple[float, ...] | None = None,
    ) -> JointThetaPsiBelief:
        if susceptibility_weights is None:
            susceptibility_weights = tuple(1.0 for _ in susceptibilities)
        if len(susceptibility_weights) != len(susceptibilities):
            raise ValueError("one susceptibility weight is required per support value")
        psi_probabilities = normalize_weights(list(susceptibility_weights))
        joint = tuple(
            theta_probability * psi_probability
            for theta_probability in theta_belief.probabilities
            for psi_probability in psi_probabilities
        )
        return cls(tuple(susceptibilities), joint)

    @classmethod
    def from_weights(
        cls,
        susceptibilities: tuple[Susceptibility, ...],
        weights: tuple[float, ...] | list[float],
    ) -> JointThetaPsiBelief:
        expected = len(THETA_STATES) * len(susceptibilities)
        if len(weights) != expected:
            raise ValueError(f"weights must contain {expected} entries")
        return cls(tuple(susceptibilities), normalize_weights(weights))

    def flat_index(self, theta: Theta, susceptibility: Susceptibility) -> int:
        theta_index = THETA_INDEX[validate_theta(theta)]
        try:
            psi_index = self.susceptibilities.index(susceptibility)
        except ValueError as exc:
            raise KeyError(susceptibility) from exc
        return theta_index * len(self.susceptibilities) + psi_index

    def probability(self, theta: Theta, susceptibility: Susceptibility) -> float:
        return self.probabilities[self.flat_index(theta, susceptibility)]

    def theta_belief(self) -> PreferenceBelief:
        psi_count = len(self.susceptibilities)
        theta_probabilities = tuple(
            math.fsum(
                self.probabilities[
                    theta_index * psi_count : (theta_index + 1) * psi_count
                ]
            )
            for theta_index in range(len(THETA_STATES))
        )
        return PreferenceBelief(theta_probabilities)

    def susceptibility_marginal(self) -> tuple[float, ...]:
        psi_count = len(self.susceptibilities)
        return tuple(
            math.fsum(
                self.probabilities[theta_index * psi_count + psi_index]
                for theta_index in range(len(THETA_STATES))
            )
            for psi_index in range(psi_count)
        )

    def marginal(self, attribute: int) -> tuple[float, float, float, float]:
        return self.theta_belief().marginal(attribute)

    def marginals(self) -> MarginalPreferenceBelief:
        return self.theta_belief().marginals()

    def sign_mass(self, attribute: int, direction: int) -> float:
        return self.theta_belief().sign_mass(attribute, direction)

    def entropy(self) -> float:
        return _entropy(self.probabilities)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "theta_psi_joint",
            "theta_states": [list(theta) for theta in THETA_STATES],
            "susceptibilities": [
                susceptibility.to_dict()
                for susceptibility in self.susceptibilities
            ],
            "probabilities": list(self.probabilities),
            "theta_marginals": self.marginals().to_dict()["probabilities"],
            "susceptibility_marginal": list(self.susceptibility_marginal()),
        }


ThetaPsiBelief = JointThetaPsiBelief
