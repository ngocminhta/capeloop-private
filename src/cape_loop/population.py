"""Finite latent-user populations and explicit initial profile seeds."""

from __future__ import annotations

from itertools import product
from typing import Sequence
from typing import Any

from .beliefs import MarginalPreferenceBelief, PreferenceBelief, THETA_STATES, THETA_VALUES
from .rng import semantic_digest
from .schemas import LatentUser, Susceptibility, Theta
from .splits import SplitManifest


INITIAL_PROFILE_KINDS = ("correct", "incorrect", "uncertain", "empty")


def susceptibility_grid(
    levels: Sequence[float] = (0.15, 0.45, 0.85),
) -> tuple[Susceptibility, ...]:
    numeric = tuple(float(level) for level in levels)
    if not numeric or any(level < 0 for level in numeric):
        raise ValueError("susceptibility levels must be non-empty and non-negative")
    return tuple(
        Susceptibility(ranking, default, suggestion)
        for ranking, default, suggestion in product(numeric, repeat=3)
    )


def theta_group_id(theta: Theta) -> str:
    return ",".join(str(value) for value in theta)


def susceptibility_group_id(susceptibility: Susceptibility) -> str:
    return ",".join(
        f"{value:.8g}"
        for value in (
            susceptibility.ranking,
            susceptibility.default,
            susceptibility.suggestion,
        )
    )


def _stable_order(values: Sequence[object], *, seed: int, key: str) -> list[object]:
    return sorted(
        values,
        key=lambda value: semantic_digest(seed, "population-order", key, value),
    )


def generate_users(
    *,
    domain_id: str,
    count: int,
    split: str,
    manifest: SplitManifest,
    susceptibility_levels: Sequence[float] = (0.15, 0.45, 0.85),
    seed: int = 1729,
) -> tuple[LatentUser, ...]:
    """Generate users only from complete theta/psi groups assigned to ``split``."""

    if count <= 0:
        raise ValueError("count must be positive")
    if split not in {"train", "development", "test"}:
        raise ValueError("split must be train, development, or test")
    theta_candidates = [
        theta
        for theta in THETA_STATES
        if manifest.theta_groups.get(theta_group_id(theta)) == split
    ]
    psi_candidates = [
        psi
        for psi in susceptibility_grid(susceptibility_levels)
        if manifest.susceptibility_groups.get(susceptibility_group_id(psi)) == split
    ]
    if not theta_candidates or not psi_candidates:
        raise ValueError(f"manifest contains no complete latent support for {split}")
    theta_order = _stable_order(
        theta_candidates, seed=seed, key=f"{domain_id}:{split}:theta"
    )
    psi_order = _stable_order(
        psi_candidates, seed=seed, key=f"{domain_id}:{split}:psi"
    )
    combinations = [
        (theta, psi)
        for theta in theta_order
        for psi in psi_order
    ]
    combinations = _stable_order(
        combinations, seed=seed, key=f"{domain_id}:{split}:joint"
    )
    users = []
    for index in range(count):
        theta, psi = combinations[index % len(combinations)]
        users.append(
            LatentUser(
                user_id=f"{domain_id}-{split}-user-{index:05d}",
                theta=theta,  # type: ignore[arg-type]
                susceptibility=psi,  # type: ignore[arg-type]
            )
        )
    return tuple(users)


def user_state_record(
    user: LatentUser, *, domain_id: str, split: str
) -> dict[str, Any]:
    """Return the versioned release record described by the public schema."""

    if not domain_id:
        raise ValueError("domain_id must not be empty")
    if split not in {"train", "development", "test"}:
        raise ValueError("unknown data split")
    return {
        "schema_version": 1,
        "user_id": user.user_id,
        "domain": domain_id,
        "theta": list(user.theta),
        "susceptibility": user.susceptibility.to_dict(),
        "split": split,
    }


def _seed_row(
    truth_value: int,
    *,
    direction: int,
    sign_mass: float,
) -> tuple[float, float, float, float]:
    if direction not in (-1, 1):
        raise ValueError("direction must be -1 or +1")
    if not 0.5 <= sign_mass < 1:
        raise ValueError("seed sign mass must lie in [0.5, 1)")
    signed_values = [value for value in THETA_VALUES if (value > 0) == (direction > 0)]
    opposite_values = [value for value in THETA_VALUES if value not in signed_values]
    target_value = abs(truth_value) * direction
    result = {value: (1.0 - sign_mass) / len(opposite_values) for value in opposite_values}
    for value in signed_values:
        result[value] = sign_mass * (0.70 if value == target_value else 0.30)
    return tuple(result[value] for value in THETA_VALUES)  # type: ignore[return-value]


def initial_profile_belief(
    truth: Theta,
    kind: str,
    *,
    profile_strength: float = 0.80,
) -> PreferenceBelief:
    """Create one of the proposal's four declared structured initial states.

    ``empty`` and ``uncertain`` are both uncommitted concepts but are distinct:
    empty is exactly uniform, while uncertain carries weak truth-aligned mass.
    Native memory represents empty with no entries.
    """

    if kind not in INITIAL_PROFILE_KINDS:
        raise ValueError(f"unknown initial profile kind: {kind}")
    if kind == "empty":
        return PreferenceBelief.uniform()
    if kind == "uncertain":
        sign_mass = 0.55
    else:
        sign_mass = profile_strength
    rows = []
    for value in truth:
        true_direction = -1 if value < 0 else 1
        direction = -true_direction if kind == "incorrect" else true_direction
        rows.append(
            _seed_row(value, direction=direction, sign_mass=sign_mass)
        )
    marginals = MarginalPreferenceBelief((rows[0], rows[1], rows[2]))
    return PreferenceBelief.from_marginals(marginals)


def add_prior_uncertainty(
    belief: PreferenceBelief,
    uncertainty: float,
) -> PreferenceBelief:
    """Mix a declared profile prior with the uniform prior.

    ``uncertainty`` is the share of total prior mass contributed by a uniform
    distribution. It is distinct from ``profile_strength``: the latter defines
    the signed content of a seed, while this parameter controls how strongly
    that seed is trusted before any interaction.
    """

    if (
        isinstance(uncertainty, bool)
        or not isinstance(uncertainty, (int, float))
        or not 0.0 <= float(uncertainty) < 1.0
    ):
        raise ValueError("uncertainty must lie in [0, 1)")
    mixture = float(uncertainty)
    if mixture == 0.0:
        return belief
    uniform = 1.0 / len(belief.probabilities)
    return PreferenceBelief(
        tuple(
            (1.0 - mixture) * probability + mixture * uniform
            for probability in belief.probabilities
        )
    )


def wrong_directions(truth: Theta) -> tuple[int, int, int]:
    return tuple(-1 if value > 0 else 1 for value in truth)  # type: ignore[return-value]
