"""Finite latent-user populations and explicit initial profile seeds."""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from itertools import product
from itertools import combinations, permutations
import math
from typing import Sequence
from typing import Any

from .beliefs import MarginalPreferenceBelief, PreferenceBelief, THETA_STATES, THETA_VALUES
from .rng import semantic_digest
from .schemas import LatentUser, Susceptibility, Theta
from .splits import (
    BALANCED_SUSCEPTIBILITY_POLICY,
    BALANCED_THETA_POLICY,
    SplitManifest,
    orthogonal_susceptibility_group_order,
    orthogonal_theta_group_order,
    susceptibility_group_key,
    theta_group_key,
)


INITIAL_PROFILE_KINDS = ("correct", "incorrect", "uncertain", "empty")
JOINT_BALANCE_HORIZONS = (4, 8, 10, 16, 20, 24, 32)
_JOINT_BALANCE_VARIANTS = 32
_JOINT_BALANCE_REFINEMENT_STEPS = 24
_FOUR_USER_CORRELATION_GUARDRAIL = 0.72


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


def susceptibility_support_for_split(
    manifest: SplitManifest,
    *,
    split: str,
    levels: Sequence[float] = (0.15, 0.45, 0.85),
) -> tuple[Susceptibility, ...]:
    """Return the prospective susceptibility support assigned to one split."""

    if split not in {"train", "development", "test"}:
        raise ValueError("split must be train, development, or test")
    support = tuple(
        susceptibility
        for susceptibility in susceptibility_grid(levels)
        if manifest.susceptibility_groups.get(
            susceptibility_group_id(susceptibility)
        )
        == split
    )
    if not support:
        raise ValueError(
            f"manifest contains no susceptibility support for {split}"
        )
    return support


def theta_group_id(theta: Theta) -> str:
    return theta_group_key(theta)


def susceptibility_group_id(susceptibility: Susceptibility) -> str:
    return susceptibility_group_key(
        (
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


def joint_cross_balance_score(
    theta_sequence: Sequence[Theta],
    susceptibility_sequence: Sequence[Susceptibility],
    *,
    horizons: Sequence[int] = JOINT_BALANCE_HORIZONS,
) -> tuple[float, float, float, float]:
    """Score outcome-free theta-by-susceptibility association.

    The categorical component is the equally weighted sum of Cramer's V
    squared over all nine theta-coordinate by susceptibility-coordinate
    contingency tables and requested horizons.  The linear component is the
    corresponding sum of squared Pearson correlations.  Lower is better.
    """

    requested = tuple(int(horizon) for horizon in horizons)
    if not requested or any(horizon <= 1 for horizon in requested):
        raise ValueError("joint-balance horizons must be integers greater than one")
    if len(theta_sequence) < max(requested) or len(susceptibility_sequence) < max(
        requested
    ):
        raise ValueError("joint-balance sequences do not cover every horizon")
    theta_levels = tuple(
        tuple(sorted({theta[coordinate] for theta in theta_sequence}))
        for coordinate in range(3)
    )
    susceptibility_levels = tuple(
        tuple(
            sorted(
                {
                    getattr(susceptibility, coordinate)
                    for susceptibility in susceptibility_sequence
                }
            )
        )
        for coordinate in ("ranking", "default", "suggestion")
    )
    categorical_energy = 0.0
    linear_energy = 0.0
    maximum_absolute_correlation = 0.0
    for horizon in requested:
        for theta_coordinate in range(3):
            theta_values = [
                theta_sequence[index][theta_coordinate]
                for index in range(horizon)
            ]
            theta_counts = Counter(theta_values)
            for susceptibility_coordinate, susceptibility_name in enumerate(
                ("ranking", "default", "suggestion")
            ):
                susceptibility_values = [
                    getattr(
                        susceptibility_sequence[index],
                        susceptibility_name,
                    )
                    for index in range(horizon)
                ]
                susceptibility_counts = Counter(susceptibility_values)
                joint_counts = Counter(
                    zip(theta_values, susceptibility_values)
                )
                chi_square = 0.0
                for theta_value in theta_levels[theta_coordinate]:
                    for susceptibility_value in susceptibility_levels[
                        susceptibility_coordinate
                    ]:
                        expected = (
                            theta_counts[theta_value]
                            * susceptibility_counts[susceptibility_value]
                            / horizon
                        )
                        if expected > 0.0:
                            chi_square += (
                                joint_counts[
                                    (theta_value, susceptibility_value)
                                ]
                                - expected
                            ) ** 2 / expected
                categorical_energy += chi_square / (
                    horizon
                    * min(
                        len(theta_levels[theta_coordinate]) - 1,
                        len(
                            susceptibility_levels[
                                susceptibility_coordinate
                            ]
                        )
                        - 1,
                    )
                )
                theta_mean = sum(theta_values) / horizon
                susceptibility_mean = (
                    sum(susceptibility_values) / horizon
                )
                numerator = sum(
                    (theta_value - theta_mean)
                    * (susceptibility_value - susceptibility_mean)
                    for theta_value, susceptibility_value in zip(
                        theta_values,
                        susceptibility_values,
                    )
                )
                denominator = math.sqrt(
                    sum(
                        (theta_value - theta_mean) ** 2
                        for theta_value in theta_values
                    )
                    * sum(
                        (
                            susceptibility_value
                            - susceptibility_mean
                        )
                        ** 2
                        for susceptibility_value in susceptibility_values
                    )
                )
                correlation = (
                    0.0 if denominator == 0.0 else numerator / denominator
                )
                linear_energy += correlation * correlation
                maximum_absolute_correlation = max(
                    maximum_absolute_correlation,
                    abs(correlation),
                )
    categorical = round(categorical_energy, 12)
    linear = round(linear_energy, 12)
    return (
        round(categorical + linear, 12),
        categorical,
        linear,
        round(maximum_absolute_correlation, 12),
    )


def _blocked_variants(
    base: tuple[object, ...],
    *,
    block_size: int,
    cycles: int,
    seed: int,
    split: str,
    dimension: str,
) -> tuple[tuple[object, ...], ...]:
    blocks = tuple(
        base[index : index + block_size]
        for index in range(0, len(base), block_size)
    )
    baseline = base * cycles
    variants = [baseline]
    for variant in range(_JOINT_BALANCE_VARIANTS - 1):
        sequence = []
        for cycle in range(cycles):
            ordered_blocks = _stable_order(
                blocks,
                seed=seed,
                key=(
                    f"joint-balance:{split}:{dimension}:variant-{variant}:"
                    f"cycle-{cycle}:blocks"
                ),
            )
            for block_index, block in enumerate(ordered_blocks):
                sequence.extend(
                    _stable_order(
                        block,
                        seed=seed,
                        key=(
                            f"joint-balance:{split}:{dimension}:"
                            f"variant-{variant}:cycle-{cycle}:"
                            f"block-{block_index}:within"
                        ),
                    )
                )
        candidate = tuple(sequence)
        if candidate not in variants:
            variants.append(candidate)
    return tuple(variants)


def _blocked_neighbors(
    sequence: tuple[object, ...],
    *,
    block_size: int,
    blocks_per_cycle: int,
) -> tuple[tuple[object, ...], ...]:
    blocks = [
        sequence[index : index + block_size]
        for index in range(0, len(sequence), block_size)
    ]
    neighbors = []
    for block_index, block in enumerate(blocks):
        for reordered in permutations(block):
            if reordered == block:
                continue
            candidate = list(blocks)
            candidate[block_index] = reordered
            neighbors.append(
                tuple(item for candidate_block in candidate for item in candidate_block)
            )
    for cycle_start in range(0, len(blocks), blocks_per_cycle):
        cycle_stop = min(cycle_start + blocks_per_cycle, len(blocks))
        for first, second in combinations(range(cycle_start, cycle_stop), 2):
            candidate = list(blocks)
            candidate[first], candidate[second] = (
                candidate[second],
                candidate[first],
            )
            neighbors.append(
                tuple(item for candidate_block in candidate for item in candidate_block)
            )
    return tuple(neighbors)


@lru_cache(maxsize=64)
def _balanced_joint_orders(
    theta_order: tuple[Theta, ...],
    susceptibility_order: tuple[Susceptibility, ...],
    *,
    seed: int,
    split: str,
) -> tuple[tuple[Theta, ...], tuple[Susceptibility, ...]]:
    """Choose cached deterministic blocks with low cross-factor association."""

    maximum_horizon = max(JOINT_BALANCE_HORIZONS)
    theta_cycles = math.ceil(maximum_horizon / len(theta_order))
    susceptibility_cycles = math.ceil(
        maximum_horizon / len(susceptibility_order)
    )
    theta_variants = _blocked_variants(
        theta_order,
        block_size=4,
        cycles=theta_cycles,
        seed=seed,
        split=split,
        dimension="theta",
    )
    susceptibility_variants = _blocked_variants(
        susceptibility_order,
        block_size=3,
        cycles=susceptibility_cycles,
        seed=seed,
        split=split,
        dimension="susceptibility",
    )

    def candidate_key(
        theta_sequence: tuple[Theta, ...],
        susceptibility_sequence: tuple[Susceptibility, ...],
    ) -> tuple[object, ...]:
        score = joint_cross_balance_score(
            theta_sequence,
            susceptibility_sequence,
        )
        four_user_correlation = joint_cross_balance_score(
            theta_sequence,
            susceptibility_sequence,
            horizons=(4,),
        )[3]
        return (
            round(
                max(
                    0.0,
                    four_user_correlation
                    - _FOUR_USER_CORRELATION_GUARDRAIL,
                ),
                12,
            ),
            *score,
            semantic_digest(
                seed,
                "joint-balance-tie",
                split,
                theta_sequence,
                susceptibility_sequence,
            ),
        )

    best_theta = theta_variants[0]
    best_susceptibility = susceptibility_variants[0]
    best_key = candidate_key(best_theta, best_susceptibility)
    for theta_candidate in theta_variants:
        for susceptibility_candidate in susceptibility_variants:
            key = candidate_key(theta_candidate, susceptibility_candidate)
            if key < best_key:
                best_key = key
                best_theta = theta_candidate  # type: ignore[assignment]
                best_susceptibility = susceptibility_candidate  # type: ignore[assignment]

    theta_blocks_per_cycle = len(theta_order) // 4
    susceptibility_blocks_per_cycle = len(susceptibility_order) // 3
    for _ in range(_JOINT_BALANCE_REFINEMENT_STEPS):
        improved_theta = best_theta
        improved_susceptibility = best_susceptibility
        improved_key = best_key
        for theta_candidate in _blocked_neighbors(
            best_theta,
            block_size=4,
            blocks_per_cycle=theta_blocks_per_cycle,
        ):
            key = candidate_key(
                theta_candidate,  # type: ignore[arg-type]
                best_susceptibility,
            )
            if key < improved_key:
                improved_key = key
                improved_theta = theta_candidate  # type: ignore[assignment]
        for susceptibility_candidate in _blocked_neighbors(
            best_susceptibility,
            block_size=3,
            blocks_per_cycle=susceptibility_blocks_per_cycle,
        ):
            key = candidate_key(
                best_theta,
                susceptibility_candidate,  # type: ignore[arg-type]
            )
            if key < improved_key:
                improved_key = key
                improved_theta = best_theta
                improved_susceptibility = susceptibility_candidate  # type: ignore[assignment]
        if improved_key >= best_key:
            break
        best_key = improved_key
        best_theta = improved_theta
        best_susceptibility = improved_susceptibility
    return best_theta, best_susceptibility


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
    psi_candidates = list(
        susceptibility_support_for_split(
            manifest,
            split=split,
            levels=susceptibility_levels,
        )
    )
    if not theta_candidates or not psi_candidates:
        raise ValueError(f"manifest contains no complete latent support for {split}")
    balanced_theta = manifest.theta_policy == BALANCED_THETA_POLICY
    balanced_susceptibility = (
        manifest.susceptibility_policy == BALANCED_SUSCEPTIBILITY_POLICY
    )
    if balanced_theta or balanced_susceptibility:
        if seed != manifest.seed:
            raise ValueError(
                "balanced population allocation requires the generation seed "
                "to match the split-manifest seed"
            )
        if balanced_theta:
            theta_candidates_by_group = {
                theta_group_id(theta): theta for theta in theta_candidates
            }
            ordered_theta_groups = orthogonal_theta_group_order(
                THETA_VALUES,
                seed=manifest.seed,
                split=split,
            )
            if set(ordered_theta_groups) != set(theta_candidates_by_group):
                raise ValueError(
                    "balanced theta manifest does not match the declared "
                    "theta values and split"
                )
            theta_order = tuple(
                theta_candidates_by_group[group]
                for group in ordered_theta_groups
            )
        if balanced_susceptibility:
            psi_candidates_by_group = {
                susceptibility_group_id(psi): psi for psi in psi_candidates
            }
            ordered_psi_groups = orthogonal_susceptibility_group_order(
                susceptibility_levels,
                seed=manifest.seed,
                split=split,
            )
            if set(ordered_psi_groups) != set(psi_candidates_by_group):
                raise ValueError(
                    "balanced susceptibility manifest does not match the "
                    "declared levels and split"
                )
            psi_order = tuple(
                psi_candidates_by_group[group]
                for group in ordered_psi_groups
            )
        if balanced_theta and balanced_susceptibility:
            joint_theta_order, joint_psi_order = _balanced_joint_orders(
                theta_order,
                psi_order,
                seed=seed,
                split=split,
            )
            return tuple(
                LatentUser(
                    user_id=f"{domain_id}-{split}-user-{index:05d}",
                    theta=joint_theta_order[
                        index % len(joint_theta_order)
                    ],
                    susceptibility=joint_psi_order[
                        index % len(joint_psi_order)
                    ],
                )
                for index in range(count)
            )
        if balanced_susceptibility:
            theta_orders = {
                group: _stable_order(
                    theta_candidates,
                    seed=seed,
                    key=f"{domain_id}:{split}:theta-for:{group}",
                )
                for group in ordered_psi_groups
            }
            users = []
            for index in range(count):
                profile_index = index % len(psi_order)
                profile_cycle = index // len(psi_order)
                psi = psi_order[profile_index]
                group = ordered_psi_groups[profile_index]
                per_profile_theta_order = theta_orders[group]
                theta = per_profile_theta_order[
                    profile_cycle % len(per_profile_theta_order)
                ]
                users.append(
                    LatentUser(
                        user_id=f"{domain_id}-{split}-user-{index:05d}",
                        theta=theta,  # type: ignore[arg-type]
                        susceptibility=psi,
                    )
                )
            return tuple(users)
        psi_orders = {
            group: _stable_order(
                psi_candidates,
                seed=seed,
                key=f"{domain_id}:{split}:psi-for:{group}",
            )
            for group in ordered_theta_groups
        }
        users = []
        for index in range(count):
            profile_index = index % len(theta_order)
            profile_cycle = index // len(theta_order)
            theta = theta_order[profile_index]
            group = ordered_theta_groups[profile_index]
            per_profile_psi_order = psi_orders[group]
            psi = per_profile_psi_order[
                profile_cycle % len(per_profile_psi_order)
            ]
            users.append(
                LatentUser(
                    user_id=f"{domain_id}-{split}-user-{index:05d}",
                    theta=theta,
                    susceptibility=psi,  # type: ignore[arg-type]
                )
            )
        return tuple(users)

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
