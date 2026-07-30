"""Leakage-resistant split manifests."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from itertools import product
from typing import Any, Iterable, Mapping, Sequence


SPLITS = ("train", "development", "test")
LEGACY_THETA_POLICY = "legacy-hash-v1"
BALANCED_THETA_POLICY = "orthogonal-balanced-v2"
THETA_POLICIES = frozenset(
    {
        LEGACY_THETA_POLICY,
        BALANCED_THETA_POLICY,
    }
)
LEGACY_SUSCEPTIBILITY_POLICY = "legacy-hash-v1"
BALANCED_SUSCEPTIBILITY_POLICY = "orthogonal-balanced-v2"
SUSCEPTIBILITY_POLICIES = frozenset(
    {
        LEGACY_SUSCEPTIBILITY_POLICY,
        BALANCED_SUSCEPTIBILITY_POLICY,
    }
)


def _bucket(key: object, seed: int) -> str:
    digest = sha256(f"{seed}\0{key!r}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big")
    # 60/20/20 split. Assignment is by complete group, never by record.
    percentile = value % 10
    if percentile < 6:
        return "train"
    if percentile < 8:
        return "development"
    return "test"


def _force_nonempty(
    assignments: dict[str, str], ordered_keys: Sequence[str]
) -> dict[str, str]:
    result = dict(assignments)
    present = set(result.values())
    for index, split in enumerate(SPLITS):
        if split not in present and ordered_keys:
            result[ordered_keys[index % len(ordered_keys)]] = split
            present.add(split)
    return result


def susceptibility_group_key(values: Sequence[float]) -> str:
    """Return the stable identifier for one complete susceptibility tuple."""

    return ",".join(f"{float(value):.8g}" for value in values)


def theta_group_key(values: Sequence[int]) -> str:
    """Return the stable identifier for one complete preference tuple."""

    return ",".join(str(int(value)) for value in values)


def _seeded_order(
    values: Sequence[Any],
    *,
    seed: int,
    namespace: str,
) -> tuple[Any, ...]:
    return tuple(
        sorted(
            values,
            key=lambda value: sha256(
                f"{seed}\0{namespace}\0{value!r}".encode("utf-8")
            ).digest(),
        )
    )


_GF4_MULTIPLICATION = (
    (0, 0, 0, 0),
    (0, 1, 2, 3),
    (0, 2, 3, 1),
    (0, 3, 1, 2),
)


def _gf4_multiply(left: int, right: int) -> int:
    return _GF4_MULTIPLICATION[left][right]


def _orthogonal_theta_rows(
    theta_values: Sequence[int],
    *,
    seed: int,
) -> tuple[tuple[str, str, int, int, int], ...]:
    """Return group, split, OA bucket, block, and within-block theta codes.

    Codes are elements of GF(4), with addition represented by XOR.  Holding
    ``third + first + second`` fixed gives a 16-cell strength-two orthogonal
    array.  Two seed-relabelled arrays go to training and one each to
    development and test, preserving complete-profile split disjointness.
    """

    values = tuple(int(value) for value in theta_values)
    if len(values) != 4 or len(set(values)) != 4:
        raise ValueError(
            f"{BALANCED_THETA_POLICY} requires exactly four unique theta "
            "values"
        )
    coordinate_codes: list[dict[int, int]] = []
    for coordinate in ("first", "second", "third"):
        ordered_indices = _seeded_order(
            tuple(range(4)),
            seed=seed,
            namespace=f"theta-v2:{coordinate}",
        )
        coordinate_codes.append(
            {
                source_index: code
                for code, source_index in enumerate(ordered_indices)
            }
        )
    bucket_order = _seeded_order(
        tuple(range(4)),
        seed=seed,
        namespace="theta-v2:buckets",
    )
    bucket_splits = {
        bucket: (
            "train"
            if position < 2
            else "development"
            if position == 2
            else "test"
        )
        for position, bucket in enumerate(bucket_order)
    }
    rows = []
    for first_index, second_index, third_index in product(
        range(4), repeat=3
    ):
        first_code = coordinate_codes[0][first_index]
        second_code = coordinate_codes[1][second_index]
        third_code = coordinate_codes[2][third_index]
        bucket = third_code ^ first_code ^ second_code
        # For a fixed intercept, second = omega*first + intercept.
        # Both omega and 1+omega are nonzero in GF(4), so every coordinate
        # traverses all four levels exactly once within each block.
        block = second_code ^ _gf4_multiply(2, first_code)
        rows.append(
            (
                theta_group_key(
                    (
                        values[first_index],
                        values[second_index],
                        values[third_index],
                    )
                ),
                bucket_splits[bucket],
                bucket,
                block,
                first_code,
            )
        )
    return tuple(rows)


def orthogonal_theta_group_order(
    theta_values: Sequence[int],
    *,
    seed: int,
    split: str,
) -> tuple[str, ...]:
    """Return deterministic four-profile balanced blocks for one v2 split."""

    if split not in SPLITS:
        raise ValueError("split must be train, development, or test")
    rows = [
        row
        for row in _orthogonal_theta_rows(theta_values, seed=seed)
        if row[1] == split
    ]
    block_ids = tuple(sorted({(row[2], row[3]) for row in rows}))
    block_order = _seeded_order(
        block_ids,
        seed=seed,
        namespace=f"theta-v2:{split}:blocks",
    )
    block_positions = {
        block_id: position for position, block_id in enumerate(block_order)
    }
    within_positions = {
        block_id: {
            code: position
            for position, code in enumerate(
                _seeded_order(
                    tuple(range(4)),
                    seed=seed,
                    namespace=(
                        f"theta-v2:{split}:bucket-{block_id[0]}:"
                        f"block-{block_id[1]}:within"
                    ),
                )
            )
        }
        for block_id in block_ids
    }
    rows.sort(
        key=lambda row: (
            block_positions[(row[2], row[3])],
            within_positions[(row[2], row[3])][row[4]],
        )
    )
    return tuple(row[0] for row in rows)


def _orthogonal_susceptibility_rows(
    susceptibility_levels: Sequence[float],
    *,
    seed: int,
) -> tuple[tuple[str, str, int, int], ...]:
    """Return group, split, block, and within-block codes for the v2 design.

    The three-way grid is partitioned by ``suggestion - ranking - default``
    modulo three after independent seed-derived relabelings.  Each split is
    therefore a nine-cell strength-two orthogonal array: every coordinate
    level occurs three times and every pair of coordinate levels occurs once.
    """

    levels = tuple(float(level) for level in susceptibility_levels)
    if len(levels) != 3 or len(set(levels)) != 3:
        raise ValueError(
            f"{BALANCED_SUSCEPTIBILITY_POLICY} requires exactly three "
            "unique susceptibility levels"
        )
    coordinate_codes: list[dict[int, int]] = []
    for coordinate in ("ranking", "default", "suggestion"):
        ordered_indices = _seeded_order(
            tuple(range(3)),
            seed=seed,
            namespace=f"susceptibility-v2:{coordinate}",
        )
        coordinate_codes.append(
            {
                source_index: code
                for code, source_index in enumerate(ordered_indices)
            }
        )
    split_order = _seeded_order(
        SPLITS,
        seed=seed,
        namespace="susceptibility-v2:splits",
    )
    rows = []
    for ranking_index, default_index, suggestion_index in product(
        range(3), repeat=3
    ):
        ranking_code = coordinate_codes[0][ranking_index]
        default_code = coordinate_codes[1][default_index]
        suggestion_code = coordinate_codes[2][suggestion_index]
        bucket = (suggestion_code - ranking_code - default_code) % 3
        rows.append(
            (
                susceptibility_group_key(
                    (
                        levels[ranking_index],
                        levels[default_index],
                        levels[suggestion_index],
                    )
                ),
                split_order[bucket],
                (default_code - ranking_code) % 3,
                ranking_code,
            )
        )
    return tuple(rows)


def orthogonal_susceptibility_group_order(
    susceptibility_levels: Sequence[float],
    *,
    seed: int,
    split: str,
) -> tuple[str, ...]:
    """Return a deterministic marginally balanced order for one v2 split."""

    if split not in SPLITS:
        raise ValueError("split must be train, development, or test")
    rows = [
        row
        for row in _orthogonal_susceptibility_rows(
            susceptibility_levels,
            seed=seed,
        )
        if row[1] == split
    ]
    block_order = _seeded_order(
        tuple(range(3)),
        seed=seed,
        namespace=f"susceptibility-v2:{split}:blocks",
    )
    block_positions = {
        block: position for position, block in enumerate(block_order)
    }
    within_positions = {
        block: {
            code: position
            for position, code in enumerate(
                _seeded_order(
                    tuple(range(3)),
                    seed=seed,
                    namespace=(
                        f"susceptibility-v2:{split}:block-{block}:within"
                    ),
                )
            )
        }
        for block in range(3)
    }
    rows.sort(
        key=lambda row: (
            block_positions[row[2]],
            within_positions[row[2]][row[3]],
        )
    )
    return tuple(row[0] for row in rows)


@dataclass(frozen=True, slots=True)
class SplitManifest:
    seed: int
    theta_groups: Mapping[str, str]
    susceptibility_groups: Mapping[str, str]
    option_templates: Mapping[str, str]
    dialogue_templates: Mapping[str, str]
    scenario_families: Mapping[str, str]
    paraphrase_templates: Mapping[str, str]
    theta_policy: str = LEGACY_THETA_POLICY
    susceptibility_policy: str = LEGACY_SUSCEPTIBILITY_POLICY

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("split seed must be non-negative")
        if self.theta_policy not in THETA_POLICIES:
            raise ValueError(f"unknown theta policy: {self.theta_policy}")
        if self.susceptibility_policy not in SUSCEPTIBILITY_POLICIES:
            raise ValueError(
                "unknown susceptibility policy: "
                f"{self.susceptibility_policy}"
            )
        for name, mapping in self.group_maps().items():
            if not mapping:
                raise ValueError(f"{name} split mapping is empty")
            if set(mapping.values()) - set(SPLITS):
                raise ValueError(f"{name} contains an unknown split")

    def group_maps(self) -> dict[str, Mapping[str, str]]:
        return {
            "theta_groups": self.theta_groups,
            "susceptibility_groups": self.susceptibility_groups,
            "option_templates": self.option_templates,
            "dialogue_templates": self.dialogue_templates,
            "scenario_families": self.scenario_families,
            "paraphrase_templates": self.paraphrase_templates,
        }

    def to_dict(self) -> dict[str, Any]:
        result = {
            "schema_version": (
                1
                if (
                    self.theta_policy == LEGACY_THETA_POLICY
                    and self.susceptibility_policy
                    == LEGACY_SUSCEPTIBILITY_POLICY
                )
                else 2
            ),
            "seed": self.seed,
            **{
                name: dict(sorted(mapping.items()))
                for name, mapping in self.group_maps().items()
            },
        }
        if self.theta_policy != LEGACY_THETA_POLICY:
            result["theta_policy"] = self.theta_policy
        if self.susceptibility_policy != LEGACY_SUSCEPTIBILITY_POLICY:
            result["susceptibility_policy"] = self.susceptibility_policy
        return result

    def assert_disjoint(self) -> None:
        """Assert that every complete group belongs to exactly one split."""

        for name, mapping in self.group_maps().items():
            per_split = {
                split: {key for key, value in mapping.items() if value == split}
                for split in SPLITS
            }
            for first_index, first in enumerate(SPLITS):
                for second in SPLITS[first_index + 1 :]:
                    overlap = per_split[first] & per_split[second]
                    if overlap:
                        raise AssertionError(
                            f"{name} leaks between {first} and {second}: {overlap}"
                        )


def build_split_manifest(
    *,
    seed: int,
    theta_values: Sequence[int] = (-2, -1, 1, 2),
    theta_policy: str = LEGACY_THETA_POLICY,
    susceptibility_levels: Sequence[float] = (0.15, 0.45, 0.85),
    susceptibility_policy: str = LEGACY_SUSCEPTIBILITY_POLICY,
    option_templates: Iterable[str] = ("travel-v1", "writing-v1", "terminal-v1"),
    dialogue_templates: Iterable[str] = (
        "choice-neutral-v1",
        "confirmation-v1",
        "direct-probe-v1",
    ),
    scenario_families: Iterable[str] = (
        "travel-hotel",
        "travel-itinerary",
        "writing-revision",
        "terminal-diagnostic",
    ),
    paraphrase_templates: Iterable[str] = (
        "accept-first",
        "accept-label",
        "keep-default",
        "terminal-neutral",
    ),
    train_option_templates: Iterable[str] = (),
    development_option_templates: Iterable[str] = (),
    test_option_templates: Iterable[str] = ("terminal-v1",),
    train_dialogue_templates: Iterable[str] = (),
    development_dialogue_templates: Iterable[str] = (),
    test_dialogue_templates: Iterable[str] = ("direct-probe-v1",),
    train_scenario_families: Iterable[str] = (),
    development_scenario_families: Iterable[str] = (),
    test_scenario_families: Iterable[str] = ("terminal-diagnostic",),
    train_paraphrase_templates: Iterable[str] = (),
    development_paraphrase_templates: Iterable[str] = (),
    test_paraphrase_templates: Iterable[str] = ("terminal-neutral",),
) -> SplitManifest:
    """Create group-disjoint assignments from stable semantic identifiers."""

    if theta_policy not in THETA_POLICIES:
        raise ValueError(
            f"theta_policy must be one of {sorted(THETA_POLICIES)}"
        )
    if susceptibility_policy not in SUSCEPTIBILITY_POLICIES:
        raise ValueError(
            f"susceptibility_policy must be one of "
            f"{sorted(SUSCEPTIBILITY_POLICIES)}"
        )

    def assign(
        keys: Iterable[str],
        *,
        reserved_train: Iterable[str] = (),
        reserved_development: Iterable[str] = (),
        reserved_test: Iterable[str] = (),
    ) -> dict[str, str]:
        ordered = sorted(set(keys))
        result = _force_nonempty(
            {key: _bucket(key, seed) for key in ordered}, ordered
        )
        reservations = {
            "train": tuple(sorted(set(reserved_train))),
            "development": tuple(sorted(set(reserved_development))),
            "test": tuple(sorted(set(reserved_test))),
        }
        reserved_sets = tuple(set(items) for items in reservations.values())
        if any(
            first & second
            for first_index, first in enumerate(reserved_sets)
            for second in reserved_sets[first_index + 1 :]
        ):
            raise ValueError("a group cannot be reserved for multiple splits")
        reserved = set().union(*reserved_sets)
        missing = reserved - set(ordered)
        if missing:
            raise ValueError(
                "reserved split groups are absent: "
                + ", ".join(sorted(missing))
            )
        for split, items in reservations.items():
            for key in items:
                result[key] = split
        available = [key for key in ordered if key not in reserved]
        # Preserve a usable train/development split even when the input contains
        # only a few explicitly named template families.
        if available:
            result[available[0]] = "train"
        if len(available) > 1:
            result[available[1]] = "development"
        return result

    theta_keys = tuple(
        theta_group_key(values)
        for values in product(theta_values, repeat=3)
    )
    if theta_policy == LEGACY_THETA_POLICY:
        theta_groups = assign(theta_keys)
    else:
        theta_groups = {
            group_id: split
            for group_id, split, _, _, _ in _orthogonal_theta_rows(
                theta_values,
                seed=seed,
            )
        }
    susceptibility_keys = tuple(
        susceptibility_group_key(values)
        for values in product(susceptibility_levels, repeat=3)
    )
    if susceptibility_policy == LEGACY_SUSCEPTIBILITY_POLICY:
        susceptibility_groups = assign(susceptibility_keys)
    else:
        susceptibility_groups = {
            group_id: split
            for group_id, split, _, _ in _orthogonal_susceptibility_rows(
                susceptibility_levels,
                seed=seed,
            )
        }
    manifest = SplitManifest(
        seed=seed,
        theta_groups=theta_groups,
        susceptibility_groups=susceptibility_groups,
        option_templates=assign(
            option_templates,
            reserved_train=train_option_templates,
            reserved_development=development_option_templates,
            reserved_test=test_option_templates,
        ),
        dialogue_templates=assign(
            dialogue_templates,
            reserved_train=train_dialogue_templates,
            reserved_development=development_dialogue_templates,
            reserved_test=test_dialogue_templates,
        ),
        scenario_families=assign(
            scenario_families,
            reserved_train=train_scenario_families,
            reserved_development=development_scenario_families,
            reserved_test=test_scenario_families,
        ),
        paraphrase_templates=assign(
            paraphrase_templates,
            reserved_train=train_paraphrase_templates,
            reserved_development=development_paraphrase_templates,
            reserved_test=test_paraphrase_templates,
        ),
        theta_policy=theta_policy,
        susceptibility_policy=susceptibility_policy,
    )
    manifest.assert_disjoint()
    return manifest


def assert_terminal_templates_held_out(
    manifest: SplitManifest, template_ids: Iterable[str]
) -> None:
    invalid = [
        template_id
        for template_id in template_ids
        if manifest.option_templates.get(template_id) != "test"
    ]
    if invalid:
        raise ValueError(
            "terminal diagnostic templates must be test-only: "
            + ", ".join(sorted(invalid))
        )
