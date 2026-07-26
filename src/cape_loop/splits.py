"""Leakage-resistant split manifests."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from itertools import product
from typing import Any, Iterable, Mapping, Sequence


SPLITS = ("train", "development", "test")


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


@dataclass(frozen=True, slots=True)
class SplitManifest:
    seed: int
    theta_groups: Mapping[str, str]
    susceptibility_groups: Mapping[str, str]
    option_templates: Mapping[str, str]
    dialogue_templates: Mapping[str, str]
    scenario_families: Mapping[str, str]
    paraphrase_templates: Mapping[str, str]

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("split seed must be non-negative")
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
        return {
            "schema_version": 1,
            "seed": self.seed,
            **{
                name: dict(sorted(mapping.items()))
                for name, mapping in self.group_maps().items()
            },
        }

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
    susceptibility_levels: Sequence[float] = (0.15, 0.45, 0.85),
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

    theta_keys = (
        ",".join(str(value) for value in values)
        for values in product(theta_values, repeat=3)
    )
    susceptibility_keys = (
        ",".join(f"{value:.8g}" for value in values)
        for values in product(susceptibility_levels, repeat=3)
    )
    manifest = SplitManifest(
        seed=seed,
        theta_groups=assign(theta_keys),
        susceptibility_groups=assign(susceptibility_keys),
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
