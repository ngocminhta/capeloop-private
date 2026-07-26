"""Stable domain definitions and controlled directional options."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

from .schemas import NUM_ATTRIBUTES, Option


DATA_SPLITS = ("train", "development", "test")
_SPLIT_TEMPLATE_CODES = {
    "train": "atlas",
    "development": "beacon",
    "test": "cedar",
}
_SPLIT_LABEL_SUFFIXES = {
    "train": "plan",
    "development": "alternative",
    "test": "choice",
}


def _split_code(split: str) -> str:
    try:
        return _SPLIT_TEMPLATE_CODES[split]
    except KeyError as exc:
        raise ValueError(f"split must be one of {DATA_SPLITS}") from exc


def option_template_id(domain_id: str, split: str) -> str:
    """Return the opaque, fixed option-template family for one data split."""

    return f"{domain_id}-option-{_split_code(split)}-v1"


def dialogue_template_id(domain_id: str, split: str) -> str:
    """Return the fixed visible dialogue-template identifier for a split."""

    return f"{domain_id}-dialogue-{_split_code(split)}-v1"


def scenario_family_id(domain_id: str, split: str) -> str:
    """Return the fixed scenario family used by a split."""

    return f"{domain_id}-scenario-{_split_code(split)}-v1"


@dataclass(frozen=True, slots=True)
class AttributeSpec:
    key: str
    negative_label: str
    positive_label: str

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("attribute key cannot be empty")
        if not self.negative_label or not self.positive_label:
            raise ValueError("attribute direction labels cannot be empty")
        if self.negative_label == self.positive_label:
            raise ValueError("attribute direction labels must differ")

    def label_for(self, direction: int) -> str:
        if direction == -1:
            return self.negative_label
        if direction == 1:
            return self.positive_label
        raise ValueError("direction must be -1 or +1")

    def to_dict(self) -> dict[str, str]:
        return {
            "key": self.key,
            "negative_label": self.negative_label,
            "positive_label": self.positive_label,
        }


@dataclass(frozen=True, slots=True)
class DomainSpec:
    """A three-attribute domain with stable full and isolated option pools."""

    domain_id: str
    attributes: tuple[AttributeSpec, AttributeSpec, AttributeSpec]
    option_pool: tuple[Option, ...]
    isolated_options: tuple[Option, ...]

    def __post_init__(self) -> None:
        if not self.domain_id:
            raise ValueError("domain_id cannot be empty")
        if len(self.attributes) != NUM_ATTRIBUTES:
            raise ValueError(f"a domain must define {NUM_ATTRIBUTES} attributes")
        all_options = self.option_pool + self.isolated_options
        ids = [option.option_id for option in all_options]
        if len(ids) != len(set(ids)):
            raise ValueError("domain option IDs must be unique")
        if any(option.domain != self.domain_id for option in all_options):
            raise ValueError("all options must carry their owning domain ID")

    def directional_option(self, attribute: int, direction: int) -> Option:
        if not 0 <= attribute < NUM_ATTRIBUTES:
            raise ValueError(f"attribute must be in [0, {NUM_ATTRIBUTES})")
        if direction not in (-1, 1):
            raise ValueError("direction must be -1 or +1")
        for option in self.isolated_options:
            if (
                option.features[attribute] * direction > 0
                and all(
                    value == 0.0
                    for index, value in enumerate(option.features)
                    if index != attribute
                )
            ):
                return option
        raise KeyError(
            f"{self.domain_id}:{attribute}:{direction:+d}"
        )

    def isolated_pair(self, attribute: int) -> tuple[Option, Option]:
        return (
            self.directional_option(attribute, -1),
            self.directional_option(attribute, 1),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain_id": self.domain_id,
            "attributes": [attribute.to_dict() for attribute in self.attributes],
            "option_pool": [option.to_dict() for option in self.option_pool],
            "isolated_options": [
                option.to_dict() for option in self.isolated_options
            ],
        }


def _make_domain(
    domain_id: str,
    attributes: tuple[AttributeSpec, AttributeSpec, AttributeSpec],
) -> DomainSpec:
    isolated: list[Option] = []
    for index, attribute in enumerate(attributes):
        for direction, suffix in ((-1, "neg"), (1, "pos")):
            features = [0.0, 0.0, 0.0]
            features[index] = 0.5 * direction
            isolated.append(
                Option(
                    option_id=f"{domain_id}_{attribute.key}_{suffix}",
                    features=(features[0], features[1], features[2]),
                    label=attribute.label_for(direction),
                    domain=domain_id,
                )
            )

    full_pool: list[Option] = []
    for directions in product((-1, 1), repeat=NUM_ATTRIBUTES):
        code = "".join("n" if direction < 0 else "p" for direction in directions)
        labels = [
            attributes[index].label_for(direction)
            for index, direction in enumerate(directions)
        ]
        full_pool.append(
            Option(
                option_id=f"{domain_id}_pool_{code}",
                features=tuple(0.5 * direction for direction in directions),
                label=" / ".join(labels),
                domain=domain_id,
            )
        )

    return DomainSpec(
        domain_id=domain_id,
        attributes=attributes,
        option_pool=tuple(full_pool),
        isolated_options=tuple(isolated),
    )


TRAVEL = _make_domain(
    "travel",
    (
        AttributeSpec("price", "budget", "premium"),
        AttributeSpec("setting", "central", "comfort"),
        AttributeSpec("planning", "convenience", "flexibility"),
    ),
)

WRITING = _make_domain(
    "writing",
    (
        AttributeSpec("length", "concise", "detailed"),
        AttributeSpec("tone", "formal", "conversational"),
        AttributeSpec("spelling", "british", "american"),
    ),
)

DOMAINS: tuple[DomainSpec, ...] = (TRAVEL, WRITING)
_DOMAIN_BY_ID = {domain.domain_id: domain for domain in DOMAINS}


def get_domain(domain_id: str) -> DomainSpec:
    try:
        return _DOMAIN_BY_ID[domain_id]
    except KeyError as exc:
        raise KeyError(f"unknown CAPE-Loop domain: {domain_id!r}") from exc


def domain_for_split(domain: DomainSpec, split: str) -> DomainSpec:
    """Clone a domain onto a feature-preserving, split-disjoint option family.

    The opaque atlas/beacon/cedar family names avoid putting the words
    ``train`` or ``test`` into an LLM-visible option. Intrinsic features are
    unchanged, so the response-model estimand is preserved while option IDs and
    labels cannot leak across fitting, calibration, and evaluation.
    """

    template_id = option_template_id(domain.domain_id, split)
    suffix = _SPLIT_LABEL_SUFFIXES[split]

    def convert(option: Option) -> Option:
        return Option(
            option_id=f"{template_id}:{option.option_id}",
            features=option.features,
            label=f"{option.label} {suffix}".strip(),
            domain=option.domain,
        )

    return DomainSpec(
        domain_id=domain.domain_id,
        attributes=domain.attributes,
        option_pool=tuple(convert(option) for option in domain.option_pool),
        isolated_options=tuple(
            convert(option) for option in domain.isolated_options
        ),
    )


def all_domains() -> tuple[DomainSpec, ...]:
    return DOMAINS
