"""Generate blinded materials for pragmatic evidence-strength validation.

This module creates study packets only. Recruitment, consent, randomization
deployment, compensation, and ethics review remain the research team's
responsibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Iterable, Mapping


CONDITIONS = ("volunteered", "balanced", "restricted", "default", "suggested")


def _require_assignment_id(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("assignment_id must be a nonempty string")
    return value


@dataclass(frozen=True, slots=True)
class StudyItem:
    item_id: str
    scenario_id: str
    condition: str
    vignette: str
    preference_claim: str
    scale_min: int = 1
    scale_max: int = 7

    def __post_init__(self) -> None:
        if self.condition not in CONDITIONS:
            raise ValueError(f"unknown human-study condition: {self.condition}")
        if not self.item_id or not self.scenario_id:
            raise ValueError("item and scenario identifiers are required")
        if not self.vignette or not self.preference_claim:
            raise ValueError("vignette and preference claim are required")
        if self.scale_min >= self.scale_max:
            raise ValueError("rating scale is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "scenario_id": self.scenario_id,
            "condition": self.condition,
            "vignette": self.vignette,
            "prompt": (
                "How strongly does this interaction support the claim that "
                f"{self.preference_claim}?"
            ),
            "scale": {
                "minimum": self.scale_min,
                "maximum": self.scale_max,
                "minimum_label": "No support",
                "maximum_label": "Very strong support",
            },
        }


def blind_and_order_items(
    items: Iterable[StudyItem], *, assignment_id: str, seed: int
) -> tuple[dict[str, Any], ...]:
    """Return a deterministic order with condition labels hidden from annotators."""

    _require_assignment_id(assignment_id)
    material = tuple(items)
    if len({item.item_id for item in material}) != len(material):
        raise ValueError("study item IDs must be unique")

    def sort_key(item: StudyItem) -> bytes:
        return sha256(
            f"{seed}\0{assignment_id}\0{item.item_id}".encode("utf-8")
        ).digest()

    result = []
    for display_index, item in enumerate(sorted(material, key=sort_key), start=1):
        value = item.to_dict()
        value.pop("condition")
        value.pop("item_id")
        value["display_id"] = f"item-{display_index:04d}"
        result.append(value)
    return tuple(result)


def build_codebook(items: Iterable[StudyItem]) -> dict[str, Mapping[str, str]]:
    """Create the separately stored, researcher-only condition codebook."""

    return {
        item.item_id: {
            "scenario_id": item.scenario_id,
            "condition": item.condition,
        }
        for item in items
    }


def build_assignment_codebook(
    items: Iterable[StudyItem],
    *,
    assignment_id: str,
    seed: int,
) -> dict[str, Mapping[str, str]]:
    """Map blinded display IDs back to source items and conditions."""

    _require_assignment_id(assignment_id)
    material = tuple(items)
    if len({item.item_id for item in material}) != len(material):
        raise ValueError("study item IDs must be unique")

    ordered = sorted(
        material,
        key=lambda item: sha256(
            f"{seed}\0{assignment_id}\0{item.item_id}".encode("utf-8")
        ).digest(),
    )
    return {
        f"item-{display_index:04d}": {
            "item_id": item.item_id,
            "scenario_id": item.scenario_id,
            "condition": item.condition,
        }
        for display_index, item in enumerate(ordered, start=1)
    }


def validate_rating_record(
    record: Mapping[str, Any], *, valid_display_ids: set[str]
) -> None:
    allowed = {"assignment_id", "display_id", "rating"}
    if set(record) != allowed:
        raise ValueError("rating record has missing or unknown fields")
    _require_assignment_id(record["assignment_id"])
    if record["display_id"] not in valid_display_ids:
        raise ValueError("unknown blinded display ID")
    rating = record["rating"]
    if isinstance(rating, bool) or not isinstance(rating, int) or not 1 <= rating <= 7:
        raise ValueError("rating must be an integer from 1 to 7")
