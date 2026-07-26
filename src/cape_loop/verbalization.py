"""Constrained verbalization that cannot change a structured choice."""

from __future__ import annotations

import re
from typing import Sequence


_TEMPLATES = (
    "Let's use {label}.",
    "{label} works for me.",
    "I'll choose {label}.",
    "Keep {label}.",
)
_UNSUPPORTED_PREFERENCE = re.compile(
    r"\b(i|we)\s+(generally\s+|always\s+|usually\s+)?"
    r"(prefer|like|love|hate|want)\b",
    re.IGNORECASE,
)


def verbalize_choice(label: str, variant: int = 0) -> str:
    """Render only a local acceptance of the already sampled option."""

    if not label or any(c in label for c in "\r\n"):
        raise ValueError("label must be a non-empty single-line string")
    return _TEMPLATES[variant % len(_TEMPLATES)].format(label=label)


def validate_surface_response(
    response: str,
    *,
    selected_label: str,
    allowed_responses: Sequence[str] | None = None,
) -> None:
    """Reject unsupported general-preference claims or a mismatched choice."""

    if not response.strip():
        raise ValueError("surface response is empty")
    if _UNSUPPORTED_PREFERENCE.search(response):
        raise ValueError("surface response invents a general preference claim")
    if allowed_responses is not None and response not in allowed_responses:
        raise ValueError("surface response is not in the semantic whitelist")
    if selected_label.casefold() not in response.casefold():
        raise ValueError("surface response does not identify the selected option")


def allowed_verbalizations(label: str) -> tuple[str, ...]:
    return tuple(template.format(label=label) for template in _TEMPLATES)

