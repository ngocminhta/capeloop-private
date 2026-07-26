"""Counter-based deterministic randomness derived from semantic SHA-256 keys."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from typing import Any, TypeVar


T = TypeVar("T")
_UINT64_RANGE = 1 << 64


def _canonical(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("semantic RNG keys cannot contain non-finite floats")
        return {"__float__": value.hex()}
    if isinstance(value, bytes):
        return {"__bytes__": value.hex()}
    if isinstance(value, Mapping):
        return {
            str(key): _canonical(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if hasattr(value, "to_dict"):
        return _canonical(value.to_dict())
    raise TypeError(f"unsupported semantic RNG key type: {type(value).__name__}")


def semantic_digest(master_seed: int, *keys: Any) -> bytes:
    """Return a stable digest independent of process hash randomization."""

    if isinstance(master_seed, bool) or not isinstance(master_seed, int):
        raise TypeError("master_seed must be an integer")
    payload = {
        "master_seed": master_seed,
        "keys": [_canonical(key) for key in keys],
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).digest()


def semantic_seed(master_seed: int, *keys: Any) -> int:
    """Return a deterministic unsigned 64-bit seed."""

    return int.from_bytes(semantic_digest(master_seed, *keys)[:8], "big")


def uniform(master_seed: int, *keys: Any) -> float:
    """Return a deterministic value strictly between zero and one."""

    integer = semantic_seed(master_seed, *keys)
    return (integer + 0.5) / _UINT64_RANGE


def gumbel(master_seed: int, *keys: Any) -> float:
    """Return a standard Gumbel variate for random-utility sampling."""

    value = uniform(master_seed, *keys)
    return -math.log(-math.log(value))


def weighted_index(
    weights: Sequence[float],
    master_seed: int,
    *keys: Any,
) -> int:
    """Sample an index from finite non-negative weights."""

    if not weights:
        raise ValueError("weights cannot be empty")
    normalized_weights: list[float] = []
    for index, weight in enumerate(weights):
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise TypeError(f"weights[{index}] must be numeric")
        numeric = float(weight)
        if not math.isfinite(numeric) or numeric < 0.0:
            raise ValueError("weights must be finite and non-negative")
        normalized_weights.append(numeric)
    total = math.fsum(normalized_weights)
    if total <= 0.0:
        raise ValueError("at least one weight must be positive")

    threshold = uniform(master_seed, *keys) * total
    cumulative = 0.0
    for index, weight in enumerate(normalized_weights):
        cumulative += weight
        if threshold < cumulative:
            return index
    return len(normalized_weights) - 1


def weighted_choice(
    items: Sequence[T],
    weights: Sequence[float],
    master_seed: int,
    *keys: Any,
) -> T:
    """Sample an item using :func:`weighted_index`."""

    if len(items) != len(weights):
        raise ValueError("items and weights must have the same length")
    return items[weighted_index(weights, master_seed, *keys)]

