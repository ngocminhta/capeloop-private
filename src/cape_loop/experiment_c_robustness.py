"""Immutable, offline cross-seed robustness review for Experiment C.

Each ordinary ``evaluation_validity`` run already performs a seeded,
complete-latent-user clustered bootstrap.  This module verifies at least two
such completed runs, requires their scientific configuration and executable
source identity to match, and compares the predeclared ranking conclusions
across distinct seeds.

The result is descriptive review evidence only.  It is written outside every
source run, never mutates a source, and always retains
``claim_status = "not_claimed"``.
"""

from __future__ import annotations

from hashlib import sha256
from math import comb, gcd, isfinite
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence
import json
import os
import shutil
import tempfile

from .artifacts import canonical_json, verify_run
from .config import AppConfig


ARTIFACT_KIND = "experiment-c-multiseed-robustness-review"
CLAIM_STATUS = "not_claimed"
MAX_SOURCE_RUNS = 32
MAX_JSON_BYTES = 32 * 1024 * 1024

RANKINGS_PATH = "metrics/experiment-c-rankings.json"
GATE_REPORT_PATH = "metrics/gate-report.json"
SUMMARY_PATH = "metrics/summary.json"
CONFIG_PATH = "config.resolved.json"
MANIFEST_PATH = "manifest.json"
CHECKSUM_PATH = "SHA256SUMS"

REVIEW_FILES = frozenset({"review.json", "manifest.json"})
COMPARISON_DIMENSIONS = (
    "point_ranking.fixed_balanced_development",
    "point_ranking.fixed_biased_development",
    "point_ranking.endogenous_closed_loop_development",
    "inferential_top_tier.fixed_balanced_development",
    "inferential_top_tier.endogenous_closed_loop_development",
    "inferential_partial_order.fixed_balanced_development",
    "inferential_partial_order.endogenous_closed_loop_development",
    "gate_5.decision_and_status",
    "esr.development_selection_sets",
)

_BOOTSTRAP_METHOD = (
    "paired percentile bootstrap over complete latent-user clusters; all "
    "domains and trajectory replicates remain grouped within each resampled unit"
)
_ALIGNMENT_KEY = [
    "split",
    "regime",
    "user_id",
    "domain_id",
    "replicate",
]

_RANKING_FIELDS = {
    "inference_unit",
    "alignment_key",
    "development_cluster_count",
    "test_cluster_count",
    "cluster_component_layout",
    "bootstrap_method",
    "open_mean_errors",
    "biased_mean_errors",
    "closed_development_mean_errors",
    "closed_test_mean_errors",
    "open_ranks",
    "biased_ranks",
    "closed_ranks",
    "open_closed_kendall_tau",
    "biased_closed_kendall_tau",
    "open_bootstrap_ranks",
    "closed_bootstrap_ranks",
    "pairwise_reversal_probabilities",
    "pairwise_tie_probabilities",
    "pairwise_open_difference_intervals",
    "pairwise_closed_difference_intervals",
    "pairwise_open_closed_shift_intervals",
    "credible_pairwise_reversals",
    "credible_reversal_basis",
    "open_partial_order",
    "closed_partial_order",
    "partial_order",
    "partial_order_basis",
    "open_loop_optimism",
    "evaluation_selection_regret",
}

_DIFFERENCE_INTERVAL_FIELDS = {
    "first_system",
    "second_system",
    "estimand",
    "estimate",
    "lower",
    "upper",
    "relation",
    "tie_tolerance",
    "replicate_count",
    "method",
}

_SHIFT_INTERVAL_FIELDS = {
    "first_system",
    "second_system",
    "open_estimand",
    "open_estimate",
    "open_lower",
    "open_upper",
    "open_relation",
    "closed_estimand",
    "closed_estimate",
    "closed_lower",
    "closed_upper",
    "closed_relation",
    "shift_estimand",
    "shift_estimate",
    "shift_lower",
    "shift_upper",
    "reversal_relation",
    "credible_reversal",
    "tie_tolerance",
    "replicate_count",
    "independent_unit_count",
    "method",
}

_ESR_FIELDS = {
    "open_selected_set",
    "closed_selected_set",
    "selection_basis",
    "evaluation_selection_regret",
    "evaluation_selection_regret_min",
    "evaluation_selection_regret_max",
    "evaluation_selection_regret_interval_envelope_lower",
    "evaluation_selection_regret_interval_envelope_upper",
    "selection_policy",
    "pair_count",
    "pairwise_closed_test_intervals",
}


def _digest(value: Any) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _file_digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _validate_digest(value: Any, name: str) -> str:
    if not (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _exact_fields(
    value: Mapping[str, Any],
    expected: set[str] | frozenset[str],
    *,
    name: str,
) -> None:
    observed = set(value)
    if observed != set(expected):
        missing = sorted(set(expected) - observed)
        unknown = sorted(observed - set(expected))
        raise ValueError(
            f"{name} fields differ; missing={missing}, unknown={unknown}"
        )


def _validate_finite_tree(value: Any, *, name: str = "JSON value") -> None:
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError(f"{name} contains a non-finite number")
        return
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, list):
        for item in value:
            _validate_finite_tree(item, name=name)
        return
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError(f"{name} contains a non-string object key")
        for item in value.values():
            _validate_finite_tree(item, name=name)
        return
    raise ValueError(f"{name} contains an unsupported value")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _strict_json_object(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key is forbidden: {key}")
        result[key] = value
    return result


def _safe_file(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if (
        not relative
        or pure.is_absolute()
        or pure.as_posix() != relative
        or "\\" in relative
        or "\x00" in relative
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ValueError(f"unsafe retained path: {relative}")
    candidate = root.joinpath(*pure.parts)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"required artifact is not a regular file: {relative}")
    resolved = candidate.resolve()
    if resolved == root or root not in resolved.parents:
        raise ValueError(f"artifact path escapes source run: {relative}")
    return candidate


def _read_json_object(path: Path, *, name: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be a regular JSON file")
    size = path.stat().st_size
    if size <= 0 or size > MAX_JSON_BYTES:
        raise ValueError(
            f"{name} byte length must lie in [1, {MAX_JSON_BYTES}]"
        )
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_strict_json_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot parse {name}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain one JSON object")
    _validate_finite_tree(value, name=name)
    return value


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    numeric = float(value)
    if not isfinite(numeric):
        raise ValueError(f"{name} must be a finite number")
    return numeric


def _validate_system_number_map(
    value: Any,
    systems: tuple[str, ...],
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != set(systems):
        raise ValueError(f"{name} must contain every configured updater exactly")
    result: dict[str, float] = {}
    for system in sorted(systems):
        numeric = _finite_number(value[system], f"{name}.{system}")
        if minimum is not None and numeric < minimum:
            raise ValueError(f"{name}.{system} is below its valid range")
        if maximum is not None and numeric > maximum:
            raise ValueError(f"{name}.{system} is above its valid range")
        result[system] = numeric
    return result


def _validate_order(
    value: Any,
    systems: tuple[str, ...],
    *,
    name: str,
) -> list[list[str]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty tier list")
    observed: list[str] = []
    result: list[list[str]] = []
    for tier in value:
        if (
            not isinstance(tier, list)
            or not tier
            or not all(isinstance(system, str) and system for system in tier)
            or tier != sorted(tier)
            or len(tier) != len(set(tier))
        ):
            raise ValueError(f"{name} contains an invalid tier")
        observed.extend(tier)
        result.append(list(tier))
    if len(observed) != len(set(observed)) or set(observed) != set(systems):
        raise ValueError(f"{name} must partition every configured updater")
    return result


def _validate_bootstrap_rank_rows(
    value: Any,
    systems: tuple[str, ...],
    *,
    name: str,
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != len(systems):
        raise ValueError(f"{name} must contain one row per updater")
    rows: list[dict[str, Any]] = []
    observed: set[str] = set()
    for raw in value:
        if not isinstance(raw, Mapping):
            raise ValueError(f"{name} rows must be objects")
        _exact_fields(
            raw,
            {"system_id", "mean_rank", "lower", "upper"},
            name=f"{name} row",
        )
        system = raw["system_id"]
        if not isinstance(system, str) or system not in systems or system in observed:
            raise ValueError(f"{name} contains an invalid system_id")
        observed.add(system)
        mean_rank = _finite_number(raw["mean_rank"], f"{name}.mean_rank")
        lower = _finite_number(raw["lower"], f"{name}.lower")
        upper = _finite_number(raw["upper"], f"{name}.upper")
        if not (
            1 <= mean_rank <= len(systems)
            and 1 <= lower <= upper <= len(systems)
        ):
            raise ValueError(f"{name} contains an invalid rank interval")
        rows.append(
            {
                "system_id": system,
                "mean_rank": mean_rank,
                "lower": lower,
                "upper": upper,
            }
        )
    if observed != set(systems):
        raise ValueError(f"{name} updater coverage is incomplete")
    return sorted(rows, key=lambda row: row["system_id"])


def _pair_ids(systems: tuple[str, ...]) -> tuple[str, ...]:
    ordered = sorted(systems)
    return tuple(
        f"{first}|{second}"
        for index, first in enumerate(ordered)
        for second in ordered[index + 1 :]
    )


def _validate_probability_map(
    value: Any,
    systems: tuple[str, ...],
    *,
    name: str,
) -> dict[str, float]:
    expected = set(_pair_ids(systems))
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{name} must contain every updater pair exactly")
    result = {}
    for pair in sorted(expected):
        numeric = _finite_number(value[pair], f"{name}.{pair}")
        if not 0.0 <= numeric <= 1.0:
            raise ValueError(f"{name}.{pair} must lie in [0, 1]")
        result[pair] = numeric
    return result


def _validate_difference_intervals(
    value: Any,
    systems: tuple[str, ...],
    *,
    name: str,
    replicates: int,
    tie_tolerance: float,
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != comb(len(systems), 2):
        raise ValueError(f"{name} must contain every updater pair exactly")
    expected_pairs = {
        tuple(pair.split("|", 1)) for pair in _pair_ids(systems)
    }
    observed: set[tuple[str, str]] = set()
    rows: list[dict[str, Any]] = []
    for raw in value:
        if not isinstance(raw, Mapping):
            raise ValueError(f"{name} rows must be objects")
        _exact_fields(raw, _DIFFERENCE_INTERVAL_FIELDS, name=f"{name} row")
        pair = (raw["first_system"], raw["second_system"])
        if pair not in expected_pairs or pair in observed:
            raise ValueError(f"{name} contains an invalid updater pair")
        observed.add(pair)
        if raw["replicate_count"] != replicates:
            raise ValueError(f"{name} bootstrap replicate count mismatch")
        if _finite_number(raw["tie_tolerance"], f"{name}.tie_tolerance") != (
            tie_tolerance
        ):
            raise ValueError(f"{name} ranking tie tolerance mismatch")
        _finite_number(raw["estimate"], f"{name}.estimate")
        lower = _finite_number(raw["lower"], f"{name}.lower")
        upper = _finite_number(raw["upper"], f"{name}.upper")
        if lower > upper:
            raise ValueError(f"{name} interval bounds are reversed")
        rows.append(dict(raw))
    if observed != expected_pairs:
        raise ValueError(f"{name} updater-pair coverage is incomplete")
    return rows


def _validate_shift_intervals(
    value: Any,
    systems: tuple[str, ...],
    *,
    replicates: int,
    tie_tolerance: float,
) -> list[dict[str, Any]]:
    name = "pairwise_open_closed_shift_intervals"
    if not isinstance(value, list) or len(value) != comb(len(systems), 2):
        raise ValueError(f"{name} must contain every updater pair exactly")
    expected_pairs = {
        tuple(pair.split("|", 1)) for pair in _pair_ids(systems)
    }
    observed: set[tuple[str, str]] = set()
    rows: list[dict[str, Any]] = []
    for raw in value:
        if not isinstance(raw, Mapping):
            raise ValueError(f"{name} rows must be objects")
        _exact_fields(raw, _SHIFT_INTERVAL_FIELDS, name=f"{name} row")
        pair = (raw["first_system"], raw["second_system"])
        if pair not in expected_pairs or pair in observed:
            raise ValueError(f"{name} contains an invalid updater pair")
        observed.add(pair)
        if raw["replicate_count"] != replicates:
            raise ValueError(f"{name} bootstrap replicate count mismatch")
        if _finite_number(raw["tie_tolerance"], f"{name}.tie_tolerance") != (
            tie_tolerance
        ):
            raise ValueError(f"{name} ranking tie tolerance mismatch")
        _positive_integer(
            raw["independent_unit_count"],
            f"{name}.independent_unit_count",
        )
        for prefix in ("open", "closed", "shift"):
            _finite_number(
                raw[f"{prefix}_estimate"],
                f"{name}.{prefix}_estimate",
            )
            lower = _finite_number(
                raw[f"{prefix}_lower"],
                f"{name}.{prefix}_lower",
            )
            upper = _finite_number(
                raw[f"{prefix}_upper"],
                f"{name}.{prefix}_upper",
            )
            if lower > upper:
                raise ValueError(f"{name} {prefix} interval bounds are reversed")
        if not isinstance(raw["credible_reversal"], bool):
            raise ValueError(f"{name}.credible_reversal must be Boolean")
        rows.append(dict(raw))
    if observed != expected_pairs:
        raise ValueError(f"{name} updater-pair coverage is incomplete")
    return rows


def _validate_esr(
    value: Any,
    systems: tuple[str, ...],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("evaluation_selection_regret must be an object")
    _exact_fields(value, _ESR_FIELDS, name="evaluation_selection_regret")
    result = dict(value)
    for field in ("open_selected_set", "closed_selected_set"):
        selected = value[field]
        if (
            not isinstance(selected, list)
            or not selected
            or not all(
                isinstance(system, str) and system in systems
                for system in selected
            )
            or selected != sorted(selected)
            or len(selected) != len(set(selected))
        ):
            raise ValueError(f"evaluation_selection_regret.{field} is invalid")
    expected_pairs = len(value["open_selected_set"]) * len(
        value["closed_selected_set"]
    )
    if value["pair_count"] != expected_pairs:
        raise ValueError("evaluation_selection_regret pair_count mismatch")
    rows = value["pairwise_closed_test_intervals"]
    if not isinstance(rows, list) or len(rows) != expected_pairs:
        raise ValueError(
            "evaluation_selection_regret test interval coverage mismatch"
        )
    row_fields = {
        "open_selected_system",
        "closed_selected_system",
        "closed_test_error_difference",
        "lower",
        "upper",
    }
    observed_pairs: set[tuple[str, str]] = set()
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise ValueError("ESR pair rows must be objects")
        _exact_fields(raw, row_fields, name="ESR pair row")
        pair = (
            raw["open_selected_system"],
            raw["closed_selected_system"],
        )
        if (
            pair[0] not in value["open_selected_set"]
            or pair[1] not in value["closed_selected_set"]
            or pair in observed_pairs
        ):
            raise ValueError("ESR pair row has an invalid selection pair")
        observed_pairs.add(pair)
        _finite_number(
            raw["closed_test_error_difference"],
            "ESR closed-test difference",
        )
        lower = _finite_number(raw["lower"], "ESR lower")
        upper = _finite_number(raw["upper"], "ESR upper")
        if lower > upper:
            raise ValueError("ESR interval bounds are reversed")
    for field in (
        "evaluation_selection_regret",
        "evaluation_selection_regret_min",
        "evaluation_selection_regret_max",
        "evaluation_selection_regret_interval_envelope_lower",
        "evaluation_selection_regret_interval_envelope_upper",
    ):
        _finite_number(value[field], f"evaluation_selection_regret.{field}")
    return result


def _validate_ranking(
    raw: Mapping[str, Any],
    *,
    systems: tuple[str, ...],
    replicates: int,
    tie_tolerance: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _exact_fields(raw, _RANKING_FIELDS, name="Experiment C rankings")
    if raw["inference_unit"] != "complete_latent_user_cluster":
        raise ValueError("Experiment C rankings use the wrong inference unit")
    if raw["alignment_key"] != _ALIGNMENT_KEY:
        raise ValueError("Experiment C ranking alignment key mismatch")
    if raw["bootstrap_method"] != _BOOTSTRAP_METHOD:
        raise ValueError("Experiment C rankings use an unknown bootstrap method")
    development_clusters = _positive_integer(
        raw["development_cluster_count"],
        "development_cluster_count",
    )
    test_clusters = _positive_integer(
        raw["test_cluster_count"],
        "test_cluster_count",
    )
    layout = raw["cluster_component_layout"]
    if not isinstance(layout, list) or not layout:
        raise ValueError("cluster_component_layout must be non-empty")
    for row in layout:
        if not isinstance(row, Mapping):
            raise ValueError("cluster_component_layout rows must be objects")
        _exact_fields(
            row,
            {"domain_id", "replicate"},
            name="cluster_component_layout row",
        )
        if not isinstance(row["domain_id"], str) or not row["domain_id"]:
            raise ValueError("cluster_component_layout domain_id is invalid")
        if (
            isinstance(row["replicate"], bool)
            or not isinstance(row["replicate"], int)
            or row["replicate"] < 0
        ):
            raise ValueError("cluster_component_layout replicate is invalid")

    for field in (
        "open_mean_errors",
        "biased_mean_errors",
        "closed_development_mean_errors",
        "closed_test_mean_errors",
        "open_loop_optimism",
    ):
        _validate_system_number_map(raw[field], systems, name=field)
    point_rankings = {
        "fixed_balanced_development": _validate_system_number_map(
            raw["open_ranks"],
            systems,
            name="open_ranks",
            minimum=1.0,
            maximum=float(len(systems)),
        ),
        "fixed_biased_development": _validate_system_number_map(
            raw["biased_ranks"],
            systems,
            name="biased_ranks",
            minimum=1.0,
            maximum=float(len(systems)),
        ),
        "endogenous_closed_loop_development": _validate_system_number_map(
            raw["closed_ranks"],
            systems,
            name="closed_ranks",
            minimum=1.0,
            maximum=float(len(systems)),
        ),
    }
    for field in ("open_closed_kendall_tau", "biased_closed_kendall_tau"):
        if raw[field] is not None:
            tau = _finite_number(raw[field], field)
            if not -1.0 <= tau <= 1.0:
                raise ValueError(f"{field} must lie in [-1, 1]")

    open_bootstrap = _validate_bootstrap_rank_rows(
        raw["open_bootstrap_ranks"],
        systems,
        name="open_bootstrap_ranks",
    )
    closed_bootstrap = _validate_bootstrap_rank_rows(
        raw["closed_bootstrap_ranks"],
        systems,
        name="closed_bootstrap_ranks",
    )
    reversal_probabilities = _validate_probability_map(
        raw["pairwise_reversal_probabilities"],
        systems,
        name="pairwise_reversal_probabilities",
    )
    tie_probabilities = _validate_probability_map(
        raw["pairwise_tie_probabilities"],
        systems,
        name="pairwise_tie_probabilities",
    )
    open_intervals = _validate_difference_intervals(
        raw["pairwise_open_difference_intervals"],
        systems,
        name="pairwise_open_difference_intervals",
        replicates=replicates,
        tie_tolerance=tie_tolerance,
    )
    closed_intervals = _validate_difference_intervals(
        raw["pairwise_closed_difference_intervals"],
        systems,
        name="pairwise_closed_difference_intervals",
        replicates=replicates,
        tie_tolerance=tie_tolerance,
    )
    shift_intervals = _validate_shift_intervals(
        raw["pairwise_open_closed_shift_intervals"],
        systems,
        replicates=replicates,
        tie_tolerance=tie_tolerance,
    )
    credible = raw["credible_pairwise_reversals"]
    if (
        not isinstance(credible, list)
        or not all(pair in _pair_ids(systems) for pair in credible)
        or credible != sorted(set(credible))
    ):
        raise ValueError("credible_pairwise_reversals is invalid")
    credible_from_intervals = sorted(
        f"{row['first_system']}|{row['second_system']}"
        for row in shift_intervals
        if row["credible_reversal"]
    )
    if credible != credible_from_intervals:
        raise ValueError(
            "credible_pairwise_reversals disagrees with retained intervals"
        )

    open_order = _validate_order(
        raw["open_partial_order"],
        systems,
        name="open_partial_order",
    )
    closed_order = _validate_order(
        raw["closed_partial_order"],
        systems,
        name="closed_partial_order",
    )
    if raw["partial_order"] != closed_order:
        raise ValueError("partial_order must alias closed_partial_order")
    esr = _validate_esr(raw["evaluation_selection_regret"], systems)
    if esr["open_selected_set"] != open_order[0]:
        raise ValueError("ESR open selection set must equal the open top tier")
    if esr["closed_selected_set"] != closed_order[0]:
        raise ValueError("ESR closed selection set must equal the closed top tier")

    observations = {
        "point_rankings": point_rankings,
        "inferential": {
            "open_top_tier": open_order[0],
            "closed_top_tier": closed_order[0],
            "open_partial_order": open_order,
            "closed_partial_order": closed_order,
        },
        "esr_selection_sets": {
            "open_selected_set": list(esr["open_selected_set"]),
            "closed_selected_set": list(esr["closed_selected_set"]),
        },
    }
    bootstrap = {
        "seeded_by_run_config": True,
        "replicate_count": replicates,
        "inference_unit": raw["inference_unit"],
        "alignment_key": list(raw["alignment_key"]),
        "method": raw["bootstrap_method"],
        "development_cluster_count": development_clusters,
        "test_cluster_count": test_clusters,
        "retained_summary_sha256": _digest(
            {
                "open_bootstrap_ranks": open_bootstrap,
                "closed_bootstrap_ranks": closed_bootstrap,
                "pairwise_reversal_probabilities": reversal_probabilities,
                "pairwise_tie_probabilities": tie_probabilities,
                "pairwise_open_difference_intervals": open_intervals,
                "pairwise_closed_difference_intervals": closed_intervals,
                "pairwise_open_closed_shift_intervals": shift_intervals,
                "open_partial_order": open_order,
                "closed_partial_order": closed_order,
                "evaluation_selection_regret": esr,
            }
        ),
    }
    return observations, bootstrap


def _validate_gate_report(
    raw: Mapping[str, Any],
) -> dict[str, Any]:
    _exact_fields(
        raw,
        {"schema_version", "claim_status", "gates"},
        name="gate report",
    )
    if raw["schema_version"] != 1 or raw["claim_status"] != CLAIM_STATUS:
        raise ValueError("gate report has invalid version or claim semantics")
    gates = raw["gates"]
    if not isinstance(gates, list) or len(gates) != 6:
        raise ValueError("gate report must contain exactly six gates")
    by_id: dict[str, Mapping[str, Any]] = {}
    for gate in gates:
        if not isinstance(gate, Mapping):
            raise ValueError("gate report entries must be objects")
        _exact_fields(
            gate,
            {
                "schema_version",
                "gate_id",
                "title",
                "evidence_scope",
                "computed_status",
                "claim_status",
                "criteria",
            },
            name="gate report entry",
        )
        gate_id = gate["gate_id"]
        if (
            not isinstance(gate_id, str)
            or gate_id in by_id
            or gate_id not in {f"gate-{index}" for index in range(1, 7)}
        ):
            raise ValueError("gate report has an invalid or duplicate gate_id")
        if (
            gate["schema_version"] != 1
            or gate["claim_status"] != CLAIM_STATUS
            or gate["computed_status"]
            not in {
                "incomplete",
                "meets_computational_checks",
                "does_not_meet_checks",
            }
        ):
            raise ValueError("gate report entry has invalid semantics")
        criteria = gate["criteria"]
        if not isinstance(criteria, list) or not criteria:
            raise ValueError("gate report entry must contain criteria")
        decisions: list[bool | None] = []
        for criterion in criteria:
            if not isinstance(criterion, Mapping):
                raise ValueError("gate criterion must be an object")
            _exact_fields(
                criterion,
                {
                    "criterion_id",
                    "description",
                    "passed",
                    "observed",
                    "requirement",
                },
                name="gate criterion",
            )
            if not (
                criterion["passed"] is None
                or isinstance(criterion["passed"], bool)
            ):
                raise ValueError("gate criterion passed must be Boolean or null")
            decisions.append(criterion["passed"])
        expected_status = (
            "incomplete"
            if any(decision is None for decision in decisions)
            else (
                "meets_computational_checks"
                if all(decisions)
                else "does_not_meet_checks"
            )
        )
        if gate["computed_status"] != expected_status:
            raise ValueError("gate computed_status disagrees with its criteria")
        by_id[gate_id] = gate
    if set(by_id) != {f"gate-{index}" for index in range(1, 7)}:
        raise ValueError("gate report coverage is incomplete")
    gate_5 = by_id["gate-5"]
    if len(gate_5["criteria"]) != 1:
        raise ValueError("Gate 5 must contain its one declared disjunction")
    criterion = gate_5["criteria"][0]
    if criterion["criterion_id"] != "evaluation-implication-disjunction":
        raise ValueError("Gate 5 criterion identity mismatch")
    return {
        "decision": criterion["passed"],
        "computed_status": gate_5["computed_status"],
    }


def _normalized_scientific_config(config: AppConfig) -> dict[str, Any]:
    payload = config.to_dict()
    run = payload["run"]
    for field in ("name", "seed", "output_root"):
        run.pop(field)
    return payload


def _source_artifact_paths(root: Path) -> dict[str, Path]:
    return {
        "checksum_manifest_sha256": _safe_file(root, CHECKSUM_PATH),
        "manifest_sha256": _safe_file(root, MANIFEST_PATH),
        "config_resolved_sha256": _safe_file(root, CONFIG_PATH),
        "rankings_sha256": _safe_file(root, RANKINGS_PATH),
        "gate_report_sha256": _safe_file(root, GATE_REPORT_PATH),
        "summary_sha256": _safe_file(root, SUMMARY_PATH),
    }


def _load_source(path: str | Path) -> tuple[dict[str, Any], AppConfig, dict[str, Any]]:
    supplied = Path(path)
    if supplied.is_symlink():
        raise ValueError(f"source run cannot be a symlink: {supplied}")
    root = supplied.resolve()
    if not root.is_dir():
        raise ValueError(f"source run is not a directory: {supplied}")
    valid, errors = verify_run(root)
    if not valid:
        raise ValueError(
            f"source run verification failed for {root}: " + "; ".join(errors)
        )
    paths = _source_artifact_paths(root)
    manifest = _read_json_object(paths["manifest_sha256"], name="run manifest")
    config_raw = _read_json_object(
        paths["config_resolved_sha256"],
        name="resolved config",
    )
    config = AppConfig.parse(config_raw)
    if config.experiment.kind != "evaluation_validity":
        raise ValueError("multi-seed review accepts only evaluation_validity runs")
    replicates = config.experiment.bootstrap_replicates
    if replicates <= 0:
        raise ValueError(
            "multi-seed review requires positive bootstrap replicates per run"
        )
    run_id = manifest.get("run_id")
    if (
        not isinstance(run_id, str)
        or not run_id
        or run_id != root.name
        or manifest.get("status") != "complete"
    ):
        raise ValueError("source run manifest identity/status is invalid")
    source_code_sha256 = _validate_digest(
        manifest.get("source_sha256"),
        "manifest.source_sha256",
    )
    rankings = _read_json_object(
        paths["rankings_sha256"],
        name="Experiment C rankings",
    )
    observations, bootstrap = _validate_ranking(
        rankings,
        systems=tuple(config.experiment.updaters),
        replicates=replicates,
        tie_tolerance=config.thresholds.ranking_tie_tolerance,
    )
    gate_report = _read_json_object(
        paths["gate_report_sha256"],
        name="gate report",
    )
    gate_5 = _validate_gate_report(gate_report)
    observations["gate_5"] = gate_5
    summary = _read_json_object(paths["summary_sha256"], name="run summary")
    if (
        summary.get("experiment") != "C"
        or summary.get("scientific_claim_status") != CLAIM_STATUS
        or summary.get("gate_5_computed_status") != gate_5["computed_status"]
    ):
        raise ValueError("Experiment C summary/gate binding is invalid")

    scientific = _normalized_scientific_config(config)
    scientific_sha256 = _digest(scientific)
    source_artifacts = {
        "run_directory_name": root.name,
        **{
            name: _file_digest(file_path)
            for name, file_path in paths.items()
        },
        "manifest_config_sha256": _validate_digest(
            manifest.get("config_sha256"),
            "manifest.config_sha256",
        ),
    }
    entry = {
        "run_id": run_id,
        "seed": config.run.seed,
        "scientific_config_sha256": scientific_sha256,
        "source_code_sha256": source_code_sha256,
        "source_artifacts": source_artifacts,
        "bootstrap_evidence": {
            "seed": config.run.seed,
            **bootstrap,
        },
        "observed_results": observations,
    }
    return entry, config, scientific


def _load_sources(
    paths: Sequence[str | Path],
) -> tuple[list[dict[str, Any]], dict[str, Any], AppConfig]:
    material = tuple(paths)
    if not 2 <= len(material) <= MAX_SOURCE_RUNS:
        raise ValueError(
            "multi-seed review requires between 2 and "
            f"{MAX_SOURCE_RUNS} source runs"
        )
    resolved_paths = []
    for path in material:
        supplied = Path(path)
        if supplied.is_symlink():
            raise ValueError(f"source run cannot be a symlink: {supplied}")
        resolved_paths.append(supplied.resolve())
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError("source run paths must be distinct")

    loaded = [_load_source(path) for path in resolved_paths]
    entries = [item[0] for item in loaded]
    configurations = [item[1] for item in loaded]
    scientific = [item[2] for item in loaded]
    seeds = [entry["seed"] for entry in entries]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Experiment C source runs must use distinct run.seed values")
    run_ids = [entry["run_id"] for entry in entries]
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("Experiment C source run IDs must be distinct")
    if any(candidate != scientific[0] for candidate in scientific[1:]):
        raise ValueError(
            "Experiment C source runs have incompatible scientific configs; "
            "only run.name, run.seed, and run.output_root may differ"
        )
    updater_sets = {
        tuple(config.experiment.updaters) for config in configurations
    }
    if len(updater_sets) != 1:
        raise ValueError("Experiment C source updater sets differ")
    ranking_thresholds = {
        config.thresholds.ranking_tie_tolerance
        for config in configurations
    }
    if len(ranking_thresholds) != 1:
        raise ValueError("Experiment C source ranking thresholds differ")
    source_code_digests = {
        entry["source_code_sha256"] for entry in entries
    }
    if len(source_code_digests) != 1:
        raise ValueError("Experiment C source runs use different source trees")
    entries.sort(key=lambda entry: (entry["seed"], entry["run_id"]))
    return entries, scientific[0], configurations[0]


def _fraction(numerator: int, denominator: int) -> dict[str, Any]:
    if (
        isinstance(numerator, bool)
        or isinstance(denominator, bool)
        or not isinstance(numerator, int)
        or not isinstance(denominator, int)
        or denominator <= 0
        or numerator < 0
        or numerator > denominator
    ):
        raise ValueError("invalid exact proportion")
    divisor = gcd(numerator, denominator)
    return {
        "numerator": numerator,
        "denominator": denominator,
        "fraction": f"{numerator // divisor}/{denominator // divisor}",
    }


def _dimension_value(entry: Mapping[str, Any], dimension: str) -> Any:
    observed = entry["observed_results"]
    paths = {
        "point_ranking.fixed_balanced_development": (
            "point_rankings",
            "fixed_balanced_development",
        ),
        "point_ranking.fixed_biased_development": (
            "point_rankings",
            "fixed_biased_development",
        ),
        "point_ranking.endogenous_closed_loop_development": (
            "point_rankings",
            "endogenous_closed_loop_development",
        ),
        "inferential_top_tier.fixed_balanced_development": (
            "inferential",
            "open_top_tier",
        ),
        "inferential_top_tier.endogenous_closed_loop_development": (
            "inferential",
            "closed_top_tier",
        ),
        "inferential_partial_order.fixed_balanced_development": (
            "inferential",
            "open_partial_order",
        ),
        "inferential_partial_order.endogenous_closed_loop_development": (
            "inferential",
            "closed_partial_order",
        ),
        "gate_5.decision_and_status": ("gate_5",),
        "esr.development_selection_sets": ("esr_selection_sets",),
    }
    value: Any = observed
    for component in paths[dimension]:
        value = value[component]
    return value


def _comparison_summary(
    entries: Sequence[Mapping[str, Any]],
    dimension: str,
) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {}
    valued_entries: list[tuple[Mapping[str, Any], Any, str]] = []
    for entry in entries:
        value = _dimension_value(entry, dimension)
        signature = _digest(value)
        valued_entries.append((entry, value, signature))
        group = groups.setdefault(
            signature,
            {
                "value_sha256": signature,
                "value": value,
                "run_ids": [],
                "seeds": [],
            },
        )
        if group["value"] != value:
            raise ValueError("SHA-256 collision in comparison values")
        group["run_ids"].append(entry["run_id"])
        group["seeds"].append(entry["seed"])
    source_count = len(entries)
    patterns = []
    for signature in sorted(groups):
        group = groups[signature]
        count = len(group["run_ids"])
        patterns.append(
            {
                **group,
                "count": count,
                "proportion": _fraction(count, source_count),
            }
        )
    modal_count = max(pattern["count"] for pattern in patterns)
    agreements = 0
    disagreements = []
    for index, (first, first_value, first_signature) in enumerate(valued_entries):
        for second, second_value, second_signature in valued_entries[index + 1 :]:
            if first_signature == second_signature:
                agreements += 1
                continue
            disagreements.append(
                {
                    "first_run_id": first["run_id"],
                    "first_seed": first["seed"],
                    "first_value": first_value,
                    "second_run_id": second["run_id"],
                    "second_seed": second["seed"],
                    "second_value": second_value,
                }
            )
    pair_count = comb(source_count, 2)
    return {
        "unanimous": len(patterns) == 1,
        "source_count": source_count,
        "distinct_pattern_count": len(patterns),
        "modal_pattern_sha256s": sorted(
            pattern["value_sha256"]
            for pattern in patterns
            if pattern["count"] == modal_count
        ),
        "modal_stability_proportion": _fraction(modal_count, source_count),
        "pairwise_agreement_proportion": _fraction(agreements, pair_count),
        "patterns": patterns,
        "disagreements": disagreements,
    }


def _comparisons(
    entries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        dimension: _comparison_summary(entries, dimension)
        for dimension in COMPARISON_DIMENSIONS
    }


def _overall(comparisons: Mapping[str, Any]) -> dict[str, Any]:
    count = len(COMPARISON_DIMENSIONS)
    unanimous = sum(
        bool(comparisons[dimension]["unanimous"])
        for dimension in COMPARISON_DIMENSIONS
    )
    return {
        "dimension_count": count,
        "unanimous_dimension_count": unanimous,
        "unanimous_dimension_proportion": _fraction(unanimous, count),
        "all_predeclared_dimensions_unanimous": unanimous == count,
        "scientific_claim_inferred": False,
    }


def _bootstrap_summary(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "evidence_unit": (
            "one independently seeded Experiment C run whose retained ranking "
            "analysis uses complete-latent-user clustered bootstrap summaries"
        ),
        "pooling_performed": False,
        "all_replicate_counts_positive": all(
            entry["bootstrap_evidence"]["replicate_count"] > 0
            for entry in entries
        ),
        "runs": [
            {
                "run_id": entry["run_id"],
                **entry["bootstrap_evidence"],
            }
            for entry in entries
        ],
    }


def _validate_source_entry(entry: Mapping[str, Any]) -> None:
    _exact_fields(
        entry,
        {
            "run_id",
            "seed",
            "scientific_config_sha256",
            "source_code_sha256",
            "source_artifacts",
            "bootstrap_evidence",
            "observed_results",
        },
        name="review source run",
    )
    if not isinstance(entry["run_id"], str) or not entry["run_id"]:
        raise ValueError("review source run_id is invalid")
    if (
        isinstance(entry["seed"], bool)
        or not isinstance(entry["seed"], int)
        or entry["seed"] < 0
    ):
        raise ValueError("review source seed is invalid")
    _validate_digest(
        entry["scientific_config_sha256"],
        "review scientific_config_sha256",
    )
    _validate_digest(entry["source_code_sha256"], "review source_code_sha256")
    artifacts = entry["source_artifacts"]
    if not isinstance(artifacts, Mapping):
        raise ValueError("review source_artifacts must be an object")
    _exact_fields(
        artifacts,
        {
            "run_directory_name",
            "checksum_manifest_sha256",
            "manifest_sha256",
            "config_resolved_sha256",
            "rankings_sha256",
            "gate_report_sha256",
            "summary_sha256",
            "manifest_config_sha256",
        },
        name="review source artifacts",
    )
    if artifacts["run_directory_name"] != entry["run_id"]:
        raise ValueError("review source directory/run ID mismatch")
    for field, value in artifacts.items():
        if field != "run_directory_name":
            _validate_digest(value, f"review source_artifacts.{field}")
    bootstrap = entry["bootstrap_evidence"]
    if not isinstance(bootstrap, Mapping):
        raise ValueError("review bootstrap_evidence must be an object")
    _exact_fields(
        bootstrap,
        {
            "seed",
            "seeded_by_run_config",
            "replicate_count",
            "inference_unit",
            "alignment_key",
            "method",
            "development_cluster_count",
            "test_cluster_count",
            "retained_summary_sha256",
        },
        name="review bootstrap evidence",
    )
    if (
        bootstrap["seed"] != entry["seed"]
        or bootstrap["seeded_by_run_config"] is not True
        or bootstrap["inference_unit"] != "complete_latent_user_cluster"
        or bootstrap["alignment_key"] != _ALIGNMENT_KEY
        or bootstrap["method"] != _BOOTSTRAP_METHOD
    ):
        raise ValueError("review bootstrap evidence semantics are invalid")
    for field in (
        "replicate_count",
        "development_cluster_count",
        "test_cluster_count",
    ):
        _positive_integer(bootstrap[field], f"review bootstrap {field}")
    _validate_digest(
        bootstrap["retained_summary_sha256"],
        "review bootstrap retained_summary_sha256",
    )
    observed = entry["observed_results"]
    if not isinstance(observed, Mapping):
        raise ValueError("review observed_results must be an object")
    _exact_fields(
        observed,
        {
            "point_rankings",
            "inferential",
            "esr_selection_sets",
            "gate_5",
        },
        name="review observed results",
    )
    point = observed["point_rankings"]
    if not isinstance(point, Mapping):
        raise ValueError("review point_rankings must be an object")
    _exact_fields(
        point,
        {
            "fixed_balanced_development",
            "fixed_biased_development",
            "endogenous_closed_loop_development",
        },
        name="review point rankings",
    )
    systems: tuple[str, ...] | None = None
    for name, ranks in point.items():
        if not isinstance(ranks, Mapping) or len(ranks) < 2:
            raise ValueError(f"review point ranking {name} is invalid")
        candidate_systems = tuple(sorted(ranks))
        if (
            not all(isinstance(system, str) and system for system in ranks)
            or (
                systems is not None
                and candidate_systems != systems
            )
        ):
            raise ValueError("review point ranking updater sets differ")
        systems = candidate_systems
        _validate_system_number_map(
            ranks,
            systems,
            name=f"review point ranking {name}",
            minimum=1.0,
            maximum=float(len(systems)),
        )
    if systems is None:
        raise ValueError("review point rankings are empty")
    inferential = observed["inferential"]
    if not isinstance(inferential, Mapping):
        raise ValueError("review inferential results must be an object")
    _exact_fields(
        inferential,
        {
            "open_top_tier",
            "closed_top_tier",
            "open_partial_order",
            "closed_partial_order",
        },
        name="review inferential results",
    )
    open_order = _validate_order(
        inferential["open_partial_order"],
        systems,
        name="review open_partial_order",
    )
    closed_order = _validate_order(
        inferential["closed_partial_order"],
        systems,
        name="review closed_partial_order",
    )
    if (
        inferential["open_top_tier"] != open_order[0]
        or inferential["closed_top_tier"] != closed_order[0]
    ):
        raise ValueError("review inferential top-tier/order binding mismatch")
    esr = observed["esr_selection_sets"]
    if not isinstance(esr, Mapping):
        raise ValueError("review ESR selection sets must be an object")
    _exact_fields(
        esr,
        {"open_selected_set", "closed_selected_set"},
        name="review ESR selection sets",
    )
    if (
        esr["open_selected_set"] != open_order[0]
        or esr["closed_selected_set"] != closed_order[0]
    ):
        raise ValueError("review ESR/top-tier binding mismatch")
    gate = observed["gate_5"]
    if not isinstance(gate, Mapping):
        raise ValueError("review Gate 5 result must be an object")
    _exact_fields(
        gate,
        {"decision", "computed_status"},
        name="review Gate 5 result",
    )
    decision = gate["decision"]
    if not (decision is None or isinstance(decision, bool)):
        raise ValueError("review Gate 5 decision must be Boolean or null")
    expected_status = (
        "incomplete"
        if decision is None
        else (
            "meets_computational_checks"
            if decision
            else "does_not_meet_checks"
        )
    )
    if gate["computed_status"] != expected_status:
        raise ValueError("review Gate 5 decision/status mismatch")


def _validate_review_payload(review: Mapping[str, Any]) -> None:
    _exact_fields(
        review,
        {
            "schema_version",
            "artifact_kind",
            "artifact_id",
            "claim_status",
            "review_scope",
            "source_count",
            "scientific_configuration",
            "validation",
            "source_runs",
            "bootstrap_evidence",
            "comparisons",
            "overall",
            "interpretation_boundary",
        },
        name="multi-seed review",
    )
    if (
        review["schema_version"] != 1
        or review["artifact_kind"] != ARTIFACT_KIND
        or review["claim_status"] != CLAIM_STATUS
        or review["review_scope"]
        != "descriptive_cross_seed_reproducibility_diagnostic"
    ):
        raise ValueError("multi-seed review identity/claim semantics are invalid")
    source_count = review["source_count"]
    if (
        isinstance(source_count, bool)
        or not isinstance(source_count, int)
        or not 2 <= source_count <= MAX_SOURCE_RUNS
    ):
        raise ValueError("multi-seed review source_count is invalid")
    source_runs = review["source_runs"]
    if not isinstance(source_runs, list) or len(source_runs) != source_count:
        raise ValueError("multi-seed review source run count mismatch")
    for entry in source_runs:
        if not isinstance(entry, Mapping):
            raise ValueError("multi-seed review source entries must be objects")
        _validate_source_entry(entry)
    seeds = [entry["seed"] for entry in source_runs]
    run_ids = [entry["run_id"] for entry in source_runs]
    if (
        len(set(seeds)) != source_count
        or len(set(run_ids)) != source_count
        or source_runs
        != sorted(source_runs, key=lambda entry: (entry["seed"], entry["run_id"]))
    ):
        raise ValueError("multi-seed review source identities are invalid")
    configuration = review["scientific_configuration"]
    if not isinstance(configuration, Mapping):
        raise ValueError("scientific_configuration must be an object")
    _exact_fields(
        configuration,
        {
            "sha256",
            "permitted_run_differences",
            "updater_ids",
            "ranking_tie_tolerance",
            "source_code_sha256",
        },
        name="scientific_configuration",
    )
    _validate_digest(configuration["sha256"], "scientific config digest")
    _validate_digest(
        configuration["source_code_sha256"],
        "scientific source code digest",
    )
    if configuration["permitted_run_differences"] != [
        "run.name",
        "run.seed",
        "run.output_root",
    ]:
        raise ValueError("scientific config normalization contract changed")
    if (
        not isinstance(configuration["updater_ids"], list)
        or len(configuration["updater_ids"]) < 2
        or not all(
            isinstance(value, str) and value
            for value in configuration["updater_ids"]
        )
        or len(configuration["updater_ids"])
        != len(set(configuration["updater_ids"]))
    ):
        raise ValueError("scientific_configuration updater_ids are invalid")
    if any(
        set(entry["observed_results"]["point_rankings"][
            "fixed_balanced_development"
        ])
        != set(configuration["updater_ids"])
        for entry in source_runs
    ):
        raise ValueError("review source observation/updater binding mismatch")
    _finite_number(
        configuration["ranking_tie_tolerance"],
        "scientific ranking_tie_tolerance",
    )
    if any(
        entry["scientific_config_sha256"] != configuration["sha256"]
        or entry["source_code_sha256"]
        != configuration["source_code_sha256"]
        for entry in source_runs
    ):
        raise ValueError("review source/configuration bindings differ")
    validation = review["validation"]
    if not isinstance(validation, Mapping):
        raise ValueError("review validation must be an object")
    expected_validation = {
        "source_runs_verified_complete": True,
        "source_runs_unchanged_during_write": True,
        "source_runs_mutated": False,
        "scientific_configuration_identical": True,
        "updater_sets_identical": True,
        "ranking_thresholds_identical": True,
        "source_code_identical": True,
        "distinct_seeds": True,
        "positive_seeded_clustered_bootstraps": True,
    }
    if dict(validation) != expected_validation:
        raise ValueError("review validation assertions are incomplete")
    expected_bootstrap = _bootstrap_summary(source_runs)
    if review["bootstrap_evidence"] != expected_bootstrap:
        raise ValueError("review bootstrap aggregate mismatch")
    comparisons = review["comparisons"]
    if not isinstance(comparisons, Mapping) or set(comparisons) != set(
        COMPARISON_DIMENSIONS
    ):
        raise ValueError("review comparison dimensions differ")
    expected_comparisons = _comparisons(source_runs)
    if comparisons != expected_comparisons:
        raise ValueError("review comparison aggregate mismatch")
    if review["overall"] != _overall(expected_comparisons):
        raise ValueError("review overall aggregate mismatch")
    if review["overall"]["scientific_claim_inferred"] is not False:
        raise ValueError("multi-seed review cannot infer a scientific claim")
    artifact_id = review["artifact_id"]
    _validate_digest(artifact_id, "review artifact_id")
    core = dict(review)
    core.pop("artifact_id")
    if artifact_id != _digest(core):
        raise ValueError("multi-seed review artifact_id mismatch")


def _write_durable(path: Path, value: str) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def create_experiment_c_multiseed_review(
    source_run_dirs: Sequence[str | Path],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Create one atomic, checksum-bound descriptive cross-seed review."""

    supplied_output = Path(output_dir)
    if supplied_output.is_symlink() or supplied_output.exists():
        raise FileExistsError(
            f"multi-seed review output already exists: {supplied_output}"
        )
    output = supplied_output.resolve()
    if output == Path(output.anchor):
        raise ValueError("multi-seed review output cannot be a filesystem root")
    entries, scientific, exemplar = _load_sources(source_run_dirs)
    source_roots = [Path(path).resolve() for path in source_run_dirs]
    for source in source_roots:
        if output == source or source in output.parents:
            raise ValueError(
                "multi-seed review output must be outside every immutable "
                f"source run: {source}"
            )
    parent = output.parent
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink():
        raise ValueError("multi-seed review output parent cannot be a symlink")
    lock = parent / f".{output.name}.multiseed.lock"
    try:
        lock_descriptor = os.open(
            lock,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
    except FileExistsError as exc:
        raise FileExistsError(
            f"multi-seed review output is locked: {output}"
        ) from exc
    os.close(lock_descriptor)
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.",
            suffix=".staging",
            dir=parent,
        )
    )
    published = False
    try:
        scientific_sha256 = _digest(scientific)
        comparisons = _comparisons(entries)
        review_core = {
            "schema_version": 1,
            "artifact_kind": ARTIFACT_KIND,
            "claim_status": CLAIM_STATUS,
            "review_scope": (
                "descriptive_cross_seed_reproducibility_diagnostic"
            ),
            "source_count": len(entries),
            "scientific_configuration": {
                "sha256": scientific_sha256,
                "permitted_run_differences": [
                    "run.name",
                    "run.seed",
                    "run.output_root",
                ],
                "updater_ids": list(exemplar.experiment.updaters),
                "ranking_tie_tolerance": (
                    exemplar.thresholds.ranking_tie_tolerance
                ),
                "source_code_sha256": entries[0]["source_code_sha256"],
            },
            "validation": {
                "source_runs_verified_complete": True,
                "source_runs_unchanged_during_write": True,
                "source_runs_mutated": False,
                "scientific_configuration_identical": True,
                "updater_sets_identical": True,
                "ranking_thresholds_identical": True,
                "source_code_identical": True,
                "distinct_seeds": True,
                "positive_seeded_clustered_bootstraps": True,
            },
            "source_runs": entries,
            "bootstrap_evidence": _bootstrap_summary(entries),
            "comparisons": comparisons,
            "overall": _overall(comparisons),
            "interpretation_boundary": (
                "Agreement and disagreement are reported exactly across "
                "verified seeded runs. No bootstrap samples are pooled, no "
                "favorable seed is selected, and this diagnostic does not "
                "establish a ranking, Gate 5, ESR, robustness, or paper claim."
            ),
        }
        artifact_id = _digest(review_core)
        review = {**review_core, "artifact_id": artifact_id}
        _write_durable(
            stage / "review.json",
            canonical_json(review) + "\n",
        )
        manifest = {
            "schema_version": 1,
            "artifact_kind": ARTIFACT_KIND,
            "artifact_id": artifact_id,
            "status": "complete",
            "claim_status": CLAIM_STATUS,
            "source_count": len(entries),
            "scientific_config_sha256": scientific_sha256,
            "review_sha256": _file_digest(stage / "review.json"),
            "source_checksum_manifest_sha256s": {
                entry["run_id"]: entry["source_artifacts"][
                    "checksum_manifest_sha256"
                ]
                for entry in entries
            },
        }
        _write_durable(
            stage / "manifest.json",
            canonical_json(manifest) + "\n",
        )
        checksum_lines = "".join(
            f"{_file_digest(stage / relative)}  {relative}\n"
            for relative in sorted(REVIEW_FILES)
        )
        _write_durable(stage / CHECKSUM_PATH, checksum_lines)
        _fsync_directory(stage)

        # Re-verify every source after all output bytes exist. This detects a
        # concurrent source mutation before the complete artifact is exposed.
        after_entries, after_scientific, _ = _load_sources(source_run_dirs)
        if after_entries != entries or after_scientific != scientific:
            raise ValueError("a source run changed while the review was written")
        valid, errors = verify_experiment_c_multiseed_review(stage)
        if not valid:
            raise ValueError(
                "staged multi-seed review failed verification: "
                + "; ".join(errors)
            )
        if output.exists() or output.is_symlink():
            raise FileExistsError(
                f"multi-seed review output already exists: {output}"
            )
        os.rename(stage, output)
        published = True
        _fsync_directory(parent)
    finally:
        if not published and stage.exists():
            shutil.rmtree(stage)
        try:
            lock.unlink()
        except FileNotFoundError:
            pass
    return {
        "artifact_id": artifact_id,
        "output_dir": str(output),
        "source_count": len(entries),
        "distinct_seed_count": len(entries),
        "all_predeclared_dimensions_unanimous": review["overall"][
            "all_predeclared_dimensions_unanimous"
        ],
        "claim_status": CLAIM_STATUS,
    }


def _verify_checksum_manifest(root: Path) -> list[str]:
    errors: list[str] = []
    checksum = root / CHECKSUM_PATH
    if checksum.is_symlink() or not checksum.is_file():
        return ["missing regular SHA256SUMS"]
    try:
        text = checksum.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        return [f"cannot read SHA256SUMS: {exc}"]
    retained: set[str] = set()
    observed_order: list[str] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            errors.append(f"blank checksum line {line_number}")
            continue
        try:
            expected, relative = line.split("  ", 1)
            _validate_digest(expected, "checksum")
        except ValueError as exc:
            errors.append(f"malformed checksum line {line_number}: {exc}")
            continue
        pure = PurePosixPath(relative)
        if (
            not relative
            or pure.is_absolute()
            or pure.as_posix() != relative
            or "\\" in relative
            or "\x00" in relative
            or any(part in {"", ".", ".."} for part in pure.parts)
        ):
            errors.append(f"unsafe checksum path on line {line_number}")
            continue
        if relative in retained:
            errors.append(f"duplicate checksum path: {relative}")
            continue
        retained.add(relative)
        observed_order.append(relative)
        candidate = root.joinpath(*pure.parts)
        if candidate.is_symlink() or not candidate.is_file():
            errors.append(f"missing regular artifact: {relative}")
            continue
        resolved = candidate.resolve()
        if resolved == root or root not in resolved.parents:
            errors.append(f"checksum path escapes review: {relative}")
            continue
        if _file_digest(candidate) != expected:
            errors.append(f"checksum mismatch: {relative}")
    if retained != REVIEW_FILES:
        errors.append("review checksum manifest has an unexpected file set")
    if observed_order != sorted(observed_order):
        errors.append("review checksum manifest paths are not sorted")
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != checksum
    }
    for relative in sorted(actual_files - retained):
        errors.append(f"unlisted artifact: {relative}")
    unexpected_nodes = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_symlink() or (path.is_dir() and path != root)
    ]
    for relative in sorted(unexpected_nodes):
        errors.append(f"unexpected review node: {relative}")
    return errors


def verify_experiment_c_multiseed_review(
    review_dir: str | Path,
    *,
    source_run_dirs: Sequence[str | Path] | None = None,
) -> tuple[bool, tuple[str, ...]]:
    """Verify checksums, review aggregates, and optional exact source bindings."""

    supplied = Path(review_dir)
    if supplied.is_symlink():
        return False, ("review directory cannot be a symlink",)
    root = supplied.resolve()
    if not root.is_dir():
        return False, ("review directory is missing",)
    errors = _verify_checksum_manifest(root)
    try:
        review = _read_json_object(root / "review.json", name="review.json")
        manifest = _read_json_object(
            root / "manifest.json",
            name="manifest.json",
        )
        _validate_review_payload(review)
        _exact_fields(
            manifest,
            {
                "schema_version",
                "artifact_kind",
                "artifact_id",
                "status",
                "claim_status",
                "source_count",
                "scientific_config_sha256",
                "review_sha256",
                "source_checksum_manifest_sha256s",
            },
            name="multi-seed manifest",
        )
        expected_checksum_bindings = {
            entry["run_id"]: entry["source_artifacts"][
                "checksum_manifest_sha256"
            ]
            for entry in review["source_runs"]
        }
        if (
            manifest["schema_version"] != 1
            or manifest["artifact_kind"] != ARTIFACT_KIND
            or manifest["artifact_id"] != review["artifact_id"]
            or manifest["status"] != "complete"
            or manifest["claim_status"] != CLAIM_STATUS
            or manifest["source_count"] != review["source_count"]
            or manifest["scientific_config_sha256"]
            != review["scientific_configuration"]["sha256"]
            or manifest["review_sha256"]
            != _file_digest(root / "review.json")
            or manifest["source_checksum_manifest_sha256s"]
            != expected_checksum_bindings
        ):
            raise ValueError("multi-seed manifest/review binding mismatch")
    except (KeyError, OSError, TypeError, ValueError) as exc:
        errors.append(f"invalid retained review content: {exc}")
        return False, tuple(errors)

    if source_run_dirs is not None:
        try:
            entries, scientific, exemplar = _load_sources(source_run_dirs)
            if entries != review["source_runs"]:
                errors.append("review/source run artifact binding mismatch")
            if _digest(scientific) != review["scientific_configuration"]["sha256"]:
                errors.append("review/source scientific config mismatch")
            if list(exemplar.experiment.updaters) != review[
                "scientific_configuration"
            ]["updater_ids"]:
                errors.append("review/source updater binding mismatch")
        except (OSError, TypeError, ValueError) as exc:
            errors.append(f"source run re-verification failed: {exc}")
    return not errors, tuple(errors)
