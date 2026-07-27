"""Provider-bound direct-statement evidence for H7.

The Experiment A choice runner cannot honestly manufacture a volunteered
preference.  This module supplies the missing external boundary:

* a verified Experiment A run deterministically yields direct-statement
  requests for every retained test user, domain, and preference attribute;
* ordinary full-context and provenance-aware requests are exactly paired;
* accepted OpenAI or OpenRouter audit records must exactly bind every replay
  response before it can become a :class:`VolunteeredPreferenceUpdate`; and
* a derived review recomputes H7's volunteered and overall Experiment A
  criteria without modifying the checksummed source run.

All plans and reviews retain ``claim_status = "not_claimed"``.  Missing,
rejected, duplicated, or mismatched evidence is an error, never an imputation.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import json
import math

from .artifacts import canonical_json, verify_run
from .domains import get_domain
from .experiments.hypothesis_estimands import (
    H7_VALID_LEARNING_RETENTION_FRACTION,
    MINIMUM_CLUSTER_COUNT,
    VolunteeredPreferenceUpdate,
    analyze_h7_volunteered_updates,
)
from .llm_exchange import (
    ATTRIBUTES,
    VALUES,
    LLMRequest,
    LLMResponse,
    read_responses,
)
from .openai_provider import read_requests
from .schemas import NUM_ATTRIBUTES, validate_theta


H7_VOLUNTEERED_PLAN_VERSION = "h7-volunteered-control-plan-v1"
H7_VOLUNTEERED_REVIEW_VERSION = "h7-volunteered-control-review-v1"
H7_SCHEMA_VERSION = 1
CLAIM_STATUS = "not_claimed"

BASELINE_UPDATER_ID = "llm_full_context"
MITIGATION_UPDATER_ID = "llm_provenance_aware"
ROLE_VIEWS: tuple[tuple[str, str], ...] = (
    (BASELINE_UPDATER_ID, "full_context"),
    (MITIGATION_UPDATER_ID, "provenance_aware"),
)

PLAN_FILENAME = "h7-volunteered-plan.json"
BINDINGS_FILENAME = "h7-volunteered-request-bindings.jsonl"
REQUESTS_FILENAME = "h7-volunteered-requests.jsonl"

ProbabilityRow = tuple[float, float, float, float]
ProbabilityRows = tuple[ProbabilityRow, ProbabilityRow, ProbabilityRow]

_SOURCE_FIELDS = {
    "run_id",
    "manifest_sha256",
    "config_file_sha256",
    "checksums_sha256",
    "population_sha256",
    "experiment_a_metrics_sha256",
    "hypothesis_estimands_sha256",
}
_POPULATION_FIELDS = {
    "schema_version",
    "user_id",
    "domain",
    "theta",
    "susceptibility",
    "split",
}


def _digest(value: Any) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class _FileSnapshot:
    """Exact bytes and named-path identity used by one derived review."""

    supplied: Path
    resolved: Path
    material: bytes
    label: str

    @property
    def sha256(self) -> str:
        return sha256(self.material).hexdigest()

    def verify_unchanged(self) -> None:
        supplied = self.supplied.absolute()
        if (
            supplied.is_symlink()
            or not supplied.is_file()
            or supplied.resolve() != self.resolved
            or supplied.read_bytes() != self.material
        ):
            raise ValueError(
                f"{self.label} changed while the H7 review was running"
            )


def _snapshot_regular_file(path: Path, *, label: str) -> _FileSnapshot:
    supplied = path.absolute()
    if supplied.is_symlink() or not supplied.is_file():
        raise ValueError(f"{label} must be a regular file, not a symlink")
    resolved = supplied.resolve()
    material = supplied.read_bytes()
    snapshot = _FileSnapshot(
        supplied=supplied,
        resolved=resolved,
        material=material,
        label=label,
    )
    snapshot.verify_unchanged()
    return snapshot


def _require_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _validate_digest(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _read_json_object(path: Path | bytes) -> Mapping[str, Any]:
    if isinstance(path, bytes):
        source_label = "<json-bytes>"
        try:
            text = path.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"{source_label}: input must be valid UTF-8"
            ) from exc
    else:
        source_label = str(path)
        text = path.read_text(encoding="utf-8")
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{source_label}: {exc}") from exc
    if not isinstance(decoded, Mapping):
        raise ValueError(f"{source_label}: expected a JSON object")
    return decoded


def _read_jsonl_objects(
    path: Path | bytes,
) -> tuple[Mapping[str, Any], ...]:
    if isinstance(path, bytes):
        source_label = "<jsonl-bytes>"
        try:
            lines = path.decode("utf-8").splitlines()
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"{source_label}: input must be valid UTF-8"
            ) from exc
    else:
        source_label = str(path)
        lines = path.read_text(encoding="utf-8").splitlines()
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            decoded = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{source_label}:{line_number}: {exc}"
            ) from exc
        if not isinstance(decoded, Mapping):
            raise ValueError(
                f"{source_label}:{line_number}: row must be an object"
            )
        rows.append(decoded)
    return tuple(rows)


def _uniform_rows() -> ProbabilityRows:
    row: ProbabilityRow = (0.25, 0.25, 0.25, 0.25)
    return (row, row, row)


def _rows_to_beliefs(rows: ProbabilityRows) -> dict[str, dict[str, float]]:
    return {
        attribute: {
            value: float(probability)
            for value, probability in zip(VALUES, rows[index])
        }
        for index, attribute in enumerate(ATTRIBUTES)
    }


def _rows_from_response(response: LLMResponse) -> ProbabilityRows:
    result: list[ProbabilityRow] = []
    for attribute in ATTRIBUTES:
        row = tuple(
            float(response.beliefs[attribute][value])
            for value in VALUES
        )
        result.append((row[0], row[1], row[2], row[3]))
    return (result[0], result[1], result[2])


def _sign_mass(
    rows: ProbabilityRows,
    *,
    attribute: int,
    direction: int,
) -> float:
    indexes = (0, 1) if direction < 0 else (2, 3)
    return rows[attribute][indexes[0]] + rows[attribute][indexes[1]]


def _clipped_logit(probability: float, *, clip: float = 1e-6) -> float:
    bounded = min(max(probability, clip), 1.0 - clip)
    return math.log(bounded / (1.0 - bounded))


def _directional_log_odds_update(
    prior: ProbabilityRows,
    posterior: ProbabilityRows,
    *,
    attribute: int,
    direction: int,
) -> float:
    return _clipped_logit(
        _sign_mass(
            posterior,
            attribute=attribute,
            direction=direction,
        )
    ) - _clipped_logit(
        _sign_mass(
            prior,
            attribute=attribute,
            direction=direction,
        )
    )


@dataclass(frozen=True, slots=True)
class H7VolunteeredCase:
    """One simulator-grounded, user-originated direct-statement case."""

    case_id: str
    user_id: str
    domain_id: str
    target_attribute: int
    target_direction: int
    surface_statement: str
    source_user_sha256: str
    prior_probabilities: ProbabilityRows = _uniform_rows()
    case_sha256: str = ""

    def __post_init__(self) -> None:
        for name in ("case_id", "user_id", "domain_id", "surface_statement"):
            _require_text(getattr(self, name), name)
        if (
            isinstance(self.target_attribute, bool)
            or not isinstance(self.target_attribute, int)
            or not 0 <= self.target_attribute < NUM_ATTRIBUTES
        ):
            raise ValueError("target_attribute must be in [0, 3)")
        if self.target_direction not in (-1, 1):
            raise ValueError("target_direction must be -1 or +1")
        _validate_digest(self.source_user_sha256, "source_user_sha256")
        domain = get_domain(self.domain_id)
        direction_label = domain.attributes[
            self.target_attribute
        ].label_for(self.target_direction)
        expected_statement = (
            f"I generally prefer {direction_label}."
        )
        if self.surface_statement != expected_statement:
            raise ValueError(
                "surface_statement does not match the registered domain "
                "attribute and direction"
            )
        expected_id = (
            f"h7-volunteered:{self.user_id}:{self.domain_id}:"
            f"attribute-{self.target_attribute + 1}"
        )
        if self.case_id != expected_id:
            raise ValueError("case_id does not match the deterministic case key")
        if self.prior_probabilities != _uniform_rows():
            raise ValueError("H7 volunteered cases use the frozen uniform prior")
        expected = _digest(self._binding_payload())
        if self.case_sha256 and self.case_sha256 != expected:
            raise ValueError("case_sha256 does not bind the case content")
        object.__setattr__(self, "case_sha256", expected)

    def _binding_payload(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "user_id": self.user_id,
            "domain_id": self.domain_id,
            "target_attribute": self.target_attribute,
            "target_direction": self.target_direction,
            "surface_statement": self.surface_statement,
            "source_user_sha256": self.source_user_sha256,
            "prior_probabilities": _rows_to_beliefs(
                self.prior_probabilities
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": H7_SCHEMA_VERSION,
            **self._binding_payload(),
            "case_sha256": self.case_sha256,
        }


@dataclass(frozen=True, slots=True)
class H7VolunteeredRequestBinding:
    """A model-visible request plus withheld case and analysis bindings."""

    case_id: str
    case_sha256: str
    user_id: str
    updater_id: str
    view: str
    llm_request: LLMRequest
    binding_sha256: str = ""

    def __post_init__(self) -> None:
        for name in ("case_id", "user_id", "updater_id", "view"):
            _require_text(getattr(self, name), name)
        _validate_digest(self.case_sha256, "case_sha256")
        expected_roles = dict(ROLE_VIEWS)
        if expected_roles.get(self.updater_id) != self.view:
            raise ValueError("H7 updater/view role is not frozen")
        if (
            self.llm_request.updater_id != self.updater_id
            or self.llm_request.view != self.view
        ):
            raise ValueError("LLM request disagrees with its H7 role binding")
        expected = _digest(self._binding_payload())
        if self.binding_sha256 and self.binding_sha256 != expected:
            raise ValueError(
                "binding_sha256 does not bind the direct-statement request"
            )
        object.__setattr__(self, "binding_sha256", expected)

    def _binding_payload(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "case_sha256": self.case_sha256,
            "user_id": self.user_id,
            "updater_id": self.updater_id,
            "view": self.view,
            "llm_request": self.llm_request.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": H7_SCHEMA_VERSION,
            **self._binding_payload(),
            "binding_sha256": self.binding_sha256,
        }


def _request_for_case(
    case: H7VolunteeredCase,
    *,
    updater_id: str,
    view: str,
) -> H7VolunteeredRequestBinding:
    observation = {
        "event_sequence": [
            {
                "response_kind": "volunteered_statement",
                "surface_response": case.surface_statement,
                "selected_option": None,
            }
        ],
        "target_attribute": case.target_attribute,
    }
    context = {
        "event_sequence": [
            {
                "domain": case.domain_id,
                "scenario_id": (
                    f"h7-volunteered:{case.domain_id}:"
                    f"attribute-{case.target_attribute + 1}"
                ),
                "wording_template_id": (
                    "h7-volunteered-user-statement-v1"
                ),
                "options": [],
                "ranking": [],
                "question_type": "volunteered_statement",
                "target_attribute": case.target_attribute,
            }
        ]
    }
    provenance = (
        {
            "event_sequence": [
                {
                    "response_source": "user",
                    "elicitation_provenance": "user_originated_unprompted",
                }
            ]
        }
        if view == "provenance_aware"
        else None
    )
    draft = LLMRequest.build(
        request_id="content-addressed-h7-volunteered-request",
        updater_id=updater_id,
        view=view,
        prior=_rows_to_beliefs(case.prior_probabilities),
        observation=observation,
        context=context,
        provenance=provenance,
    )
    request = LLMRequest(
        request_id=f"h7:{case.case_sha256}:{updater_id}",
        updater_id=draft.updater_id,
        view=draft.view,
        payload=draft.payload,
        system_instruction=draft.system_instruction,
        prompt_sha256=draft.prompt_sha256,
    )
    return H7VolunteeredRequestBinding(
        case_id=case.case_id,
        case_sha256=case.case_sha256,
        user_id=case.user_id,
        updater_id=updater_id,
        view=view,
        llm_request=request,
    )


@dataclass(frozen=True, slots=True)
class H7VolunteeredCollectionPlan:
    """Complete, source-bound H7 direct-statement collection packet."""

    source_run: Mapping[str, str]
    cases: tuple[H7VolunteeredCase, ...]
    request_bindings: tuple[H7VolunteeredRequestBinding, ...]
    plan_sha256: str = ""

    def __post_init__(self) -> None:
        source = dict(self.source_run)
        if set(source) != _SOURCE_FIELDS:
            raise ValueError("H7 source-run binding fields are not exact")
        for name, value in source.items():
            if name == "run_id":
                _require_text(value, name)
            else:
                _validate_digest(value, name)
        cases = tuple(self.cases)
        bindings = tuple(self.request_bindings)
        if not cases:
            raise ValueError("H7 collection plan requires at least one case")
        if len({case.case_id for case in cases}) != len(cases):
            raise ValueError("H7 collection plan contains duplicate case IDs")
        if len({case.user_id for case in cases}) < MINIMUM_CLUSTER_COUNT:
            raise ValueError(
                "H7 collection requires at least two independent test users"
            )
        case_by_id = {case.case_id: case for case in cases}
        expected_keys = {
            (case.case_id, updater_id)
            for case in cases
            for updater_id, _ in ROLE_VIEWS
        }
        observed_keys = {
            (binding.case_id, binding.updater_id)
            for binding in bindings
        }
        if observed_keys != expected_keys or len(bindings) != len(expected_keys):
            raise ValueError(
                "H7 request bindings must exactly cross every case and role"
            )
        if len(
            {binding.llm_request.request_id for binding in bindings}
        ) != len(bindings):
            raise ValueError("H7 request IDs must be unique")
        for binding in bindings:
            case = case_by_id[binding.case_id]
            if (
                binding.case_sha256 != case.case_sha256
                or binding.user_id != case.user_id
            ):
                raise ValueError("H7 request binding disagrees with its case")
        object.__setattr__(self, "source_run", source)
        object.__setattr__(self, "cases", cases)
        object.__setattr__(self, "request_bindings", bindings)
        expected = _digest(self._binding_payload())
        if self.plan_sha256 and self.plan_sha256 != expected:
            raise ValueError("plan_sha256 does not bind the collection plan")
        object.__setattr__(self, "plan_sha256", expected)

    @property
    def requests(self) -> tuple[LLMRequest, ...]:
        return tuple(
            binding.llm_request for binding in self.request_bindings
        )

    def _binding_payload(self) -> dict[str, Any]:
        return {
            "plan_version": H7_VOLUNTEERED_PLAN_VERSION,
            "source_run": dict(self.source_run),
            "roles": [
                {"updater_id": updater_id, "view": view}
                for updater_id, view in ROLE_VIEWS
            ],
            "cases": [case.to_dict() for case in self.cases],
            "request_bindings": [
                binding.to_dict() for binding in self.request_bindings
            ],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": H7_SCHEMA_VERSION,
            "artifact_kind": "h7_volunteered_collection_plan",
            **self._binding_payload(),
            "case_count": len(self.cases),
            "request_count": len(self.request_bindings),
            "independent_user_count": len(
                {case.user_id for case in self.cases}
            ),
            "plan_sha256": self.plan_sha256,
            "claim_status": CLAIM_STATUS,
            "interpretation": (
                "This is a provider-neutral collection plan, not empirical "
                "evidence. Every planned response is required; no missing "
                "direct-statement outcome may be imputed."
            ),
        }


def _validated_test_population(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    test_rows: list[Mapping[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for index, raw in enumerate(rows):
        if set(raw) != _POPULATION_FIELDS:
            raise ValueError(
                f"population row {index} fields do not match schema version 1"
            )
        if raw["schema_version"] != 1:
            raise ValueError("population schema_version must be 1")
        user_id = _require_text(raw["user_id"], "population user_id")
        domain_id = _require_text(raw["domain"], "population domain")
        get_domain(domain_id)
        theta_raw = raw["theta"]
        if (
            not isinstance(theta_raw, Sequence)
            or isinstance(theta_raw, (str, bytes))
        ):
            raise ValueError("population theta must be an array")
        validate_theta(theta_raw)
        if raw["split"] not in {"train", "development", "test"}:
            raise ValueError("population split is invalid")
        if not isinstance(raw["susceptibility"], Mapping):
            raise ValueError("population susceptibility must be an object")
        if raw["split"] != "test":
            continue
        key = (user_id, domain_id)
        if key in seen:
            raise ValueError(f"duplicate test population row {key}")
        seen.add(key)
        test_rows.append(raw)
    if not test_rows:
        raise ValueError("source run has no retained test population")
    if len({str(row["user_id"]) for row in test_rows}) < MINIMUM_CLUSTER_COUNT:
        raise ValueError("H7 requires at least two retained test users")
    return tuple(
        sorted(
            test_rows,
            key=lambda row: (str(row["user_id"]), str(row["domain"])),
        )
    )


def build_h7_volunteered_collection_plan(
    population_rows: Sequence[Mapping[str, Any]],
    *,
    source_run: Mapping[str, str],
) -> H7VolunteeredCollectionPlan:
    """Build all direct-statement cases without sampling or cherry-picking."""

    cases: list[H7VolunteeredCase] = []
    for raw in _validated_test_population(population_rows):
        theta_raw = raw["theta"]
        assert isinstance(theta_raw, Sequence)
        theta = validate_theta(theta_raw)
        user_id = str(raw["user_id"])
        domain_id = str(raw["domain"])
        domain = get_domain(domain_id)
        source_user_sha256 = _digest(raw)
        for attribute in range(NUM_ATTRIBUTES):
            direction = 1 if theta[attribute] > 0 else -1
            statement = (
                "I generally prefer "
                f"{domain.attributes[attribute].label_for(direction)}."
            )
            cases.append(
                H7VolunteeredCase(
                    case_id=(
                        f"h7-volunteered:{user_id}:{domain_id}:"
                        f"attribute-{attribute + 1}"
                    ),
                    user_id=user_id,
                    domain_id=domain_id,
                    target_attribute=attribute,
                    target_direction=direction,
                    surface_statement=statement,
                    source_user_sha256=source_user_sha256,
                )
            )
    ordered_cases = tuple(
        sorted(
            cases,
            key=lambda case: (
                case.user_id,
                case.domain_id,
                case.target_attribute,
            ),
        )
    )
    bindings = tuple(
        _request_for_case(
            case,
            updater_id=updater_id,
            view=view,
        )
        for case in ordered_cases
        for updater_id, view in ROLE_VIEWS
    )
    return H7VolunteeredCollectionPlan(
        source_run=dict(source_run),
        cases=ordered_cases,
        request_bindings=bindings,
    )


@dataclass(frozen=True, slots=True)
class VerifiedH7Source:
    """Read-only materials retained by one verified Experiment A run."""

    run_dir: Path
    source_run: Mapping[str, str]
    plan: H7VolunteeredCollectionPlan
    hypothesis_estimands: Mapping[str, Any]
    bootstrap_replicates: int
    confidence_level: float
    seed: int
    _snapshots: tuple[_FileSnapshot, ...]

    def verify_unchanged(self) -> None:
        """Revalidate the source run and every byte snapshot used to load it."""

        for snapshot in self._snapshots:
            snapshot.verify_unchanged()
        valid, errors = verify_run(self.run_dir)
        if not valid:
            raise ValueError(
                "H7 source run changed during review: "
                + "; ".join(errors)
            )
        for snapshot in self._snapshots:
            snapshot.verify_unchanged()


def load_verified_h7_source(run_dir: str | Path) -> VerifiedH7Source:
    """Verify and load the exact source artifacts needed by H7 review."""

    supplied_root = Path(run_dir).absolute()
    if supplied_root.is_symlink() or not supplied_root.is_dir():
        raise ValueError(
            "H7 source run must be a directory, not a symlink"
        )
    root = supplied_root.resolve()
    valid, errors = verify_run(root)
    if not valid:
        raise ValueError(
            "H7 source run is not verified: " + "; ".join(errors)
        )
    paths = {
        "manifest": root / "manifest.json",
        "config": root / "config.resolved.json",
        "checksums": root / "SHA256SUMS",
        "population": root / "population" / "users.jsonl",
        "metrics": root / "metrics" / "experiment-a.jsonl",
        "hypothesis": (
            root / "metrics" / "experiment-a-hypothesis-estimands.json"
        ),
    }
    missing = sorted(
        str(path.relative_to(root))
        for path in paths.values()
        if not path.is_file()
    )
    if missing:
        raise ValueError(
            "H7 source run lacks required artifacts: " + ", ".join(missing)
        )
    snapshots = {
        name: _snapshot_regular_file(
            path,
            label=f"H7 source {name}",
        )
        for name, path in paths.items()
    }
    manifest = _read_json_object(snapshots["manifest"].material)
    config = _read_json_object(snapshots["config"].material)
    if config.get("experiment", {}).get("kind") != "provenance_audit":
        raise ValueError("H7 review requires an Experiment A source run")
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("H7 source manifest has no valid run_id")
    run_settings = config.get("run")
    if not isinstance(run_settings, Mapping):
        raise ValueError("H7 source config has no run settings")
    seed = run_settings.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("H7 source run seed must be an integer")

    population = _read_jsonl_objects(snapshots["population"].material)
    test_population = _validated_test_population(population)
    expected_pairs = {
        (str(row["user_id"]), str(row["domain"]))
        for row in test_population
    }
    metrics = _read_jsonl_objects(snapshots["metrics"].material)
    coverage: dict[str, set[tuple[str, str]]] = {
        updater_id: set()
        for updater_id, _ in ROLE_VIEWS
    }
    for row in metrics:
        if row.get("response_mode") != "controlled_anchor":
            continue
        updater_id = row.get("updater_id")
        if updater_id not in coverage:
            continue
        user_id = row.get("user_id")
        domain_id = row.get("domain")
        if isinstance(user_id, str) and isinstance(domain_id, str):
            coverage[updater_id].add((user_id, domain_id))
    missing_coverage = {
        updater_id: sorted(expected_pairs - observed)
        for updater_id, observed in coverage.items()
        if expected_pairs - observed
    }
    if missing_coverage:
        raise ValueError(
            "H7 source metrics lack controlled-anchor role coverage: "
            + canonical_json(missing_coverage)
        )

    hypothesis = _read_json_object(snapshots["hypothesis"].material)
    if (
        hypothesis.get("analysis") != "experiment_a_hypothesis_estimands"
        or hypothesis.get("claim_status") != CLAIM_STATUS
    ):
        raise ValueError(
            "H7 source hypothesis artifact has the wrong analysis or claim status"
        )
    replicates = hypothesis.get("bootstrap_replicates")
    confidence_level = hypothesis.get("confidence_level")
    if (
        isinstance(replicates, bool)
        or not isinstance(replicates, int)
        or replicates <= 0
    ):
        raise ValueError("H7 source bootstrap_replicates must be positive")
    if (
        isinstance(confidence_level, bool)
        or not isinstance(confidence_level, (int, float))
        or not 0.0 < float(confidence_level) < 1.0
    ):
        raise ValueError("H7 source confidence_level is invalid")

    source_run = {
        "run_id": run_id,
        "manifest_sha256": snapshots["manifest"].sha256,
        "config_file_sha256": snapshots["config"].sha256,
        "checksums_sha256": snapshots["checksums"].sha256,
        "population_sha256": snapshots["population"].sha256,
        "experiment_a_metrics_sha256": snapshots["metrics"].sha256,
        "hypothesis_estimands_sha256": snapshots["hypothesis"].sha256,
    }
    plan = build_h7_volunteered_collection_plan(
        population,
        source_run=source_run,
    )
    source = VerifiedH7Source(
        run_dir=root,
        source_run=source_run,
        plan=plan,
        hypothesis_estimands=hypothesis,
        bootstrap_replicates=replicates,
        confidence_level=float(confidence_level),
        seed=seed,
        _snapshots=tuple(snapshots.values()),
    )
    source.verify_unchanged()
    return source


def write_h7_plan_directory(
    output_dir: str | Path,
    plan: H7VolunteeredCollectionPlan,
) -> tuple[Path, Path, Path]:
    """Write the three deterministic provider-neutral plan files."""

    root = Path(output_dir)
    if root.exists():
        raise FileExistsError(f"H7 plan output already exists: {root}")
    root.mkdir(parents=True)
    plan_path = root / PLAN_FILENAME
    bindings_path = root / BINDINGS_FILENAME
    requests_path = root / REQUESTS_FILENAME
    plan_path.write_text(
        json.dumps(plan.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    bindings_path.write_text(
        "".join(
            canonical_json(binding.to_dict()) + "\n"
            for binding in plan.request_bindings
        ),
        encoding="utf-8",
    )
    requests_path.write_text(
        "".join(
            canonical_json(request.to_dict()) + "\n"
            for request in plan.requests
        ),
        encoding="utf-8",
    )
    return plan_path, bindings_path, requests_path


@dataclass(frozen=True, slots=True)
class H7ReviewInputSnapshots:
    """Exact external files consumed by one H7 review computation."""

    plan_root: Path
    plan_root_resolved: Path
    plan: _FileSnapshot
    bindings: _FileSnapshot
    requests: _FileSnapshot
    responses: _FileSnapshot
    provider_audit: _FileSnapshot

    def verify_unchanged(self) -> None:
        root = self.plan_root.absolute()
        if (
            root.is_symlink()
            or not root.is_dir()
            or root.resolve() != self.plan_root_resolved
        ):
            raise ValueError(
                "H7 plan directory changed while the review was running"
            )
        for snapshot in (
            self.plan,
            self.bindings,
            self.requests,
            self.responses,
            self.provider_audit,
        ):
            snapshot.verify_unchanged()


def snapshot_h7_review_inputs(
    plan_dir: str | Path,
    responses_path: str | Path,
    provider_audit_path: str | Path,
) -> H7ReviewInputSnapshots:
    """Capture the exact regular-file bytes that a review will parse."""

    root = Path(plan_dir).absolute()
    if root.is_symlink() or not root.is_dir():
        raise ValueError(
            "H7 plan input must be a directory, not a symlink"
        )
    snapshots = H7ReviewInputSnapshots(
        plan_root=root,
        plan_root_resolved=root.resolve(),
        plan=_snapshot_regular_file(
            root / PLAN_FILENAME,
            label="H7 retained plan",
        ),
        bindings=_snapshot_regular_file(
            root / BINDINGS_FILENAME,
            label="H7 retained request bindings",
        ),
        requests=_snapshot_regular_file(
            root / REQUESTS_FILENAME,
            label="H7 retained provider requests",
        ),
        responses=_snapshot_regular_file(
            Path(responses_path),
            label="H7 provider responses",
        ),
        provider_audit=_snapshot_regular_file(
            Path(provider_audit_path),
            label="H7 provider audit",
        ),
    )
    snapshots.verify_unchanged()
    return snapshots


def _validate_h7_plan_snapshots(
    plan_snapshot: _FileSnapshot,
    bindings_snapshot: _FileSnapshot,
    requests_snapshot: _FileSnapshot,
    expected: H7VolunteeredCollectionPlan,
) -> Mapping[str, str]:
    if _read_json_object(plan_snapshot.material) != expected.to_dict():
        raise ValueError(
            "retained H7 plan does not match the verified source run"
        )
    bindings = _read_jsonl_objects(bindings_snapshot.material)
    if bindings != tuple(
        binding.to_dict() for binding in expected.request_bindings
    ):
        raise ValueError(
            "retained H7 request bindings do not match regenerated bindings"
        )
    if read_requests(requests_snapshot.material) != expected.requests:
        raise ValueError(
            "retained H7 provider requests do not match regenerated requests"
        )
    return {
        "plan_file_sha256": plan_snapshot.sha256,
        "bindings_file_sha256": bindings_snapshot.sha256,
        "requests_file_sha256": requests_snapshot.sha256,
    }


def validate_h7_plan_directory(
    plan_dir: str | Path,
    expected: H7VolunteeredCollectionPlan,
) -> Mapping[str, str]:
    """Require every retained plan representation to match regeneration."""

    root = Path(plan_dir)
    root_absolute = root.absolute()
    if root_absolute.is_symlink() or not root_absolute.is_dir():
        raise ValueError(
            "H7 plan input must be a directory, not a symlink"
        )
    root_resolved = root_absolute.resolve()
    plan_snapshot = _snapshot_regular_file(
        root / PLAN_FILENAME,
        label="H7 retained plan",
    )
    bindings_snapshot = _snapshot_regular_file(
        root / BINDINGS_FILENAME,
        label="H7 retained request bindings",
    )
    requests_snapshot = _snapshot_regular_file(
        root / REQUESTS_FILENAME,
        label="H7 retained provider requests",
    )
    result = _validate_h7_plan_snapshots(
        plan_snapshot,
        bindings_snapshot,
        requests_snapshot,
        expected,
    )
    if (
        root_absolute.is_symlink()
        or not root_absolute.is_dir()
        or root_absolute.resolve() != root_resolved
    ):
        raise ValueError(
            "H7 plan directory changed while it was validated"
        )
    for snapshot in (
        plan_snapshot,
        bindings_snapshot,
        requests_snapshot,
    ):
        snapshot.verify_unchanged()
    return result


@dataclass(frozen=True, slots=True)
class H7VolunteeredEvidenceRecord:
    """One accepted provider result and its derived directional update."""

    case_id: str
    user_id: str
    domain_id: str
    target_attribute: int
    target_direction: int
    updater_id: str
    provider: str
    model_id: str
    request_id: str
    prompt_sha256: str
    request_body_sha256: str
    raw_response_sha256: str
    audit_record_sha256: str
    prior_probabilities: ProbabilityRows
    posterior_probabilities: ProbabilityRows
    directional_log_odds_update: float

    def to_update(self) -> VolunteeredPreferenceUpdate:
        return VolunteeredPreferenceUpdate(
            case_id=self.case_id,
            user_id=self.user_id,
            updater_id=self.updater_id,
            directional_log_odds_update=(
                self.directional_log_odds_update
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": H7_SCHEMA_VERSION,
            "case_id": self.case_id,
            "user_id": self.user_id,
            "domain_id": self.domain_id,
            "target_attribute": self.target_attribute,
            "target_direction": self.target_direction,
            "updater_id": self.updater_id,
            "provider": self.provider,
            "model_id": self.model_id,
            "request_id": self.request_id,
            "prompt_sha256": self.prompt_sha256,
            "request_body_sha256": self.request_body_sha256,
            "raw_response_sha256": self.raw_response_sha256,
            "audit_record_sha256": self.audit_record_sha256,
            "prior_probabilities": _rows_to_beliefs(
                self.prior_probabilities
            ),
            "posterior_probabilities": _rows_to_beliefs(
                self.posterior_probabilities
            ),
            "directional_log_odds_update": (
                self.directional_log_odds_update
            ),
            "claim_status": CLAIM_STATUS,
        }


def _validate_provider_audit(
    audit: Mapping[str, Any],
    *,
    expected_request: LLMRequest,
    response: LLMResponse,
) -> tuple[str, str, str]:
    provider = audit.get("provider")
    if provider not in {"openai", "openrouter"}:
        raise ValueError("H7 audit provider must be openai or openrouter")
    if audit.get("acceptance_status") != "accepted":
        raise ValueError(
            f"H7 provider audit is not accepted for {response.request_id}"
        )
    if (
        audit.get("request_id") != expected_request.request_id
        or audit.get("prompt_sha256") != expected_request.prompt_sha256
    ):
        raise ValueError("H7 provider audit request/prompt binding mismatch")
    if (
        response.request_id != expected_request.request_id
        or response.prompt_sha256 != expected_request.prompt_sha256
    ):
        raise ValueError("H7 replay response request/prompt binding mismatch")
    model_returned = audit.get("model_returned")
    if model_returned != response.model_id:
        raise ValueError("H7 provider audit model does not match replay response")
    raw_digest = response.raw_response_sha256
    _validate_digest(raw_digest, "response raw_response_sha256")
    if audit.get("raw_response_sha256") != raw_digest:
        raise ValueError(
            "H7 provider audit raw-response digest does not match replay response"
        )
    body_digest = _validate_digest(
        audit.get("request_body_sha256"),
        "request_body_sha256",
    )
    if not isinstance(audit.get("raw_response"), Mapping):
        raise ValueError("H7 provider audit must retain a redacted raw response")
    if not isinstance(audit.get("provider_response_id"), str) or not audit[
        "provider_response_id"
    ]:
        raise ValueError("H7 provider audit lacks a provider response ID")
    attempts = audit.get("transport_attempts", audit.get("attempts"))
    if (
        isinstance(attempts, bool)
        or not isinstance(attempts, int)
        or attempts < 1
    ):
        raise ValueError("H7 provider audit lacks a positive attempt count")
    embedded = audit.get("replay_response")
    if not isinstance(embedded, Mapping):
        raise ValueError("H7 provider audit lacks its replay response")
    if LLMResponse.parse(embedded) != response:
        raise ValueError(
            "H7 provider audit embedded response differs from response JSONL"
        )
    if provider == "openrouter":
        if (
            audit.get("gateway") != "openrouter"
            or audit.get("first_party_origin_claimed") is not False
            or not isinstance(audit.get("upstream_provider"), str)
            or not audit["upstream_provider"]
            or not isinstance(audit.get("upstream_model"), str)
            or not audit["upstream_model"]
        ):
            raise ValueError("H7 OpenRouter audit lacks strict route identity")
    return provider, str(model_returned), body_digest


def volunteered_updates_from_provider_evidence(
    plan: H7VolunteeredCollectionPlan,
    responses: Iterable[LLMResponse],
    provider_audits: Sequence[Mapping[str, Any]],
) -> tuple[
    tuple[VolunteeredPreferenceUpdate, ...],
    tuple[H7VolunteeredEvidenceRecord, ...],
]:
    """Convert exact provider evidence; reject any coverage or pairing defect."""

    response_rows = tuple(responses)
    if len({row.request_id for row in response_rows}) != len(response_rows):
        raise ValueError("H7 responses contain duplicate request IDs")
    response_by_id = {row.request_id: row for row in response_rows}
    expected_by_id = {
        binding.llm_request.request_id: binding
        for binding in plan.request_bindings
    }
    missing = sorted(set(expected_by_id) - set(response_by_id))
    unexpected = sorted(set(response_by_id) - set(expected_by_id))
    if missing or unexpected:
        raise ValueError(
            "H7 response coverage mismatch; "
            f"missing={missing}, unexpected={unexpected}"
        )

    audit_by_id: dict[str, Mapping[str, Any]] = {}
    for audit in provider_audits:
        request_id = audit.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("H7 provider audit lacks a request_id")
        if request_id in audit_by_id:
            raise ValueError(f"duplicate H7 provider audit: {request_id}")
        audit_by_id[request_id] = audit
    missing_audits = sorted(set(expected_by_id) - set(audit_by_id))
    unexpected_audits = sorted(set(audit_by_id) - set(expected_by_id))
    if missing_audits or unexpected_audits:
        raise ValueError(
            "H7 provider-audit coverage mismatch; "
            f"missing={missing_audits}, unexpected={unexpected_audits}"
        )

    cases = {case.case_id: case for case in plan.cases}
    evidence: list[H7VolunteeredEvidenceRecord] = []
    for binding in plan.request_bindings:
        request = binding.llm_request
        response = response_by_id[request.request_id]
        audit = audit_by_id[request.request_id]
        provider, model_id, body_digest = _validate_provider_audit(
            audit,
            expected_request=request,
            response=response,
        )
        case = cases[binding.case_id]
        posterior = _rows_from_response(response)
        evidence.append(
            H7VolunteeredEvidenceRecord(
                case_id=case.case_id,
                user_id=case.user_id,
                domain_id=case.domain_id,
                target_attribute=case.target_attribute,
                target_direction=case.target_direction,
                updater_id=binding.updater_id,
                provider=provider,
                model_id=model_id,
                request_id=request.request_id,
                prompt_sha256=request.prompt_sha256,
                request_body_sha256=body_digest,
                raw_response_sha256=str(response.raw_response_sha256),
                audit_record_sha256=_digest(audit),
                prior_probabilities=case.prior_probabilities,
                posterior_probabilities=posterior,
                directional_log_odds_update=(
                    _directional_log_odds_update(
                        case.prior_probabilities,
                        posterior,
                        attribute=case.target_attribute,
                        direction=case.target_direction,
                    )
                ),
            )
        )

    pair_identity: dict[str, set[tuple[str, str]]] = {}
    for record in evidence:
        pair_identity.setdefault(record.case_id, set()).add(
            (record.provider, record.model_id)
        )
    mismatched_cases = sorted(
        case_id
        for case_id, identities in pair_identity.items()
        if len(identities) != 1
    )
    if mismatched_cases:
        raise ValueError(
            "H7 paired updater conditions must use the same provider/model: "
            + ", ".join(mismatched_cases)
        )
    identities = {
        (record.provider, record.model_id) for record in evidence
    }
    if len(identities) != 1:
        raise ValueError(
            "one H7 review must hold provider and model fixed across all cases"
        )
    updates = tuple(record.to_update() for record in evidence)
    return updates, tuple(evidence)


def _source_h7_payload(source: VerifiedH7Source) -> Mapping[str, Any]:
    artifact = source.hypothesis_estimands
    constants = artifact.get("frozen_decision_constants")
    if not isinstance(constants, Mapping):
        raise ValueError("source H7 artifact lacks frozen decision constants")
    if (
        constants.get("h7_valid_learning_retention_fraction")
        != H7_VALID_LEARNING_RETENTION_FRACTION
        or constants.get("minimum_cluster_count") != MINIMUM_CLUSTER_COUNT
    ):
        raise ValueError("source H7 frozen constants differ from implementation")
    hypotheses = artifact.get("hypotheses")
    if not isinstance(hypotheses, Mapping):
        raise ValueError("source hypothesis artifact lacks hypotheses")
    h7 = hypotheses.get("H7")
    if not isinstance(h7, Mapping):
        raise ValueError("source hypothesis artifact lacks H7")
    if (
        h7.get("claim_status") != CLAIM_STATUS
        or h7.get("baseline_updater_id") != BASELINE_UPDATER_ID
        or h7.get("mitigation_updater_id") != MITIGATION_UPDATER_ID
        or h7.get("response_mode") != "controlled_anchor"
    ):
        raise ValueError("source H7 identity or claim status is invalid")
    volunteered = h7.get("volunteered_valid_learning")
    if (
        not isinstance(volunteered, Mapping)
        or volunteered.get("pair_count") != 0
        or volunteered.get("criterion_met") is not None
    ):
        raise ValueError(
            "source H7 volunteered component must be explicitly unfilled"
        )
    return h7


def _recomputed_h7(
    source_h7: Mapping[str, Any],
    volunteered: Mapping[str, Any],
) -> dict[str, Any]:
    result = deepcopy(dict(source_h7))
    superiority = result.get("superiority_estimands")
    balanced = result.get("balanced_valid_learning")
    required = result.get("required_superiority_mechanisms")
    qualifying = result.get("qualifying_superiority_mechanisms")
    if (
        not isinstance(superiority, Sequence)
        or isinstance(superiority, (str, bytes))
        or not isinstance(balanced, Mapping)
        or isinstance(required, bool)
        or not isinstance(required, int)
        or not isinstance(qualifying, Sequence)
        or isinstance(qualifying, (str, bytes))
    ):
        raise ValueError("source H7 component fields are malformed")
    evaluable_count = sum(
        isinstance(item, Mapping) and item.get("criterion_met") is not None
        for item in superiority
    )
    complete = (
        evaluable_count >= required
        and balanced.get("criterion_met") is not None
        and volunteered.get("criterion_met") is not None
    )
    criterion_met = (
        len(qualifying) >= required
        and balanced.get("criterion_met") is True
        and volunteered.get("criterion_met") is True
        if complete
        else None
    )
    result["volunteered_valid_learning"] = dict(volunteered)
    result["complete"] = complete
    result["criterion_met"] = criterion_met
    result["computed_status"] = (
        "incomplete"
        if criterion_met is None
        else "criterion_met"
        if criterion_met
        else "criterion_not_met"
    )
    result["claim_status"] = CLAIM_STATUS
    return result


def create_h7_volunteered_review(
    source: VerifiedH7Source,
    plan_dir: str | Path,
    responses_path: str | Path,
    provider_audit_path: str | Path,
    *,
    input_snapshots: H7ReviewInputSnapshots | None = None,
) -> dict[str, Any]:
    """Create a deterministic derived H7 review from immutable inputs."""

    if input_snapshots is None:
        snapshots = snapshot_h7_review_inputs(
            plan_dir,
            responses_path,
            provider_audit_path,
        )
    else:
        snapshots = input_snapshots
        expected_paths = (
            Path(plan_dir).absolute(),
            Path(responses_path).absolute(),
            Path(provider_audit_path).absolute(),
        )
        observed_paths = (
            snapshots.plan_root,
            snapshots.responses.supplied,
            snapshots.provider_audit.supplied,
        )
        if observed_paths != expected_paths:
            raise ValueError(
                "H7 input snapshots do not bind the supplied input paths"
            )
        snapshots.verify_unchanged()
    source.verify_unchanged()
    plan_files = _validate_h7_plan_snapshots(
        snapshots.plan,
        snapshots.bindings,
        snapshots.requests,
        source.plan,
    )
    updates, evidence = volunteered_updates_from_provider_evidence(
        source.plan,
        read_responses(snapshots.responses.material),
        _read_jsonl_objects(snapshots.provider_audit.material),
    )
    refreshed_source = load_verified_h7_source(source.run_dir)
    if (
        refreshed_source.source_run != source.source_run
        or refreshed_source.plan != source.plan
        or refreshed_source.hypothesis_estimands
        != source.hypothesis_estimands
    ):
        raise ValueError("H7 source run changed during review")
    volunteered = analyze_h7_volunteered_updates(
        updates,
        replicates=source.bootstrap_replicates,
        confidence_level=source.confidence_level,
        minimum_cluster_count=MINIMUM_CLUSTER_COUNT,
        seed=source.seed,
    )
    source_h7 = _source_h7_payload(source)
    provider_model = sorted(
        {(record.provider, record.model_id) for record in evidence}
    )
    if len(provider_model) != 1:
        raise ValueError("H7 review requires one fixed provider/model identity")
    provider, model_id = provider_model[0]
    payload: dict[str, Any] = {
        "schema_version": H7_SCHEMA_VERSION,
        "artifact_kind": "h7_volunteered_control_review",
        "review_version": H7_VOLUNTEERED_REVIEW_VERSION,
        "source_run": dict(source.source_run),
        "collection_plan": {
            "plan_sha256": source.plan.plan_sha256,
            **dict(plan_files),
        },
        "provider_evidence": {
            "provider": provider,
            "model_id": model_id,
            "response_file_sha256": snapshots.responses.sha256,
            "provider_audit_file_sha256": snapshots.provider_audit.sha256,
            "response_count": len(evidence),
            "accepted_audit_count": len(evidence),
            "exact_coverage": True,
            "same_provider_model_within_pair": True,
        },
        "analysis_settings": {
            "bootstrap_replicates": source.bootstrap_replicates,
            "confidence_level": source.confidence_level,
            "seed": source.seed,
            "independent_unit": "complete latent user",
            "retention_fraction": H7_VALID_LEARNING_RETENTION_FRACTION,
            "minimum_cluster_count": MINIMUM_CLUSTER_COUNT,
        },
        "volunteered_preference_updates": [
            update.to_dict() for update in updates
        ],
        "provider_bound_evidence": [
            record.to_dict() for record in evidence
        ],
        "source_h7_sha256": _digest(source_h7),
        "recomputed_h7": _recomputed_h7(
            source_h7,
            volunteered.to_dict(),
        ),
        "recomputation_scope": {
            "source_components_reused": [
                "policy_conditioned_acue_superiority",
                "balanced_valid_learning",
            ],
            "components_recomputed": [
                "volunteered_valid_learning",
                "experiment_a_h7_complete",
                "experiment_a_h7_criterion_met",
            ],
            "source_run_modified": False,
            "missing_values_imputed": False,
        },
        "claim_status": CLAIM_STATUS,
        "interpretation": (
            "This checksum-bound review supplies H7's Experiment A "
            "volunteered positive-control component only. A computed result "
            "is not a paper claim, and H7's separate closed-loop component "
            "remains required."
        ),
    }
    payload["review_sha256"] = _digest(payload)
    source.verify_unchanged()
    snapshots.verify_unchanged()
    return payload


def verify_h7_volunteered_review(
    source_run_dir: str | Path,
    plan_dir: str | Path,
    responses_path: str | Path,
    provider_audit_path: str | Path,
    review_path: str | Path,
) -> tuple[bool, tuple[str, ...]]:
    """Recompute the full derived review and compare it byte-independently."""

    try:
        source = load_verified_h7_source(source_run_dir)
        expected = create_h7_volunteered_review(
            source,
            plan_dir,
            responses_path,
            provider_audit_path,
        )
        retained = _read_json_object(Path(review_path))
        errors: list[str] = []
        if retained != expected:
            errors.append(
                "retained H7 review differs from exact recomputation"
            )
        review_digest = retained.get("review_sha256")
        if isinstance(review_digest, str):
            without_digest = dict(retained)
            without_digest.pop("review_sha256", None)
            if review_digest != _digest(without_digest):
                errors.append("retained H7 review_sha256 is invalid")
        else:
            errors.append("retained H7 review lacks review_sha256")
        return not errors, tuple(errors)
    except (OSError, TypeError, ValueError) as exc:
        return False, (str(exc),)
