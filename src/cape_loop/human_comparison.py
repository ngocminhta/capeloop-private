"""Matched human-versus-model provenance-sensitivity analysis for H8.

Human ratings and model profile updates are measured on different raw scales.
The comparison therefore uses a dimensionless within-source discount:

``(balanced evidence - policy-conditioned evidence) / balanced evidence``.

Human ratings are first shifted to the declared zero-support endpoint of the
seven-point scale. Model inputs must explicitly attest that zero means no
evidential update. The ratio is invariant to positive rescaling, but not to an
arbitrary additive shift; the input contract makes that limitation explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence
import json
import math

from .decoder_study import (
    Codebook,
    HumanCollectionRecord,
    HumanImportAudit,
    validate_human_collection,
)
from .human_study import CONDITIONS
from .rng import weighted_index
from .statistics import percentile


POLICY_CONDITIONS = ("restricted", "default", "suggested")
EVIDENCE_METRIC = "positive_part_anchor_directional_log_odds_update"
SOURCE_ROLES = (
    "fitted_action_aware",
    "ordinary_llm",
    "provenance_aware_llm",
)
EXPERIMENT_A_UPDATER_ROLES = {
    "fitted_action_aware": "fitted_action_aware",
    "llm_response_only": "ordinary_llm",
    "llm_full_context": "ordinary_llm",
    "llm_provenance_aware": "provenance_aware_llm",
}


def _require_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _finite_nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return parsed


def _require_sha256(value: object, name: str) -> str:
    parsed = _require_text(value, name)
    if len(parsed) != 64 or any(
        character not in "0123456789abcdef" for character in parsed
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return parsed


@dataclass(frozen=True, slots=True)
class ModelEvidenceStrength:
    """One model/reference evidence-strength observation for Experiment D.

    ``cluster_id`` is the independent resampling unit (normally one latent
    user/case), while ``scenario_id`` binds balanced and policy-conditioned
    observations inside that cluster.
    """

    source_run_id: str
    source_artifact_sha256: str
    source_record_id: str
    source_id: str
    source_role: str
    cluster_id: str
    scenario_id: str
    condition: str
    evidence_strength: float
    evidence_metric: str
    zero_means_no_evidence: bool
    evaluation_split: str = "test"

    def __post_init__(self) -> None:
        for name in (
            "source_run_id",
            "source_record_id",
            "source_id",
            "cluster_id",
            "scenario_id",
            "evidence_metric",
        ):
            _require_text(getattr(self, name), name)
        _require_sha256(
            self.source_artifact_sha256,
            "source_artifact_sha256",
        )
        if self.source_role not in SOURCE_ROLES:
            raise ValueError(f"source_role must be one of {SOURCE_ROLES}")
        if self.condition not in CONDITIONS:
            raise ValueError(f"condition must be one of {CONDITIONS}")
        if self.evidence_metric != EVIDENCE_METRIC:
            raise ValueError(
                f"evidence_metric must be {EVIDENCE_METRIC!r}"
            )
        if self.evaluation_split != "test":
            raise ValueError("H8 model evidence must come from the test split")
        if self.zero_means_no_evidence is not True:
            raise ValueError(
                "H8 comparison requires zero_means_no_evidence = true"
            )
        object.__setattr__(
            self,
            "evidence_strength",
            _finite_nonnegative(
                self.evidence_strength,
                "evidence_strength",
            ),
        )

    @classmethod
    def parse(cls, raw: Mapping[str, Any]) -> "ModelEvidenceStrength":
        allowed = {
            "schema_version",
            "source_run_id",
            "source_artifact_sha256",
            "source_record_id",
            "source_id",
            "source_role",
            "cluster_id",
            "scenario_id",
            "condition",
            "evidence_strength",
            "evidence_metric",
            "zero_means_no_evidence",
            "evaluation_split",
        }
        if set(raw) != allowed:
            raise ValueError(
                "model evidence row has missing or unknown fields: "
                + json.dumps(
                    {
                        "missing": sorted(allowed - set(raw)),
                        "unknown": sorted(set(raw) - allowed),
                    },
                    sort_keys=True,
                )
            )
        if raw["schema_version"] != 1:
            raise ValueError("model evidence schema_version must be 1")
        return cls(
            source_run_id=raw["source_run_id"],
            source_artifact_sha256=raw["source_artifact_sha256"],
            source_record_id=raw["source_record_id"],
            source_id=raw["source_id"],
            source_role=raw["source_role"],
            cluster_id=raw["cluster_id"],
            scenario_id=raw["scenario_id"],
            condition=raw["condition"],
            evidence_strength=raw["evidence_strength"],
            evidence_metric=raw["evidence_metric"],
            zero_means_no_evidence=raw["zero_means_no_evidence"],
            evaluation_split=raw["evaluation_split"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "source_run_id": self.source_run_id,
            "source_artifact_sha256": self.source_artifact_sha256,
            "source_record_id": self.source_record_id,
            "source_id": self.source_id,
            "source_role": self.source_role,
            "cluster_id": self.cluster_id,
            "scenario_id": self.scenario_id,
            "condition": self.condition,
            "evidence_strength": self.evidence_strength,
            "evidence_metric": self.evidence_metric,
            "zero_means_no_evidence": self.zero_means_no_evidence,
            "evaluation_split": self.evaluation_split,
        }


def read_model_evidence_strengths(
    path: str | Path | bytes,
) -> tuple[ModelEvidenceStrength, ...]:
    if isinstance(path, bytes):
        source_label = "<model-evidence-bytes>"
        try:
            lines = path.decode("utf-8").splitlines()
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"{source_label}: input must be valid UTF-8"
            ) from exc
    else:
        source = Path(path)
        source_label = str(source)
        lines = source.read_text(encoding="utf-8").splitlines()
    rows: list[ModelEvidenceStrength] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            decoded = json.loads(line)
            if not isinstance(decoded, Mapping):
                raise ValueError("record must be a JSON object")
            rows.append(ModelEvidenceStrength.parse(decoded))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{source_label}:{line_number}: {exc}"
            ) from exc
    keys = [
        (
            row.source_id,
            row.cluster_id,
            row.scenario_id,
            row.condition,
        )
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate model evidence source/cluster/scenario/condition")
    return tuple(rows)


def _metric_text(raw: Mapping[str, Any], name: str) -> str:
    value = raw.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Experiment A metric {name} must be nonempty text")
    return value


def _metric_int(
    raw: Mapping[str, Any],
    name: str,
    *,
    allowed: set[int],
) -> int:
    value = raw.get(name)
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value not in allowed
    ):
        raise ValueError(
            f"Experiment A metric {name} must be one of {sorted(allowed)}"
        )
    return value


def _metric_float(raw: Mapping[str, Any], name: str) -> float:
    value = raw.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Experiment A metric {name} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"Experiment A metric {name} must be finite")
    return parsed


def convert_experiment_a_metrics_to_model_evidence(
    metric_rows: Iterable[Mapping[str, Any]],
    *,
    source_run_id: str,
    source_artifact_sha256: str,
    sources: Mapping[str, str],
    test_user_domain_pairs: set[tuple[str, str]] | None = None,
) -> tuple[ModelEvidenceStrength, ...]:
    """Convert retained test-only Experiment A rows without fabricating cells.

    ``sources`` maps a public evidence-source ID to an Experiment A updater ID.
    Only fitted-aware and actual ``llm_*`` updater IDs are admissible. The
    converter retains controlled-anchor balanced/policy rows and never creates
    the volunteered-statement condition absent from Experiment A.
    """

    _require_text(source_run_id, "source_run_id")
    _require_sha256(source_artifact_sha256, "source_artifact_sha256")
    if not sources:
        raise ValueError("at least one Experiment A evidence source is required")
    updater_to_source: dict[str, str] = {}
    for source_id, updater_id in sources.items():
        _require_text(source_id, "source_id")
        _require_text(updater_id, "updater_id")
        if updater_id not in EXPERIMENT_A_UPDATER_ROLES:
            raise ValueError(
                "Experiment A H8 conversion accepts only "
                + ", ".join(sorted(EXPERIMENT_A_UPDATER_ROLES))
            )
        if updater_id in updater_to_source:
            raise ValueError(
                f"Experiment A updater {updater_id!r} was selected twice"
            )
        updater_to_source[updater_id] = source_id

    converted: list[ModelEvidenceStrength] = []
    seen_updaters: set[str] = set()
    for raw in metric_rows:
        if not isinstance(raw, Mapping):
            raise ValueError("Experiment A metric rows must be JSON objects")
        updater_id = raw.get("updater_id")
        if updater_id not in updater_to_source:
            continue
        if _metric_text(raw, "response_mode") != "controlled_anchor":
            continue
        mechanism = _metric_text(raw, "mechanism")
        if mechanism not in ("balanced", *POLICY_CONDITIONS):
            raise ValueError(
                f"unsupported Experiment A mechanism for H8: {mechanism}"
            )
        user_id = _metric_text(raw, "user_id")
        domain = _metric_text(raw, "domain")
        if (
            test_user_domain_pairs is not None
            and (user_id, domain) not in test_user_domain_pairs
        ):
            raise ValueError(
                "Experiment A H8 evidence row is not bound to a retained "
                f"test user/domain: {user_id}/{domain}"
            )
        trial_id = _metric_text(raw, "trial_id")
        prior_stratum = _metric_text(raw, "prior_stratum")
        target_attribute = _metric_int(
            raw,
            "target_attribute",
            allowed={0, 1, 2},
        )
        anchor_direction = _metric_int(
            raw,
            "anchor_direction",
            allowed={-1, 1},
        )
        prior_strength = _metric_float(raw, "prior_strength")
        directional_update = (
            anchor_direction * _metric_float(raw, "log_odds_update")
        )
        scenario_payload = {
            "domain": domain,
            "target_attribute": target_attribute,
            "anchor_direction": anchor_direction,
            "prior_stratum": prior_stratum,
            "prior_strength": prior_strength,
        }
        scenario_id = "experiment-a:" + sha256(
            json.dumps(
                scenario_payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()
        source_id = updater_to_source[updater_id]
        converted.append(
            ModelEvidenceStrength(
                source_run_id=source_run_id,
                source_artifact_sha256=source_artifact_sha256,
                source_record_id=f"{trial_id}:{updater_id}",
                source_id=source_id,
                source_role=EXPERIMENT_A_UPDATER_ROLES[updater_id],
                cluster_id=user_id,
                scenario_id=scenario_id,
                condition=mechanism,
                evidence_strength=max(0.0, directional_update),
                evidence_metric=EVIDENCE_METRIC,
                zero_means_no_evidence=True,
                evaluation_split="test",
            )
        )
        seen_updaters.add(updater_id)
    missing = sorted(set(updater_to_source) - seen_updaters)
    if missing:
        raise ValueError(
            "selected Experiment A updater has no controlled-anchor rows: "
            + ", ".join(missing)
        )
    keys = [
        (
            row.source_id,
            row.cluster_id,
            row.scenario_id,
            row.condition,
        )
        for row in converted
    ]
    if len(keys) != len(set(keys)):
        raise ValueError(
            "converted Experiment A evidence has duplicate "
            "source/cluster/scenario/condition rows"
        )
    return tuple(
        sorted(
            converted,
            key=lambda row: (
                row.source_id,
                row.cluster_id,
                row.scenario_id,
                row.condition,
            ),
        )
    )


@dataclass(frozen=True, slots=True)
class H8SourceContrast:
    source_id: str
    source_role: str
    mechanism: str
    human_cluster_count: int
    model_cluster_count: int
    human_paired_scenario_count: int
    model_paired_scenario_count: int
    human_mean_discount: float
    model_mean_discount: float
    human_minus_model_discount: float
    bootstrap_lower: float
    bootstrap_upper: float
    criterion_evaluable: bool
    criterion_met: bool | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_role": self.source_role,
            "mechanism": self.mechanism,
            "human_cluster_count": self.human_cluster_count,
            "model_cluster_count": self.model_cluster_count,
            "human_paired_scenario_count": self.human_paired_scenario_count,
            "model_paired_scenario_count": self.model_paired_scenario_count,
            "human_mean_discount": self.human_mean_discount,
            "model_mean_discount": self.model_mean_discount,
            "human_minus_model_discount": self.human_minus_model_discount,
            "bootstrap_lower": self.bootstrap_lower,
            "bootstrap_upper": self.bootstrap_upper,
            "criterion_evaluable": self.criterion_evaluable,
            "criterion_met": self.criterion_met,
        }


@dataclass(frozen=True, slots=True)
class H8Analysis:
    primary_llm_source_id: str
    minimum_clusters: int
    bootstrap_replicates: int
    human_import_audit: HumanImportAudit
    contrasts: tuple[H8SourceContrast, ...]
    complete_fitted_aware_source_ids: tuple[str, ...]
    qualifying_primary_llm_mechanisms: tuple[str, ...]
    missing_primary_llm_mechanisms: tuple[str, ...]
    computed_status: str
    criterion_met: bool | None
    claim_status: str = "not_claimed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "hypothesis": "H8",
            "estimand": (
                "human provenance-discount fraction minus model "
                "provenance-discount fraction"
            ),
            "discount_definition": (
                "(balanced evidence - policy-conditioned evidence) / "
                "balanced evidence"
            ),
            "human_scale": (
                "Likert support shifted to zero at the declared scale minimum"
            ),
            "model_scale_requirement": (
                "nonnegative evidence strength with an attested no-update zero; "
                "the contrast is invariant only to positive multiplicative "
                "rescaling"
            ),
            "inference": (
                "independent percentile bootstrap over pair-complete human "
                "participant and model test-user clusters after within-cluster "
                "balanced/policy scenario pairing; clusters are equally "
                "weighted after averaging their paired scenarios"
            ),
            "primary_llm_source_id": self.primary_llm_source_id,
            "primary_source_selection_boundary": (
                "The CLI requires a caller-declared ordinary-LLM source ID; "
                "the software cannot verify that selection was temporally "
                "preregistered."
            ),
            "minimum_clusters": self.minimum_clusters,
            "bootstrap_replicates": self.bootstrap_replicates,
            "human_import_audit": self.human_import_audit.to_dict(),
            "contrasts": [row.to_dict() for row in self.contrasts],
            "complete_fitted_aware_source_ids": list(
                self.complete_fitted_aware_source_ids
            ),
            "qualifying_primary_llm_mechanisms": list(
                self.qualifying_primary_llm_mechanisms
            ),
            "missing_primary_llm_mechanisms": list(
                self.missing_primary_llm_mechanisms
            ),
            "criterion": (
                "For the preregistered primary ordinary-LLM source, the 95% "
                "lower bound on human-minus-model discount is above zero on "
                "at least two of restricted/default/suggested, with at least "
                "the minimum pair-complete clusters in both samples. At least "
                "one fitted-aware source must independently cover all three "
                "mechanisms at the same minimum."
            ),
            "computed_status": self.computed_status,
            "criterion_met": self.criterion_met,
            "claim_status": self.claim_status,
            "interpretation_boundary": (
                "Human ratings validate pragmatic evidential ordering; they "
                "are not treated as access to metaphysical true preferences."
            ),
        }


def _cluster_discounts(
    rows: Sequence[tuple[str, str, str, float]],
    *,
    mechanism: str,
    zero_tolerance: float,
) -> tuple[tuple[float, ...], int]:
    """Reduce paired scenario discounts to one equally weighted value/cluster."""

    grouped: dict[tuple[str, str, str], list[float]] = {}
    for cluster_id, scenario_id, condition, value in rows:
        grouped.setdefault((cluster_id, scenario_id, condition), []).append(value)
    scenario_discounts: dict[str, list[float]] = {}
    paired_scenarios = 0
    cluster_scenarios = sorted(
        {(cluster, scenario) for cluster, scenario, _ in grouped}
    )
    for cluster_id, scenario_id in cluster_scenarios:
        balanced_values = grouped.get((cluster_id, scenario_id, "balanced"))
        mechanism_values = grouped.get((cluster_id, scenario_id, mechanism))
        if not balanced_values or not mechanism_values:
            continue
        balanced = mean(balanced_values)
        if balanced <= zero_tolerance:
            continue
        conditioned = mean(mechanism_values)
        scenario_discounts.setdefault(cluster_id, []).append(
            (balanced - conditioned) / balanced
        )
        paired_scenarios += 1
    return (
        tuple(
            mean(scenario_discounts[cluster_id])
            for cluster_id in sorted(scenario_discounts)
        ),
        paired_scenarios,
    )


def _independent_bootstrap_difference(
    first: Sequence[float],
    second: Sequence[float],
    *,
    replicates: int,
    seed: int,
    namespace: str,
) -> tuple[float, float, float]:
    if not first or not second:
        raise ValueError("independent bootstrap samples cannot be empty")
    observed = mean(first) - mean(second)
    first_weights = [1.0] * len(first)
    second_weights = [1.0] * len(second)
    draws: list[float] = []
    for replicate in range(replicates):
        first_indexes = [
            weighted_index(
                first_weights,
                seed,
                namespace,
                "human",
                replicate,
                draw,
            )
            for draw in range(len(first))
        ]
        second_indexes = [
            weighted_index(
                second_weights,
                seed,
                namespace,
                "model",
                replicate,
                draw,
            )
            for draw in range(len(second))
        ]
        draws.append(
            mean(first[index] for index in first_indexes)
            - mean(second[index] for index in second_indexes)
        )
    return observed, percentile(draws, 0.025), percentile(draws, 0.975)


def analyze_h8_human_model_comparison(
    human_records: Iterable[HumanCollectionRecord],
    model_records: Iterable[ModelEvidenceStrength],
    *,
    assignment_codebooks: Codebook,
    expected_assignment_protocol_id: str,
    expected_consent_version: str,
    expected_blinding_version: str,
    primary_llm_source_id: str,
    bootstrap_replicates: int = 2000,
    minimum_clusters: int = 8,
    seed: int = 1729,
    zero_tolerance: float = 1e-12,
) -> H8Analysis:
    """Compare complete-cluster human and model provenance discounts."""

    _require_text(primary_llm_source_id, "primary_llm_source_id")
    if (
        isinstance(bootstrap_replicates, bool)
        or not isinstance(bootstrap_replicates, int)
        or bootstrap_replicates <= 0
    ):
        raise ValueError("bootstrap_replicates must be positive")
    if (
        isinstance(minimum_clusters, bool)
        or not isinstance(minimum_clusters, int)
        or minimum_clusters < 2
    ):
        raise ValueError("minimum_clusters must be an integer of at least two")
    if not math.isfinite(zero_tolerance) or zero_tolerance < 0.0:
        raise ValueError("zero_tolerance must be finite and nonnegative")

    human_material = tuple(human_records)
    human_import_audit = validate_human_collection(
        human_material,
        assignment_codebooks=assignment_codebooks,
        expected_assignment_protocol_id=expected_assignment_protocol_id,
        expected_consent_version=expected_consent_version,
        expected_blinding_version=expected_blinding_version,
    )
    human_rows: list[tuple[str, str, str, float]] = []
    for row in human_material:
        if not row.consented or not row.comprehension_passed:
            continue
        entry = assignment_codebooks[row.assignment_id][row.display_id]
        # StudyItem fixes the rating scale at 1..7. Shift its declared minimum
        # to zero before computing a within-source ratio.
        normalized_support = (float(row.rating) - 1.0) / 6.0
        human_rows.append(
            (
                row.participant_code,
                entry["scenario_id"],
                entry["condition"],
                normalized_support,
            )
        )
    if not human_rows:
        raise ValueError("no eligible human ratings remain for H8")

    model_material = tuple(model_records)
    model_keys = [
        (
            row.source_id,
            row.cluster_id,
            row.scenario_id,
            row.condition,
        )
        for row in model_material
    ]
    if len(model_keys) != len(set(model_keys)):
        raise ValueError("duplicate model evidence source/cluster/scenario/condition")
    source_metadata: dict[str, set[tuple[str, str, str, str]]] = {}
    for row in model_material:
        source_metadata.setdefault(row.source_id, set()).add(
            (
                row.source_role,
                row.evidence_metric,
                row.source_run_id,
                row.source_artifact_sha256,
            )
        )
    inconsistent = sorted(
        source_id
        for source_id, metadata in source_metadata.items()
        if len(metadata) != 1
    )
    if inconsistent:
        raise ValueError(
            "model source changes role, metric, run, or artifact binding: "
            + ", ".join(inconsistent)
        )
    primary_metadata = source_metadata.get(primary_llm_source_id)
    if (
        primary_metadata is not None
        and next(iter(primary_metadata))[0] != "ordinary_llm"
    ):
        raise ValueError("primary_llm_source_id must identify an ordinary_llm")

    contrasts: list[H8SourceContrast] = []
    missing_primary: list[str] = []
    qualifying_primary: list[str] = []
    for mechanism in POLICY_CONDITIONS:
        human_discounts, human_pairs = _cluster_discounts(
            human_rows,
            mechanism=mechanism,
            zero_tolerance=zero_tolerance,
        )
        primary_seen = False
        for source_id in sorted(source_metadata):
            source_role, _, _, _ = next(iter(source_metadata[source_id]))
            source_rows = [
                (
                    row.cluster_id,
                    row.scenario_id,
                    row.condition,
                    row.evidence_strength,
                )
                for row in model_material
                if row.source_id == source_id
            ]
            model_discounts, model_pairs = _cluster_discounts(
                source_rows,
                mechanism=mechanism,
                zero_tolerance=zero_tolerance,
            )
            if not human_discounts or not model_discounts:
                continue
            estimate, lower, upper = _independent_bootstrap_difference(
                human_discounts,
                model_discounts,
                replicates=bootstrap_replicates,
                seed=seed,
                namespace=f"h8:{source_id}:{mechanism}",
            )
            evaluable = (
                len(human_discounts) >= minimum_clusters
                and len(model_discounts) >= minimum_clusters
            )
            met = lower > 0.0 if evaluable else None
            contrasts.append(
                H8SourceContrast(
                    source_id=source_id,
                    source_role=source_role,
                    mechanism=mechanism,
                    human_cluster_count=len(human_discounts),
                    model_cluster_count=len(model_discounts),
                    human_paired_scenario_count=human_pairs,
                    model_paired_scenario_count=model_pairs,
                    human_mean_discount=mean(human_discounts),
                    model_mean_discount=mean(model_discounts),
                    human_minus_model_discount=estimate,
                    bootstrap_lower=lower,
                    bootstrap_upper=upper,
                    criterion_evaluable=evaluable,
                    criterion_met=met,
                )
            )
            if source_id == primary_llm_source_id:
                primary_seen = True
                if met is True:
                    qualifying_primary.append(mechanism)
        if not primary_seen:
            missing_primary.append(mechanism)

    complete_fitted_sources = tuple(
        source_id
        for source_id in sorted(source_metadata)
        if next(iter(source_metadata[source_id]))[0] == "fitted_action_aware"
        and all(
            any(
                row.source_id == source_id
                and row.mechanism == mechanism
                and row.criterion_evaluable
                for row in contrasts
            )
            for mechanism in POLICY_CONDITIONS
        )
    )
    complete = (
        primary_llm_source_id in source_metadata
        and not missing_primary
        and bool(complete_fitted_sources)
        and all(
            any(
                row.source_id == primary_llm_source_id
                and row.mechanism == mechanism
                and row.criterion_evaluable
                for row in contrasts
            )
            for mechanism in POLICY_CONDITIONS
        )
    )
    return H8Analysis(
        primary_llm_source_id=primary_llm_source_id,
        minimum_clusters=minimum_clusters,
        bootstrap_replicates=bootstrap_replicates,
        human_import_audit=human_import_audit,
        contrasts=tuple(contrasts),
        complete_fitted_aware_source_ids=complete_fitted_sources,
        qualifying_primary_llm_mechanisms=tuple(
            mechanism
            for mechanism in POLICY_CONDITIONS
            if mechanism in qualifying_primary
        ),
        missing_primary_llm_mechanisms=tuple(missing_primary),
        computed_status="computed" if complete else "incomplete",
        criterion_met=(
            len(set(qualifying_primary)) >= 2 if complete else None
        ),
    )
