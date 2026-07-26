"""Machine-readable scientific stage-gate diagnostics.

Gate reports expose computed checks but never promote a smoke/pilot run into a
paper claim. ``claim_status`` stays ``"not_claimed"`` until researchers attach a
frozen analysis protocol and explicitly review the retained evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping
import math


@dataclass(frozen=True, slots=True)
class GateCriterion:
    criterion_id: str
    description: str
    passed: bool | None
    observed: Any
    requirement: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "criterion_id": self.criterion_id,
            "description": self.description,
            "passed": self.passed,
            "observed": self.observed,
            "requirement": self.requirement,
        }


@dataclass(frozen=True, slots=True)
class GateReport:
    gate_id: str
    title: str
    criteria: tuple[GateCriterion, ...]
    evidence_scope: str = "diagnostic"
    claim_status: str = "not_claimed"

    @property
    def computed_status(self) -> str:
        decisions = tuple(criterion.passed for criterion in self.criteria)
        if not decisions or any(decision is None for decision in decisions):
            return "incomplete"
        return "meets_computational_checks" if all(decisions) else "does_not_meet_checks"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "gate_id": self.gate_id,
            "title": self.title,
            "evidence_scope": self.evidence_scope,
            "computed_status": self.computed_status,
            "claim_status": self.claim_status,
            "criteria": [criterion.to_dict() for criterion in self.criteria],
        }


def _finite_mean(values: Iterable[float]) -> float | None:
    material = tuple(float(value) for value in values if math.isfinite(float(value)))
    return None if not material else math.fsum(material) / len(material)


def gate_1_from_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    material_gap: float = 0.01,
    required_mechanisms: int = 2,
    full_context_updater_id: str = "full_context_blind",
    held_out_paraphrase_verified: bool | None = None,
) -> GateReport:
    """Evaluate Gate 1 from naturally sampled Experiment A Brier rows."""

    natural = [
        row
        for row in rows
        if row.get("response_mode") == "naturally_sampled"
        and "brier" in row
    ]

    def mean_for(updater: str, *, mechanism: str | None = None) -> float | None:
        return _finite_mean(
            row["brier"]
            for row in natural
            if row.get("updater_id") == updater
            and (mechanism is None or row.get("mechanism") == mechanism)
        )

    aware = mean_for("fitted_action_aware")
    unaware = mean_for("fitted_action_unaware")
    full = mean_for(full_context_updater_id)
    mechanisms = sorted(
        {
            str(row.get("mechanism"))
            for row in natural
            if row.get("mechanism")
            and row.get("mechanism") != "balanced"
        }
    )
    domains = sorted(
        {
            str(row.get("domain"))
            for row in natural
            if row.get("domain") in {"travel", "writing"}
        }
    )

    def mean_for_domain(
        updater: str,
        domain: str,
        *,
        mechanism: str | None = None,
    ) -> float | None:
        return _finite_mean(
            row["brier"]
            for row in natural
            if row.get("updater_id") == updater
            and row.get("domain") == domain
            and (
                mechanism is None
                or row.get("mechanism") == mechanism
            )
        )

    domain_aware_gaps = {
        domain: (
            None
            if mean_for_domain("fitted_action_aware", domain) is None
            or mean_for_domain("fitted_action_unaware", domain) is None
            else mean_for_domain("fitted_action_unaware", domain)
            - mean_for_domain("fitted_action_aware", domain)
        )
        for domain in domains
    }
    mechanism_domain_gaps = {
        mechanism: {
            domain: (
                None
                if mean_for_domain(
                    full_context_updater_id,
                    domain,
                    mechanism=mechanism,
                )
                is None
                or mean_for_domain(
                    "fitted_action_aware",
                    domain,
                    mechanism=mechanism,
                )
                is None
                else mean_for_domain(
                    full_context_updater_id,
                    domain,
                    mechanism=mechanism,
                )
                - mean_for_domain(
                    "fitted_action_aware",
                    domain,
                    mechanism=mechanism,
                )
            )
            for domain in domains
        }
        for mechanism in mechanisms
    }
    material_mechanisms = sum(
        len(domain_gaps) == 2
        and all(
            gap is not None and gap > material_gap
            for gap in domain_gaps.values()
        )
        for domain_gaps in mechanism_domain_gaps.values()
    )
    return GateReport(
        gate_id="gate-1",
        title="Learnable provenance gap",
        criteria=(
            GateCriterion(
                "aware-beats-unaware",
                "Fitted action-aware inference has lower held-out Brier error.",
                (
                    None
                    if set(domains) != {"travel", "writing"}
                    or any(value is None for value in domain_aware_gaps.values())
                    else all(
                        value is not None and value > 0
                        for value in domain_aware_gaps.values()
                    )
                ),
                {
                    "aware": aware,
                    "unaware": unaware,
                    "unaware_minus_aware_by_domain": domain_aware_gaps,
                },
                "aware < unaware separately in travel and writing",
            ),
            GateCriterion(
                "full-context-gap",
                "The declared full-context writer remains materially worse than fitted aware.",
                None if full is None or aware is None else full - aware > material_gap,
                {
                    "full_context_updater_id": full_context_updater_id,
                    "full_context": full,
                    "aware": aware,
                    "material_gap": material_gap,
                },
                f"full_context - aware > {material_gap}",
            ),
            GateCriterion(
                "mechanism-transfer",
                "The material gap appears across enough provenance mechanisms.",
                material_mechanisms >= required_mechanisms if mechanisms else None,
                {
                    "material_mechanisms": material_mechanisms,
                    "required": required_mechanisms,
                    "gaps_by_domain": mechanism_domain_gaps,
                },
                (
                    f"at least {required_mechanisms} mechanisms with a "
                    "material gap in both domains"
                ),
            ),
            GateCriterion(
                "domain-transfer",
                "Both declared domains are represented.",
                set(domains) == {"travel", "writing"} if natural else None,
                domains,
                "travel and writing",
            ),
            GateCriterion(
                "held-out-paraphrase-transfer",
                "The effect transfers to held-out surface paraphrases.",
                held_out_paraphrase_verified,
                {"verified": held_out_paraphrase_verified},
                "complete held-out paraphrase evaluation",
            ),
        ),
    )


def gate_2_and_3_from_trajectories(
    rows: Iterable[Mapping[str, Any]],
    *,
    target_updater_ids: Iterable[str] | None = None,
    inferential_evidence: Mapping[str, Any] | None = None,
) -> tuple[GateReport, GateReport]:
    material = tuple(rows)
    targets = (
        None
        if target_updater_ids is None
        else set(target_updater_ids)
    )
    soft = [
        row
        for row in material
        if row.get("policy_id") == "soft_profile_conditioned"
        and row.get("initial_profile") == "incorrect"
        and (
            targets is None
            or row.get("updater_id") in targets
        )
    ]
    cases = [row for row in soft if bool(row.get("is_self_confirming"))]
    self_confirming = len(cases)
    positive_lcg = sum(
        float(row.get("cumulative_lcg", 0.0)) > 0 for row in cases
    )
    influenced = sum(
        bool(row.get("profile_changed_later_action")) for row in cases
    )
    attribution = _finite_mean(
        float(row["attribution_cost"])
        for row in soft
        if row.get("attribution_cost") is not None
    )
    mechanisms = {
        str(mechanism)
        for row in cases
        for mechanism in row.get("mechanisms", ())
    }

    def inference_metric(metric_id: str) -> Mapping[str, Any] | None:
        if not isinstance(inferential_evidence, Mapping):
            return None
        metrics = inferential_evidence.get("metrics")
        if not isinstance(metrics, Mapping):
            return None
        value = metrics.get(metric_id)
        return value if isinstance(value, Mapping) else None

    def adequate(metric: Mapping[str, Any] | None) -> bool | None:
        if metric is None or metric.get("adequacy_status") == "not_computed":
            return None
        return bool(metric.get("adequate", False))

    def lower_above_zero(
        metric: Mapping[str, Any] | None,
    ) -> bool | None:
        if metric is None:
            return None
        lower = metric.get("lower")
        if not isinstance(lower, (int, float)) or not math.isfinite(
            float(lower)
        ):
            return None
        return float(lower) > 0.0

    lcg_interval = inference_metric("mean_cumulative_lcg")
    self_confirming_interval = inference_metric(
        "self_confirming_profile_rate"
    )
    attribution_interval = inference_metric("profile_attribution_cost")
    gate_2 = GateReport(
        gate_id="gate-2",
        title="Nontrivial soft self-confirmation",
        criteria=(
            GateCriterion(
                "soft-mechanism-coverage",
                "Soft self-confirmation appears under a presentation channel.",
                len(mechanisms) >= 1 if cases else (False if soft else None),
                {
                    "mechanisms_among_five_clause_cases": sorted(mechanisms),
                    "five_clause_cases": len(cases),
                },
                "at least one of ranking, default, suggestion",
            ),
            GateCriterion(
                "soft-alternatives",
                "Counter-profile alternatives remained available.",
                (
                    all(
                        bool(row.get("counter_profile_available", False))
                        for row in cases
                    )
                    if cases
                    else (False if soft else None)
                ),
                {"eligible_five_clause_cases": len(cases)},
                "all five-clause cases retain both directions",
            ),
            GateCriterion(
                "positive-lcg",
                "At least one incorrect-seed trajectory has positive LCG.",
                positive_lcg > 0 if soft else None,
                {"positive_lcg": positive_lcg},
                "positive_lcg > 0",
            ),
            GateCriterion(
                "later-action",
                "A strengthened profile changes a later action.",
                influenced > 0 if soft else None,
                {"influenced": influenced},
                "influenced > 0",
            ),
            GateCriterion(
                "five-clause-cases",
                "At least one case satisfies all five definitional clauses.",
                self_confirming > 0 if soft else None,
                {"self_confirming": self_confirming},
                "self_confirming > 0",
            ),
            GateCriterion(
                "independent-user-cluster-adequacy",
                (
                    "The primary LCG and five-clause-rate intervals use enough "
                    "independent latent-user clusters."
                ),
                (
                    None
                    if adequate(lcg_interval) is None
                    or adequate(self_confirming_interval) is None
                    else bool(
                        adequate(lcg_interval)
                        and adequate(self_confirming_interval)
                    )
                ),
                {
                    "mean_cumulative_lcg": lcg_interval,
                    "self_confirming_profile_rate": (
                        self_confirming_interval
                    ),
                },
                "both primary user-clustered intervals are adequate",
            ),
            GateCriterion(
                "positive-lcg-clustered-interval",
                "The user-clustered LCG interval excludes zero.",
                lower_above_zero(lcg_interval),
                lcg_interval,
                "95% interval lower bound > 0",
            ),
            GateCriterion(
                "nonzero-five-clause-rate-clustered-interval",
                (
                    "The user-clustered five-clause profile-rate interval "
                    "excludes zero."
                ),
                lower_above_zero(self_confirming_interval),
                self_confirming_interval,
                "95% interval lower bound > 0",
            ),
        ),
    )
    gate_3 = GateReport(
        gate_id="gate-3",
        title="Attribution beyond evidence selection",
        criteria=(
            GateCriterion(
                "same-history-attribution",
                "System error exceeds the aware shadow on the same histories.",
                None if attribution is None else attribution > 0,
                {"mean_attribution_cost": attribution},
                "mean attribution cost > 0",
            ),
            GateCriterion(
                "independent-user-cluster-adequacy",
                (
                    "The primary same-history attribution interval uses "
                    "enough independent latent-user clusters."
                ),
                adequate(attribution_interval),
                attribution_interval,
                "primary user-clustered interval is adequate",
            ),
            GateCriterion(
                "positive-attribution-clustered-interval",
                (
                    "The user-clustered same-history attribution interval "
                    "excludes zero."
                ),
                lower_above_zero(attribution_interval),
                attribution_interval,
                "95% interval lower bound > 0",
            ),
        ),
    )
    return gate_2, gate_3


def incomplete_gate(gate_id: int, title: str, reason: str) -> GateReport:
    return GateReport(
        gate_id=f"gate-{gate_id}",
        title=title,
        criteria=(
            GateCriterion(
                "not-evaluated",
                reason,
                None,
                None,
                "complete the required retained experiment",
            ),
        ),
    )
