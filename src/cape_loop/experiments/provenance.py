"""Experiment A: controlled and naturally sampled provenance audits."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from itertools import product
import json
import math
from statistics import mean
from typing import Any, Mapping, Sequence

from ..beliefs import JointThetaPsiBelief, PreferenceBelief
from ..domains import (
    DATA_SPLITS,
    DOMAINS,
    DomainSpec,
    dialogue_template_id,
    scenario_family_id,
)
from ..elicitation import MECHANISMS, build_matched_anchor_set
from ..fitting import AwareConditionalLogitModel
from ..metrics import (
    action_conditioned_update_error,
    marginal_brier,
    marginal_kl,
    marginal_l1,
    update_direction_accuracy_details,
)
from ..response import RandomUtilityModel
from ..rng import weighted_index
from ..schemas import (
    InteractionContext,
    LatentUser,
    Observation,
    PolicyProvenance,
    Susceptibility,
    THETA_VALUES,
)
from ..statistics import (
    ClusterRobustOLSResult,
    IntervalEstimate,
    MarginalForecast,
    PairedContrast,
    RawCalibratedComparison,
    compare_raw_and_calibrated_forecasts,
    fit_cluster_robust_ols,
    paired_cluster_contrast,
    paired_cluster_interaction,
    percentile,
)
from ..updaters import (
    ExactActionAwareUpdater,
    FittedActionAwareUpdater,
    ProfileUpdater,
    build_updater_registry,
    make_update_view,
)


@dataclass(frozen=True, slots=True)
class ExperimentARow:
    """One updater evaluated on one matched context and response."""

    trial_id: str
    user_id: str
    domain_id: str
    target_attribute: int
    anchor_direction: int
    prior_stratum: str
    prior_strength: float
    mechanism: str
    response_mode: str
    updater_id: str
    selected_option_id: str
    anchor_option_id: str
    context: InteractionContext
    provenance: PolicyProvenance
    observation: Observation
    prior: PreferenceBelief
    posterior: PreferenceBelief
    fitted_aware_posterior: PreferenceBelief
    exact_posterior: PreferenceBelief
    posterior_theta_psi: JointThetaPsiBelief | None
    exact_theta_psi: JointThetaPsiBelief | None
    acue: float
    fitted_aware_kl: float
    exact_kl: float
    brier: float
    fitted_aware_brier: float
    excess_brier: float
    update_direction_accuracy: float | None
    update_direction_evaluated_components: int
    update_direction_excluded_components: int
    update_magnitude: float
    evidence_weight: float

    @property
    def anchor_selected(self) -> bool:
        return self.selected_option_id == self.anchor_option_id

    @property
    def log_odds_update(self) -> float:
        """Target-attribute positive-direction log-odds update.

        This follows the directional-log-odds definition in the statistical
        plan. Boundary values are clipped only for this numerical diagnostic;
        the retained probability vectors remain unchanged.
        """

        return _positive_log_odds_update(
            self.prior,
            self.posterior,
            self.target_attribute,
        )

    @property
    def fitted_aware_log_odds_update(self) -> float:
        return _positive_log_odds_update(
            self.prior,
            self.fitted_aware_posterior,
            self.target_attribute,
        )

    @property
    def fitted_evidence_strength(self) -> float:
        """Absolute fitted-aware update toward the controlled anchor."""

        return abs(
            _directional_log_odds_update(
                self.prior,
                self.fitted_aware_posterior,
                self.target_attribute,
                self.anchor_direction,
            )
        )

    def to_dict(
        self,
        *,
        include_joint_states: bool = True,
    ) -> dict[str, Any]:
        result = {
            "trial_id": self.trial_id,
            "exact_reference_id": self.trial_id,
            "user_id": self.user_id,
            "domain_id": self.domain_id,
            "target_attribute": self.target_attribute,
            "anchor_direction": self.anchor_direction,
            "prior_stratum": self.prior_stratum,
            "prior_strength": self.prior_strength,
            "mechanism": self.mechanism,
            "response_mode": self.response_mode,
            "updater_id": self.updater_id,
            "selected_option_id": self.selected_option_id,
            "anchor_option_id": self.anchor_option_id,
            "anchor_selected": self.anchor_selected,
            "context": self.context.to_dict(),
            "policy_provenance": self.provenance.to_dict(),
            "observation": self.observation.to_dict(),
            "prior": self.prior.to_dict(),
            "posterior": self.posterior.to_dict(),
            "fitted_aware_posterior": self.fitted_aware_posterior.to_dict(),
            "prior_marginals": self.prior.marginals().to_dict()["probabilities"],
            "posterior_marginals": (
                self.posterior.marginals().to_dict()["probabilities"]
            ),
            "fitted_aware_marginals": (
                self.fitted_aware_posterior.marginals().to_dict()[
                    "probabilities"
                ]
            ),
            "exact_marginals": (
                self.exact_posterior.marginals().to_dict()["probabilities"]
            ),
            "metrics": {
                "acue": self.acue,
                "fitted_aware_kl": self.fitted_aware_kl,
                "exact_kl": self.exact_kl,
                "brier": self.brier,
                "fitted_aware_brier": self.fitted_aware_brier,
                "excess_brier": self.excess_brier,
                "update_direction_accuracy": self.update_direction_accuracy,
                "update_direction_evaluated_components": (
                    self.update_direction_evaluated_components
                ),
                "update_direction_excluded_components": (
                    self.update_direction_excluded_components
                ),
                "update_magnitude": self.update_magnitude,
                "evidence_weight": self.evidence_weight,
                "log_odds_update": self.log_odds_update,
                "fitted_aware_log_odds_update": (
                    self.fitted_aware_log_odds_update
                ),
                "fitted_evidence_strength": self.fitted_evidence_strength,
            },
        }
        if include_joint_states:
            result["exact_posterior"] = self.exact_posterior.to_dict()
            result["posterior_theta_psi"] = (
                None
                if self.posterior_theta_psi is None
                else self.posterior_theta_psi.to_dict()
            )
            result["exact_theta_psi"] = (
                None
                if self.exact_theta_psi is None
                else self.exact_theta_psi.to_dict()
            )
        return result


@dataclass(frozen=True, slots=True)
class ExcludedMatchedSet:
    user_id: str
    domain_id: str
    target_attribute: int
    anchor_direction: int
    minimum_probability: float
    choice_probabilities: tuple[tuple[str, float], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "domain_id": self.domain_id,
            "target_attribute": self.target_attribute,
            "anchor_direction": self.anchor_direction,
            "minimum_probability": self.minimum_probability,
            "choice_probabilities": dict(self.choice_probabilities),
        }


@dataclass(frozen=True, slots=True)
class ExperimentAResult:
    rows: tuple[ExperimentARow, ...]
    excluded: tuple[ExcludedMatchedSet, ...]
    updater_views: tuple[tuple[str, str], ...]

    @property
    def controlled_rows(self) -> tuple[ExperimentARow, ...]:
        return tuple(row for row in self.rows if row.response_mode == "controlled_anchor")

    @property
    def natural_rows(self) -> tuple[ExperimentARow, ...]:
        return tuple(row for row in self.rows if row.response_mode == "naturally_sampled")

    def summary(self) -> dict[str, Any]:
        by_updater: dict[str, list[ExperimentARow]] = {}
        for row in self.rows:
            by_updater.setdefault(row.updater_id, []).append(row)
        return {
            "row_count": len(self.rows),
            "controlled_row_count": len(self.controlled_rows),
            "natural_row_count": len(self.natural_rows),
            "excluded_matched_sets": len(self.excluded),
            "prior_strata": [
                {
                    "prior_stratum": prior_stratum,
                    "prior_strength": prior_strength,
                }
                for prior_stratum, prior_strength in sorted(
                    {
                        (row.prior_stratum, row.prior_strength)
                        for row in self.rows
                    }
                )
            ],
            "updaters": {
                updater_id: {
                    "mean_acue": math.fsum(item.acue for item in items)
                    / len(items),
                    "mean_excess_brier": math.fsum(
                        item.excess_brier for item in items
                    )
                    / len(items),
                }
                for updater_id, items in sorted(by_updater.items())
            },
        }

    def oracle_update_slopes(
        self,
        *,
        response_mode: str = "controlled_anchor",
        replicates: int = 2000,
        seed: int = 1729,
    ) -> tuple[OracleUpdateSlope, ...]:
        """Fit the proposal's updater-versus-aware directional slopes."""

        return estimate_oracle_update_slopes(
            self.rows,
            response_mode=response_mode,
            replicates=replicates,
            seed=seed,
        )

    def evidence_strength_analysis(
        self,
        *,
        response_mode: str = "controlled_anchor",
        volunteered_strengths: Mapping[str, float] | None = None,
    ) -> EvidenceStrengthAnalysis:
        """Derive, rather than assume, the fitted evidence-strength ordering."""

        return fitted_evidence_strength_ordering(
            self.rows,
            response_mode=response_mode,
            volunteered_strengths=volunteered_strengths,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment": "A",
            "rows": [row.to_dict() for row in self.rows],
            "excluded": [item.to_dict() for item in self.excluded],
            "updater_views": dict(self.updater_views),
            "summary": self.summary(),
        }


def default_audit_users() -> tuple[LatentUser, ...]:
    """A small deterministic smoke population; paper runs should pass a split."""

    return (
        LatentUser(
            "audit-user-negative",
            (-2, -1, 1),
            Susceptibility(ranking=0.35, default=0.80, suggestion=0.65),
        ),
        LatentUser(
            "audit-user-positive",
            (2, 1, -1),
            Susceptibility(ranking=0.15, default=0.45, suggestion=0.85),
        ),
    )


def _prior_strength_label(strength: float) -> str:
    return "truth-mixture-" + format(strength, ".8g").replace(".", "p")


def _truth_aligned_prior(
    theta: tuple[int, int, int],
    strength: float,
) -> PreferenceBelief:
    """Mix a uniform joint prior with truth-aligned mass.

    Experiment A crosses this declared concentration factor while keeping the
    same prior within every updater/mechanism matched set. Truth alignment is
    balanced by the latent population and avoids conflating prior direction
    with prior strength.
    """

    if (
        isinstance(strength, bool)
        or not isinstance(strength, (int, float))
        or not math.isfinite(float(strength))
        or not 0.0 <= float(strength) < 1.0
    ):
        raise ValueError("Experiment A prior strengths must lie in [0, 1)")
    concentration = float(strength)
    uniform = PreferenceBelief.uniform()
    point = PreferenceBelief.point_mass(theta)
    return PreferenceBelief(
        tuple(
            (1.0 - concentration) * baseline
            + concentration * target
            for baseline, target in zip(
                uniform.probabilities,
                point.probabilities,
            )
        )
    )


@dataclass(frozen=True, slots=True)
class ExperimentAControlCase:
    """One fixed positive/negative control protocol.

    These records specify the stimulus and expected diagnostic direction. They
    intentionally do not manufacture an ``InteractionContext`` when the core
    choice schema cannot faithfully represent volunteered, indifferent, or
    randomized evidence.
    """

    control_id: str
    polarity: str
    signal_kind: str
    turn_count: int
    construction: str
    expected_diagnostic: str
    execution_stage: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "control_id": self.control_id,
            "polarity": self.polarity,
            "signal_kind": self.signal_kind,
            "turn_count": self.turn_count,
            "construction": self.construction,
            "expected_diagnostic": self.expected_diagnostic,
            "execution_stage": self.execution_stage,
        }


@dataclass(frozen=True, slots=True)
class ExperimentAControlBattery:
    """Versioned preregistration artifact for all proposal controls."""

    battery_id: str
    cases: tuple[ExperimentAControlCase, ...]
    battery_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "battery_id": self.battery_id,
            "battery_sha256": self.battery_sha256,
            "status": "fixed_protocol_not_scored_by_one_step_choice_runner",
            "cases": [case.to_dict() for case in self.cases],
            "interpretation": (
                "This artifact fixes the control stimuli and expected "
                "diagnostics. Outcomes require the declared longitudinal, "
                "direct-statement, or randomized-response executor; absent "
                "outcomes must not be imputed from anchor-choice rows."
            ),
        }


def build_experiment_a_control_battery() -> ExperimentAControlBattery:
    """Return the proposal's fixed positive and negative control protocol."""

    cases = (
        ExperimentAControlCase(
            "positive-volunteered-preference",
            "positive",
            "explicit_volunteered_preference",
            1,
            (
                "A user-originated, unprompted statement names one target "
                "attribute and direction; no option, default, ranking, or "
                "agent suggestion is present."
            ),
            (
                "The updater moves toward the volunteered direction and does "
                "not apply a provenance discount."
            ),
            "direct_statement_executor",
        ),
        ExperimentAControlCase(
            "positive-repeated-balanced-cross-context",
            "positive",
            "repeated_balanced_cross_context_choices",
            3,
            (
                "Three independently worded balanced choices share a target "
                "direction but use disjoint scenario and option templates."
            ),
            (
                "Evidence accumulates in the repeated direction without "
                "being treated as policy-conditioned acceptance."
            ),
            "longitudinal_control_executor",
        ),
        ExperimentAControlCase(
            "positive-direct-correction",
            "positive",
            "direct_correction",
            1,
            (
                "A neutral direct statement explicitly corrects one existing "
                "profile attribute after the same frozen prehistory."
            ),
            (
                "The corrected attribute moves toward the stated direction; "
                "recovery time is analyzed separately as correction debt."
            ),
            "correction_debt_executor",
        ),
        ExperimentAControlCase(
            "negative-indifferent-response",
            "negative",
            "indifferent_response",
            1,
            (
                "The user explicitly reports indifference between options "
                "matched on non-target utility."
            ),
            "No directional target-preference update is justified.",
            "indifference_response_executor",
        ),
        ExperimentAControlCase(
            "negative-random-choice",
            "negative",
            "random_choice",
            3,
            (
                "Selections are generated by a registered randomization "
                "device independent of theta and presentation context."
            ),
            (
                "Aggregate directional update is null; individual randomized "
                "choices are not interpreted as preference evidence."
            ),
            "randomized_response_executor",
        ),
        ExperimentAControlCase(
            "negative-nondistinguishing-response",
            "negative",
            "target_nondistinguishing_response",
            1,
            (
                "All displayed options have the same feature value on the "
                "declared target attribute while differing only elsewhere."
            ),
            "No update to the target-attribute marginal is justified.",
            "nondistinguishing_context_executor",
        ),
    )
    canonical = json.dumps(
        [case.to_dict() for case in cases],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return ExperimentAControlBattery(
        battery_id="experiment-a-controls-v1",
        cases=cases,
        battery_sha256=sha256(canonical).hexdigest(),
    )


def _default_updaters(
    response_model: RandomUtilityModel,
    aware_model: AwareConditionalLogitModel,
) -> dict[str, ProfileUpdater]:
    return build_updater_registry(
        (
            "no_update",
            "exact_action_aware",
            "fitted_action_aware",
            "fitted_action_unaware",
            "response_only",
            "full_context_blind",
            "provenance_discount",
            "provenance_aware",
            "conservative",
        ),
        response_model=response_model,
        aware_model=aware_model,
    )


def _validate_registry(
    updaters: Mapping[str, ProfileUpdater],
) -> dict[str, ProfileUpdater]:
    if not updaters:
        raise ValueError("Experiment A requires at least one updater")
    result = dict(updaters)
    for key, updater in result.items():
        if key != updater.updater_id:
            raise ValueError(
                f"updater registry key {key!r} differs from {updater.updater_id!r}"
            )
    return result


def run_provenance_audit(
    *,
    users: Sequence[LatentUser] | None = None,
    domains: Sequence[DomainSpec] = DOMAINS,
    updaters: Mapping[str, ProfileUpdater] | None = None,
    prior: PreferenceBelief | None = None,
    prior_strengths: Sequence[float] | None = None,
    response_model: RandomUtilityModel | None = None,
    fitted_aware_model: AwareConditionalLogitModel | None = None,
    mechanisms: Sequence[str] = MECHANISMS,
    response_modes: Sequence[str] = (
        "controlled_anchor",
        "naturally_sampled",
    ),
    minimum_probability: float = 0.05,
    direction_tolerance: float = 1e-9,
    seed: int = 1729,
    data_split: str = "test",
) -> ExperimentAResult:
    """Run both estimands while retaining one row per declared updater.

    Controlled rows hold the anchor response fixed.  Natural rows draw a
    separate response in every action context, as required by the proposal.
    """

    population = tuple(default_audit_users() if users is None else users)
    domain_specs = tuple(domains)
    if not population or not domain_specs:
        raise ValueError("Experiment A requires users and domains")
    if len({user.user_id for user in population}) != len(population):
        raise ValueError("Experiment A user IDs must be unique")
    requested_mechanisms = tuple(mechanisms)
    if not requested_mechanisms or not set(requested_mechanisms) <= set(MECHANISMS):
        raise ValueError(f"mechanisms must be selected from {MECHANISMS}")
    requested_modes = tuple(response_modes)
    allowed_modes = {"controlled_anchor", "naturally_sampled"}
    if not requested_modes or not set(requested_modes) <= allowed_modes:
        raise ValueError(f"response_modes must be selected from {sorted(allowed_modes)}")
    if direction_tolerance < 0:
        raise ValueError("direction_tolerance must be non-negative")
    if data_split not in DATA_SPLITS:
        raise ValueError(f"data_split must be one of {DATA_SPLITS}")

    if prior is not None and prior_strengths is not None:
        raise ValueError(
            "supply either an explicit Experiment A prior or prior strengths, "
            "not both"
        )
    raw_prior_strengths = (
        (0.0,) if prior_strengths is None else tuple(prior_strengths)
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        for value in raw_prior_strengths
    ):
        raise TypeError("Experiment A prior strengths must be numeric")
    requested_prior_strengths = tuple(
        float(value) for value in raw_prior_strengths
    )
    if prior is None:
        if not requested_prior_strengths:
            raise ValueError("Experiment A prior strengths cannot be empty")
        if len(set(requested_prior_strengths)) != len(
            requested_prior_strengths
        ):
            raise ValueError("Experiment A prior strengths must be distinct")
        for strength in requested_prior_strengths:
            _truth_aligned_prior((1, 1, 1), strength)
    declared_response = response_model or RandomUtilityModel()
    aware_model = fitted_aware_model or AwareConditionalLogitModel(
        (
            declared_response.beta,
            declared_response.ranking_scale * 0.35,
            declared_response.default_scale * 0.80,
            declared_response.suggestion_scale * 0.65,
        )
    )
    registry = _validate_registry(
        _default_updaters(declared_response, aware_model)
        if updaters is None
        else updaters
    )
    aware_reference: ProfileUpdater = registry.get(
        "fitted_action_aware",
        FittedActionAwareUpdater(aware_model),
    )
    # A configured paper run may supply the full Cartesian susceptibility grid.
    # Preserve that exact reference rather than replacing it with smoke defaults.
    exact_reference: ProfileUpdater = registry.get(
        "exact_action_aware",
        ExactActionAwareUpdater(declared_response),
    )

    rows: list[ExperimentARow] = []
    excluded: list[ExcludedMatchedSet] = []
    for domain in domain_specs:
        for user in population:
            for target_attribute, anchor_direction in product(
                range(3),
                (-1, 1),
            ):
                scenario_id = (
                    f"{scenario_family_id(domain.domain_id, data_split)}:"
                    "experiment-a:"
                    f"attribute-{target_attribute}:direction-{anchor_direction:+d}"
                )
                matched = build_matched_anchor_set(
                    domain,
                    target_attribute=target_attribute,
                    anchor_direction=anchor_direction,
                    scenario_id=scenario_id,
                    wording_template=dialogue_template_id(
                        domain.domain_id,
                        data_split,
                    ),
                )
                probabilities = matched.choice_probabilities(
                    user,
                    declared_response,
                )
                if not matched.eligible(
                    user,
                    declared_response,
                    minimum_probability=minimum_probability,
                ):
                    excluded.append(
                        ExcludedMatchedSet(
                            user.user_id,
                            domain.domain_id,
                            target_attribute,
                            anchor_direction,
                            minimum_probability,
                            tuple(sorted(probabilities.items())),
                        )
                    )
                    continue

                if prior is None:
                    prior_specs = tuple(
                        (
                            _prior_strength_label(strength),
                            strength,
                            _truth_aligned_prior(user.theta, strength),
                        )
                        for strength in requested_prior_strengths
                    )
                else:
                    normalized_strength = 1.0 - (
                        prior.entropy() / math.log(len(prior.probabilities))
                    )
                    prior_specs = (
                        ("explicit-custom", normalized_strength, prior),
                    )

                for (
                    prior_stratum,
                    prior_strength,
                    prior_belief,
                    mechanism,
                ) in (
                    (
                        prior_label,
                        prior_concentration,
                        prior_candidate,
                        mechanism_id,
                    )
                    for (
                        prior_label,
                        prior_concentration,
                        prior_candidate,
                    ) in prior_specs
                    for mechanism_id in requested_mechanisms
                ):
                    context = matched.context(mechanism)
                    provenance = PolicyProvenance(
                        policy_id=f"provenance_audit_{mechanism}",
                        policy_version="v1",
                        profile_snapshot=tuple(
                            (
                                f"attribute_{index + 1}",
                                expectation,
                            )
                            for index, expectation in enumerate(
                                prior_belief.expected_theta()
                            )
                        ),
                        random_seed=seed,
                        presentation_mechanism={
                            "balanced": "balanced",
                            "restricted": "restriction",
                            "default": "default",
                            "suggested": "suggestion",
                        }[mechanism],
                        profile_conditioned=False,
                    )
                    for response_mode in requested_modes:
                        event_id = (
                            f"{scenario_id}:user-{user.user_id}:"
                            f"prior-{prior_stratum}:"
                            f"{mechanism}:{response_mode}"
                        )
                        if response_mode == "controlled_anchor":
                            observation = Observation(
                                matched.anchor_option_id,
                                choice_noise_key=(
                                    f"{scenario_id}:controlled-anchor"
                                ),
                            )
                        else:
                            observation = declared_response.sample(
                                user.theta,
                                user.susceptibility,
                                context,
                                seed,
                                noise_key=(
                                    "experiment-a-natural",
                                    user.user_id,
                                    domain.domain_id,
                                    target_attribute,
                                    anchor_direction,
                                    mechanism,
                                ),
                            )

                        aware_state = aware_reference.initial_state(prior_belief)
                        aware_view = make_update_view(
                            aware_reference.view_kind,
                            context,
                            observation,
                            provenance,
                            event_id=f"{event_id}:aware-reference",
                        )
                        aware_result = aware_reference.update(
                            aware_state,
                            aware_view,
                        )
                        aware_after = aware_result.state.belief

                        exact_state = exact_reference.initial_state(prior_belief)
                        exact_view = make_update_view(
                            exact_reference.view_kind,
                            context,
                            observation,
                            provenance,
                            event_id=f"{event_id}:exact-reference",
                        )
                        exact_result = exact_reference.update(
                            exact_state,
                            exact_view,
                        )
                        exact_after = exact_result.state.belief

                        for updater_id, updater in registry.items():
                            state = updater.initial_state(prior_belief)
                            view = make_update_view(
                                updater.view_kind,
                                context,
                                observation,
                                provenance,
                                event_id=f"{event_id}:{updater_id}",
                            )
                            result = updater.update(state, view)
                            posterior = result.state.belief
                            (
                                direction_accuracy,
                                direction_evaluated,
                                direction_excluded,
                            ) = update_direction_accuracy_details(
                                prior_belief,
                                posterior,
                                prior_belief,
                                aware_after,
                                tolerance=direction_tolerance,
                            )
                            rows.append(
                                ExperimentARow(
                                    trial_id=event_id,
                                    user_id=user.user_id,
                                    domain_id=domain.domain_id,
                                    target_attribute=target_attribute,
                                    anchor_direction=anchor_direction,
                                    prior_stratum=prior_stratum,
                                    prior_strength=prior_strength,
                                    mechanism=mechanism,
                                    response_mode=response_mode,
                                    updater_id=updater_id,
                                    selected_option_id=(
                                        observation.selected_option_id
                                    ),
                                    anchor_option_id=matched.anchor_option_id,
                                    context=context,
                                    provenance=provenance,
                                    observation=observation,
                                    prior=prior_belief,
                                    posterior=posterior,
                                    fitted_aware_posterior=aware_after,
                                    exact_posterior=exact_after,
                                    posterior_theta_psi=(
                                        result.state.joint_belief
                                    ),
                                    exact_theta_psi=(
                                        exact_result.state.joint_belief
                                    ),
                                    acue=action_conditioned_update_error(
                                        prior_belief,
                                        posterior,
                                        prior_belief,
                                        aware_after,
                                    ),
                                    fitted_aware_kl=marginal_kl(
                                        aware_after,
                                        posterior,
                                    ),
                                    exact_kl=marginal_kl(
                                        exact_after,
                                        posterior,
                                    ),
                                    brier=marginal_brier(
                                        posterior,
                                        user.theta,
                                    ),
                                    fitted_aware_brier=marginal_brier(
                                        aware_after,
                                        user.theta,
                                    ),
                                    excess_brier=(
                                        marginal_brier(posterior, user.theta)
                                        - marginal_brier(
                                            aware_after,
                                            user.theta,
                                        )
                                    ),
                                    update_direction_accuracy=(
                                        direction_accuracy
                                    ),
                                    update_direction_evaluated_components=(
                                        direction_evaluated
                                    ),
                                    update_direction_excluded_components=(
                                        direction_excluded
                                    ),
                                    update_magnitude=marginal_l1(
                                        prior_belief,
                                        posterior,
                                    )
                                    / 3.0,
                                    evidence_weight=float(
                                        result.diagnostic(
                                            "evidence_weight",
                                            math.nan,
                                        )
                                    ),
                                )
                            )

    return ExperimentAResult(
        rows=tuple(rows),
        excluded=tuple(excluded),
        updater_views=tuple(
            sorted(
                (
                    updater_id,
                    updater.view_kind.value,
                )
                for updater_id, updater in registry.items()
            )
        ),
    )


def _clipped_logit(probability: float, *, clip: float = 1e-6) -> float:
    if not 0 < clip < 0.5:
        raise ValueError("log-odds clip must lie in (0, 0.5)")
    bounded = min(max(float(probability), clip), 1.0 - clip)
    return math.log(bounded / (1.0 - bounded))


def _positive_log_odds_update(
    prior: PreferenceBelief,
    posterior: PreferenceBelief,
    attribute: int,
    *,
    clip: float = 1e-6,
) -> float:
    return _clipped_logit(
        posterior.sign_mass(attribute, 1),
        clip=clip,
    ) - _clipped_logit(
        prior.sign_mass(attribute, 1),
        clip=clip,
    )


def _directional_log_odds_update(
    prior: PreferenceBelief,
    posterior: PreferenceBelief,
    attribute: int,
    direction: int,
    *,
    clip: float = 1e-6,
) -> float:
    return _clipped_logit(
        posterior.sign_mass(attribute, direction),
        clip=clip,
    ) - _clipped_logit(
        prior.sign_mass(attribute, direction),
        clip=clip,
    )


def _fit_line(
    x_values: Sequence[float],
    y_values: Sequence[float],
) -> tuple[float, float, tuple[float, ...]]:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        raise ValueError("slope estimation needs at least two paired observations")
    x_mean = mean(x_values)
    y_mean = mean(y_values)
    denominator = math.fsum((value - x_mean) ** 2 for value in x_values)
    if denominator <= 1e-18:
        raise ValueError("aware log-odds updates have no estimable variation")
    slope = (
        math.fsum(
            (x_value - x_mean) * (y_value - y_mean)
            for x_value, y_value in zip(x_values, y_values)
        )
        / denominator
    )
    intercept = y_mean - slope * x_mean
    residuals = tuple(
        y_value - (intercept + slope * x_value)
        for x_value, y_value in zip(x_values, y_values)
    )
    return intercept, slope, residuals


@dataclass(frozen=True, slots=True)
class MechanismResidual:
    mechanism: str
    observation_count: int
    mean_residual: float
    root_mean_squared_residual: float

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "mechanism": self.mechanism,
            "observation_count": self.observation_count,
            "mean_residual": self.mean_residual,
            "root_mean_squared_residual": self.root_mean_squared_residual,
        }


@dataclass(frozen=True, slots=True)
class OracleUpdateSlope:
    """Directional log-odds slope against the fitted action-aware update."""

    updater_id: str
    response_mode: str
    observation_count: int
    user_cluster_count: int
    intercept: float
    slope: float
    root_mean_squared_residual: float
    mechanism_residuals: tuple[MechanismResidual, ...]
    slope_interval: IntervalEstimate | None
    inference_status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "updater_id": self.updater_id,
            "response_mode": self.response_mode,
            "observation_count": self.observation_count,
            "user_cluster_count": self.user_cluster_count,
            "intercept": self.intercept,
            "slope": self.slope,
            "root_mean_squared_residual": self.root_mean_squared_residual,
            "mechanism_residuals": [
                residual.to_dict() for residual in self.mechanism_residuals
            ],
            "slope_interval": (
                None
                if self.slope_interval is None
                else self.slope_interval.to_dict()
            ),
            "inference_status": self.inference_status,
            "estimand": (
                "system target-attribute positive-direction log-odds update "
                "regressed on the fitted action-aware update"
            ),
        }


def estimate_oracle_update_slopes(
    rows: Sequence[ExperimentARow],
    *,
    response_mode: str = "controlled_anchor",
    replicates: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 1729,
) -> tuple[OracleUpdateSlope, ...]:
    """Estimate updater slopes and complete-user bootstrap intervals."""

    if replicates <= 0:
        raise ValueError("slope bootstrap requires positive replicates")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must lie in (0, 1)")
    selected = tuple(row for row in rows if row.response_mode == response_mode)
    if not selected:
        raise ValueError(f"no Experiment A rows for response mode {response_mode!r}")
    results = []
    updater_ids = sorted({row.updater_id for row in selected})
    for updater_id in updater_ids:
        updater_rows = tuple(
            row for row in selected if row.updater_id == updater_id
        )
        x_values = tuple(
            row.fitted_aware_log_odds_update for row in updater_rows
        )
        y_values = tuple(row.log_odds_update for row in updater_rows)
        intercept, slope, residuals = _fit_line(x_values, y_values)
        residual_rows = []
        for mechanism in sorted({row.mechanism for row in updater_rows}):
            values = tuple(
                residual
                for row, residual in zip(updater_rows, residuals)
                if row.mechanism == mechanism
            )
            residual_rows.append(
                MechanismResidual(
                    mechanism=mechanism,
                    observation_count=len(values),
                    mean_residual=mean(values),
                    root_mean_squared_residual=math.sqrt(
                        math.fsum(value * value for value in values)
                        / len(values)
                    ),
                )
            )

        by_user: dict[str, list[ExperimentARow]] = {}
        for row in updater_rows:
            by_user.setdefault(row.user_id, []).append(row)
        user_ids = sorted(by_user)
        slope_draws: list[float] = []
        if len(user_ids) >= 2:
            weights = [1.0] * len(user_ids)
            for replicate in range(replicates):
                sampled_rows = [
                    row
                    for draw in range(len(user_ids))
                    for row in by_user[
                        user_ids[
                            weighted_index(
                                weights,
                                seed,
                                "oracle-update-slope",
                                response_mode,
                                updater_id,
                                replicate,
                                draw,
                            )
                        ]
                    ]
                ]
                try:
                    _, draw_slope, _ = _fit_line(
                        [
                            row.fitted_aware_log_odds_update
                            for row in sampled_rows
                        ],
                        [row.log_odds_update for row in sampled_rows],
                    )
                except ValueError:
                    continue
                slope_draws.append(draw_slope)
        if slope_draws:
            tail = (1.0 - confidence_level) / 2.0
            slope_interval = IntervalEstimate(
                estimate=slope,
                lower=percentile(slope_draws, tail),
                upper=percentile(slope_draws, 1.0 - tail),
                confidence_level=confidence_level,
                method=(
                    "percentile bootstrap resampling complete latent users"
                ),
                cluster_count=len(user_ids),
                replicate_count=len(slope_draws),
            )
            status = (
                "descriptive_and_cluster_bootstrap; inferential adequacy still "
                "depends on the preregistered participant count"
            )
        else:
            slope_interval = None
            status = (
                "descriptive_only; at least two independently sampled users "
                "with estimable update variation are required for resampling"
            )
        results.append(
            OracleUpdateSlope(
                updater_id=updater_id,
                response_mode=response_mode,
                observation_count=len(updater_rows),
                user_cluster_count=len(user_ids),
                intercept=intercept,
                slope=slope,
                root_mean_squared_residual=math.sqrt(
                    math.fsum(value * value for value in residuals)
                    / len(residuals)
                ),
                mechanism_residuals=tuple(residual_rows),
                slope_interval=slope_interval,
                inference_status=status,
            )
        )
    return tuple(results)


def experiment_a_matched_set_id(row: ExperimentARow) -> str:
    """Return the mechanism-independent identifier for one matched set."""

    return (
        f"{row.user_id}|{row.domain_id}|attribute-{row.target_attribute}|"
        f"direction-{row.anchor_direction:+d}|prior-{row.prior_stratum}|"
        f"{row.response_mode}"
    )


@dataclass(frozen=True, slots=True)
class EvidenceRankGroup:
    rank: int
    mechanisms: tuple[str, ...]
    mean_strength: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "mechanisms": list(self.mechanisms),
            "mean_strength": self.mean_strength,
        }


def _rank_strengths(
    strengths: Mapping[str, float],
    *,
    tie_tolerance: float,
) -> tuple[EvidenceRankGroup, ...]:
    ordered = sorted(strengths, key=lambda key: (-strengths[key], key))
    groups = []
    index = 0
    while index < len(ordered):
        end = index + 1
        while (
            end < len(ordered)
            and abs(
                strengths[ordered[end]] - strengths[ordered[index]]
            )
            <= tie_tolerance
        ):
            end += 1
        labels = tuple(sorted(ordered[index:end]))
        groups.append(
            EvidenceRankGroup(
                rank=len(groups) + 1,
                mechanisms=labels,
                mean_strength=mean(strengths[label] for label in labels),
            )
        )
        index = end
    return tuple(groups)


@dataclass(frozen=True, slots=True)
class MatchedEvidenceStrength:
    matched_set_id: str
    strengths: tuple[tuple[str, float], ...]
    fitted_ordering: tuple[EvidenceRankGroup, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "matched_set_id": self.matched_set_id,
            "strengths": dict(self.strengths),
            "fitted_ordering": [
                rank_group.to_dict()
                for rank_group in self.fitted_ordering
            ],
        }


@dataclass(frozen=True, slots=True)
class EvidenceStrengthAnalysis:
    """Fitted-model-derived ordering without a universal hard-coded ranking."""

    response_mode: str
    matched_sets: tuple[MatchedEvidenceStrength, ...]
    aggregate_strengths: tuple[tuple[str, float], ...]
    aggregate_ordering: tuple[EvidenceRankGroup, ...]
    volunteered_control_coverage: int
    volunteered_control_status: str
    positive_control_note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "response_mode": self.response_mode,
            "reference": "fitted_action_aware",
            "strength_definition": (
                "absolute fitted-aware target-direction log-odds update"
            ),
            "matched_sets": [item.to_dict() for item in self.matched_sets],
            "aggregate_strengths": dict(self.aggregate_strengths),
            "aggregate_ordering": [
                rank_group.to_dict()
                for rank_group in self.aggregate_ordering
            ],
            "volunteered_control_coverage": self.volunteered_control_coverage,
            "volunteered_control_status": self.volunteered_control_status,
            "positive_control_note": self.positive_control_note,
        }


def fitted_evidence_strength_ordering(
    rows: Sequence[ExperimentARow],
    *,
    response_mode: str = "controlled_anchor",
    volunteered_strengths: Mapping[str, float] | None = None,
    tie_tolerance: float = 1e-9,
) -> EvidenceStrengthAnalysis:
    """Derive matched and aggregate evidence orderings from the fitted model.

    The current choice schema cannot encode an unprompted volunteered
    preference as another :class:`InteractionContext`.  Callers may supply an
    externally evaluated positive-control strength keyed by
    :func:`experiment_a_matched_set_id`; its provenance remains explicit.
    """

    if tie_tolerance < 0:
        raise ValueError("tie_tolerance must be non-negative")
    selected = tuple(row for row in rows if row.response_mode == response_mode)
    if not selected:
        raise ValueError(f"no Experiment A rows for response mode {response_mode!r}")
    unique_cells: dict[tuple[str, str], ExperimentARow] = {}
    for row in selected:
        matched_set_id = experiment_a_matched_set_id(row)
        key = (matched_set_id, row.mechanism)
        existing = unique_cells.get(key)
        if existing is not None:
            if any(
                abs(first - second) > 1e-12
                for first, second in zip(
                    existing.fitted_aware_posterior.probabilities,
                    row.fitted_aware_posterior.probabilities,
                )
            ):
                raise ValueError(
                    "updater rows disagree on their fitted-aware reference"
                )
            continue
        unique_cells[key] = row

    external = {} if volunteered_strengths is None else dict(volunteered_strengths)
    if any(
        not math.isfinite(value) or value < 0.0
        for value in external.values()
    ):
        raise ValueError("volunteered strengths must be finite and non-negative")
    by_set: dict[str, dict[str, float]] = {}
    for (matched_set_id, mechanism), row in unique_cells.items():
        by_set.setdefault(matched_set_id, {})[
            mechanism
        ] = row.fitted_evidence_strength
    matched_results = []
    volunteer_count = 0
    for matched_set_id, strengths in sorted(by_set.items()):
        complete_strengths = dict(strengths)
        if matched_set_id in external:
            complete_strengths["volunteered"] = float(external[matched_set_id])
            volunteer_count += 1
        matched_results.append(
            MatchedEvidenceStrength(
                matched_set_id=matched_set_id,
                strengths=tuple(sorted(complete_strengths.items())),
                fitted_ordering=_rank_strengths(
                    complete_strengths,
                    tie_tolerance=tie_tolerance,
                ),
            )
        )
    labels = sorted(
        {
            label
            for result in matched_results
            for label, _ in result.strengths
        }
    )
    aggregate = {
        label: mean(
            dict(result.strengths)[label]
            for result in matched_results
            if label in dict(result.strengths)
        )
        for label in labels
    }
    if not external:
        volunteer_status = (
            "not represented by the choice-context schema; no volunteered "
            "positive-control values were supplied"
        )
    elif volunteer_count == len(matched_results):
        volunteer_status = (
            "externally supplied volunteered positive control available for "
            "every matched set"
        )
    else:
        volunteer_status = (
            "externally supplied volunteered positive control available for "
            f"{volunteer_count} of {len(matched_results)} matched sets"
        )
    return EvidenceStrengthAnalysis(
        response_mode=response_mode,
        matched_sets=tuple(matched_results),
        aggregate_strengths=tuple(sorted(aggregate.items())),
        aggregate_ordering=_rank_strengths(
            aggregate,
            tie_tolerance=tie_tolerance,
        ),
        volunteered_control_coverage=volunteer_count,
        volunteered_control_status=volunteer_status,
        positive_control_note=(
            "The ordering is empirical under the fitted response model. "
            "Repeated balanced-choice and direct-correction controls require "
            "longitudinal records and are not inferred from one-step rows."
        ),
    )


_CONFIRMATORY_METRICS = {
    "acue",
    "fitted_aware_kl",
    "exact_kl",
    "brier",
    "fitted_aware_brier",
    "excess_brier",
    "update_magnitude",
    "evidence_weight",
    "log_odds_update",
    "fitted_aware_log_odds_update",
    "fitted_evidence_strength",
}


def _metric_value(row: ExperimentARow, metric: str) -> float:
    if metric not in _CONFIRMATORY_METRICS:
        raise ValueError(
            f"metric must be selected from {sorted(_CONFIRMATORY_METRICS)}"
        )
    value = float(getattr(row, metric))
    if not math.isfinite(value):
        raise ValueError(
            f"metric {metric!r} is non-finite for {row.trial_id}/{row.updater_id}"
        )
    return value


def _paired_cell_key(
    row: ExperimentARow,
) -> tuple[str, str, int, int, str]:
    return (
        row.user_id,
        row.domain_id,
        row.target_attribute,
        row.anchor_direction,
        row.prior_stratum,
    )


def experiment_a_mechanism_contrasts(
    rows: Sequence[ExperimentARow],
    *,
    first_mechanism: str,
    second_mechanism: str,
    metric: str = "acue",
    response_mode: str = "controlled_anchor",
    replicates: int = 2000,
    seed: int = 1729,
) -> tuple[PairedContrast, ...]:
    """Return within-matched-set mechanism contrasts for every updater."""

    if first_mechanism == second_mechanism:
        raise ValueError("mechanism contrast requires distinct mechanisms")
    selected = tuple(row for row in rows if row.response_mode == response_mode)
    cells = {
        (_paired_cell_key(row), row.updater_id, row.mechanism): row
        for row in selected
    }
    results = []
    for updater_id in sorted({row.updater_id for row in selected}):
        pair_keys = sorted(
            {
                key
                for key, candidate_updater, mechanism in cells
                if candidate_updater == updater_id
                and mechanism == first_mechanism
                and (key, updater_id, second_mechanism) in cells
            }
        )
        if not pair_keys:
            continue
        first = [
            _metric_value(
                cells[(key, updater_id, first_mechanism)],
                metric,
            )
            for key in pair_keys
        ]
        second = [
            _metric_value(
                cells[(key, updater_id, second_mechanism)],
                metric,
            )
            for key in pair_keys
        ]
        results.append(
            paired_cluster_contrast(
                first,
                second,
                [key[0] for key in pair_keys],
                contrast_id=(
                    f"experiment-a:{metric}:{updater_id}:"
                    f"{first_mechanism}-vs-{second_mechanism}"
                ),
                first_label=f"{updater_id}/{first_mechanism}",
                second_label=f"{updater_id}/{second_mechanism}",
                replicates=replicates,
                seed=seed,
            )
        )
    return tuple(results)


def experiment_a_updater_mechanism_interaction(
    rows: Sequence[ExperimentARow],
    *,
    first_updater: str,
    second_updater: str,
    treated_mechanism: str,
    reference_mechanism: str,
    metric: str = "acue",
    response_mode: str = "controlled_anchor",
    replicates: int = 2000,
    seed: int = 1729,
) -> PairedContrast:
    """Estimate the proposal's paired updater-by-mechanism interaction."""

    if first_updater == second_updater:
        raise ValueError("interaction requires distinct updaters")
    if treated_mechanism == reference_mechanism:
        raise ValueError("interaction requires distinct mechanisms")
    selected = tuple(row for row in rows if row.response_mode == response_mode)
    cells = {
        (_paired_cell_key(row), row.updater_id, row.mechanism): row
        for row in selected
    }
    required = (
        (first_updater, treated_mechanism),
        (first_updater, reference_mechanism),
        (second_updater, treated_mechanism),
        (second_updater, reference_mechanism),
    )
    pair_keys = sorted(
        {
            key
            for key, _, _ in cells
            if all((key, updater, mechanism) in cells for updater, mechanism in required)
        }
    )
    if not pair_keys:
        raise ValueError("no complete matched cells for the requested interaction")

    def values(updater: str, mechanism: str) -> list[float]:
        return [
            _metric_value(cells[(key, updater, mechanism)], metric)
            for key in pair_keys
        ]

    return paired_cluster_interaction(
        values(first_updater, treated_mechanism),
        values(first_updater, reference_mechanism),
        values(second_updater, treated_mechanism),
        values(second_updater, reference_mechanism),
        [key[0] for key in pair_keys],
        contrast_id=(
            f"experiment-a:{metric}:{first_updater}-vs-{second_updater}:"
            f"{treated_mechanism}-vs-{reference_mechanism}"
        ),
        first_label=first_updater,
        second_label=second_updater,
        treated_label=treated_mechanism,
        reference_label=reference_mechanism,
        replicates=replicates,
        seed=seed,
    )


def fit_experiment_a_marginal_ols(
    rows: Sequence[ExperimentARow],
    *,
    outcome: str = "acue",
    response_mode: str = "naturally_sampled",
    updater_reference: str | None = None,
    mechanism_reference: str = "balanced",
) -> ClusterRobustOLSResult:
    """Fit an updater-by-mechanism marginal model with user-clustered CR1 SEs.

    This dependency-free model covers the fixed updater, mechanism, interaction,
    domain, and varying prior-entropy terms. It does not fit the proposal's user
    random slopes or scenario random intercept, so it is a robustness analysis
    rather than the preregistered mixed-effects model.
    """

    selected = tuple(row for row in rows if row.response_mode == response_mode)
    if not selected:
        raise ValueError(f"no Experiment A rows for response mode {response_mode!r}")
    updaters = sorted({row.updater_id for row in selected})
    mechanisms = sorted({row.mechanism for row in selected})
    domains = sorted({row.domain_id for row in selected})
    updater_base = (
        (
            "fitted_action_aware"
            if "fitted_action_aware" in updaters
            else updaters[0]
        )
        if updater_reference is None
        else updater_reference
    )
    if updater_base not in updaters:
        raise ValueError(f"unknown updater reference {updater_base!r}")
    if mechanism_reference not in mechanisms:
        raise ValueError(f"unknown mechanism reference {mechanism_reference!r}")
    updater_levels = [value for value in updaters if value != updater_base]
    mechanism_levels = [
        value for value in mechanisms if value != mechanism_reference
    ]
    domain_base = domains[0]
    domain_levels = [value for value in domains if value != domain_base]
    prior_strengths = [row.prior_strength for row in selected]
    include_prior_strength = (
        max(prior_strengths) - min(prior_strengths) > 1e-12
    )
    names = ["intercept"]
    names.extend(f"updater[{value}]" for value in updater_levels)
    names.extend(f"mechanism[{value}]" for value in mechanism_levels)
    names.extend(
        f"updater[{updater}]:mechanism[{mechanism}]"
        for updater in updater_levels
        for mechanism in mechanism_levels
    )
    names.extend(f"domain[{value}]" for value in domain_levels)
    if include_prior_strength:
        names.append("prior_strength")
    design = []
    for row in selected:
        design_row = [1.0]
        design_row.extend(
            1.0 if row.updater_id == value else 0.0
            for value in updater_levels
        )
        design_row.extend(
            1.0 if row.mechanism == value else 0.0
            for value in mechanism_levels
        )
        design_row.extend(
            (
                1.0
                if row.updater_id == updater
                and row.mechanism == mechanism
                else 0.0
            )
            for updater in updater_levels
            for mechanism in mechanism_levels
        )
        design_row.extend(
            1.0 if row.domain_id == value else 0.0
            for value in domain_levels
        )
        if include_prior_strength:
            design_row.append(row.prior_strength)
        design.append(design_row)
    return fit_cluster_robust_ols(
        design,
        [_metric_value(row, outcome) for row in selected],
        [row.user_id for row in selected],
        names,
        model_label=(
            f"Experiment A {outcome}: updater * mechanism + domain"
            + (" + prior_strength" if include_prior_strength else "")
            + "; marginal OLS working-independence model"
        ),
    )


def compare_experiment_a_raw_calibrated(
    raw_rows: Sequence[ExperimentARow],
    calibrated_rows: Sequence[ExperimentARow],
    *,
    true_theta_by_user: Mapping[str, tuple[int, int, int]],
    bin_count: int = 10,
) -> RawCalibratedComparison:
    """Compare two otherwise identical A runs at marginal-outcome level."""

    def keyed(
        candidates: Sequence[ExperimentARow],
    ) -> dict[tuple[str, str], ExperimentARow]:
        result = {}
        for row in candidates:
            key = (row.trial_id, row.updater_id)
            if key in result:
                raise ValueError(f"duplicate Experiment A row key {key}")
            result[key] = row
        return result

    raw = keyed(raw_rows)
    calibrated = keyed(calibrated_rows)
    if set(raw) != set(calibrated):
        raise ValueError("raw and calibrated runs must contain identical row keys")
    forecasts = []
    for key in sorted(raw):
        raw_row = raw[key]
        calibrated_row = calibrated[key]
        if (
            raw_row.user_id != calibrated_row.user_id
            or raw_row.context != calibrated_row.context
            or raw_row.observation != calibrated_row.observation
        ):
            raise ValueError(
                "raw/calibrated comparison requires identical users and histories"
            )
        try:
            true_theta = true_theta_by_user[raw_row.user_id]
        except KeyError as exc:
            raise ValueError(
                f"missing true theta for user {raw_row.user_id!r}"
            ) from exc
        for attribute, true_value in enumerate(true_theta):
            try:
                true_index = THETA_VALUES.index(true_value)
            except ValueError as exc:
                raise ValueError(
                    f"invalid theta value {true_value!r} for {raw_row.user_id}"
                ) from exc
            forecasts.append(
                MarginalForecast(
                    record_id=(
                        f"{raw_row.trial_id}:{raw_row.updater_id}:"
                        f"attribute-{attribute}"
                    ),
                    cluster_id=raw_row.user_id,
                    raw_probabilities=raw_row.posterior.marginal(attribute),
                    calibrated_probabilities=(
                        calibrated_row.posterior.marginal(attribute)
                    ),
                    true_index=true_index,
                )
            )
    return compare_raw_and_calibrated_forecasts(
        forecasts,
        bin_count=bin_count,
    )


@dataclass(frozen=True, slots=True)
class ExperimentAConfirmatoryResult:
    """Serializable Experiment A analysis components ready for artifact writing."""

    oracle_update_slopes: tuple[OracleUpdateSlope, ...]
    evidence_strength: EvidenceStrengthAnalysis
    mechanism_contrasts: tuple[PairedContrast, ...] = ()
    updater_mechanism_interactions: tuple[PairedContrast, ...] = ()
    marginal_regression: ClusterRobustOLSResult | None = None
    raw_calibrated_comparison: RawCalibratedComparison | None = None
    bootstrap_replicates: int = 0
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis": "experiment_a_confirmatory",
            "independent_unit": "complete latent user",
            "bootstrap_replicates": self.bootstrap_replicates,
            "oracle_update_slopes": [
                result.to_dict() for result in self.oracle_update_slopes
            ],
            "evidence_strength": self.evidence_strength.to_dict(),
            "mechanism_contrasts": [
                contrast.to_dict() for contrast in self.mechanism_contrasts
            ],
            "updater_mechanism_interactions": [
                contrast.to_dict()
                for contrast in self.updater_mechanism_interactions
            ],
            "marginal_regression": (
                None
                if self.marginal_regression is None
                else self.marginal_regression.to_dict()
            ),
            "raw_calibrated_comparison": (
                None
                if self.raw_calibrated_comparison is None
                else self.raw_calibrated_comparison.to_dict()
            ),
            "notes": list(self.notes),
        }


# Runner-facing name matching the paper.
run_experiment_a = run_provenance_audit
