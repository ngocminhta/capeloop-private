"""Simulation-based pilot power and multiplicity helpers.

Experiment B uses a deliberately narrow planning estimand: a complete-user
three-way difference in differences in differences (DDD) for terminal error.
The simulator resamples those complete-user pilot contrasts.  It is a bounded,
dependency-free planning approximation and is not the confirmatory
mixed-effects analysis declared in the proposal.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from statistics import NormalDist, mean, stdev
from typing import Any, Mapping, Protocol, Sequence

from .rng import weighted_index
from .statistics import simulate_paired_cluster_power


EXPERIMENT_B_POWER_SAMPLE_SIZES = (16, 32, 64, 128)
EXPERIMENT_B_POWER_ALPHA = 0.05
EXPERIMENT_B_TARGET_POWER = 0.80
EXPERIMENT_B_POWER_MINIMUM_SIMULATIONS = 200
EXPERIMENT_B_POWER_MAXIMUM_SIMULATIONS = 10_000
EXPERIMENT_B_POWER_MAXIMUM_SAMPLE_SIZE = max(
    EXPERIMENT_B_POWER_SAMPLE_SIZES
)
EXPERIMENT_B_POWER_MC_CONFIDENCE_LEVEL = 0.95


@dataclass(frozen=True, slots=True)
class PowerEstimate:
    sample_size: int
    simulations: int
    alpha: float
    estimated_power: float
    pilot_effect: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "sample_size": self.sample_size,
            "simulations": self.simulations,
            "alpha": self.alpha,
            "estimated_power": self.estimated_power,
            "pilot_effect": self.pilot_effect,
        }


def paired_pilot_power(
    paired_differences: Sequence[float],
    sample_sizes: Sequence[int],
    *,
    simulations: int = 2000,
    alpha: float = 0.05,
    seed: int = 1729,
) -> tuple[PowerEstimate, ...]:
    """Bootstrap paired trajectory effects and use a two-sided normal test."""

    pilot = tuple(float(value) for value in paired_differences)
    if len(pilot) < 2:
        raise ValueError("pilot power requires at least two paired trajectories")
    if simulations <= 0:
        raise ValueError("simulations must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    if any(size < 2 for size in sample_sizes):
        raise ValueError("sample sizes must be at least two")
    threshold = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    weights = [1.0] * len(pilot)
    estimates = []
    for sample_size in sample_sizes:
        rejections = 0
        for simulation in range(simulations):
            sample = [
                pilot[
                    weighted_index(
                        weights,
                        seed,
                        "power",
                        sample_size,
                        simulation,
                        draw,
                    )
                ]
                for draw in range(sample_size)
            ]
            standard_error = stdev(sample) / math.sqrt(sample_size)
            if standard_error == 0:
                reject = mean(sample) != 0
            else:
                reject = abs(mean(sample) / standard_error) >= threshold
            rejections += int(reject)
        estimates.append(
            PowerEstimate(
                sample_size=sample_size,
                simulations=simulations,
                alpha=alpha,
                estimated_power=rejections / simulations,
                pilot_effect=mean(pilot),
            )
        )
    return tuple(estimates)


class ExperimentBPowerTrajectory(Protocol):
    """Fields needed from one retained closed-loop trajectory."""

    trajectory_id: str
    crn_key: str
    user_id: str
    domain_id: str
    updater_id: str
    policy_id: str
    initial_profile_condition: str

    @property
    def terminal_error(self) -> float: ...


@dataclass(frozen=True, slots=True)
class ExperimentBUserInteraction:
    """One complete user's terminal-error three-way interaction."""

    user_id: str
    stratum_count: int
    interaction: float

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "user_id": self.user_id,
            "stratum_count": self.stratum_count,
            "interaction": self.interaction,
        }


@dataclass(frozen=True, slots=True)
class ExperimentBPilotInteractions:
    """Complete-user pilot inputs for the frozen Experiment B power estimand."""

    target_updater_id: str
    reference_updater_id: str
    treated_policy_id: str
    reference_policy_id: str
    focal_initial_profile: str
    reference_initial_profile: str
    eligible_users: tuple[ExperimentBUserInteraction, ...]
    excluded_users: tuple[tuple[str, str], ...]
    pilot_input_sha256: str
    contributing_trajectory_count: int

    @property
    def differences(self) -> tuple[float, ...]:
        return tuple(row.interaction for row in self.eligible_users)

    def to_dict(self) -> dict[str, Any]:
        return {
            "eligible_user_count": len(self.eligible_users),
            "excluded_user_count": len(self.excluded_users),
            "contributing_trajectory_count": (
                self.contributing_trajectory_count
            ),
            "pilot_input_sha256": self.pilot_input_sha256,
            "user_level_interactions": [
                row.to_dict() for row in self.eligible_users
            ],
            "excluded_users": [
                {"user_id": user_id, "reason": reason}
                for user_id, reason in self.excluded_users
            ],
        }


def bounded_experiment_b_simulations(requested: int) -> int:
    """Apply the frozen lower and upper bounds to B power simulations."""

    if isinstance(requested, bool) or not isinstance(requested, int):
        raise ValueError("requested simulations must be an integer")
    if requested < 0:
        raise ValueError("requested simulations must be non-negative")
    return min(
        EXPERIMENT_B_POWER_MAXIMUM_SIMULATIONS,
        max(EXPERIMENT_B_POWER_MINIMUM_SIMULATIONS, requested),
    )


def _experiment_b_replicate_id(crn_key: str) -> str | None:
    """Read the runner's explicit replicate suffix without parsing user IDs."""

    marker = ":replicate-"
    _, found, suffix = crn_key.rpartition(marker)
    if not found or not suffix.isdigit():
        return None
    return f"replicate-{int(suffix)}"


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def experiment_b_pilot_interactions(
    trajectories: Sequence[ExperimentBPowerTrajectory],
    *,
    target_updater_id: str,
    reference_updater_id: str = "fitted_action_aware",
    treated_policy_id: str = "soft_profile_conditioned",
    reference_policy_id: str = "balanced",
    focal_initial_profile: str = "incorrect",
    reference_initial_profile: str = "correct",
) -> ExperimentBPilotInteractions:
    """Reduce a crossed B pilot to complete-user terminal-error DDD values.

    For each domain×replicate stratum, the scalar contrast is::

        (
          (target, treated policy) - (target, reference policy)
          - (reference updater, treated policy)
          + (reference updater, reference policy)
        )[focal profile]
        -
        (
          (target, treated policy) - (target, reference policy)
          - (reference updater, treated policy)
          + (reference updater, reference policy)
        )[reference profile]

    Stratum-level contrasts are averaged within user.  A user is retained only
    when every required cell exists for every one of that user's candidate
    strata.  This makes the complete latent user, rather than a turn or repeated
    trajectory, the independent pilot unit.
    """

    identifiers = (
        target_updater_id,
        reference_updater_id,
        treated_policy_id,
        reference_policy_id,
        focal_initial_profile,
        reference_initial_profile,
    )
    if any(not isinstance(item, str) or not item for item in identifiers):
        raise ValueError("Experiment B power factor identifiers must be non-empty")
    if target_updater_id == reference_updater_id:
        raise ValueError("target and reference updaters must differ")
    if treated_policy_id == reference_policy_id:
        raise ValueError("treated and reference policies must differ")
    if focal_initial_profile == reference_initial_profile:
        raise ValueError("focal and reference initial profiles must differ")

    required_updaters = (target_updater_id, reference_updater_id)
    required_policies = (treated_policy_id, reference_policy_id)
    required_profiles = (
        focal_initial_profile,
        reference_initial_profile,
    )
    required_factors = tuple(
        (updater_id, policy_id, initial_profile)
        for updater_id in required_updaters
        for policy_id in required_policies
        for initial_profile in required_profiles
    )
    relevant = tuple(
        trajectory
        for trajectory in trajectories
        if trajectory.updater_id in required_updaters
        and trajectory.policy_id in required_policies
        and trajectory.initial_profile_condition in required_profiles
    )
    all_user_ids = sorted(
        {
            trajectory.user_id
            for trajectory in trajectories
        }
    )
    candidate_strata_by_user: dict[str, set[tuple[str, str]]] = {}
    for trajectory in trajectories:
        replicate_id = _experiment_b_replicate_id(trajectory.crn_key)
        if replicate_id is not None:
            candidate_strata_by_user.setdefault(
                trajectory.user_id,
                set(),
            ).add((trajectory.domain_id, replicate_id))
    cells: dict[
        tuple[str, str, str, str, str, str],
        ExperimentBPowerTrajectory,
    ] = {}
    malformed_by_user: dict[str, list[str]] = {}
    source_rows = []
    for trajectory in sorted(
        relevant,
        key=lambda row: (
            row.user_id,
            row.domain_id,
            row.crn_key,
            row.updater_id,
            row.policy_id,
            row.initial_profile_condition,
            row.trajectory_id,
        ),
    ):
        replicate_id = _experiment_b_replicate_id(trajectory.crn_key)
        if replicate_id is None:
            malformed_by_user.setdefault(trajectory.user_id, []).append(
                trajectory.trajectory_id
            )
            continue
        terminal_error = float(trajectory.terminal_error)
        if not math.isfinite(terminal_error):
            raise ValueError("Experiment B power terminal errors must be finite")
        key = (
            trajectory.user_id,
            trajectory.domain_id,
            replicate_id,
            trajectory.updater_id,
            trajectory.policy_id,
            trajectory.initial_profile_condition,
        )
        if key in cells:
            raise ValueError(
                "duplicate Experiment B power cell for "
                + "|".join(key)
            )
        cells[key] = trajectory
        source_rows.append(
            {
                "trajectory_id": trajectory.trajectory_id,
                "user_id": trajectory.user_id,
                "domain_id": trajectory.domain_id,
                "replicate_id": replicate_id,
                "updater_id": trajectory.updater_id,
                "policy_id": trajectory.policy_id,
                "initial_profile_condition": (
                    trajectory.initial_profile_condition
                ),
                "terminal_error": terminal_error,
            }
        )

    eligible: list[ExperimentBUserInteraction] = []
    excluded: list[tuple[str, str]] = []
    for user_id in all_user_ids:
        if user_id in malformed_by_user:
            excluded.append(
                (
                    user_id,
                    "one or more relevant trajectories have a non-canonical "
                    "CRN replicate key",
                )
            )
            continue
        strata = sorted(candidate_strata_by_user.get(user_id, set()))
        if not strata:
            excluded.append(
                (user_id, "no trajectories for the frozen power contrast")
            )
            continue
        missing = [
            (
                domain_id,
                replicate_id,
                updater_id,
                policy_id,
                initial_profile,
            )
            for domain_id, replicate_id in strata
            for updater_id, policy_id, initial_profile in required_factors
            if (
                user_id,
                domain_id,
                replicate_id,
                updater_id,
                policy_id,
                initial_profile,
            )
            not in cells
        ]
        if missing:
            excluded.append(
                (
                    user_id,
                    f"incomplete crossed block ({len(missing)} required "
                    "cell(s) missing)",
                )
            )
            continue

        stratum_interactions = []
        for domain_id, replicate_id in strata:
            prefix = (user_id, domain_id, replicate_id)

            def error(
                updater_id: str,
                policy_id: str,
                initial_profile: str,
            ) -> float:
                return float(
                    cells[
                        (
                            *prefix,
                            updater_id,
                            policy_id,
                            initial_profile,
                        )
                    ].terminal_error
                )

            def updater_policy_interaction(initial_profile: str) -> float:
                return (
                    error(
                        target_updater_id,
                        treated_policy_id,
                        initial_profile,
                    )
                    - error(
                        target_updater_id,
                        reference_policy_id,
                        initial_profile,
                    )
                    - error(
                        reference_updater_id,
                        treated_policy_id,
                        initial_profile,
                    )
                    + error(
                        reference_updater_id,
                        reference_policy_id,
                        initial_profile,
                    )
                )

            stratum_interactions.append(
                updater_policy_interaction(focal_initial_profile)
                - updater_policy_interaction(reference_initial_profile)
            )
        eligible.append(
            ExperimentBUserInteraction(
                user_id=user_id,
                stratum_count=len(strata),
                interaction=mean(stratum_interactions),
            )
        )

    factor_spec = {
        "target_updater_id": target_updater_id,
        "reference_updater_id": reference_updater_id,
        "treated_policy_id": treated_policy_id,
        "reference_policy_id": reference_policy_id,
        "focal_initial_profile": focal_initial_profile,
        "reference_initial_profile": reference_initial_profile,
    }
    return ExperimentBPilotInteractions(
        target_updater_id=target_updater_id,
        reference_updater_id=reference_updater_id,
        treated_policy_id=treated_policy_id,
        reference_policy_id=reference_policy_id,
        focal_initial_profile=focal_initial_profile,
        reference_initial_profile=reference_initial_profile,
        eligible_users=tuple(eligible),
        excluded_users=tuple(excluded),
        pilot_input_sha256=_canonical_sha256(
            {
                "factor_spec": factor_spec,
                "source_trajectories": source_rows,
            }
        ),
        contributing_trajectory_count=len(source_rows),
    )


def _wilson_interval(
    successes: int,
    trials: int,
    *,
    confidence_level: float,
) -> tuple[float, float]:
    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("Wilson interval needs valid binomial counts")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0, 1)")
    z = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / trials
            + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def experiment_b_pilot_power(
    trajectories: Sequence[ExperimentBPowerTrajectory],
    *,
    target_updater_id: str | None,
    reference_updater_id: str = "fitted_action_aware",
    sample_sizes: Sequence[int] = EXPERIMENT_B_POWER_SAMPLE_SIZES,
    simulations: int = EXPERIMENT_B_POWER_MINIMUM_SIMULATIONS,
    alpha: float = EXPERIMENT_B_POWER_ALPHA,
    target_power: float = EXPERIMENT_B_TARGET_POWER,
    seed: int = 1729,
) -> dict[str, Any]:
    """Build the machine-readable Experiment B pilot-design artifact."""

    sizes = tuple(sample_sizes)
    if (
        not sizes
        or any(
            isinstance(size, bool) or not isinstance(size, int)
            for size in sizes
        )
        or len(sizes) != len(set(sizes))
        or tuple(sorted(sizes)) != sizes
        or any(size < 2 for size in sizes)
        or any(size > EXPERIMENT_B_POWER_MAXIMUM_SAMPLE_SIZE for size in sizes)
    ):
        raise ValueError(
            "Experiment B power sample sizes must be unique, increasing, "
            f"and lie in [2, {EXPERIMENT_B_POWER_MAXIMUM_SAMPLE_SIZE}]"
        )
    if (
        isinstance(simulations, bool)
        or not isinstance(simulations, int)
        or not 1 <= simulations <= EXPERIMENT_B_POWER_MAXIMUM_SIMULATIONS
    ):
        raise ValueError(
            "Experiment B power simulations must be an integer in "
            f"[1, {EXPERIMENT_B_POWER_MAXIMUM_SIMULATIONS}]"
        )
    if not 0.0 < alpha < 1.0:
        raise ValueError("Experiment B power alpha must lie in (0, 1)")
    if not 0.0 < target_power < 1.0:
        raise ValueError("Experiment B target power must lie in (0, 1)")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Experiment B power seed must be non-negative")

    estimand_formula = (
        "[(target_soft - target_balanced) - "
        "(fitted_aware_soft - fitted_aware_balanced)]_incorrect - "
        "[(target_soft - target_balanced) - "
        "(fitted_aware_soft - fitted_aware_balanced)]_correct"
    )
    base: dict[str, Any] = {
        "schema_version": 1,
        "analysis_id": "experiment-b-pilot-power-v1",
        "artifact_role": "pilot_design_evidence",
        "scientific_claim_status": "not_claimed",
        "outcome": "terminal marginal Brier error",
        "independent_unit": "complete latent user",
        "estimand": (
            "Experiment B Updater × Policy × InitialProfile terminal-error "
            "interaction, focused on incorrect initialization"
        ),
        "estimand_formula": estimand_formula,
        "factor_contrast": {
            "target_updater_id": target_updater_id,
            "reference_updater_id": reference_updater_id,
            "treated_policy_id": "soft_profile_conditioned",
            "reference_policy_id": "balanced",
            "focal_initial_profile": "incorrect",
            "reference_initial_profile": "correct",
        },
        "candidate_user_counts": list(sizes),
        "simulation": {
            "method": (
                "complete-user nonparametric simulation from centered "
                "empirical pilot DDD residuals with a two-sided "
                "normal-reference one-sample test"
            ),
            "seed": seed,
            "alpha": alpha,
            "simulations_per_candidate": simulations,
            "maximum_simulations_per_candidate": (
                EXPERIMENT_B_POWER_MAXIMUM_SIMULATIONS
            ),
            "maximum_candidate_user_count": (
                EXPERIMENT_B_POWER_MAXIMUM_SAMPLE_SIZE
            ),
            "bounded": True,
            "monte_carlo_confidence_level": (
                EXPERIMENT_B_POWER_MC_CONFIDENCE_LEVEL
            ),
            "monte_carlo_interval_method": "Wilson binomial interval",
            "assumptions": [
                (
                    "Complete latent users are the independent resampling "
                    "units; domains and repeated trajectories are averaged "
                    "within user."
                ),
                (
                    "All eight target/reference updater × policy × initial-"
                    "profile cells must be present in every retained stratum."
                ),
                (
                    "The target effect is the configured pilot mean; centered "
                    "complete-user residuals represent future heterogeneity."
                ),
                (
                    "The normal-reference test is a planning approximation, "
                    "not the confirmatory mixed-effects model."
                ),
            ],
        },
        "decision_rule": {
            "target_power": target_power,
            "rule": (
                "select the smallest frozen candidate whose lower 95% Wilson "
                "Monte Carlo bound is at least the target power"
            ),
            "automatic_sample_size_commitment": False,
        },
        "limitations": [
            (
                "This artifact supports pilot design only and is not empirical "
                "evidence for a paper claim."
            ),
            (
                "Pilot-effect reuse can be optimistic; final sample-size "
                "selection requires investigator review and preregistration."
            ),
            (
                "The power approximation does not replace the optional "
                "confirmatory mixed-effects pipeline."
            ),
        ],
    }
    if target_updater_id is None:
        return {
            **base,
            "status": "not_estimable",
            "reason": (
                "the configured pilot has neither llm_full_context nor "
                "full_context_blind as the frozen target updater"
            ),
            "pilot": {
                "eligible_user_count": 0,
                "excluded_user_count": 0,
                "contributing_trajectory_count": 0,
                "pilot_input_sha256": None,
                "user_level_interactions": [],
                "excluded_users": [],
            },
            "points": [],
            "decision": {
                "status": "not_estimable",
                "selected_user_count": None,
            },
        }

    pilot = experiment_b_pilot_interactions(
        trajectories,
        target_updater_id=target_updater_id,
        reference_updater_id=reference_updater_id,
    )
    base["pilot"] = pilot.to_dict()
    if len(pilot.eligible_users) < 2:
        return {
            **base,
            "status": "not_estimable",
            "reason": (
                "power simulation requires at least two complete independent "
                "user-level pilot three-way interactions"
            ),
            "points": [],
            "decision": {
                "status": "not_estimable",
                "selected_user_count": None,
            },
        }

    calculated = simulate_paired_cluster_power(
        pilot.differences,
        sizes,
        estimand=str(base["estimand"]),
        simulations=simulations,
        alpha=alpha,
        seed=seed,
    )
    points = []
    for point in calculated.points:
        rejections = point.rejections
        lower, upper = _wilson_interval(
            rejections,
            point.simulations,
            confidence_level=EXPERIMENT_B_POWER_MC_CONFIDENCE_LEVEL,
        )
        points.append(
            {
                **point.to_dict(),
                "rejections": rejections,
                "monte_carlo_standard_error": math.sqrt(
                    point.estimated_power
                    * (1.0 - point.estimated_power)
                    / point.simulations
                ),
                "monte_carlo_lower": lower,
                "monte_carlo_upper": upper,
                "meets_decision_threshold": lower >= target_power,
            }
        )
    selected = next(
        (
            int(point["sample_size"])
            for point in points
            if point["meets_decision_threshold"]
        ),
        None,
    )
    return {
        **base,
        "status": "estimated_from_configured_pilot",
        "target_effect": mean(pilot.differences),
        "target_effect_source": (
            "mean complete-user interaction in the configured pilot"
        ),
        "points": points,
        "decision": {
            "status": (
                "candidate_meets_threshold"
                if selected is not None
                else "no_frozen_candidate_meets_threshold"
            ),
            "selected_user_count": selected,
            "requires_investigator_review": True,
        },
    }


def format_experiment_b_power_summary(payload: Mapping[str, Any]) -> str:
    """Render the B power artifact as a concise Markdown planning summary."""

    status = str(payload["status"])
    factor = payload["factor_contrast"]
    simulation = payload["simulation"]
    decision_rule = payload["decision_rule"]
    pilot = payload["pilot"]
    lines = [
        "# Experiment B pilot-power summary",
        "",
        f"- Status: `{status}`",
        "- Artifact role: pilot-design evidence",
        "- Scientific claim status: `not_claimed`",
        f"- Outcome: {payload['outcome']}",
        f"- Independent unit: {payload['independent_unit']}",
        (
            "- Frozen contrast: "
            f"`{factor['target_updater_id']}` versus "
            f"`{factor['reference_updater_id']}` × "
            f"`{factor['treated_policy_id']}` versus "
            f"`{factor['reference_policy_id']}` × "
            f"`{factor['focal_initial_profile']}` versus "
            f"`{factor['reference_initial_profile']}`"
        ),
        f"- Estimand formula: `{payload['estimand_formula']}`",
        (
            "- Eligible complete users: "
            f"{pilot['eligible_user_count']} "
            f"(excluded: {pilot['excluded_user_count']})"
        ),
        (
            "- Pilot input SHA-256: "
            f"`{pilot['pilot_input_sha256']}`"
        ),
        (
            "- Simulation: "
            f"{simulation['simulations_per_candidate']} replicates per "
            f"candidate, seed {simulation['seed']}, "
            f"two-sided alpha {simulation['alpha']}"
        ),
        (
            "- Decision threshold: lower 95% Wilson Monte Carlo bound "
            f"≥ {decision_rule['target_power']:.2f}"
        ),
        "",
    ]
    reason = payload.get("reason")
    if reason:
        lines.extend([f"Reason: {reason}", ""])
    points = payload.get("points", [])
    if points:
        lines.extend(
            [
                "| Complete users | Estimated power | MC SE | 95% MC interval | Threshold met |",
                "| ---: | ---: | ---: | ---: | :---: |",
            ]
        )
        for point in points:
            lines.append(
                "| "
                f"{point['sample_size']} | "
                f"{point['estimated_power']:.4f} | "
                f"{point['monte_carlo_standard_error']:.4f} | "
                f"[{point['monte_carlo_lower']:.4f}, "
                f"{point['monte_carlo_upper']:.4f}] | "
                f"{'yes' if point['meets_decision_threshold'] else 'no'} |"
            )
        lines.append("")
    decision = payload["decision"]
    selected = decision["selected_user_count"]
    lines.extend(
        [
            (
                "Planning decision: no frozen candidate was selected."
                if selected is None
                else (
                    "Planning decision: the first candidate satisfying the "
                    f"computational rule is {selected} complete users."
                )
            ),
            "",
            (
                "This is a bounded pilot-design calculation, not empirical "
                "evidence for the paper. Any final sample-size commitment "
                "requires investigator review and preregistration; the "
                "confirmatory mixed-effects model remains separate."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def benjamini_hochberg(
    p_values: Mapping[str, float],
    *,
    alpha: float = 0.05,
) -> dict[str, bool]:
    """Return false-discovery-rate rejection decisions for secondary tests."""

    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    if any(not 0 <= value <= 1 for value in p_values.values()):
        raise ValueError("p-values must lie in [0, 1]")
    ordered = sorted(p_values, key=lambda key: (p_values[key], key))
    largest = -1
    count = len(ordered)
    for index, key in enumerate(ordered, start=1):
        if p_values[key] <= alpha * index / max(count, 1):
            largest = index
    return {
        key: index <= largest
        for index, key in enumerate(ordered, start=1)
    }
