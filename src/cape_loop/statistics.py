"""Deterministic bootstrap, confirmatory-analysis, and ranking helpers.

The cluster-aware regression in this module is a marginal ordinary least
squares model with a CR1 cluster-robust covariance estimate.  It is useful as a
transparent dependency-free robustness analysis, but is deliberately not
called a mixed-effects model or a GLMM.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from statistics import NormalDist, mean, stdev
from typing import Any, Mapping, Sequence

from .rng import weighted_index


def percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot take a percentile of no values")
    if not 0 <= probability <= 1:
        raise ValueError("probability must lie in [0, 1]")
    ordered = sorted(values)
    location = probability * (len(ordered) - 1)
    lower = math.floor(location)
    upper = math.ceil(location)
    if lower == upper:
        return ordered[lower]
    fraction = location - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def paired_bootstrap_mean_difference(
    first: Sequence[float],
    second: Sequence[float],
    *,
    replicates: int = 1000,
    seed: int = 1729,
) -> tuple[float, float, float]:
    if len(first) != len(second) or not first:
        raise ValueError("paired samples must have equal non-zero length")
    if replicates <= 0:
        raise ValueError("paired bootstrap requires positive replicates")
    observed = mean(a - b for a, b in zip(first, second))
    draws = []
    weights = [1.0] * len(first)
    for replicate in range(replicates):
        indexes = [
            weighted_index(weights, seed, "paired-bootstrap", replicate, draw)
            for draw in range(len(first))
        ]
        draws.append(mean(first[index] - second[index] for index in indexes))
    return observed, percentile(draws, 0.025), percentile(draws, 0.975)


@dataclass(frozen=True, slots=True)
class IntervalEstimate:
    """A point estimate and deterministic resampling interval."""

    estimate: float
    lower: float
    upper: float
    confidence_level: float
    method: str
    cluster_count: int
    replicate_count: int

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "estimate": self.estimate,
            "lower": self.lower,
            "upper": self.upper,
            "confidence_level": self.confidence_level,
            "method": self.method,
            "cluster_count": self.cluster_count,
            "replicate_count": self.replicate_count,
        }


def clustered_bootstrap_mean(
    values: Sequence[float],
    cluster_ids: Sequence[str],
    *,
    replicates: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 1729,
    namespace: str = "clustered-bootstrap",
) -> IntervalEstimate:
    """Bootstrap an equally weighted mean over independent clusters.

    Rows within a participant or trajectory may be dependent.  This function
    first reduces each cluster to its mean and then resamples complete clusters.
    Consequently a participant with more retained turns does not receive more
    inferential weight merely because it contributed more rows.
    """

    if len(values) != len(cluster_ids) or not values:
        raise ValueError("values and cluster_ids must have equal non-zero length")
    if replicates <= 0:
        raise ValueError("clustered bootstrap requires positive replicates")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must lie in (0, 1)")
    numeric = tuple(float(value) for value in values)
    if any(not math.isfinite(value) for value in numeric):
        raise ValueError("bootstrap values must be finite")
    grouped: dict[str, list[float]] = {}
    for cluster_id, value in zip(cluster_ids, numeric):
        if not cluster_id:
            raise ValueError("cluster IDs must be non-empty")
        grouped.setdefault(cluster_id, []).append(value)
    cluster_means = tuple(
        mean(grouped[cluster_id]) for cluster_id in sorted(grouped)
    )
    observed = mean(cluster_means)
    weights = [1.0] * len(cluster_means)
    draws = []
    for replicate in range(replicates):
        indexes = [
            weighted_index(
                weights,
                seed,
                namespace,
                replicate,
                draw,
            )
            for draw in range(len(cluster_means))
        ]
        draws.append(mean(cluster_means[index] for index in indexes))
    tail = (1.0 - confidence_level) / 2.0
    return IntervalEstimate(
        estimate=observed,
        lower=percentile(draws, tail),
        upper=percentile(draws, 1.0 - tail),
        confidence_level=confidence_level,
        method="percentile bootstrap over equally weighted complete clusters",
        cluster_count=len(cluster_means),
        replicate_count=replicates,
    )


@dataclass(frozen=True, slots=True)
class PairedContrast:
    """A paired contrast or difference-in-differences estimate."""

    contrast_id: str
    expression: str
    pair_count: int
    interval: IntervalEstimate

    @property
    def estimate(self) -> float:
        return self.interval.estimate

    def to_dict(self) -> dict[str, Any]:
        return {
            "contrast_id": self.contrast_id,
            "expression": self.expression,
            "pair_count": self.pair_count,
            "interval": self.interval.to_dict(),
        }


def paired_cluster_contrast(
    first: Sequence[float],
    second: Sequence[float],
    cluster_ids: Sequence[str],
    *,
    contrast_id: str,
    first_label: str,
    second_label: str,
    replicates: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 1729,
) -> PairedContrast:
    """Estimate ``first - second`` while resampling the pairing clusters."""

    if len(first) != len(second) or len(first) != len(cluster_ids) or not first:
        raise ValueError("paired contrast inputs must have equal non-zero length")
    differences = tuple(
        float(first_value) - float(second_value)
        for first_value, second_value in zip(first, second)
    )
    interval = clustered_bootstrap_mean(
        differences,
        cluster_ids,
        replicates=replicates,
        confidence_level=confidence_level,
        seed=seed,
        namespace=f"paired-contrast:{contrast_id}",
    )
    return PairedContrast(
        contrast_id=contrast_id,
        expression=f"{first_label} - {second_label}",
        pair_count=len(differences),
        interval=interval,
    )


def paired_cluster_interaction(
    first_treated: Sequence[float],
    first_reference: Sequence[float],
    second_treated: Sequence[float],
    second_reference: Sequence[float],
    cluster_ids: Sequence[str],
    *,
    contrast_id: str,
    first_label: str,
    second_label: str,
    treated_label: str,
    reference_label: str,
    replicates: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 1729,
) -> PairedContrast:
    """Estimate a paired difference-in-differences interaction."""

    lengths = {
        len(first_treated),
        len(first_reference),
        len(second_treated),
        len(second_reference),
        len(cluster_ids),
    }
    if len(lengths) != 1 or next(iter(lengths)) == 0:
        raise ValueError("interaction inputs must have equal non-zero length")
    first_difference = tuple(
        float(treated) - float(reference)
        for treated, reference in zip(first_treated, first_reference)
    )
    second_difference = tuple(
        float(treated) - float(reference)
        for treated, reference in zip(second_treated, second_reference)
    )
    return paired_cluster_contrast(
        first_difference,
        second_difference,
        cluster_ids,
        contrast_id=contrast_id,
        first_label=f"{first_label}[{treated_label} - {reference_label}]",
        second_label=f"{second_label}[{treated_label} - {reference_label}]",
        replicates=replicates,
        confidence_level=confidence_level,
        seed=seed,
    )


@dataclass(frozen=True, slots=True)
class MarginalForecast:
    """One multiclass forecast, optionally before and after calibration."""

    record_id: str
    cluster_id: str
    raw_probabilities: tuple[float, ...]
    calibrated_probabilities: tuple[float, ...]
    true_index: int

    def __post_init__(self) -> None:
        if not self.record_id or not self.cluster_id:
            raise ValueError("forecast record and cluster IDs must be non-empty")
        if (
            not self.raw_probabilities
            or len(self.raw_probabilities) != len(self.calibrated_probabilities)
        ):
            raise ValueError("raw and calibrated forecasts need equal non-zero size")
        for probabilities in (
            self.raw_probabilities,
            self.calibrated_probabilities,
        ):
            if any(
                not math.isfinite(value) or value < 0.0
                for value in probabilities
            ):
                raise ValueError("forecast probabilities must be finite and non-negative")
            if not math.isclose(
                math.fsum(probabilities),
                1.0,
                rel_tol=1e-9,
                abs_tol=1e-12,
            ):
                raise ValueError("forecast probabilities must sum to one")
        if not 0 <= self.true_index < len(self.raw_probabilities):
            raise ValueError("true_index is outside the forecast")


@dataclass(frozen=True, slots=True)
class ForecastScore:
    record_id: str
    cluster_id: str
    variant: str
    brier: float
    negative_log_likelihood: float
    true_class_probability: float
    top_class_correct: bool

    def to_dict(self) -> dict[str, float | str | bool]:
        return {
            "record_id": self.record_id,
            "cluster_id": self.cluster_id,
            "variant": self.variant,
            "brier": self.brier,
            "negative_log_likelihood": self.negative_log_likelihood,
            "true_class_probability": self.true_class_probability,
            "top_class_correct": self.top_class_correct,
        }


@dataclass(frozen=True, slots=True)
class ReliabilityBin:
    """A fixed-width, pooled one-vs-rest multiclass reliability bin."""

    variant: str
    bin_index: int
    lower_bound: float
    upper_bound: float
    prediction_count: int
    mean_predicted_probability: float | None
    observed_frequency: float | None
    calibration_gap: float | None

    def to_dict(self) -> dict[str, float | int | str | None]:
        return {
            "variant": self.variant,
            "bin_index": self.bin_index,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "prediction_count": self.prediction_count,
            "mean_predicted_probability": self.mean_predicted_probability,
            "observed_frequency": self.observed_frequency,
            "calibration_gap": self.calibration_gap,
        }


@dataclass(frozen=True, slots=True)
class RawCalibratedComparison:
    scores: tuple[ForecastScore, ...]
    reliability_bins: tuple[ReliabilityBin, ...]
    raw_mean_brier: float
    calibrated_mean_brier: float
    raw_mean_negative_log_likelihood: float
    calibrated_mean_negative_log_likelihood: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "scores": [score.to_dict() for score in self.scores],
            "reliability_bins": [
                reliability_bin.to_dict()
                for reliability_bin in self.reliability_bins
            ],
            "summary": {
                "raw_mean_brier": self.raw_mean_brier,
                "calibrated_mean_brier": self.calibrated_mean_brier,
                "calibrated_minus_raw_mean_brier": (
                    self.calibrated_mean_brier - self.raw_mean_brier
                ),
                "raw_mean_negative_log_likelihood": (
                    self.raw_mean_negative_log_likelihood
                ),
                "calibrated_mean_negative_log_likelihood": (
                    self.calibrated_mean_negative_log_likelihood
                ),
                "calibrated_minus_raw_mean_negative_log_likelihood": (
                    self.calibrated_mean_negative_log_likelihood
                    - self.raw_mean_negative_log_likelihood
                ),
            },
        }


def compare_raw_and_calibrated_forecasts(
    forecasts: Sequence[MarginalForecast],
    *,
    bin_count: int = 10,
) -> RawCalibratedComparison:
    """Score raw/calibrated forecasts and emit fixed-bin reliability rows.

    Reliability uses every class as a one-vs-rest probability/outcome pair.
    This avoids the invalid construction in which the probability assigned to
    the realized class is always compared with an outcome of one.
    """

    if not forecasts:
        raise ValueError("at least one forecast is required")
    if bin_count <= 0:
        raise ValueError("bin_count must be positive")
    scores: list[ForecastScore] = []
    reliability: dict[tuple[str, int], list[tuple[float, float]]] = {}
    for forecast in forecasts:
        for variant, probabilities in (
            ("raw", forecast.raw_probabilities),
            ("calibrated", forecast.calibrated_probabilities),
        ):
            true_probability = probabilities[forecast.true_index]
            scores.append(
                ForecastScore(
                    record_id=forecast.record_id,
                    cluster_id=forecast.cluster_id,
                    variant=variant,
                    brier=math.fsum(
                        (
                            probability
                            - (1.0 if index == forecast.true_index else 0.0)
                        )
                        ** 2
                        for index, probability in enumerate(probabilities)
                    ),
                    negative_log_likelihood=-math.log(
                        max(true_probability, 1e-15)
                    ),
                    true_class_probability=true_probability,
                    top_class_correct=(
                        max(
                            range(len(probabilities)),
                            key=lambda index: (probabilities[index], -index),
                        )
                        == forecast.true_index
                    ),
                )
            )
            for index, probability in enumerate(probabilities):
                bin_index = min(int(probability * bin_count), bin_count - 1)
                reliability.setdefault((variant, bin_index), []).append(
                    (
                        probability,
                        1.0 if index == forecast.true_index else 0.0,
                    )
                )
    rows: list[ReliabilityBin] = []
    for variant in ("raw", "calibrated"):
        for bin_index in range(bin_count):
            pairs = reliability.get((variant, bin_index), [])
            mean_probability = (
                None if not pairs else mean(pair[0] for pair in pairs)
            )
            observed = None if not pairs else mean(pair[1] for pair in pairs)
            rows.append(
                ReliabilityBin(
                    variant=variant,
                    bin_index=bin_index,
                    lower_bound=bin_index / bin_count,
                    upper_bound=(bin_index + 1) / bin_count,
                    prediction_count=len(pairs),
                    mean_predicted_probability=mean_probability,
                    observed_frequency=observed,
                    calibration_gap=(
                        None
                        if mean_probability is None or observed is None
                        else observed - mean_probability
                    ),
                )
            )
    raw_scores = tuple(score for score in scores if score.variant == "raw")
    calibrated_scores = tuple(
        score for score in scores if score.variant == "calibrated"
    )
    return RawCalibratedComparison(
        scores=tuple(scores),
        reliability_bins=tuple(rows),
        raw_mean_brier=mean(score.brier for score in raw_scores),
        calibrated_mean_brier=mean(
            score.brier for score in calibrated_scores
        ),
        raw_mean_negative_log_likelihood=mean(
            score.negative_log_likelihood for score in raw_scores
        ),
        calibrated_mean_negative_log_likelihood=mean(
            score.negative_log_likelihood for score in calibrated_scores
        ),
    )


@dataclass(frozen=True, slots=True)
class MultiplicityDecision:
    hypothesis_id: str
    raw_p_value: float
    adjusted_p_value: float
    rejection_threshold: float
    reject: bool
    rank: int

    def to_dict(self) -> dict[str, float | int | str | bool]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "raw_p_value": self.raw_p_value,
            "adjusted_p_value": self.adjusted_p_value,
            "rejection_threshold": self.rejection_threshold,
            "reject": self.reject,
            "rank": self.rank,
        }


@dataclass(frozen=True, slots=True)
class HolmMultiplicityResult:
    alpha: float
    method: str
    decisions: tuple[MultiplicityDecision, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "method": self.method,
            "decisions": [
                decision.to_dict() for decision in self.decisions
            ],
        }


def holm_bonferroni(
    p_values: Mapping[str, float],
    *,
    alpha: float = 0.05,
) -> HolmMultiplicityResult:
    """Apply Holm's family-wise-error-rate step-down procedure."""

    if not p_values:
        raise ValueError("Holm correction requires at least one hypothesis")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    if any(
        not math.isfinite(value) or not 0 <= value <= 1
        for value in p_values.values()
    ):
        raise ValueError("p-values must be finite and lie in [0, 1]")
    ordered = sorted(p_values, key=lambda key: (p_values[key], key))
    count = len(ordered)
    adjusted: dict[str, float] = {}
    running_adjusted = 0.0
    continue_rejecting = True
    decisions = []
    for index, hypothesis_id in enumerate(ordered):
        remaining = count - index
        raw = float(p_values[hypothesis_id])
        running_adjusted = max(running_adjusted, remaining * raw)
        threshold = alpha / remaining
        reject = continue_rejecting and raw <= threshold
        if not reject:
            continue_rejecting = False
        adjusted[hypothesis_id] = min(1.0, running_adjusted)
        decisions.append(
            MultiplicityDecision(
                hypothesis_id=hypothesis_id,
                raw_p_value=raw,
                adjusted_p_value=adjusted[hypothesis_id],
                rejection_threshold=threshold,
                reject=reject,
                rank=index + 1,
            )
        )
    return HolmMultiplicityResult(
        alpha=alpha,
        method="Holm step-down family-wise error-rate correction",
        decisions=tuple(decisions),
    )


def _matrix_inverse(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    size = len(matrix)
    if size == 0 or any(len(row) != size for row in matrix):
        raise ValueError("matrix must be non-empty and square")
    augmented = [
        [float(value) for value in row]
        + [1.0 if row_index == column else 0.0 for column in range(size)]
        for row_index, row in enumerate(matrix)
    ]
    for column in range(size):
        pivot = max(
            range(column, size),
            key=lambda row_index: abs(augmented[row_index][column]),
        )
        scale = max(
            1.0,
            max(abs(augmented[row_index][column]) for row_index in range(size)),
        )
        if abs(augmented[pivot][column]) <= 1e-12 * scale:
            raise ValueError("design matrix is rank deficient")
        augmented[column], augmented[pivot] = (
            augmented[pivot],
            augmented[column],
        )
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row_index in range(size):
            if row_index == column:
                continue
            factor = augmented[row_index][column]
            if factor == 0.0:
                continue
            augmented[row_index] = [
                value - factor * pivot_value
                for value, pivot_value in zip(
                    augmented[row_index],
                    augmented[column],
                )
            ]
    return [row[size:] for row in augmented]


def _matrix_multiply(
    first: Sequence[Sequence[float]],
    second: Sequence[Sequence[float]],
) -> list[list[float]]:
    if not first or not second or len(first[0]) != len(second):
        raise ValueError("incompatible matrix dimensions")
    transposed = list(zip(*second))
    return [
        [
            math.fsum(left * right for left, right in zip(row, column))
            for column in transposed
        ]
        for row in first
    ]


@dataclass(frozen=True, slots=True)
class ClusterRobustCoefficient:
    name: str
    estimate: float
    standard_error: float
    z_statistic: float | None
    p_value: float | None
    lower: float
    upper: float

    def to_dict(self) -> dict[str, float | str | None]:
        return {
            "name": self.name,
            "estimate": self.estimate,
            "standard_error": self.standard_error,
            "z_statistic": self.z_statistic,
            "p_value": self.p_value,
            "lower": self.lower,
            "upper": self.upper,
        }


@dataclass(frozen=True, slots=True)
class ClusterRobustOLSResult:
    """Marginal OLS coefficients with CR1 cluster-robust covariance."""

    model_label: str
    coefficient_names: tuple[str, ...]
    coefficients: tuple[ClusterRobustCoefficient, ...]
    observation_count: int
    cluster_count: int
    residual_degrees_of_freedom: int
    r_squared: float
    covariance_method: str
    inference_note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_label": self.model_label,
            "coefficient_names": list(self.coefficient_names),
            "coefficients": [
                coefficient.to_dict() for coefficient in self.coefficients
            ],
            "observation_count": self.observation_count,
            "cluster_count": self.cluster_count,
            "residual_degrees_of_freedom": self.residual_degrees_of_freedom,
            "r_squared": self.r_squared,
            "covariance_method": self.covariance_method,
            "inference_note": self.inference_note,
        }


def fit_cluster_robust_ols(
    design: Sequence[Sequence[float]],
    outcomes: Sequence[float],
    cluster_ids: Sequence[str],
    coefficient_names: Sequence[str],
    *,
    model_label: str = "marginal OLS working-independence model",
    confidence_level: float = 0.95,
) -> ClusterRobustOLSResult:
    """Fit OLS and a participant-clustered CR1 sandwich covariance.

    The p-values and intervals use a normal reference distribution.  This is a
    large-cluster approximation; the returned metadata says so explicitly.
    """

    observation_count = len(outcomes)
    if (
        len(design) != observation_count
        or len(cluster_ids) != observation_count
        or observation_count == 0
    ):
        raise ValueError("design, outcomes, and clusters need equal non-zero length")
    parameter_count = len(coefficient_names)
    if parameter_count == 0 or any(len(row) != parameter_count for row in design):
        raise ValueError("design width must equal coefficient_names length")
    if observation_count <= parameter_count:
        raise ValueError("OLS requires more observations than coefficients")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must lie in (0, 1)")
    x = [[float(value) for value in row] for row in design]
    y = [float(value) for value in outcomes]
    if any(not math.isfinite(value) for row in x for value in row) or any(
        not math.isfinite(value) for value in y
    ):
        raise ValueError("OLS inputs must be finite")
    unique_clusters = sorted(set(cluster_ids))
    if len(unique_clusters) < 2 or any(not cluster for cluster in cluster_ids):
        raise ValueError("cluster-robust covariance needs at least two clusters")

    xtx = [
        [
            math.fsum(row[left] * row[right] for row in x)
            for right in range(parameter_count)
        ]
        for left in range(parameter_count)
    ]
    xtx_inverse = _matrix_inverse(xtx)
    xty = [
        [math.fsum(row[column] * outcome for row, outcome in zip(x, y))]
        for column in range(parameter_count)
    ]
    beta = [
        value[0] for value in _matrix_multiply(xtx_inverse, xty)
    ]
    fitted = [
        math.fsum(value * coefficient for value, coefficient in zip(row, beta))
        for row in x
    ]
    residuals = [
        outcome - prediction for outcome, prediction in zip(y, fitted)
    ]

    cluster_scores: dict[str, list[float]] = {
        cluster: [0.0] * parameter_count for cluster in unique_clusters
    }
    for row, residual, cluster in zip(x, residuals, cluster_ids):
        score = cluster_scores[cluster]
        for column in range(parameter_count):
            score[column] += row[column] * residual
    meat = [
        [
            math.fsum(
                score[left] * score[right]
                for score in cluster_scores.values()
            )
            for right in range(parameter_count)
        ]
        for left in range(parameter_count)
    ]
    covariance = _matrix_multiply(
        _matrix_multiply(xtx_inverse, meat),
        xtx_inverse,
    )
    cluster_count = len(unique_clusters)
    correction = (
        cluster_count
        / (cluster_count - 1)
        * (observation_count - 1)
        / (observation_count - parameter_count)
    )
    covariance = [
        [value * correction for value in row] for row in covariance
    ]
    normal = NormalDist()
    critical = normal.inv_cdf(0.5 + confidence_level / 2.0)
    coefficient_rows = []
    for index, name in enumerate(coefficient_names):
        variance = max(0.0, covariance[index][index])
        standard_error = math.sqrt(variance)
        if standard_error == 0.0:
            z_statistic = None
            p_value = None
        else:
            z_statistic = beta[index] / standard_error
            p_value = 2.0 * (
                1.0 - normal.cdf(abs(z_statistic))
            )
        coefficient_rows.append(
            ClusterRobustCoefficient(
                name=str(name),
                estimate=beta[index],
                standard_error=standard_error,
                z_statistic=z_statistic,
                p_value=p_value,
                lower=beta[index] - critical * standard_error,
                upper=beta[index] + critical * standard_error,
            )
        )
    outcome_mean = mean(y)
    total_sum_squares = math.fsum(
        (outcome - outcome_mean) ** 2 for outcome in y
    )
    residual_sum_squares = math.fsum(
        residual * residual for residual in residuals
    )
    r_squared = (
        math.nan
        if total_sum_squares == 0.0
        else 1.0 - residual_sum_squares / total_sum_squares
    )
    return ClusterRobustOLSResult(
        model_label=model_label,
        coefficient_names=tuple(str(name) for name in coefficient_names),
        coefficients=tuple(coefficient_rows),
        observation_count=observation_count,
        cluster_count=cluster_count,
        residual_degrees_of_freedom=observation_count - parameter_count,
        r_squared=r_squared,
        covariance_method=(
            "CR1 sandwich covariance clustered by independent participant/trajectory"
        ),
        inference_note=(
            "Normal-reference intervals are a large-cluster approximation; "
            "this marginal working-independence OLS is not a mixed-effects model."
        ),
    )


@dataclass(frozen=True, slots=True)
class PowerAnalysisPoint:
    sample_size: int
    rejections: int
    simulations: int
    estimated_power: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "sample_size": self.sample_size,
            "rejections": self.rejections,
            "simulations": self.simulations,
            "estimated_power": self.estimated_power,
        }


@dataclass(frozen=True, slots=True)
class PowerAnalysisArtifact:
    estimand: str
    method: str
    seed: int
    alpha: float
    target_effect: float
    pilot_cluster_count: int
    points: tuple[PowerAnalysisPoint, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimand": self.estimand,
            "method": self.method,
            "seed": self.seed,
            "alpha": self.alpha,
            "target_effect": self.target_effect,
            "pilot_cluster_count": self.pilot_cluster_count,
            "points": [point.to_dict() for point in self.points],
        }


def simulate_paired_cluster_power(
    pilot_cluster_differences: Sequence[float],
    sample_sizes: Sequence[int],
    *,
    estimand: str,
    target_effect: float | None = None,
    simulations: int = 2000,
    alpha: float = 0.05,
    seed: int = 1729,
) -> PowerAnalysisArtifact:
    """Simulate paired-cluster power from centered empirical pilot residuals.

    A two-sided normal-reference test is used, matching the declared
    approximation in the artifact.  ``target_effect`` makes sensitivity to the
    assumed effect explicit rather than silently reusing the noisy pilot mean.
    """

    pilot = tuple(float(value) for value in pilot_cluster_differences)
    if len(pilot) < 2 or any(not math.isfinite(value) for value in pilot):
        raise ValueError("power analysis needs at least two finite pilot clusters")
    if not sample_sizes or any(size < 2 for size in sample_sizes):
        raise ValueError("power-analysis sample sizes must be at least two")
    if simulations <= 0:
        raise ValueError("simulations must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    effect = mean(pilot) if target_effect is None else float(target_effect)
    if not math.isfinite(effect):
        raise ValueError("target_effect must be finite")
    residuals = tuple(value - mean(pilot) for value in pilot)
    threshold = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    weights = [1.0] * len(residuals)
    points = []
    for sample_size in sample_sizes:
        rejections = 0
        for simulation in range(simulations):
            sample = [
                effect
                + residuals[
                    weighted_index(
                        weights,
                        seed,
                        "paired-cluster-power",
                        estimand,
                        sample_size,
                        simulation,
                        draw,
                    )
                ]
                for draw in range(sample_size)
            ]
            standard_error = stdev(sample) / math.sqrt(sample_size)
            reject = (
                mean(sample) != 0.0
                if standard_error == 0.0
                else abs(mean(sample) / standard_error) >= threshold
            )
            rejections += int(reject)
        points.append(
            PowerAnalysisPoint(
                sample_size=sample_size,
                rejections=rejections,
                simulations=simulations,
                estimated_power=rejections / simulations,
            )
        )
    return PowerAnalysisArtifact(
        estimand=estimand,
        method=(
            "paired-cluster simulation from centered empirical pilot residuals "
            "with a two-sided normal-reference test"
        ),
        seed=seed,
        alpha=alpha,
        target_effect=effect,
        pilot_cluster_count=len(pilot),
        points=tuple(points),
    )


@dataclass(frozen=True, slots=True)
class ConfirmatoryAnalysisResult:
    """Serializable bundle of explicitly named confirmatory estimands."""

    analysis_id: str
    independent_unit: str
    intervals: tuple[tuple[str, IntervalEstimate], ...] = ()
    contrasts: tuple[PairedContrast, ...] = ()
    regressions: tuple[ClusterRobustOLSResult, ...] = ()
    multiplicity: HolmMultiplicityResult | None = None
    power: PowerAnalysisArtifact | None = None
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis_id": self.analysis_id,
            "independent_unit": self.independent_unit,
            "intervals": {
                name: interval.to_dict() for name, interval in self.intervals
            },
            "contrasts": [
                contrast.to_dict() for contrast in self.contrasts
            ],
            "regressions": [
                regression.to_dict() for regression in self.regressions
            ],
            "multiplicity": (
                None if self.multiplicity is None else self.multiplicity.to_dict()
            ),
            "power": None if self.power is None else self.power.to_dict(),
            "notes": list(self.notes),
        }


def kendall_tau_b(
    first: Mapping[str, float],
    second: Mapping[str, float],
    *,
    tie_tolerance: float = 1e-12,
) -> float:
    if set(first) != set(second):
        raise ValueError("rankings must contain identical systems")
    names = sorted(first)
    concordant = discordant = first_ties = second_ties = 0
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            delta_first = first[left] - first[right]
            delta_second = second[left] - second[right]
            first_tie = abs(delta_first) <= tie_tolerance
            second_tie = abs(delta_second) <= tie_tolerance
            if first_tie and second_tie:
                continue
            if first_tie:
                first_ties += 1
            elif second_tie:
                second_ties += 1
            elif delta_first * delta_second > 0:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + first_ties)
        * (concordant + discordant + second_ties)
    )
    return math.nan if denominator == 0 else (concordant - discordant) / denominator


def ranks_from_errors(
    errors: Mapping[str, float], *, tie_tolerance: float = 1e-6
) -> dict[str, float]:
    """Lower error receives better rank; statistical ties share average rank."""

    ordered = sorted(errors, key=lambda name: (errors[name], name))
    ranks: dict[str, float] = {}
    index = 0
    while index < len(ordered):
        end = index + 1
        while (
            end < len(ordered)
            and abs(errors[ordered[end]] - errors[ordered[index]]) <= tie_tolerance
        ):
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        for name in ordered[index:end]:
            ranks[name] = average_rank
        index = end
    return ranks


@dataclass(frozen=True, slots=True)
class BootstrapRankSummary:
    system_id: str
    mean_rank: float
    lower: float
    upper: float


def bootstrap_ranks(
    errors_by_system: Mapping[str, Sequence[float]],
    *,
    replicates: int,
    seed: int,
    tie_tolerance: float = 1e-6,
) -> tuple[BootstrapRankSummary, ...]:
    if not errors_by_system:
        raise ValueError("no systems supplied")
    if len(errors_by_system) < 2:
        raise ValueError("rank bootstrap requires at least two systems")
    if replicates <= 0:
        raise ValueError("rank bootstrap requires positive replicates")
    if tie_tolerance < 0:
        raise ValueError("tie_tolerance must be non-negative")
    lengths = {len(values) for values in errors_by_system.values()}
    if len(lengths) != 1 or next(iter(lengths)) == 0:
        raise ValueError("systems need equal non-empty paired trajectory samples")
    count = next(iter(lengths))
    rank_draws = {system: [] for system in errors_by_system}
    weights = [1.0] * count
    for replicate in range(replicates):
        indexes = [
            weighted_index(weights, seed, "rank-bootstrap", replicate, draw)
            for draw in range(count)
        ]
        errors = {
            system: mean(values[index] for index in indexes)
            for system, values in errors_by_system.items()
        }
        for system, rank in ranks_from_errors(
            errors, tie_tolerance=tie_tolerance
        ).items():
            rank_draws[system].append(rank)
    summaries = []
    for system, values in sorted(rank_draws.items()):
        summaries.append(
            BootstrapRankSummary(
                system,
                mean(values),
                percentile(values, 0.025),
                percentile(values, 0.975),
            )
        )
    return tuple(summaries)


def pairwise_reversal_probability(
    first_errors: Mapping[str, Sequence[float]],
    second_errors: Mapping[str, Sequence[float]],
    *,
    replicates: int,
    seed: int,
    tie_tolerance: float = 1e-6,
) -> dict[str, float]:
    reversals, _ = pairwise_reversal_and_tie_probability(
        first_errors,
        second_errors,
        replicates=replicates,
        seed=seed,
        tie_tolerance=tie_tolerance,
    )
    return reversals


def pairwise_reversal_and_tie_probability(
    first_errors: Mapping[str, Sequence[float]],
    second_errors: Mapping[str, Sequence[float]],
    *,
    replicates: int,
    seed: int,
    tie_tolerance: float = 1e-6,
) -> tuple[dict[str, float], dict[str, float]]:
    """Return reversal and ambiguous/tied-draw probabilities per system pair."""

    if set(first_errors) != set(second_errors):
        raise ValueError("regimes must contain identical systems")
    if replicates <= 0:
        raise ValueError("reversal bootstrap requires positive replicates")
    if tie_tolerance < 0:
        raise ValueError("tie_tolerance must be non-negative")
    systems = sorted(first_errors)
    lengths = {len(values) for values in first_errors.values()} | {
        len(values) for values in second_errors.values()
    }
    if len(lengths) != 1 or next(iter(lengths)) == 0:
        raise ValueError("all paired samples must have the same non-zero length")
    count = next(iter(lengths))
    reversed_counts: Counter[str] = Counter()
    tie_counts: Counter[str] = Counter()
    weights = [1.0] * count
    for replicate in range(replicates):
        indexes = [
            weighted_index(weights, seed, "reversal-bootstrap", replicate, draw)
            for draw in range(count)
        ]
        for left_index, left in enumerate(systems):
            for right in systems[left_index + 1 :]:
                first_delta = mean(
                    first_errors[left][i] - first_errors[right][i] for i in indexes
                )
                second_delta = mean(
                    second_errors[left][i] - second_errors[right][i] for i in indexes
                )
                pair_id = f"{left}|{right}"
                if (
                    abs(first_delta) <= tie_tolerance
                    or abs(second_delta) <= tie_tolerance
                ):
                    tie_counts[pair_id] += 1
                elif first_delta * second_delta < 0:
                    reversed_counts[pair_id] += 1
    reversals = {
        f"{left}|{right}": reversed_counts[f"{left}|{right}"] / replicates
        for left_index, left in enumerate(systems)
        for right in systems[left_index + 1 :]
    }
    ties = {
        f"{left}|{right}": tie_counts[f"{left}|{right}"] / replicates
        for left_index, left in enumerate(systems)
        for right in systems[left_index + 1 :]
    }
    return reversals, ties


def evaluation_selection_regret(
    open_dev: Mapping[str, float],
    closed_dev: Mapping[str, float],
    closed_test: Mapping[str, float],
) -> dict[str, float | str]:
    if not set(open_dev) == set(closed_dev) == set(closed_test):
        raise ValueError("all ESR inputs must contain the same systems")
    open_selected = min(open_dev, key=lambda name: (open_dev[name], name))
    closed_selected = min(closed_dev, key=lambda name: (closed_dev[name], name))
    return {
        "open_selected": open_selected,
        "closed_selected": closed_selected,
        "open_selected_closed_test_error": closed_test[open_selected],
        "closed_selected_closed_test_error": closed_test[closed_selected],
        "evaluation_selection_regret": (
            closed_test[open_selected] - closed_test[closed_selected]
        ),
    }


@dataclass(frozen=True, slots=True)
class PairwiseDifferenceInterval:
    """Paired-bootstrap uncertainty for one system-error difference."""

    first_system: str
    second_system: str
    estimate: float
    lower: float
    upper: float
    relation: str
    tie_tolerance: float
    replicate_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "first_system": self.first_system,
            "second_system": self.second_system,
            "estimand": (
                f"mean_error[{self.first_system}] - "
                f"mean_error[{self.second_system}]"
            ),
            "estimate": self.estimate,
            "lower": self.lower,
            "upper": self.upper,
            "relation": self.relation,
            "tie_tolerance": self.tie_tolerance,
            "replicate_count": self.replicate_count,
            "method": "paired independent-unit percentile bootstrap",
        }


@dataclass(frozen=True, slots=True)
class PairwiseRegimeShiftInterval:
    """Joint paired inference for one system pair across two regimes.

    The estimands are formed from the same independent-unit indexes in every
    bootstrap draw:

    ``open = error[first] - error[second]``
    ``closed = error[first] - error[second]``
    ``shift = closed - open``

    A credible reversal requires both regime-specific intervals to clear the
    declared tie region in opposite directions *and* the shift interval to
    clear that region in the corresponding direction.  Marginal rank
    intervals are deliberately not used.
    """

    first_system: str
    second_system: str
    open_estimate: float
    open_lower: float
    open_upper: float
    open_relation: str
    closed_estimate: float
    closed_lower: float
    closed_upper: float
    closed_relation: str
    shift_estimate: float
    shift_lower: float
    shift_upper: float
    reversal_relation: str
    tie_tolerance: float
    replicate_count: int
    independent_unit_count: int

    @property
    def credible_reversal(self) -> bool:
        return self.reversal_relation in {
            "first_better_open_second_better_closed",
            "second_better_open_first_better_closed",
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "first_system": self.first_system,
            "second_system": self.second_system,
            "open_estimand": (
                f"mean_open_error[{self.first_system}] - "
                f"mean_open_error[{self.second_system}]"
            ),
            "open_estimate": self.open_estimate,
            "open_lower": self.open_lower,
            "open_upper": self.open_upper,
            "open_relation": self.open_relation,
            "closed_estimand": (
                f"mean_closed_error[{self.first_system}] - "
                f"mean_closed_error[{self.second_system}]"
            ),
            "closed_estimate": self.closed_estimate,
            "closed_lower": self.closed_lower,
            "closed_upper": self.closed_upper,
            "closed_relation": self.closed_relation,
            "shift_estimand": (
                "(closed first-minus-second error) - "
                "(open first-minus-second error)"
            ),
            "shift_estimate": self.shift_estimate,
            "shift_lower": self.shift_lower,
            "shift_upper": self.shift_upper,
            "reversal_relation": self.reversal_relation,
            "credible_reversal": self.credible_reversal,
            "tie_tolerance": self.tie_tolerance,
            "replicate_count": self.replicate_count,
            "independent_unit_count": self.independent_unit_count,
            "method": (
                "joint paired percentile bootstrap over the same independent "
                "units in open and closed regimes"
            ),
        }


def _difference_relation(
    lower: float,
    upper: float,
    *,
    tie_tolerance: float,
) -> str:
    if upper < -tie_tolerance:
        return "first_better"
    if lower > tie_tolerance:
        return "second_better"
    return "inferentially_undetermined"


def paired_system_regime_shift_intervals(
    open_errors: Mapping[str, Sequence[float]],
    closed_errors: Mapping[str, Sequence[float]],
    *,
    replicates: int,
    seed: int,
    tie_tolerance: float = 1e-6,
    confidence_level: float = 0.95,
) -> tuple[PairwiseRegimeShiftInterval, ...]:
    """Infer open/closed pair order and its shift with one paired bootstrap."""

    if set(open_errors) != set(closed_errors):
        raise ValueError("regimes must contain identical systems")
    systems = sorted(open_errors)
    if len(systems) < 2:
        raise ValueError("regime-shift intervals require at least two systems")
    if replicates <= 0:
        raise ValueError("regime-shift intervals require positive replicates")
    if tie_tolerance < 0:
        raise ValueError("tie_tolerance must be non-negative")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must lie in (0, 1)")
    lengths = {
        len(values) for values in open_errors.values()
    } | {
        len(values) for values in closed_errors.values()
    }
    if len(lengths) != 1 or next(iter(lengths)) == 0:
        raise ValueError(
            "all regime samples need equal non-empty paired independent units"
        )
    count = next(iter(lengths))
    weights = [1.0] * count
    tail = (1.0 - confidence_level) / 2.0
    output = []
    for first_index, first in enumerate(systems):
        for second in systems[first_index + 1 :]:
            open_unit_differences = tuple(
                first_value - second_value
                for first_value, second_value in zip(
                    open_errors[first],
                    open_errors[second],
                )
            )
            closed_unit_differences = tuple(
                first_value - second_value
                for first_value, second_value in zip(
                    closed_errors[first],
                    closed_errors[second],
                )
            )
            open_estimate = mean(open_unit_differences)
            closed_estimate = mean(closed_unit_differences)
            shift_estimate = closed_estimate - open_estimate
            open_draws = []
            closed_draws = []
            shift_draws = []
            for replicate in range(replicates):
                indexes = [
                    weighted_index(
                        weights,
                        seed,
                        "paired-system-regime-shift",
                        first,
                        second,
                        replicate,
                        draw,
                    )
                    for draw in range(count)
                ]
                open_draw = mean(
                    open_unit_differences[index] for index in indexes
                )
                closed_draw = mean(
                    closed_unit_differences[index] for index in indexes
                )
                open_draws.append(open_draw)
                closed_draws.append(closed_draw)
                shift_draws.append(closed_draw - open_draw)
            open_lower = percentile(open_draws, tail)
            open_upper = percentile(open_draws, 1.0 - tail)
            closed_lower = percentile(closed_draws, tail)
            closed_upper = percentile(closed_draws, 1.0 - tail)
            shift_lower = percentile(shift_draws, tail)
            shift_upper = percentile(shift_draws, 1.0 - tail)
            open_relation = _difference_relation(
                open_lower,
                open_upper,
                tie_tolerance=tie_tolerance,
            )
            closed_relation = _difference_relation(
                closed_lower,
                closed_upper,
                tie_tolerance=tie_tolerance,
            )
            if (
                open_relation == "first_better"
                and closed_relation == "second_better"
                and shift_lower > tie_tolerance
            ):
                reversal_relation = (
                    "first_better_open_second_better_closed"
                )
            elif (
                open_relation == "second_better"
                and closed_relation == "first_better"
                and shift_upper < -tie_tolerance
            ):
                reversal_relation = (
                    "second_better_open_first_better_closed"
                )
            else:
                reversal_relation = "no_credible_reversal"
            output.append(
                PairwiseRegimeShiftInterval(
                    first_system=first,
                    second_system=second,
                    open_estimate=open_estimate,
                    open_lower=open_lower,
                    open_upper=open_upper,
                    open_relation=open_relation,
                    closed_estimate=closed_estimate,
                    closed_lower=closed_lower,
                    closed_upper=closed_upper,
                    closed_relation=closed_relation,
                    shift_estimate=shift_estimate,
                    shift_lower=shift_lower,
                    shift_upper=shift_upper,
                    reversal_relation=reversal_relation,
                    tie_tolerance=tie_tolerance,
                    replicate_count=replicates,
                    independent_unit_count=count,
                )
            )
    return tuple(output)


def paired_system_difference_intervals(
    errors_by_system: Mapping[str, Sequence[float]],
    *,
    replicates: int,
    seed: int,
    tie_tolerance: float = 1e-6,
    confidence_level: float = 0.95,
) -> tuple[PairwiseDifferenceInterval, ...]:
    """Infer pairwise dominance only when a paired interval clears zero."""

    systems = sorted(errors_by_system)
    if len(systems) < 2:
        raise ValueError("pairwise intervals require at least two systems")
    if replicates <= 0:
        raise ValueError("pairwise intervals require positive replicates")
    if tie_tolerance < 0:
        raise ValueError("tie_tolerance must be non-negative")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must lie in (0, 1)")
    lengths = {len(values) for values in errors_by_system.values()}
    if len(lengths) != 1 or next(iter(lengths)) == 0:
        raise ValueError("systems need equal non-empty paired samples")
    count = next(iter(lengths))
    weights = [1.0] * count
    tail = (1.0 - confidence_level) / 2.0
    output = []
    for first_index, first in enumerate(systems):
        for second in systems[first_index + 1 :]:
            observed = mean(
                first_value - second_value
                for first_value, second_value in zip(
                    errors_by_system[first],
                    errors_by_system[second],
                )
            )
            draws = []
            for replicate in range(replicates):
                indexes = [
                    weighted_index(
                        weights,
                        seed,
                        "paired-system-difference",
                        first,
                        second,
                        replicate,
                        draw,
                    )
                    for draw in range(count)
                ]
                draws.append(
                    mean(
                        errors_by_system[first][index]
                        - errors_by_system[second][index]
                        for index in indexes
                    )
                )
            lower = percentile(draws, tail)
            upper = percentile(draws, 1.0 - tail)
            if upper < -tie_tolerance:
                relation = "first_better"
            elif lower > tie_tolerance:
                relation = "second_better"
            else:
                relation = "inferentially_undetermined"
            output.append(
                PairwiseDifferenceInterval(
                    first_system=first,
                    second_system=second,
                    estimate=observed,
                    lower=lower,
                    upper=upper,
                    relation=relation,
                    tie_tolerance=tie_tolerance,
                    replicate_count=replicates,
                )
            )
    return tuple(output)


def inferential_partial_order(
    systems: Sequence[str],
    intervals: Sequence[PairwiseDifferenceInterval],
) -> tuple[tuple[str, ...], ...]:
    """Return topological tiers from interval-supported pairwise dominance."""

    remaining = set(systems)
    if len(remaining) != len(tuple(systems)) or not remaining:
        raise ValueError("systems must be distinct and non-empty")
    expected_pairs = {
        tuple(sorted((first, second)))
        for index, first in enumerate(sorted(remaining))
        for second in sorted(remaining)[index + 1 :]
    }
    observed_pairs = {
        tuple(sorted((interval.first_system, interval.second_system)))
        for interval in intervals
    }
    if expected_pairs != observed_pairs:
        raise ValueError("intervals do not cover every system pair exactly")
    dominates: dict[str, set[str]] = {system: set() for system in remaining}
    for interval in intervals:
        if interval.relation == "first_better":
            dominates[interval.first_system].add(interval.second_system)
        elif interval.relation == "second_better":
            dominates[interval.second_system].add(interval.first_system)
    tiers = []
    while remaining:
        dominated = {
            loser
            for winner in remaining
            for loser in dominates[winner]
            if loser in remaining
        }
        tier = sorted(remaining - dominated)
        if not tier:
            # Sampling uncertainty can yield non-transitive dominance. Preserve
            # the cycle as one unresolved tier rather than breaking by ID.
            tier = sorted(remaining)
        tiers.append(tuple(tier))
        remaining.difference_update(tier)
    return tuple(tiers)


def tied_evaluation_selection_regret(
    open_dev: Mapping[str, float],
    closed_dev: Mapping[str, float],
    closed_test: Mapping[str, float],
    *,
    tie_tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Return set-valued selections and the full ESR range under ties."""

    if not set(open_dev) == set(closed_dev) == set(closed_test):
        raise ValueError("all ESR inputs must contain the same systems")
    if not open_dev:
        raise ValueError("ESR requires at least one system")
    if tie_tolerance < 0:
        raise ValueError("tie_tolerance must be non-negative")
    open_minimum = min(open_dev.values())
    closed_minimum = min(closed_dev.values())
    open_selected = tuple(
        sorted(
            system
            for system, value in open_dev.items()
            if value <= open_minimum + tie_tolerance
        )
    )
    closed_selected = tuple(
        sorted(
            system
            for system, value in closed_dev.items()
            if value <= closed_minimum + tie_tolerance
        )
    )
    values = tuple(
        closed_test[open_system] - closed_test[closed_system]
        for open_system in open_selected
        for closed_system in closed_selected
    )
    return {
        "open_selected_set": open_selected,
        "closed_selected_set": closed_selected,
        "selection_tie_tolerance": tie_tolerance,
        "evaluation_selection_regret": mean(values),
        "evaluation_selection_regret_min": min(values),
        "evaluation_selection_regret_max": max(values),
        "selection_policy": (
            "uniform expectation over every open-selected × "
            "closed-selected tied-system pair"
        ),
        "pair_count": len(values),
    }


def inferential_tier_evaluation_selection_regret(
    open_top_tier: Sequence[str],
    closed_top_tier: Sequence[str],
    closed_test_errors: Mapping[str, float],
    closed_test_intervals: Sequence[PairwiseDifferenceInterval],
) -> dict[str, Any]:
    """Evaluate every development-inferential top-tier selection on test.

    The top tiers must be obtained from paired development error-difference
    intervals.  The returned test interval is an envelope over paired
    complete-unit intervals for every open-tier × closed-tier deployment
    contrast.  It is intentionally conservative when development evidence
    leaves more than one candidate in either tier.
    """

    open_selected = tuple(sorted(set(open_top_tier)))
    closed_selected = tuple(sorted(set(closed_top_tier)))
    systems = set(closed_test_errors)
    if not open_selected or not closed_selected:
        raise ValueError("inferential ESR requires non-empty top tiers")
    if (
        set(open_selected) - systems
        or set(closed_selected) - systems
    ):
        raise ValueError("ESR top tiers contain an unknown system")
    interval_by_pair = {
        (interval.first_system, interval.second_system): interval
        for interval in closed_test_intervals
    }
    if len(interval_by_pair) != len(closed_test_intervals):
        raise ValueError("closed-test intervals contain duplicate pairs")

    def interval_for(first: str, second: str) -> tuple[float, float]:
        if first == second:
            return 0.0, 0.0
        canonical = tuple(sorted((first, second)))
        try:
            interval = interval_by_pair[canonical]
        except KeyError as exc:
            raise ValueError(
                f"missing closed-test interval for {canonical}"
            ) from exc
        if (first, second) == canonical:
            return interval.lower, interval.upper
        return -interval.upper, -interval.lower

    values = []
    lower_bounds = []
    upper_bounds = []
    pair_rows = []
    for open_system in open_selected:
        for closed_system in closed_selected:
            estimate = (
                closed_test_errors[open_system]
                - closed_test_errors[closed_system]
            )
            lower, upper = interval_for(open_system, closed_system)
            values.append(estimate)
            lower_bounds.append(lower)
            upper_bounds.append(upper)
            pair_rows.append(
                {
                    "open_selected_system": open_system,
                    "closed_selected_system": closed_system,
                    "closed_test_error_difference": estimate,
                    "lower": lower,
                    "upper": upper,
                }
            )
    return {
        "open_selected_set": open_selected,
        "closed_selected_set": closed_selected,
        "selection_basis": (
            "paired development error-difference confidence-set top tiers"
        ),
        "evaluation_selection_regret": mean(values),
        "evaluation_selection_regret_min": min(values),
        "evaluation_selection_regret_max": max(values),
        "evaluation_selection_regret_interval_envelope_lower": min(
            lower_bounds
        ),
        "evaluation_selection_regret_interval_envelope_upper": max(
            upper_bounds
        ),
        "selection_policy": (
            "uniform descriptive mean over every open-top-tier × "
            "closed-top-tier pair; claims use the conservative paired-test "
            "interval envelope"
        ),
        "pair_count": len(values),
        "pairwise_closed_test_intervals": pair_rows,
    }
