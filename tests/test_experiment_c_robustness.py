from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from contextlib import redirect_stdout
from io import StringIO
import json
import unittest

from cape_loop.artifacts import RunArtifacts, verify_run
from cape_loop.cli import main
from cape_loop.config import AppConfig, load_config
from cape_loop.experiment_c_robustness import (
    CLAIM_STATUS,
    COMPARISON_DIMENSIONS,
    create_experiment_c_multiseed_review,
    verify_experiment_c_multiseed_review,
)
from cape_loop.gates import GateCriterion, GateReport, incomplete_gate
from cape_loop.runner import run_experiment


SYSTEMS = ("response_only", "full_context_blind")
PAIR = "full_context_blind|response_only"


def _config(
    *,
    seed: int,
    name: str,
    output_root: str,
    ranking_tie_tolerance: float = 1e-6,
) -> AppConfig:
    raw = json.loads(load_config("configs/evaluation.toml").canonical_json())
    raw["run"].update(
        {
            "seed": seed,
            "name": name,
            "output_root": output_root,
        }
    )
    raw["experiment"].update(
        {
            "updaters": list(SYSTEMS),
            "users": 2,
            "trajectories_per_cell": 1,
            "turns": 1,
            "bootstrap_replicates": 20,
        }
    )
    raw["inference"].update(
        {
            "training_interactions": 24,
            "fit_steps": 10,
            "learning_rate": 0.04,
            "l2": 0.001,
        }
    )
    raw["thresholds"]["ranking_tie_tolerance"] = ranking_tie_tolerance
    return AppConfig.parse(raw)


def _difference_interval(
    *,
    estimate: float,
    lower: float,
    upper: float,
    relation: str,
    tie_tolerance: float,
) -> dict[str, object]:
    return {
        "first_system": "full_context_blind",
        "second_system": "response_only",
        "estimand": (
            "mean_error[full_context_blind] - mean_error[response_only]"
        ),
        "estimate": estimate,
        "lower": lower,
        "upper": upper,
        "relation": relation,
        "tie_tolerance": tie_tolerance,
        "replicate_count": 20,
        "method": "paired independent-unit percentile bootstrap",
    }


def _shift_interval(
    *,
    estimate: float,
    lower: float,
    upper: float,
    relation: str,
    tie_tolerance: float,
) -> dict[str, object]:
    return {
        "first_system": "full_context_blind",
        "second_system": "response_only",
        "open_estimand": (
            "mean_open_error[full_context_blind] - "
            "mean_open_error[response_only]"
        ),
        "open_estimate": estimate,
        "open_lower": lower,
        "open_upper": upper,
        "open_relation": relation,
        "closed_estimand": (
            "mean_closed_error[full_context_blind] - "
            "mean_closed_error[response_only]"
        ),
        "closed_estimate": estimate,
        "closed_lower": lower,
        "closed_upper": upper,
        "closed_relation": relation,
        "shift_estimand": (
            "(closed first-minus-second error) - "
            "(open first-minus-second error)"
        ),
        "shift_estimate": 0.0,
        "shift_lower": 0.0,
        "shift_upper": 0.0,
        "reversal_relation": "no_credible_reversal",
        "credible_reversal": False,
        "tie_tolerance": tie_tolerance,
        "replicate_count": 20,
        "independent_unit_count": 2,
        "method": (
            "joint paired percentile bootstrap over the same independent "
            "units in open and closed regimes"
        ),
    }


def _rankings(
    *,
    winner: str,
    tie_tolerance: float = 1e-6,
) -> dict[str, object]:
    loser = next(system for system in SYSTEMS if system != winner)
    ranks = {winner: 1.0, loser: 2.0}
    errors = {winner: 0.1, loser: 0.2}
    first_wins = winner == "full_context_blind"
    estimate = -0.1 if first_wins else 0.1
    lower = -0.15 if first_wins else 0.05
    upper = -0.05 if first_wins else 0.15
    relation = "first_better" if first_wins else "second_better"
    order = [[winner], [loser]]
    bootstrap_rows = [
        {
            "system_id": system,
            "mean_rank": ranks[system],
            "lower": ranks[system],
            "upper": ranks[system],
        }
        for system in sorted(SYSTEMS)
    ]
    esr_pair = {
        "open_selected_system": winner,
        "closed_selected_system": winner,
        "closed_test_error_difference": 0.0,
        "lower": 0.0,
        "upper": 0.0,
    }
    return {
        "inference_unit": "complete_latent_user_cluster",
        "alignment_key": [
            "split",
            "regime",
            "user_id",
            "domain_id",
            "replicate",
        ],
        "development_cluster_count": 2,
        "test_cluster_count": 2,
        "cluster_component_layout": [
            {"domain_id": "travel", "replicate": 0},
            {"domain_id": "writing", "replicate": 0},
        ],
        "bootstrap_method": (
            "paired percentile bootstrap over complete latent-user clusters; "
            "all domains and trajectory replicates remain grouped within each "
            "resampled unit"
        ),
        "open_mean_errors": errors,
        "biased_mean_errors": errors,
        "closed_development_mean_errors": errors,
        "closed_test_mean_errors": errors,
        "open_ranks": ranks,
        "biased_ranks": ranks,
        "closed_ranks": ranks,
        "open_closed_kendall_tau": 1.0,
        "biased_closed_kendall_tau": 1.0,
        "open_bootstrap_ranks": bootstrap_rows,
        "closed_bootstrap_ranks": bootstrap_rows,
        "pairwise_reversal_probabilities": {PAIR: 0.0},
        "pairwise_tie_probabilities": {PAIR: 0.0},
        "pairwise_open_difference_intervals": [
            _difference_interval(
                estimate=estimate,
                lower=lower,
                upper=upper,
                relation=relation,
                tie_tolerance=tie_tolerance,
            )
        ],
        "pairwise_closed_difference_intervals": [
            _difference_interval(
                estimate=estimate,
                lower=lower,
                upper=upper,
                relation=relation,
                tie_tolerance=tie_tolerance,
            )
        ],
        "pairwise_open_closed_shift_intervals": [
            _shift_interval(
                estimate=estimate,
                lower=lower,
                upper=upper,
                relation=relation,
                tie_tolerance=tie_tolerance,
            )
        ],
        "credible_pairwise_reversals": [],
        "credible_reversal_basis": (
            "joint paired open/closed complete-user error-difference intervals "
            "clear the tie region in opposite directions and their "
            "difference-of-differences interval clears it too"
        ),
        "open_partial_order": order,
        "closed_partial_order": order,
        "partial_order": order,
        "partial_order_basis": (
            "paired development error-difference intervals by regime; "
            "partial_order is the closed-development alias, and same-tier "
            "systems are not separated by interval-supported dominance"
        ),
        "open_loop_optimism": {system: 0.0 for system in SYSTEMS},
        "evaluation_selection_regret": {
            "open_selected_set": [winner],
            "closed_selected_set": [winner],
            "selection_basis": (
                "paired development error-difference confidence-set top tiers"
            ),
            "evaluation_selection_regret": 0.0,
            "evaluation_selection_regret_min": 0.0,
            "evaluation_selection_regret_max": 0.0,
            "evaluation_selection_regret_interval_envelope_lower": 0.0,
            "evaluation_selection_regret_interval_envelope_upper": 0.0,
            "selection_policy": (
                "uniform descriptive mean over every open-top-tier × "
                "closed-top-tier pair; claims use the conservative paired-test "
                "interval envelope"
            ),
            "pair_count": 1,
            "pairwise_closed_test_intervals": [esr_pair],
        },
    }


def _gate_report(*, decision: bool) -> dict[str, object]:
    titles = {
        1: "Learnable provenance gap",
        2: "Nontrivial soft self-confirmation",
        3: "Attribution beyond evidence selection",
        4: "Native-system validity",
        5: "Evaluation implication",
        6: "Robustness",
    }
    reports = [
        incomplete_gate(index, titles[index], "not evaluated in fixture")
        for index in (1, 2, 3, 4)
    ]
    reports.append(
        GateReport(
            gate_id="gate-5",
            title=titles[5],
            criteria=(
                GateCriterion(
                    "evaluation-implication-disjunction",
                    "fixture decision",
                    decision,
                    {"fixture": True},
                    "fixture requirement",
                ),
            ),
        )
    )
    reports.append(incomplete_gate(6, titles[6], "not evaluated in fixture"))
    return {
        "schema_version": 1,
        "claim_status": CLAIM_STATUS,
        "gates": [report.to_dict() for report in reports],
    }


def _source_run(
    root: Path,
    *,
    seed: int,
    winner: str,
    suffix: str,
    ranking_tie_tolerance: float = 1e-6,
) -> Path:
    config = _config(
        seed=seed,
        name=f"c-seed-{suffix}",
        output_root=f"ignored-{suffix}",
        ranking_tie_tolerance=ranking_tie_tolerance,
    )
    run = RunArtifacts.create(config, root=root)
    run.write_json(
        "metrics/experiment-c-rankings.json",
        _rankings(
            winner=winner,
            tie_tolerance=ranking_tie_tolerance,
        ),
    )
    decision = winner == "full_context_blind"
    gate = _gate_report(decision=decision)
    run.write_json("metrics/gate-report.json", gate)
    gate_status = (
        "meets_computational_checks"
        if decision
        else "does_not_meet_checks"
    )
    run.finalize(
        {
            "experiment": "C",
            "scientific_claim_status": CLAIM_STATUS,
            "gate_5_computed_status": gate_status,
        }
    )
    valid, errors = verify_run(run.path)
    if not valid:
        raise AssertionError(errors)
    return run.path


class ExperimentCMultiseedRobustnessTests(unittest.TestCase):
    def test_real_runner_artifacts_are_admitted(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "sources"
            first_result = run_experiment(
                _config(
                    seed=31,
                    name="actual-c-a",
                    output_root="ignored-a",
                ),
                output_root=source_root,
            )
            second_result = run_experiment(
                _config(
                    seed=37,
                    name="actual-c-b",
                    output_root="ignored-b",
                ),
                output_root=source_root,
            )
            first = Path(first_result["run_dir"])
            second = Path(second_result["run_dir"])
            output = root / "review"
            create_experiment_c_multiseed_review((first, second), output)
            valid, errors = verify_experiment_c_multiseed_review(
                output,
                source_run_dirs=(first, second),
            )
            self.assertTrue(valid, errors)

    def test_stable_review_is_atomic_checksum_bound_and_not_claimed(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first = _source_run(
                root / "sources",
                seed=11,
                winner="response_only",
                suffix="a",
            )
            second = _source_run(
                root / "sources",
                seed=29,
                winner="response_only",
                suffix="b",
            )
            output = root / "reviews" / "stable"
            result = create_experiment_c_multiseed_review(
                (second, first),
                output,
            )
            self.assertEqual(result["claim_status"], CLAIM_STATUS)
            self.assertTrue(result["all_predeclared_dimensions_unanimous"])
            valid, errors = verify_experiment_c_multiseed_review(
                output,
                source_run_dirs=(first, second),
            )
            self.assertTrue(valid, errors)
            review = json.loads(
                (output / "review.json").read_text(encoding="utf-8")
            )
            self.assertEqual(review["claim_status"], CLAIM_STATUS)
            self.assertEqual(
                review["overall"]["unanimous_dimension_proportion"]["fraction"],
                "1/1",
            )
            self.assertEqual(
                set(review["comparisons"]),
                set(COMPARISON_DIMENSIONS),
            )
            self.assertTrue(
                all(
                    comparison["pairwise_agreement_proportion"]["fraction"]
                    == "1/1"
                    for comparison in review["comparisons"].values()
                )
            )
            with self.assertRaises(FileExistsError):
                create_experiment_c_multiseed_review((first, second), output)
            cli_output = root / "reviews" / "cli"
            with redirect_stdout(StringIO()):
                self.assertEqual(
                    main(
                        [
                            "experiment-c-robustness",
                            "review",
                            str(cli_output),
                            str(first),
                            str(second),
                        ]
                    ),
                    0,
                )
                self.assertEqual(
                    main(
                        [
                            "experiment-c-robustness",
                            "verify",
                            str(cli_output),
                            "--source-run",
                            str(first),
                            "--source-run",
                            str(second),
                        ]
                    ),
                    0,
                )

    def test_unstable_review_reports_exact_disagreements_without_claim(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first = _source_run(
                root / "sources",
                seed=3,
                winner="response_only",
                suffix="a",
            )
            second = _source_run(
                root / "sources",
                seed=7,
                winner="full_context_blind",
                suffix="b",
            )
            output = root / "review"
            create_experiment_c_multiseed_review((first, second), output)
            review = json.loads(
                (output / "review.json").read_text(encoding="utf-8")
            )
            point = review["comparisons"][
                "point_ranking.fixed_balanced_development"
            ]
            gate = review["comparisons"]["gate_5.decision_and_status"]
            self.assertFalse(point["unanimous"])
            self.assertEqual(
                point["modal_stability_proportion"]["fraction"],
                "1/2",
            )
            self.assertEqual(
                point["pairwise_agreement_proportion"]["fraction"],
                "0/1",
            )
            self.assertEqual(len(point["disagreements"]), 1)
            self.assertFalse(gate["unanimous"])
            self.assertFalse(
                review["overall"]["all_predeclared_dimensions_unanimous"]
            )
            self.assertFalse(review["overall"]["scientific_claim_inferred"])
            self.assertEqual(review["claim_status"], CLAIM_STATUS)

    def test_duplicate_seed_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first = _source_run(
                root / "sources",
                seed=5,
                winner="response_only",
                suffix="a",
            )
            second = _source_run(
                root / "sources",
                seed=5,
                winner="response_only",
                suffix="b",
            )
            with self.assertRaisesRegex(ValueError, "distinct run.seed"):
                create_experiment_c_multiseed_review(
                    (first, second),
                    root / "review",
                )

    def test_incompatible_scientific_config_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first = _source_run(
                root / "sources",
                seed=13,
                winner="response_only",
                suffix="a",
            )
            second = _source_run(
                root / "sources",
                seed=17,
                winner="response_only",
                suffix="b",
                ranking_tie_tolerance=1e-5,
            )
            with self.assertRaisesRegex(ValueError, "incompatible scientific"):
                create_experiment_c_multiseed_review(
                    (first, second),
                    root / "review",
                )

    def test_review_tampering_is_detected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first = _source_run(
                root / "sources",
                seed=19,
                winner="response_only",
                suffix="a",
            )
            second = _source_run(
                root / "sources",
                seed=23,
                winner="response_only",
                suffix="b",
            )
            output = root / "review"
            create_experiment_c_multiseed_review((first, second), output)
            review_path = output / "review.json"
            review_path.write_text(
                review_path.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
            valid, errors = verify_experiment_c_multiseed_review(output)
            self.assertFalse(valid)
            self.assertTrue(
                any("checksum mismatch: review.json" in error for error in errors)
            )


if __name__ == "__main__":
    unittest.main()
