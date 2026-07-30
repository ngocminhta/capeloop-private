from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from cape_loop.analysis_export import (
    export_compact_analysis,
    verify_compact_analysis,
)
from cape_loop.artifacts import RunArtifacts, verify_run
from cape_loop.config import AppConfig, ExperimentSection, RunSection


def _uniform_belief() -> dict[str, object]:
    return {
        "kind": "theta_joint",
        "marginals": [[0.25, 0.25, 0.25, 0.25]] * 3,
    }


def _point_belief(theta: list[int]) -> dict[str, object]:
    values = (-2, -1, 1, 2)
    return {
        "kind": "theta_joint",
        "marginals": [
            [1.0 if value == truth else 0.0 for value in values]
            for truth in theta
        ],
    }


class CompactAnalysisExportTests(unittest.TestCase):
    def _run(
        self,
        root: Path,
        *,
        name: str,
        kind: str,
        turns: int = 1,
    ) -> RunArtifacts:
        experiment_kwargs: dict[str, object] = {
            "kind": kind,
            "turns": turns,
        }
        if kind != "provenance_audit":
            experiment_kwargs.update(
                {
                    "mechanisms": ("ranking", "default", "suggestion"),
                    "response_modes": ("naturally_sampled",),
                }
            )
        if kind == "evaluation_validity":
            experiment_kwargs.update(
                {
                    "policies": (
                        "balanced",
                        "fixed_bias",
                        "soft_profile_conditioned",
                    ),
                    "updaters": (
                        "response_only",
                        "provenance_aware",
                    ),
                    "bootstrap_replicates": 1,
                }
            )
        config = AppConfig(
            run=RunSection(name=name, seed=7),
            experiment=ExperimentSection(**experiment_kwargs),
        )
        return RunArtifacts.create(config, root=root)

    def test_exports_legacy_experiment_a_without_mutating_source(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            run = self._run(
                root / "runs",
                name="legacy-a",
                kind="provenance_audit",
            )
            rows = [
                {
                    "schema_version": 1,
                    "trial_id": f"trial-{index}",
                    "user_id": "user-1",
                    "domain_id": "travel",
                    "context": {"scenario_id": f"scenario-{index}"},
                    "updater_id": "fitted_action_aware",
                    "mechanism": "balanced",
                    "prior_strength": 0.35,
                    "response_mode": mode,
                    "metrics": {"acue": 0.1 + index / 100},
                }
                for index, mode in enumerate(
                    ("controlled_anchor", "naturally_sampled"),
                    start=1,
                )
            ]
            run.write_jsonl("events/experiment-a.jsonl", rows)
            run.write_jsonl("events/experiment-a-exclusions.jsonl", ())
            run.finalize(
                {
                    "experiment": "A",
                    "scientific_claim_status": "not_claimed",
                    "row_count": len(rows),
                    "natural_row_count": 1,
                    "excluded_matched_sets": 0,
                }
            )
            before = {
                path.relative_to(run.path).as_posix(): path.read_bytes()
                for path in run.path.rglob("*")
                if path.is_file()
            }

            bundle = export_compact_analysis(
                run.path,
                root / "compact-a",
            )

            self.assertEqual(bundle.experiment, "A")
            self.assertEqual(bundle.row_count, 2)
            self.assertEqual(
                {
                    path.name
                    for path in bundle.path.iterdir()
                    if path.is_file()
                },
                {"manifest.json", "analysis-rows.jsonl", "SHA256SUMS"},
            )
            compact_rows = [
                json.loads(line)
                for line in (
                    bundle.path / "analysis-rows.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]
            natural_row = compact_rows[1]
            self.assertAlmostEqual(natural_row.pop("update_error"), 0.12)
            self.assertEqual(
                natural_row,
                {
                    "schema_version": 1,
                    "source_record_index": 2,
                    "trial_id": "trial-2",
                    "user_id": "user-1",
                    "domain_id": "travel",
                    "scenario_id": "scenario-2",
                    "updater_id": "fitted_action_aware",
                    "mechanism": "balanced",
                    "prior_strength": 0.35,
                    "response_mode": "naturally_sampled",
                },
            )
            manifest = json.loads(
                (bundle.path / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertFalse(manifest["source_input_is_runner_compact"])
            self.assertEqual(
                manifest["source_input_file"],
                "events/experiment-a.jsonl",
            )
            valid, errors = verify_compact_analysis(bundle.path)
            self.assertTrue(valid, errors)
            valid, errors = verify_run(run.path)
            self.assertTrue(valid, errors)
            after = {
                path.relative_to(run.path).as_posix(): path.read_bytes()
                for path in run.path.rglob("*")
                if path.is_file()
            }
            self.assertEqual(before, after)

    def test_exports_legacy_experiment_b_as_turn_rows(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            run = self._run(
                root / "runs",
                name="legacy-b",
                kind="closed_loop",
                turns=2,
            )
            theta = [2, 1, -1]
            trajectory = {
                "schema_version": 1,
                "trajectory_id": "trajectory-1",
                "crn_key": "common-history-1",
                "user_id": "user-1",
                "domain_id": "travel",
                "updater_id": "fitted_action_aware",
                "policy_id": "balanced",
                "initial_profile_condition": "incorrect",
                "same_history_shadow": True,
                "theta": theta,
                "terminal_error": 0.0,
                "turns": [
                    {
                        "turn": 0,
                        "scenario_id": "scenario-1",
                        "theta_snapshot": theta,
                        "belief_after": _uniform_belief(),
                    },
                    {
                        "turn": 1,
                        "scenario_id": "scenario-2",
                        "theta_snapshot": theta,
                        "belief_after": _point_belief(theta),
                    },
                ],
            }
            run.write_jsonl(
                "events/experiment-b-trajectories.jsonl",
                (trajectory,),
            )
            run.finalize(
                {
                    "experiment": "B",
                    "scientific_claim_status": "not_claimed",
                    "trajectories": 1,
                }
            )

            bundle = export_compact_analysis(
                run.path,
                root / "compact-b",
            )

            rows = [
                json.loads(line)
                for line in (
                    bundle.path / "analysis-rows.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(rows), 2)
            self.assertEqual(
                [row["source_turn_index"] for row in rows],
                [0, 1],
            )
            self.assertEqual([row["turn"] for row in rows], [1, 2])
            self.assertEqual(
                [row["scenario_id"] for row in rows],
                ["scenario-1", "scenario-2"],
            )
            self.assertAlmostEqual(rows[0]["terminal_error"], 0.75)
            self.assertEqual(rows[1]["terminal_error"], 0.0)
            self.assertTrue(all(row["same_history_shadow"] for row in rows))
            valid, errors = verify_compact_analysis(bundle.path)
            self.assertTrue(valid, errors)

    def test_exports_legacy_experiment_c_and_detects_tampering(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            run = self._run(
                root / "runs",
                name="legacy-c",
                kind="evaluation_validity",
            )
            run.write_jsonl(
                "metrics/experiment-c.jsonl",
                (
                    {
                        "schema_version": 1,
                        "split": "test",
                        "regime": "endogenous_closed_loop",
                        "replicate": 0,
                        "user_id": "user-1",
                        "domain_id": "travel",
                        "updater_id": "provenance_aware",
                        "profile_error": 0.2,
                        "behavioral_accuracy": 0.8,
                        "cross_context_accuracy": 0.75,
                        "intrinsic_regret": 0.1,
                        "score_basis": "system_structured_projection",
                        "history_digest": "a" * 64,
                        "battery_id": "travel:terminal",
                        "battery_digest": "b" * 64,
                        "ranking_score": {
                            "profile_brier": 0.2,
                            "behavioral_accuracy": 0.8,
                            "cross_context_accuracy": 0.75,
                            "mean_intrinsic_regret": 0.1,
                        },
                    },
                ),
            )
            run.finalize(
                {
                    "experiment": "C",
                    "scientific_claim_status": "not_claimed",
                    "evaluation_rows": 1,
                }
            )
            bundle = export_compact_analysis(
                run.path,
                root / "compact-c",
            )
            compact_row = json.loads(
                (bundle.path / "analysis-rows.jsonl")
                .read_text(encoding="utf-8")
                .strip()
            )
            self.assertNotIn("ranking_score", compact_row)
            self.assertEqual(compact_row["profile_error"], 0.2)
            valid, errors = verify_compact_analysis(bundle.path)
            self.assertTrue(valid, errors)

            (bundle.path / "analysis-rows.jsonl").write_text(
                "{}\n",
                encoding="utf-8",
            )
            valid, errors = verify_compact_analysis(bundle.path)
            self.assertFalse(valid)
            self.assertTrue(
                any("checksum mismatch" in error for error in errors)
            )

    def test_rejects_output_inside_immutable_source_run(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            run = self._run(
                root / "runs",
                name="legacy-a-contained",
                kind="provenance_audit",
            )
            run.write_jsonl(
                "events/experiment-a.jsonl",
                (
                    {
                        "schema_version": 1,
                        "trial_id": "trial-1",
                        "user_id": "user-1",
                        "domain_id": "travel",
                        "context": {"scenario_id": "scenario-1"},
                        "updater_id": "fitted_action_aware",
                        "mechanism": "balanced",
                        "prior_strength": 0.35,
                        "response_mode": "naturally_sampled",
                        "metrics": {"acue": 0.1},
                    },
                ),
            )
            run.finalize(
                {
                    "experiment": "A",
                    "scientific_claim_status": "not_claimed",
                    "row_count": 1,
                }
            )
            with self.assertRaisesRegex(ValueError, "outside"):
                export_compact_analysis(
                    run.path,
                    run.path / "compact",
                )


if __name__ == "__main__":
    unittest.main()
