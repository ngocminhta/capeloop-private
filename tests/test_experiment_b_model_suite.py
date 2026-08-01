from __future__ import annotations

from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import io
import json
import unittest

from cape_loop.artifacts import config_digest
from cape_loop.cli import build_parser, main as cli_main
from cape_loop.config import AppConfig, load_config
from cape_loop.experiment_b_model_suite import (
    DEFAULT_SUITE_PATH,
    build_experiment_b_model_suite_plan,
    load_experiment_b_model_suite,
    orchestrate_experiment_b_model_suite,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = (
    REPOSITORY_ROOT / "configs" / "live" / "experiment_b_openrouter.toml"
)


class ExperimentBModelSuiteTests(unittest.TestCase):
    def test_cli_defaults_to_plan_and_exposes_offline_audit(self) -> None:
        with TemporaryDirectory() as directory:
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                status = cli_main(
                    [
                        "experiment-b",
                        "model-suite",
                        str(BASE_CONFIG),
                        "--output-root",
                        directory,
                    ]
                )
            plan = json.loads(stdout.getvalue())
        self.assertEqual(status, 0)
        self.assertEqual(plan["status"], "planned")
        self.assertFalse(plan["live_execution"])
        self.assertFalse(plan["credential_read"])

        args = build_parser().parse_args(
            [
                "experiment-b",
                "manipulation-audit",
                str(BASE_CONFIG),
                "audit-output",
                "--response-seeds",
                "3",
            ]
        )
        self.assertEqual(args.response_seeds, 3)
        self.assertFalse(hasattr(args, "execute_live"))

    def test_plan_freezes_primary_panel_and_targeted_secondary_arm(self) -> None:
        suite = load_experiment_b_model_suite()
        self.assertEqual(len(suite.arms), 4)
        self.assertTrue(suite.project_generated)
        self.assertEqual(suite.source_status, "project-authored-frozen-protocol")
        self.assertEqual(suite.license, "Apache-2.0")
        self.assertEqual(
            suite.expected_consumer,
            "cape-loop experiment-b model-suite",
        )
        with TemporaryDirectory() as directory:
            plan, configs = build_experiment_b_model_suite_plan(
                BASE_CONFIG,
                output_root=directory,
            )

        self.assertEqual(plan["status"], "planned")
        self.assertFalse(plan["live_execution"])
        self.assertFalse(plan["credential_read"])
        self.assertEqual(
            [record["model"] for record in plan["arms"]],
            [
                "google/gemini-3.6-flash",
                "openai/gpt-5.6-luna",
                "mistralai/mistral-large-2512",
                "deepseek/deepseek-v4-flash",
            ],
        )
        self.assertEqual(
            [record["physical_http_attempt_upper_bound"] for record in plan["arms"]],
            [636, 636, 636, 252],
        )
        self.assertEqual(plan["primary_physical_attempt_upper_bound"], 1908)
        self.assertEqual(plan["secondary_physical_attempt_upper_bound"], 252)
        self.assertEqual(plan["total_physical_attempt_upper_bound"], 2160)
        self.assertEqual(
            [config.llm.mode for config in configs],
            ["openrouter"] * 4,
        )
        self.assertEqual(
            [config.llm.reasoning_effort for config in configs],
            ["minimal", "low", "", ""],
        )
        self.assertEqual(
            [config.llm.openrouter_upstream_provider for config in configs],
            ["google-vertex/global", "", "", ""],
        )
        base = load_config(BASE_CONFIG)
        for config in configs[:3]:
            self.assertEqual(config.experiment, base.experiment)
        targeted = configs[3]
        self.assertEqual(
            targeted.experiment.initial_profile_conditions,
            ("incorrect",),
        )
        self.assertEqual(
            targeted.experiment.policies,
            ("balanced", "soft_profile_conditioned"),
        )
        self.assertFalse(plan["arms"][3]["primary_analysis_eligible"])
        self.assertTrue(
            plan["arms"][3]["secondary_never_pooled_with_primary"]
        )
        self.assertEqual(
            len({record["run_name"] for record in plan["arms"]}),
            4,
        )
        self.assertEqual(
            len({record["run_directory"] for record in plan["arms"]}),
            4,
        )

    def test_loader_rejects_a_primary_secondary_boundary_change(self) -> None:
        payload = json.loads(DEFAULT_SUITE_PATH.read_text(encoding="utf-8"))
        payload["analysis_policy"][
            "secondary_may_be_pooled_with_primary"
        ] = True
        with TemporaryDirectory() as directory:
            changed = Path(directory) / "changed-suite.json"
            changed.write_text(
                json.dumps(payload),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError,
                "primary/secondary boundary",
            ):
                load_experiment_b_model_suite(changed)

    def test_orchestrator_requires_explicit_execution_and_runs_sequentially(
        self,
    ) -> None:
        calls: list[dict[str, object]] = []

        def fake_run(
            config: AppConfig,
            **kwargs: object,
        ) -> dict[str, object]:
            calls.append({"config": config, **kwargs})
            run_id = f"{config.run.name}-{config_digest(config)[:12]}"
            return {
                "run_dir": str(Path(str(kwargs["output_root"])) / run_id),
                "reused": False,
                "summary": {"status": "fixture"},
            }

        with TemporaryDirectory() as directory:
            with patch(
                "cape_loop.runner.run_experiment",
                side_effect=fake_run,
            ) as run_mock:
                planned = orchestrate_experiment_b_model_suite(
                    BASE_CONFIG,
                    output_root=directory,
                    execute_live=False,
                )
                run_mock.assert_not_called()
                completed = orchestrate_experiment_b_model_suite(
                    BASE_CONFIG,
                    output_root=directory,
                    execute_live=True,
                )

        self.assertEqual(planned["status"], "planned")
        self.assertEqual(completed["status"], "complete")
        self.assertTrue(completed["live_execution"])
        self.assertEqual(len(calls), 4)
        self.assertEqual(
            [
                call["config"].llm.model  # type: ignore[index,union-attr]
                for call in calls
            ],
            [
                "google/gemini-3.6-flash",
                "openai/gpt-5.6-luna",
                "mistralai/mistral-large-2512",
                "deepseek/deepseek-v4-flash",
            ],
        )
        self.assertTrue(all(call["execute_live"] is True for call in calls))
        self.assertTrue(all(call["source_config"] is None for call in calls))
        self.assertEqual(
            [record["execution_status"] for record in completed["arms"]],
            ["complete"] * 4,
        )


if __name__ == "__main__":
    unittest.main()
