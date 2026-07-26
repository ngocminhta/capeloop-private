from __future__ import annotations

from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import io
import json
import unittest

from cape_loop.artifacts import config_digest
from cape_loop.cli import main as cli_main
from cape_loop.config import AppConfig
from cape_loop.evaluation_suite import (
    orchestrate_openai_evaluation_suite,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PRIMARY = REPOSITORY_ROOT / "configs" / "openai_primary.toml"
REPLICATION = REPOSITORY_ROOT / "configs" / "openai_replication.toml"


class OpenAIEvaluationSuiteTests(unittest.TestCase):
    def test_default_command_is_a_credential_free_plan(self) -> None:
        with TemporaryDirectory() as directory:
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                status = cli_main(
                    [
                        "llm",
                        "evaluation-suite",
                        str(PRIMARY),
                        str(REPLICATION),
                        "--output-root",
                        directory,
                    ]
                )
            self.assertEqual(status, 0)
            index = json.loads(stdout.getvalue())
            self.assertEqual(index["status"], "planned")
            self.assertFalse(index["live_execution"])
            self.assertFalse(index["credential_read"])
            self.assertFalse(
                index["distinct_model_family_robustness_claimed"]
            )
            self.assertEqual(
                [role["role"] for role in index["roles"]],
                ["primary", "replication"],
            )
            self.assertEqual(
                len({role["run_id"] for role in index["roles"]}),
                2,
            )
            self.assertEqual(
                {
                    role["conservative_request_upper_bound"]
                    for role in index["roles"]
                },
                {752},
            )
            self.assertTrue(
                all(role["request_headroom"] == 148 for role in index["roles"])
            )
            self.assertEqual(
                len(
                    {
                        role["journal_directory"]
                        for role in index["roles"]
                    }
                ),
                2,
            )
            self.assertTrue(Path(index["index_path"]).is_file())

            with patch(
                "cape_loop.openai_provider.urlopen",
                side_effect=AssertionError("suite plan attempted network"),
            ):
                keyless = orchestrate_openai_evaluation_suite(
                    PRIMARY,
                    REPLICATION,
                    output_root=Path(directory) / "keyless",
                )
            self.assertFalse(keyless["credential_read"])

    def test_index_cannot_be_repointed_to_a_different_suite(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            index_path = root / "suite-index.json"
            first = orchestrate_openai_evaluation_suite(
                PRIMARY,
                REPLICATION,
                output_root=root / "one",
                index_path=index_path,
            )
            self.assertEqual(first["status"], "planned")
            with self.assertRaisesRegex(
                ValueError,
                "different config/run identities",
            ):
                orchestrate_openai_evaluation_suite(
                    PRIMARY,
                    REPLICATION,
                    output_root=root / "two",
                    index_path=index_path,
                )

    def test_explicit_execution_uses_each_immutable_config_and_run(self) -> None:
        calls: list[dict[str, object]] = []

        def fake_run(
            config: AppConfig,
            **kwargs: object,
        ) -> dict[str, object]:
            calls.append({"config": config, **kwargs})
            digest = config_digest(config)
            run_name = config.run.name
            run_dir = Path(str(kwargs["output_root"])) / (
                f"{run_name}-{digest[:12]}"
            )
            return {
                "run_dir": str(run_dir),
                "reused": False,
                "summary": {"status": "fixture"},
            }

        with TemporaryDirectory() as directory:
            with patch(
                "cape_loop.runner.run_experiment",
                side_effect=fake_run,
            ):
                index = orchestrate_openai_evaluation_suite(
                    PRIMARY,
                    REPLICATION,
                    output_root=directory,
                    execute_live=True,
                )
            self.assertEqual(index["status"], "complete")
            self.assertEqual(len(calls), 2)
            self.assertTrue(
                all(call["execute_live"] is True for call in calls)
            )
            self.assertEqual(
                [call["config"].llm.model_role for call in calls],  # type: ignore[index,union-attr]
                ["primary", "replication"],
            )
            self.assertEqual(
                [call["config"].llm.max_requests for call in calls],  # type: ignore[index,union-attr]
                [900, 900],
            )
            self.assertEqual(
                len(
                    {
                        role["run_directory"]
                        for role in index["roles"]
                    }
                ),
                2,
            )


if __name__ == "__main__":
    unittest.main()
