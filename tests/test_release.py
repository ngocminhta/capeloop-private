from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib
import json
import unittest

from cape_loop.artifacts import RunArtifacts
from cape_loop.artifacts import config_digest
from cape_loop.config import AppConfig, RunSection
from cape_loop.release import freeze_run, verify_frozen_artifact


class ReleaseArtifactTests(unittest.TestCase):
    @staticmethod
    def _source_toml(config: AppConfig) -> str:
        return (
            "schema_version = 1\n\n"
            "[run]\n"
            f"name = {json.dumps(config.run.name)}\n"
            f"seed = {config.run.seed}\n"
            f"output_root = {json.dumps(config.run.output_root)}\n"
            f"deterministic = {str(config.run.deterministic).lower()}\n"
        )

    def test_verified_run_freezes_deterministically(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = AppConfig(
                run=RunSection(
                    name="freeze-source",
                    seed=7,
                    output_root=str(root / "runs"),
                )
            )
            run = RunArtifacts.create(config, root=root / "runs")
            run.write_json("metrics/example.json", {"schema_version": 1, "value": 3})
            run.finalize({"status": "test"})

            first = freeze_run(run.path, root / "first.tar")
            second = freeze_run(run.path, root / "second.tar")
            self.assertEqual(first.archive_sha256, second.archive_sha256)
            self.assertEqual(
                (root / "first.tar").read_bytes(),
                (root / "second.tar").read_bytes(),
            )
            valid, errors = verify_frozen_artifact(root / "first.tar")
            self.assertTrue(valid, errors)

            payload = bytearray((root / "first.tar").read_bytes())
            payload[-1] ^= 1
            (root / "first.tar").write_bytes(payload)
            valid, errors = verify_frozen_artifact(root / "first.tar")
            self.assertFalse(valid)
            self.assertIn("frozen archive SHA-256 mismatch", errors)

    def test_freeze_requires_a_traceable_config_origin(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = AppConfig(
                run=RunSection(
                    name="untraceable-config",
                    seed=7,
                    output_root=str(root / "runs"),
                )
            )
            run = RunArtifacts.create(config, root=root / "runs")
            manifest_path = run.path / "manifest.json"
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            manifest.pop("config_origin")
            run.write_json("manifest.json", manifest)
            run.finalize({"status": "test"})
            with self.assertRaisesRegex(
                ValueError,
                "lacks config.source.toml",
            ):
                freeze_run(run.path, root / "untraceable.tar")

    def test_retained_toml_source_satisfies_the_freeze_gate(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = AppConfig(
                run=RunSection(
                    name="retained-source",
                    seed=7,
                    output_root=str(root / "runs"),
                )
            )
            run = RunArtifacts.create(
                config,
                root=root / "runs",
                config_origin={
                    "kind": "toml_file",
                    "retained_file": "config.source.toml",
                    "source_filename": "retained.toml",
                    "source_sha256": hashlib.sha256(
                        self._source_toml(config).encode("utf-8")
                    ).hexdigest(),
                    "config_sha256": config_digest(config),
                },
            )
            run.write_text(
                "config.source.toml",
                self._source_toml(config),
            )
            run.finalize({"status": "test"})
            frozen = freeze_run(run.path, root / "retained.tar")
            self.assertTrue(frozen.archive_path.is_file())

    def test_mismatched_retained_toml_cannot_be_verified_or_frozen(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = AppConfig(
                run=RunSection(
                    name="source-mismatch",
                    seed=7,
                    output_root=str(root / "runs"),
                )
            )
            run = RunArtifacts.create(config, root=root / "runs")
            different = AppConfig(
                run=RunSection(
                    name="different-source",
                    seed=7,
                    output_root=str(root / "runs"),
                )
            )
            run.write_text(
                "config.source.toml",
                self._source_toml(different),
            )
            run.finalize({"status": "test"})
            from cape_loop.artifacts import verify_run

            valid, errors = verify_run(run.path)
            self.assertFalse(valid)
            self.assertTrue(
                any(
                    "source TOML config digest" in error
                    for error in errors
                ),
                errors,
            )
            with self.assertRaisesRegex(
                ValueError,
                "source TOML config digest",
            ):
                freeze_run(run.path, root / "mismatch.tar")


if __name__ == "__main__":
    unittest.main()
