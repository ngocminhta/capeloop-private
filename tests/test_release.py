from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, Mapping
import hashlib
import io
import json
import tarfile
import unittest
from unittest.mock import patch

from cape_loop.artifacts import RunArtifacts
from cape_loop.artifacts import canonical_json
from cape_loop.artifacts import config_digest
from cape_loop.artifacts import retained_config_digest
from cape_loop.artifacts import verify_run
from cape_loop.config import AppConfig, RunSection
from cape_loop.release import (
    freeze_run,
    frozen_manifest_path,
    verify_frozen_artifact,
)


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

    @staticmethod
    def _rewrite_frozen_archive(
        source: Path,
        destination: Path,
        rewrite: Callable[[str, bytes], bytes],
        *,
        reverse_members: bool = False,
        extra_pax_headers: Mapping[str, str] | None = None,
        omit_member: Callable[[str], bool] | None = None,
    ) -> Path:
        """Repack a test archive while only refreshing its outer digest."""

        payloads: list[tuple[tarfile.TarInfo, bytes]] = []
        with tarfile.open(source, mode="r:") as bundle:
            for member in bundle.getmembers():
                if omit_member is not None and omit_member(member.name):
                    continue
                extracted = bundle.extractfile(member)
                if extracted is None:
                    raise AssertionError(
                        f"test archive member is unreadable: {member.name}"
                    )
                payloads.append(
                    (member, rewrite(member.name, extracted.read()))
                )
        with tarfile.open(
            destination,
            mode="w",
            format=tarfile.PAX_FORMAT,
        ) as bundle:
            ordered_payloads = (
                reversed(payloads) if reverse_members else payloads
            )
            for original, payload in ordered_payloads:
                member = tarfile.TarInfo(original.name)
                member.size = len(payload)
                member.mode = original.mode
                member.mtime = original.mtime
                member.uid = original.uid
                member.gid = original.gid
                member.uname = original.uname
                member.gname = original.gname
                member.pax_headers.update(extra_pax_headers or {})
                bundle.addfile(member, io.BytesIO(payload))

        source_sidecar = frozen_manifest_path(source)
        destination_sidecar = frozen_manifest_path(destination)
        manifest = json.loads(source_sidecar.read_text(encoding="utf-8"))
        manifest["archive_file"] = destination.name
        manifest["archive_sha256"] = hashlib.sha256(
            destination.read_bytes()
        ).hexdigest()
        manifest["file_count"] = len(payloads)
        checksums_payload = next(
            payload
            for member, payload in payloads
            if member.name.endswith("/SHA256SUMS")
        )
        manifest["source_checksums_sha256"] = hashlib.sha256(
            checksums_payload
        ).hexdigest()
        destination_sidecar.write_text(
            canonical_json(manifest) + "\n",
            encoding="utf-8",
        )
        return destination

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

    def test_freeze_destination_cannot_mutate_the_source_run(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = AppConfig(
                run=RunSection(
                    name="immutable-freeze-source",
                    seed=7,
                    output_root=str(root / "runs"),
                )
            )
            run = RunArtifacts.create(config, root=root / "runs")
            run.finalize({"status": "test"})
            before = tuple(
                sorted(
                    path.relative_to(run.path).as_posix()
                    for path in run.path.rglob("*")
                )
            )
            archive = run.path / "nested" / "paper.tar"

            with self.assertRaisesRegex(
                ValueError,
                "must remain outside the immutable source run",
            ):
                freeze_run(run.path, archive)

            self.assertFalse(archive.parent.exists())
            after = tuple(
                sorted(
                    path.relative_to(run.path).as_posix()
                    for path in run.path.rglob("*")
                )
            )
            self.assertEqual(after, before)
            valid, errors = verify_run(run.path)
            self.assertTrue(valid, errors)

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

    def test_source_presence_and_config_origin_must_match_before_freeze(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = AppConfig(
                run=RunSection(
                    name="source-origin-mismatch",
                    seed=7,
                    output_root=str(root / "runs"),
                )
            )
            run = RunArtifacts.create(config, root=root / "runs")
            run.write_text(
                "config.source.toml",
                self._source_toml(config),
            )
            run.finalize({"status": "test"})

            valid, errors = verify_run(run.path)
            self.assertFalse(valid)
            self.assertIn(
                "config.source.toml requires a TOML config origin",
                errors,
            )
            archive = root / "mismatched-origin.tar"
            with self.assertRaisesRegex(
                ValueError,
                "config.source.toml requires a TOML config origin",
            ):
                freeze_run(run.path, archive)
            self.assertFalse(archive.exists())
            self.assertFalse(frozen_manifest_path(archive).exists())

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
                    "source TOML config does not resolve" in error
                    for error in errors
                ),
                errors,
            )
            with self.assertRaisesRegex(
                ValueError,
                "source TOML config does not resolve",
            ):
                freeze_run(run.path, root / "mismatch.tar")

    def test_historical_resolved_defaults_remain_verifiable(self) -> None:
        """New parser defaults must not invalidate immutable older runs."""

        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = AppConfig(
                run=RunSection(
                    name="historical-caf\u00e9-default",
                    seed=7,
                    output_root=str(root / "runs"),
                )
            )
            source_toml = self._source_toml(config)
            run = RunArtifacts.create(
                config,
                root=root / "runs",
                config_origin={
                    "kind": "toml_file",
                    "retained_file": "config.source.toml",
                    "source_filename": "historical.toml",
                    "source_sha256": hashlib.sha256(
                        source_toml.encode("utf-8")
                    ).hexdigest(),
                    "config_sha256": config_digest(config),
                },
            )
            run.write_text("config.source.toml", source_toml)
            run.finalize({"status": "test"})

            resolved_path = run.path / "config.resolved.json"
            resolved = json.loads(
                resolved_path.read_text(encoding="utf-8")
            )
            self.assertEqual(
                retained_config_digest(resolved),
                config_digest(config),
            )
            resolved["sensitivity"].pop("design")
            historical_digest = retained_config_digest(resolved)
            resolved_path.write_text(
                canonical_json(resolved) + "\n",
                encoding="utf-8",
            )

            manifest_path = run.path / "manifest.json"
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            manifest["config_sha256"] = historical_digest
            manifest["config_origin"]["config_sha256"] = historical_digest
            run.write_json("manifest.json", manifest)
            run.write_checksums()

            valid, errors = verify_run(run.path)
            self.assertTrue(valid, errors)
            frozen = freeze_run(run.path, root / "historical.tar")
            valid, errors = verify_frozen_artifact(frozen.archive_path)
            self.assertTrue(valid, errors)

    def test_malformed_resolved_root_is_invalid_instead_of_raising(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = AppConfig(
                run=RunSection(
                    name="malformed-resolved-root",
                    seed=7,
                    output_root=str(root / "runs"),
                )
            )
            run = RunArtifacts.create(config, root=root / "runs")
            run.finalize({"status": "test"})

            resolved_path = run.path / "config.resolved.json"
            resolved_path.write_text("[]\n", encoding="utf-8")
            run.write_checksums()
            valid, errors = verify_run(run.path)
            self.assertFalse(valid)
            self.assertIn(
                "invalid config.resolved.json: "
                "resolved config must be an object",
                errors,
            )

            resolved_path.write_text(
                canonical_json(config.to_dict()) + "\n",
                encoding="utf-8",
            )
            run.write_checksums()
            frozen = freeze_run(run.path, root / "original.tar")
            tampered = self._rewrite_frozen_archive(
                frozen.archive_path,
                root / "malformed.tar",
                lambda name, payload: (
                    b"[]\n"
                    if name.endswith("/config.resolved.json")
                    else payload
                ),
            )
            valid, errors = verify_frozen_artifact(tampered)
            self.assertFalse(valid)
            self.assertIn(
                "cannot validate frozen resolved config: "
                "resolved config must be an object",
                errors,
            )

    def test_frozen_verifier_rejects_stale_source_and_manifest_bindings(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = AppConfig(
                run=RunSection(
                    name="binding-hardening",
                    seed=7,
                    output_root=str(root / "runs"),
                )
            )
            source_toml = self._source_toml(config)
            run = RunArtifacts.create(
                config,
                root=root / "runs",
                config_origin={
                    "kind": "toml_file",
                    "retained_file": "config.source.toml",
                    "source_filename": "binding.toml",
                    "source_sha256": hashlib.sha256(
                        source_toml.encode("utf-8")
                    ).hexdigest(),
                    "config_sha256": config_digest(config),
                },
            )
            run.write_text("config.source.toml", source_toml)
            run.finalize({"status": "test"})
            frozen = freeze_run(run.path, root / "original.tar")

            stale_source = self._rewrite_frozen_archive(
                frozen.archive_path,
                root / "stale-source.tar",
                lambda name, payload: (
                    payload + b"# digest-matched outer archive only\n"
                    if name.endswith("/config.source.toml")
                    else payload
                ),
            )
            valid, errors = verify_frozen_artifact(stale_source)
            self.assertFalse(valid)
            self.assertIn(
                "frozen member checksum mismatch: config.source.toml",
                errors,
            )
            self.assertIn(
                "retained source TOML SHA-256 mismatch",
                errors,
            )

            def rewrite_manifest(name: str, payload: bytes) -> bytes:
                if not name.endswith("/manifest.json"):
                    return payload
                embedded = json.loads(payload.decode("utf-8"))
                embedded["config_sha256"] = "0" * 64
                return (canonical_json(embedded) + "\n").encode("utf-8")

            stale_manifest = self._rewrite_frozen_archive(
                frozen.archive_path,
                root / "stale-manifest.tar",
                rewrite_manifest,
            )
            valid, errors = verify_frozen_artifact(stale_manifest)
            self.assertFalse(valid)
            self.assertIn(
                "frozen embedded manifest SHA-256 mismatch",
                errors,
            )
            self.assertIn(
                "embedded run manifest config digest mismatch",
                errors,
            )

    def test_frozen_verifier_requires_canonical_order_and_pax_metadata(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = AppConfig(
                run=RunSection(
                    name="canonical-tar-layout",
                    seed=7,
                    output_root=str(root / "runs"),
                )
            )
            run = RunArtifacts.create(config, root=root / "runs")
            run.write_text("a.txt", "a")
            run.write_text("z.txt", "z")
            run.finalize({"status": "test"})
            frozen = freeze_run(run.path, root / "original.tar")

            reversed_archive = self._rewrite_frozen_archive(
                frozen.archive_path,
                root / "reversed.tar",
                lambda _name, payload: payload,
                reverse_members=True,
            )
            valid, errors = verify_frozen_artifact(reversed_archive)
            self.assertFalse(valid)
            self.assertIn(
                "frozen archive members are not in canonical order",
                errors,
            )

            unexpected_pax = self._rewrite_frozen_archive(
                frozen.archive_path,
                root / "unexpected-pax.tar",
                lambda _name, payload: payload,
                extra_pax_headers={"comment": "noncanonical metadata"},
            )
            valid, errors = verify_frozen_artifact(unexpected_pax)
            self.assertFalse(valid)
            self.assertTrue(
                any(
                    error.startswith("unexpected PAX metadata for member:")
                    for error in errors
                ),
                errors,
            )

    def test_frozen_verifier_bounds_sidecar_before_json_decode(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = AppConfig(
                run=RunSection(
                    name="bounded-sidecar",
                    seed=7,
                    output_root=str(root / "runs"),
                )
            )
            run = RunArtifacts.create(config, root=root / "runs")
            run.finalize({"status": "test"})
            frozen = freeze_run(run.path, root / "bounded.tar")
            frozen.manifest_path.write_bytes(b"{" + (b" " * 32))

            with patch(
                "cape_loop.release._MAX_FROZEN_MANIFEST_BYTES",
                16,
            ):
                valid, errors = verify_frozen_artifact(
                    frozen.archive_path
                )
            self.assertFalse(valid)
            self.assertEqual(
                errors,
                ("cannot read frozen manifest: "
                 "frozen manifest exceeds 16 bytes",),
            )

    def test_frozen_verifier_requires_embedded_run_summary(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = AppConfig(
                run=RunSection(
                    name="summary-required",
                    seed=7,
                    output_root=str(root / "runs"),
                )
            )
            run = RunArtifacts.create(config, root=root / "runs")
            run.finalize({"status": "test"})
            frozen = freeze_run(run.path, root / "original.tar")

            def rewrite_checksums(name: str, payload: bytes) -> bytes:
                if not name.endswith("/SHA256SUMS"):
                    return payload
                retained = [
                    line
                    for line in payload.decode("utf-8").splitlines()
                    if not line.endswith("  metrics/summary.json")
                ]
                return ("\n".join(retained) + "\n").encode("utf-8")

            without_summary = self._rewrite_frozen_archive(
                frozen.archive_path,
                root / "without-summary.tar",
                rewrite_checksums,
                omit_member=lambda name: name.endswith(
                    "/metrics/summary.json"
                ),
            )
            valid, errors = verify_frozen_artifact(without_summary)
            self.assertFalse(valid)
            self.assertEqual(
                errors,
                ("frozen archive is missing metrics/summary.json",),
            )


if __name__ == "__main__":
    unittest.main()
