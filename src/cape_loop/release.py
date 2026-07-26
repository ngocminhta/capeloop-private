"""Deterministic freezing and verification of paper-facing run artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any
import io
import json
import os
import tarfile
import tempfile
import tomllib

from .artifacts import config_digest, verify_run
from .config import AppConfig


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frozen_manifest_path(archive_path: str | Path) -> Path:
    archive = Path(archive_path)
    return archive.with_name(archive.name + ".manifest.json")


@dataclass(frozen=True, slots=True)
class FrozenArtifact:
    archive_path: Path
    manifest_path: Path
    archive_sha256: str
    file_count: int
    source_run_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "archive_path": str(self.archive_path),
            "manifest_path": str(self.manifest_path),
            "archive_sha256": self.archive_sha256,
            "file_count": self.file_count,
            "source_run_id": self.source_run_id,
        }


def freeze_run(
    run_dir: str | Path,
    archive_path: str | Path,
) -> FrozenArtifact:
    """Freeze one complete verified run into a deterministic uncompressed tar.

    The destination and its sidecar must not already exist. Tar member metadata
    is normalized so freezing the same verified run twice yields identical
    bytes independent of local ownership and wall-clock time.
    """

    source = Path(run_dir).resolve()
    archive = Path(archive_path)
    sidecar = frozen_manifest_path(archive)
    if archive.suffix != ".tar":
        raise ValueError("frozen artifact path must end in .tar")
    if archive.exists() or sidecar.exists():
        raise FileExistsError(
            "frozen archive and sidecar destinations must be absent"
        )
    valid, errors = verify_run(source)
    if not valid:
        raise ValueError(
            "cannot freeze an unverified run: " + "; ".join(errors)
        )
    files = tuple(
        path
        for path in sorted(source.rglob("*"))
        if path.is_file() and not path.is_symlink()
    )
    if not files:
        raise ValueError("verified run contains no files")
    if any(path.is_symlink() for path in source.rglob("*")):
        raise ValueError("frozen runs cannot contain symbolic links")
    run_manifest = json.loads(
        (source / "manifest.json").read_text(encoding="utf-8")
    )
    source_run_id = str(run_manifest.get("run_id", ""))
    if not source_run_id:
        raise ValueError("run manifest does not contain run_id")
    retained_config_source = (source / "config.source.toml").is_file()
    config_origin = run_manifest.get("config_origin")
    if not retained_config_source:
        if not isinstance(config_origin, dict):
            raise ValueError(
                "paper artifact lacks config.source.toml and a recorded "
                "programmatic config origin"
            )
        if (
            config_origin.get("kind") != "programmatic"
            or not isinstance(config_origin.get("descriptor"), str)
            or not config_origin["descriptor"].strip()
            or config_origin.get("config_sha256")
            != run_manifest.get("config_sha256")
        ):
            raise ValueError(
                "paper artifact's programmatic config origin is incomplete "
                "or does not bind the resolved config"
            )

    archive.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{archive.name}.",
        suffix=".tmp",
        dir=archive.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with tarfile.open(temporary, mode="w", format=tarfile.PAX_FORMAT) as bundle:
            for path in files:
                relative = path.relative_to(source)
                data = path.read_bytes()
                info = tarfile.TarInfo(
                    name=str(PurePosixPath(source.name) / PurePosixPath(relative))
                )
                info.size = len(data)
                info.mode = 0o644
                info.mtime = 0
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                bundle.addfile(info, io.BytesIO(data))
        archive_sha256 = _sha256_file(temporary)
        manifest = {
            "schema_version": 1,
            "artifact_kind": "cape-loop-verified-run-tar",
            "archive_file": archive.name,
            "archive_sha256": archive_sha256,
            "compression": "none",
            "deterministic_tar_metadata": True,
            "source_run_directory": source.name,
            "source_run_id": source_run_id,
            "source_manifest_sha256": sha256(
                (source / "manifest.json").read_bytes()
            ).hexdigest(),
            "source_checksums_sha256": sha256(
                (source / "SHA256SUMS").read_bytes()
            ).hexdigest(),
            "source_config_resolved_sha256": run_manifest.get(
                "config_sha256"
            ),
            "config_source_retained": retained_config_source,
            "config_origin": config_origin,
            "file_count": len(files),
        }
        sidecar_descriptor, sidecar_temporary_name = tempfile.mkstemp(
            prefix=f".{sidecar.name}.",
            suffix=".tmp",
            dir=sidecar.parent,
        )
        os.close(sidecar_descriptor)
        sidecar_temporary = Path(sidecar_temporary_name)
        try:
            sidecar_temporary.write_bytes(_canonical_json(manifest))
            os.replace(temporary, archive)
            os.replace(sidecar_temporary, sidecar)
        finally:
            if sidecar_temporary.exists():
                sidecar_temporary.unlink()
    finally:
        if temporary.exists():
            temporary.unlink()
    return FrozenArtifact(
        archive_path=archive,
        manifest_path=sidecar,
        archive_sha256=archive_sha256,
        file_count=len(files),
        source_run_id=source_run_id,
    )


def verify_frozen_artifact(
    archive_path: str | Path,
) -> tuple[bool, tuple[str, ...]]:
    """Verify the sidecar digest and safe deterministic tar member inventory."""

    archive = Path(archive_path)
    sidecar = frozen_manifest_path(archive)
    errors: list[str] = []
    try:
        manifest = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, (f"cannot read frozen manifest: {exc}",)
    if manifest.get("schema_version") != 1:
        errors.append("frozen manifest schema_version must be 1")
    if manifest.get("archive_file") != archive.name:
        errors.append("frozen manifest archive_file mismatch")
    try:
        observed_sha256 = _sha256_file(archive)
    except OSError as exc:
        return False, (f"cannot read frozen archive: {exc}",)
    if manifest.get("archive_sha256") != observed_sha256:
        errors.append("frozen archive SHA-256 mismatch")
    try:
        with tarfile.open(archive, mode="r:") as bundle:
            members = bundle.getmembers()
            names = [member.name for member in members]
            if len(names) != len(set(names)):
                errors.append("frozen archive has duplicate member names")
            for member in members:
                path = PurePosixPath(member.name)
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or len(path.parts) < 2
                ):
                    errors.append(f"unsafe frozen member path: {member.name}")
                if not member.isfile():
                    errors.append(
                        f"frozen archive member is not a regular file: {member.name}"
                    )
                if (
                    member.mtime != 0
                    or member.uid != 0
                    or member.gid != 0
                    or member.mode != 0o644
                ):
                    errors.append(
                        f"non-deterministic metadata for member: {member.name}"
                    )
            if manifest.get("file_count") != len(members):
                errors.append("frozen archive file_count mismatch")
            source_directory = manifest.get("source_run_directory")
            retained_source_name = (
                f"{source_directory}/config.source.toml"
                if isinstance(source_directory, str)
                else ""
            )
            source_retained = retained_source_name in names
            if manifest.get("config_source_retained") != source_retained:
                errors.append(
                    "frozen manifest config-source retention mismatch"
                )
            if not source_retained:
                origin = manifest.get("config_origin")
                if not (
                    isinstance(origin, dict)
                    and origin.get("kind") == "programmatic"
                    and isinstance(origin.get("descriptor"), str)
                    and origin["descriptor"].strip()
                    and isinstance(origin.get("config_sha256"), str)
                    and len(origin["config_sha256"]) == 64
                    and origin.get("config_sha256")
                    == manifest.get("source_config_resolved_sha256")
                ):
                    errors.append(
                        "frozen artifact lacks a retained source config or "
                        "complete programmatic config origin"
                    )
            else:
                try:
                    source_member = bundle.getmember(retained_source_name)
                    extracted = bundle.extractfile(source_member)
                    if extracted is None:
                        raise ValueError("source config member is unreadable")
                    source_raw = tomllib.loads(
                        extracted.read().decode("utf-8")
                    )
                    source_config = AppConfig.parse(source_raw)
                    if config_digest(source_config) != manifest.get(
                        "source_config_resolved_sha256"
                    ):
                        errors.append(
                            "frozen source TOML does not bind the resolved "
                            "config digest"
                        )
                except (
                    KeyError,
                    TypeError,
                    UnicodeDecodeError,
                    ValueError,
                    tomllib.TOMLDecodeError,
                ) as exc:
                    errors.append(
                        f"cannot validate frozen source TOML: {exc}"
                    )
    except (OSError, tarfile.TarError) as exc:
        errors.append(f"cannot inspect frozen archive: {exc}")
    return not errors, tuple(errors)
