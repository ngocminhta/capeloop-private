"""Deterministic freezing and verification of paper-facing run artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any, Mapping
import json
import os
import tarfile
import tempfile
import tomllib

from .artifacts import (
    _validated_config_origin,
    retained_config_digest,
    verify_run,
)
from .config import AppConfig


_MAX_CONTROL_MEMBER_BYTES = 32 * 1024 * 1024
_MAX_FROZEN_MANIFEST_BYTES = 32 * 1024 * 1024


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


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_member_bytes(
    bundle: tarfile.TarFile,
    member: tarfile.TarInfo,
    *,
    name: str,
) -> bytes:
    if member.size > _MAX_CONTROL_MEMBER_BYTES:
        raise ValueError(
            f"{name} exceeds {_MAX_CONTROL_MEMBER_BYTES} bytes"
        )
    extracted = bundle.extractfile(member)
    if extracted is None:
        raise ValueError(f"{name} is unreadable")
    return extracted.read()


def _member_sha256(
    bundle: tarfile.TarFile,
    member: tarfile.TarInfo,
) -> str:
    extracted = bundle.extractfile(member)
    if extracted is None:
        raise ValueError(f"archive member is unreadable: {member.name}")
    digest = sha256()
    for block in iter(lambda: extracted.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def _read_frozen_manifest_bytes(path: Path) -> bytes:
    with path.open("rb") as handle:
        payload = handle.read(_MAX_FROZEN_MANIFEST_BYTES + 1)
    if len(payload) > _MAX_FROZEN_MANIFEST_BYTES:
        raise ValueError(
            "frozen manifest exceeds "
            f"{_MAX_FROZEN_MANIFEST_BYTES} bytes"
        )
    return payload


def _expected_frozen_pax_headers(
    member: tarfile.TarInfo,
) -> dict[str, str]:
    """Return only the PAX fields required by the canonical writer."""

    expected: dict[str, str] = {}
    try:
        member.name.encode("ascii", "strict")
    except UnicodeEncodeError:
        expected["path"] = member.name
    else:
        if len(member.name) > tarfile.LENGTH_NAME:
            expected["path"] = member.name
    if not 0 <= member.size < 8 ** (12 - 1):
        expected["size"] = str(member.size)
    return expected


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
    resolved_destinations = (
        archive.resolve(),
        sidecar.resolve(),
    )
    if any(
        destination == source or source in destination.parents
        for destination in resolved_destinations
    ):
        raise ValueError(
            "frozen archive and sidecar destinations must remain outside "
            "the immutable source run"
        )
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
        sorted(
            (
                path
                for path in source.rglob("*")
                if path.is_file() and not path.is_symlink()
            ),
            key=lambda path: path.relative_to(source).as_posix(),
        )
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
                info = tarfile.TarInfo(
                    name=str(PurePosixPath(source.name) / PurePosixPath(relative))
                )
                info.size = path.stat().st_size
                info.mode = 0o644
                info.mtime = 0
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                with path.open("rb") as handle:
                    bundle.addfile(info, handle)
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
        sidecar_payload = _canonical_json(manifest)
        if len(sidecar_payload) > _MAX_FROZEN_MANIFEST_BYTES:
            raise ValueError(
                "frozen manifest exceeds "
                f"{_MAX_FROZEN_MANIFEST_BYTES} bytes"
            )
        sidecar_descriptor, sidecar_temporary_name = tempfile.mkstemp(
            prefix=f".{sidecar.name}.",
            suffix=".tmp",
            dir=sidecar.parent,
        )
        os.close(sidecar_descriptor)
        sidecar_temporary = Path(sidecar_temporary_name)
        try:
            sidecar_temporary.write_bytes(sidecar_payload)
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
    """Verify a frozen run without extracting untrusted archive members."""

    archive = Path(archive_path)
    sidecar = frozen_manifest_path(archive)
    errors: list[str] = []
    try:
        decoded_manifest = json.loads(
            _read_frozen_manifest_bytes(sidecar).decode("utf-8")
        )
    except (
        OSError,
        UnicodeDecodeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        return False, (f"cannot read frozen manifest: {exc}",)
    if not isinstance(decoded_manifest, Mapping):
        return False, ("frozen manifest must be an object",)
    manifest = decoded_manifest
    if manifest.get("schema_version") != 1:
        errors.append("frozen manifest schema_version must be 1")
    if manifest.get("artifact_kind") != "cape-loop-verified-run-tar":
        errors.append("frozen manifest artifact_kind mismatch")
    if manifest.get("archive_file") != archive.name:
        errors.append("frozen manifest archive_file mismatch")
    if manifest.get("compression") != "none":
        errors.append("frozen manifest compression mismatch")
    if manifest.get("deterministic_tar_metadata") is not True:
        errors.append("frozen manifest deterministic metadata flag mismatch")
    source_directory = manifest.get("source_run_directory")
    source_directory_valid = (
        isinstance(source_directory, str)
        and bool(source_directory)
        and "\\" not in source_directory
        and "\x00" not in source_directory
        and PurePosixPath(source_directory).parts == (source_directory,)
        and source_directory not in {".", ".."}
    )
    if not source_directory_valid:
        errors.append("frozen manifest source_run_directory is unsafe")
    source_run_id = manifest.get("source_run_id")
    if not isinstance(source_run_id, str) or not source_run_id:
        errors.append("frozen manifest source_run_id is invalid")
    elif source_directory_valid and source_run_id != source_directory:
        errors.append("frozen manifest source run identity mismatch")
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
            if names != sorted(names):
                errors.append(
                    "frozen archive members are not in canonical order"
                )
            members_by_name: dict[str, tarfile.TarInfo] = {}
            for member in members:
                path = PurePosixPath(member.name)
                if (
                    path.is_absolute()
                    or "\\" in member.name
                    or "\x00" in member.name
                    or ".." in path.parts
                    or len(path.parts) < 2
                    or path.as_posix() != member.name
                ):
                    errors.append(f"unsafe frozen member path: {member.name}")
                    continue
                if source_directory_valid and path.parts[0] != source_directory:
                    errors.append(
                        "frozen archive member is outside the source run: "
                        f"{member.name}"
                    )
                if not member.isfile():
                    errors.append(
                        f"frozen archive member is not a regular file: {member.name}"
                    )
                expected_pax_headers = _expected_frozen_pax_headers(member)
                if member.pax_headers != expected_pax_headers:
                    errors.append(
                        "unexpected PAX metadata for member: "
                        f"{member.name}"
                    )
                if (
                    member.mtime != 0
                    or member.uid != 0
                    or member.gid != 0
                    or member.mode != 0o644
                    or member.uname != ""
                    or member.gname != ""
                ):
                    errors.append(
                        f"non-deterministic metadata for member: {member.name}"
                    )
                if member.name not in members_by_name:
                    members_by_name[member.name] = member
            file_count = manifest.get("file_count")
            if (
                isinstance(file_count, bool)
                or not isinstance(file_count, int)
                or file_count != len(members)
            ):
                errors.append("frozen archive file_count mismatch")

            if not source_directory_valid:
                return False, tuple(errors)
            source_prefix = f"{source_directory}/"
            embedded_manifest_name = source_prefix + "manifest.json"
            checksums_name = source_prefix + "SHA256SUMS"
            resolved_name = (
                source_prefix + "config.resolved.json"
            )
            summary_name = source_prefix + "metrics/summary.json"
            if summary_name not in members_by_name:
                errors.append(
                    "frozen archive is missing metrics/summary.json"
                )

            embedded_manifest_bytes: bytes | None = None
            embedded_manifest: Mapping[str, Any] | None = None
            embedded_manifest_member = members_by_name.get(
                embedded_manifest_name
            )
            if embedded_manifest_member is None:
                errors.append("frozen archive is missing manifest.json")
            else:
                try:
                    embedded_manifest_bytes = _read_member_bytes(
                        bundle,
                        embedded_manifest_member,
                        name="embedded run manifest",
                    )
                    parsed_manifest = json.loads(
                        embedded_manifest_bytes.decode("utf-8")
                    )
                    if not isinstance(parsed_manifest, Mapping):
                        raise TypeError(
                            "embedded run manifest must be an object"
                        )
                    embedded_manifest = parsed_manifest
                except (
                    TypeError,
                    UnicodeDecodeError,
                    ValueError,
                    json.JSONDecodeError,
                ) as exc:
                    errors.append(
                        f"cannot validate embedded run manifest: {exc}"
                    )
            expected_manifest_sha256 = manifest.get(
                "source_manifest_sha256"
            )
            if not _is_sha256(expected_manifest_sha256):
                errors.append(
                    "frozen manifest source_manifest_sha256 is invalid"
                )
            elif (
                embedded_manifest_bytes is not None
                and sha256(embedded_manifest_bytes).hexdigest()
                != expected_manifest_sha256
            ):
                errors.append("frozen embedded manifest SHA-256 mismatch")

            checksums_bytes: bytes | None = None
            checksums_member = members_by_name.get(checksums_name)
            if checksums_member is None:
                errors.append("frozen archive is missing SHA256SUMS")
            else:
                try:
                    checksums_bytes = _read_member_bytes(
                        bundle,
                        checksums_member,
                        name="embedded SHA256SUMS",
                    )
                except ValueError as exc:
                    errors.append(
                        f"cannot validate embedded SHA256SUMS: {exc}"
                    )
            expected_checksums_sha256 = manifest.get(
                "source_checksums_sha256"
            )
            if not _is_sha256(expected_checksums_sha256):
                errors.append(
                    "frozen manifest source_checksums_sha256 is invalid"
                )
            elif (
                checksums_bytes is not None
                and sha256(checksums_bytes).hexdigest()
                != expected_checksums_sha256
            ):
                errors.append("frozen embedded SHA256SUMS digest mismatch")

            if checksums_bytes is not None:
                retained_paths: set[str] = set()
                try:
                    checksum_lines = checksums_bytes.decode(
                        "utf-8"
                    ).splitlines()
                except UnicodeDecodeError as exc:
                    errors.append(
                        f"cannot decode embedded SHA256SUMS: {exc}"
                    )
                    checksum_lines = []
                for line_number, line in enumerate(
                    checksum_lines,
                    start=1,
                ):
                    if not line.strip():
                        continue
                    try:
                        expected, relative = line.split("  ", 1)
                    except ValueError:
                        errors.append(
                            "malformed embedded checksum line "
                            f"{line_number}"
                        )
                        continue
                    relative_path = PurePosixPath(relative)
                    if not _is_sha256(expected):
                        errors.append(
                            "invalid embedded SHA-256 on line "
                            f"{line_number}"
                        )
                        continue
                    if (
                        not relative
                        or relative_path.is_absolute()
                        or "\\" in relative
                        or "\x00" in relative
                        or any(
                            part in {"", ".", ".."}
                            for part in relative_path.parts
                        )
                        or relative_path.as_posix() != relative
                        or relative == "SHA256SUMS"
                    ):
                        errors.append(
                            "unsafe embedded checksum path on line "
                            f"{line_number}"
                        )
                        continue
                    if relative in retained_paths:
                        errors.append(
                            "duplicate embedded checksum path on line "
                            f"{line_number}"
                        )
                        continue
                    retained_paths.add(relative)
                    retained_member = members_by_name.get(
                        source_prefix + relative
                    )
                    if retained_member is None:
                        errors.append(
                            f"missing frozen member listed by checksum: "
                            f"{relative}"
                        )
                        continue
                    try:
                        actual = _member_sha256(
                            bundle,
                            retained_member,
                        )
                    except ValueError as exc:
                        errors.append(str(exc))
                        continue
                    if actual != expected:
                        errors.append(
                            f"frozen member checksum mismatch: {relative}"
                        )
                actual_paths = {
                    name.removeprefix(source_prefix)
                    for name in members_by_name
                    if name.startswith(source_prefix)
                    and name != checksums_name
                }
                for relative in sorted(actual_paths - retained_paths):
                    errors.append(
                        f"unlisted frozen run artifact: {relative}"
                    )

            resolved_config: AppConfig | None = None
            resolved_raw: Mapping[str, Any] | None = None
            retained_digest: str | None = None
            resolved_member = members_by_name.get(resolved_name)
            if resolved_member is None:
                errors.append(
                    "frozen archive is missing config.resolved.json"
                )
            else:
                try:
                    parsed_resolved = json.loads(
                        _read_member_bytes(
                            bundle,
                            resolved_member,
                            name="resolved config member",
                        ).decode("utf-8")
                    )
                    if not isinstance(parsed_resolved, Mapping):
                        raise TypeError(
                            "resolved config must be an object"
                        )
                    resolved_raw = parsed_resolved
                    resolved_config = AppConfig.parse(resolved_raw)
                    retained_digest = retained_config_digest(resolved_raw)
                    if retained_digest != manifest.get(
                        "source_config_resolved_sha256"
                    ):
                        errors.append(
                            "frozen resolved config digest mismatch"
                        )
                except (
                    KeyError,
                    TypeError,
                    UnicodeDecodeError,
                    ValueError,
                    json.JSONDecodeError,
                ) as exc:
                    errors.append(
                        f"cannot validate frozen resolved config: {exc}"
                    )

            embedded_config_sha256: Any = None
            embedded_origin: Any = None
            if embedded_manifest is not None:
                if (
                    embedded_manifest.get("schema_version") != 1
                    or embedded_manifest.get("status") != "complete"
                ):
                    errors.append(
                        "embedded run manifest identity/status is invalid"
                    )
                if (
                    embedded_manifest.get("run_id") != source_run_id
                    or embedded_manifest.get("run_id")
                    != source_directory
                ):
                    errors.append(
                        "embedded run manifest run identity mismatch"
                    )
                embedded_config_sha256 = embedded_manifest.get(
                    "config_sha256"
                )
                if not _is_sha256(embedded_config_sha256):
                    errors.append(
                        "embedded run manifest config digest is invalid"
                    )
                elif (
                    embedded_config_sha256
                    != manifest.get("source_config_resolved_sha256")
                ):
                    errors.append(
                        "embedded run manifest config digest mismatch"
                    )
                if (
                    retained_digest is not None
                    and embedded_config_sha256 != retained_digest
                ):
                    errors.append(
                        "embedded run manifest does not bind the resolved "
                        "config"
                    )
                embedded_origin = embedded_manifest.get("config_origin")
                if embedded_origin != manifest.get("config_origin"):
                    errors.append(
                        "frozen sidecar config_origin does not match the "
                        "embedded run manifest"
                    )
                if _is_sha256(embedded_config_sha256):
                    try:
                        if not isinstance(embedded_origin, Mapping):
                            raise ValueError(
                                "embedded config_origin must be an object"
                            )
                        _validated_config_origin(
                            embedded_origin,
                            expected_config_sha256=embedded_config_sha256,
                        )
                    except ValueError as exc:
                        errors.append(
                            f"invalid embedded config_origin: {exc}"
                        )

            retained_source_name = (
                source_prefix + "config.source.toml"
            )
            source_member = members_by_name.get(retained_source_name)
            source_retained = source_member is not None
            if (
                not isinstance(
                    manifest.get("config_source_retained"),
                    bool,
                )
                or manifest.get("config_source_retained")
                is not source_retained
            ):
                errors.append(
                    "frozen manifest config-source retention mismatch"
                )
            if not source_retained:
                if not (
                    isinstance(embedded_origin, Mapping)
                    and embedded_origin.get("kind") == "programmatic"
                ):
                    errors.append(
                        "frozen artifact lacks a retained source config or "
                        "complete programmatic config origin"
                    )
            else:
                try:
                    assert source_member is not None
                    source_bytes = _read_member_bytes(
                        bundle,
                        source_member,
                        name="source config member",
                    )
                    if not (
                        isinstance(embedded_origin, Mapping)
                        and embedded_origin.get("kind") == "toml_file"
                    ):
                        errors.append(
                            "retained source TOML lacks a TOML config origin"
                        )
                    elif (
                        embedded_origin.get("source_sha256")
                        != sha256(source_bytes).hexdigest()
                    ):
                        errors.append(
                            "retained source TOML SHA-256 mismatch"
                        )
                    source_raw = tomllib.loads(
                        source_bytes.decode("utf-8")
                    )
                    source_config = AppConfig.parse(source_raw)
                    if (
                        resolved_config is not None
                        and source_config.to_dict()
                        != resolved_config.to_dict()
                    ):
                        errors.append(
                            "frozen source TOML does not resolve to the "
                            "retained config"
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
