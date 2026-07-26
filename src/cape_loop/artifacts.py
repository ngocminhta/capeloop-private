"""Self-describing run artifacts and checksum verification."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping
import json
import platform
import subprocess
import sys

from .config import AppConfig, load_config


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def config_digest(config: AppConfig) -> str:
    return sha256(config.canonical_json().encode("utf-8")).hexdigest()


def _validated_config_origin(
    origin: Mapping[str, Any],
    *,
    expected_config_sha256: str,
) -> dict[str, Any]:
    payload = dict(origin)
    kind = payload.get("kind")
    if kind == "programmatic":
        required = {"kind", "descriptor", "config_sha256"}
        if set(payload) != required:
            raise ValueError(
                "programmatic config_origin fields must be exactly "
                + ", ".join(sorted(required))
            )
        if (
            not isinstance(payload.get("descriptor"), str)
            or not payload["descriptor"].strip()
        ):
            raise ValueError(
                "programmatic config_origin descriptor must be non-empty"
            )
    elif kind == "toml_file":
        required = {
            "kind",
            "retained_file",
            "source_filename",
            "source_sha256",
            "config_sha256",
        }
        if set(payload) != required:
            raise ValueError(
                "TOML config_origin fields must be exactly "
                + ", ".join(sorted(required))
            )
        if payload.get("retained_file") != "config.source.toml":
            raise ValueError(
                "TOML config_origin must retain config.source.toml"
            )
        if (
            not isinstance(payload.get("source_filename"), str)
            or not payload["source_filename"].strip()
        ):
            raise ValueError(
                "TOML config_origin source_filename must be non-empty"
            )
        source_sha256 = payload.get("source_sha256")
        if not (
            isinstance(source_sha256, str)
            and len(source_sha256) == 64
            and all(character in "0123456789abcdef" for character in source_sha256)
        ):
            raise ValueError(
                "TOML config_origin source_sha256 must be lowercase SHA-256"
            )
    else:
        raise ValueError(
            "config_origin kind must be programmatic or toml_file"
        )
    if payload.get("config_sha256") != expected_config_sha256:
        raise ValueError(
            "config_origin config_sha256 does not bind the resolved config"
        )
    return payload


def _git_revision(workdir: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=workdir,
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def source_tree_digest(workdir: str | Path) -> str:
    """Hash the executable source and packaging declaration in stable order."""

    root = Path(workdir).resolve()
    candidates = list((root / "src" / "cape_loop").rglob("*.py"))
    typed_marker = root / "src" / "cape_loop" / "py.typed"
    package_config = root / "pyproject.toml"
    for path in (typed_marker, package_config):
        if path.is_file():
            candidates.append(path)
    if not candidates:
        raise ValueError(f"no CAPE-Loop source found under {root}")
    digest = sha256()
    for path in sorted(set(candidates), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


@dataclass(slots=True)
class RunArtifacts:
    path: Path
    config: AppConfig

    @classmethod
    def create(
        cls,
        config: AppConfig,
        *,
        root: str | Path | None = None,
        exist_ok: bool = False,
        config_origin: Mapping[str, Any] | None = None,
    ) -> "RunArtifacts":
        output_root = Path(root or config.run.output_root)
        run_id = f"{config.run.name}-{config_digest(config)[:12]}"
        path = output_root / run_id
        path.mkdir(parents=True, exist_ok=exist_ok)
        for child in ("events", "metrics", "models", "tables", "figures", "llm"):
            (path / child).mkdir(exist_ok=True)
        run = cls(path=path, config=config)
        resolved_config_sha256 = config_digest(config)
        origin = _validated_config_origin(
            config_origin
            or {
                "kind": "programmatic",
                "descriptor": (
                    "AppConfig supplied directly to "
                    "RunArtifacts.create"
                ),
                "config_sha256": resolved_config_sha256,
            },
            expected_config_sha256=resolved_config_sha256,
        )
        run.write_json("config.resolved.json", config.to_dict())
        run.write_json(
            "environment.json",
            {
                "python": sys.version,
                "implementation": platform.python_implementation(),
                "platform": platform.platform(),
                "executable": sys.executable,
            },
        )
        source_root = Path(__file__).resolve().parents[2]
        run.write_json(
            "manifest.json",
            {
                "schema_version": 1,
                "run_id": run_id,
                "config_sha256": resolved_config_sha256,
                "config_origin": origin,
                "git_revision": _git_revision(source_root),
                "source_sha256": source_tree_digest(source_root),
                "deterministic": config.run.deterministic,
                "status": "created",
            },
        )
        return run

    def _resolve(self, relative: str | Path) -> Path:
        candidate = (self.path / relative).resolve()
        root = self.path.resolve()
        if candidate != root and root not in candidate.parents:
            raise ValueError("artifact path escapes the run directory")
        return candidate

    def write_json(self, relative: str | Path, value: Any) -> Path:
        destination = self._resolve(relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(canonical_json(value) + "\n", encoding="utf-8")
        return destination

    def write_jsonl(self, relative: str | Path, rows: Iterable[Any]) -> Path:
        destination = self._resolve(relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(canonical_json(row) + "\n")
        return destination

    def write_text(self, relative: str | Path, value: str) -> Path:
        destination = self._resolve(relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(value, encoding="utf-8")
        return destination

    def write_bytes(self, relative: str | Path, value: bytes) -> Path:
        destination = self._resolve(relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(value)
        return destination

    def finalize(self, summary: Mapping[str, Any]) -> None:
        self.write_json("metrics/summary.json", summary)
        manifest_path = self.path / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["status"] = "complete"
        self.write_json("manifest.json", manifest)
        if self.config.artifacts.checksum_manifest:
            self.write_checksums()

    def write_checksums(self) -> Path:
        checksum_path = self.path / "SHA256SUMS"
        lines: list[str] = []
        for file_path in sorted(self.path.rglob("*")):
            if not file_path.is_file() or file_path == checksum_path:
                continue
            relative = file_path.relative_to(self.path).as_posix()
            digest = sha256(file_path.read_bytes()).hexdigest()
            lines.append(f"{digest}  {relative}")
        checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return checksum_path


def verify_run(path: str | Path) -> tuple[bool, tuple[str, ...]]:
    run_path = Path(path).resolve()
    checksum_path = run_path / "SHA256SUMS"
    errors: list[str] = []
    if not checksum_path.is_file():
        return False, ("missing SHA256SUMS",)
    retained_paths: set[str] = set()
    for line_number, line in enumerate(
        checksum_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            expected, relative = line.split("  ", 1)
        except ValueError:
            errors.append(f"malformed checksum line {line_number}")
            continue
        if (
            len(expected) != 64
            or expected.lower() != expected
            or any(character not in "0123456789abcdef" for character in expected)
        ):
            errors.append(f"invalid SHA-256 on line {line_number}")
            continue
        relative_path = PurePosixPath(relative)
        if (
            not relative
            or relative_path.is_absolute()
            or "\\" in relative
            or "\x00" in relative
            or any(part in {"", ".", ".."} for part in relative_path.parts)
            or relative_path.as_posix() != relative
        ):
            errors.append(f"unsafe checksum path on line {line_number}")
            continue
        unresolved_path = run_path.joinpath(*relative_path.parts)
        file_path = unresolved_path.resolve()
        if file_path == run_path or run_path not in file_path.parents:
            errors.append(f"checksum path escapes run directory: {relative}")
            continue
        canonical_relative = file_path.relative_to(run_path).as_posix()
        if canonical_relative in retained_paths:
            errors.append(f"duplicate checksum path on line {line_number}")
            continue
        retained_paths.add(canonical_relative)
        if not file_path.is_file():
            errors.append(f"missing {relative}")
            continue
        actual = sha256(file_path.read_bytes()).hexdigest()
        if not expected == actual:
            errors.append(f"checksum mismatch: {relative}")
    actual_paths = {
        file_path.relative_to(run_path).as_posix()
        for file_path in run_path.rglob("*")
        if file_path.is_file() and file_path != checksum_path
    }
    for relative in sorted(actual_paths - retained_paths):
        errors.append(f"unlisted artifact: {relative}")

    manifest_path = run_path / "manifest.json"
    summary_path = run_path / "metrics" / "summary.json"
    manifest: Mapping[str, Any] | None = None
    if not manifest_path.is_file():
        errors.append("missing manifest.json")
    else:
        try:
            decoded = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(decoded, Mapping):
                raise TypeError("manifest is not an object")
            manifest = decoded
        except (OSError, TypeError, json.JSONDecodeError) as exc:
            errors.append(f"invalid manifest.json: {exc}")
    if manifest is not None:
        if manifest.get("status") != "complete":
            errors.append("run manifest status is not complete")
        if manifest.get("run_id") != run_path.name:
            errors.append("run manifest ID does not match directory name")
        resolved_path = run_path / "config.resolved.json"
        resolved_config_sha256: str | None = None
        if not resolved_path.is_file():
            errors.append("missing config.resolved.json")
        else:
            try:
                resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
                resolved_config = AppConfig.parse(resolved)
                resolved_config_sha256 = config_digest(resolved_config)
                if (
                    manifest.get("config_sha256")
                    != resolved_config_sha256
                ):
                    errors.append(
                        "manifest config digest does not match resolved config"
                    )
            except (
                OSError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ) as exc:
                errors.append(f"invalid config.resolved.json: {exc}")
        source_config_path = run_path / "config.source.toml"
        origin = manifest.get("config_origin")
        if source_config_path.is_file():
            try:
                source_config = load_config(source_config_path)
                source_config_sha256 = config_digest(source_config)
                if (
                    resolved_config_sha256 is not None
                    and source_config_sha256 != resolved_config_sha256
                ):
                    errors.append(
                        "source TOML config digest does not match resolved "
                        "config"
                    )
                if isinstance(origin, Mapping) and origin.get(
                    "kind"
                ) == "toml_file":
                    if origin.get("retained_file") != "config.source.toml":
                        errors.append(
                            "TOML config origin retained_file mismatch"
                        )
                    if origin.get("source_sha256") != sha256(
                        source_config_path.read_bytes()
                    ).hexdigest():
                        errors.append(
                            "TOML config origin source SHA-256 mismatch"
                        )
                    if origin.get("config_sha256") != manifest.get(
                        "config_sha256"
                    ):
                        errors.append(
                            "TOML config origin resolved digest mismatch"
                        )
            except (OSError, TypeError, ValueError) as exc:
                errors.append(f"invalid config.source.toml: {exc}")
        elif not (
            isinstance(origin, Mapping)
            and origin.get("kind") == "programmatic"
            and isinstance(origin.get("descriptor"), str)
            and origin["descriptor"].strip()
            and origin.get("config_sha256")
            == manifest.get("config_sha256")
        ):
            errors.append(
                "run lacks config.source.toml and a hash-bound "
                "programmatic config origin"
            )
    if not summary_path.is_file():
        errors.append("missing metrics/summary.json")
    return not errors, tuple(errors)
