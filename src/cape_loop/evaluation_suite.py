"""Immutable two-role orchestration for the paper's OpenAI evaluation suite."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping
import json
import os
import tempfile

from .artifacts import canonical_json, config_digest, file_sha256
from .config import AppConfig, load_config
from .llm_preflight import (
    build_llm_request_preflight,
    require_live_llm_budget,
)
from .openai_provider import DEFAULT_OPENAI_MODEL_ROLES


SUITE_ID = "openai-gpt-5.6-paper-evaluation-v1"
SUITE_ROLES = ("primary", "replication")


def _source_sha256(path: Path) -> str:
    return file_sha256(path)


def _resolved_role(config: AppConfig) -> tuple[str, str]:
    declaration = DEFAULT_OPENAI_MODEL_ROLES[config.llm.model_role]
    return (
        config.llm.model or declaration.model,
        config.llm.reasoning_effort or declaration.reasoning_effort,
    )


def _matched_design(config: AppConfig) -> dict[str, Any]:
    """Normalize only the fields that must differ across suite roles."""

    payload = config.to_dict()
    payload["run"]["name"] = "<suite-role-run>"
    payload["run"]["output_root"] = "<suite-output-root>"
    payload["llm"]["model_role"] = "<suite-role>"
    payload["llm"]["model"] = "<suite-model-variant>"
    payload["llm"]["reasoning_effort"] = "<suite-reasoning-effort>"
    return payload


def _experiment_a_request_upper_bound(config: AppConfig) -> int:
    """Return a conservative provider-call bound for the checked A workflow."""

    if config.experiment.kind != "provenance_audit":
        raise ValueError(
            "the OpenAI paper suite currently supports Experiment A only"
        )
    preflight = build_llm_request_preflight(config)
    if preflight is None:
        raise ValueError("the OpenAI paper suite requires an LLM updater")
    return int(preflight["logical_completion_upper_bound"])


def _validate_role_config(
    role: str,
    path: Path,
    config: AppConfig,
) -> None:
    if config.llm.mode != "openai":
        raise ValueError(
            f"suite role {role!r} must use llm.mode = 'openai'"
        )
    if config.llm.model_role != role:
        raise ValueError(
            f"suite role {role!r} requires llm.model_role = {role!r}"
        )
    if not any(
        updater_id.startswith("llm_")
        for updater_id in config.experiment.updaters
    ):
        raise ValueError(
            f"suite role {role!r} has no LLM updater to evaluate"
        )
    try:
        require_live_llm_budget(config)
    except ValueError as exc:
        raise ValueError(
            f"suite role {role!r} fails credential-free budget preflight: "
            f"{exc}"
        ) from exc
    declared = DEFAULT_OPENAI_MODEL_ROLES[role]
    model, effort = _resolved_role(config)
    if model != declared.model or effort != declared.reasoning_effort:
        raise ValueError(
            f"suite role {role!r} must retain its declared immutable "
            f"model/effort ({declared.model}, {declared.reasoning_effort}); "
            f"received ({model}, {effort}) in {path}"
        )


def _journal_directory(
    config: AppConfig,
    *,
    output_root: Path,
    run_id: str,
) -> Path:
    journal_root = (
        Path(config.llm.journal_dir)
        if config.llm.journal_dir
        else output_root / ".llm-journals"
    )
    return (
        journal_root / run_id / config.llm.model_role
    ).resolve()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(
                json.dumps(
                    payload,
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _index_identity(payload: Mapping[str, Any]) -> tuple[Any, ...]:
    roles = payload.get("roles")
    if not isinstance(roles, list):
        return ()
    role_identities = []
    for role in roles:
        if not isinstance(role, Mapping):
            return ()
        role_identities.append(
            (
                role.get("role"),
                role.get("config_source_sha256"),
                role.get("config_sha256"),
                role.get("run_id"),
                role.get("run_directory"),
                role.get("journal_directory"),
            )
        )
    return (
        payload.get("suite_id"),
        payload.get("suite_sha256"),
        tuple(role_identities),
    )


def _guard_existing_index(
    destination: Path,
    plan: Mapping[str, Any],
    *,
    execute_live: bool,
    allow_existing: bool,
) -> None:
    if not destination.exists():
        return
    try:
        existing = json.loads(destination.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"existing suite index is unreadable: {destination}: {exc}"
        ) from exc
    if not isinstance(existing, Mapping):
        raise ValueError("existing suite index must be a JSON object")
    if _index_identity(existing) != _index_identity(plan):
        raise ValueError(
            "refusing to overwrite a suite index with different config/run "
            f"identities: {destination}"
        )
    status = existing.get("status")
    if not execute_live:
        if status != "planned":
            raise ValueError(
                "same-identity planning can refresh only an existing planned "
                f"index; found status={status!r}"
            )
        return
    if status == "planned":
        return
    if status == "complete" and allow_existing:
        return
    raise ValueError(
        "live suite execution cannot replace this same-identity index "
        f"status={status!r}; use --allow-existing only for a completed "
        "verified suite or preserve/remove the index during explicit recovery"
    )


def build_openai_evaluation_suite_plan(
    primary_config: str | Path,
    replication_config: str | Path,
    *,
    output_root: str | Path | None = None,
    index_path: str | Path | None = None,
) -> tuple[dict[str, Any], tuple[AppConfig, AppConfig], Path]:
    """Validate and materialize a credential-free, immutable suite plan."""

    config_paths = tuple(
        Path(path).resolve()
        for path in (primary_config, replication_config)
    )
    if config_paths[0] == config_paths[1]:
        raise ValueError("primary and replication config files must differ")
    configs = tuple(load_config(path) for path in config_paths)
    for role, path, config in zip(SUITE_ROLES, config_paths, configs):
        _validate_role_config(role, path, config)
    if _matched_design(configs[0]) != _matched_design(configs[1]):
        raise ValueError(
            "primary and replication configs are not a matched design after "
            "normalizing only role-specific run/model fields"
        )

    resolved_output_root = Path(
        output_root
        if output_root is not None
        else configs[0].run.output_root
    ).resolve()
    source_digests = tuple(_source_sha256(path) for path in config_paths)
    config_digests = tuple(config_digest(config) for config in configs)
    suite_digest = sha256(
        canonical_json(
            {
                "suite_id": SUITE_ID,
                "sources": list(source_digests),
                "configs": list(config_digests),
            }
        ).encode("utf-8")
    ).hexdigest()
    roles: list[dict[str, Any]] = []
    for role, path, source_digest, config, digest in zip(
        SUITE_ROLES,
        config_paths,
        source_digests,
        configs,
        config_digests,
    ):
        model, effort = _resolved_role(config)
        preflight = build_llm_request_preflight(config)
        if preflight is None:  # already rejected by _validate_role_config
            raise AssertionError("validated suite role lost its LLM updater")
        run_id = f"{config.run.name}-{digest[:12]}"
        run_directory = (resolved_output_root / run_id).resolve()
        roles.append(
            {
                "role": role,
                "config_source": str(path),
                "config_source_sha256": source_digest,
                "config_sha256": digest,
                "run_id": run_id,
                "run_directory": str(run_directory),
                "journal_directory": str(
                    _journal_directory(
                        config,
                        output_root=resolved_output_root,
                        run_id=run_id,
                    )
                ),
                "model": model,
                "reasoning_effort": effort,
                "max_output_tokens": config.llm.max_output_tokens,
                "max_requests": config.llm.max_requests,
                "max_total_tokens": config.llm.max_total_tokens,
                "conservative_request_upper_bound": (
                    _experiment_a_request_upper_bound(config)
                ),
                "retry_expansion_factor": preflight[
                    "retry_expansion_factor"
                ],
                "physical_http_attempt_upper_bound": preflight[
                    "physical_http_attempt_upper_bound"
                ],
                "maximum_output_token_allocation": preflight[
                    "maximum_output_token_allocation"
                ],
                "output_token_headroom_before_input": preflight[
                    "output_token_headroom_before_input"
                ],
                "adaptive_input_token_preflight": preflight[
                    "adaptive_input_token_preflight"
                ],
                "within_declared_retry_expanded_bounds": preflight[
                    "within_declared_retry_expanded_bounds"
                ],
                "request_headroom": (
                    config.llm.max_requests
                    - int(
                        preflight["physical_http_attempt_upper_bound"]
                    )
                ),
                "execution_status": "planned",
                "result": None,
            }
        )

    run_directories = {role["run_directory"] for role in roles}
    journal_directories = {role["journal_directory"] for role in roles}
    if len(run_directories) != len(SUITE_ROLES):
        raise ValueError("suite roles resolve to a shared run directory")
    if len(journal_directories) != len(SUITE_ROLES):
        raise ValueError("suite roles resolve to a shared journal directory")

    destination = (
        Path(index_path).resolve()
        if index_path is not None
        else (
            resolved_output_root
            / f"{SUITE_ID}-{suite_digest[:12]}.index.json"
        ).resolve()
    )
    for run_directory in (Path(path) for path in run_directories):
        if destination == run_directory or run_directory in destination.parents:
            raise ValueError(
                "combined suite index must remain outside each immutable run "
                "artifact directory"
            )

    plan = {
        "schema_version": 1,
        "suite_id": SUITE_ID,
        "suite_sha256": suite_digest,
        "status": "planned",
        "live_execution": False,
        "credential_read": False,
        "output_root": str(resolved_output_root),
        "index_path": str(destination),
        "roles": roles,
        "output_isolation": {
            "run_directories_shared": False,
            "journal_directories_shared": False,
            "combined_index_outside_run_directories": True,
        },
        "budget_enforcement": (
            "Each role receives a fresh provider ledger with only that "
            "config's max_requests and max_total_tokens ceilings. The suite "
            "also rejects a retry-expanded Experiment A request/output "
            "allocation bound that exceeds either ceiling before live "
            "execution; adaptive prompt-input tokens remain enforced before "
            "each request."
        ),
        "replication_scope": (
            "GPT-5.6 model-variant/tier replication; not distinct-family "
            "robustness."
        ),
        "distinct_model_family_robustness_claimed": False,
    }
    return plan, (configs[0], configs[1]), destination


def orchestrate_openai_evaluation_suite(
    primary_config: str | Path,
    replication_config: str | Path,
    *,
    output_root: str | Path | None = None,
    index_path: str | Path | None = None,
    execute_live: bool = False,
    allow_existing: bool = False,
) -> dict[str, Any]:
    """Plan by default or explicitly execute isolated primary/replication runs."""

    index, configs, destination = build_openai_evaluation_suite_plan(
        primary_config,
        replication_config,
        output_root=output_root,
        index_path=index_path,
    )
    _guard_existing_index(
        destination,
        index,
        execute_live=execute_live,
        allow_existing=allow_existing,
    )
    _atomic_write_json(destination, index)
    if not execute_live:
        return index

    from .runner import run_experiment

    index["status"] = "executing"
    index["live_execution"] = True
    _atomic_write_json(destination, index)
    for role_record, config in zip(index["roles"], configs):
        role_record["execution_status"] = "executing"
        _atomic_write_json(destination, index)
        source_path = Path(role_record["config_source"])
        try:
            if _source_sha256(source_path) != role_record[
                "config_source_sha256"
            ]:
                raise ValueError(
                    f"{role_record['role']} config source changed after "
                    "suite planning and before execution"
                )
            result = run_experiment(
                config,
                output_root=index["output_root"],
                allow_existing=allow_existing,
                source_config=source_path,
                execute_live=True,
            )
            if _source_sha256(source_path) != role_record[
                "config_source_sha256"
            ]:
                raise ValueError(
                    f"{role_record['role']} config source changed during "
                    "suite execution"
                )
            if (
                Path(result["run_dir"]).resolve()
                != Path(role_record["run_directory"]).resolve()
            ):
                raise ValueError(
                    f"{role_record['role']} runner returned an unexpected "
                    "run directory"
                )
        except Exception as exc:
            role_record["execution_status"] = "failed"
            role_record["failure"] = {
                "exception_type": type(exc).__name__,
                "message": str(exc),
            }
            index["status"] = "failed"
            _atomic_write_json(destination, index)
            raise
        role_record["execution_status"] = "complete"
        role_record["result"] = result
        _atomic_write_json(destination, index)
    index["status"] = "complete"
    _atomic_write_json(destination, index)
    return index
