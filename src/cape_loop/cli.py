"""Command-line interface for the offline CAPE-Loop reference implementation."""

from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence
import json
import os
import platform
import sys
import tempfile

from . import __version__
from .analysis_export import (
    export_compact_analysis,
    verify_compact_analysis,
)
from .artifacts import file_sha256, verify_run
from .config import ConfigError, load_config
from .control_study import (
    build_control_llm_exchange,
    build_experiment_a_control_plan,
    execute_control_llm_exchange,
    read_control_request_bindings,
)
from .correction_debt import run_correction_debt_experiment
from .evaluation_suite import orchestrate_openai_evaluation_suite
from .external_decoder_providers import (
    ANTHROPIC_DEFAULT_MODEL,
    ANTHROPIC_OFFICIAL_ORIGIN,
    GEMINI_DEFAULT_MODEL,
    GEMINI_OFFICIAL_ORIGIN,
    ExternalDecoderProvider,
    ExternalDecoderProviderConfig,
    ExternalDecoderProviderError,
    _ExclusiveCollectionLock,
    execute_external_decoder_collection,
    plan_external_decoder_collection,
)
from .decoder_study import (
    ExternalDecoderRequest,
    analyze_external_decoders,
    analyze_human_evidence_strength,
    external_decoder_judgment_from_response,
    external_decoder_llm_request,
    read_decoder_truth_labels,
    read_external_decoder_judgments,
    read_external_decoder_requests,
    read_human_collection,
    validate_external_decoder_import,
)
from .human_study import (
    CONDITIONS,
    StudyItem,
    blind_and_order_items,
    build_assignment_codebook,
)
from .human_comparison import (
    analyze_h8_human_model_comparison,
    convert_experiment_a_metrics_to_model_evidence,
    read_model_evidence_strengths,
)
from .h7_control_review import (
    create_h7_volunteered_review,
    load_verified_h7_source,
    snapshot_h7_review_inputs,
    verify_h7_volunteered_review,
    write_h7_plan_directory,
)
from .experiment_c_review import (
    import_experiment_c_external_rescore,
    verify_experiment_c_external_rescore,
)
from .experiment_c_robustness import (
    create_experiment_c_multiseed_review,
    verify_experiment_c_multiseed_review,
)
from .gate_review import (
    DIRECT_FIRST_PARTY_COLLECTION_PROVENANCE,
    OPENROUTER_COLLECTION_PROVENANCE,
    import_native_gate_review,
    verify_gate_review,
)
from .conversation_surfaces import load_conversation_bank
from .experiments.provenance import default_audit_users
from .llm_exchange import (
    CompletionProvider,
    LLMRequest,
    ReplayProvider,
    read_responses,
)
from .one_scenario import run_one_scenario
from .openai_provider import (
    DEFAULT_OPENAI_MODEL_ROLES,
    OpenAIProviderConfig,
    OpenAIProviderError,
    OpenAIResponsesProvider,
    ResumableOpenAICompletionProvider,
    execute_jsonl,
    read_requests,
)
from .openrouter_provider import (
    OPENROUTER_EXAMPLE_MODEL,
    OPENROUTER_MODELS_URL,
    OpenRouterChatProvider,
    OpenRouterProviderConfig,
    OpenRouterProviderError,
    ResumableOpenRouterCompletionProvider,
    execute_openrouter_jsonl,
)
from .openrouter_conversation_provider import (
    OpenRouterConversationConfig,
    OpenRouterConversationProvider,
    generate_conversation_bank,
)
from .openrouter_decoder_collection import (
    OPENROUTER_COLLECTION_LOCKS,
    OPENROUTER_DECODER_MAX_OUTPUT_TOKENS,
    OPENROUTER_DECODER_MAX_REQUESTS,
    OPENROUTER_DECODER_MAX_RETRIES,
    OPENROUTER_DECODER_MAX_TOTAL_TOKENS,
    SELECTED_OPENROUTER_DECODER_MODELS,
    SELECTED_OPENROUTER_REASONING_EFFORTS,
    build_openrouter_decoder_collection_plan,
    build_openrouter_decoder_execution_manifest,
    openrouter_decoder_family,
    openrouter_decoder_identity,
    openrouter_decoder_source_descriptor,
    openrouter_source_execution_summary,
)
from .native_action_provider import (
    OpenAINativeActionProvider,
    execute_openai_native_actions,
    plan_openai_native_actions,
)
from .release import freeze_run, verify_frozen_artifact
from .robustness_review import (
    build_gate6_cross_run_review,
    verify_gate6_cross_run_review,
)
from .schema_export import export_schemas
from .schema_export import SCHEMAS
from .scenarios import load_scenario_catalog


class _ExternalCollectionDirAction(argparse.Action):
    """Store a collection path together with the flag's provenance contract."""

    def __init__(
        self,
        option_strings: Sequence[str],
        dest: str,
        *,
        provenance_mode: str,
        **kwargs: Any,
    ) -> None:
        self.provenance_mode = provenance_mode
        super().__init__(option_strings, dest, **kwargs)

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: object,
        option_string: str | None = None,
    ) -> None:
        setattr(namespace, self.dest, values)
        setattr(
            namespace,
            "external_collection_provenance_mode",
            self.provenance_mode,
        )


def _json(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)


def _retry_expanded_budget(
    *,
    request_count: int,
    conservative_tokens: int,
    max_retries: int,
    max_requests: int,
    max_total_tokens: int,
) -> dict[str, Any]:
    """Report feasibility under the physical-attempt accounting contract."""

    attempts_per_request = max_retries + 1
    theoretical_attempts = request_count * attempts_per_request
    theoretical_tokens = conservative_tokens * attempts_per_request
    return {
        "initial_transport_attempt_count": request_count,
        "maximum_attempts_per_request": attempts_per_request,
        "theoretical_max_transport_attempts": theoretical_attempts,
        "theoretical_max_tokens_with_all_retries": theoretical_tokens,
        "request_budget_unit": "physical_http_attempt",
        "within_declared_budget": (
            theoretical_attempts <= max_requests
            and theoretical_tokens <= max_total_tokens
        ),
    }


def _atomic_write_text(path: Path, value: str) -> None:
    """Replace one UTF-8 text file only after its complete content is durable."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(name)
    try:
        with os.fdopen(
            descriptor,
            "w",
            encoding="utf-8",
            newline="\n",
        ) as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _fsync_directory(path: Path) -> None:
    """Durably persist a same-directory artifact publication."""

    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_new_text(path: Path, value: str) -> None:
    """Publish one immutable text artifact without replacing an existing path."""

    destination = path.absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.is_symlink():
        raise ValueError("artifact output parent cannot be a symlink")
    if destination.is_symlink() or destination.exists():
        raise FileExistsError(f"artifact output already exists: {destination}")
    lock = destination.parent / f".{destination.name}.publication.lock"
    try:
        lock_descriptor = os.open(
            lock,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
    except FileExistsError as exc:
        raise FileExistsError(
            f"artifact output is locked: {destination}"
        ) from exc
    descriptor = -1
    temporary: Path | None = None
    try:
        try:
            os.write(lock_descriptor, f"{os.getpid()}\n".encode("ascii"))
            os.fsync(lock_descriptor)
        finally:
            os.close(lock_descriptor)
        if destination.is_symlink() or destination.exists():
            raise FileExistsError(
                f"artifact output already exists: {destination}"
            )
        descriptor, name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
        )
        temporary = Path(name)
        with os.fdopen(
            descriptor,
            "w",
            encoding="utf-8",
            newline="\n",
        ) as handle:
            descriptor = -1
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        if destination.is_symlink() or destination.exists():
            raise FileExistsError(
                f"artifact output already exists: {destination}"
            )
        os.rename(temporary, destination)
        temporary = None
        _fsync_directory(destination.parent)
    finally:
        try:
            if descriptor >= 0:
                os.close(descriptor)
            if temporary is not None and temporary.exists():
                temporary.unlink()
        finally:
            try:
                lock.unlink()
            except FileNotFoundError:
                pass


def _snapshot_regular_file(
    path: Path,
    *,
    label: str,
) -> tuple[Path, bytes]:
    """Read one regular input exactly once and bind its resolved identity."""

    supplied = path.absolute()
    if supplied.is_symlink() or not supplied.is_file():
        raise ValueError(f"{label} must be a regular file, not a symlink")
    resolved = supplied.resolve()
    material = supplied.read_bytes()
    if (
        supplied.is_symlink()
        or not supplied.is_file()
        or supplied.resolve() != resolved
    ):
        raise ValueError(f"{label} changed while it was being read")
    return resolved, material


def _verify_file_snapshot(
    path: Path,
    *,
    resolved: Path,
    material: bytes,
    label: str,
) -> None:
    """Fail closed if a named input no longer denotes its parsed snapshot."""

    supplied = path.absolute()
    if (
        supplied.is_symlink()
        or not supplied.is_file()
        or supplied.resolve() != resolved
        or supplied.read_bytes() != material
    ):
        raise ValueError(f"{label} changed while the analysis was running")


def _containing_run(path: Path) -> Path | None:
    """Identify a run-shaped ancestor without mutating or trusting its content."""

    resolved = path.resolve()
    anchor = resolved if resolved.is_dir() else resolved.parent
    candidates = (anchor, *anchor.parents)
    for candidate in candidates:
        if all(
            (candidate / filename).is_file()
            for filename in (
                "manifest.json",
                "config.resolved.json",
                "SHA256SUMS",
            )
        ):
            return candidate
    return None


def _reject_output_inside_input_run(
    input_path: Path,
    output_path: Path,
) -> None:
    """Keep external collection plans and results outside immutable runs."""

    resolved_input = input_path.resolve()
    resolved_output = output_path.resolve()
    if resolved_output == resolved_input:
        raise ValueError(
            "collection output cannot overwrite its input: "
            f"{resolved_input}"
        )
    source_run = _containing_run(input_path)
    if source_run is None:
        return
    if (
        resolved_output == source_run
        or source_run in resolved_output.parents
    ):
        raise ValueError(
            "collection output must be outside the immutable source "
            f"run: {source_run}"
        )


def _doctor(_: argparse.Namespace) -> int:
    checks = {
        "cape_loop_version": __version__,
        "python": platform.python_version(),
        "python_supported": sys.version_info >= (3, 11),
        "core_dependencies": "standard library only",
        "network_required": False,
    }
    try:
        from .domains import get_domain

        checks["domains"] = [get_domain("travel").domain_id, get_domain("writing").domain_id]
        checks["domain_registry_ok"] = True
    except Exception as exc:  # doctor should report, not crash
        checks["domain_registry_ok"] = False
        checks["domain_error"] = f"{type(exc).__name__}: {exc}"
    print(_json(checks))
    return 0 if checks["python_supported"] and checks["domain_registry_ok"] else 1


class _SingleRequestJournal:
    """Write the one model-visible request before delegating its live call."""

    def __init__(
        self,
        provider: CompletionProvider,
        *,
        request_path: Path,
    ) -> None:
        self.provider = provider
        self.request_path = request_path
        self.call_count = 0

    def complete(self, request: LLMRequest):
        if self.call_count:
            raise RuntimeError(
                "one-scenario request journal received more than one call"
            )
        self.call_count += 1
        serialized = json.dumps(
            request.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        _atomic_write_new_text(self.request_path, serialized + "\n")
        return self.provider.complete(request)


def _demo_one_scenario(args: argparse.Namespace) -> int:
    """Execute one explanatory scenario with one physical OpenRouter attempt."""

    if not args.execute_live:
        raise ValueError(
            "one-scenario OpenRouter execution requires the explicit "
            "--execute-live flag"
        )

    loaded = load_scenario_catalog(
        args.scenario_catalog,
        expected_sha256=file_sha256(args.scenario_catalog),
    )
    conversation_bank = load_conversation_bank(args.conversation_bank)
    conversation_bank.validate_catalog(loaded.catalog)
    loaded.catalog.scenario(args.scenario_id)

    output = args.output_dir.absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.parent.is_symlink():
        raise ValueError("one-scenario output parent cannot be a symlink")
    try:
        output.mkdir()
    except FileExistsError as exc:
        raise FileExistsError(
            "one-scenario output must be a new directory so this command "
            f"cannot silently resume zero calls: {output}"
        ) from exc
    llm_dir = output / "llm"
    llm_dir.mkdir()

    raw_provider = OpenRouterChatProvider(
        OpenRouterProviderConfig(
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            api_key_env=args.api_key_env,
            upstream_provider=args.upstream_provider,
            allow_fallbacks=False,
            require_parameters=True,
            data_collection="deny",
            zdr=args.zdr,
            timeout_seconds=args.timeout_seconds,
            max_retries=0,
            max_output_tokens=2048,
            max_requests=1,
            max_total_tokens=10_000,
            live_execution=True,
        )
    )
    adapter = ResumableOpenRouterCompletionProvider(
        raw_provider,
        responses_path=llm_dir / "responses.jsonl",
        audit_path=llm_dir / "provider-audit.jsonl",
        attempts_path=llm_dir / "provider-attempts.jsonl",
    )
    journaled_provider = _SingleRequestJournal(
        adapter,
        request_path=llm_dir / "requests.jsonl",
    )
    result = run_one_scenario(
        catalog=loaded.catalog,
        conversation_bank=conversation_bank,
        scenario_id=args.scenario_id,
        user=default_audit_users()[0],
        provider=journaled_provider,
        mechanism=args.mechanism,
        seed=args.seed,
    )

    provider_manifest = adapter.to_manifest()
    expected_execution = {
        "requests_used": 1,
        "requests_executed": 1,
        "requests_resumed": 0,
        "transport_attempt_count": 1,
    }
    mismatches = {
        key: {
            "expected": expected,
            "observed": provider_manifest.get(key),
        }
        for key, expected in expected_execution.items()
        if provider_manifest.get(key) != expected
    }
    if journaled_provider.call_count != 1:
        mismatches["request_journal_call_count"] = {
            "expected": 1,
            "observed": journaled_provider.call_count,
        }
    if mismatches:
        raise RuntimeError(
            "one-scenario physical-call invariant failed: "
            + json.dumps(mismatches, sort_keys=True)
        )

    result_path = output / "result.json"
    conversation_path = output / "conversation.jsonl"
    readable_path = output / "conversation.md"
    provider_manifest_path = llm_dir / "provider-manifest.json"
    _atomic_write_new_text(
        result_path,
        _json(result.to_dict()) + "\n",
    )
    _atomic_write_new_text(
        conversation_path,
        "".join(
            json.dumps(
                record,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
            for record in result.jsonl_records()
        ),
    )
    _atomic_write_new_text(
        readable_path,
        result.render_markdown(),
    )
    _atomic_write_new_text(
        provider_manifest_path,
        _json(provider_manifest) + "\n",
    )

    audit = adapter.used_audit_records[0]
    print(
        _json(
            {
                "status": "completed",
                "scope": "one_scenario_diagnostic_demo",
                "paper_eligible": False,
                "claim_eligible": False,
                "provider": "openrouter",
                "model_requested": args.model,
                "model_returned": result.model_id,
                "upstream_provider_returned": audit.get(
                    "upstream_provider"
                ),
                "physical_openrouter_calls": 1,
                "scenario_id": result.scenario_id,
                "mechanism": result.mechanism,
                "selected_option": result.selected_option_label,
                "metrics": dict(result.metrics),
                "readable_log": str(readable_path),
                "machine_result": str(result_path),
                "provider_audit": str(
                    llm_dir / "provider-audit.jsonl"
                ),
            }
        )
    )
    return 0


def _config_validate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    from .llm_preflight import require_live_llm_budget

    require_live_llm_budget(config)
    print(_json(config.to_dict()))
    return 0


def _run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    from .runner import run_experiment

    result = run_experiment(
        config,
        output_root=args.output_root,
        allow_existing=args.allow_existing,
        source_config=args.config,
        execute_live=args.execute_live,
        resume_failed_live=args.resume_failed_live,
    )
    print(_json(result))
    return 0


def _verify(args: argparse.Namespace) -> int:
    ok, errors = verify_run(args.run_dir)
    print(_json({"run_dir": str(args.run_dir), "valid": ok, "errors": list(errors)}))
    return 0 if ok else 1


def _gate_review_import_native(args: argparse.Namespace) -> int:
    result = import_native_gate_review(
        run_dir=args.run_dir,
        requests_path=args.requests,
        judgments_path=args.judgments,
        truth_labels_path=args.truth_labels,
        native_collection_dir=args.native_collection_dir,
        source_review_path=args.source_review,
        output_dir=args.output_dir,
        external_collection_dir=args.external_collection_dir,
        external_collection_provenance_mode=(
            args.external_collection_provenance_mode
        ),
        allow_reviewed_generic_decoders=(
            args.allow_reviewed_generic_decoders
        ),
    )
    print(_json(result))
    return 0


def _gate_review_verify(args: argparse.Namespace) -> int:
    ok, errors = verify_gate_review(args.review_dir)
    print(
        _json(
            {
                "review_dir": str(args.review_dir),
                "valid": ok,
                "errors": list(errors),
            }
        )
    )
    return 0 if ok else 1


def _schema_export(args: argparse.Namespace) -> int:
    written = export_schemas(args.destination)
    print(_json({"written": [str(path) for path in written]}))
    return 0


def _conversations_generate_openrouter(
    args: argparse.Namespace,
) -> int:
    if args.output.exists():
        raise FileExistsError(
            f"conversation bank already exists: {args.output}"
        )
    log_path = (
        args.log
        if args.log is not None
        else args.output.with_name(
            args.output.stem + ".generation.jsonl"
        )
    )
    if log_path.exists():
        raise FileExistsError(
            f"conversation generation log already exists: {log_path}"
        )
    loaded = load_scenario_catalog(
        args.catalog,
        expected_sha256=file_sha256(args.catalog),
    )
    provider = OpenRouterConversationProvider(
        OpenRouterConversationConfig(
            model=args.model,
            timeout_seconds=args.timeout_seconds,
            max_output_tokens=args.max_output_tokens,
            max_requests=args.max_requests,
            max_total_tokens=args.max_total_tokens,
            upstream_provider=args.upstream_provider,
            live_execution=args.execute_live,
        )
    )
    bank = None
    try:
        bank = generate_conversation_bank(
            loaded.catalog,
            provider,
            bank_id=args.bank_id,
        )
    finally:
        logs = [
            *provider.request_logs,
            *provider.result_logs,
        ]
        if logs:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open(
                "w",
                encoding="utf-8",
                newline="\n",
            ) as handle:
                for record in logs:
                    handle.write(
                        json.dumps(
                            record,
                            sort_keys=True,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
    assert bank is not None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            bank.to_dict(),
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        _json(
            {
                "output": str(args.output),
                "generation_log": str(log_path),
                "bank_id": bank.bank_id,
                "scenario_count": len(bank.templates),
                "model": args.model,
                "runtime_choice_source": "mathematical_simulator",
                "runtime_language_source": "frozen_llm_templates",
            }
        )
    )
    return 0


def _llm_validate(args: argparse.Namespace) -> int:
    responses = read_responses(args.responses)
    print(
        _json(
            {
                "valid": True,
                "responses": len(responses),
                "models": sorted({response.model_id for response in responses}),
            }
        )
    )
    return 0


def _resolved_model(
    role_name: str,
    model_override: str,
    effort_override: str,
) -> tuple[str, str]:
    role = DEFAULT_OPENAI_MODEL_ROLES[role_name]
    return (
        model_override or role.model,
        effort_override or role.reasoning_effort,
    )


def _openai_cli_config(
    args: argparse.Namespace,
    *,
    live_execution: bool,
    role_name: str | None = None,
) -> OpenAIProviderConfig:
    declared_role = role_name or args.role
    model, effort = _resolved_model(
        declared_role,
        getattr(args, "model", ""),
        getattr(args, "reasoning_effort", ""),
    )
    return OpenAIProviderConfig(
        model=model,
        reasoning_effort=effort,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        allow_custom_base_url=args.allow_custom_base_url,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
        max_output_tokens=args.max_output_tokens,
        max_requests=args.max_requests,
        max_total_tokens=args.max_total_tokens,
        live_execution=live_execution,
    )


def _openrouter_cli_config(
    args: argparse.Namespace,
    *,
    live_execution: bool,
) -> OpenRouterProviderConfig:
    return OpenRouterProviderConfig(
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        allow_custom_base_url=args.allow_custom_base_url,
        upstream_provider=args.upstream_provider,
        allow_fallbacks=args.allow_fallbacks,
        require_parameters=not args.allow_unsupported_parameters,
        data_collection=args.data_collection,
        zdr=args.zdr,
        http_referer=args.http_referer,
        app_title=args.app_title,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
        max_output_tokens=args.max_output_tokens,
        max_requests=args.max_requests,
        max_total_tokens=args.max_total_tokens,
        live_execution=live_execution,
    )


def _llm_models(_: argparse.Namespace) -> int:
    print(
        _json(
            {
                "suite_id": "openai-gpt-5.6-paper-evaluation-v1",
                "roles": [
                    {
                        "role": role.role,
                        "model": role.model,
                        "reasoning_effort": role.reasoning_effort,
                        "purpose": role.purpose,
                    }
                    for role in DEFAULT_OPENAI_MODEL_ROLES.values()
                ],
                "within_condition_rule": (
                    "Hold model and reasoning effort fixed across updater "
                    "information views."
                ),
                "openrouter": {
                    "supported": True,
                    "example_model": OPENROUTER_EXAMPLE_MODEL,
                    "catalog": OPENROUTER_MODELS_URL,
                    "selection_rule": (
                        "Use one explicit author/model slug; aliases, "
                        "openrouter/auto, and model fallback arrays are "
                        "excluded from reproducible runs."
                    ),
                    "strict_gate4_first_party_origin": False,
                },
                "selected_decoder_pair": {
                    "provenance_mode": (
                        "selected_openrouter_gateway_collection"
                    ),
                    "shared_gateway": True,
                    "first_party_origin_claimed": False,
                    "statistical_independence_claimed": False,
                    "sources": [
                        {
                            "model": model,
                            "reasoning_effort": (
                                SELECTED_OPENROUTER_REASONING_EFFORTS[
                                    model
                                ]
                            ),
                            "decoder_family_id": (
                                openrouter_decoder_family(model)
                            ),
                        }
                        for model in SELECTED_OPENROUTER_DECODER_MODELS
                    ],
                    "optional_direct_adapters_retained": [
                        {
                            "provider": "anthropic",
                            "model": ANTHROPIC_DEFAULT_MODEL,
                        },
                        {
                            "provider": "google_gemini",
                            "model": GEMINI_DEFAULT_MODEL,
                        },
                    ],
                },
            }
        )
    )
    return 0


def _llm_plan(args: argparse.Namespace) -> int:
    requests = read_requests(args.requests)
    config = _openai_cli_config(args, live_execution=False)
    provider = OpenAIResponsesProvider(config)
    prepared = tuple(provider.prepare(request) for request in requests)
    conservative_tokens = sum(
        request.estimated_max_tokens for request in prepared
    )
    retry_budget = _retry_expanded_budget(
        request_count=len(prepared),
        conservative_tokens=conservative_tokens,
        max_retries=config.max_retries,
        max_requests=config.max_requests,
        max_total_tokens=config.max_total_tokens,
    )
    print(
        _json(
            {
                "live_execution": False,
                "credential_read": False,
                "request_count": len(prepared),
                "model": config.model,
                "reasoning_effort": config.reasoning_effort,
                "api_key_env": config.api_key_env,
                "conservative_max_tokens": conservative_tokens,
                "max_requests": config.max_requests,
                "max_total_tokens": config.max_total_tokens,
                **retry_budget,
                "request_body_sha256": [
                    {
                        "request_id": request.request_id,
                        "sha256": item.body_sha256,
                    }
                    for request, item in zip(requests, prepared)
                ],
            }
        )
    )
    return 0 if retry_budget["within_declared_budget"] else 1


def _llm_execute(args: argparse.Namespace) -> int:
    if not args.execute_live:
        raise ValueError(
            "live execution requires the explicit --execute-live flag"
        )
    provider = OpenAIResponsesProvider(
        _openai_cli_config(args, live_execution=True)
    )
    if not args.responses.exists() and not args.audit.exists():
        requests = read_requests(args.requests)
        conservative_tokens = sum(
            provider.prepare(request).estimated_max_tokens
            for request in requests
        )
        retry_budget = _retry_expanded_budget(
            request_count=len(requests),
            conservative_tokens=conservative_tokens,
            max_retries=provider.config.max_retries,
            max_requests=provider.config.max_requests,
            max_total_tokens=provider.config.max_total_tokens,
        )
        if not retry_budget["within_declared_budget"]:
            raise ValueError(
                "fresh execution would exceed a declared hard budget; run "
                "`cape-loop llm plan` and increase budgets deliberately"
            )
    summary = execute_jsonl(
        provider,
        args.requests,
        responses_path=args.responses,
        audit_path=args.audit,
    )
    print(_json(summary.to_dict()))
    return 0


def _llm_plan_openrouter(args: argparse.Namespace) -> int:
    requests = read_requests(args.requests)
    config = _openrouter_cli_config(args, live_execution=False)
    provider = OpenRouterChatProvider(config)
    prepared = tuple(provider.prepare(request) for request in requests)
    conservative_tokens = sum(
        request.estimated_max_tokens for request in prepared
    )
    retry_budget = _retry_expanded_budget(
        request_count=len(prepared),
        conservative_tokens=conservative_tokens,
        max_retries=config.max_retries,
        max_requests=config.max_requests,
        max_total_tokens=config.max_total_tokens,
    )
    print(
        _json(
            {
                "provider": "openrouter",
                "gateway": "openrouter",
                "live_execution": False,
                "credential_read": False,
                "request_count": len(prepared),
                "model": config.model,
                "reasoning_effort": config.reasoning_effort or None,
                "api_key_env": config.api_key_env,
                "endpoint": config.endpoint,
                "upstream_provider_constraint": (
                    config.upstream_provider or None
                ),
                "provider_preferences": config.provider_preferences(),
                "response_cache_enabled": False,
                "router_metadata_requested": True,
                "router_transforms_accepted": False,
                "first_party_origin_claimed": False,
                "conservative_max_tokens": conservative_tokens,
                "max_requests": config.max_requests,
                "max_retries_per_logical_request": config.max_retries,
                "max_total_tokens": config.max_total_tokens,
                **retry_budget,
                "request_body_sha256": [
                    {
                        "request_id": request.request_id,
                        "sha256": item.body_sha256,
                    }
                    for request, item in zip(requests, prepared)
                ],
            }
        )
    )
    return 0 if retry_budget["within_declared_budget"] else 1


def _llm_execute_openrouter(args: argparse.Namespace) -> int:
    if not args.execute_live:
        raise ValueError(
            "live OpenRouter execution requires the explicit "
            "--execute-live flag"
        )
    provider = OpenRouterChatProvider(
        _openrouter_cli_config(args, live_execution=True)
    )
    if not args.responses.exists() and not args.audit.exists():
        requests = read_requests(args.requests)
        conservative_tokens = sum(
            provider.prepare(request).estimated_max_tokens
            for request in requests
        )
        retry_budget = _retry_expanded_budget(
            request_count=len(requests),
            conservative_tokens=conservative_tokens,
            max_retries=provider.config.max_retries,
            max_requests=provider.config.max_requests,
            max_total_tokens=provider.config.max_total_tokens,
        )
        if not retry_budget["within_declared_budget"]:
            raise ValueError(
                "fresh OpenRouter execution would exceed a declared hard "
                "budget; run `cape-loop llm plan-openrouter` and increase "
                "budgets deliberately"
            )
    summary = execute_openrouter_jsonl(
        provider,
        args.requests,
        responses_path=args.responses,
        audit_path=args.audit,
    )
    print(_json(summary.to_dict()))
    return 0


def _llm_evaluation_suite(args: argparse.Namespace) -> int:
    result = orchestrate_openai_evaluation_suite(
        args.primary_config,
        args.replication_config,
        output_root=args.output_root,
        index_path=args.index,
        execute_live=args.execute_live,
        allow_existing=args.allow_existing,
    )
    print(_json(result))
    return 0


def _decoder_validate(args: argparse.Namespace) -> int:
    requests = read_external_decoder_requests(args.requests)
    judgments = read_external_decoder_judgments(args.judgments)
    audit = validate_external_decoder_import(
        requests,
        judgments,
        minimum_sources_per_request=args.minimum_sources,
        require_distinct_families=not args.allow_same_family,
    )
    print(_json(audit.to_dict()))
    return 0 if audit.source_design_eligible else 1


def _decoder_analyze(args: argparse.Namespace) -> int:
    analysis = analyze_external_decoders(
        read_external_decoder_requests(args.requests),
        read_external_decoder_judgments(args.judgments),
        read_decoder_truth_labels(args.truth_labels),
        reliability_bins=args.reliability_bins,
    )
    payload = analysis.to_dict()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(_json(payload) + "\n", encoding="utf-8")
    print(_json(payload))
    return 0


_GENERIC_DECODER_COMMAND_LOCK_NAME = ".external-decoder-command.lock"


def _openai_decoder_roles(args: argparse.Namespace) -> tuple[str, ...]:
    roles = tuple(args.roles)
    if len(roles) != len(set(roles)):
        raise ValueError("OpenAI decoder roles must not contain duplicates")
    return roles


def _decoder_execute_openai(args: argparse.Namespace) -> int:
    if not args.execute_live:
        raise ValueError(
            "decoder execution requires the explicit --execute-live flag"
        )
    _openai_decoder_roles(args)
    _reject_output_inside_input_run(args.requests, args.output_dir)
    requests = read_external_decoder_requests(args.requests)
    output = Path(args.output_dir)
    with _ExclusiveCollectionLock(
        output / _GENERIC_DECODER_COMMAND_LOCK_NAME
    ):
        return _decoder_execute_openai_locked(
            args,
            output=output,
            requests=requests,
        )


def _decoder_execute_openai_locked(
    args: argparse.Namespace,
    *,
    output: Path,
    requests: Sequence[ExternalDecoderRequest],
) -> int:
    """Reconcile, dispatch, and publish one OpenAI decoder transaction."""

    prepared_runs = []
    for role_name in _openai_decoder_roles(args):
        role = DEFAULT_OPENAI_MODEL_ROLES[role_name]
        model, _ = _resolved_model(role_name, "", "")
        instance_id = (
            "openai-"
            + role_name
            + "-"
            + "".join(
                character if character.isalnum() else "-"
                for character in model
            )
        )
        provider = OpenAIResponsesProvider(
            _openai_cli_config(
                args,
                live_execution=True,
                role_name=role_name,
            )
        )
        journal = output / "journals" / role_name
        adapter = ResumableOpenAICompletionProvider(
            provider,
            responses_path=journal / "responses.jsonl",
            audit_path=journal / "provider-audit.jsonl",
        )
        provider_requests = tuple(
            external_decoder_llm_request(
                request,
                decoder_instance_id=instance_id,
            )
            for request in requests
        )
        adapter.require_static_corpus_capacity(provider_requests)
        prepared_runs.append(
            (
                role_name,
                role,
                model,
                instance_id,
                adapter,
                provider_requests,
            )
        )

    judgments = []
    manifests = []
    for (
        role_name,
        role,
        model,
        instance_id,
        adapter,
        provider_requests,
    ) in prepared_runs:
        for request, provider_request in zip(requests, provider_requests):
            response = adapter.complete(provider_request)
            judgments.append(
                external_decoder_judgment_from_response(
                    request,
                    response,
                    decoder_instance_id=instance_id,
                    decoder_family_id=model,
                    source_descriptor=f"openai-responses:{model}",
                )
            )
        manifests.append(
            {
                **adapter.to_manifest(),
                "role": role_name,
                "purpose": role.purpose,
            }
        )
    judgment_path = output / "judgments.jsonl"
    _atomic_write_text(
        judgment_path,
        "".join(
            json.dumps(
                judgment.to_dict(),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            + "\n"
            for judgment in judgments
        ),
    )
    design_audit = validate_external_decoder_import(requests, judgments)
    _atomic_write_text(
        output / "execution-manifest.json",
        _json(
            {
                "schema_version": 1,
                "request_count": len(requests),
                "judgment_count": len(judgments),
                "roles": list(_openai_decoder_roles(args)),
                "provider_runs": manifests,
                "source_design_audit": design_audit.to_dict(),
                "statistical_independence_claimed": False,
                "credentials_retained": False,
            }
        )
        + "\n",
    )
    print(
        _json(
            {
                "output_dir": str(output),
                "judgments": len(judgments),
                "source_design_eligible": (
                    design_audit.source_design_eligible
                ),
                "statistical_independence_claimed": False,
            }
        )
    )
    return 0


def _decoder_plan_openai(args: argparse.Namespace) -> int:
    roles = _openai_decoder_roles(args)
    requests = read_external_decoder_requests(args.requests)
    sources = []
    all_within_budget = True
    for role_name in roles:
        role = DEFAULT_OPENAI_MODEL_ROLES[role_name]
        provider = OpenAIResponsesProvider(
            _openai_cli_config(
                args,
                live_execution=False,
                role_name=role_name,
            )
        )
        prepared = tuple(
            provider.prepare(
                external_decoder_llm_request(
                    request,
                    decoder_instance_id=f"plan-{role_name}",
                )
            )
            for request in requests
        )
        conservative_tokens = sum(
            request.estimated_max_tokens for request in prepared
        )
        retry_budget = _retry_expanded_budget(
            request_count=len(prepared),
            conservative_tokens=conservative_tokens,
            max_retries=provider.config.max_retries,
            max_requests=provider.config.max_requests,
            max_total_tokens=provider.config.max_total_tokens,
        )
        within_budget = bool(retry_budget["within_declared_budget"])
        all_within_budget = all_within_budget and within_budget
        sources.append(
            {
                "role": role_name,
                "model": role.model,
                "reasoning_effort": role.reasoning_effort,
                "request_count": len(prepared),
                "conservative_max_tokens": conservative_tokens,
                "max_requests": provider.config.max_requests,
                "max_total_tokens": provider.config.max_total_tokens,
                **retry_budget,
            }
        )
    print(
        _json(
            {
                "live_execution": False,
                "credential_read": False,
                "decoder_source_count": len(sources),
                "sources": sources,
                "all_within_declared_budget": all_within_budget,
                "statistical_independence_claimed": False,
            }
        )
    )
    return 0 if all_within_budget else 1


def _openrouter_decoder_models(
    args: argparse.Namespace,
) -> tuple[str, ...]:
    model = getattr(args, "model", None)
    additional = tuple(getattr(args, "additional_model", ()))
    if model is None:
        if additional:
            raise ValueError(
                "--additional-model requires an explicit --model"
            )
        models = SELECTED_OPENROUTER_DECODER_MODELS
    else:
        models = (model, *additional)
    if len(models) != len(set(models)):
        raise ValueError("OpenRouter decoder models must not contain duplicates")
    return models


_OPENROUTER_DECODER_REASONING_EFFORTS = frozenset(
    {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
)


def _openrouter_model_effort_overrides(
    args: argparse.Namespace,
) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in getattr(args, "model_reasoning_effort", ()):
        model, separator, effort = raw.partition("=")
        if (
            not separator
            or not model
            or effort not in _OPENROUTER_DECODER_REASONING_EFFORTS
        ):
            raise ValueError(
                "--model-reasoning-effort must use MODEL=EFFORT with effort "
                "one of none, minimal, low, medium, high, xhigh, or max"
            )
        if model in result:
            raise ValueError(
                "OpenRouter per-model reasoning overrides must be unique"
            )
        result[model] = effort
    models = set(_openrouter_decoder_models(args))
    unexpected = sorted(set(result) - models)
    if unexpected:
        raise ValueError(
            "reasoning override references an unselected model: "
            + ", ".join(unexpected)
        )
    return result


def _openrouter_decoder_reasoning_effort(
    args: argparse.Namespace,
    model: str,
) -> str:
    override = _openrouter_model_effort_overrides(args).get(model)
    if override is not None:
        return override
    global_effort = getattr(args, "reasoning_effort", "")
    if global_effort:
        return global_effort
    return SELECTED_OPENROUTER_REASONING_EFFORTS.get(model, "")


def _openrouter_decoder_config(
    args: argparse.Namespace,
    *,
    model: str,
    live_execution: bool,
) -> OpenRouterProviderConfig:
    prepared = argparse.Namespace(**vars(args))
    prepared.model = model
    prepared.reasoning_effort = _openrouter_decoder_reasoning_effort(
        args,
        model,
    )
    return _openrouter_cli_config(
        prepared,
        live_execution=live_execution,
    )


def _decoder_plan_openrouter(args: argparse.Namespace) -> int:
    requests = read_external_decoder_requests(args.requests)
    configs = tuple(
        _openrouter_decoder_config(
            args,
            model=model,
            live_execution=False,
        )
        for model in _openrouter_decoder_models(args)
    )
    plan = build_openrouter_decoder_collection_plan(requests, configs)
    print(_json(plan))
    return 0 if plan["all_within_declared_budget"] is True else 1


def _decoder_execute_openrouter(args: argparse.Namespace) -> int:
    if not args.execute_live:
        raise ValueError(
            "OpenRouter decoder execution requires the explicit "
            "--execute-live flag"
        )
    models = _openrouter_decoder_models(args)
    _reject_output_inside_input_run(args.requests, args.output_dir)
    requests = read_external_decoder_requests(args.requests)
    output = Path(args.output_dir)
    if output.parent.is_symlink():
        raise ValueError("OpenRouter decoder output parent cannot be a symlink")
    if output.exists() and (output.is_symlink() or not output.is_dir()):
        raise ValueError(
            "OpenRouter decoder output must be a safe directory"
        )
    with _ExclusiveCollectionLock(
        output / OPENROUTER_COLLECTION_LOCKS[0]
    ):
        _require_safe_openrouter_decoder_output(output, models=models)
        with _ExclusiveCollectionLock(
            output / OPENROUTER_COLLECTION_LOCKS[1]
        ):
            return _decoder_execute_openrouter_locked(
                args,
                output=output,
                requests=requests,
            )


def _require_safe_openrouter_decoder_output(
    output: Path,
    *,
    models: Sequence[str],
) -> None:
    """Reject unsafe resume paths before any credential can be read."""

    if output.parent.is_symlink():
        raise ValueError("OpenRouter decoder output parent cannot be a symlink")
    if output.exists() and (output.is_symlink() or not output.is_dir()):
        raise ValueError(
            "OpenRouter decoder output must be a safe directory"
        )
    if not output.exists():
        return
    allowed_root = {
        *OPENROUTER_COLLECTION_LOCKS,
        "collection-plan.json",
        "transport-attempts.jsonl",
        "provider-audit.jsonl",
        "judgments.jsonl",
        "execution-manifest.json",
        "journals",
    }
    actual_root = {item.name for item in output.iterdir()}
    unexpected = sorted(actual_root - allowed_root)
    if unexpected:
        raise ValueError(
            "OpenRouter decoder output contains unexpected entries: "
            + ", ".join(unexpected)
        )
    for name in actual_root - {"journals"}:
        candidate = output / name
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError(
                f"unsafe OpenRouter decoder resume file: {name}"
            )
    journals = output / "journals"
    if not journals.exists():
        return
    if journals.is_symlink() or not journals.is_dir():
        raise ValueError(
            "OpenRouter decoder journals must be a safe directory"
        )
    expected_journals = {
        openrouter_decoder_identity(model)[0] for model in models
    }
    unexpected_journals = sorted(
        {item.name for item in journals.iterdir()} - expected_journals
    )
    if unexpected_journals:
        raise ValueError(
            "OpenRouter decoder output contains unexpected model journals: "
            + ", ".join(unexpected_journals)
        )
    allowed_journal_files = {
        "provider-audit-transport-attempts.jsonl",
        "provider-audit.jsonl",
        "responses.jsonl",
    }
    for journal in journals.iterdir():
        if journal.is_symlink() or not journal.is_dir():
            raise ValueError(
                "OpenRouter model journal must be a safe directory"
            )
        unexpected_files = sorted(
            {item.name for item in journal.iterdir()} - allowed_journal_files
        )
        if unexpected_files:
            raise ValueError(
                "OpenRouter model journal contains unexpected files: "
                + ", ".join(unexpected_files)
            )
        for item in journal.iterdir():
            if item.is_symlink() or not item.is_file():
                raise ValueError(
                    "OpenRouter model journal contains an unsafe file"
                )


def _decoder_execute_openrouter_locked(
    args: argparse.Namespace,
    *,
    output: Path,
    requests: Sequence[ExternalDecoderRequest],
) -> int:
    """Reconcile, dispatch, and publish one OpenRouter decoder transaction."""

    requests = tuple(sorted(requests, key=lambda row: row.request_id))
    planning_configs = tuple(
        _openrouter_decoder_config(
            args,
            model=model,
            live_execution=False,
        )
        for model in _openrouter_decoder_models(args)
    )
    plan = build_openrouter_decoder_collection_plan(
        requests,
        planning_configs,
    )
    if plan["all_within_declared_budget"] is not True:
        raise ValueError(
            "remaining retry-expanded corpus would exceed the OpenRouter "
            "decoder collection hard budget before any provider call"
        )
    plan_path = output / "collection-plan.json"
    if plan_path.exists():
        if plan_path.is_symlink() or not plan_path.is_file():
            raise ValueError(
                "existing OpenRouter collection plan is not a safe file"
            )
        try:
            retained_plan = json.loads(plan_path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                "existing OpenRouter collection plan is invalid JSON"
            ) from exc
        if retained_plan != plan:
            raise ValueError(
                "existing OpenRouter collection plan differs from the current "
                "request, model, effort, route, or budget configuration"
            )
    else:
        _atomic_write_text(plan_path, _json(plan) + "\n")

    prepared_runs = []
    for index, model in enumerate(plan["models"], start=1):
        config = _openrouter_decoder_config(
            args,
            model=model,
            live_execution=True,
        )
        provider = OpenRouterChatProvider(config)
        model_digest, instance_id = openrouter_decoder_identity(model)
        provider_requests = tuple(
            external_decoder_llm_request(
                request,
                decoder_instance_id=instance_id,
            )
            for request in requests
        )
        journal = output / "journals" / model_digest
        adapter = ResumableOpenRouterCompletionProvider(
            provider,
            responses_path=journal / "responses.jsonl",
            audit_path=journal / "provider-audit.jsonl",
        )
        adapter.require_static_corpus_capacity(provider_requests)
        prepared_runs.append(
            (index, model, instance_id, adapter, provider_requests)
        )

    judgments = []
    manifests = []
    for (
        index,
        model,
        instance_id,
        adapter,
        provider_requests,
    ) in prepared_runs:
        for request, provider_request in zip(requests, provider_requests):
            response = adapter.complete(provider_request)
            judgments.append(
                external_decoder_judgment_from_response(
                    request,
                    response,
                    decoder_instance_id=instance_id,
                    decoder_family_id=openrouter_decoder_family(model),
                    source_descriptor=openrouter_decoder_source_descriptor(
                        model
                    ),
                )
            )
        audits = tuple(adapter.used_audit_records)
        source_summary = openrouter_source_execution_summary(
            source_index=index,
            config=adapter.provider.config,
            decoder_instance_id=instance_id,
            request_count=len(provider_requests),
            transport_attempt_count=adapter.provider.budget.request_count,
            total_tokens=adapter.provider.budget.total_tokens,
            audits=audits,
        )
        manifests.append(source_summary)
    judgment_path = output / "judgments.jsonl"
    _atomic_write_text(
        judgment_path,
        "".join(
            json.dumps(
                judgment.to_dict(),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            + "\n"
            for judgment in judgments
        ),
    )
    aggregate_audits = [
        audit
        for _, _, _, adapter, _ in prepared_runs
        for audit in adapter.used_audit_records
    ]
    aggregate_attempts = [
        attempt
        for _, _, _, adapter, _ in prepared_runs
        for attempt in adapter.used_attempt_records
    ]
    _atomic_write_text(
        output / "provider-audit.jsonl",
        "".join(
            json.dumps(
                dict(audit),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            + "\n"
            for audit in aggregate_audits
        ),
    )
    _atomic_write_text(
        output / "transport-attempts.jsonl",
        "".join(
            json.dumps(
                dict(attempt),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            + "\n"
            for attempt in aggregate_attempts
        ),
    )
    design_audit = validate_external_decoder_import(requests, judgments)
    execution_manifest = build_openrouter_decoder_execution_manifest(
        root=output,
        plan=plan,
        source_runs=manifests,
        source_design_audit=design_audit.to_dict(),
    )
    _atomic_write_text(
        output / "execution-manifest.json",
        _json(execution_manifest) + "\n",
    )
    print(
        _json(
            {
                "output_dir": str(output),
                "judgments": len(judgments),
                "source_design_eligible": (
                    design_audit.source_design_eligible
                ),
                "eligible_for_reviewed_shared_gateway_admission": (
                    execution_manifest[
                        "eligible_for_reviewed_shared_gateway_admission"
                    ]
                ),
                "shared_gateway": True,
                "first_party_origin_claimed": False,
                "strict_first_party_gate4_eligible": False,
                "statistical_independence_claimed": False,
            }
        )
    )
    return 0


def _external_decoder_configs(
    args: argparse.Namespace,
    *,
    live_execution: bool,
) -> tuple[ExternalDecoderProviderConfig, ...]:
    if (
        args.max_requests_per_source > 900
        or args.max_total_tokens_per_source > 6_000_000
        or args.max_output_tokens > 1_024
    ):
        raise ValueError(
            "strict Gate 4 decoder collection cannot exceed the approved "
            "per-source ceilings: 900 physical attempts, 6000000 total "
            "tokens, and 1024 output tokens"
        )
    shared = {
        "timeout_seconds": args.timeout_seconds,
        "max_retries": args.max_retries,
        "max_output_tokens": args.max_output_tokens,
        "max_requests": args.max_requests_per_source,
        "max_total_tokens": args.max_total_tokens_per_source,
        "live_execution": live_execution,
    }
    return (
        ExternalDecoderProviderConfig(
            provider="anthropic",
            model=args.anthropic_model,
            api_key_env=args.anthropic_api_key_env,
            base_url=args.anthropic_base_url,
            allow_custom_base_url=args.allow_custom_anthropic_base_url,
            **shared,
        ),
        ExternalDecoderProviderConfig(
            provider="google_gemini",
            model=args.gemini_model,
            api_key_env=args.gemini_api_key_env,
            base_url=args.gemini_base_url,
            allow_custom_base_url=args.allow_custom_gemini_base_url,
            **shared,
        ),
    )


def _decoder_plan_distinct(args: argparse.Namespace) -> int:
    configs = _external_decoder_configs(args, live_execution=False)
    requests = read_external_decoder_requests(args.requests)
    plan = plan_external_decoder_collection(requests, configs)
    if args.output is not None:
        _reject_output_inside_input_run(args.requests, args.output)
        _atomic_write_text(args.output, _json(plan) + "\n")
    print(_json(plan))
    return 0


def _decoder_execute_distinct(args: argparse.Namespace) -> int:
    if not args.execute_live:
        raise ValueError(
            "distinct decoder execution requires the explicit "
            "--execute-live flag"
        )
    planning_configs = _external_decoder_configs(
        args,
        live_execution=False,
    )
    _reject_output_inside_input_run(args.requests, args.output_dir)
    requests = read_external_decoder_requests(args.requests)
    plan = plan_external_decoder_collection(requests, planning_configs)
    output = Path(args.output_dir)
    with _ExclusiveCollectionLock(
        output / ".external-decoder-command.lock"
    ):
        return _decoder_execute_distinct_locked(
            args,
            requests=requests,
            plan=plan,
            output=output,
        )


def _decoder_execute_distinct_locked(
    args: argparse.Namespace,
    *,
    requests: tuple[object, ...],
    plan: Mapping[str, object],
    output: Path,
) -> int:
    """Run planning, provider collection, and manifesting under one lock."""

    plan_path = output / "collection-plan.json"
    plan_text = _json(plan) + "\n"
    if plan_path.exists() and plan_path.read_text(
        encoding="utf-8"
    ) != plan_text:
        raise ValueError(
            "existing distinct-decoder plan has a different request, model, "
            "origin, or budget identity"
        )
    _atomic_write_text(plan_path, plan_text)

    providers = tuple(
        ExternalDecoderProvider(config)
        for config in _external_decoder_configs(
            args,
            live_execution=True,
        )
    )
    judgments_path = output / "judgments.jsonl"
    audit_path = output / "provider-audit.jsonl"
    attempt_path = output / "transport-attempts.jsonl"
    summary = execute_external_decoder_collection(
        providers,
        requests,
        judgments_path=judgments_path,
        audit_path=audit_path,
        attempt_path=attempt_path,
    )
    judgments = read_external_decoder_judgments(judgments_path)
    source_design = validate_external_decoder_import(
        requests,
        judgments,
        minimum_sources_per_request=2,
        require_distinct_families=True,
    )
    portable_summary = {
        **summary.to_dict(),
        "judgments_path": judgments_path.name,
        "audit_path": audit_path.name,
        "attempt_path": attempt_path.name,
        "repaired_trailing_files": [
            Path(path).name
            for path in summary.repaired_trailing_files
        ],
    }
    manifest = {
        "schema_version": 1,
        "kind": "distinct-external-decoder-collection",
        "status": "complete",
        "claim_status": "not_claimed",
        "collection_plan_sha256": file_sha256(plan_path),
        "judgments_sha256": file_sha256(judgments_path),
        "provider_audit_sha256": file_sha256(audit_path),
        "transport_attempts_sha256": file_sha256(attempt_path),
        "execution_summary": portable_summary,
        "source_design_audit": source_design.to_dict(),
        "distinct_provider_model_families": True,
        "statistical_independence_claimed": False,
        "responsible_researcher_source_review_required": True,
        "credentials_retained": False,
    }
    _atomic_write_text(
        output / "execution-manifest.json",
        _json(manifest) + "\n",
    )
    print(
        _json(
            {
                **summary.to_dict(),
                "output_dir": str(output),
                "source_design_eligible": (
                    source_design.source_design_eligible
                ),
                "statistical_independence_claimed": False,
                "claim_status": "not_claimed",
            }
        )
    )
    return 0


def _native_action_plan_openai(args: argparse.Namespace) -> int:
    _require_gate4_native_cli_caps(args)
    if args.output is not None:
        _reject_output_inside_input_run(args.run_dir, args.output)
    config = _openai_cli_config(
        args,
        live_execution=False,
        role_name="primary",
    )
    plan = plan_openai_native_actions(args.run_dir, config)
    if args.output is not None:
        _atomic_write_text(args.output, _json(plan) + "\n")
    print(_json(plan))
    return 0 if plan["within_declared_budget"] else 1


def _native_action_execute_openai(args: argparse.Namespace) -> int:
    if not args.execute_live:
        raise ValueError(
            "native action execution requires the explicit --execute-live flag"
        )
    _require_gate4_native_cli_caps(args)
    provider = OpenAINativeActionProvider(
        _openai_cli_config(
            args,
            live_execution=True,
            role_name="primary",
        )
    )
    result = execute_openai_native_actions(
        args.run_dir,
        args.output_dir,
        provider,
    )
    print(_json(result))
    return 0


def _require_gate4_native_cli_caps(args: argparse.Namespace) -> None:
    if (
        args.max_requests > 900
        or args.max_total_tokens > 6_000_000
        or args.max_output_tokens > 4_096
    ):
        raise ValueError(
            "strict Gate 4 native-action collection cannot exceed the "
            "approved ceilings: 900 physical attempts, 6000000 total "
            "tokens, and 4096 output tokens"
        )


def _load_assignment_codebooks(path: Path | bytes) -> dict[str, object]:
    if isinstance(path, bytes):
        try:
            decoded = json.loads(path)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                "human-study codebook bytes are not valid UTF-8 JSON"
            ) from exc
    else:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError("human-study codebook must be a JSON object")
    if {
        "assignment_id",
        "items_by_display_id",
    } <= set(decoded):
        assignment_id = decoded["assignment_id"]
        items = decoded["items_by_display_id"]
        if not isinstance(assignment_id, str) or not isinstance(items, dict):
            raise ValueError("human-study codebook packet is malformed")
        return {assignment_id: items}
    assignments = decoded.get("assignments", decoded)
    if not isinstance(assignments, dict):
        raise ValueError("human-study assignments must be an object")
    return assignments


def _human_study_analyze(args: argparse.Namespace) -> int:
    result = analyze_human_evidence_strength(
        read_human_collection(args.responses),
        assignment_codebooks=_load_assignment_codebooks(args.codebook),
        expected_assignment_protocol_id=args.assignment_protocol_id,
        expected_consent_version=args.consent_version,
        expected_blinding_version=args.blinding_version,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    payload = result.to_dict()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(_json(payload) + "\n", encoding="utf-8")
    print(_json(payload))
    return 0


def _human_study_compare(args: argparse.Namespace) -> int:
    snapshots = {
        "human responses": (
            args.responses,
            *_snapshot_regular_file(
                args.responses,
                label="human responses",
            ),
        ),
        "assignment codebook": (
            args.codebook,
            *_snapshot_regular_file(
                args.codebook,
                label="assignment codebook",
            ),
        ),
        "model evidence": (
            args.model_evidence,
            *_snapshot_regular_file(
                args.model_evidence,
                label="model evidence",
            ),
        ),
    }
    resolved_output = args.output.resolve()
    if any(
        resolved_output == resolved
        for _, resolved, _ in snapshots.values()
    ):
        raise ValueError("H8 output cannot overwrite an evidence input")
    response_material = snapshots["human responses"][2]
    codebook_material = snapshots["assignment codebook"][2]
    evidence_material = snapshots["model evidence"][2]
    result = analyze_h8_human_model_comparison(
        read_human_collection(response_material),
        read_model_evidence_strengths(evidence_material),
        assignment_codebooks=_load_assignment_codebooks(codebook_material),
        expected_assignment_protocol_id=args.assignment_protocol_id,
        expected_consent_version=args.consent_version,
        expected_blinding_version=args.blinding_version,
        primary_llm_source_id=args.primary_llm_source_id,
        bootstrap_replicates=args.bootstrap_replicates,
        minimum_clusters=args.minimum_clusters,
        seed=args.seed,
    )
    payload = {
        **result.to_dict(),
        "artifact_kind": "h8_human_model_comparison",
        "input_artifacts": {
            "human_responses_sha256": sha256(
                response_material
            ).hexdigest(),
            "assignment_codebook_sha256": sha256(
                codebook_material
            ).hexdigest(),
            "model_evidence_sha256": sha256(
                evidence_material
            ).hexdigest(),
        },
    }
    for label, (path, resolved, material) in snapshots.items():
        _verify_file_snapshot(
            path,
            resolved=resolved,
            material=material,
            label=label,
        )
    _atomic_write_new_text(args.output, _json(payload) + "\n")
    print(
        _json(
            {
                "output": str(args.output),
                "computed_status": result.computed_status,
                "criterion_met": result.criterion_met,
                "claim_status": result.claim_status,
            }
        )
    )
    return 0


def _parse_h8_sources(values: list[str]) -> dict[str, str]:
    sources: dict[str, str] = {}
    for value in values:
        source_id, separator, updater_id = value.partition("=")
        if not separator or not source_id.strip() or not updater_id.strip():
            raise ValueError(
                "--source must have the form SOURCE_ID=EXPERIMENT_A_UPDATER_ID"
            )
        if source_id in sources:
            raise ValueError(f"duplicate H8 source ID: {source_id}")
        sources[source_id] = updater_id
    return sources


def _read_jsonl_objects(
    path: Path | bytes,
) -> tuple[Mapping[str, Any], ...]:
    if isinstance(path, bytes):
        source_label = "<jsonl-bytes>"
        try:
            lines = path.decode("utf-8").splitlines()
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"{source_label}: input must be valid UTF-8"
            ) from exc
    else:
        source_label = str(path)
        lines = path.read_text(encoding="utf-8").splitlines()
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            decoded = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{source_label}:{line_number}: {exc}"
            ) from exc
        if not isinstance(decoded, Mapping):
            raise ValueError(
                f"{source_label}:{line_number}: row must be an object"
            )
        rows.append(decoded)
    return tuple(rows)


def _human_study_evidence_from_experiment_a(
    args: argparse.Namespace,
) -> int:
    valid, errors = verify_run(args.run_dir)
    if not valid:
        raise ValueError(
            "Experiment A source run is not verified: " + "; ".join(errors)
    )
    _reject_output_inside_input_run(args.run_dir, args.output)
    source_paths = {
        "Experiment A config": args.run_dir / "config.resolved.json",
        "Experiment A manifest": args.run_dir / "manifest.json",
        "Experiment A population": (
            args.run_dir / "population" / "users.jsonl"
        ),
        "Experiment A metrics": (
            args.run_dir / "metrics" / "experiment-a.jsonl"
        ),
    }
    source_snapshots = {
        label: (
            path,
            *_snapshot_regular_file(path, label=label),
        )
        for label, path in source_paths.items()
    }
    config_material = source_snapshots["Experiment A config"][2]
    manifest_material = source_snapshots["Experiment A manifest"][2]
    population_material = source_snapshots["Experiment A population"][2]
    metric_material = source_snapshots["Experiment A metrics"][2]
    config = json.loads(config_material)
    if config.get("experiment", {}).get("kind") != "provenance_audit":
        raise ValueError("H8 evidence conversion requires an Experiment A run")
    manifest = json.loads(manifest_material)
    source_run_id = manifest.get("run_id")
    if not isinstance(source_run_id, str) or not source_run_id:
        raise ValueError("Experiment A manifest has no valid run_id")
    test_pairs: set[tuple[str, str]] = set()
    for row in _read_jsonl_objects(population_material):
        if row.get("split") != "test":
            continue
        user_id = row.get("user_id")
        domain = row.get("domain")
        if isinstance(user_id, str) and isinstance(domain, str):
            test_pairs.add((user_id, domain))
    if not test_pairs:
        raise ValueError("Experiment A run has no retained test user/domain rows")
    metric_digest = sha256(metric_material).hexdigest()
    evidence = convert_experiment_a_metrics_to_model_evidence(
        _read_jsonl_objects(metric_material),
        source_run_id=source_run_id,
        source_artifact_sha256=metric_digest,
        sources=_parse_h8_sources(args.source),
        test_user_domain_pairs=test_pairs,
    )
    valid, errors = verify_run(args.run_dir)
    if not valid:
        raise ValueError(
            "Experiment A source run changed before H8 publication: "
            + "; ".join(errors)
        )
    for label, (path, resolved, material) in source_snapshots.items():
        _verify_file_snapshot(
            path,
            resolved=resolved,
            material=material,
            label=label,
        )
    _atomic_write_new_text(
        args.output,
        "".join(
            json.dumps(
                row.to_dict(),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            + "\n"
            for row in evidence
        ),
    )
    print(
        _json(
            {
                "output": str(args.output),
                "source_run_id": source_run_id,
                "source_artifact_sha256": metric_digest,
                "record_count": len(evidence),
                "source_ids": sorted({row.source_id for row in evidence}),
                "conditions": sorted({row.condition for row in evidence}),
                "volunteered_rows_synthesized": 0,
                "claim_status": "not_claimed",
            }
        )
    )
    return 0


def _correction_debt_run(args: argparse.Namespace) -> int:
    if args.output.exists():
        raise FileExistsError(
            f"correction-debt output already exists: {args.output}"
        )
    if args.truth_map is not None:
        decoded = json.loads(args.truth_map.read_text(encoding="utf-8"))
        if not isinstance(decoded, dict):
            raise ValueError("truth map must be a JSON object")
        truth = {str(key): value for key, value in decoded.items()}
    else:
        truth = {
            f"pair-{index + 1:04d}": (-1 if index % 2 else 1)
            for index in range(args.pairs)
        }
    result = run_correction_debt_experiment(
        pair_truth_directions=truth,
        stage_gate_authorized=args.stage_gate_authorized,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(_json(result.to_dict()) + "\n", encoding="utf-8")
    print(
        _json(
            {
                "output": str(args.output),
                "pairs": len(truth),
                "adapter_id": result.adapter_id,
                "scientific_claim_status": "diagnostic_reference_only",
            }
        )
    )
    return 0


def _control_study_analyze(args: argparse.Namespace) -> int:
    """Regenerate fixed bindings and atomically score imported responses."""

    if args.output.exists():
        raise FileExistsError(
            f"control-study output already exists: {args.output}"
        )
    _reject_output_inside_input_run(args.bindings, args.output)
    _reject_output_inside_input_run(args.responses, args.output)
    binding_resolved, binding_material = _snapshot_regular_file(
        args.bindings,
        label="control-study bindings",
    )
    response_resolved, response_material = _snapshot_regular_file(
        args.responses,
        label="control-study responses",
    )
    retained_bindings = read_control_request_bindings(binding_material)
    if not retained_bindings:
        raise ValueError("control-study bindings cannot be empty")
    updater_ids = {
        binding.llm_request.updater_id
        for binding in retained_bindings
    }
    views = {
        binding.llm_request.view
        for binding in retained_bindings
    }
    if len(updater_ids) != 1 or len(views) != 1:
        raise ValueError(
            "control-study bindings must use one updater ID and one view"
        )
    updater_id = next(iter(updater_ids))
    view = next(iter(views))
    plan = build_experiment_a_control_plan()
    exchange = build_control_llm_exchange(
        plan,
        updater_id=updater_id,
        view=view,
    )
    if retained_bindings != exchange.requests:
        raise ValueError(
            "control-study bindings do not exactly match the regenerated "
            "fixed plan and exchange"
        )
    report = execute_control_llm_exchange(
        plan,
        exchange,
        ReplayProvider(read_responses(response_material)),
        execution_mode="provider_replay",
        source_descriptor=args.source_descriptor,
    )
    payload = {
        "schema_version": 1,
        "analysis_kind": "experiment_a_control_provider_replay",
        "plan_sha256": plan.plan_sha256,
        "exchange_sha256": exchange.exchange_sha256,
        "binding_file_sha256": sha256(binding_material).hexdigest(),
        "response_file_sha256": sha256(response_material).hexdigest(),
        "request_count": len(exchange.requests),
        "report": report.to_dict(),
        "claim_status": "not_claimed",
    }
    _verify_file_snapshot(
        args.bindings,
        resolved=binding_resolved,
        material=binding_material,
        label="control-study bindings",
    )
    _verify_file_snapshot(
        args.responses,
        resolved=response_resolved,
        material=response_material,
        label="control-study responses",
    )
    _atomic_write_new_text(args.output, _json(payload) + "\n")
    print(
        _json(
            {
                "output": str(args.output),
                "plan_sha256": plan.plan_sha256,
                "request_count": len(exchange.requests),
                "criterion_pass_count": report.criterion_pass_count,
                "claim_status": "not_claimed",
            }
        )
    )
    return 0


def _control_study_h7_plan(args: argparse.Namespace) -> int:
    """Materialize the immutable direct-statement request corpus."""

    _reject_output_inside_input_run(args.run_dir, args.output_dir)
    source = load_verified_h7_source(args.run_dir)
    plan_path, bindings_path, requests_path = write_h7_plan_directory(
        args.output_dir,
        source.plan,
    )
    print(
        _json(
            {
                "output_dir": str(args.output_dir),
                "plan": str(plan_path),
                "bindings": str(bindings_path),
                "requests": str(requests_path),
                "source_run_id": source.source_run["run_id"],
                "plan_sha256": source.plan.plan_sha256,
                "case_count": len(source.plan.cases),
                "request_count": len(source.plan.requests),
                "independent_user_count": len(
                    {case.user_id for case in source.plan.cases}
                ),
                "claim_status": "not_claimed",
            }
        )
    )
    return 0


def _control_study_h7_review(args: argparse.Namespace) -> int:
    """Create a separate H7 review without mutating its source run."""

    if args.output.exists():
        raise FileExistsError(
            f"H7 volunteered review output already exists: {args.output}"
        )
    for input_path in (
        args.run_dir,
        args.plan_dir,
        args.responses,
        args.provider_audit,
    ):
        _reject_output_inside_input_run(input_path, args.output)
    source = load_verified_h7_source(args.run_dir)
    input_snapshots = snapshot_h7_review_inputs(
        args.plan_dir,
        args.responses,
        args.provider_audit,
    )
    payload = create_h7_volunteered_review(
        source,
        args.plan_dir,
        args.responses,
        args.provider_audit,
        input_snapshots=input_snapshots,
    )
    source.verify_unchanged()
    input_snapshots.verify_unchanged()
    _atomic_write_new_text(args.output, _json(payload) + "\n")
    recomputed = payload["recomputed_h7"]
    assert isinstance(recomputed, Mapping)
    print(
        _json(
            {
                "output": str(args.output),
                "source_run_id": source.source_run["run_id"],
                "plan_sha256": source.plan.plan_sha256,
                "review_sha256": payload["review_sha256"],
                "volunteered_pair_count": recomputed[
                    "volunteered_valid_learning"
                ]["pair_count"],
                "computed_status": recomputed["computed_status"],
                "criterion_met": recomputed["criterion_met"],
                "source_run_modified": False,
                "missing_values_imputed": False,
                "claim_status": "not_claimed",
            }
        )
    )
    return 0


def _control_study_h7_verify(args: argparse.Namespace) -> int:
    ok, errors = verify_h7_volunteered_review(
        args.run_dir,
        args.plan_dir,
        args.responses,
        args.provider_audit,
        args.review,
    )
    print(_json({"valid": ok, "errors": list(errors)}))
    return 0 if ok else 1


def _artifact_freeze(args: argparse.Namespace) -> int:
    print(_json(freeze_run(args.run_dir, args.archive).to_dict()))
    return 0


def _artifact_verify(args: argparse.Namespace) -> int:
    valid, errors = verify_frozen_artifact(args.archive)
    print(
        _json(
            {
                "archive": str(args.archive),
                "valid": valid,
                "errors": list(errors),
            }
        )
    )
    return 0 if valid else 1


def _artifact_compact(args: argparse.Namespace) -> int:
    print(
        _json(
            export_compact_analysis(
                args.run_dir,
                args.output_dir,
            ).to_dict()
        )
    )
    return 0


def _artifact_verify_compact(args: argparse.Namespace) -> int:
    valid, errors = verify_compact_analysis(args.bundle_dir)
    print(
        _json(
            {
                "bundle_dir": str(args.bundle_dir),
                "valid": valid,
                "errors": list(errors),
            }
        )
    )
    return 0 if valid else 1


def _experiment_c_decoder_import(args: argparse.Namespace) -> int:
    result = import_experiment_c_external_rescore(
        run_dir=args.run_dir,
        judgments_path=args.judgments,
        output_dir=args.output_dir,
        external_collection_dir=args.external_collection_dir,
        external_collection_provenance_mode=(
            args.external_collection_provenance_mode
        ),
        allow_reviewed_generic_decoders=(
            args.allow_reviewed_generic_decoders
        ),
    )
    print(_json(result))
    return 0


def _experiment_c_decoder_verify(args: argparse.Namespace) -> int:
    valid, errors = verify_experiment_c_external_rescore(
        args.review_dir,
        source_run_dir=args.source_run,
    )
    print(_json({"valid": valid, "errors": list(errors)}))
    return 0 if valid else 1


def _experiment_c_robustness_review(args: argparse.Namespace) -> int:
    result = create_experiment_c_multiseed_review(
        args.source_runs,
        args.output_dir,
    )
    print(_json(result))
    return 0


def _experiment_c_robustness_verify(args: argparse.Namespace) -> int:
    valid, errors = verify_experiment_c_multiseed_review(
        args.review_dir,
        source_run_dirs=args.source_runs,
    )
    print(_json({"valid": valid, "errors": list(errors)}))
    return 0 if valid else 1


def _gate6_review_build(args: argparse.Namespace) -> int:
    result = build_gate6_cross_run_review(
        declaration_path=args.declaration,
        output_dir=args.output_dir,
    )
    print(_json(result))
    return 0


def _gate6_review_verify(args: argparse.Namespace) -> int:
    valid, errors = verify_gate6_cross_run_review(
        args.review_dir,
        reverify_sources=args.reverify_sources,
    )
    print(
        _json(
            {
                "review_dir": str(args.review_dir),
                "sources_reverified": args.reverify_sources,
                "valid": valid,
                "errors": list(errors),
            }
        )
    )
    return 0 if valid else 1


def _study_templates() -> tuple[StudyItem, ...]:
    domain_templates = {
        "travel": (
            "The user chose the budget hotel.",
            "the user generally prefers budget hotels",
            'The user stated, "I generally prefer budget hotels." ',
        ),
        "writing": (
            "The user chose the concise draft.",
            "the user generally prefers concise writing",
            'The user stated, "I generally prefer concise writing." ',
        ),
    }
    condition_vignettes = {
        "balanced": "The user saw a balanced opposite-direction choice. ",
        "restricted": "Only same-direction alternatives were offered. ",
        "default": "Opposite alternatives remained available, but the chosen option was preselected. ",
        "suggested": "Opposite alternatives remained available, but the agent recommended the chosen option. ",
    }
    items = []
    for domain, (outcome, claim, volunteered) in domain_templates.items():
        for condition in CONDITIONS:
            setup = (
                volunteered
                if condition == "volunteered"
                else condition_vignettes[condition]
            )
            items.append(
                StudyItem(
                    item_id=f"{domain}-{condition}",
                    scenario_id=f"{domain}-anchor",
                    condition=condition,
                    vignette=setup + outcome,
                    preference_claim=claim,
                )
            )
    return tuple(items)


def _human_study_generate(args: argparse.Namespace) -> int:
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError(
            "human-study output directory must be absent or empty"
        )
    output.mkdir(parents=True, exist_ok=True)
    items = _study_templates()
    participant = blind_and_order_items(
        items, assignment_id=args.assignment_id, seed=args.seed
    )
    participant_path = output / "participant-items.jsonl"
    participant_path.write_text(
        "".join(
            json.dumps(
                {
                    "schema_version": 1,
                    "assignment_id": args.assignment_id,
                    "assignment_protocol_id": args.assignment_protocol_id,
                    "consent_version": args.consent_version,
                    "blinding_version": args.blinding_version,
                    "comprehension_check_id": (
                        args.comprehension_check_id
                    ),
                    **item,
                },
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            + "\n"
            for item in participant
        ),
        encoding="utf-8",
    )
    codebook_path = output / "researcher-codebook.json"
    codebook_path.write_text(
        _json(
            {
                "schema_version": 1,
                "assignment_id": args.assignment_id,
                "assignment_protocol_id": args.assignment_protocol_id,
                "consent_version": args.consent_version,
                "blinding_version": args.blinding_version,
                "comprehension_check_id": args.comprehension_check_id,
                "items_by_display_id": build_assignment_codebook(
                    items,
                    assignment_id=args.assignment_id,
                    seed=args.seed,
                ),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    order_path = output / "order-manifest.json"
    order_path.write_text(
        _json(
            {
                "schema_version": 1,
                "assignment_id": args.assignment_id,
                "assignment_protocol_id": args.assignment_protocol_id,
                "seed": args.seed,
                "display_ids": [item["display_id"] for item in participant],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    response_schema_path = output / "human-rating.schema.json"
    response_schema_path.write_text(
        _json(SCHEMAS["human-rating"]) + "\n",
        encoding="utf-8",
    )
    readme_path = output / "README.md"
    readme_path.write_text(
        "# Human-study packet\n\n"
        "This packet is a material-generation aid, not an ethics approval or a "
        "deployed survey. Obtain the required review, consent, privacy, and "
        "compensation approvals before collecting participant data.\n\n"
        f"Assignment protocol: `{args.assignment_protocol_id}`  \n"
        f"Consent version: `{args.consent_version}`  \n"
        f"Blinding version: `{args.blinding_version}`  \n"
        f"Comprehension check: `{args.comprehension_check_id}`\n",
        encoding="utf-8",
    )
    retained = (
        participant_path,
        codebook_path,
        order_path,
        response_schema_path,
        readme_path,
    )
    (output / "packet-manifest.json").write_text(
        _json(
            {
                "schema_version": 1,
                "assignment_id": args.assignment_id,
                "assignment_protocol_id": args.assignment_protocol_id,
                "files": {
                    path.name: file_sha256(path)
                    for path in retained
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        _json(
            {
                "output_dir": str(output),
                "items": len(items),
                "participant_file": participant_path.name,
                "manifest": "packet-manifest.json",
            }
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cape-loop",
        description="Causal-provenance evaluation for persistent LLM profiles.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    doctor = commands.add_parser("doctor", help="check the local offline runtime")
    doctor.set_defaults(handler=_doctor)

    demo = commands.add_parser(
        "demo",
        help="small explanatory workflows that are not paper evidence",
    )
    demo_commands = demo.add_subparsers(
        dest="demo_command",
        required=True,
    )
    one_scenario = demo_commands.add_parser(
        "one-scenario",
        help=(
            "run one frozen scenario and exactly one physical OpenRouter "
            "profile-update attempt"
        ),
    )
    one_scenario.add_argument(
        "output_dir",
        type=Path,
        help="new directory for the readable log and provider journals",
    )
    one_scenario.add_argument(
        "--scenario-catalog",
        type=Path,
        default=Path("data/scenarios/scenario-catalog-v1.json"),
    )
    one_scenario.add_argument(
        "--conversation-bank",
        type=Path,
        default=Path("data/scenarios/conversation-templates-v1.json"),
    )
    one_scenario.add_argument(
        "--scenario-id",
        default="travel-scenario-atlas-lodging-price-01",
        help="one exact frozen scenario ID",
    )
    one_scenario.add_argument(
        "--mechanism",
        choices=("balanced", "restricted", "default", "suggested"),
        default="balanced",
    )
    one_scenario.add_argument("--seed", type=int, default=1729)
    one_scenario.add_argument(
        "--model",
        default=OPENROUTER_EXAMPLE_MODEL,
        help=(
            "exact OpenRouter author/model slug; change only this value to "
            "test another model"
        ),
    )
    one_scenario.add_argument(
        "--reasoning-effort",
        choices=(
            "",
            "none",
            "minimal",
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
        ),
        default="",
    )
    one_scenario.add_argument(
        "--upstream-provider",
        default="",
        help="optional exact OpenRouter upstream provider constraint",
    )
    one_scenario.add_argument(
        "--api-key-env",
        default="OPENROUTER_API_KEY",
    )
    one_scenario.add_argument(
        "--timeout-seconds",
        type=float,
        default=120.0,
    )
    one_scenario.add_argument(
        "--zdr",
        action="store_true",
        help="require an OpenRouter zero-data-retention endpoint",
    )
    one_scenario.add_argument(
        "--execute-live",
        action="store_true",
        help="authorize this one paid OpenRouter attempt",
    )
    one_scenario.set_defaults(handler=_demo_one_scenario)

    config = commands.add_parser("config", help="configuration commands")
    config_commands = config.add_subparsers(dest="config_command", required=True)
    validate = config_commands.add_parser("validate", help="validate a TOML config")
    validate.add_argument("config", type=Path)
    validate.set_defaults(handler=_config_validate)

    run = commands.add_parser("run", help="run an experiment configuration")
    run.add_argument("config", type=Path)
    run.add_argument("--output-root", type=Path)
    run.add_argument(
        "--allow-existing",
        action="store_true",
        help="reuse the deterministic run directory if it already exists",
    )
    run.add_argument(
        "--execute-live",
        action="store_true",
        help=(
            "authorize configured live provider calls subject to hard budgets"
        ),
    )
    run.add_argument(
        "--resume-failed-live",
        action="store_true",
        help=(
            "preserve a failed live artifact and resume provider journals "
            "into a fresh attempt at the same deterministic run path"
        ),
    )
    run.set_defaults(handler=_run)

    verify = commands.add_parser("verify", help="verify a run's SHA-256 manifest")
    verify.add_argument("run_dir", type=Path)
    verify.set_defaults(handler=_verify)

    gate_review = commands.add_parser(
        "gate-review",
        help="append-only external-evidence reviews for scientific gates",
    )
    gate_review_commands = gate_review.add_subparsers(
        dest="gate_review_command",
        required=True,
    )
    import_native = gate_review_commands.add_parser(
        "import-native",
        help=(
            "bind reviewed external decoders and a fully validated OpenAI "
            "native-action collection to a verified completed Experiment B run"
        ),
    )
    import_native.add_argument("run_dir", type=Path)
    import_native.add_argument("requests", type=Path)
    import_native.add_argument("judgments", type=Path)
    import_native.add_argument("truth_labels", type=Path)
    import_native.add_argument(
        "native_collection_dir",
        type=Path,
        help=(
            "complete gpt-5.6-sol/medium OpenAI collection directory, not a "
            "standalone native-actions.jsonl file"
        ),
    )
    import_native.add_argument("source_review", type=Path)
    import_native.add_argument("output_dir", type=Path)
    external_provenance = import_native.add_mutually_exclusive_group(
        required=True
    )
    external_provenance.add_argument(
        "--external-collection-dir",
        type=Path,
        action=_ExternalCollectionDirAction,
        provenance_mode=DIRECT_FIRST_PARTY_COLLECTION_PROVENANCE,
        help=(
            "complete official Anthropic/Gemini distinct-decoder collection; "
            "its judgments must be byte-identical to JUDGMENTS"
        ),
    )
    external_provenance.add_argument(
        "--openrouter-collection-dir",
        dest="external_collection_dir",
        type=Path,
        action=_ExternalCollectionDirAction,
        provenance_mode=OPENROUTER_COLLECTION_PROVENANCE,
        help=(
            "complete audited Claude/Gemini OpenRouter collection; gateway "
            "provenance is validated without claiming first-party origin"
        ),
    )
    external_provenance.add_argument(
        "--allow-reviewed-generic-decoders",
        action="store_true",
        help=(
            "explicitly admit generic or manual judgments under the supplied "
            "responsible-researcher source review without asserting official "
            "provider-collection provenance"
        ),
    )
    import_native.set_defaults(external_collection_provenance_mode=None)
    import_native.set_defaults(handler=_gate_review_import_native)
    verify_review = gate_review_commands.add_parser(
        "verify",
        help="verify a checksum-bound Gate 4 review artifact",
    )
    verify_review.add_argument("review_dir", type=Path)
    verify_review.set_defaults(handler=_gate_review_verify)

    gate6_review = commands.add_parser(
        "gate6-review",
        help=(
            "immutable offline cross-family Gate 6 robustness review"
        ),
    )
    gate6_review_commands = gate6_review.add_subparsers(
        dest="gate6_review_command",
        required=True,
    )
    build_gate6 = gate6_review_commands.add_parser(
        "build",
        help=(
            "bind explicit sensitivity/Experiment A run pairs and recompute "
            "all six Gate 6 clauses"
        ),
    )
    build_gate6.add_argument("declaration", type=Path)
    build_gate6.add_argument("output_dir", type=Path)
    build_gate6.set_defaults(handler=_gate6_review_build)
    verify_gate6 = gate6_review_commands.add_parser(
        "verify",
        help=(
            "verify a Gate 6 review, optionally re-reading every source run"
        ),
    )
    verify_gate6.add_argument("review_dir", type=Path)
    verify_gate6.add_argument(
        "--reverify-sources",
        action="store_true",
        help=(
            "re-verify all declared source runs and compare recomputed evidence"
        ),
    )
    verify_gate6.set_defaults(handler=_gate6_review_verify)

    experiment_c_decoder = commands.add_parser(
        "experiment-c-decoder",
        help=(
            "append-only external-decoder rescoring for a completed "
            "Experiment C run"
        ),
    )
    experiment_c_decoder_commands = experiment_c_decoder.add_subparsers(
        dest="experiment_c_decoder_command",
        required=True,
    )
    import_c_decoder = experiment_c_decoder_commands.add_parser(
        "import",
        help=(
            "validate exactly two external decoder families, calibrate on "
            "development rows only, and rerun C rankings, ESR, and Gate 5"
        ),
    )
    import_c_decoder.add_argument("run_dir", type=Path)
    import_c_decoder.add_argument("judgments", type=Path)
    import_c_decoder.add_argument("output_dir", type=Path)
    c_decoder_provenance = import_c_decoder.add_mutually_exclusive_group(
        required=True,
    )
    c_decoder_provenance.add_argument(
        "--external-collection-dir",
        type=Path,
        action=_ExternalCollectionDirAction,
        provenance_mode=DIRECT_FIRST_PARTY_COLLECTION_PROVENANCE,
        help=(
            "validate the complete locked first-party Anthropic/Gemini "
            "collection that produced JUDGMENTS"
        ),
    )
    c_decoder_provenance.add_argument(
        "--openrouter-collection-dir",
        dest="external_collection_dir",
        type=Path,
        action=_ExternalCollectionDirAction,
        provenance_mode=OPENROUTER_COLLECTION_PROVENANCE,
        help=(
            "validate the complete audited Claude/Gemini OpenRouter "
            "collection that produced JUDGMENTS"
        ),
    )
    c_decoder_provenance.add_argument(
        "--allow-reviewed-generic-decoders",
        action="store_true",
        help=(
            "accept caller-declared family/source metadata without claiming "
            "provider-validated provenance"
        ),
    )
    import_c_decoder.set_defaults(
        external_collection_provenance_mode=None
    )
    import_c_decoder.set_defaults(handler=_experiment_c_decoder_import)
    verify_c_decoder = experiment_c_decoder_commands.add_parser(
        "verify",
        help="verify an immutable Experiment C external-decoder rescore",
    )
    verify_c_decoder.add_argument("review_dir", type=Path)
    verify_c_decoder.add_argument(
        "--source-run",
        type=Path,
        help=(
            "optionally re-verify the completed source run and its exact "
            "cryptographic binding"
        ),
    )
    verify_c_decoder.set_defaults(handler=_experiment_c_decoder_verify)

    experiment_c_robustness = commands.add_parser(
        "experiment-c-robustness",
        help=(
            "immutable offline cross-seed ranking robustness review for "
            "completed Experiment C runs"
        ),
    )
    robustness_commands = experiment_c_robustness.add_subparsers(
        dest="experiment_c_robustness_command",
        required=True,
    )
    review_robustness = robustness_commands.add_parser(
        "review",
        help=(
            "compare verified distinct-seed clustered-bootstrap ranking "
            "results without making a scientific claim"
        ),
    )
    review_robustness.add_argument("output_dir", type=Path)
    review_robustness.add_argument(
        "source_runs",
        type=Path,
        nargs="+",
        metavar="SOURCE_RUN",
        help="two or more compatible completed evaluation_validity runs",
    )
    review_robustness.set_defaults(
        handler=_experiment_c_robustness_review
    )
    verify_robustness = robustness_commands.add_parser(
        "verify",
        help="verify the review and optionally re-bind all source runs",
    )
    verify_robustness.add_argument("review_dir", type=Path)
    verify_robustness.add_argument(
        "--source-run",
        dest="source_runs",
        type=Path,
        action="append",
        help=(
            "repeat for every source to re-verify exact source bindings"
        ),
    )
    verify_robustness.set_defaults(
        handler=_experiment_c_robustness_verify
    )

    schema = commands.add_parser("schema", help="JSON Schema commands")
    schema_commands = schema.add_subparsers(dest="schema_command", required=True)
    export = schema_commands.add_parser("export", help="export public JSON Schemas")
    export.add_argument("destination", type=Path, nargs="?", default=Path("schemas"))
    export.set_defaults(handler=_schema_export)

    conversations = commands.add_parser(
        "conversations",
        help="author and validate frozen hybrid conversation templates",
    )
    conversation_commands = conversations.add_subparsers(
        dest="conversation_command",
        required=True,
    )
    generate_conversations = conversation_commands.add_parser(
        "generate-openrouter",
        help=(
            "make one OpenRouter authoring call per scenario; the model "
            "writes language templates but never chooses user actions"
        ),
    )
    generate_conversations.add_argument("catalog", type=Path)
    generate_conversations.add_argument("output", type=Path)
    generate_conversations.add_argument(
        "--log",
        type=Path,
        help=(
            "readable JSONL request/result log; defaults beside OUTPUT"
        ),
    )
    generate_conversations.add_argument(
        "--bank-id",
        default="cape-loop-conversation-templates-v1",
    )
    generate_conversations.add_argument(
        "--model",
        default="anthropic/claude-sonnet-5",
        help="one pinned OpenRouter author/model slug",
    )
    generate_conversations.add_argument(
        "--upstream-provider",
        default="",
        help="optional exact OpenRouter provider route",
    )
    generate_conversations.add_argument(
        "--timeout-seconds",
        type=float,
        default=60.0,
    )
    generate_conversations.add_argument(
        "--max-output-tokens",
        type=int,
        default=1200,
    )
    generate_conversations.add_argument(
        "--max-requests",
        type=int,
        default=32,
    )
    generate_conversations.add_argument(
        "--max-total-tokens",
        type=int,
        default=500_000,
    )
    generate_conversations.add_argument(
        "--execute-live",
        action="store_true",
        help="authorize the paid OpenRouter authoring calls",
    )
    generate_conversations.set_defaults(
        handler=_conversations_generate_openrouter
    )

    llm = commands.add_parser("llm", help="provider-neutral LLM exchange")
    llm_commands = llm.add_subparsers(dest="llm_command", required=True)
    models = llm_commands.add_parser(
        "models", help="show the versioned default paper model suite"
    )
    models.set_defaults(handler=_llm_models)
    validate_llm = llm_commands.add_parser(
        "validate", help="strictly validate imported JSONL responses"
    )
    validate_llm.add_argument("responses", type=Path)
    validate_llm.set_defaults(handler=_llm_validate)

    def add_openai_arguments(
        command: argparse.ArgumentParser,
        *,
        include_role: bool = True,
    ) -> None:
        if include_role:
            command.add_argument(
                "--role",
                choices=tuple(DEFAULT_OPENAI_MODEL_ROLES),
                default="primary",
            )
            command.add_argument(
                "--model",
                default="",
                help="explicit model override; empty uses the selected role",
            )
            command.add_argument(
                "--reasoning-effort",
                choices=("", "none", "low", "medium", "high", "xhigh", "max"),
                default="",
            )
        command.add_argument("--api-key-env", default="OPENAI_API_KEY")
        command.add_argument("--base-url", default="https://api.openai.com")
        command.add_argument(
            "--allow-custom-base-url",
            action="store_true",
            help=(
                "allow sending the configured API key to a reviewed "
                "non-OpenAI HTTPS endpoint"
            ),
        )
        command.add_argument("--timeout-seconds", type=float, default=180.0)
        command.add_argument("--max-retries", type=int, default=4)
        command.add_argument("--max-output-tokens", type=int, default=4096)
        command.add_argument("--max-requests", type=int, default=100)
        command.add_argument("--max-total-tokens", type=int, default=500_000)

    def add_openrouter_arguments(
        command: argparse.ArgumentParser,
    ) -> None:
        command.add_argument(
            "--model",
            default=OPENROUTER_EXAMPLE_MODEL,
            help=(
                "exact OpenRouter author/model slug; change this one value "
                "to switch models"
            ),
        )
        command.add_argument(
            "--reasoning-effort",
            choices=(
                "",
                "none",
                "minimal",
                "low",
                "medium",
                "high",
                "xhigh",
                "max",
            ),
            default="",
            help="omit by default so models without reasoning controls work",
        )
        command.add_argument(
            "--upstream-provider",
            default="",
            help=(
                "optional exact OpenRouter provider slug to place in both "
                "provider.only and provider.order"
            ),
        )
        command.add_argument(
            "--allow-fallbacks",
            action="store_true",
            help="allow OpenRouter to try another endpoint for the same model",
        )
        command.add_argument(
            "--allow-unsupported-parameters",
            action="store_true",
            help=(
                "permit routes that may ignore requested parameters; "
                "not recommended for research runs"
            ),
        )
        command.add_argument(
            "--data-collection",
            choices=("deny", "allow"),
            default="deny",
        )
        command.add_argument(
            "--zdr",
            action="store_true",
            help="require an OpenRouter zero-data-retention endpoint",
        )
        command.add_argument(
            "--http-referer",
            default="",
            help="optional public repository/site URL for app attribution",
        )
        command.add_argument(
            "--app-title",
            default="CAPE-Loop",
            help="optional OpenRouter app-attribution title",
        )
        command.add_argument(
            "--api-key-env",
            default="OPENROUTER_API_KEY",
        )
        command.add_argument(
            "--base-url",
            default="https://openrouter.ai/api",
        )
        command.add_argument(
            "--allow-custom-base-url",
            action="store_true",
            help=(
                "allow sending a dedicated non-default credential to a "
                "reviewed non-OpenRouter HTTPS endpoint"
            ),
        )
        command.add_argument("--timeout-seconds", type=float, default=180.0)
        command.add_argument("--max-retries", type=int, default=2)
        command.add_argument("--max-output-tokens", type=int, default=4096)
        command.add_argument("--max-requests", type=int, default=100)
        command.add_argument("--max-total-tokens", type=int, default=500_000)

    plan_llm = llm_commands.add_parser(
        "plan",
        help="dry-run request bodies and conservative budgets without a key",
    )
    plan_llm.add_argument("requests", type=Path)
    add_openai_arguments(plan_llm)
    plan_llm.set_defaults(handler=_llm_plan)

    execute_llm = llm_commands.add_parser(
        "execute-openai",
        help="resumably execute a static request JSONL with OpenAI",
    )
    execute_llm.add_argument("requests", type=Path)
    execute_llm.add_argument("responses", type=Path)
    execute_llm.add_argument("audit", type=Path)
    execute_llm.add_argument("--execute-live", action="store_true")
    add_openai_arguments(execute_llm)
    execute_llm.set_defaults(handler=_llm_execute)

    plan_openrouter = llm_commands.add_parser(
        "plan-openrouter",
        help=(
            "dry-run OpenRouter request bodies, routes, and budgets without "
            "reading a key"
        ),
    )
    plan_openrouter.add_argument("requests", type=Path)
    add_openrouter_arguments(plan_openrouter)
    plan_openrouter.set_defaults(handler=_llm_plan_openrouter)

    execute_openrouter = llm_commands.add_parser(
        "execute-openrouter",
        help="resumably execute static request JSONL through OpenRouter",
    )
    execute_openrouter.add_argument("requests", type=Path)
    execute_openrouter.add_argument("responses", type=Path)
    execute_openrouter.add_argument("audit", type=Path)
    execute_openrouter.add_argument("--execute-live", action="store_true")
    add_openrouter_arguments(execute_openrouter)
    execute_openrouter.set_defaults(handler=_llm_execute_openrouter)

    evaluation_suite = llm_commands.add_parser(
        "evaluation-suite",
        help=(
            "plan or explicitly execute isolated primary and replication "
            "paper configs"
        ),
    )
    evaluation_suite.add_argument("primary_config", type=Path)
    evaluation_suite.add_argument("replication_config", type=Path)
    evaluation_suite.add_argument("--output-root", type=Path)
    evaluation_suite.add_argument("--index", type=Path)
    evaluation_suite.add_argument(
        "--execute-live",
        action="store_true",
        help=(
            "authorize both role-specific live runs under their own hard "
            "budgets"
        ),
    )
    evaluation_suite.add_argument(
        "--allow-existing",
        action="store_true",
        help="reuse only complete verified role artifacts",
    )
    evaluation_suite.set_defaults(handler=_llm_evaluation_suite)

    artifact = commands.add_parser(
        "artifact",
        help="freeze, compact, and verify run-derived artifacts",
    )
    artifact_commands = artifact.add_subparsers(
        dest="artifact_command", required=True
    )
    freeze = artifact_commands.add_parser(
        "freeze", help="freeze a completed verified run to deterministic tar"
    )
    freeze.add_argument("run_dir", type=Path)
    freeze.add_argument("archive", type=Path)
    freeze.set_defaults(handler=_artifact_freeze)
    verify_artifact = artifact_commands.add_parser(
        "verify", help="verify a frozen tar and its sidecar"
    )
    verify_artifact.add_argument("archive", type=Path)
    verify_artifact.set_defaults(handler=_artifact_verify)
    compact = artifact_commands.add_parser(
        "compact",
        help=(
            "export checksum-bound row-level analysis data without changing "
            "the completed source run"
        ),
    )
    compact.add_argument("run_dir", type=Path)
    compact.add_argument("output_dir", type=Path)
    compact.set_defaults(handler=_artifact_compact)
    verify_compact = artifact_commands.add_parser(
        "verify-compact",
        help="verify a compact analysis bundle",
    )
    verify_compact.add_argument("bundle_dir", type=Path)
    verify_compact.set_defaults(handler=_artifact_verify_compact)

    decoder = commands.add_parser(
        "decoder-study",
        help="external native-state decoder exchange and analysis",
    )
    decoder_commands = decoder.add_subparsers(
        dest="decoder_command", required=True
    )
    validate_decoder = decoder_commands.add_parser(
        "validate", help="validate decoder hashes, blinding, and source design"
    )
    validate_decoder.add_argument("requests", type=Path)
    validate_decoder.add_argument("judgments", type=Path)
    validate_decoder.add_argument("--minimum-sources", type=int, default=2)
    validate_decoder.add_argument("--allow-same-family", action="store_true")
    validate_decoder.set_defaults(handler=_decoder_validate)
    analyze_decoder = decoder_commands.add_parser(
        "analyze",
        help="fit development calibration and analyze held-out test judgments",
    )
    analyze_decoder.add_argument("requests", type=Path)
    analyze_decoder.add_argument("judgments", type=Path)
    analyze_decoder.add_argument("truth_labels", type=Path)
    analyze_decoder.add_argument("--reliability-bins", type=int, default=10)
    analyze_decoder.add_argument("--output", type=Path)
    analyze_decoder.set_defaults(handler=_decoder_analyze)
    plan_decoder = decoder_commands.add_parser(
        "plan-openai",
        help="dry-run the two-source decoder budget without reading a key",
    )
    plan_decoder.add_argument("requests", type=Path)
    plan_decoder.add_argument(
        "--roles",
        nargs="+",
        choices=("replication", "decoder"),
        default=["replication", "decoder"],
    )
    add_openai_arguments(plan_decoder, include_role=False)
    plan_decoder.set_defaults(handler=_decoder_plan_openai)
    execute_decoder = decoder_commands.add_parser(
        "execute-openai",
        help=(
            "collect two blinded OpenAI decoder sources resumably; this "
            "does not establish statistical independence"
        ),
    )
    execute_decoder.add_argument("requests", type=Path)
    execute_decoder.add_argument("output_dir", type=Path)
    execute_decoder.add_argument(
        "--roles",
        nargs="+",
        choices=("replication", "decoder"),
        default=["replication", "decoder"],
    )
    execute_decoder.add_argument("--execute-live", action="store_true")
    add_openai_arguments(execute_decoder, include_role=False)
    execute_decoder.set_defaults(handler=_decoder_execute_openai)

    plan_openrouter_decoder = decoder_commands.add_parser(
        "plan-openrouter",
        help=(
            "dry-run routed OpenRouter decoder sources without reading a key"
        ),
    )
    plan_openrouter_decoder.add_argument("requests", type=Path)
    add_openrouter_arguments(plan_openrouter_decoder)
    plan_openrouter_decoder.add_argument(
        "--additional-model",
        action="append",
        default=[],
        help=(
            "add another exact OpenRouter model as a separately journaled "
            "decoder source; repeat as needed"
        ),
    )
    plan_openrouter_decoder.add_argument(
        "--model-reasoning-effort",
        action="append",
        default=[],
        metavar="MODEL=EFFORT",
        help=(
            "override reasoning effort for one selected model; the default "
            "pair uses Claude=low and Gemini=minimal"
        ),
    )
    plan_openrouter_decoder.set_defaults(
        handler=_decoder_plan_openrouter,
        model=None,
        max_retries=OPENROUTER_DECODER_MAX_RETRIES,
        max_output_tokens=OPENROUTER_DECODER_MAX_OUTPUT_TOKENS,
        max_requests=OPENROUTER_DECODER_MAX_REQUESTS,
        max_total_tokens=OPENROUTER_DECODER_MAX_TOTAL_TOKENS,
    )

    execute_openrouter_decoder = decoder_commands.add_parser(
        "execute-openrouter",
        help=(
            "collect routed OpenRouter decoder sources; these remain "
            "ineligible for strict first-party Gate 4 provenance"
        ),
    )
    execute_openrouter_decoder.add_argument("requests", type=Path)
    execute_openrouter_decoder.add_argument("output_dir", type=Path)
    execute_openrouter_decoder.add_argument(
        "--execute-live",
        action="store_true",
    )
    add_openrouter_arguments(execute_openrouter_decoder)
    execute_openrouter_decoder.add_argument(
        "--additional-model",
        action="append",
        default=[],
        help=(
            "add another exact OpenRouter model as a separately journaled "
            "decoder source; repeat as needed"
        ),
    )
    execute_openrouter_decoder.add_argument(
        "--model-reasoning-effort",
        action="append",
        default=[],
        metavar="MODEL=EFFORT",
        help=(
            "override reasoning effort for one selected model; the default "
            "pair uses Claude=low and Gemini=minimal"
        ),
    )
    execute_openrouter_decoder.set_defaults(
        handler=_decoder_execute_openrouter,
        model=None,
        max_retries=OPENROUTER_DECODER_MAX_RETRIES,
        max_output_tokens=OPENROUTER_DECODER_MAX_OUTPUT_TOKENS,
        max_requests=OPENROUTER_DECODER_MAX_REQUESTS,
        max_total_tokens=OPENROUTER_DECODER_MAX_TOTAL_TOKENS,
    )

    def add_distinct_decoder_arguments(
        command: argparse.ArgumentParser,
    ) -> None:
        command.add_argument(
            "--anthropic-model",
            default=ANTHROPIC_DEFAULT_MODEL,
        )
        command.add_argument(
            "--gemini-model",
            default=GEMINI_DEFAULT_MODEL,
        )
        command.add_argument(
            "--anthropic-api-key-env",
            default="ANTHROPIC_API_KEY",
        )
        command.add_argument(
            "--gemini-api-key-env",
            default="GEMINI_API_KEY",
        )
        command.add_argument(
            "--anthropic-base-url",
            default=ANTHROPIC_OFFICIAL_ORIGIN,
        )
        command.add_argument(
            "--gemini-base-url",
            default=GEMINI_OFFICIAL_ORIGIN,
        )
        command.add_argument(
            "--allow-custom-anthropic-base-url",
            action="store_true",
            help=(
                "allow a reviewed non-Anthropic HTTPS origin; also requires "
                "a dedicated non-default credential environment variable"
            ),
        )
        command.add_argument(
            "--allow-custom-gemini-base-url",
            action="store_true",
            help=(
                "allow a reviewed non-Google HTTPS origin; also requires a "
                "dedicated non-default credential environment variable"
            ),
        )
        command.add_argument("--timeout-seconds", type=float, default=180.0)
        command.add_argument(
            "--max-retries",
            type=int,
            default=0,
            help=(
                "retry count per logical request; strict Gate 4 defaults to "
                "zero so the complete corpus fits the approved physical-"
                "attempt ceiling"
            ),
        )
        command.add_argument("--max-output-tokens", type=int, default=1024)
        command.add_argument(
            "--max-requests-per-source",
            type=int,
            default=900,
        )
        command.add_argument(
            "--max-total-tokens-per-source",
            type=int,
            default=6_000_000,
        )

    plan_distinct_decoder = decoder_commands.add_parser(
        "plan-distinct",
        help=(
            "plan the Anthropic Sonnet and Google Gemini decoder pair "
            "without reading either key"
        ),
    )
    plan_distinct_decoder.add_argument("requests", type=Path)
    plan_distinct_decoder.add_argument("--output", type=Path)
    add_distinct_decoder_arguments(plan_distinct_decoder)
    plan_distinct_decoder.set_defaults(handler=_decoder_plan_distinct)

    execute_distinct_decoder = decoder_commands.add_parser(
        "execute-distinct",
        help=(
            "resumably collect the two first-party, distinct-family decoder "
            "sources; researcher source review remains required"
        ),
    )
    execute_distinct_decoder.add_argument("requests", type=Path)
    execute_distinct_decoder.add_argument("output_dir", type=Path)
    execute_distinct_decoder.add_argument(
        "--execute-live",
        action="store_true",
    )
    add_distinct_decoder_arguments(execute_distinct_decoder)
    execute_distinct_decoder.set_defaults(handler=_decoder_execute_distinct)

    native_action = commands.add_parser(
        "native-action",
        help=(
            "record terminal actions produced directly from retained native "
            "memory for Gate 4"
        ),
    )
    native_action_commands = native_action.add_subparsers(
        dest="native_action_command",
        required=True,
    )
    plan_native_action = native_action_commands.add_parser(
        "plan-openai",
        help=(
            "plan exact gpt-5.6-sol native-state actions without reading a key"
        ),
    )
    plan_native_action.add_argument("run_dir", type=Path)
    plan_native_action.add_argument("--output", type=Path)
    add_openai_arguments(plan_native_action, include_role=False)
    plan_native_action.add_argument(
        "--model",
        default=DEFAULT_OPENAI_MODEL_ROLES["primary"].model,
    )
    plan_native_action.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high", "xhigh", "max"),
        default=DEFAULT_OPENAI_MODEL_ROLES["primary"].reasoning_effort,
    )
    plan_native_action.set_defaults(
        handler=_native_action_plan_openai,
        max_retries=0,
        max_requests=900,
        max_total_tokens=6_000_000,
    )

    execute_native_action = native_action_commands.add_parser(
        "execute-openai",
        help=(
            "resumably collect schema-bound native actions with explicit "
            "live authorization"
        ),
    )
    execute_native_action.add_argument("run_dir", type=Path)
    execute_native_action.add_argument("output_dir", type=Path)
    execute_native_action.add_argument("--execute-live", action="store_true")
    add_openai_arguments(execute_native_action, include_role=False)
    execute_native_action.add_argument(
        "--model",
        default=DEFAULT_OPENAI_MODEL_ROLES["primary"].model,
    )
    execute_native_action.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high", "xhigh", "max"),
        default=DEFAULT_OPENAI_MODEL_ROLES["primary"].reasoning_effort,
    )
    execute_native_action.set_defaults(
        handler=_native_action_execute_openai,
        max_retries=0,
        max_requests=900,
        max_total_tokens=6_000_000,
    )

    human = commands.add_parser("human-study", help="human-study material helpers")
    human_commands = human.add_subparsers(dest="human_command", required=True)
    generate = human_commands.add_parser(
        "generate", help="generate a blinded pragmatic-validation packet"
    )
    generate.add_argument("output_dir", type=Path)
    generate.add_argument("--assignment-id", default="template")
    generate.add_argument("--seed", type=int, default=1729)
    generate.add_argument(
        "--assignment-protocol-id",
        default="cape-loop-human-assignment-v1",
    )
    generate.add_argument("--consent-version", default="consent-v1")
    generate.add_argument("--blinding-version", default="blinding-v1")
    generate.add_argument(
        "--comprehension-check-id",
        default="comprehension-v1",
    )
    generate.set_defaults(handler=_human_study_generate)
    analyze_human = human_commands.add_parser(
        "analyze",
        help="validate and analyze collected de-identified ratings",
    )
    analyze_human.add_argument("responses", type=Path)
    analyze_human.add_argument("codebook", type=Path)
    analyze_human.add_argument(
        "--assignment-protocol-id",
        default="cape-loop-human-assignment-v1",
    )
    analyze_human.add_argument("--consent-version", default="consent-v1")
    analyze_human.add_argument("--blinding-version", default="blinding-v1")
    analyze_human.add_argument(
        "--bootstrap-replicates", type=int, default=1000
    )
    analyze_human.add_argument("--seed", type=int, default=1729)
    analyze_human.add_argument("--output", type=Path)
    analyze_human.set_defaults(handler=_human_study_analyze)
    compare_human = human_commands.add_parser(
        "compare",
        help=(
            "atomically compare de-identified human ratings with fitted-aware "
            "and model evidence for H8"
        ),
    )
    compare_human.add_argument("responses", type=Path)
    compare_human.add_argument("codebook", type=Path)
    compare_human.add_argument("model_evidence", type=Path)
    compare_human.add_argument("output", type=Path)
    compare_human.add_argument(
        "--primary-llm-source-id",
        required=True,
        help=(
            "ordinary-LLM evidence source selected by the external "
            "preregistration"
        ),
    )
    compare_human.add_argument(
        "--assignment-protocol-id",
        default="cape-loop-human-assignment-v1",
    )
    compare_human.add_argument("--consent-version", default="consent-v1")
    compare_human.add_argument("--blinding-version", default="blinding-v1")
    compare_human.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=2000,
    )
    compare_human.add_argument("--minimum-clusters", type=int, default=8)
    compare_human.add_argument("--seed", type=int, default=1729)
    compare_human.set_defaults(handler=_human_study_compare)
    evidence_from_a = human_commands.add_parser(
        "evidence-from-experiment-a",
        help=(
            "convert verified test-only controlled-anchor Experiment A "
            "metrics into the strict H8 evidence exchange"
        ),
    )
    evidence_from_a.add_argument("run_dir", type=Path)
    evidence_from_a.add_argument("output", type=Path)
    evidence_from_a.add_argument(
        "--source",
        action="append",
        required=True,
        metavar="SOURCE_ID=UPDATER_ID",
        help=(
            "repeat for fitted_action_aware or an actual llm_* updater; "
            "structured proxy updaters are rejected"
        ),
    )
    evidence_from_a.set_defaults(
        handler=_human_study_evidence_from_experiment_a
    )

    correction = commands.add_parser(
        "correction-debt",
        help="stage-gated correction-debt protocol diagnostics",
    )
    correction_commands = correction.add_subparsers(
        dest="correction_command", required=True
    )
    correction_run = correction_commands.add_parser(
        "run", help="run the exact-pair reference protocol"
    )
    correction_run.add_argument("output", type=Path)
    correction_run.add_argument("--truth-map", type=Path)
    correction_run.add_argument("--pairs", type=int, default=32)
    correction_run.add_argument(
        "--stage-gate-authorized",
        action="store_true",
        help="confirm that prerequisite evidence has been reviewed",
    )
    correction_run.set_defaults(handler=_correction_debt_run)

    control_study = commands.add_parser(
        "control-study",
        help="verify and score the separate Experiment A six-control exchange",
    )
    control_study_commands = control_study.add_subparsers(
        dest="control_study_command",
        required=True,
    )
    analyze_controls = control_study_commands.add_parser(
        "analyze",
        help=(
            "regenerate fixed plan/bindings and atomically score response JSONL"
        ),
    )
    analyze_controls.add_argument("bindings", type=Path)
    analyze_controls.add_argument("responses", type=Path)
    analyze_controls.add_argument("output", type=Path)
    analyze_controls.add_argument(
        "--source-descriptor",
        default=(
            "provider responses imported for checksum-bound offline "
            "Experiment A control scoring"
        ),
    )
    analyze_controls.set_defaults(handler=_control_study_analyze)
    h7_plan = control_study_commands.add_parser(
        "h7-plan",
        help=(
            "build paired direct-statement requests from a verified "
            "Experiment A run"
        ),
    )
    h7_plan.add_argument("run_dir", type=Path)
    h7_plan.add_argument("output_dir", type=Path)
    h7_plan.set_defaults(handler=_control_study_h7_plan)
    h7_review = control_study_commands.add_parser(
        "h7-review",
        help=(
            "bind accepted provider evidence and recompute H7 in a new artifact"
        ),
    )
    h7_review.add_argument("run_dir", type=Path)
    h7_review.add_argument("plan_dir", type=Path)
    h7_review.add_argument("responses", type=Path)
    h7_review.add_argument("provider_audit", type=Path)
    h7_review.add_argument("output", type=Path)
    h7_review.set_defaults(handler=_control_study_h7_review)
    h7_verify = control_study_commands.add_parser(
        "h7-verify",
        help="recompute and verify a provider-bound H7 review",
    )
    h7_verify.add_argument("run_dir", type=Path)
    h7_verify.add_argument("plan_dir", type=Path)
    h7_verify.add_argument("responses", type=Path)
    h7_verify.add_argument("provider_audit", type=Path)
    h7_verify.add_argument("review", type=Path)
    h7_verify.set_defaults(handler=_control_study_h7_verify)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (
        ConfigError,
        ExternalDecoderProviderError,
        OpenAIProviderError,
        OpenRouterProviderError,
        OSError,
        ValueError,
        KeyError,
    ) as exc:
        parser.error(str(exc))
    return 2
