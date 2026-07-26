"""Command-line interface for the offline CAPE-Loop reference implementation."""

from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path
from typing import Mapping
import json
import os
import platform
import sys
import tempfile

from . import __version__
from .artifacts import verify_run
from .config import ConfigError, load_config
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
from .gate_review import import_native_gate_review, verify_gate_review
from .llm_exchange import read_responses
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
from .native_action_provider import (
    OpenAINativeActionProvider,
    execute_openai_native_actions,
    plan_openai_native_actions,
)
from .release import freeze_run, verify_frozen_artifact
from .schema_export import export_schemas
from .schema_export import SCHEMAS


def _json(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)


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


def _config_validate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
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
    within_budget = (
        len(prepared) <= config.max_requests
        and conservative_tokens <= config.max_total_tokens
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
                "within_declared_budget": within_budget,
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
    return 0 if within_budget else 1


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
        if (
            len(requests) > provider.config.max_requests
            or conservative_tokens > provider.config.max_total_tokens
        ):
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
    within_budget = (
        len(prepared) <= config.max_requests
        and conservative_tokens <= config.max_total_tokens
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
                "request_budget_unit": "physical_http_attempt",
                "max_retries_per_logical_request": config.max_retries,
                "max_total_tokens": config.max_total_tokens,
                "within_declared_budget": within_budget,
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
    return 0 if within_budget else 1


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
        if (
            len(requests) > provider.config.max_requests
            or conservative_tokens > provider.config.max_total_tokens
        ):
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


def _decoder_execute_openai(args: argparse.Namespace) -> int:
    if not args.execute_live:
        raise ValueError(
            "decoder execution requires the explicit --execute-live flag"
        )
    requests = read_external_decoder_requests(args.requests)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for role_name in args.roles:
        if not (output / "journals" / role_name).exists():
            planning_provider = OpenAIResponsesProvider(
                _openai_cli_config(
                    args,
                    live_execution=False,
                    role_name=role_name,
                )
            )
            prepared_requests = (
                external_decoder_llm_request(
                    request,
                    decoder_instance_id=f"plan-{role_name}",
                )
                for request in requests
            )
            conservative_tokens = sum(
                planning_provider.prepare(request).estimated_max_tokens
                for request in prepared_requests
            )
            if (
                len(requests) > planning_provider.config.max_requests
                or conservative_tokens
                > planning_provider.config.max_total_tokens
            ):
                raise ValueError(
                    f"decoder role {role_name!r} would exceed its hard "
                    "budget before any request; run decoder-study plan-openai"
                )
    judgments = []
    manifests = []
    for role_name in args.roles:
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
        for request in requests:
            response = adapter.complete(
                external_decoder_llm_request(
                    request,
                    decoder_instance_id=instance_id,
                )
            )
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
    judgment_path.write_text(
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
        encoding="utf-8",
    )
    design_audit = validate_external_decoder_import(requests, judgments)
    (output / "execution-manifest.json").write_text(
        _json(
            {
                "schema_version": 1,
                "request_count": len(requests),
                "judgment_count": len(judgments),
                "roles": list(args.roles),
                "provider_runs": manifests,
                "source_design_audit": design_audit.to_dict(),
                "statistical_independence_claimed": False,
                "credentials_retained": False,
            }
        )
        + "\n",
        encoding="utf-8",
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
    requests = read_external_decoder_requests(args.requests)
    sources = []
    all_within_budget = True
    for role_name in args.roles:
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
        within_budget = (
            len(prepared) <= provider.config.max_requests
            and conservative_tokens <= provider.config.max_total_tokens
        )
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
                "within_declared_budget": within_budget,
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
    models = (args.model, *args.additional_model)
    if len(models) != len(set(models)):
        raise ValueError("OpenRouter decoder models must not contain duplicates")
    return models


def _openrouter_decoder_config(
    args: argparse.Namespace,
    *,
    model: str,
    live_execution: bool,
) -> OpenRouterProviderConfig:
    prepared = argparse.Namespace(**vars(args))
    prepared.model = model
    return _openrouter_cli_config(
        prepared,
        live_execution=live_execution,
    )


def _decoder_plan_openrouter(args: argparse.Namespace) -> int:
    requests = read_external_decoder_requests(args.requests)
    sources = []
    all_within_budget = True
    for index, model in enumerate(_openrouter_decoder_models(args), start=1):
        provider = OpenRouterChatProvider(
            _openrouter_decoder_config(
                args,
                model=model,
                live_execution=False,
            )
        )
        prepared = tuple(
            provider.prepare(
                external_decoder_llm_request(
                    request,
                    decoder_instance_id=f"plan-openrouter-{index}",
                )
            )
            for request in requests
        )
        conservative_tokens = sum(
            request.estimated_max_tokens for request in prepared
        )
        within_budget = (
            len(prepared) <= provider.config.max_requests
            and conservative_tokens <= provider.config.max_total_tokens
        )
        all_within_budget = all_within_budget and within_budget
        sources.append(
            {
                "gateway": "openrouter",
                "model": model,
                "reasoning_effort": (
                    provider.config.reasoning_effort or None
                ),
                "upstream_provider_constraint": (
                    provider.config.upstream_provider or None
                ),
                "provider_preferences": (
                    provider.config.provider_preferences()
                ),
                "request_count": len(prepared),
                "conservative_max_tokens": conservative_tokens,
                "max_requests": provider.config.max_requests,
                "request_budget_unit": "physical_http_attempt",
                "max_retries_per_logical_request": (
                    provider.config.max_retries
                ),
                "max_total_tokens": provider.config.max_total_tokens,
                "within_declared_budget": within_budget,
            }
        )
    print(
        _json(
            {
                "provider": "openrouter",
                "live_execution": False,
                "credential_read": False,
                "decoder_source_count": len(sources),
                "sources": sources,
                "all_within_declared_budget": all_within_budget,
                "response_cache_enabled": False,
                "router_metadata_requested": True,
                "first_party_origin_claimed": False,
                "strict_gate4_eligible": False,
                "statistical_independence_claimed": False,
            }
        )
    )
    return 0 if all_within_budget else 1


def _decoder_execute_openrouter(args: argparse.Namespace) -> int:
    if not args.execute_live:
        raise ValueError(
            "OpenRouter decoder execution requires the explicit "
            "--execute-live flag"
        )
    requests = read_external_decoder_requests(args.requests)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    judgments = []
    manifests = []
    for index, model in enumerate(_openrouter_decoder_models(args), start=1):
        config = _openrouter_decoder_config(
            args,
            model=model,
            live_execution=True,
        )
        provider = OpenRouterChatProvider(config)
        model_digest = sha256(model.encode("utf-8")).hexdigest()[:12]
        journal = output / "journals" / model_digest
        if not journal.exists():
            conservative_tokens = sum(
                provider.prepare(
                    external_decoder_llm_request(
                        request,
                        decoder_instance_id=(
                            f"plan-openrouter-{index}"
                        ),
                    )
                ).estimated_max_tokens
                for request in requests
            )
            if (
                len(requests) > config.max_requests
                or conservative_tokens > config.max_total_tokens
            ):
                raise ValueError(
                    f"OpenRouter decoder model {model!r} would exceed its "
                    "hard budget before any request; run "
                    "decoder-study plan-openrouter"
                )
        instance_id = f"openrouter-{model_digest}"
        adapter = ResumableOpenRouterCompletionProvider(
            provider,
            responses_path=journal / "responses.jsonl",
            audit_path=journal / "provider-audit.jsonl",
        )
        for request in requests:
            response = adapter.complete(
                external_decoder_llm_request(
                    request,
                    decoder_instance_id=instance_id,
                )
            )
            judgments.append(
                external_decoder_judgment_from_response(
                    request,
                    response,
                    decoder_instance_id=instance_id,
                    decoder_family_id=model,
                    source_descriptor=(
                        f"openrouter-chat:{model};"
                        "first-party-origin=false"
                    ),
                )
            )
        manifests.append(
            {
                **adapter.to_manifest(),
                "source_index": index,
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
                "gateway": "openrouter",
                "request_count": len(requests),
                "judgment_count": len(judgments),
                "models": list(_openrouter_decoder_models(args)),
                "provider_runs": manifests,
                "source_design_audit": design_audit.to_dict(),
                "response_cache_enabled": False,
                "router_metadata_requested": True,
                "first_party_origin_claimed": False,
                "strict_gate4_eligible": False,
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
                "strict_gate4_eligible": False,
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
    requests = read_external_decoder_requests(args.requests)
    plan = plan_external_decoder_collection(
        requests,
        _external_decoder_configs(args, live_execution=False),
    )
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
    _reject_output_inside_input_run(args.requests, args.output_dir)
    requests = read_external_decoder_requests(args.requests)
    planning_configs = _external_decoder_configs(
        args,
        live_execution=False,
    )
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
        "collection_plan_sha256": sha256(
            plan_path.read_bytes()
        ).hexdigest(),
        "judgments_sha256": sha256(
            judgments_path.read_bytes()
        ).hexdigest(),
        "provider_audit_sha256": sha256(
            audit_path.read_bytes()
        ).hexdigest(),
        "transport_attempts_sha256": sha256(
            attempt_path.read_bytes()
        ).hexdigest(),
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


def _load_assignment_codebooks(path: Path) -> dict[str, object]:
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
                    path.name: sha256(path.read_bytes()).hexdigest()
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
        help=(
            "complete official Anthropic/Gemini distinct-decoder collection; "
            "its judgments must be byte-identical to JUDGMENTS"
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
    import_native.set_defaults(handler=_gate_review_import_native)
    verify_review = gate_review_commands.add_parser(
        "verify",
        help="verify a checksum-bound Gate 4 review artifact",
    )
    verify_review.add_argument("review_dir", type=Path)
    verify_review.set_defaults(handler=_gate_review_verify)

    schema = commands.add_parser("schema", help="JSON Schema commands")
    schema_commands = schema.add_subparsers(dest="schema_command", required=True)
    export = schema_commands.add_parser("export", help="export public JSON Schemas")
    export.add_argument("destination", type=Path, nargs="?", default=Path("schemas"))
    export.set_defaults(handler=_schema_export)

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
        "artifact", help="freeze and verify paper-facing run archives"
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
    plan_openrouter_decoder.set_defaults(
        handler=_decoder_plan_openrouter
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
    execute_openrouter_decoder.set_defaults(
        handler=_decoder_execute_openrouter
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
        command.add_argument("--max-retries", type=int, default=4)
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
