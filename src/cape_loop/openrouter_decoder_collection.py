"""Auditable two-family OpenRouter external-decoder collections.

This module keeps shared-gateway evidence distinct from the direct first-party
Anthropic/Gemini collector.  A selected OpenRouter collection proves that the
gateway request, returned model, routing metadata, durable attempt ledger, and
replay response all agree.  It never claims first-party origin or statistically
independent errors.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence
import json

from .artifacts import canonical_json, file_sha256
from .decoder_study import (
    ExternalDecoderJudgment,
    ExternalDecoderRequest,
    external_decoder_judgment_from_response,
    external_decoder_llm_request,
    read_external_decoder_judgments,
    validate_external_decoder_import,
)
from .llm_exchange import LLMResponse, read_responses
from .openai_provider import _read_audit_records
from .openrouter_provider import (
    OPENROUTER_OFFICIAL_BASE_URL,
    OpenRouterChatProvider,
    OpenRouterProviderConfig,
)
from .provider_attempts import (
    DurableProviderAttemptLedger,
    default_attempt_path,
)


OPENROUTER_CLAUDE_DECODER_MODEL = "anthropic/claude-sonnet-5"
OPENROUTER_GEMINI_DECODER_MODEL = "google/gemini-3.6-flash"
SELECTED_OPENROUTER_DECODER_MODELS = (
    OPENROUTER_CLAUDE_DECODER_MODEL,
    OPENROUTER_GEMINI_DECODER_MODEL,
)
SELECTED_OPENROUTER_DECODER_FAMILIES: Mapping[str, str] = MappingProxyType(
    {
        OPENROUTER_CLAUDE_DECODER_MODEL: "anthropic-claude",
        OPENROUTER_GEMINI_DECODER_MODEL: "google-gemini",
    }
)
SELECTED_OPENROUTER_REASONING_EFFORTS: Mapping[str, str] = MappingProxyType(
    {
        OPENROUTER_CLAUDE_DECODER_MODEL: "low",
        OPENROUTER_GEMINI_DECODER_MODEL: "minimal",
    }
)

OPENROUTER_DECODER_MAX_REQUESTS = 900
OPENROUTER_DECODER_MAX_TOTAL_TOKENS = 6_000_000
OPENROUTER_DECODER_MAX_OUTPUT_TOKENS = 1_024
OPENROUTER_DECODER_MAX_RETRIES = 0

OPENROUTER_COLLECTION_KIND = (
    "openrouter-distinct-external-decoder-collection"
)
OPENROUTER_COLLECTION_PLAN_KIND = (
    "openrouter-external-decoder-collection-plan"
)
OPENROUTER_COLLECTION_LOCKS = (
    ".external-decoder-command.lock",
    ".external-decoder-collection.lock",
)
OPENROUTER_COLLECTION_FILES: Mapping[str, str] = MappingProxyType(
    {
        "decoder_collection_plan": "collection-plan.json",
        "decoder_transport_attempts": "transport-attempts.jsonl",
        "decoder_provider_audit": "provider-audit.jsonl",
        "decoder_judgments": "judgments.jsonl",
        "decoder_execution_manifest": "execution-manifest.json",
    }
)


def openrouter_decoder_identity(model: str) -> tuple[str, str]:
    """Return the journal digest and stable decoder instance identity."""

    model_digest = sha256(model.encode("utf-8")).hexdigest()[:12]
    return model_digest, f"openrouter-{model_digest}"


def openrouter_decoder_family(model: str) -> str:
    """Return the declared family, retaining generic model-level behavior."""

    return SELECTED_OPENROUTER_DECODER_FAMILIES.get(model, model)


def openrouter_decoder_source_descriptor(model: str) -> str:
    """Return stable judgment metadata without overstating route provenance."""

    return (
        f"openrouter-chat:{model};"
        f"family={openrouter_decoder_family(model)};"
        "shared-gateway=openrouter;"
        "first-party-origin=false"
    )


def is_selected_openrouter_pair(models: Sequence[str]) -> bool:
    return (
        len(models) == len(SELECTED_OPENROUTER_DECODER_MODELS)
        and set(models) == set(SELECTED_OPENROUTER_DECODER_MODELS)
    )


def _retry_budget(
    *,
    request_count: int,
    conservative_tokens: int,
    max_retries: int,
    max_requests: int,
    max_total_tokens: int,
) -> dict[str, int | bool]:
    multiplier = max_retries + 1
    attempts = request_count * multiplier
    tokens = conservative_tokens * multiplier
    return {
        "initial_transport_attempt_count": request_count,
        "maximum_attempts_per_request": multiplier,
        "theoretical_max_transport_attempts": attempts,
        "theoretical_max_tokens_with_all_retries": tokens,
        "within_declared_budget": (
            attempts <= max_requests and tokens <= max_total_tokens
        ),
    }


def build_openrouter_decoder_collection_plan(
    requests: Sequence[ExternalDecoderRequest],
    configs: Sequence[OpenRouterProviderConfig],
) -> dict[str, Any]:
    """Build the exact credential-free plan shared by plan and execution."""

    request_rows = tuple(sorted(requests, key=lambda row: row.request_id))
    if not request_rows:
        raise ValueError("at least one external decoder request is required")
    if len({row.request_id for row in request_rows}) != len(request_rows):
        raise ValueError("external decoder requests contain duplicate IDs")
    configured = tuple(sorted(configs, key=lambda item: item.model))
    if not configured:
        raise ValueError("at least one OpenRouter decoder model is required")
    if len({item.model for item in configured}) != len(configured):
        raise ValueError("OpenRouter decoder models must not contain duplicates")

    sources: list[dict[str, Any]] = []
    for config in configured:
        model_digest, instance_id = openrouter_decoder_identity(config.model)
        provider = OpenRouterChatProvider(config)
        provider_requests = tuple(
            external_decoder_llm_request(
                request,
                decoder_instance_id=instance_id,
            )
            for request in request_rows
        )
        prepared = tuple(
            provider.prepare(request) for request in provider_requests
        )
        conservative_tokens = sum(
            item.estimated_max_tokens for item in prepared
        )
        retry_budget = _retry_budget(
            request_count=len(prepared),
            conservative_tokens=conservative_tokens,
            max_retries=config.max_retries,
            max_requests=config.max_requests,
            max_total_tokens=config.max_total_tokens,
        )
        sources.append(
            {
                "gateway": "openrouter",
                "model": config.model,
                "decoder_family_id": openrouter_decoder_family(config.model),
                "decoder_instance_id": instance_id,
                "model_digest": model_digest,
                "source_descriptor": openrouter_decoder_source_descriptor(
                    config.model
                ),
                "reasoning_effort": config.reasoning_effort or None,
                "api_key_env": config.api_key_env,
                "base_url": config.base_url,
                "allow_custom_base_url": config.allow_custom_base_url,
                "endpoint": config.endpoint,
                "upstream_provider_constraint": (
                    config.upstream_provider or None
                ),
                "allow_fallbacks": config.allow_fallbacks,
                "require_parameters": config.require_parameters,
                "data_collection": config.data_collection,
                "zdr": config.zdr,
                "http_referer": config.http_referer,
                "app_title": config.app_title,
                "provider_preferences": config.provider_preferences(),
                "timeout_seconds": config.timeout_seconds,
                "max_retries": config.max_retries,
                "initial_backoff_seconds": (
                    config.initial_backoff_seconds
                ),
                "max_backoff_seconds": config.max_backoff_seconds,
                "jitter_fraction": config.jitter_fraction,
                "max_output_tokens": config.max_output_tokens,
                "max_requests": config.max_requests,
                "max_total_tokens": config.max_total_tokens,
                "request_count": len(prepared),
                "conservative_max_tokens": conservative_tokens,
                **retry_budget,
                "request_body_sha256": [
                    {
                        "request_id": source_request.request_id,
                        "provider_request_id": provider_request.request_id,
                        "sha256": prepared_request.body_sha256,
                        "estimated_max_tokens": (
                            prepared_request.estimated_max_tokens
                        ),
                    }
                    for (
                        source_request,
                        provider_request,
                        prepared_request,
                    ) in zip(request_rows, provider_requests, prepared)
                ],
            }
        )
    models = tuple(source["model"] for source in sources)
    families = {source["decoder_family_id"] for source in sources}
    selected_pair = is_selected_openrouter_pair(models)
    selected_admission = selected_pair and all(
        config.reasoning_effort
        == SELECTED_OPENROUTER_REASONING_EFFORTS[config.model]
        and config.api_key_env == "OPENROUTER_API_KEY"
        and config.base_url == OPENROUTER_OFFICIAL_BASE_URL
        and not config.allow_custom_base_url
        and not config.upstream_provider
        and not config.allow_fallbacks
        and config.require_parameters
        and config.data_collection == "deny"
        and config.max_retries == OPENROUTER_DECODER_MAX_RETRIES
        and config.max_output_tokens
        <= OPENROUTER_DECODER_MAX_OUTPUT_TOKENS
        and config.max_requests <= OPENROUTER_DECODER_MAX_REQUESTS
        and config.max_total_tokens
        <= OPENROUTER_DECODER_MAX_TOTAL_TOKENS
        for config in configured
    )
    core: dict[str, Any] = {
        "schema_version": 1,
        "kind": OPENROUTER_COLLECTION_PLAN_KIND,
        "provider": "openrouter",
        "gateway": "openrouter",
        "request_count": len(request_rows),
        "decoder_source_count": len(sources),
        "models": list(models),
        "sources": sources,
        "all_within_declared_budget": all(
            source["within_declared_budget"] is True for source in sources
        ),
        "selected_claude_gemini_pair": selected_pair,
        "distinct_model_families": len(families) == len(sources),
        "credential_read": False,
        "response_cache_enabled": False,
        "router_metadata_requested": True,
        "shared_gateway": True,
        "first_party_origin_claimed": False,
        "strict_first_party_gate4_eligible": False,
        "eligible_for_reviewed_shared_gateway_admission": selected_admission,
        "statistical_independence_claimed": False,
    }
    return {
        **core,
        "plan_sha256": sha256(
            canonical_json(core).encode("utf-8")
        ).hexdigest(),
    }


def _config_from_source(source: Mapping[str, Any]) -> OpenRouterProviderConfig:
    try:
        return OpenRouterProviderConfig(
            model=source["model"],
            reasoning_effort=source["reasoning_effort"] or "",
            api_key_env=source["api_key_env"],
            base_url=source["base_url"],
            allow_custom_base_url=source["allow_custom_base_url"],
            upstream_provider=source["upstream_provider_constraint"] or "",
            allow_fallbacks=source["allow_fallbacks"],
            require_parameters=source["require_parameters"],
            data_collection=source["data_collection"],
            zdr=source["zdr"],
            http_referer=source["http_referer"],
            app_title=source["app_title"],
            timeout_seconds=source["timeout_seconds"],
            max_retries=source["max_retries"],
            initial_backoff_seconds=source["initial_backoff_seconds"],
            max_backoff_seconds=source["max_backoff_seconds"],
            jitter_fraction=source["jitter_fraction"],
            max_output_tokens=source["max_output_tokens"],
            max_requests=source["max_requests"],
            max_total_tokens=source["max_total_tokens"],
            live_execution=False,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "OpenRouter collection plan has an invalid source configuration"
        ) from exc


def openrouter_source_execution_summary(
    *,
    source_index: int,
    config: OpenRouterProviderConfig,
    decoder_instance_id: str,
    request_count: int,
    transport_attempt_count: int,
    total_tokens: int,
    audits: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the normalized source summary used by execution and verification."""

    model_digest, expected_instance = openrouter_decoder_identity(config.model)
    if decoder_instance_id != expected_instance:
        raise ValueError("OpenRouter decoder instance identity mismatch")
    return {
        "source_index": source_index,
        "gateway": "openrouter",
        "model": config.model,
        "decoder_family_id": openrouter_decoder_family(config.model),
        "decoder_instance_id": decoder_instance_id,
        "reasoning_effort": config.reasoning_effort or None,
        "provider_preferences": config.provider_preferences(),
        "request_count": request_count,
        "transport_attempt_count": transport_attempt_count,
        "total_tokens": total_tokens,
        "journal_directory": f"journals/{model_digest}",
        "upstream_providers_returned": sorted(
            {str(audit["upstream_provider"]) for audit in audits}
        ),
        "upstream_models_returned": sorted(
            {str(audit["upstream_model"]) for audit in audits}
        ),
        "routing_strategies_returned": sorted(
            {str(audit["routing_strategy"]) for audit in audits}
        ),
        "shared_gateway": True,
        "first_party_origin_claimed": False,
    }


def _file_entry(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    return {
        "path": relative,
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }


def build_openrouter_decoder_execution_manifest(
    *,
    root: Path,
    plan: Mapping[str, Any],
    source_runs: Sequence[Mapping[str, Any]],
    source_design_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the completed collection and all resumable source journals."""

    journal_files = []
    for source in source_runs:
        journal = str(source["journal_directory"])
        for filename in (
            "provider-audit-transport-attempts.jsonl",
            "provider-audit.jsonl",
            "responses.jsonl",
        ):
            journal_files.append(_file_entry(root, f"{journal}/{filename}"))
    core: dict[str, Any] = {
        "schema_version": 1,
        "kind": OPENROUTER_COLLECTION_KIND,
        "status": "complete",
        "claim_status": "not_claimed",
        "gateway": "openrouter",
        "collection_plan_sha256": file_sha256(
            root / OPENROUTER_COLLECTION_FILES["decoder_collection_plan"]
        ),
        "transport_attempts_sha256": file_sha256(
            root
            / OPENROUTER_COLLECTION_FILES["decoder_transport_attempts"]
        ),
        "provider_audit_sha256": file_sha256(
            root / OPENROUTER_COLLECTION_FILES["decoder_provider_audit"]
        ),
        "judgments_sha256": file_sha256(
            root / OPENROUTER_COLLECTION_FILES["decoder_judgments"]
        ),
        "plan_identity_sha256": plan["plan_sha256"],
        "request_count": plan["request_count"],
        "judgment_count": (
            int(plan["request_count"]) * int(plan["decoder_source_count"])
        ),
        "models": list(plan["models"]),
        "source_runs": [dict(source) for source in source_runs],
        "journal_files": sorted(
            journal_files,
            key=lambda item: str(item["path"]),
        ),
        "source_design_audit": dict(source_design_audit),
        "selected_claude_gemini_pair": (
            plan["selected_claude_gemini_pair"]
        ),
        "shared_gateway": True,
        "gateway_provider_provenance_auditable": True,
        "first_party_origin_claimed": False,
        "strict_first_party_gate4_eligible": False,
        "eligible_for_reviewed_shared_gateway_admission": (
            plan["eligible_for_reviewed_shared_gateway_admission"]
        ),
        "statistical_independence_claimed": False,
        "responsible_researcher_source_review_required": True,
        "credentials_retained": False,
    }
    return {
        **core,
        "collection_id": sha256(
            canonical_json(core).encode("utf-8")
        ).hexdigest(),
    }


def _safe_json_object(path: Path, *, name: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be a safe regular JSON file")
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is invalid JSON: {exc}") from exc
    if not isinstance(decoded, Mapping):
        raise ValueError(f"{name} must contain one JSON object")
    return dict(decoded)


def _canonical_jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(canonical_json(dict(row)) + "\n" for row in rows).encode(
        "utf-8"
    )


def _manifest_entry(
    path: Path,
    *,
    record_count: int | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "filename": path.name,
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }
    if record_count is not None:
        entry["record_count"] = record_count
    return entry


def is_openrouter_decoder_collection(path: str | Path) -> bool:
    """Identify the selected collection kind without accepting its contents."""

    root = Path(path)
    manifest = root / "execution-manifest.json"
    if root.is_symlink() or not root.is_dir() or manifest.is_symlink():
        return False
    try:
        decoded = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return (
        isinstance(decoded, Mapping)
        and decoded.get("kind") == OPENROUTER_COLLECTION_KIND
    )


def validate_openrouter_decoder_collection(
    collection_dir: str | Path,
    *,
    requests: Sequence[ExternalDecoderRequest],
    judgments_path: str | Path,
) -> tuple[
    tuple[ExternalDecoderJudgment, ...],
    dict[str, dict[str, Any]],
    dict[str, Any],
]:
    """Read-only validation of one complete selected OpenRouter collection."""

    unresolved = Path(collection_dir)
    if unresolved.is_symlink() or not unresolved.is_dir():
        raise ValueError(
            "OpenRouter decoder evidence requires a safe collection directory"
        )
    root = unresolved.resolve()
    required_root = {
        *OPENROUTER_COLLECTION_FILES.values(),
        *OPENROUTER_COLLECTION_LOCKS,
        "journals",
    }
    actual_root = {item.name for item in root.iterdir()}
    if actual_root != required_root:
        raise ValueError(
            "OpenRouter decoder collection has missing or unexpected entries: "
            + canonical_json(
                {
                    "missing": sorted(required_root - actual_root),
                    "unexpected": sorted(actual_root - required_root),
                }
            )
        )
    for lock_name in OPENROUTER_COLLECTION_LOCKS:
        lock = root / lock_name
        if lock.is_symlink() or not lock.is_file():
            raise ValueError("OpenRouter collection lacks its safe lock files")
    journals = root / "journals"
    if journals.is_symlink() or not journals.is_dir():
        raise ValueError("OpenRouter collection journals must be a directory")
    for filename in OPENROUTER_COLLECTION_FILES.values():
        candidate = root / filename
        if (
            candidate.is_symlink()
            or not candidate.is_file()
            or candidate.resolve().parent != root
        ):
            raise ValueError(
                f"OpenRouter collection file is unsafe: {filename}"
            )

    plan_path = root / "collection-plan.json"
    plan = _safe_json_object(plan_path, name="OpenRouter collection plan")
    raw_sources = plan.get("sources")
    if (
        not isinstance(raw_sources, list)
        or any(not isinstance(source, Mapping) for source in raw_sources)
    ):
        raise ValueError("OpenRouter collection plan sources are invalid")
    configs = tuple(_config_from_source(source) for source in raw_sources)
    if not is_selected_openrouter_pair(
        tuple(config.model for config in configs)
    ):
        raise ValueError(
            "audited OpenRouter admission requires the selected Claude and "
            "Gemini pair"
        )
    expected_efforts = {
        config.model: config.reasoning_effort for config in configs
    }
    if expected_efforts != dict(SELECTED_OPENROUTER_REASONING_EFFORTS):
        raise ValueError(
            "selected OpenRouter decoder reasoning efforts must be "
            "Claude=low and Gemini=minimal"
        )
    for config in configs:
        if (
            config.base_url != OPENROUTER_OFFICIAL_BASE_URL
            or config.allow_custom_base_url
            or config.api_key_env != "OPENROUTER_API_KEY"
            or config.upstream_provider
            or config.allow_fallbacks
            or not config.require_parameters
            or config.data_collection != "deny"
            or config.max_retries != OPENROUTER_DECODER_MAX_RETRIES
            or config.max_output_tokens
            > OPENROUTER_DECODER_MAX_OUTPUT_TOKENS
            or config.max_requests > OPENROUTER_DECODER_MAX_REQUESTS
            or config.max_total_tokens
            > OPENROUTER_DECODER_MAX_TOTAL_TOKENS
        ):
            raise ValueError(
                "selected OpenRouter decoder collection violates its official "
                "gateway, route, privacy, retry, or budget policy"
            )
    expected_plan = build_openrouter_decoder_collection_plan(
        tuple(requests),
        configs,
    )
    if plan != expected_plan:
        raise ValueError(
            "OpenRouter collection plan does not match the retained requests "
            "and implemented adapters"
        )
    if plan.get("all_within_declared_budget") is not True:
        raise ValueError("OpenRouter collection plan exceeds its hard budgets")

    expected_journal_names = {
        openrouter_decoder_identity(config.model)[0] for config in configs
    }
    actual_journal_names = {item.name for item in journals.iterdir()}
    if actual_journal_names != expected_journal_names:
        raise ValueError(
            "OpenRouter collection has missing or unexpected model journals"
        )

    expected_judgments: list[ExternalDecoderJudgment] = []
    aggregate_audits: list[Mapping[str, Any]] = []
    aggregate_attempts: list[Mapping[str, Any]] = []
    source_runs: list[dict[str, Any]] = []
    request_rows = tuple(sorted(requests, key=lambda row: row.request_id))
    for source_index, config in enumerate(configs, start=1):
        model_digest, instance_id = openrouter_decoder_identity(config.model)
        journal = journals / model_digest
        expected_files = {
            "provider-audit-transport-attempts.jsonl",
            "provider-audit.jsonl",
            "responses.jsonl",
        }
        if (
            journal.is_symlink()
            or not journal.is_dir()
            or {item.name for item in journal.iterdir()} != expected_files
            or any(
                item.is_symlink() or not item.is_file()
                for item in journal.iterdir()
            )
        ):
            raise ValueError(
                f"OpenRouter journal {model_digest} is incomplete or unsafe"
            )
        provider = OpenRouterChatProvider(config)
        provider_requests = tuple(
            external_decoder_llm_request(
                request,
                decoder_instance_id=instance_id,
            )
            for request in request_rows
        )
        request_by_id = {
            request.request_id: request for request in provider_requests
        }
        audit_path = journal / "provider-audit.jsonl"
        response_path = journal / "responses.jsonl"
        attempt_path = default_attempt_path(audit_path)
        ledger = DurableProviderAttemptLedger(
            attempt_path,
            provider_name="openrouter",
            model_requested=config.model,
        )
        ledger.assert_safe_to_resume()
        ledger.validate_requests(request_by_id, provider)
        audits = _read_audit_records(
            audit_path,
            provider_name="openrouter",
        )
        responses = {
            response.request_id: response
            for response in read_responses(response_path)
        }
        expected_ids = set(request_by_id)
        embedded = ledger.embedded_final_audits()
        if (
            set(audits) != expected_ids
            or set(responses) != expected_ids
            or set(embedded) != expected_ids
        ):
            raise ValueError(
                "OpenRouter journal does not cover every decoder request"
            )
        ordered_audits = []
        for source_request, provider_request in zip(
            request_rows,
            provider_requests,
        ):
            audit = audits[provider_request.request_id]
            if audit != embedded[provider_request.request_id]:
                raise ValueError(
                    "OpenRouter provider audit differs from its durable "
                    "attempt settlement"
                )
            if audit.get("acceptance_status") != "accepted":
                raise ValueError(
                    "OpenRouter collection contains a rejected provider result"
                )
            provider.validate_resumed_audit(
                audit,
                request=provider_request,
                prepared=provider.prepare(provider_request),
            )
            replay = LLMResponse.parse(audit["replay_response"])
            if responses[provider_request.request_id].to_dict() != (
                replay.to_dict()
            ):
                raise ValueError(
                    "OpenRouter response journal differs from provider audit"
                )
            expected_judgments.append(
                external_decoder_judgment_from_response(
                    source_request,
                    replay,
                    decoder_instance_id=instance_id,
                    decoder_family_id=openrouter_decoder_family(config.model),
                    source_descriptor=openrouter_decoder_source_descriptor(
                        config.model
                    ),
                )
            )
            ordered_audits.append(audit)
        attempts, tokens = ledger.accounting()
        source_runs.append(
            openrouter_source_execution_summary(
                source_index=source_index,
                config=config,
                decoder_instance_id=instance_id,
                request_count=len(request_rows),
                transport_attempt_count=attempts,
                total_tokens=tokens,
                audits=ordered_audits,
            )
        )
        aggregate_audits.extend(ordered_audits)
        aggregate_attempts.extend(
            ledger.events_for_request_ids(
                tuple(request.request_id for request in provider_requests)
            )
        )

    audit_path = root / "provider-audit.jsonl"
    attempt_path = root / "transport-attempts.jsonl"
    judgment_path = root / "judgments.jsonl"
    expected_audit_bytes = _canonical_jsonl(aggregate_audits)
    expected_attempt_bytes = _canonical_jsonl(aggregate_attempts)
    expected_judgment_bytes = _canonical_jsonl(
        tuple(judgment.to_dict() for judgment in expected_judgments)
    )
    if audit_path.read_bytes() != expected_audit_bytes:
        raise ValueError(
            "aggregate OpenRouter audits differ from the model journals"
        )
    if attempt_path.read_bytes() != expected_attempt_bytes:
        raise ValueError(
            "aggregate OpenRouter attempts differ from the model journals"
        )
    if judgment_path.read_bytes() != expected_judgment_bytes:
        raise ValueError(
            "OpenRouter judgments differ from accepted provider audits"
        )
    supplied_judgments = Path(judgments_path)
    if (
        supplied_judgments.is_symlink()
        or not supplied_judgments.is_file()
        or supplied_judgments.read_bytes() != expected_judgment_bytes
    ):
        raise ValueError(
            "supplied judgments must be byte-identical to the selected "
            "OpenRouter collection"
        )
    judgments = tuple(expected_judgments)
    retained_judgments = read_external_decoder_judgments(judgment_path)
    if retained_judgments != judgments:
        raise ValueError("OpenRouter judgment parsing changed collection order")
    source_design = validate_external_decoder_import(
        request_rows,
        judgments,
        minimum_sources_per_request=2,
        require_distinct_families=True,
    )
    if (
        not source_design.complete_coverage
        or not source_design.source_design_eligible
    ):
        raise ValueError(
            "OpenRouter collection lacks complete distinct-family coverage"
        )
    manifest_path = root / "execution-manifest.json"
    manifest = _safe_json_object(
        manifest_path,
        name="OpenRouter execution manifest",
    )
    expected_manifest = build_openrouter_decoder_execution_manifest(
        root=root,
        plan=plan,
        source_runs=source_runs,
        source_design_audit=source_design.to_dict(),
    )
    if manifest != expected_manifest:
        raise ValueError(
            "OpenRouter execution manifest does not match its collection"
        )

    inputs = {
        "decoder_collection_plan": _manifest_entry(plan_path),
        "decoder_transport_attempts": _manifest_entry(
            attempt_path,
            record_count=len(aggregate_attempts),
        ),
        "decoder_provider_audit": _manifest_entry(
            audit_path,
            record_count=len(aggregate_audits),
        ),
        "decoder_judgments": _manifest_entry(
            judgment_path,
            record_count=len(judgments),
        ),
        "decoder_execution_manifest": _manifest_entry(manifest_path),
    }
    summary = {
        "provenance_mode": "selected_openrouter_gateway_collection",
        "collection_status": "complete",
        "gateway": "openrouter",
        "models": list(plan["models"]),
        "decoder_family_ids": sorted(
            {judgment.decoder_family_id for judgment in judgments}
        ),
        "request_count": len(request_rows),
        "source_count": len(configs),
        "judgment_count": len(judgments),
        "source_runs": source_runs,
        "all_collection_files_digest_bound": True,
        "plan_rebuilt_from_retained_requests": True,
        "attempt_journals_validated": True,
        "provider_audits_validated": True,
        "judgments_match_accepted_provider_audits": True,
        "shared_gateway": True,
        "first_party_origin_claimed": False,
        "strict_first_party_gate4_eligible": False,
        "eligible_for_responsible_researcher_review": True,
        "statistical_independence_claimed": False,
    }
    return judgments, inputs, summary
