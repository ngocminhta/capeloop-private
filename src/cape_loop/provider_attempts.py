"""Durable physical-attempt evidence for direct LLM providers.

The public provider audit is written only after a provider call returns to its
caller.  A process can therefore fail after a paid HTTP response but before
that audit append.  This module closes that gap with an fsynced, append-only
started/settled journal around every physical transport attempt.

An unresolved start, or a previous invocation that ended without a final
accepted/rejected audit, is intentionally not retried on process restart.
Those states require manual billing review.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence
import json
import os
import re
import stat

from .file_lock import try_file_lock, unlock_file


PROVIDER_ATTEMPT_KIND = "llm-provider-transport-attempt"
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_OUTCOMES = frozenset(
    {
        "transport_error",
        "http_error",
        "invalid_response",
        "rejected_provider_result",
        "success",
    }
)


class ProviderAttemptManualReviewRequired(RuntimeError):
    """Automatic execution is unsafe given retained attempt evidence."""


class ProviderExecutionLocked(RuntimeError):
    """Another process owns the provider journal transaction."""


class ExclusiveProviderExecutionLock:
    """Nonblocking process lock covering reconciliation through final append."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._descriptor: int | None = None

    def __enter__(self) -> "ExclusiveProviderExecutionLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(self.path, flags, 0o600)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ValueError(
                    "provider execution lock must be a regular file"
                )
            if not try_file_lock(descriptor):
                raise ProviderExecutionLocked(
                    "another provider executor holds the journal lock; wait "
                    "for it to finish or review the owning process"
                )
        except Exception:
            os.close(descriptor)
            raise
        self._descriptor = descriptor
        return self

    def __exit__(self, *_: object) -> None:
        descriptor = self._descriptor
        self._descriptor = None
        if descriptor is not None:
            try:
                unlock_file(descriptor)
            finally:
                os.close(descriptor)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _digest(value: Any) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _validate_digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _validate_nonempty(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\r" in value
        or "\n" in value
    ):
        raise ValueError(f"{name} must be a non-empty single-line string")
    return value


def _validate_timestamp(value: Any, name: str) -> str:
    text = _validate_nonempty(value, name)
    if not text.endswith("Z") or "T" not in text:
        raise ValueError(f"{name} must be an ISO-8601 UTC timestamp")
    return text


def _audit_usage_total(audit: Mapping[str, Any]) -> int | None:
    usage = audit.get("usage")
    if not isinstance(usage, Mapping):
        return None
    total = usage.get("total_tokens")
    if isinstance(total, int) and not isinstance(total, bool) and total >= 0:
        return total
    for input_name, output_name in (
        ("input_tokens", "output_tokens"),
        ("prompt_tokens", "completion_tokens"),
    ):
        input_tokens = usage.get(input_name)
        output_tokens = usage.get(output_name)
        if all(
            isinstance(value, int)
            and not isinstance(value, bool)
            and value >= 0
            for value in (input_tokens, output_tokens)
        ):
            return int(input_tokens) + int(output_tokens)
    return None


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    """Append one canonical JSON line and fsync it before returning."""

    path.parent.mkdir(parents=True, exist_ok=True)
    line = (canonical_json(value) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(
        path,
        flags,
        0o600,
    )
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("provider journal must be a regular file")
        offset = 0
        while offset < len(line):
            offset += os.write(descriptor, line[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def repair_trailing_jsonl(path: Path) -> bool:
    """Repair only a crash-truncated final JSONL record."""

    flags = os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        return False
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("provider journal must be a regular file")
        with os.fdopen(descriptor, "r+b", closefd=False) as handle:
            material = handle.read()
            if not material or material.endswith(b"\n"):
                return False
            last_newline = material.rfind(b"\n")
            tail_start = last_newline + 1
            tail = material[tail_start:]
            try:
                json.loads(tail.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                handle.seek(0)
                handle.truncate(tail_start)
            else:
                handle.seek(0, os.SEEK_END)
                handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    return True


def default_attempt_path(audit_path: str | Path) -> Path:
    """Derive a collision-resistant attempt journal from an audit path."""

    audit = Path(audit_path)
    suffix = audit.suffix or ".jsonl"
    return audit.with_name(f"{audit.stem}-transport-attempts{suffix}")


class DurableProviderAttemptLedger:
    """Validated, append-only evidence for physical provider attempts."""

    def __init__(
        self,
        path: str | Path,
        *,
        provider_name: str,
        model_requested: str,
    ) -> None:
        self.path = Path(path)
        self.provider_name = _validate_nonempty(
            provider_name,
            "provider_name",
        )
        self.model_requested = _validate_nonempty(
            model_requested,
            "model_requested",
        )
        self.starts: dict[str, dict[str, Any]] = {}
        self.settlements: dict[str, dict[str, Any]] = {}
        self._ordinals: dict[str, int] = {}
        if self.path.exists():
            if self.path.is_symlink() or not self.path.is_file():
                raise ValueError(
                    "provider attempt journal must be a regular file"
                )
            self._read()

    def _read(self) -> None:
        with self.path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    decoded = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ProviderAttemptManualReviewRequired(
                        f"{self.path}:{line_number}: malformed durable provider "
                        "attempt journal; manual review is required"
                    ) from exc
                if not isinstance(decoded, Mapping):
                    raise ValueError(
                        f"{self.path}:{line_number}: attempt event must be an "
                        "object"
                    )
                row = dict(decoded)
                if (
                    row.get("schema_version") != 1
                    or row.get("kind") != PROVIDER_ATTEMPT_KIND
                ):
                    raise ValueError(
                        f"{self.path}:{line_number}: unknown provider-attempt "
                        "schema"
                    )
                event = row.get("event")
                if event == "started":
                    self._parse_start(row, line_number=line_number)
                elif event == "settled":
                    self._parse_settlement(row, line_number=line_number)
                else:
                    raise ValueError(
                        f"{self.path}:{line_number}: invalid attempt event"
                    )

    @property
    def unresolved_attempt_ids(self) -> tuple[str, ...]:
        return tuple(
            attempt_id
            for attempt_id in self.starts
            if attempt_id not in self.settlements
        )

    def _parse_start(
        self,
        row: dict[str, Any],
        *,
        line_number: int,
    ) -> None:
        allowed = {
            "schema_version",
            "kind",
            "event",
            "attempt_id",
            "provider",
            "request_id",
            "prompt_sha256",
            "endpoint",
            "request_body_sha256",
            "model_requested",
            "client_request_id",
            "idempotency_key",
            "estimated_max_tokens",
            "attempt_ordinal",
            "started_at",
        }
        if set(row) != allowed:
            raise ValueError(
                f"{self.path}:{line_number}: invalid started-attempt fields"
            )
        if self.unresolved_attempt_ids:
            raise ProviderAttemptManualReviewRequired(
                f"{self.path}:{line_number}: an unresolved provider attempt "
                "precedes another event; manual review is required"
            )
        attempt_id = _validate_digest(row.get("attempt_id"), "attempt_id")
        if row.get("provider") != self.provider_name:
            raise ValueError(
                f"{self.path}:{line_number}: provider identity differs from "
                "the configured provider"
            )
        if row.get("model_requested") != self.model_requested:
            raise ValueError(
                f"{self.path}:{line_number}: requested model differs from "
                "the configured model"
            )
        request_id = _validate_nonempty(row.get("request_id"), "request_id")
        _validate_digest(row.get("prompt_sha256"), "prompt_sha256")
        _validate_digest(
            row.get("request_body_sha256"),
            "request_body_sha256",
        )
        _validate_nonempty(row.get("endpoint"), "endpoint")
        _validate_nonempty(
            row.get("client_request_id"),
            "client_request_id",
        )
        idempotency_key = row.get("idempotency_key")
        if idempotency_key is not None:
            _validate_nonempty(idempotency_key, "idempotency_key")
        estimated = row.get("estimated_max_tokens")
        ordinal = row.get("attempt_ordinal")
        if (
            not isinstance(estimated, int)
            or isinstance(estimated, bool)
            or estimated <= 0
            or not isinstance(ordinal, int)
            or isinstance(ordinal, bool)
            or ordinal <= 0
        ):
            raise ValueError(
                f"{self.path}:{line_number}: invalid attempt budget or ordinal"
            )
        _validate_timestamp(row.get("started_at"), "started_at")
        if attempt_id in self.starts or attempt_id in self.settlements:
            raise ValueError(
                f"{self.path}:{line_number}: duplicate attempt identity"
            )
        expected_ordinal = self._ordinals.get(request_id, 0) + 1
        if ordinal != expected_ordinal:
            raise ValueError(
                f"{self.path}:{line_number}: nonsequential attempt ordinal"
            )
        binding = {
            key: row[key]
            for key in allowed
            if key not in {"schema_version", "kind", "event", "attempt_id"}
        }
        if attempt_id != _digest(binding):
            raise ValueError(
                f"{self.path}:{line_number}: attempt digest mismatch"
            )
        self.starts[attempt_id] = row
        self._ordinals[request_id] = ordinal

    def _parse_settlement(
        self,
        row: dict[str, Any],
        *,
        line_number: int,
    ) -> None:
        allowed = {
            "schema_version",
            "kind",
            "event",
            "attempt_id",
            "settled_at",
            "outcome",
            "automatic_retry_safe",
            "http_status",
            "charged_tokens",
            "server_request_id",
            "response_body_sha256",
            "response_record",
            "provider_audit",
        }
        if set(row) != allowed:
            raise ValueError(
                f"{self.path}:{line_number}: invalid settlement fields"
            )
        attempt_id = _validate_digest(row.get("attempt_id"), "attempt_id")
        if attempt_id not in self.starts:
            raise ValueError(
                f"{self.path}:{line_number}: settlement lacks its start"
            )
        if attempt_id in self.settlements:
            raise ValueError(
                f"{self.path}:{line_number}: duplicate attempt settlement"
            )
        _validate_timestamp(row.get("settled_at"), "settled_at")
        outcome = row.get("outcome")
        if outcome not in _OUTCOMES:
            raise ValueError(
                f"{self.path}:{line_number}: invalid attempt outcome"
            )
        if not isinstance(row.get("automatic_retry_safe"), bool):
            raise ValueError(
                f"{self.path}:{line_number}: automatic_retry_safe must be "
                "Boolean"
            )
        status = row.get("http_status")
        if status is not None and (
            not isinstance(status, int)
            or isinstance(status, bool)
            or not 100 <= status <= 599
        ):
            raise ValueError(
                f"{self.path}:{line_number}: invalid HTTP status"
            )
        charged = row.get("charged_tokens")
        if (
            not isinstance(charged, int)
            or isinstance(charged, bool)
            or charged < 0
            or charged
            > self.starts[attempt_id]["estimated_max_tokens"]
        ):
            raise ValueError(
                f"{self.path}:{line_number}: invalid charged token count"
            )
        server_request_id = row.get("server_request_id")
        if server_request_id is not None:
            _validate_nonempty(server_request_id, "server_request_id")
        response_digest = row.get("response_body_sha256")
        if response_digest is not None:
            _validate_digest(response_digest, "response_body_sha256")
        response_record = row.get("response_record")
        if response_record is not None and not isinstance(
            response_record,
            Mapping,
        ):
            raise ValueError(
                f"{self.path}:{line_number}: response_record must be an object"
            )
        audit = row.get("provider_audit")
        if audit is not None and not isinstance(audit, Mapping):
            raise ValueError(
                f"{self.path}:{line_number}: provider_audit must be an object"
            )
        if outcome in {"success", "rejected_provider_result"}:
            if not isinstance(audit, Mapping):
                raise ValueError(
                    f"{self.path}:{line_number}: final attempt lacks its audit"
                )
            accepted = audit.get("acceptance_status", "accepted") == "accepted"
            if accepted != (outcome == "success"):
                raise ValueError(
                    f"{self.path}:{line_number}: settlement/audit acceptance "
                    "status mismatch"
                )
            if row["automatic_retry_safe"]:
                raise ValueError(
                    f"{self.path}:{line_number}: a final attempt cannot be "
                    "marked retry-safe"
                )
        elif audit is not None:
            raise ValueError(
                f"{self.path}:{line_number}: nonfinal attempt embeds an audit"
            )
        if outcome == "transport_error" and status is not None:
            raise ValueError(
                f"{self.path}:{line_number}: transport error has HTTP status"
            )
        self.settlements[attempt_id] = row

    def start(
        self,
        request: Any,
        prepared: Any,
        *,
        started_at: str,
    ) -> str:
        if self.unresolved_attempt_ids:
            raise ProviderAttemptManualReviewRequired(
                "provider transport-attempt journal has an unresolved started "
                "attempt; its billing outcome is unknown and manual review is "
                "required before retry"
            )
        request_id = _validate_nonempty(request.request_id, "request_id")
        ordinal = self._ordinals.get(request_id, 0) + 1
        binding = {
            "provider": self.provider_name,
            "request_id": request_id,
            "prompt_sha256": request.prompt_sha256,
            "endpoint": prepared.endpoint,
            "request_body_sha256": prepared.body_sha256,
            "model_requested": self.model_requested,
            "client_request_id": prepared.client_request_id,
            "idempotency_key": getattr(prepared, "idempotency_key", None),
            "estimated_max_tokens": prepared.estimated_max_tokens,
            "attempt_ordinal": ordinal,
            "started_at": started_at,
        }
        attempt_id = _digest(binding)
        record = {
            "schema_version": 1,
            "kind": PROVIDER_ATTEMPT_KIND,
            "event": "started",
            "attempt_id": attempt_id,
            **binding,
        }
        self._parse_start(dict(record), line_number=0)
        try:
            append_jsonl(self.path, record)
        except Exception:
            self.starts.pop(attempt_id, None)
            self._ordinals[request_id] = ordinal - 1
            raise
        return attempt_id

    def settle(
        self,
        attempt_id: str,
        *,
        settled_at: str,
        outcome: str,
        automatic_retry_safe: bool,
        http_status: int | None,
        charged_tokens: int,
        server_request_id: str | None,
        response_body_sha256: str | None,
        response_record: Mapping[str, Any] | None,
        provider_audit: Mapping[str, Any] | None,
    ) -> None:
        record = {
            "schema_version": 1,
            "kind": PROVIDER_ATTEMPT_KIND,
            "event": "settled",
            "attempt_id": attempt_id,
            "settled_at": settled_at,
            "outcome": outcome,
            "automatic_retry_safe": automatic_retry_safe,
            "http_status": http_status,
            "charged_tokens": charged_tokens,
            "server_request_id": server_request_id,
            "response_body_sha256": response_body_sha256,
            "response_record": (
                dict(response_record)
                if response_record is not None
                else None
            ),
            "provider_audit": (
                dict(provider_audit)
                if provider_audit is not None
                else None
            ),
        }
        shadow = dict(self.settlements)
        self._parse_settlement(dict(record), line_number=0)
        try:
            append_jsonl(self.path, record)
        except Exception:
            self.settlements = shadow
            raise

    def validate_request(
        self,
        request: Any,
        provider: Any,
    ) -> None:
        """Validate every retained start for one request against current input."""

        prepared = provider.prepare(request)
        expected = {
            "provider": self.provider_name,
            "prompt_sha256": request.prompt_sha256,
            "endpoint": prepared.endpoint,
            "request_body_sha256": prepared.body_sha256,
            "model_requested": provider.config.model,
            "client_request_id": prepared.client_request_id,
            "idempotency_key": getattr(prepared, "idempotency_key", None),
            "estimated_max_tokens": prepared.estimated_max_tokens,
        }
        matching = [
            (attempt_id, start)
            for attempt_id, start in self.starts.items()
            if start["request_id"] == request.request_id
        ]
        for attempt_id, start in matching:
            if any(start.get(key) != value for key, value in expected.items()):
                raise ValueError(
                    "provider transport-attempt journal does not match the "
                    f"current request/model/origin for {request.request_id}"
                )
            settlement = self.settlements.get(attempt_id)
            if settlement is None:
                continue
            if (
                settlement["outcome"]
                in {"transport_error", "http_error", "invalid_response"}
                and settlement["charged_tokens"]
                != prepared.estimated_max_tokens
            ):
                raise ValueError(
                    "provider attempt journal has a nonconservative failed "
                    "attempt charge"
                )
            audit = settlement.get("provider_audit")
            if not isinstance(audit, Mapping):
                continue
            if audit.get("request_id") != request.request_id:
                raise ValueError(
                    "provider attempt settlement audit has the wrong request"
                )
            audit_binding = {
                "provider": self.provider_name,
                "prompt_sha256": request.prompt_sha256,
                "request_body_sha256": prepared.body_sha256,
                "model_requested": provider.config.model,
                "client_request_id": prepared.client_request_id,
            }
            idempotency_key = getattr(prepared, "idempotency_key", None)
            if idempotency_key is not None:
                audit_binding["idempotency_key"] = idempotency_key
            if any(
                audit.get(key) != value
                for key, value in audit_binding.items()
            ):
                raise ValueError(
                    "provider attempt settlement audit has the wrong "
                    "request/model/body binding"
                )
            if (
                settlement.get("response_body_sha256")
                != audit.get("raw_response_sha256")
            ):
                raise ValueError(
                    "provider attempt response/audit digest mismatch"
                )
            if audit.get("attempts") != start["attempt_ordinal"]:
                raise ValueError(
                    "provider audit attempt count differs from its physical "
                    "attempt journal"
                )
            expected_charge = _audit_usage_total(audit)
            if expected_charge is None:
                expected_charge = prepared.estimated_max_tokens
            if settlement["charged_tokens"] != expected_charge:
                raise ValueError(
                    "provider attempt journal charge differs from its final "
                    "audit usage"
                )
            expected_server_id = audit.get(
                "server_request_id",
                audit.get("generation_id"),
            )
            if (
                settlement.get("server_request_id") != expected_server_id
                or settlement.get("response_record")
                != audit.get("raw_response")
                or not isinstance(settlement.get("http_status"), int)
                or not 200 <= settlement["http_status"] <= 299
            ):
                raise ValueError(
                    "provider attempt response metadata differs from its final "
                    "audit"
                )

    def validate_requests(
        self,
        request_by_id: Mapping[str, Any],
        provider: Any,
    ) -> None:
        retained_ids = {start["request_id"] for start in self.starts.values()}
        unexpected = sorted(retained_ids - set(request_by_id))
        if unexpected:
            raise ValueError(
                "provider attempt journal references unexpected requests: "
                + ", ".join(unexpected)
            )
        for request_id in sorted(retained_ids):
            self.validate_request(request_by_id[request_id], provider)

    def accounting(self) -> tuple[int, int]:
        """Return settled physical attempts and conservative charged tokens."""

        return (
            len(self.settlements),
            sum(
                int(settlement["charged_tokens"])
                for settlement in self.settlements.values()
            ),
        )

    def embedded_final_audits(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for attempt_id, settlement in self.settlements.items():
            audit = settlement.get("provider_audit")
            if not isinstance(audit, Mapping):
                continue
            request_id = self.starts[attempt_id]["request_id"]
            if request_id in result:
                raise ValueError(
                    "provider attempt journal contains multiple final audits "
                    f"for {request_id}"
                )
            result[request_id] = dict(audit)
        return result

    def requests_without_final_audit(self) -> tuple[str, ...]:
        final = set(self.embedded_final_audits())
        attempted = {
            self.starts[attempt_id]["request_id"]
            for attempt_id in self.settlements
        }
        return tuple(sorted(attempted - final))

    def assert_safe_to_resume(self) -> None:
        if self.unresolved_attempt_ids:
            raise ProviderAttemptManualReviewRequired(
                "provider transport-attempt journal contains an unresolved "
                "started attempt; billing outcome is unknown and manual review "
                "is required before any retry"
            )
        nonfinal = self.requests_without_final_audit()
        if nonfinal:
            raise ProviderAttemptManualReviewRequired(
                "provider transport attempts exist without a final embedded "
                "accepted/rejected audit; manual review is required before "
                "retry: "
                + ", ".join(nonfinal)
            )

    def events_for_request_ids(
        self,
        request_ids: Sequence[str],
    ) -> tuple[Mapping[str, Any], ...]:
        retained = set(request_ids)
        events: list[Mapping[str, Any]] = []
        for attempt_id, start in self.starts.items():
            if start["request_id"] not in retained:
                continue
            events.append(start)
            settlement = self.settlements.get(attempt_id)
            if settlement is not None:
                events.append(settlement)
        return tuple(events)
