from __future__ import annotations

from enum import Enum
import re
from typing import Any, Iterable


class ErrorCategory(str, Enum):
    PROVIDER_ERROR = "PROVIDER_ERROR"
    PROVIDER_TIMEOUT = "PROVIDER_TIMEOUT"
    RATE_LIMITED = "RATE_LIMITED"
    INVALID_STRUCTURED_OUTPUT = "INVALID_STRUCTURED_OUTPUT"
    UNKNOWN_TOOL = "UNKNOWN_TOOL"
    INVALID_ARGUMENTS = "INVALID_ARGUMENTS"
    TOOL_RUNTIME_FAILURE = "TOOL_RUNTIME_FAILURE"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    SAFETY_BLOCKED = "SAFETY_BLOCKED"
    DIAGNOSIS_ERROR = "DIAGNOSIS_ERROR"
    PATCH_REJECTED = "PATCH_REJECTED"
    TASK_FAILED = "TASK_FAILED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class FailureLayer(str, Enum):
    DNS = "DNS"
    CONNECTION = "CONNECTION"
    TLS = "TLS"
    TIMEOUT = "TIMEOUT"
    HTTP = "HTTP"
    RESPONSE_PARSE = "RESPONSE_PARSE"
    UNKNOWN = "UNKNOWN"


_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(authorization|x-api-key|api[_-]?key|access[_-]?token)"
    r"\s*[:=]\s*([^\s,;]+)"
)
_BEARER = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+")


def sanitize_provider_text(
    value: Any, *, secrets: Iterable[str] = (), limit: int = 512,
) -> str:
    """Return a compact Provider diagnostic without credentials or large bodies."""
    text = "" if value is None else str(value)
    for secret in secrets:
        if secret:
            text = text.replace(secret, "[REDACTED]")
    text = _BEARER.sub("Bearer [REDACTED]", text)
    text = _SECRET_ASSIGNMENT.sub(lambda match: f"{match.group(1)}=[REDACTED]", text)
    text = " ".join(text.split())
    if len(text) > limit:
        text = text[: max(0, limit - 15)] + "...[TRUNCATED]"
    return text


class ProviderError(RuntimeError):
    category = ErrorCategory.PROVIDER_ERROR

    def __init__(
        self,
        sanitized_message: str,
        *,
        status_code: int | None = None,
        provider_error_code: str | None = None,
        request_id: str | None = None,
        retryable: bool = False,
        attempt_number: int = 1,
        failure_layer: FailureLayer | str = FailureLayer.UNKNOWN,
        exception_type: str | None = None,
        latency_ms: float = 0.0,
    ) -> None:
        message = sanitize_provider_text(sanitized_message)
        super().__init__(message)
        self.status_code = status_code
        self.provider_error_code = (
            sanitize_provider_text(provider_error_code, limit=128)
            if provider_error_code is not None else None
        )
        self.sanitized_message = message
        self.request_id = (
            sanitize_provider_text(request_id, limit=128)
            if request_id is not None else None
        )
        self.retryable = bool(retryable)
        self.attempt_number = max(0, int(attempt_number))
        self.failure_layer = FailureLayer(failure_layer)
        self.exception_type = (
            sanitize_provider_text(exception_type, limit=128)
            if exception_type is not None else None
        )
        self.latency_ms = max(0.0, float(latency_ms))

    @property
    def actual_network_attempts(self) -> int:
        return self.attempt_number

    def public_dict(self) -> dict[str, Any]:
        category = self.category.value if isinstance(self.category, ErrorCategory) else str(self.category)
        return {
            "category": category,
            "error_category": category,
            "status_code": self.status_code,
            "http_status": self.status_code,
            "provider_error_code": self.provider_error_code,
            "sanitized_message": self.sanitized_message,
            "request_id": self.request_id,
            "retryable": self.retryable,
            "attempt_number": self.attempt_number,
            "actual_network_attempts": self.actual_network_attempts,
            "failure_layer": self.failure_layer.value,
            "exception_type": self.exception_type,
            "latency_ms": self.latency_ms,
        }


class ProviderTimeout(ProviderError):
    category = ErrorCategory.PROVIDER_TIMEOUT

    def __init__(self, sanitized_message: str, **kwargs: Any) -> None:
        kwargs.setdefault("retryable", True)
        kwargs.setdefault("failure_layer", FailureLayer.TIMEOUT)
        super().__init__(sanitized_message, **kwargs)


class RateLimited(ProviderError):
    category = ErrorCategory.RATE_LIMITED

    def __init__(self, sanitized_message: str, **kwargs: Any) -> None:
        kwargs.setdefault("status_code", 429)
        kwargs.setdefault("retryable", True)
        kwargs.setdefault("failure_layer", FailureLayer.HTTP)
        super().__init__(sanitized_message, **kwargs)


class InvalidStructuredOutput(ValueError):
    category = ErrorCategory.INVALID_STRUCTURED_OUTPUT
