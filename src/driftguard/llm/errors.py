from __future__ import annotations

from enum import Enum


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


class ProviderError(RuntimeError):
    category = ErrorCategory.PROVIDER_ERROR


class ProviderTimeout(ProviderError):
    category = ErrorCategory.PROVIDER_TIMEOUT


class RateLimited(ProviderError):
    category = ErrorCategory.RATE_LIMITED


class InvalidStructuredOutput(ValueError):
    category = ErrorCategory.INVALID_STRUCTURED_OUTPUT

