from __future__ import annotations

from copy import deepcopy
from typing import Any


SENSITIVE_KEYS = {"api_key", "authorization", "x-api-key", "access_token"}


def redact(value: Any, secrets: tuple[str, ...] = ()) -> Any:
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if str(key).lower() in SENSITIVE_KEYS else redact(child, secrets)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [redact(child, secrets) for child in value]
    if isinstance(value, tuple):
        return tuple(redact(child, secrets) for child in value)
    if isinstance(value, str):
        result = value
        for secret in secrets:
            if secret:
                result = result.replace(secret, "[REDACTED]")
        return result
    return deepcopy(value)


def assert_secret_absent(value: Any, secret: str | None) -> None:
    if secret and secret in repr(value):
        raise ValueError("API key leaked into a public artifact")

