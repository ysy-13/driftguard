from __future__ import annotations

from copy import deepcopy
import re
from typing import Any

from jsonschema import Draft202012Validator

from .registry import ToolContract


class InputValidationError(ValueError):
    def __init__(self, message: str, field: str | None = None):
        super().__init__(message)
        self.field = field


def _field_from_error(error: Any) -> str | None:
    if error.absolute_path:
        return str(next(iter(error.absolute_path)))
    match = re.search(r"'([^']+)' is a required property", error.message)
    if match:
        return match.group(1)
    match = re.search(r"\('([^']+)' was unexpected\)", error.message)
    return match.group(1) if match else None


class InputValidator:
    def validate(self, contract: ToolContract, arguments: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(arguments, dict):
            raise InputValidationError("Tool arguments must be an object.")
        normalized = deepcopy(arguments)
        for name, field_schema in contract.input_schema.get("properties", {}).items():
            if name not in normalized and "default" in field_schema:
                normalized[name] = deepcopy(field_schema["default"])

        errors = sorted(
            Draft202012Validator(contract.input_schema).iter_errors(normalized),
            key=lambda item: (list(item.absolute_path), item.message),
        )
        if errors:
            error = errors[0]
            raise InputValidationError(error.message, _field_from_error(error))

        if contract.body_schema is not None:
            body = {name: normalized[name] for name in contract.body_properties if name in normalized}
            body_errors = sorted(
                Draft202012Validator(contract.body_schema).iter_errors(body),
                key=lambda item: (list(item.absolute_path), item.message),
            )
            if body_errors:
                error = body_errors[0]
                raise InputValidationError(error.message, _field_from_error(error))
        return normalized
