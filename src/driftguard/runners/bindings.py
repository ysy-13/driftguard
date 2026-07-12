from __future__ import annotations

import re
from typing import Any


REFERENCE = re.compile(r"^\$steps\.([^.]+)\.(.+)$")


class BindingError(ValueError):
    pass


def resolve_binding(reference: str, completed_steps: dict[str, dict[str, Any]]) -> Any:
    match = REFERENCE.fullmatch(reference)
    if match is None:
        raise BindingError(f"invalid step binding: {reference}")
    step_id, path = match.groups()
    if step_id not in completed_steps:
        raise BindingError(f"step {step_id!r} has not completed")
    current: Any = completed_steps[step_id]
    for field in path.split("."):
        if not isinstance(current, dict) or field not in current:
            raise BindingError(f"field {field!r} does not exist in {reference}")
        current = current[field]
    return current


def resolve_bindings(value: Any, completed_steps: dict[str, dict[str, Any]]) -> Any:
    if isinstance(value, str) and value.startswith("$steps."):
        return resolve_binding(value, completed_steps)
    if isinstance(value, dict):
        return {key: resolve_bindings(child, completed_steps) for key, child in value.items()}
    if isinstance(value, list):
        return [resolve_bindings(child, completed_steps) for child in value]
    return value
