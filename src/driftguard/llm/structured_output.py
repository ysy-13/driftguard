from __future__ import annotations

import json
from typing import Any, Callable

from jsonschema import Draft202012Validator

from .errors import InvalidStructuredOutput


def parse_structured_output(raw_text: str, schema: dict[str, Any]) -> dict[str, Any]:
    try:
        value = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise InvalidStructuredOutput(f"invalid JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise InvalidStructuredOutput("structured output root must be an object")
    errors = sorted(Draft202012Validator(schema).iter_errors(value), key=lambda item: list(item.path))
    if errors:
        raise InvalidStructuredOutput(errors[0].message)
    return value


def parse_with_repairs(
    raw_text: str,
    schema: dict[str, Any],
    repair: Callable[[str, str], str],
    max_repairs: int,
) -> tuple[dict[str, Any], int, str]:
    current = raw_text
    for attempt in range(max_repairs + 1):
        try:
            return parse_structured_output(current, schema), attempt, current
        except InvalidStructuredOutput as exc:
            if attempt == max_repairs:
                raise
            current = repair(current, str(exc))
    raise AssertionError("unreachable")

