from __future__ import annotations

from typing import Any

from driftguard.runners.bindings import resolve_binding
from driftguard.sandbox.permissions import ROLE_ORDER


def evaluate_condition(condition: dict[str, Any], completed_steps: dict[str, dict[str, Any]]) -> bool:
    operator = condition["operator"]
    if operator == "all":
        return all(evaluate_condition(child, completed_steps) for child in condition["conditions"])
    actual = resolve_binding(condition["source"], completed_steps)
    expected = condition.get("value")
    if operator == "eq":
        return actual == expected
    if operator == "neq":
        return actual != expected
    if operator == "is_null":
        return actual is None
    if operator == "not_null":
        return actual is not None
    if operator == "role_at_least":
        return actual in ROLE_ORDER and expected in ROLE_ORDER and ROLE_ORDER[actual] >= ROLE_ORDER[expected]
    if operator == "role_below":
        return actual in ROLE_ORDER and expected in ROLE_ORDER and ROLE_ORDER[actual] < ROLE_ORDER[expected]
    raise ValueError(f"unsupported condition operator: {operator}")
