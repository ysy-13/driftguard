from __future__ import annotations

from typing import Any


ROLE_ORDER = {"read": 0, "triage": 1, "write": 2, "maintain": 3, "admin": 4}
BASE_PERMISSION = {
    "read": "read",
    "triage": "read",
    "write": "write",
    "maintain": "write",
    "admin": "admin",
}


def role_at_least(actual: str, required: str) -> bool:
    return actual in ROLE_ORDER and required in ROLE_ORDER and ROLE_ORDER[actual] >= ROLE_ORDER[required]


def base_permission_for(role: str) -> str:
    try:
        return BASE_PERMISSION[role]
    except KeyError as exc:
        raise ValueError(f"unknown role: {role}") from exc


def actor_has_permission(state: dict[str, Any], actor_id: str, permission: str) -> bool:
    actor = state.get("actors", {}).get(actor_id)
    return actor is not None and permission in actor.get("permissions", [])
