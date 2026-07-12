from __future__ import annotations

from typing import Any

from driftguard.sandbox.errors import NotFoundError, PreconditionError
from driftguard.sandbox.permissions import base_permission_for
from .common import require_member, require_repository


def add_member(state: dict[str, Any], arguments: dict[str, Any], timestamp: str | None) -> dict[str, Any]:
    repository = require_repository(state, arguments["repo_id"])
    username = arguments["username"]
    if username not in state["users"]:
        raise NotFoundError(f"User {username!r} was not found.", "username")
    existing = repository["members"].get(username)
    if existing is not None and existing.get("membership_state") == "active":
        raise PreconditionError("The user is already an active member.", "username")
    role = arguments["role"]
    member = {
        "repo_id": arguments["repo_id"],
        "username": username,
        "role": role,
        "base_permission": base_permission_for(role),
        "membership_state": "active",
        "added_at": timestamp,
        "updated_at": timestamp,
    }
    repository["members"][username] = member
    return member


def get_member(state: dict[str, Any], arguments: dict[str, Any], timestamp: str | None) -> dict[str, Any]:
    return require_member(require_repository(state, arguments["repo_id"]), arguments["username"])


def update_member_role(state: dict[str, Any], arguments: dict[str, Any], timestamp: str | None) -> dict[str, Any]:
    repository = require_repository(state, arguments["repo_id"])
    member = require_member(repository, arguments["username"])
    role = arguments["role"]
    if member["role"] == "admin" and role != "admin":
        admin_count = sum(
            item.get("membership_state") == "active" and item.get("role") == "admin"
            for item in repository["members"].values()
        )
        if admin_count <= 1:
            raise PreconditionError("The repository's last admin cannot be demoted.", "role")
    member["role"] = role
    member["base_permission"] = base_permission_for(role)
    member["updated_at"] = timestamp
    return member
