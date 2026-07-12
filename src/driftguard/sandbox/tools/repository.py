from __future__ import annotations

from typing import Any

from driftguard.sandbox.errors import PreconditionError
from .common import require_repository


def get_repository(state: dict[str, Any], arguments: dict[str, Any], timestamp: str | None) -> dict[str, Any]:
    return require_repository(state, arguments["repo_id"])


def update_repository(state: dict[str, Any], arguments: dict[str, Any], timestamp: str | None) -> dict[str, Any]:
    repository = require_repository(state, arguments["repo_id"])
    if repository["archived"]:
        raise PreconditionError("Archived repositories cannot be updated.")
    if "default_branch" in arguments and arguments["default_branch"] not in repository["branches"]:
        raise PreconditionError("The supplied default_branch does not exist.", "default_branch")
    for field in ("description", "default_branch", "issues_enabled"):
        if field in arguments:
            repository[field] = arguments[field]
    return repository
