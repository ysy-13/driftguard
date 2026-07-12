from __future__ import annotations

from typing import Any

from driftguard.sandbox.errors import NotFoundError, PreconditionError
from driftguard.sandbox.permissions import role_at_least
from .common import require_issue, require_repository


def create_issue(state: dict[str, Any], arguments: dict[str, Any], timestamp: str | None) -> dict[str, Any]:
    repository = require_repository(state, arguments["repo_id"])
    if not repository["issues_enabled"]:
        raise PreconditionError("Issues are disabled for this repository.")
    issue_id = state["next_ids"][arguments["repo_id"]]["issue_id"]
    state["next_ids"][arguments["repo_id"]]["issue_id"] += 1
    issue = {
        "issue_id": issue_id,
        "repo_id": arguments["repo_id"],
        "title": arguments["title"],
        "body": arguments["body"],
        "priority": arguments["priority"],
        "state": "open",
        "assignee": None,
        "resolution": None,
        "created_at": timestamp,
        "updated_at": timestamp,
        "closed_at": None,
    }
    repository["issues"][str(issue_id)] = issue
    return issue


def get_issue(state: dict[str, Any], arguments: dict[str, Any], timestamp: str | None) -> dict[str, Any]:
    return require_issue(require_repository(state, arguments["repo_id"]), arguments["issue_id"])


def assign_issue(state: dict[str, Any], arguments: dict[str, Any], timestamp: str | None) -> dict[str, Any]:
    repository = require_repository(state, arguments["repo_id"])
    issue = require_issue(repository, arguments["issue_id"])
    if issue["state"] != "open":
        raise PreconditionError("Only an open issue can be assigned.", "issue_id")
    username = arguments["assignee"]
    if username not in state["users"]:
        raise NotFoundError(f"User {username!r} was not found.", "assignee")
    member = repository["members"].get(username)
    if member is None or member.get("membership_state") != "active":
        raise PreconditionError("The assignee is not an active repository member.", "assignee")
    if not role_at_least(member["role"], "triage"):
        raise PreconditionError("The assignee must have at least the triage role.", "assignee")
    issue["assignee"] = username
    issue["updated_at"] = timestamp
    return issue


def close_issue(state: dict[str, Any], arguments: dict[str, Any], timestamp: str | None) -> dict[str, Any]:
    repository = require_repository(state, arguments["repo_id"])
    issue = require_issue(repository, arguments["issue_id"])
    if issue["state"] != "open":
        raise PreconditionError("Only an open issue can be closed.", "issue_id")
    issue["state"] = "closed"
    issue["resolution"] = arguments["resolution"]
    issue["closed_at"] = timestamp
    issue["updated_at"] = timestamp
    return issue
