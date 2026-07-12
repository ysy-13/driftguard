from __future__ import annotations

from typing import Any

from driftguard.sandbox.errors import NotFoundError


def require_repository(state: dict[str, Any], repo_id: str) -> dict[str, Any]:
    repository = state["repositories"].get(repo_id)
    if repository is None:
        raise NotFoundError(f"Repository {repo_id!r} was not found.", "repo_id")
    return repository


def require_issue(repository: dict[str, Any], issue_id: int) -> dict[str, Any]:
    issue = repository["issues"].get(str(issue_id))
    if issue is None:
        raise NotFoundError(f"Issue {issue_id} was not found.", "issue_id")
    return issue


def require_run(repository: dict[str, Any], run_id: int) -> dict[str, Any]:
    run = repository["pipeline_runs"].get(str(run_id))
    if run is None:
        raise NotFoundError(f"Pipeline run {run_id} was not found.", "run_id")
    return run


def require_member(repository: dict[str, Any], username: str) -> dict[str, Any]:
    member = repository["members"].get(username)
    if member is None or member.get("membership_state") != "active":
        raise NotFoundError(f"Active member {username!r} was not found.", "username")
    return member
