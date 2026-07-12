from __future__ import annotations

from typing import Any

from driftguard.sandbox.errors import NotFoundError, PreconditionError
from .common import require_repository, require_run


def trigger_pipeline(state: dict[str, Any], arguments: dict[str, Any], timestamp: str | None) -> dict[str, Any]:
    repository = require_repository(state, arguments["repo_id"])
    workflow = repository["workflows"].get(arguments["workflow_id"])
    if workflow is None:
        raise NotFoundError(f"Workflow {arguments['workflow_id']!r} was not found.", "workflow_id")
    if workflow["state"] != "active":
        raise PreconditionError("The workflow is not active.", "workflow_id")
    if arguments["ref"] not in repository["branches"]:
        raise PreconditionError("The requested ref does not exist.", "ref")
    run_id = state["next_ids"][arguments["repo_id"]]["run_id"]
    state["next_ids"][arguments["repo_id"]]["run_id"] += 1
    run = {
        "run_id": run_id,
        "repo_id": arguments["repo_id"],
        "workflow_id": arguments["workflow_id"],
        "ref": arguments["ref"],
        "status": "queued",
        "conclusion": None,
        "attempt": 1,
        "retried_from": None,
        "created_at": timestamp,
        "updated_at": timestamp,
    }
    repository["pipeline_runs"][str(run_id)] = run
    return run


def get_pipeline_status(state: dict[str, Any], arguments: dict[str, Any], timestamp: str | None) -> dict[str, Any]:
    return require_run(require_repository(state, arguments["repo_id"]), arguments["run_id"])


def retry_pipeline(state: dict[str, Any], arguments: dict[str, Any], timestamp: str | None) -> dict[str, Any]:
    repository = require_repository(state, arguments["repo_id"])
    run = require_run(repository, arguments["run_id"])
    if run["status"] != "completed":
        raise PreconditionError("Only a completed pipeline run can be retried.", "run_id")
    if run["conclusion"] not in {"failure", "cancelled"}:
        raise PreconditionError("Only a failed or cancelled pipeline run can be retried.", "run_id")
    previous_attempt = run["attempt"]
    run["attempt"] = previous_attempt + 1
    run["retried_from"] = previous_attempt
    run["status"] = "queued"
    run["conclusion"] = None
    run["updated_at"] = timestamp
    return run
