from __future__ import annotations

from copy import deepcopy
from typing import Any

from driftguard.sandbox.result import ToolResult


class ObservationNormalizer:
    def normalize(
        self,
        signature: dict[str, Any],
        result: ToolResult,
        pre_state: dict[str, Any],
        post_state: dict[str, Any],
        arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        channel = signature["channel"]
        valid = result.status_code == signature["http_status"] and result.ok == signature["response_success"]
        if channel == "response_error":
            error = result.payload.get("error", {})
            valid = (
                valid
                and error.get("code") == signature["error_code"]
                and error.get("field") == signature.get("field")
                and error.get("message") == signature["message_pattern"]
            )
        elif channel == "missing_output_field":
            valid = valid and signature["field"] not in result.payload.get("data", {})
        elif channel == "state_mismatch":
            arguments = arguments or {}
            data = result.payload.get("data", {})
            repo_id = arguments.get("repo_id")
            repository = post_state.get("repositories", {}).get(repo_id, {})
            expected = signature.get("expected_effect")
            mismatch = False
            if expected == "issue.state == closed":
                issue = repository.get("issues", {}).get(str(arguments.get("issue_id")), {})
                mismatch = issue.get("state") != data.get("state")
            elif expected == "member.role == requested role":
                member = repository.get("members", {}).get(arguments.get("username"), {})
                mismatch = member.get("role") != data.get("role")
            elif expected == "membership is active and usable":
                member = repository.get("members", {}).get(arguments.get("username"), {})
                mismatch = member.get("membership_state") != "active"
            elif expected == "original run attempt increments":
                original = repository.get("pipeline_runs", {}).get(str(arguments.get("run_id")), {})
                mismatch = data.get("run_id") != arguments.get("run_id") or original.get("attempt") == 1
            elif expected == "default_branch == requested branch":
                mismatch = repository.get("default_branch") != arguments.get("default_branch")
            valid = valid and mismatch
        if not valid:
            return {"channel": "unexpected", "http_status": result.status_code, "response_success": result.ok}
        return deepcopy(signature)
