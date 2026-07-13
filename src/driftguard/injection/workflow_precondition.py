from __future__ import annotations

from copy import deepcopy
from typing import Any

from driftguard.sandbox.result import ToolResult
from .base import MutationStrategy


class WorkflowPreconditionStrategy(MutationStrategy):
    phase = "workflow_precondition"

    def before_validation(
        self,
        arguments: dict[str, Any],
        pre_state: dict[str, Any],
        context: Any,
    ) -> tuple[dict[str, Any], ToolResult | None]:
        translated = deepcopy(arguments)
        operation = self.case["runtime_mutation"]["operation"]
        repo = pre_state["repositories"].get(arguments.get("repo_id"), {})
        reject = False
        if operation == "add_prerequisite":
            issue = repo.get("issues", {}).get(str(arguments.get("issue_id")), {})
            assignee = issue.get("assignee")
            member = repo.get("members", {}).get(assignee, {})
            reject = not assignee or member.get("membership_state") != "active"
        elif operation == "add_verification_binding":
            token_field = "verification_token" if self.case["target_tool"] == "retry_pipeline" else "membership_verification_token"
            token = translated.pop(token_field, None)
            if self.case["target_tool"] == "retry_pipeline":
                reject = not context.session.consume_token(
                    token, "pipeline", arguments["repo_id"], str(arguments["run_id"]), context.episode_index
                )
            else:
                reject = not context.session.consume_token(
                    token, "membership", arguments["repo_id"], str(arguments["assignee"]), context.episode_index
                )
        elif operation == "add_transition_step":
            roles = ["read", "triage", "write", "maintain", "admin"]
            current = repo.get("members", {}).get(arguments.get("username"), {}).get("role")
            requested = arguments.get("role")
            reject = current in roles and requested in roles and roles.index(requested) - roles.index(current) > 1
        elif operation == "restrict_initial_transition":
            reject = translated.get("role", "read") != "read"
        return translated, self.reject() if reject else None
