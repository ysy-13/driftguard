from __future__ import annotations

from copy import deepcopy
from typing import Any

from driftguard.runtime.pending_effects import PendingEffect
from .base import MutationStrategy


class StateEffectStrategy(MutationStrategy):
    phase = "state_effect"

    def after_handler(
        self,
        working_state: dict[str, Any],
        pre_state: dict[str, Any],
        arguments: dict[str, Any],
        data: dict[str, Any],
        context: Any,
    ) -> dict[str, Any]:
        operation = self.case["runtime_mutation"]["operation"]
        tool = self.case["target_tool"]
        repo_id = arguments["repo_id"]
        response = deepcopy(data)
        if operation == "change_effect_timing":
            if tool == "close_issue":
                key = str(arguments["issue_id"])
                context.pending_effects.add(PendingEffect("get_issue", repo_id, ("issues", key), deepcopy(data)))
                working_state["repositories"][repo_id]["issues"][key] = deepcopy(pre_state["repositories"][repo_id]["issues"][key])
            elif tool == "update_member_role":
                key = arguments["username"]
                context.pending_effects.add(PendingEffect("get_member", repo_id, ("members", key), deepcopy(data)))
                working_state["repositories"][repo_id]["members"][key] = deepcopy(pre_state["repositories"][repo_id]["members"][key])
            elif tool == "update_repository":
                value = data["default_branch"]
                context.pending_effects.add(PendingEffect("get_repository", repo_id, ("default_branch",), value))
                working_state["repositories"][repo_id]["default_branch"] = pre_state["repositories"][repo_id]["default_branch"]
        elif operation == "replace_state_effect":
            key = arguments["username"]
            active = deepcopy(data)
            working_state["repositories"][repo_id]["members"].pop(key, None)
            context.pending_effects.add(PendingEffect("get_member", repo_id, ("members", key), active))
        elif operation == "change_resource_identity":
            original_key = str(arguments["run_id"])
            original = deepcopy(pre_state["repositories"][repo_id]["pipeline_runs"][original_key])
            new_id = working_state["next_ids"][repo_id]["run_id"]
            working_state["next_ids"][repo_id]["run_id"] += 1
            new_run = deepcopy(data)
            new_run["run_id"] = new_id
            new_run["attempt"] = 1
            new_run["retried_from"] = original["attempt"]
            new_run["created_at"] = new_run["updated_at"]
            working_state["repositories"][repo_id]["pipeline_runs"][original_key] = original
            working_state["repositories"][repo_id]["pipeline_runs"][str(new_id)] = new_run
            response = new_run
        return response
