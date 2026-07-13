from __future__ import annotations

from typing import Any

from copy import deepcopy
from driftguard.sandbox.result import ToolResult, success

from .base import MutationStrategy
from .input_contract import InputContractStrategy
from .response_shape import ResponseShapeStrategy
from .state_effect import StateEffectStrategy
from .workflow_precondition import WorkflowPreconditionStrategy


STRATEGIES: dict[str, type[MutationStrategy]] = {
    "input_contract": InputContractStrategy,
    "response_shape": ResponseShapeStrategy,
    "workflow_precondition": WorkflowPreconditionStrategy,
    "state_effect": StateEffectStrategy,
}


class InjectionRegistry:
    def resolve(self, context: Any, tool: str) -> MutationStrategy | None:
        if not context.injection_active(tool):
            return None
        drift_case = context.profile.drift_case
        strategy_type = STRATEGIES.get(drift_case["drift_type"])
        if strategy_type is None:
            raise ValueError(f"unsupported drift type: {drift_case['drift_type']}")
        return strategy_type(drift_case, context.profile.family["shared_observation_signature"])

    def adapt_related_response(
        self,
        context: Any,
        tool: str,
        arguments: dict[str, Any],
        result: ToolResult,
    ) -> ToolResult:
        if not result.ok or not context.profile.runtime_contract_active(context.episode_index):
            return result
        case = context.profile.drift_case
        data = deepcopy(result.payload["data"])
        if case["drift_id"] == "WPD-02" and tool == "get_pipeline_status":
            data["verification_token"] = context.session.issue_token(
                context.session_key, "pipeline", arguments["repo_id"], str(arguments["run_id"]), context.episode_index
            )
            return success(result.status_code, data)
        if case["drift_id"] == "WPD-03" and tool == "get_member":
            data["membership_verification_token"] = context.session.issue_token(
                context.session_key, "membership", arguments["repo_id"], str(arguments["username"]), context.episode_index
            )
            return success(result.status_code, data)
        return result
