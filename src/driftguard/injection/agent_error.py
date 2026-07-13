from __future__ import annotations

from copy import deepcopy
from typing import Any

from driftguard.runtime.execution_profile import ExecutionMode
from driftguard.sandbox.result import ToolResult


REQUESTED_PATHS = {
    "RSD-01": ["data.issue_id"],
    "RSD-02": ["data.status", "data.conclusion"],
    "RSD-03": ["data.role", "data.base_permission"],
    "RSD-04": ["data.default_branch"],
    "RSD-05": ["data.attempt"],
}


def _paths(value: Any, prefix: str = "") -> list[str]:
    if not isinstance(value, dict):
        return [prefix]
    paths: list[str] = []
    for key, child in value.items():
        path = f"{prefix}.{key}" if prefix else key
        paths.extend(_paths(child, path))
    return paths


class AgentFaultInjector:
    """Produces agent-side plans and interpretation records without changing runtime responses."""

    @staticmethod
    def metadata(context: Any) -> dict[str, Any]:
        return {"fault_layer": "agent_behavior", "episode": context.episode_index}

    def active(self, context: Any) -> bool:
        return context.profile.mode == ExecutionMode.AGENT_ERROR and context.agent_fault_active

    def plan(self, context: Any) -> dict[str, Any]:
        drift_type = context.profile.drift_case["drift_type"]
        category = "response_interpretation" if drift_type == "response_shape" else drift_type
        return {"category": category, "active": self.active(context), "modifies_runtime": False}

    def inject_call(self, context: Any, proposed: dict[str, Any]) -> dict[str, Any]:
        faulty = deepcopy(proposed)
        if not self.active(context):
            return faulty
        mutation = context.profile.drift_case["runtime_mutation"]
        operation = mutation["operation"]
        if operation == "add_required":
            field = (
                next(name for name in mutation["after"] if name not in mutation["before"])
                if isinstance(mutation["after"], list)
                else mutation["target"].split("/")[-1]
            )
            faulty.pop(field, None)
        elif operation == "rename_input_field":
            old, new = mutation["before"], mutation["after"]
            if new in faulty:
                faulty[old] = faulty.pop(new)
        elif operation == "replace_enum_value":
            old_values, new_values = mutation["before"], mutation["after"]
            old = next(value for value in old_values if value not in new_values)
            new = next(value for value in new_values if value not in old_values)
            field = mutation["target"].split("/")[-2]
            if faulty.get(field) == new:
                faulty[field] = old
        elif operation == "add_verification_binding":
            faulty.pop("verification_token", None)
            faulty.pop("membership_verification_token", None)
        return faulty

    def interpret(self, context: Any, result: ToolResult) -> dict[str, Any] | None:
        if not self.active(context):
            return None
        drift_id = context.profile.drift_case["drift_id"]
        requested = REQUESTED_PATHS.get(drift_id)
        if requested is None:
            if context.profile.drift_case["drift_type"] == "state_effect":
                requested = ["canonical_postcondition"]
            else:
                return None
        available = _paths(result.payload)
        record = {
            "interpretation_status": "failed",
            "requested_path": requested[0] if len(requested) == 1 else requested,
            "available_paths": available,
        }
        context.session.response_interpretation = deepcopy(record)
        return record


AgentErrorHook = AgentFaultInjector
