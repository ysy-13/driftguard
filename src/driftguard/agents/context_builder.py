from __future__ import annotations

import json
from typing import Any

from driftguard.evidence.leakage_guard import assert_agent_visible


class AgentContextBuilder:
    def build(
        self,
        base_prompt: str,
        policy_prompt: str,
        task_instruction: str,
        displayed_spec: dict[str, Any],
        observations: list[dict[str, Any]],
        remaining_budget: dict[str, int],
    ) -> tuple[dict[str, str], ...]:
        payload = {
            "task": task_instruction,
            "displayed_tools": _tool_summary(displayed_spec),
            "observations": observations,
            "remaining_budget": remaining_budget,
        }
        assert_agent_visible(payload)
        return (
            {"role": "system", "content": base_prompt + "\n" + policy_prompt},
            {"role": "user", "content": json.dumps(payload, sort_keys=True)},
        )


def _tool_summary(spec: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"tool_id": operation["operationId"], "summary": operation.get("summary", ""), "path": path, "method": method}
        for path, item in spec.get("paths", {}).items()
        for method, operation in item.items()
        if isinstance(operation, dict) and "operationId" in operation
    ]

