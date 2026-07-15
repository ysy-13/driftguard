from __future__ import annotations

import json
from typing import Any

from driftguard.evidence.leakage_guard import assert_agent_visible

from .tool_catalog import ToolCatalogRenderer


class AgentContextBuilder:
    def __init__(self, renderer: ToolCatalogRenderer | None = None, full_catalog: bool = True) -> None:
        self.renderer = renderer or ToolCatalogRenderer()
        self.full_catalog = full_catalog
        self.last_catalog_fingerprint = ""

    def build(
        self,
        base_prompt: str,
        policy_prompt: str,
        task_instruction: str,
        displayed_spec: dict[str, Any],
        observations: list[dict[str, Any]],
        remaining_budget: dict[str, int],
        policy_capabilities: dict[str, Any] | None = None,
    ) -> tuple[dict[str, str], ...]:
        catalog = self.renderer.render(displayed_spec) if self.full_catalog else _legacy_tool_summary(displayed_spec)
        self.last_catalog_fingerprint = self.renderer.fingerprint(displayed_spec) if self.full_catalog else ""
        payload = {
            "task": task_instruction,
            "displayed_tools": catalog,
            "observations": observations,
            "remaining_budget": remaining_budget,
        }
        if self.full_catalog:
            payload["policy_capabilities"] = policy_capabilities or {}
        assert_agent_visible(payload)
        return (
            {"role": "system", "content": base_prompt + "\n" + policy_prompt},
            {"role": "user", "content": json.dumps(payload, sort_keys=True)},
        )


def _legacy_tool_summary(spec: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"tool_id": operation["operationId"], "summary": operation.get("summary", ""), "path": path, "method": method}
        for path, item in spec.get("paths", {}).items()
        for method, operation in item.items()
        if isinstance(operation, dict) and "operationId" in operation
    ]


# Historical Phase 10 audit import. New v3 execution uses ToolCatalogRenderer.
_tool_summary = _legacy_tool_summary
