from __future__ import annotations

from typing import Any

from .models import ToolSpecPatch


class GroundTruthPatchEvaluator:
    """Late-bound evaluator; this object must only be created after transfer."""

    def evaluate(self, patch: ToolSpecPatch | None, expected_patch: dict[str, Any], execution: dict[str, Any]) -> dict[str, bool]:
        if patch is None:
            return {
                "semantic_patch": False, "exact_target": False, "operation": False,
                "immediate_repair": False, "future_transfer": False,
            }
        actual = patch.semantic_extensions["x-driftguard-patch-semantics"]
        expected_operation = expected_patch["operations"][0]
        operation_correct = all(
            _normalize(actual.get(key)) == _normalize(expected_operation.get(key))
            for key in ("operation", "target", "before", "after")
        )
        target_correct = patch.target_tool_id == expected_patch["target_tool"]
        return {
            "semantic_patch": target_correct and operation_correct,
            "exact_target": target_correct,
            "operation": operation_correct,
            "immediate_repair": bool(execution.get("repair_run", {}).get("passed")),
            "future_transfer": bool(execution.get("future_transfer", {}).get("patched_success")),
        }


def _normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalize(child) for key, child in sorted(value.items())}
    if isinstance(value, list):
        return [_normalize(child) for child in value]
    return value

