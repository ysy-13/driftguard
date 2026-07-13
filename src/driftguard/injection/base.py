from __future__ import annotations

from copy import deepcopy
from typing import Any

from driftguard.sandbox.result import ToolResult, failure


class MutationStrategy:
    phase = "none"

    def __init__(self, drift_case: dict[str, Any], signature: dict[str, Any]):
        self.case = deepcopy(drift_case)
        self.signature = deepcopy(signature)

    def reject(self) -> ToolResult:
        return failure(
            int(self.signature["http_status"]),
            str(self.signature["error_code"]),
            str(self.signature["message_pattern"]),
            self.signature.get("field"),
        )

    def before_validation(
        self,
        arguments: dict[str, Any],
        pre_state: dict[str, Any],
        context: Any,
    ) -> tuple[dict[str, Any], ToolResult | None]:
        return deepcopy(arguments), None

    def after_handler(
        self,
        working_state: dict[str, Any],
        pre_state: dict[str, Any],
        arguments: dict[str, Any],
        data: dict[str, Any],
        context: Any,
    ) -> dict[str, Any]:
        return data

    def mutate_response(self, result: ToolResult) -> ToolResult:
        return result
