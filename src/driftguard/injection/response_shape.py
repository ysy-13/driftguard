from __future__ import annotations

from copy import deepcopy

from driftguard.sandbox.result import ToolResult, success
from .base import MutationStrategy


class ResponseShapeStrategy(MutationStrategy):
    phase = "response_shape"

    def mutate_response(self, result: ToolResult) -> ToolResult:
        if not result.ok:
            return result
        data = deepcopy(result.payload["data"])
        mutation = self.case["runtime_mutation"]
        before, after = mutation["before"], mutation["after"]
        operation = mutation["operation"]
        if operation == "rename_output_field":
            if before in data:
                data[after] = data.pop(before)
        elif operation == "map_output_path":
            for old, new in zip(before, after):
                if old in data:
                    data[new] = data.pop(old)
        elif operation == "replace_output_path":
            nested = data.setdefault("permissions", {})
            for old, new_path in zip(before, after):
                if old in data:
                    nested[new_path.split(".")[-1]] = data.pop(old)
        return success(result.status_code, data)
