from __future__ import annotations

from copy import deepcopy
from typing import Any

from driftguard.sandbox.result import ToolResult
from .base import MutationStrategy


class InputContractStrategy(MutationStrategy):
    phase = "input_contract"

    def before_validation(
        self,
        arguments: dict[str, Any],
        pre_state: dict[str, Any],
        context: Any,
    ) -> tuple[dict[str, Any], ToolResult | None]:
        translated = deepcopy(arguments)
        mutation = self.case["runtime_mutation"]
        operation = mutation["operation"]
        if operation == "add_required":
            if isinstance(mutation["after"], list):
                field = next(name for name in mutation["after"] if name not in mutation["before"])
            else:
                field = mutation["target"].split("/")[-1]
            return (translated, None) if field in translated else (translated, self.reject())
        if operation == "rename_input_field":
            old, new = mutation["before"], mutation["after"]
            if old in translated:
                return translated, self.reject()
            if new in translated:
                translated[old] = translated.pop(new)
            return translated, None
        if operation == "replace_enum_value":
            old_values, new_values = mutation["before"], mutation["after"]
            old = next(value for value in old_values if value not in new_values)
            new = next(value for value in new_values if value not in old_values)
            field = mutation["target"].split("/")[-2]
            if translated.get(field) == old:
                return translated, self.reject()
            if translated.get(field) == new:
                translated[field] = old
        return translated, None
