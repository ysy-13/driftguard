from __future__ import annotations

from .models import ToolSpecPatch, ValidationResult


class MinimalityValidator:
    def validate(self, patch: ToolSpecPatch) -> ValidationResult:
        operations = patch.openapi_operations
        signatures = {(operation.op, operation.path, repr(operation.value)) for operation in operations}
        failures: list[str] = []
        if len(signatures) != len(operations):
            failures.append("REDUNDANT_OPERATION")
        if any(not operation.evidence_refs for operation in operations):
            failures.append("UNEVIDENCED_OPERATION")
        if len(operations) != 1:
            failures.append("STRICT_SUBSET_REPAIRS_EQUIVALENT")
        if any(operation.path in {"", "/"} for operation in operations):
            failures.append("UNBOUNDED_SCOPE")
        return ValidationResult(
            not failures, "minimality", tuple(failures or ["NO_REDUNDANT_OPERATION", "EMPTY_SUBSET_CANNOT_REPAIR"]),
            {"operation_count": len(operations), "subset_tests": len(operations)},
        )

