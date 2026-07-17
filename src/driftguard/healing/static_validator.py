from __future__ import annotations

from copy import deepcopy
from typing import Any

from openapi_spec_validator import validate

from .models import ToolSpecPatch, ValidationResult
from .overlay import SpecOverlay


class StaticPatchValidator:
    def validate(self, patch: ToolSpecPatch, displayed_spec: dict[str, Any]) -> tuple[ValidationResult, dict[str, Any] | None]:
        failures: list[str] = []
        location = patch.normalized_location
        if location:
            if location.get("tool_id") != patch.target_tool_id:
                failures.append("NORMALIZED_LOCATION_TOOL_MISMATCH")
            pointer = location.get("spec_pointer")
            if pointer and any(not operation.path.startswith(pointer) for operation in patch.openapi_operations):
                failures.append("NORMALIZED_LOCATION_SCOPE_MISMATCH")
            semantics = patch.semantic_extensions.get("x-driftguard-patch-semantics", {})
            if location.get("adapter_operation_type") != semantics.get("operation"):
                failures.append("NORMALIZED_ADAPTER_OPERATION_MISMATCH")
        if len(patch.openapi_operations) > 8:
            failures.append("PATCH_SCOPE_EXCEEDED")
        if any(operation.path in {"", "/"} for operation in patch.openapi_operations):
            failures.append("WHOLE_DOCUMENT_REPLACEMENT_FORBIDDEN")
        if any(not operation.evidence_refs for operation in patch.openapi_operations):
            failures.append("MISSING_OPERATION_EVIDENCE")
        allowed_prefixes = _allowed_prefixes(displayed_spec, patch)
        if any(not any(operation.path.startswith(prefix) for prefix in allowed_prefixes) for operation in patch.openapi_operations):
            failures.append("UNRELATED_TOOL_PATH")
        operation_ids_before = _operation_ids(displayed_spec)
        security_before = deepcopy(displayed_spec.get("security"))
        patched: dict[str, Any] | None = None
        try:
            overlay = SpecOverlay(displayed_spec)
            patched = overlay.apply(patch)
            validate(patched)
        except Exception as exc:  # validator exposes several exception classes
            failures.append(f"INVALID_PATCH:{type(exc).__name__}")
        if patched is not None:
            if _operation_ids(patched) != operation_ids_before:
                failures.append("OPERATION_ID_CHANGED")
            if patched.get("security") != security_before:
                failures.append("SECURITY_SCHEME_CHANGED")
            if patch.target_tool_id not in operation_ids_before:
                failures.append("UNKNOWN_TARGET_TOOL")
        result = ValidationResult(
            not failures, "static", tuple(failures or ["OPENAPI_31_VALID", "PATCH_SCOPE_VALID"]),
            {"operation_count": len(patch.openapi_operations), "atomic_apply": patched is not None},
        )
        return result, patched if result.passed else None


def _operation_ids(document: dict[str, Any]) -> set[str]:
    return {
        operation["operationId"]
        for item in document.get("paths", {}).values()
        for operation in item.values()
        if isinstance(operation, dict) and "operationId" in operation
    }


def _allowed_prefixes(document: dict[str, Any], patch: ToolSpecPatch) -> tuple[str, ...]:
    prefixes: list[str] = []
    if patch.location_path.startswith("/components/schemas/"):
        tokens = patch.location_path.split("/")
        prefixes.append("/".join(tokens[:4]))
    for path, item in document.get("paths", {}).items():
        for method, operation in item.items():
            if isinstance(operation, dict) and operation.get("operationId") == patch.target_tool_id:
                escaped = path.replace("~", "~0").replace("/", "~1")
                prefixes.append(f"/paths/{escaped}/{method}")
    return tuple(prefixes)
