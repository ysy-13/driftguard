from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


# These are the semantic operations already consumed by Phase 8's
# RepairExecutor/category generators.  This module only exposes that contract.
ADAPTER_OPERATIONS: dict[str, dict[str, tuple[str, ...]]] = {
    "ICD": {
        "add_required": ("operation", "target", "before", "after", "request_transform"),
        "rename_input_field": ("operation", "target", "before", "after", "request_transform"),
        "replace_enum_value": ("operation", "target", "before", "after", "request_transform"),
    },
    "RSD": {
        "rename_output_field": ("operation", "target", "before", "after", "response_mapping"),
        "map_output_path": ("operation", "target", "before", "after", "response_mapping"),
        "replace_output_path": ("operation", "target", "before", "after", "response_mapping"),
    },
    "WPD": {
        "add_prerequisite": ("operation", "target", "before", "after", "workflow"),
        "add_verification_binding": ("operation", "target", "before", "after", "workflow"),
        "add_transition_step": ("operation", "target", "before", "after", "workflow"),
        "restrict_initial_transition": ("operation", "target", "before", "after", "workflow"),
    },
    "SED": {
        "add_postcondition_verification": ("operation", "target", "before", "after", "observation_policy"),
        "change_resource_identity": ("operation", "target", "before", "after", "observation_policy"),
    },
}


def operation_types(category: str | None = None) -> tuple[str, ...]:
    if category is not None:
        return tuple(ADAPTER_OPERATIONS.get(category, {}))
    return tuple(operation for rules in ADAPTER_OPERATIONS.values() for operation in rules)


def rendered_contract() -> dict[str, Any]:
    return {
        "source_of_truth": "Phase 8 ToolSpecPatch, SpecOverlay and validators",
        "json_patch_operations": ["add", "remove", "replace", "test"],
        "adapter_operations": {
            category: {
                operation: {"required_semantic_fields": list(required)}
                for operation, required in rules.items()
            }
            for category, rules in ADAPTER_OPERATIONS.items()
        },
        "normalized_location_fields": [
            "tool_id", "layer", "spec_pointer", "runtime_path", "adapter_operation_type",
        ],
        "scope_rule": "Every JSON Patch path must remain under the selected operation or referenced component schema.",
    }


def neutral_example() -> dict[str, Any]:
    return {
        "target_tool_id": "set_thermostat",
        "drift_category": "SED",
        "location": {
            "tool_id": "set_thermostat",
            "layer": "state_effect",
            "spec_pointer": "/paths/~1devices~1{device_key}~1temperature/put/x-observation-policy",
            "runtime_path": "device.temperature_celsius",
            "adapter_operation_type": "add_postcondition_verification",
        },
        "openapi_operations": [{
            "op": "add",
            "path": "/paths/~1devices~1{device_key}~1temperature/put/x-observation-policy",
            "value": {
                "kind": "read_after_write",
                "confirmation_tool": "read_thermostat",
                "condition": "temperature_celsius matches request",
            },
            "evidence_refs": ["visible-event-a", "visible-probe-b"],
            "reason_code": "VISIBLE_STATE_CONFIRMATION_REQUIRED",
        }],
        "semantic_extensions": {"x-driftguard-patch-semantics": {
            "operation": "add_postcondition_verification",
            "target": "/adapter/set_thermostat/postcondition",
            "before": ["set_thermostat"],
            "after": ["set_thermostat", "read_thermostat", "verify temperature_celsius"],
            "observation_policy": {
                "kind": "read_after_write",
                "confirmation_tool": "read_thermostat",
                "condition": "temperature_celsius matches request",
            },
        }},
        "evidence_refs": ["visible-event-a", "visible-probe-b"],
        "expected_agent_behavior_change": "Confirm the persisted device value after a successful write.",
        "confidence": 0.9,
        "concise_reason": "A successful write response conflicts with the subsequent visible read.",
    }


def validate_semantics(category: str, location: Mapping[str, Any], extensions: Mapping[str, Any]) -> None:
    semantics = extensions.get("x-driftguard-patch-semantics")
    if not isinstance(semantics, Mapping):
        raise ValueError("missing x-driftguard-patch-semantics")
    operation = semantics.get("operation")
    if operation != location.get("adapter_operation_type"):
        raise ValueError("adapter operation differs from normalized location")
    required = ADAPTER_OPERATIONS.get(category, {}).get(str(operation))
    if required is None:
        raise ValueError("adapter operation is not allowed for drift category")
    missing = [field for field in required if field not in semantics]
    if missing:
        raise ValueError("adapter semantics missing required fields: " + ", ".join(missing))
    target = semantics.get("target")
    expected_prefix = f"/adapter/{location.get('tool_id')}"
    if not isinstance(target, str) or not target.startswith(expected_prefix):
        raise ValueError("adapter semantic target is outside normalized tool scope")


def schema_fragment(category: str) -> dict[str, Any]:
    return deepcopy(rendered_contract()["adapter_operations"].get(category, {}))
