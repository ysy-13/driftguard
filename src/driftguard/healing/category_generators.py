from __future__ import annotations

import re
from typing import Any

from driftguard.diagnosis.models import LocalizationResult
from driftguard.evidence.models import AgentView, EvidenceEvent, thaw


def operation_path_for_tool(spec: dict[str, Any], tool_id: str) -> str:
    for path, item in spec.get("paths", {}).items():
        for method, operation in item.items():
            if isinstance(operation, dict) and operation.get("operationId") == tool_id:
                escaped = path.replace("~", "~0").replace("/", "~1")
                return f"/paths/{escaped}/{method}"
    raise ValueError(f"tool is absent from displayed specification: {tool_id}")


def _schema_at(spec: dict[str, Any], pointer: str) -> Any:
    current: Any = spec
    for raw in pointer[1:].split("/"):
        token = raw.replace("~1", "/").replace("~0", "~")
        current = current[token]
    return current


def _visible_data(event: EvidenceEvent) -> dict[str, Any]:
    response = thaw(event.visible_runtime_response)
    return response.get("payload", {}).get("data", {})


def _field_from_message(message: str, label: str) -> str | None:
    match = re.search(rf"{label}:\s*([A-Za-z_][A-Za-z0-9_]*)", message)
    return match.group(1) if match else None


def generate_icd(spec: dict[str, Any], location: LocalizationResult, event: EvidenceEvent) -> tuple[dict[str, Any], str, Any]:
    request = thaw(event.displayed_request)
    observation = thaw(event.normalized_observation)
    message = str(observation.get("message_pattern", ""))
    field = str(observation.get("field") or location.location_path.rsplit("/", 1)[-1])
    semantic_target = f"/adapter/{location.tool_id}"

    if "Invalid enum" in message:
        old = request[field]
        # A semantic role alias is a general vocabulary transformation, not a
        # benchmark-case lookup. Candidate validation decides whether it works.
        aliases = {"write": "developer"}
        new = aliases.get(str(old), f"{old}_v2")
        values = list(_schema_at(spec, location.location_path or ""))
        replacement = [new if value == old else value for value in values]
        semantics = {
            "operation": "replace_enum_value", "target": f"{semantic_target}/role_aliases",
            "before": values, "after": replacement,
            "request_transform": {"kind": "enum_alias", "field": field, "from": old, "to": new},
        }
        return semantics, location.location_path or "", replacement

    missing = _field_from_message(message, "Missing required field")
    unknown = _field_from_message(message, "Unknown field")
    if unknown and missing and unknown != missing:
        semantics = {
            "operation": "rename_input_field", "target": f"{semantic_target}/input",
            "before": unknown, "after": missing,
            "request_transform": {"kind": "rename", "from": unknown, "to": missing},
        }
        return semantics, (location.location_path or "") + "/x-driftguard-runtime-name", missing
    displayed_location_field = (location.location_path or "").rsplit("/", 1)[-1]
    if missing and "/properties/" in (location.location_path or "") and displayed_location_field != missing:
        semantics = {
            "operation": "rename_input_field", "target": f"{semantic_target}/input",
            "before": displayed_location_field, "after": missing,
            "request_transform": {"kind": "rename", "from": displayed_location_field, "to": missing},
        }
        return semantics, (location.location_path or "") + "/x-driftguard-runtime-name", missing
    if missing and missing not in request:
        location_path = location.location_path or ""
        schema_root = location_path.split("/properties/", 1)[0]
        if schema_root.endswith("/required"):
            schema_root = schema_root.rsplit("/", 1)[0]
        required_path = location.location_path if (location.location_path or "").endswith("/required") else schema_root + "/required"
        before = list(_schema_at(spec, required_path)) if required_path.endswith("/required") and _pointer_exists(spec, required_path) else []
        after = before + [missing]
        value: Any = missing if _pointer_exists(spec, required_path) else after
        patch_path = required_path + "/-" if _pointer_exists(spec, required_path) else required_path
        default_schema = _schema_at(spec, schema_root + f"/properties/{missing}")
        semantics = {
            "operation": "add_required", "target": f"{semantic_target}/required",
            "before": before if before else default_schema,
            "after": after if before else {"type": default_schema.get("type"), "required": True},
            "request_transform": {"kind": "supply_required", "field": missing, "value": default_schema.get("default", True)},
        }
        return semantics, patch_path, value

    # A missing runtime field paired with a displayed field is a rename even
    # when the runtime only reports the new side of the pair.
    if missing:
        displayed_candidates = [key for key in request if key not in {"repo_id", "issue_id", "run_id", "workflow_id", "username"}]
        old = displayed_candidates[-1] if displayed_candidates else field
        semantics = {
            "operation": "rename_input_field", "target": f"{semantic_target}/input",
            "before": old, "after": missing,
            "request_transform": {"kind": "rename", "from": old, "to": missing},
        }
        return semantics, (location.location_path or "") + "/x-driftguard-runtime-name", missing
    raise ValueError("visible ICD evidence does not identify a minimal change")


def _pointer_exists(document: dict[str, Any], pointer: str) -> bool:
    try:
        _schema_at(document, pointer)
        return True
    except (KeyError, IndexError, TypeError):
        return False


def generate_rsd(spec: dict[str, Any], location: LocalizationResult, event: EvidenceEvent) -> tuple[dict[str, Any], str, Any]:
    data = _visible_data(event)
    observation = thaw(event.normalized_observation)
    missing = str(observation.get("field"))
    semantic_target = f"/adapter/{location.tool_id}/output"
    if missing == "issue_id" and "id" in data:
        operation, before, after = "rename_output_field", "id", "issue_id"
        mapping = {"issue_id": "id"}
    elif missing == "status" and {"state", "result"}.issubset(data):
        operation, before, after = "map_output_path", ["state", "result"], ["status", "conclusion"]
        mapping = {"status": "state", "conclusion": "result"}
    elif missing == "role" and isinstance(data.get("permissions"), dict):
        operation = "replace_output_path"
        before, after = ["permissions.role_name", "permissions.base"], ["role", "base_permission"]
        mapping = {"role": "permissions.role_name", "base_permission": "permissions.base"}
    elif missing == "default_branch" and "defaultBranch" in data:
        operation, before, after = "rename_output_field", "defaultBranch", "default_branch"
        mapping = {"default_branch": "defaultBranch"}
    elif missing == "attempt" and "run_attempt" in data:
        operation, before, after = "rename_output_field", "run_attempt", "attempt"
        mapping = {"attempt": "run_attempt"}
    else:
        raise ValueError("visible response does not expose a unique parser mapping")
    semantics = {
        "operation": operation, "target": semantic_target, "before": before, "after": after,
        "response_mapping": mapping,
    }
    base = location.location_path or ""
    patch_path = (base if not base.endswith("/properties") else base.rsplit("/", 1)[0]) + "/x-driftguard-runtime-path"
    return semantics, patch_path, mapping


def generate_wpd(spec: dict[str, Any], location: LocalizationResult, event: EvidenceEvent) -> tuple[dict[str, Any], str, Any]:
    message = str(thaw(event.normalized_observation).get("message_pattern", ""))
    tool = str(location.tool_id)
    target = f"/adapter/{tool}/workflow"
    if "active assignee" in message:
        operation, before, after = "add_prerequisite", ["The issue exists and is open."], ["get_issue", "assign_issue_if_null", "close_issue"]
        workflow = {"kind": "active_assignee", "read": "get_issue", "write": "assign_issue"}
    elif "run state must be verified" in message:
        operation, before, after = "add_verification_binding", ["retry_pipeline"], ["get_pipeline_status", "bind verification_token", "retry_pipeline"]
        workflow = {"kind": "verification_binding", "read": "get_pipeline_status", "binding": "verification_token"}
    elif "Assignee validation" in message:
        operation, before, after = "add_verification_binding", ["assign_issue"], ["get_member", "bind membership_verification_token", "assign_issue"]
        workflow = {"kind": "verification_binding", "read": "get_member", "binding": "membership_verification_token"}
    elif "Role transition" in message:
        operation, before, after = "add_transition_step", ["read", "write"], ["read", "triage", "write"]
        workflow = {"kind": "ordered_transition", "intermediate": "triage"}
    elif "Initial role must be read" in message:
        operation, before, after = "restrict_initial_transition", ["add_member with target role"], ["add_member role=read", "update_member_role to target"]
        workflow = {"kind": "initial_then_promote", "initial_role": "read", "promotion_tool": "update_member_role"}
    else:
        raise ValueError("visible workflow rejection has no supported safe prerequisite")
    semantics = {"operation": operation, "target": target, "before": before, "after": after, "workflow": workflow}
    op_root = operation_path_for_tool(spec, tool)
    return semantics, op_root + "/x-driftguard-preconditions-v2", workflow


def generate_sed(spec: dict[str, Any], location: LocalizationResult, event: EvidenceEvent) -> tuple[dict[str, Any], str, Any]:
    observation = thaw(event.normalized_observation)
    tool = str(location.tool_id)
    target = f"/adapter/{tool}"
    observed = str(observation.get("observed_effect", ""))
    if "new run" in observed:
        operation, suffix = "change_resource_identity", "/effect"
        before, after = ["update original run"], ["read response run_id", "track new run", "query new run"]
        policy = {"kind": "new_resource", "confirmation_tool": "get_pipeline_status", "identity_field": "run_id"}
    else:
        operation, suffix = "add_postcondition_verification", "/postcondition"
        confirmation = {
            "close_issue": ("get_issue", "state=closed"),
            "update_member_role": ("get_member", "role and base_permission"),
            "add_member": ("get_member", "active"),
            "update_repository": ("get_repository", "default_branch"),
        }[tool]
        if tool == "add_member":
            before, after = ["add_member", "dependent write"], ["add_member", "get_member", "verify active", "dependent write"]
        elif tool == "update_repository":
            before, after = ["update_repository", "dependent operation"], ["update_repository", "get_repository", "verify default_branch", "dependent operation"]
        else:
            before, after = [tool], [tool, confirmation[0], f"verify {confirmation[1]}"]
        policy = {"kind": "read_after_write", "confirmation_tool": confirmation[0], "condition": confirmation[1]}
    semantics = {"operation": operation, "target": target + suffix, "before": before, "after": after, "observation_policy": policy}
    op_root = operation_path_for_tool(spec, tool)
    return semantics, op_root + "/x-driftguard-observation-policy", policy
