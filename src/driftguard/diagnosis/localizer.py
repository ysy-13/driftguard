from __future__ import annotations

from typing import Any

from driftguard.evidence.models import AgentView, thaw
from .models import LocalizationResult


def _operation(spec: dict[str, Any], tool: str) -> tuple[str, str, dict[str, Any]]:
    for path, path_item in spec.get("paths", {}).items():
        for method, operation in path_item.items():
            if isinstance(operation, dict) and operation.get("operationId") == tool:
                return path, method, operation
    raise ValueError(f"displayed specification has no operation {tool}")


def _ref_name(value: dict[str, Any]) -> str | None:
    ref = value.get("$ref") if isinstance(value, dict) else None
    return ref.rsplit("/", 1)[-1] if isinstance(ref, str) else None


def _request_schema(spec: dict[str, Any], operation: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    raw = operation.get("requestBody", {}).get("content", {}).get("application/json", {}).get("schema", {})
    name = _ref_name(raw)
    return name, spec.get("components", {}).get("schemas", {}).get(name, {}) if name else raw


def _response_data_schema(spec: dict[str, Any], operation: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    responses = operation.get("responses", {})
    code = sorted(key for key in responses if str(key).startswith("2"))[0]
    raw = responses[code]["content"]["application/json"]["schema"]
    envelope_name = _ref_name(raw)
    envelope = spec["components"]["schemas"].get(envelope_name, {}) if envelope_name else raw
    data = envelope.get("properties", {}).get("data", {})
    name = _ref_name(data)
    return name, spec["components"]["schemas"].get(name, {}) if name else data


class DriftLocalizer:
    def localize(self, view: AgentView, label: str) -> LocalizationResult:
        failure = next(
            (event for event in view.trace.events if event.normalized_observation), None
        )
        tool = failure.tool_id if failure else None
        if label != "PD" or failure is None or tool is None:
            return LocalizationResult(tool, "UNKNOWN", "unknown", None, "", "", (), 0.0)
        observation = thaw(failure.normalized_observation)
        spec = thaw(view.displayed_spec)
        path, method, operation = _operation(spec, tool)
        channel = observation.get("channel")
        field = observation.get("field")
        message = observation.get("message_pattern", "")
        if channel == "response_error":
            name, schema = _request_schema(spec, operation)
            properties = schema.get("properties", {})
            request = thaw(failure.displayed_request)
            actual_field = field
            field_not_displayed = field not in properties
            if field_not_displayed:
                candidates = [key for key in request if key in properties and (key in str(field) or str(field) in key)]
                actual_field = candidates[0] if candidates else field
            if "Invalid enum" in message:
                location_type = "request_field"
                location = f"/components/schemas/{name}/properties/{actual_field}/enum"
            elif "Unknown field" in message or field_not_displayed:
                location_type = "request_field"
                location = f"/components/schemas/{name}/properties/{actual_field}"
            elif properties.get(actual_field, {}).get("type") == "boolean":
                location_type = "request_requiredness"
                location = f"/components/schemas/{name}/properties/{actual_field}"
            elif "Missing required" in message:
                location_type = "request_requiredness"
                location = f"/components/schemas/{name}/required"
            else:
                location_type = "request_field"
                location = f"/components/schemas/{name}/properties/{actual_field}"
            category = "ICD"
        elif channel == "missing_output_field":
            name, _ = _response_data_schema(spec, operation)
            multiple = " or " in message.lower()
            location_type = "response_shape" if multiple else "response_field"
            location = f"/components/schemas/{name}/properties" + ("" if multiple else f"/{field}")
            category = "RSD"
        elif channel == "workflow_rejection":
            name, schema = _request_schema(spec, operation)
            if field and field in schema.get("properties", {}) and "enum" in schema["properties"][field]:
                location = f"/components/schemas/{name}/properties/{field}/enum"
            else:
                escaped = path.replace("~", "~0").replace("/", "~1")
                location = f"/paths/{escaped}/{method}/x-preconditions"
            location_type, category = "workflow_precondition", "WPD"
        elif channel == "state_mismatch":
            escaped = path.replace("~", "~0").replace("/", "~1")
            location = f"/paths/{escaped}/{method}/x-state-effects"
            location_type, category = "state_effect", "SED"
        else:
            return LocalizationResult(tool, "UNKNOWN", "unknown", None, "", "", (failure.event_id,), 0.0)
        return LocalizationResult(
            tool, category, location_type, location,
            "behavior described by displayed specification",
            message or str(observation.get("observed_effect", "runtime behavior differs")),
            (failure.event_id,), 1.0,
        )
