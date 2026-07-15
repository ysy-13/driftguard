from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import re
from typing import Any

from driftguard.contracts.registry import HTTP_METHODS
from driftguard.evidence.leakage_guard import assert_agent_visible


class ToolCatalogRenderer:
    """Render only the supplied displayed OpenAPI document into a stable Agent view."""

    def render(self, displayed_spec: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(displayed_spec, dict):
            raise TypeError("displayed specification must be an object")
        tools = []
        for path in sorted(displayed_spec.get("paths", {})):
            path_item = self._resolve(displayed_spec, displayed_spec["paths"][path], ())
            for method in sorted(set(path_item) & HTTP_METHODS):
                operation = self._resolve(displayed_spec, path_item[method], ())
                if "operationId" not in operation:
                    continue
                parameters = [
                    self._resolve(displayed_spec, item, ())
                    for item in (*path_item.get("parameters", ()), *operation.get("parameters", ()))
                ]
                path_query: list[dict[str, Any]] = []
                flat_properties: dict[str, Any] = {}
                flat_required: list[str] = []
                for parameter in parameters:
                    schema = self._resolve(displayed_spec, parameter.get("schema", {}), ())
                    row = {
                        "name": parameter["name"], "location": parameter["in"],
                        "required": bool(parameter.get("required", False)), "schema": schema,
                    }
                    if parameter.get("description"):
                        row["description"] = parameter["description"]
                    path_query.append(row)
                    flat_properties[parameter["name"]] = schema
                    if row["required"]:
                        flat_required.append(parameter["name"])
                request_body = operation.get("requestBody")
                body_schema = None
                body_required = False
                if request_body:
                    request_body = self._resolve(displayed_spec, request_body, ())
                    body_required = bool(request_body.get("required", False))
                    body_schema = self._media_schema(displayed_spec, request_body)
                    flat_properties.update(body_schema.get("properties", {}))
                    flat_required.extend(body_schema.get("required", ()))
                input_schema = {
                    "type": "object", "additionalProperties": False,
                    "properties": flat_properties, "required": flat_required,
                }
                responses: dict[str, Any] = {}
                for status, raw_response in sorted(operation.get("responses", {}).items()):
                    response = self._resolve(displayed_spec, raw_response, ())
                    item: dict[str, Any] = {"description": response.get("description", "")}
                    if response.get("content"):
                        item["schema"] = self._media_schema(displayed_spec, response)
                    responses[str(status)] = item
                success_codes = [code for code in responses if code.isdigit() and code.startswith("2")]
                success_code = min(success_codes, key=int) if success_codes else None
                tools.append({
                    "tool_id": operation["operationId"],
                    "operation_id": operation["operationId"],
                    "name": operation.get("summary", operation["operationId"]),
                    "summary": operation.get("summary", ""),
                    "description": operation.get("description", ""),
                    "method": method.upper(), "path": path,
                    "parameters": path_query,
                    "request_body_required": body_required,
                    "request_body_schema": body_schema,
                    "input_schema": input_schema,
                    "success_response": None if success_code is None else {
                        "status": int(success_code), **responses[success_code],
                    },
                    "public_error_responses": {
                        code: value for code, value in responses.items() if code != success_code
                    },
                    "permission_requirement": operation.get("x-required-permission"),
                    "workflow": {
                        "preconditions": deepcopy(operation.get("x-preconditions", [])),
                        "state_effects": deepcopy(operation.get("x-state-effects", [])),
                        "effect_type": operation.get("x-effect-type"),
                        "idempotent": operation.get("x-idempotent"),
                    },
                })
        catalog = {"catalog_version": "agent-visible-v1", "tools": tools}
        assert_agent_visible(catalog)
        encoded = json.dumps(catalog, sort_keys=True).lower()
        hidden_terms = ("runtime_contract", "runtime_profile", "source_drift_id", "expected_patch", "ground_truth", '"drift_id"')
        if any(term in encoded for term in hidden_terms) or re.search(r"\b(?:icd|rsd|wpd|sed)-\d{2}\b", encoded):
            raise ValueError("rendered Tool Catalog contains hidden runtime or drift metadata")
        return catalog

    def stable_json(self, displayed_spec: dict[str, Any]) -> str:
        return json.dumps(self.render(displayed_spec), sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    def fingerprint(self, displayed_spec: dict[str, Any]) -> str:
        return hashlib.sha256(self.stable_json(displayed_spec).encode()).hexdigest()

    def _media_schema(self, document: dict[str, Any], value: dict[str, Any]) -> dict[str, Any]:
        content = value.get("content", {})
        media = content.get("application/json") or next(iter(content.values()), {})
        return self._resolve(document, media.get("schema", {}), ())

    def _resolve(self, document: dict[str, Any], value: Any, stack: tuple[str, ...]) -> Any:
        if isinstance(value, dict) and "$ref" in value:
            ref = value["$ref"]
            if not isinstance(ref, str) or not ref.startswith("#/"):
                raise ValueError(f"unsupported displayed-spec reference: {ref!r}")
            if ref in stack:
                return {"type": "object", "description": f"recursive reference {ref}"}
            current: Any = document
            for raw in ref[2:].split("/"):
                token = raw.replace("~1", "/").replace("~0", "~")
                if not isinstance(current, dict) or token not in current:
                    raise ValueError(f"unresolvable displayed-spec reference: {ref}")
                current = current[token]
            siblings = {key: child for key, child in value.items() if key != "$ref"}
            resolved = self._resolve(document, current, stack + (ref,))
            if siblings:
                if not isinstance(resolved, dict):
                    raise ValueError("$ref siblings require an object target")
                resolved = {**resolved, **self._resolve(document, siblings, stack)}
            return resolved
        if isinstance(value, dict):
            return {key: self._resolve(document, child, stack) for key, child in value.items()}
        if isinstance(value, list):
            return [self._resolve(document, child, stack) for child in value]
        return deepcopy(value)
