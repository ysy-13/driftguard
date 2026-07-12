#!/usr/bin/env python3
"""Validate DriftGuard's canonical OpenAPI contract and benchmark invariants."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Iterator

import yaml


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "benchmark" / "openapi" / "driftguard_openapi_v1.yaml"
HTTP_METHODS = {"get", "put", "post", "delete", "options", "head", "patch", "trace"}
EXPECTED_OPERATION_BINDINGS = (
    ("T01", "get", "/repositories/{repo_id}", "get_repository", "read", True, "repository:read", "200", "RepositorySuccess"),
    ("T02", "patch", "/repositories/{repo_id}", "update_repository", "write", True, "repository:admin", "200", "RepositorySuccess"),
    ("T03", "post", "/repositories/{repo_id}/issues", "create_issue", "write", False, "issues:write", "201", "IssueSuccess"),
    ("T04", "get", "/repositories/{repo_id}/issues/{issue_id}", "get_issue", "read", True, "issues:read", "200", "IssueSuccess"),
    ("T05", "put", "/repositories/{repo_id}/issues/{issue_id}/assignee", "assign_issue", "write", True, "issues:write", "200", "IssueSuccess"),
    ("T06", "post", "/repositories/{repo_id}/issues/{issue_id}/close", "close_issue", "write", True, "issues:write", "200", "IssueSuccess"),
    ("T07", "post", "/repositories/{repo_id}/workflows/{workflow_id}/runs", "trigger_pipeline", "write", False, "actions:write", "201", "PipelineSuccess"),
    ("T08", "get", "/repositories/{repo_id}/pipeline-runs/{run_id}", "get_pipeline_status", "read", True, "actions:read", "200", "PipelineSuccess"),
    ("T09", "post", "/repositories/{repo_id}/pipeline-runs/{run_id}/retry", "retry_pipeline", "write", False, "actions:write", "201", "PipelineSuccess"),
    ("T10", "put", "/repositories/{repo_id}/members/{username}", "add_member", "write", "conditional", "repository:admin", "201", "MemberSuccess"),
    ("T11", "get", "/repositories/{repo_id}/members/{username}", "get_member", "read", True, "metadata:read", "200", "MemberSuccess"),
    ("T12", "patch", "/repositories/{repo_id}/members/{username}", "update_member_role", "write", True, "repository:admin", "200", "MemberSuccess"),
)
EXPECTED_OPERATION_IDS = {binding[3] for binding in EXPECTED_OPERATION_BINDINGS}
EXPECTED_TOOL_IDS = {binding[0] for binding in EXPECTED_OPERATION_BINDINGS}
EXPECTED_MODEL_FIELDS = {
    "Repository": {
        "repo_id", "name", "description", "visibility", "default_branch",
        "issues_enabled", "archived",
    },
    "Issue": {
        "issue_id", "repo_id", "title", "body", "priority", "state",
        "assignee", "resolution", "created_at", "updated_at", "closed_at",
    },
    "PipelineRun": {
        "run_id", "repo_id", "workflow_id", "ref", "status", "conclusion",
        "attempt", "retried_from", "created_at", "updated_at",
    },
    "Member": {
        "repo_id", "username", "role", "base_permission", "membership_state",
        "added_at", "updated_at",
    },
}
EXPECTED_ERROR_CODES = {
    "BAD_REQUEST", "PERMISSION_DENIED", "NOT_FOUND", "PRECONDITION_FAILED",
    "VALIDATION_ERROR", "RATE_LIMITED", "SERVICE_UNAVAILABLE",
}
REQUIRED_EXTENSIONS = {
    "x-canonical-tool-id",
    "x-canonical-contract-version",
    "x-effect-type",
    "x-idempotent",
    "x-required-permission",
    "x-preconditions",
    "x-state-effects",
}


class ContractValidationError(Exception):
    """Raised when one or more canonical contract invariants fail."""


def load_spec(path: Path = SPEC_PATH) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            document = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as exc:
        raise ContractValidationError(f"{path}: YAML could not be parsed: {exc}") from exc
    if not isinstance(document, dict):
        raise ContractValidationError(f"{path}: document root must be a mapping")
    return document


def iter_operations(spec: dict[str, Any]) -> Iterator[tuple[str, str, dict[str, Any]]]:
    for path, path_item in spec.get("paths", {}).items():
        if not isinstance(path_item, dict):
            continue
        for method, operation in path_item.items():
            if method.lower() in HTTP_METHODS and isinstance(operation, dict):
                yield path, method.lower(), operation


def resolve_ref(spec: dict[str, Any], ref: str) -> Any:
    if not ref.startswith("#/"):
        raise ContractValidationError(f"external reference is not allowed: {ref}")
    current: Any = spec
    for raw_token in ref[2:].split("/"):
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, dict) or token not in current:
            raise ContractValidationError(f"unresolvable internal reference: {ref}")
        current = current[token]
    return current


def dereference(spec: dict[str, Any], value: Any) -> Any:
    seen: set[str] = set()
    while isinstance(value, dict) and set(value) == {"$ref"}:
        ref = value["$ref"]
        if ref in seen:
            raise ContractValidationError(f"circular reference chain: {ref}")
        seen.add(ref)
        value = resolve_ref(spec, ref)
    return value


def iter_refs(value: Any, location: str = "$") -> Iterator[tuple[str, str]]:
    if isinstance(value, dict):
        for key, child in value.items():
            child_location = f"{location}.{key}"
            if key == "$ref" and isinstance(child, str):
                yield child_location, child
            else:
                yield from iter_refs(child, child_location)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from iter_refs(child, f"{location}[{index}]")


def _iter_key_values(
    value: Any, target_key: str, location: str = "$"
) -> Iterator[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            child_location = f"{location}.{key}"
            if key == target_key:
                yield child_location, child
            yield from _iter_key_values(child, target_key, child_location)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _iter_key_values(child, target_key, f"{location}[{index}]")


def schema(spec: dict[str, Any], name: str) -> dict[str, Any]:
    return spec["components"]["schemas"][name]


def validate_contract(spec: dict[str, Any]) -> None:
    errors: list[str] = []

    def expect(location: str, actual: Any, expected: Any) -> None:
        if actual != expected:
            errors.append(f"{location}: expected {expected!r}, got {actual!r}")

    version = str(spec.get("openapi", ""))
    if not version.startswith("3.1."):
        errors.append(f"$.openapi: expected a 3.1.x version, got {version!r}")

    expect("$.info.title", spec.get("info", {}).get("title"), "DriftGuard Software Engineering Sandbox API")
    expect("$.info.version", str(spec.get("info", {}).get("version")), "1.0.0")
    expect("$.servers", spec.get("servers"), [{"url": "http://localhost:8000"}])
    expect(
        "$.tags",
        {tag.get("name") for tag in spec.get("tags", []) if isinstance(tag, dict)},
        {"Repository", "Issue", "Pipeline", "Membership"},
    )
    expect("$.security", spec.get("security"), [{"SandboxBearerAuth": []}])
    bearer = spec.get("components", {}).get("securitySchemes", {}).get("SandboxBearerAuth", {})
    expect("$.components.securitySchemes.SandboxBearerAuth.type", bearer.get("type"), "http")
    expect("$.components.securitySchemes.SandboxBearerAuth.scheme", bearer.get("scheme"), "bearer")
    expect(
        "$.components.securitySchemes.SandboxBearerAuth.bearerFormat",
        bearer.get("bearerFormat"),
        "sandbox-token",
    )

    operations = list(iter_operations(spec))
    if len(operations) != 12:
        errors.append(f"$.paths: expected exactly 12 operations, found {len(operations)}")

    operation_ids = [operation.get("operationId") for _, _, operation in operations]
    operation_id_set = set(operation_ids)
    if operation_id_set != EXPECTED_OPERATION_IDS:
        errors.append(
            "$.paths: operationId mismatch; "
            f"missing={sorted(EXPECTED_OPERATION_IDS - operation_id_set, key=str)}, "
            f"extra={sorted(operation_id_set - EXPECTED_OPERATION_IDS, key=str)}"
        )
    if len(operation_ids) != len(operation_id_set):
        errors.append("$.paths: operationId values must be unique")

    tool_ids = [operation.get("x-canonical-tool-id") for _, _, operation in operations]
    tool_id_set = set(tool_ids)
    if tool_id_set != EXPECTED_TOOL_IDS:
        errors.append(
            "$.paths: canonical tool ID mismatch; "
            f"missing={sorted(EXPECTED_TOOL_IDS - tool_id_set, key=str)}, "
            f"extra={sorted(tool_id_set - EXPECTED_TOOL_IDS, key=str)}"
        )
    if len(tool_ids) != len(tool_id_set):
        errors.append("$.paths: x-canonical-tool-id values must be unique")

    actual_by_location = {(method, path): operation for path, method, operation in operations}
    expected_locations = {(binding[1], binding[2]) for binding in EXPECTED_OPERATION_BINDINGS}
    if set(actual_by_location) != expected_locations:
        errors.append(
            "$.paths: path/method bindings mismatch; "
            f"missing={sorted(expected_locations - set(actual_by_location))}, "
            f"extra={sorted(set(actual_by_location) - expected_locations)}"
        )
    for tool_id, method, path, operation_id, effect, idempotent, permission, status, success_name in EXPECTED_OPERATION_BINDINGS:
        operation = actual_by_location.get((method, path))
        if operation is None:
            continue
        location = f"$.paths.{path}.{method}"
        expect(f"{location}.operationId", operation.get("operationId"), operation_id)
        expect(f"{location}.x-canonical-tool-id", operation.get("x-canonical-tool-id"), tool_id)
        expect(f"{location}.x-effect-type", operation.get("x-effect-type"), effect)
        expect(f"{location}.x-idempotent", operation.get("x-idempotent"), idempotent)
        expect(f"{location}.x-required-permission", operation.get("x-required-permission"), permission)
        responses = operation.get("responses", {})
        actual_success_statuses = {str(code) for code in responses if str(code).startswith("2")}
        expect(f"{location}.responses success statuses", actual_success_statuses, {status})
        try:
            success_ref = responses[status]["content"]["application/json"]["schema"]["$ref"]
            expect(
                f"{location}.responses.{status}.content.application/json.schema.$ref",
                success_ref,
                f"#/components/schemas/{success_name}",
            )
            error_ref = responses["default"]["$ref"]
            expect(f"{location}.responses.default.$ref", error_ref, "#/components/responses/ApiErrorResponse")
        except (KeyError, TypeError) as exc:
            errors.append(f"{location}.responses: missing canonical response structure: {exc}")

    for path, method, operation in operations:
        location = f"$.paths.{path}.{method}"
        missing = REQUIRED_EXTENSIONS - operation.keys()
        if missing:
            errors.append(f"{location}: missing extensions {sorted(missing)}")
        if operation.get("x-canonical-contract-version") != "1.0":
            errors.append(f"{location}.x-canonical-contract-version: must be '1.0'")
        if operation.get("x-effect-type") not in {"read", "write"}:
            errors.append(f"{location}.x-effect-type: must be read or write")
        if not operation.get("x-preconditions"):
            errors.append(f"{location}.x-preconditions: must not be empty")
        if operation.get("x-effect-type") == "write" and not operation.get("x-state-effects"):
            errors.append(f"{location}.x-state-effects: write operation effects must not be empty")
        if operation.get("x-effect-type") == "read" and operation.get("x-state-effects") != []:
            errors.append(f"{location}.x-state-effects: read operation must have no state effects")

        declared: dict[tuple[str, str], dict[str, Any]] = {}
        combined_parameters = list(spec["paths"][path].get("parameters", [])) + list(
            operation.get("parameters", [])
        )
        for parameter_value in combined_parameters:
            try:
                parameter = dereference(spec, parameter_value)
            except ContractValidationError as exc:
                errors.append(f"{location}.parameters: {exc}")
                continue
            if isinstance(parameter, dict):
                declared[(parameter.get("name"), parameter.get("in"))] = parameter
        template_parameters = set(re.findall(r"{([^{}]+)}", path))
        declared_path_parameters = {
            name for name, where in declared if where == "path"
        }
        if declared_path_parameters != template_parameters:
            errors.append(
                f"{location}.parameters: path declarations must exactly match the template; "
                f"declared={sorted(declared_path_parameters, key=str)}, "
                f"template={sorted(template_parameters)}"
            )
        for parameter_name in template_parameters:
            parameter = declared.get((parameter_name, "path"))
            if parameter is None:
                errors.append(f"{location}: path parameter {parameter_name!r} is not declared")
            elif parameter.get("required") is not True:
                errors.append(f"{location}: path parameter {parameter_name!r} must be required=true")

        success_responses = [
            response
            for status, response in operation.get("responses", {}).items()
            if str(status).startswith("2")
        ]
        if not success_responses:
            errors.append(f"{location}.responses: missing success response")
        for response_value in success_responses:
            try:
                response = dereference(spec, response_value)
                success_schema = dereference(
                    spec, response["content"]["application/json"]["schema"]
                )
                required = set(success_schema.get("required", []))
                properties = set(success_schema.get("properties", {}))
                if not {"ok", "data"}.issubset(required & properties):
                    errors.append(f"{location}.responses: success schema must require ok and data")
            except (ContractValidationError, KeyError, TypeError) as exc:
                errors.append(f"{location}.responses: invalid success schema: {exc}")

    for location, ref in iter_refs(spec):
        try:
            resolve_ref(spec, ref)
        except ContractValidationError as exc:
            errors.append(f"{location}: {exc}")

    try:
        api_error = schema(spec, "ApiError")
        if not {"ok", "error"}.issubset(
            set(api_error.get("required", [])) & set(api_error.get("properties", {}))
        ):
            errors.append("$.components.schemas.ApiError: must require ok and error")
        error_detail = schema(spec, "ErrorDetail")
        error_codes = set(error_detail["properties"]["code"]["enum"])
        expect("$.components.schemas.ErrorDetail error codes", error_codes, EXPECTED_ERROR_CODES)
        expect(
            "$.components.schemas.ErrorDetail.required",
            set(error_detail.get("required", [])),
            {"code", "message", "field", "retryable"},
        )
    except KeyError as exc:
        errors.append(f"$.components.schemas: missing error schema field {exc}")

    try:
        create_request = schema(spec, "CreateIssueRequest")
        create_required = set(create_request.get("required", []))
        if "title" not in create_required or "priority" in create_required:
            errors.append("CreateIssueRequest: title must be required and priority optional")
        expect("CreateIssueRequest.priority.default", create_request["properties"]["priority"].get("default"), "medium")
        expect(
            "CreateIssueRequest.priority.enum",
            set(create_request["properties"]["priority"].get("enum", [])),
            {"low", "medium", "high"},
        )
        trigger_properties = schema(spec, "TriggerPipelineRequest")["properties"]
        if "ref" not in trigger_properties or "branch" in trigger_properties:
            errors.append("TriggerPipelineRequest: canonical input is ref, not branch")
        assign_properties = schema(spec, "AssignIssueRequest")["properties"]
        if "assignee" not in assign_properties or "assignee_username" in assign_properties:
            errors.append("AssignIssueRequest: canonical input is assignee")
        retry_request = schema(spec, "RetryPipelineRequest")
        if "failed_only" in set(retry_request.get("required", [])):
            errors.append("RetryPipelineRequest: failed_only must remain optional")
        expect("RetryPipelineRequest.failed_only.default", retry_request["properties"]["failed_only"].get("default"), True)
        member = schema(spec, "Member")
        roles = set(member["properties"]["role"]["enum"])
        states = set(member["properties"]["membership_state"]["enum"])
        canonical_roles = {"read", "triage", "write", "maintain", "admin"}
        expect("Member.role.enum", roles, canonical_roles)
        add_member_role = schema(spec, "AddMemberRequest")["properties"]["role"]
        expect("AddMemberRequest.role.enum", set(add_member_role["enum"]), canonical_roles)
        expect("AddMemberRequest.role.default", add_member_role.get("default"), "read")
        expect(
            "UpdateMemberRoleRequest.role.enum",
            set(schema(spec, "UpdateMemberRoleRequest")["properties"]["role"]["enum"]),
            canonical_roles,
        )
        if states != {"active"}:
            errors.append("Member.membership_state: canonical v1 permits only active")
        pipeline_properties = schema(spec, "PipelineRun")["properties"]
        if "attempt" not in pipeline_properties or "run_attempt" in pipeline_properties:
            errors.append("PipelineRun: canonical field is attempt")
        repository_properties = schema(spec, "Repository")["properties"]
        if "default_branch" not in repository_properties or "defaultBranch" in repository_properties:
            errors.append("Repository: canonical field is default_branch")
        issue_properties = schema(spec, "Issue")["properties"]
        if "issue_id" not in issue_properties or "id" in issue_properties:
            errors.append("Issue: canonical identifier is issue_id")

        by_id = {operation["operationId"]: operation for _, _, operation in operations}
        close_preconditions = " ".join(by_id["close_issue"]["x-preconditions"]).lower()
        if "assignee" in close_preconditions:
            errors.append("close_issue: canonical v1 must not require an assignee")
        retry_effects = " ".join(by_id["retry_pipeline"]["x-state-effects"]).lower()
        if "existing run" not in retry_effects or "run_id is preserved" not in retry_effects:
            errors.append("retry_pipeline: effects must update the existing run and preserve run_id")
        add_member_text = " ".join(
            by_id["add_member"]["x-preconditions"] + by_id["add_member"]["x-state-effects"]
        ).lower()
        if "promote" in add_member_text or "first add as read" in add_member_text:
            errors.append("add_member: canonical v1 must allow direct assignment of any valid role")
    except (KeyError, TypeError) as exc:
        errors.append(f"canonical drift checks could not inspect required field: {exc}")

    for model_name, expected_fields in EXPECTED_MODEL_FIELDS.items():
        try:
            model = schema(spec, model_name)
            expect(f"{model_name}.properties", set(model.get("properties", {})), expected_fields)
            expect(f"{model_name}.required", set(model.get("required", [])), expected_fields)
            expect(f"{model_name}.additionalProperties", model.get("additionalProperties"), False)
        except KeyError as exc:
            errors.append(f"$.components.schemas.{model_name}: missing field {exc}")

    for model_name, fields in {
        "Issue": {"created_at", "updated_at", "closed_at"},
        "PipelineRun": {"created_at", "updated_at"},
        "Member": {"added_at", "updated_at"},
    }.items():
        for field in fields:
            field_schema = schema(spec, model_name).get("properties", {}).get(field, {})
            expect(f"{model_name}.{field}.format", field_schema.get("format"), "date-time")

    nullable_fields = {
        ("Issue", "assignee"), ("Issue", "resolution"), ("Issue", "closed_at"),
        ("PipelineRun", "conclusion"), ("PipelineRun", "retried_from"),
    }
    for model_name, field in nullable_fields:
        actual_type = schema(spec, model_name)["properties"][field].get("type")
        if not isinstance(actual_type, list) or "null" not in actual_type:
            errors.append(f"{model_name}.{field}.type: must use an OpenAPI 3.1 null union")
    for location, value in _iter_key_values(spec, "nullable"):
        errors.append(f"{location}: nullable is an OpenAPI 3.0 keyword and is forbidden ({value!r})")

    try:
        input_value = schema(spec, "PipelineInputValue")
        variants = input_value.get("anyOf")
        expect(
            "PipelineInputValue.anyOf types",
            {variant.get("type") for variant in variants or []},
            {"string", "number", "integer", "boolean"},
        )
        if "oneOf" in input_value:
            errors.append("PipelineInputValue: use anyOf because JSON Schema number includes integer")
    except (KeyError, TypeError) as exc:
        errors.append(f"PipelineInputValue: invalid canonical schema: {exc}")

    if errors:
        raise ContractValidationError("\n".join(f"- {error}" for error in errors))


def validate_with_library(spec: dict[str, Any]) -> str:
    try:
        from openapi_spec_validator import validate
    except ImportError as exc:
        raise ContractValidationError(
            "openapi-spec-validator is not installed; run `python -m pip install -r requirements.txt`"
        ) from exc
    try:
        validate(spec)
    except Exception as exc:  # library exception types vary by release
        raise ContractValidationError(f"formal OpenAPI validation failed: {exc}") from exc
    return "valid"


def main() -> int:
    try:
        spec = load_spec()
        validate_contract(spec)
        validate_with_library(spec)
    except ContractValidationError as exc:
        print(f"OpenAPI validation failed:\n{exc}", file=sys.stderr)
        return 1

    print("OpenAPI 3.1 validation passed.")
    print("Operations: 12")
    print("Tool IDs: T01-T12")
    print("Internal references: valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
