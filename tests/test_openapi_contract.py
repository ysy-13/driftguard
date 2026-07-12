from __future__ import annotations

from copy import deepcopy
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.validate_openapi as validator  # noqa: E402
from scripts.validate_openapi import (  # noqa: E402
    ContractValidationError,
    EXPECTED_OPERATION_BINDINGS,
    EXPECTED_OPERATION_IDS,
    EXPECTED_TOOL_IDS,
    REQUIRED_EXTENSIONS,
    dereference,
    iter_operations,
    iter_refs,
    load_spec,
    resolve_ref,
    schema,
    validate_contract,
    validate_with_library,
)


@pytest.fixture(scope="module")
def spec():
    return load_spec()


@pytest.fixture(scope="module")
def operations(spec):
    return list(iter_operations(spec))


@pytest.fixture(scope="module")
def by_id(operations):
    return {operation["operationId"]: operation for _, _, operation in operations}


def test_openapi_version(spec):
    assert spec["openapi"].startswith("3.1.")


def test_exact_operation_count(operations):
    assert len(operations) == 12


def test_expected_operation_ids(operations):
    assert {operation["operationId"] for _, _, operation in operations} == EXPECTED_OPERATION_IDS


def test_exact_tool_path_method_bindings(operations):
    actual = {
        (
            operation["x-canonical-tool-id"],
            method,
            path,
            operation["operationId"],
            operation["x-effect-type"],
            operation["x-idempotent"],
            operation["x-required-permission"],
        )
        for path, method, operation in operations
    }
    expected = {binding[:7] for binding in EXPECTED_OPERATION_BINDINGS}
    assert actual == expected


def test_unique_tool_ids(operations):
    tool_ids = [operation["x-canonical-tool-id"] for _, _, operation in operations]
    assert set(tool_ids) == EXPECTED_TOOL_IDS
    assert len(tool_ids) == len(set(tool_ids))


def test_required_extensions_present(operations):
    for _, _, operation in operations:
        assert REQUIRED_EXTENSIONS <= operation.keys()
        assert operation["x-canonical-contract-version"] == "1.0"
        assert operation["x-preconditions"]


def test_all_internal_refs_resolve(spec):
    refs = list(iter_refs(spec))
    assert refs
    for _, ref in refs:
        assert resolve_ref(spec, ref) is not None


def test_path_parameters_are_required(spec, operations):
    import re

    for path, _, operation in operations:
        parameters = [dereference(spec, item) for item in operation.get("parameters", [])]
        for name in re.findall(r"{([^{}]+)}", path):
            matches = [p for p in parameters if p.get("name") == name and p.get("in") == "path"]
            assert len(matches) == 1
            assert matches[0]["required"] is True


def test_read_write_classification(by_id):
    reads = {"get_repository", "get_issue", "get_pipeline_status", "get_member"}
    for operation_id, operation in by_id.items():
        expected = "read" if operation_id in reads else "write"
        assert operation["x-effect-type"] == expected
        if expected == "write":
            assert operation["x-state-effects"]


def test_idempotency_metadata(by_id):
    false_ids = {"create_issue", "trigger_pipeline", "retry_pipeline"}
    for operation_id, operation in by_id.items():
        expected = "conditional" if operation_id == "add_member" else operation_id not in false_ids
        assert operation["x-idempotent"] == expected


def test_canonical_input_fields(spec):
    create_issue = schema(spec, "CreateIssueRequest")
    assert create_issue["required"] == ["title"]
    assert "priority" not in create_issue["required"]
    assert create_issue["properties"]["priority"]["default"] == "medium"
    assert create_issue["properties"]["title"]["minLength"] == 1
    assert create_issue["properties"]["title"]["maxLength"] == 256
    assert create_issue["properties"]["body"]["maxLength"] == 10000

    trigger = schema(spec, "TriggerPipelineRequest")
    assert "ref" in trigger["properties"]
    assert "branch" not in trigger["properties"]

    assign = schema(spec, "AssignIssueRequest")
    assert set(assign["properties"]) == {"assignee"}

    retry = schema(spec, "RetryPipelineRequest")
    assert "failed_only" not in retry.get("required", [])
    assert retry["properties"]["failed_only"]["default"] is True

    pipeline_input = schema(spec, "PipelineInputValue")
    assert "oneOf" not in pipeline_input
    assert {item["type"] for item in pipeline_input["anyOf"]} == {
        "string", "number", "integer", "boolean"
    }

    update_repository = schema(spec, "UpdateRepositoryRequest")
    assert update_repository["minProperties"] == 1
    assert update_repository["properties"]["description"]["maxLength"] == 500

    roles = {"read", "triage", "write", "maintain", "admin"}
    add_member = schema(spec, "AddMemberRequest")
    assert set(add_member["properties"]["role"]["enum"]) == roles
    assert add_member["properties"]["role"]["default"] == "read"
    assert set(schema(spec, "UpdateMemberRoleRequest")["properties"]["role"]["enum"]) == roles

    for request_name in (
        "UpdateRepositoryRequest", "CreateIssueRequest", "AssignIssueRequest",
        "CloseIssueRequest", "TriggerPipelineRequest", "RetryPipelineRequest",
        "AddMemberRequest", "UpdateMemberRoleRequest",
    ):
        assert schema(spec, request_name)["additionalProperties"] is False


def test_request_body_required_flags(by_id):
    expected = {
        "get_repository": None,
        "update_repository": True,
        "create_issue": True,
        "get_issue": None,
        "assign_issue": True,
        "close_issue": False,
        "trigger_pipeline": True,
        "get_pipeline_status": None,
        "retry_pipeline": False,
        "add_member": False,
        "get_member": None,
        "update_member_role": True,
    }
    for operation_id, required in expected.items():
        request_body = by_id[operation_id].get("requestBody")
        if required is None:
            assert request_body is None
        else:
            assert request_body["required"] is required


def test_canonical_output_fields(spec):
    repository = schema(spec, "Repository")
    assert "default_branch" in repository["properties"]
    assert "defaultBranch" not in repository["properties"]

    issue = schema(spec, "Issue")
    assert "issue_id" in issue["properties"]
    assert "id" not in issue["properties"]

    pipeline = schema(spec, "PipelineRun")
    assert "attempt" in pipeline["properties"]
    assert "run_attempt" not in pipeline["properties"]

    for success_name in ("RepositorySuccess", "IssueSuccess", "PipelineSuccess", "MemberSuccess"):
        success = schema(spec, success_name)
        assert {"ok", "data"} <= set(success["required"])
        assert success["properties"]["ok"]["const"] is True

    expected_fields = {
        "Repository": {"repo_id", "name", "description", "visibility", "default_branch", "issues_enabled", "archived"},
        "Issue": {"issue_id", "repo_id", "title", "body", "priority", "state", "assignee", "resolution", "created_at", "updated_at", "closed_at"},
        "PipelineRun": {"run_id", "repo_id", "workflow_id", "ref", "status", "conclusion", "attempt", "retried_from", "created_at", "updated_at"},
        "Member": {"repo_id", "username", "role", "base_permission", "membership_state", "added_at", "updated_at"},
    }
    for model_name, fields in expected_fields.items():
        model = schema(spec, model_name)
        assert set(model["properties"]) == fields
        assert set(model["required"]) == fields
        assert model["additionalProperties"] is False


def test_error_codes(spec):
    expected = {
        "BAD_REQUEST",
        "PERMISSION_DENIED",
        "NOT_FOUND",
        "PRECONDITION_FAILED",
        "VALIDATION_ERROR",
        "RATE_LIMITED",
        "SERVICE_UNAVAILABLE",
    }
    assert set(schema(spec, "ErrorDetail")["properties"]["code"]["enum"]) == expected
    api_error = schema(spec, "ApiError")
    assert {"ok", "error"} <= set(api_error["required"])
    assert api_error["properties"]["ok"]["const"] is False
    assert set(schema(spec, "ErrorDetail")["required"]) == {
        "code", "message", "field", "retryable"
    }


def test_no_drift_cases_leak_into_canonical_v1(spec, by_id):
    member = schema(spec, "Member")
    assert "write" in member["properties"]["role"]["enum"]
    assert "developer" not in member["properties"]["role"]["enum"]
    assert member["properties"]["membership_state"]["enum"] == ["active"]
    assert "assignee" not in " ".join(by_id["close_issue"]["x-preconditions"]).lower()
    retry_effects = " ".join(by_id["retry_pipeline"]["x-state-effects"]).lower()
    assert "existing run" in retry_effects
    assert "run_id is preserved" in retry_effects
    assert "DRIFT_DETECTED" not in schema(spec, "ErrorDetail")["properties"]["code"]["enum"]
    validate_contract(spec)


def test_formal_openapi_validation(spec):
    assert validate_with_library(spec) == "valid"


def test_invalid_yaml_is_rejected(tmp_path):
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("paths: [unterminated", encoding="utf-8")
    with pytest.raises(ContractValidationError, match="YAML could not be parsed"):
        load_spec(invalid)


def test_main_returns_nonzero_for_invalid_yaml(tmp_path, monkeypatch, capsys):
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("paths: [unterminated", encoding="utf-8")
    monkeypatch.setattr(validator, "load_spec", lambda: load_spec(invalid))
    assert validator.main() == 1
    assert "OpenAPI validation failed" in capsys.readouterr().err


def _delete_operation(spec):
    spec["paths"]["/repositories/{repo_id}"].pop("get")


def _duplicate_tool_id(spec):
    spec["paths"]["/repositories/{repo_id}"]["patch"]["x-canonical-tool-id"] = "T01"


def _require_priority(spec):
    schema(spec, "CreateIssueRequest")["required"].append("priority")


def _rename_ref_to_branch(spec):
    request = schema(spec, "TriggerPipelineRequest")
    request["properties"]["branch"] = request["properties"].pop("ref")
    request["required"] = ["branch"]


def _rename_assignee(spec):
    request = schema(spec, "AssignIssueRequest")
    request["properties"]["assignee_username"] = request["properties"].pop("assignee")
    request["required"] = ["assignee_username"]


def _replace_write_role(spec):
    roles = schema(spec, "Member")["properties"]["role"]["enum"]
    roles[roles.index("write")] = "developer"


def _add_pending_state(spec):
    schema(spec, "Member")["properties"]["membership_state"]["enum"].append("pending")


def _rename_attempt(spec):
    model = schema(spec, "PipelineRun")
    model["properties"]["run_attempt"] = model["properties"].pop("attempt")
    model["required"][model["required"].index("attempt")] = "run_attempt"


def _rename_default_branch(spec):
    model = schema(spec, "Repository")
    model["properties"]["defaultBranch"] = model["properties"].pop("default_branch")
    model["required"][model["required"].index("default_branch")] = "defaultBranch"


def _add_drift_error(spec):
    schema(spec, "ErrorDetail")["properties"]["code"]["enum"].append("DRIFT_DETECTED")


def _add_invalid_ref(spec):
    schema(spec, "IssueSuccess")["properties"]["data"]["$ref"] = "#/components/schemas/Missing"


def _remove_path_parameter(spec):
    spec["paths"]["/repositories/{repo_id}"]["get"]["parameters"] = []


def _remove_state_effects(spec):
    spec["paths"]["/repositories/{repo_id}"]["patch"].pop("x-state-effects")


def _swap_unique_tool_ids(spec):
    get = spec["paths"]["/repositories/{repo_id}"]["get"]
    patch = spec["paths"]["/repositories/{repo_id}"]["patch"]
    get["x-canonical-tool-id"], patch["x-canonical-tool-id"] = (
        patch["x-canonical-tool-id"], get["x-canonical-tool-id"]
    )


@pytest.mark.parametrize(
    "mutator",
    [
        _delete_operation,
        _duplicate_tool_id,
        _require_priority,
        _rename_ref_to_branch,
        _rename_assignee,
        _replace_write_role,
        _add_pending_state,
        _rename_attempt,
        _rename_default_branch,
        _add_drift_error,
        _add_invalid_ref,
        _remove_path_parameter,
        _remove_state_effects,
        _swap_unique_tool_ids,
    ],
    ids=lambda mutator: mutator.__name__.removeprefix("_"),
)
def test_negative_mutations_are_rejected(spec, mutator):
    mutated = deepcopy(spec)
    mutator(mutated)
    with pytest.raises(ContractValidationError):
        validate_contract(mutated)
