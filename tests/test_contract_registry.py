from __future__ import annotations

import pytest

from driftguard.contracts import ContractRegistry
from driftguard.contracts.input_validator import InputValidationError, InputValidator


def test_registry_has_exact_twelve_tools():
    registry = ContractRegistry.from_openapi()
    assert len(registry.operation_ids()) == 12
    assert set(registry.operation_ids()) == {
        "get_repository", "update_repository", "create_issue", "get_issue",
        "assign_issue", "close_issue", "trigger_pipeline", "get_pipeline_status",
        "retry_pipeline", "add_member", "get_member", "update_member_role",
    }


def test_registry_reads_method_path_permission_and_effect():
    contract = ContractRegistry.from_openapi().require("assign_issue")
    assert contract.method == "put"
    assert contract.path == "/repositories/{repo_id}/issues/{issue_id}/assignee"
    assert contract.required_permission == "issues:write"
    assert contract.effect_type == "write"
    assert contract.success_status_code == 200


def test_flat_input_schema_merges_path_and_body():
    contract = ContractRegistry.from_openapi().require("trigger_pipeline")
    assert set(contract.input_schema["properties"]) == {"repo_id", "workflow_id", "ref", "inputs"}
    assert set(contract.input_schema["required"]) == {"repo_id", "workflow_id", "ref"}


def test_defaults_are_loaded_from_openapi():
    registry = ContractRegistry.from_openapi()
    validator = InputValidator()
    issue = validator.validate(registry.require("create_issue"), {"repo_id": "R1", "title": "x"})
    retry = validator.validate(registry.require("retry_pipeline"), {"repo_id": "R1", "run_id": 501})
    member = validator.validate(registry.require("add_member"), {"repo_id": "R1", "username": "erin"})
    assert issue["body"] == "" and issue["priority"] == "medium"
    assert retry["failed_only"] is True
    assert member["role"] == "read"


@pytest.mark.parametrize(
    "tool,arguments,field",
    [
        ("create_issue", {"repo_id": "R1"}, "title"),
        ("create_issue", {"repo_id": "R1", "title": "x", "priority": "urgent"}, "priority"),
        ("get_issue", {"repo_id": "R1", "issue_id": "101"}, "issue_id"),
        ("get_repository", {"repo_id": "R1", "extra": True}, "extra"),
    ],
)
def test_input_validation_rejects_invalid_arguments(tool, arguments, field):
    registry = ContractRegistry.from_openapi()
    with pytest.raises(InputValidationError) as exc:
        InputValidator().validate(registry.require(tool), arguments)
    assert exc.value.field == field


def test_update_repository_requires_mutable_field():
    registry = ContractRegistry.from_openapi()
    with pytest.raises(InputValidationError):
        InputValidator().validate(registry.require("update_repository"), {"repo_id": "R1"})
