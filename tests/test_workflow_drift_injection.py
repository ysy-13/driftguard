import pytest

from conftest import injection_context
from driftguard.runners.injection_conformance_runner import CASE_ARGUMENTS
from driftguard.sandbox import SandboxService


@pytest.mark.parametrize("drift_id", ["WPD-01", "WPD-02", "WPD-03", "WPD-04", "WPD-05"])
def test_all_workflow_preconditions_execute(injection_catalog, drift_id):
    context = injection_context(injection_catalog, drift_id, "PD")
    case = injection_catalog[0][drift_id]
    result = SandboxService(execution_context=context).call_tool(
        case["target_tool"], CASE_ARGUMENTS[drift_id], "agent_admin"
    )
    assert result.status_code == 409
    assert result.payload["error"]["code"] == "PRECONDITION_FAILED"


@pytest.mark.parametrize(
    "drift_id,arguments",
    [
        ("WPD-01", {"repo_id": "R1", "issue_id": 102}),
        ("WPD-04", {"repo_id": "R1", "username": "dave", "role": "triage"}),
        ("WPD-05", {"repo_id": "R1", "username": "erin", "role": "read"}),
    ],
)
def test_satisfied_runtime_precondition_reaches_canonical_handler(injection_catalog, drift_id, arguments):
    context = injection_context(injection_catalog, drift_id, "PD")
    case = injection_catalog[0][drift_id]
    result = SandboxService(execution_context=context).call_tool(case["target_tool"], arguments, "agent_admin")
    assert result.ok


def test_pipeline_verification_token_is_deterministic_bound_and_consumed(injection_catalog):
    context = injection_context(injection_catalog, "WPD-02", "PD")
    service = SandboxService(execution_context=context)
    read = service.call_tool("get_pipeline_status", {"repo_id": "R1", "run_id": 501}, "agent_admin")
    token = read.payload["data"]["verification_token"]
    assert token == service.call_tool("get_pipeline_status", {"repo_id": "R1", "run_id": 501}, "agent_admin").payload["data"]["verification_token"]
    wrong = service.call_tool("retry_pipeline", {"repo_id": "R2", "run_id": 601, "verification_token": token}, "agent_admin")
    assert not wrong.ok
    assert service.call_tool("retry_pipeline", {"repo_id": "R1", "run_id": 501, "verification_token": token}, "agent_admin").ok
    assert not service.call_tool("retry_pipeline", {"repo_id": "R1", "run_id": 501, "verification_token": token}, "agent_admin").ok


def test_membership_token_sequence_and_cross_scenario_rejection(injection_catalog):
    first = injection_context(injection_catalog, "WPD-03", "PD")
    service = SandboxService(execution_context=first)
    read = service.call_tool("get_member", {"repo_id": "R1", "username": "bob"}, "agent_admin")
    token = read.payload["data"]["membership_verification_token"]
    second = injection_context(injection_catalog, "WPD-03", "PD")
    rejected = SandboxService(execution_context=second).call_tool(
        "assign_issue", {"repo_id": "R1", "issue_id": 101, "assignee": "bob", "membership_verification_token": token}, "agent_admin"
    )
    assert not rejected.ok
    accepted = service.call_tool(
        "assign_issue", {"repo_id": "R1", "issue_id": 101, "assignee": "bob", "membership_verification_token": token}, "agent_admin"
    )
    assert accepted.ok
