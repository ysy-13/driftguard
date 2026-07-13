import pytest

from conftest import injection_context
from driftguard.runners.injection_conformance_runner import CASE_ARGUMENTS
from driftguard.sandbox import SandboxService


@pytest.mark.parametrize("drift_id", ["ICD-01", "ICD-02", "ICD-03", "ICD-04", "ICD-05"])
def test_all_input_contract_mutations_execute(injection_catalog, drift_id):
    context = injection_context(injection_catalog, drift_id, "PD")
    case = injection_catalog[0][drift_id]
    result = SandboxService(execution_context=context).call_tool(
        case["target_tool"], CASE_ARGUMENTS[drift_id], "agent_admin"
    )
    signature = injection_catalog[1][drift_id]["shared_observation_signature"]
    assert result.status_code == signature["http_status"]
    assert result.payload["error"]["field"] == signature["field"]


@pytest.mark.parametrize(
    "drift_id,arguments",
    [
        ("ICD-01", {"repo_id": "R1", "title": "explicit", "priority": "high"}),
        ("ICD-02", {"repo_id": "R1", "workflow_id": "ci", "branch": "main"}),
        ("ICD-03", {"repo_id": "R1", "issue_id": 101, "assignee_username": "bob"}),
        ("ICD-04", {"repo_id": "R1", "run_id": 501, "failed_only": True}),
        ("ICD-05", {"repo_id": "R1", "username": "carol", "role": "developer"}),
    ],
)
def test_runtime_conformant_input_is_translated_to_canonical_handler(injection_catalog, drift_id, arguments):
    context = injection_context(injection_catalog, drift_id, "PD")
    case = injection_catalog[0][drift_id]
    result = SandboxService(execution_context=context).call_tool(case["target_tool"], arguments, "agent_admin")
    assert result.ok
