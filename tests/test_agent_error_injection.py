import pytest

from conftest import injection_context
from driftguard.injection import AgentErrorHook
from driftguard.runners.injection_conformance_runner import CASE_ARGUMENTS, InjectionConformanceRunner
from driftguard.sandbox import SandboxService


@pytest.mark.parametrize("drift_id", ["ICD-01", "RSD-01", "WPD-01", "SED-01"])
def test_only_agent_fault_is_episode_scoped_while_aligned_new_runtime_remains(injection_catalog, drift_id):
    context = injection_context(injection_catalog, drift_id, "AE")
    assert context.contracts.contracts_equal
    assert context.contracts.displayed.source == "canonical_v1_plus_runtime_rule"
    assert AgentErrorHook().active(context)
    context.set_episode(4)
    assert not AgentErrorHook().active(context)
    assert context.injection_active(context.profile.target_tool)
    assert context.contracts.contracts_equal


def test_response_interpretation_fault_does_not_modify_runtime_or_raw_response(injection_catalog):
    context = injection_context(injection_catalog, "RSD-01", "AE")
    service = SandboxService(execution_context=context)
    result = service.call_tool("create_issue", CASE_ARGUMENTS["RSD-01"], "agent_admin")
    before = result.to_dict()
    record = AgentErrorHook().interpret(context, result)
    assert result.to_dict() == before
    assert record["interpretation_status"] == "failed"
    audit = service.call_log()[-1]
    assert "issue_id" in audit["canonical_response"]["payload"]["data"]
    assert "id" in audit["runtime_response"]["payload"]["data"]


def test_input_agent_fault_transforms_new_contract_call_to_old_shape(injection_catalog):
    context = injection_context(injection_catalog, "ICD-02", "AE")
    proposed = {"repo_id": "R1", "workflow_id": "ci", "branch": "main"}
    faulty = AgentErrorHook().inject_call(context, proposed)
    assert faulty == {"repo_id": "R1", "workflow_id": "ci", "ref": "main"}
    assert proposed["branch"] == "main"


def test_all_agent_errors_are_corrected_without_patch():
    report = InjectionConformanceRunner().run()
    assert report["summary"]["agent_error_executed"] == 20
    assert report["summary"]["agent_error_corrected"] == 20
