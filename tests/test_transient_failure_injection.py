import pytest

from conftest import injection_context
from driftguard.runners.injection_conformance_runner import CASE_ARGUMENTS
from driftguard.sandbox import SandboxService


@pytest.mark.parametrize("drift_id", ["ICD-01", "RSD-02", "WPD-03", "SED-05"])
def test_transient_failure_recovers(injection_catalog, drift_id):
    context = injection_context(injection_catalog, drift_id, "TF")
    case = injection_catalog[0][drift_id]
    first_service = SandboxService(execution_context=context)
    first = first_service.call_tool(case["target_tool"], CASE_ARGUMENTS[drift_id], "agent_admin")
    context.set_episode(4)
    second_service = SandboxService(execution_context=context)
    second = second_service.call_tool(case["target_tool"], CASE_ARGUMENTS[drift_id], "agent_admin")
    assert first.to_dict() != second.to_dict() or first_service.store.snapshot() != second_service.store.snapshot()
    assert second.ok
    assert context.contracts.contracts_equal


def test_transient_consumes_after_one_hit_in_same_episode(injection_catalog):
    context = injection_context(injection_catalog, "ICD-01", "TF")
    first = SandboxService(execution_context=context).call_tool("create_issue", CASE_ARGUMENTS["ICD-01"], "agent_admin")
    second = SandboxService(execution_context=context).call_tool("create_issue", CASE_ARGUMENTS["ICD-01"], "agent_admin")
    assert not first.ok and second.ok
