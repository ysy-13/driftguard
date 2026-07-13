import pytest

from conftest import injection_context
from driftguard.runners.injection_conformance_runner import CASE_ARGUMENTS
from driftguard.sandbox import SandboxService


@pytest.mark.parametrize("drift_id", ["ICD-02", "RSD-03", "WPD-04", "SED-04"])
def test_persistent_drift_survives_independent_call(injection_catalog, drift_id):
    context = injection_context(injection_catalog, drift_id, "PD")
    case = injection_catalog[0][drift_id]
    first = SandboxService(execution_context=context).call_tool(case["target_tool"], CASE_ARGUMENTS[drift_id], "agent_admin")
    context.set_episode(4)
    second = SandboxService(execution_context=context).call_tool(case["target_tool"], CASE_ARGUMENTS[drift_id], "agent_admin")
    assert first.to_dict() == second.to_dict()
    assert not context.contracts.contracts_equal
