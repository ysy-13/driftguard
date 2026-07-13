import pytest

from conftest import injection_context
from driftguard.runners.injection_conformance_runner import CASE_ARGUMENTS
from driftguard.sandbox import SandboxService


@pytest.mark.parametrize("drift_id", ["RSD-01", "RSD-02", "RSD-03", "RSD-04", "RSD-05"])
def test_all_response_shape_mutations_remove_canonical_field(injection_catalog, drift_id):
    context = injection_context(injection_catalog, drift_id, "PD")
    case = injection_catalog[0][drift_id]
    result = SandboxService(execution_context=context).call_tool(
        case["target_tool"], CASE_ARGUMENTS[drift_id], "agent_admin"
    )
    missing = injection_catalog[1][drift_id]["shared_observation_signature"]["field"]
    assert result.ok and missing not in result.payload["data"]
    audit = SandboxService(execution_context=injection_context(injection_catalog, drift_id, "PD"))
    before = audit.store.snapshot()
    audit.call_tool(case["target_tool"], CASE_ARGUMENTS[drift_id], "agent_admin")
    record = audit.call_log()[-1]
    assert record["canonical_response"] != record["runtime_response"]
    if case["drift_type"] == "response_shape" and case["target_tool"] in {"get_pipeline_status", "get_member", "get_repository"}:
        assert audit.store.snapshot() == before
