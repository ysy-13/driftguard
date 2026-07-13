import pytest

from conftest import injection_context
from driftguard.runners.injection_conformance_runner import CASE_ARGUMENTS, InjectionConformanceRunner
from driftguard.sandbox import SandboxService


@pytest.mark.parametrize("drift_id", ["SED-01", "SED-02", "SED-03", "SED-04", "SED-05"])
def test_all_state_effect_mutations_execute(injection_catalog, drift_id):
    context = injection_context(injection_catalog, drift_id, "PD")
    case = injection_catalog[0][drift_id]
    service = SandboxService(execution_context=context)
    before = service.store.snapshot()
    result = service.call_tool(case["target_tool"], CASE_ARGUMENTS[drift_id], "agent_admin")
    assert result.ok
    assert InjectionConformanceRunner._state_effect_matches(drift_id, before, service.store.snapshot())


def test_deferred_effect_materializes_on_read(injection_catalog):
    context = injection_context(injection_catalog, "SED-01", "PD")
    service = SandboxService(execution_context=context)
    service.call_tool("close_issue", CASE_ARGUMENTS["SED-01"], "agent_admin")
    result = service.call_tool("get_issue", {"repo_id": "R1", "issue_id": 101}, "agent_admin")
    assert result.payload["data"]["state"] == "closed"


def test_pending_membership_stays_out_of_business_state_until_get(injection_catalog):
    context = injection_context(injection_catalog, "SED-03", "PD")
    service = SandboxService(execution_context=context)
    result = service.call_tool("add_member", CASE_ARGUMENTS["SED-03"], "agent_admin")
    assert result.payload["data"]["membership_state"] == "active"
    assert "erin" not in service.store.snapshot()["repositories"]["R1"]["members"]
    premature = service.call_tool("assign_issue", {"repo_id": "R1", "issue_id": 101, "assignee": "erin"}, "agent_admin")
    assert not premature.ok
    assert service.call_tool("get_member", {"repo_id": "R1", "username": "erin"}, "agent_admin").ok
    assert service.call_tool("assign_issue", {"repo_id": "R1", "issue_id": 101, "assignee": "erin"}, "agent_admin").ok


def test_sed04_adds_new_identity_without_modifying_original(injection_catalog):
    context = injection_context(injection_catalog, "SED-04", "PD")
    service = SandboxService(execution_context=context)
    before = service.store.snapshot()
    result = service.call_tool("retry_pipeline", CASE_ARGUMENTS["SED-04"], "agent_admin")
    after = service.store.snapshot()
    assert result.payload["data"]["run_id"] == before["next_ids"]["R1"]["run_id"]
    assert after["repositories"]["R1"]["pipeline_runs"]["501"] == before["repositories"]["R1"]["pipeline_runs"]["501"]
    diff = service.call_log()[-1]["state_diff"]
    assert any(item["op"] == "add" and "/pipeline_runs/504" in item["path"] for item in diff)
    assert not any(item["path"].startswith("/repositories/R1/pipeline_runs/501/") for item in diff)


def test_sed05_only_delays_default_branch(injection_catalog):
    context = injection_context(injection_catalog, "SED-05", "PD")
    service = SandboxService(execution_context=context)
    result = service.call_tool(
        "update_repository", {"repo_id": "R1", "default_branch": "develop", "description": "immediate"}, "agent_admin"
    )
    state = service.store.snapshot()["repositories"]["R1"]
    assert result.payload["data"]["default_branch"] == "develop"
    assert state["default_branch"] == "main"
    assert state["description"] == "immediate"


def test_reset_clears_pending_effect(injection_catalog):
    context = injection_context(injection_catalog, "SED-01", "PD")
    service = SandboxService(execution_context=context)
    service.call_tool("close_issue", CASE_ARGUMENTS["SED-01"], "agent_admin")
    assert context.pending_effects._items
    context.reset()
    assert not context.pending_effects._items
