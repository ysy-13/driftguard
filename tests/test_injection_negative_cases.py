from copy import deepcopy

from conftest import injection_context
from driftguard.contracts.loader import load_openapi
from driftguard.runners.injection_conformance_runner import CASE_ARGUMENTS
from driftguard.sandbox import SandboxService


def test_non_target_tool_and_normal_context_are_unaffected(injection_catalog):
    context = injection_context(injection_catalog, "ICD-01", "PD")
    service = SandboxService(execution_context=context)
    assert service.call_tool("get_repository", {"repo_id": "R1"}, "agent_admin").ok
    assert SandboxService().call_tool("create_issue", CASE_ARGUMENTS["ICD-01"], "agent_admin").ok


def test_transient_is_atomic_consumed_and_resettable(injection_catalog):
    context = injection_context(injection_catalog, "ICD-01", "TF")
    service = SandboxService(execution_context=context)
    state, clock = service.store.snapshot(), service.clock.now()
    assert not service.call_tool("create_issue", CASE_ARGUMENTS["ICD-01"], "agent_admin").ok
    assert service.store.snapshot() == state and service.clock.now() == clock
    assert context.profile.scenario_id in context.session.consumed_transient
    context.reset()
    assert context.profile.scenario_id not in context.session.consumed_transient
    assert not SandboxService(execution_context=context).call_tool("create_issue", CASE_ARGUMENTS["ICD-01"], "agent_admin").ok


def test_runtime_only_input_field_is_removed_before_canonical_audit(injection_catalog):
    context = injection_context(injection_catalog, "ICD-02", "PD")
    service = SandboxService(execution_context=context)
    assert service.call_tool("trigger_pipeline", {"repo_id": "R1", "workflow_id": "ci", "branch": "main"}, "agent_admin").ok
    arguments = service.call_log()[-1]["arguments"]
    assert "ref" in arguments and "branch" not in arguments


def test_snapshot_mutation_cannot_modify_canonical(injection_catalog):
    canonical = load_openapi()
    context = injection_context(injection_catalog, "RSD-01", "PD")
    copy = context.runtime_contract
    copy["info"]["title"] = "bad mutation"
    assert load_openapi() == canonical
    assert context.runtime_contract["info"]["title"] != "bad mutation"
