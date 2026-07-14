from copy import deepcopy

import pytest

from driftguard.diagnosis.probe_executor import ProbeExecutor
from driftguard.diagnosis.probes import ProbeBudgetExceeded, ProbePlan, ProbePlanner, UnsafeProbeError
from driftguard.sandbox import SandboxService
from phase7_helpers import build_view


def plan(probe_type="read_after_write"):
    return ProbePlan("P1", probe_type, "get_repository", (), True, True)


def test_probe_planner_covers_channel_and_budget():
    plans = ProbePlanner(3).plan(build_view("SINGLE"))
    assert [item.probe_type for item in plans] == [
        "exact_retry", "local_schema_check", "independent_instance_reproduction"
    ]
    with pytest.raises(ProbeBudgetExceeded):
        ProbePlanner(2).plan(build_view("SINGLE"))


@pytest.mark.parametrize("probe_type", ["read_after_write", "repeated_read"])
def test_isolated_probe_does_not_pollute_main_state(probe_type):
    service = SandboxService()
    before = service.store.snapshot()
    result = ProbeExecutor().execute(
        plan(probe_type), service,
        lambda fork: fork.call_tool("update_repository", {"repo_id": "R1", "description": "fork"}, "agent_admin").ok,
    )
    assert result["state_unchanged"]
    assert service.store.snapshot() == before


def test_unsafe_probe_is_rejected_and_counted():
    executor = ProbeExecutor()
    with pytest.raises(UnsafeProbeError):
        executor.execute(plan("delete_repository"), SandboxService(), lambda fork: None)
    assert executor.unsafe_probe_count == 1


def test_probe_detects_direct_main_state_pollution():
    service = SandboxService()
    changed = deepcopy(service.store.snapshot())
    changed["repositories"]["R1"]["description"] = "polluted"
    with pytest.raises(RuntimeError, match="polluted"):
        ProbeExecutor().execute(plan(), service, lambda fork: service.store.commit(changed))
