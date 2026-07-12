from __future__ import annotations

from driftguard.sandbox.deterministic_clock import DeterministicClock
from driftguard.sandbox.state_store import StateStore


def test_fixture_load_and_snapshot():
    store = StateStore.from_fixture()
    snapshot = store.snapshot()
    assert snapshot["fixture_id"] == "S0"
    assert set(snapshot["repositories"]) == {"R1", "R2", "R3"}


def test_snapshot_is_deep_copy():
    store = StateStore.from_fixture()
    snapshot = store.snapshot()
    snapshot["repositories"]["R1"]["description"] = "tampered"
    assert store.snapshot()["repositories"]["R1"]["description"] == "Legacy API service"


def test_reset_restores_initial_state():
    store = StateStore.from_fixture()
    state = store.snapshot()
    state["repositories"]["R1"]["description"] = "changed"
    store.commit(state)
    store.reset()
    assert store.snapshot()["repositories"]["R1"]["description"] == "Legacy API service"


def test_next_ids_are_deterministic_and_resettable():
    store = StateStore.from_fixture()
    assert store.allocate_issue_id("R1") == 105
    assert store.allocate_run_id("R1") == 504
    store.reset()
    assert store.allocate_issue_id("R1") == 105
    assert store.allocate_run_id("R1") == 504


def test_clock_advances_only_explicitly_and_resets():
    clock = DeterministicClock("2026-01-01T00:00:00Z")
    assert clock.now() == "2026-01-01T00:00:00Z"
    assert clock.peek_next() == "2026-01-01T00:00:01Z"
    assert clock.now() == "2026-01-01T00:00:00Z"
    assert clock.advance_write() == "2026-01-01T00:00:01Z"
    clock.reset()
    assert clock.now() == "2026-01-01T00:00:00Z"


def test_service_reset_isolates_tasks(service):
    service.call_tool("update_repository", {"repo_id": "R1", "description": "changed"}, "agent_admin")
    service.reset()
    assert service.store.snapshot()["repositories"]["R1"]["description"] == "Legacy API service"
    assert service.clock.now() == "2026-01-01T00:00:00Z"
