from __future__ import annotations

from copy import deepcopy

import pytest

from driftguard.contracts.loader import load_openapi
from driftguard.live import LiveEvidenceBridge, LiveHealingSession, LiveHistoryStore, LiveScope


def _bridge(scenario: str = "public-live-a", provider: str = "mock") -> LiveEvidenceBridge:
    return LiveEvidenceBridge(LiveScope(scenario, provider, "driftguard_llm", 0), load_openapi())


def _completed(bridge: LiveEvidenceBridge, episode: int = 3, context: str = "ctx-1"):
    return bridge.emit(
        "tool_response", episode, source_event="test.controller.tool_response",
        execution_context_id=context, tool_id="create_issue",
        visible_runtime_response={"status_code": 422, "error": {"code": "VALIDATION_ERROR"}},
        normalized_observation={"channel": "response_error", "task_complete": False},
        visible_state_diff=[], before_state_hash="a", after_state_hash="a",
        probe_metadata={"independent_failure": True, "regression_requirements_identified": True},
    )


def test_controller_event_becomes_live_evidence_with_traceable_provenance():
    bridge = _bridge()
    event = _completed(bridge)
    assert event.correlation_id
    assert event.provenance["source"] == "controller_event"
    assert event.provenance["controller_event"] == "test.controller.tool_response"
    assert event.provenance["execution_context_id"] == "ctx-1"


def test_live_evidence_sequence_is_contiguous_and_episode_time_is_monotonic():
    bridge = _bridge()
    first = _completed(bridge, 3, "ctx-a")
    second = _completed(bridge, 4, "ctx-b")
    assert (first.sequence_number, second.sequence_number) == (1, 2)
    with pytest.raises(ValueError, match="backward"):
        _completed(bridge, 3, "ctx-c")


def test_scenario_provider_method_and_repetition_are_isolated():
    left, right = _bridge("public-left", "deepseek"), _bridge("public-right", "dashscope")
    _completed(left)
    assert len(left.trace.events) == 1 and not right.trace.events
    assert left.scope.key != right.scope.key and left.trace.trace_id != right.trace.trace_id


def test_agent_view_rejects_future_evidence_and_can_freeze_past_episode():
    bridge = _bridge()
    _completed(bridge, 3, "ctx-a")
    _completed(bridge, 4, "ctx-b")
    assert len(bridge.agent_view(3).trace.events) == 1
    with pytest.raises(ValueError, match="future"):
        bridge.agent_view(5)


@pytest.mark.parametrize("key", ["runtime_profile", "variant", "ground_truth", "source_drift_id", "expected_patch"])
def test_hidden_benchmark_material_cannot_enter_live_evidence(key: str):
    bridge = _bridge()
    with pytest.raises(ValueError, match="hidden"):
        bridge.emit("task_received", 1, source_event="test", **{key: "forbidden"})


def test_history_contains_only_completed_real_controller_events_from_past():
    bridge, history = _bridge(), LiveHistoryStore()
    event = _completed(bridge, 3)
    history.append_completed(bridge.scope, event)
    assert not history.before(bridge.scope, 3)
    assert history.before(bridge.scope, 4)[0].event_id == event.event_id
    proposed = bridge.emit("tool_call_proposed", 4, source_event="test", tool_id="create_issue")
    with pytest.raises(ValueError, match="completed"):
        history.append_completed(bridge.scope, proposed)


def test_history_never_crosses_scenario_or_provider_scope():
    left, right, history = _bridge("public-left"), _bridge("public-right"), LiveHistoryStore()
    history.append_completed(left.scope, _completed(left, 3))
    assert history.size(left.scope) == 1 and history.size(right.scope) == 0


def test_live_session_reset_clears_trace_history_and_patch_registry():
    session = LiveHealingSession(LiveScope("public-reset", "mock", "driftguard_llm", 0), load_openapi())
    event = _completed(session.bridge)
    session.history.append_completed(session.scope, event)
    session.reset()
    assert not session.bridge.trace.events
    assert session.history.size(session.scope) == 0
    assert len(session.patch_registry) == 0


def test_agent_view_contains_no_raw_canonical_response_alias():
    bridge = _bridge()
    _completed(bridge)
    encoded = str(bridge.agent_view().trace.to_dict()).lower()
    assert "canonical_response" not in encoded


def test_displayed_spec_is_copied_and_cannot_be_mutated_by_caller():
    spec = load_openapi()
    bridge = LiveEvidenceBridge(LiveScope("public-copy", "mock", "driftguard_llm", 0), spec)
    original = deepcopy(bridge.displayed_spec)
    spec["info"]["title"] = "mutated"
    assert bridge.displayed_spec == original
