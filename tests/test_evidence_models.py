from dataclasses import FrozenInstanceError

import pytest

from driftguard.evidence import EvidenceCollector, EvidenceEvent, EvidenceStore, HistoryRecord, HistoryStore
from driftguard.evidence.models import EvidenceTrace


def event(sequence=1, episode=1, public_id="public-safe"):
    return EvidenceEvent(
        f"E{sequence}", "T", public_id, episode, sequence, "task_received",
        probe_metadata={"safe": True},
    )


def test_evidence_event_is_deeply_immutable():
    value = event()
    with pytest.raises(FrozenInstanceError):
        value.sequence_number = 2
    with pytest.raises(TypeError):
        value.probe_metadata["safe"] = False


def test_trace_requires_contiguous_order_and_no_episode_reversal():
    trace = EvidenceTrace("T", "public-safe").append(event())
    with pytest.raises(ValueError):
        trace.append(event(3, 1))
    with pytest.raises(ValueError):
        EvidenceTrace("T", "public-safe", (event(1, 2), event(2, 1)))


def test_store_reset_removes_all_evidence():
    store = EvidenceStore("T", "public-safe")
    store.append(event())
    store.reset()
    assert store.trace.events == ()


def test_history_is_time_and_scenario_isolated():
    history = HistoryStore()
    history.append(HistoryRecord("H1", "public-a", "get_repository", {}, {}, [], True, 2, "f", "t"))
    history.append(HistoryRecord("H2", "public-b", "get_repository", {}, {}, [], True, 1, "f", "t"))
    assert history.before("public-a", 2) == ()
    assert [record.record_id for record in history.before("public-a", 3)] == ["H1"]
    assert [record.record_id for record in history.before("public-b", 3)] == ["H2"]
