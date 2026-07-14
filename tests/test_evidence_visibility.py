import pytest

from driftguard.contracts.loader import load_openapi
from driftguard.evidence import AgentView, EvidenceEvent, EvidenceStore, EvaluatorView
from driftguard.evidence.leakage_guard import EvidenceLeakageError, assert_agent_visible
from driftguard.evidence.models import EvidenceTrace


@pytest.mark.parametrize("payload", [
    {"ground_truth_label": "x"}, {"evaluator_metadata": {}},
    {"runtime_profile": {}}, {"source_drift_id": "x"},
])
def test_leakage_guard_rejects_hidden_fields(payload):
    with pytest.raises(EvidenceLeakageError):
        assert_agent_visible(payload)


@pytest.mark.parametrize("public_id", ["public-M01-AE", "x_TF", "PD-case"])
def test_public_id_cannot_reveal_variant(public_id):
    with pytest.raises(EvidenceLeakageError):
        EvidenceTrace("T", public_id)


def test_agent_view_rejects_hidden_runtime_contract():
    with pytest.raises(EvidenceLeakageError):
        AgentView(EvidenceTrace("T", "public-safe"), {"runtime_contract": {}}, 1)


def test_leakage_guard_rejects_injected_drift_marker():
    with pytest.raises(EvidenceLeakageError):
        assert_agent_visible({"runtime_response": {"message": "DRIFT_DETECTED"}})


def test_evaluator_metadata_is_not_part_of_agent_view():
    agent = AgentView(EvidenceTrace("T", "public-safe"), load_openapi(), 1)
    evaluator = EvaluatorView(agent, {"ground_truth_label": "persistent_drift"})
    assert not hasattr(agent, "metadata")
    assert evaluator.metadata["ground_truth_label"] == "persistent_drift"


def test_agent_view_cannot_read_future_episode():
    event = EvidenceEvent("E1", "T", "public-safe", 5, 1, "task_received")
    with pytest.raises(ValueError, match="future"):
        AgentView(EvidenceTrace("T", "public-safe", (event,)), load_openapi(), 4)
