import json

import pytest

from driftguard.experiments import BudgetExhausted, BudgetTracker, ExperimentBudget
from driftguard.experiments.checkpoint import CheckpointStore
from driftguard.experiments.metrics import aggregate_metrics, attribution_metrics, confusion_matrix
from driftguard.healing import PatchRegistry
from driftguard.evidence import HistoryStore


def test_tool_probe_format_and_token_budgets_are_enforced():
    tracker = BudgetTracker(ExperimentBudget(max_tool_calls=1, max_probe_calls=0, max_format_repairs=0, max_input_tokens=2))
    tracker.consume_tool()
    with pytest.raises(BudgetExhausted):
        tracker.consume_tool()
    with pytest.raises(BudgetExhausted):
        BudgetTracker(ExperimentBudget(max_probe_calls=0)).consume_tool(probe=True)
    with pytest.raises(BudgetExhausted):
        tracker.consume_format_repair()
    with pytest.raises(BudgetExhausted):
        BudgetTracker(ExperimentBudget(max_input_tokens=1)).consume_llm(2, 0)


def test_atomic_checkpoint_and_resume_read(tmp_path):
    store = CheckpointStore(tmp_path)
    store.write("abc", {"value": 1})
    assert store.has("abc") and store.read("abc") == {"value": 1}
    assert not list((tmp_path / "records").glob("*.tmp"))


def test_scenario_and_repetition_state_containers_are_isolated():
    first_registry, second_registry = PatchRegistry(), PatchRegistry()
    first_history, second_history = HistoryStore(), HistoryStore()
    assert first_registry is not second_registry and first_history is not second_history
    assert len(first_registry) == len(second_registry) == 0


def test_confusion_matrix_and_macro_metrics():
    rows = [
        {"_expected": "AE", "predicted_class": "AE"},
        {"_expected": "TF", "predicted_class": "PD"},
        {"_expected": "PD", "predicted_class": "PD"},
    ]
    matrix = confusion_matrix(rows)
    assert matrix["AE"]["AE"] == 1 and matrix["TF"]["PD"] == 1
    metrics = attribution_metrics(rows)
    assert metrics["accuracy"] == 2 / 3 and 0 <= metrics["macro_f1"] <= 1


def test_efficiency_token_and_budget_metrics():
    row = {
        "final_task_success": True, "immediate_repair": True, "future_transfer": False,
        "forbidden_side_effects": 0, "unsafe_action": False, "termination_reason": "TASK_SUCCESS",
        "patch_proposed": True, "patch_accepted": True, "false_patch": False,
        "llm_calls": 2, "tool_calls": 3, "probe_calls": 1, "input_tokens": 10,
        "output_tokens": 5, "latency_ms": 12, "_expected": "PD", "predicted_class": "PD",
    }
    metrics = aggregate_metrics([row])
    assert metrics["average_input_tokens"] == 10 and metrics["average_output_tokens"] == 5
    assert metrics["successful_task_average_interactions"] == 5

