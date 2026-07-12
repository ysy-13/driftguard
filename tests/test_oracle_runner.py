from __future__ import annotations

import pytest

from driftguard.runners.bindings import BindingError, resolve_binding, resolve_bindings
from driftguard.runners.conditions import evaluate_condition
from driftguard.runners.oracle_runner import OracleRunner


def test_binding_resolves_nested_output():
    steps = {"s1": {"data": {"issue": {"id": 105}}}}
    assert resolve_binding("$steps.s1.data.issue.id", steps) == 105
    assert resolve_bindings({"issue_id": "$steps.s1.data.issue.id"}, steps) == {"issue_id": 105}


def test_binding_rejects_missing_step_and_field():
    with pytest.raises(BindingError):
        resolve_binding("$steps.s2.data.id", {})
    with pytest.raises(BindingError):
        resolve_binding("$steps.s1.data.missing", {"s1": {"data": {}}})


@pytest.mark.parametrize(
    "condition,expected",
    [
        ({"source": "$steps.s1.data.value", "operator": "eq", "value": 1}, True),
        ({"source": "$steps.s1.data.value", "operator": "neq", "value": 2}, True),
        ({"source": "$steps.s1.data.nullable", "operator": "is_null"}, True),
        ({"source": "$steps.s1.data.value", "operator": "not_null"}, True),
        ({"source": "$steps.s1.data.role", "operator": "role_below", "value": "write"}, True),
        ({"source": "$steps.s1.data.role", "operator": "role_at_least", "value": "triage"}, True),
    ],
)
def test_condition_operators(condition, expected):
    steps = {"s1": {"data": {"value": 1, "nullable": None, "role": "triage"}}}
    assert evaluate_condition(condition, steps) is expected


def test_all_condition_and_false_condition():
    steps = {"s1": {"data": {"a": True, "b": "open"}}}
    condition = {
        "operator": "all",
        "conditions": [
            {"source": "$steps.s1.data.a", "operator": "eq", "value": True},
            {"source": "$steps.s1.data.b", "operator": "eq", "value": "open"},
        ],
    }
    assert evaluate_condition(condition, steps)
    condition["conditions"][1]["value"] = "closed"
    assert not evaluate_condition(condition, steps)


def test_oracle_has_exact_tasks_and_all_pass():
    results = OracleRunner().run_all(replay=2)
    assert results["summary"] == {
        "tasks": 32,
        "passed": 32,
        "failed": 0,
        "forbidden_side_effects": 0,
        "deterministic_replay_passed": 32,
    }
    assert all(task["assertion_results"]["within_tool_budget"] for task in results["tasks"])
    assert all(task["assertion_results"]["answer_assertions_passed"] for task in results["tasks"])


def test_false_condition_is_skipped_without_tool_call():
    runner = OracleRunner()
    task = {
        "task_id": "X01", "actor_id": "agent_admin", "max_tool_calls": 2,
        "oracle_plan": [
            {"step_id": "s1", "tool": "get_issue", "arguments": {"repo_id": "R1", "issue_id": 102}},
            {"step_id": "s2", "when": {"source": "$steps.s1.data.state", "operator": "eq", "value": "closed"}, "tool": "close_issue", "arguments": {"repo_id": "R1", "issue_id": 102}},
        ],
        "success_assertions": [], "answer_assertions": [],
        "forbidden_assertions": [{"operator": "unchanged_outside", "allowed_paths": []}],
    }
    result = runner.run_once(task)
    assert result["actual_tool_calls"] == 1
    assert result["skipped_steps"] == ["s2"]


def test_replay_ids_timestamps_diff_and_results_match():
    runner = OracleRunner()
    first = runner.run_once(runner.task("F03"))
    second = runner.run_once(runner.task("F03"))
    assert first == second
    assert first["final_diff"]
    assert first["oracle_steps"][0]["payload"]["data"]["issue_id"] == 105
    assert first["oracle_steps"][0]["payload"]["data"]["created_at"] == "2026-01-01T00:00:01Z"
