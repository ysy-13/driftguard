from __future__ import annotations

from copy import deepcopy

from driftguard.sandbox.evaluator import TaskEvaluator


def base_task():
    return {
        "task_id": "X01",
        "success_assertions": [{"target": "state", "path": "/value", "operator": "eq", "value": 2}],
        "answer_assertions": [{"target": "answer", "key": "nullable", "operator": "is_null"}],
        "forbidden_assertions": [{"operator": "unchanged_outside", "allowed_paths": ["/value"]}],
        "max_tool_calls": 1,
    }


def test_evaluator_success_and_null_answer():
    result = TaskEvaluator().evaluate(base_task(), {"value": 1}, {"value": 2}, {"nullable": None}, 1, 1)
    assert result.task_success


def test_missing_answer_fails():
    result = TaskEvaluator().evaluate(base_task(), {"value": 1}, {"value": 2}, {}, 1, 1)
    assert not result.answer_assertions_passed


def test_forbidden_side_effect_fails_and_wildcard_allows():
    task = base_task()
    final = {"value": 2, "other": True}
    result = TaskEvaluator().evaluate(task, {"value": 1}, final, {"nullable": None}, 1, 1)
    assert not result.forbidden_assertions_passed
    task["forbidden_assertions"][0]["allowed_paths"].append("/other/**")
    result = TaskEvaluator().evaluate(task, {"value": 1}, final, {"nullable": None}, 1, 1)
    assert result.forbidden_assertions_passed


def test_tool_budget_failure():
    result = TaskEvaluator().evaluate(base_task(), {"value": 1}, {"value": 2}, {"nullable": None}, 2, 1)
    assert not result.within_tool_budget
