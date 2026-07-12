from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from driftguard.sandbox.diff import path_is_allowed, state_diff


MISSING = object()
_NO_DEFAULT = object()


def resolve_pointer(document: Any, pointer: str, default: Any = _NO_DEFAULT) -> Any:
    if pointer == "":
        return document
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise ValueError(f"invalid JSON Pointer: {pointer!r}")
    current = document
    for raw_token in pointer[1:].split("/"):
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict) and token in current:
            current = current[token]
        elif isinstance(current, list) and token.isdigit() and int(token) < len(current):
            current = current[int(token)]
        elif default is not _NO_DEFAULT:
            return default
        else:
            raise KeyError(pointer)
    return current


def _assert_value(actual: Any, exists: bool, assertion: dict[str, Any]) -> bool:
    operator = assertion["operator"]
    if operator == "eq":
        return exists and actual == assertion.get("value")
    if operator == "exists":
        return exists
    if operator == "not_exists":
        return not exists
    if operator == "is_null":
        return exists and actual is None
    if operator == "not_null":
        return exists and actual is not None
    raise ValueError(f"unsupported assertion operator: {operator}")


@dataclass(frozen=True)
class EvaluationResult:
    task_id: str
    task_success: bool
    state_assertions_passed: bool
    answer_assertions_passed: bool
    forbidden_assertions_passed: bool
    within_tool_budget: bool
    actual_tool_calls: int
    oracle_tool_calls: int
    failures: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_success": self.task_success,
            "state_assertions_passed": self.state_assertions_passed,
            "answer_assertions_passed": self.answer_assertions_passed,
            "forbidden_assertions_passed": self.forbidden_assertions_passed,
            "within_tool_budget": self.within_tool_budget,
            "actual_tool_calls": self.actual_tool_calls,
            "oracle_tool_calls": self.oracle_tool_calls,
            "failures": list(self.failures),
        }


class TaskEvaluator:
    def evaluate(
        self,
        task: dict[str, Any],
        initial_state: dict[str, Any],
        final_state: dict[str, Any],
        answer: dict[str, Any],
        actual_tool_calls: int,
        oracle_tool_calls: int,
        execution_failures: list[str] | None = None,
    ) -> EvaluationResult:
        failures = list(execution_failures or [])

        state_ok = True
        for assertion in task["success_assertions"]:
            value = resolve_pointer(final_state, assertion["path"], MISSING)
            passed = _assert_value(value, value is not MISSING, assertion)
            if not passed:
                state_ok = False
                failures.append(f"state assertion failed: {assertion}")

        answer_ok = True
        for assertion in task["answer_assertions"]:
            key = assertion["key"]
            value = answer.get(key, MISSING)
            passed = _assert_value(value, value is not MISSING, assertion)
            if not passed:
                answer_ok = False
                failures.append(f"answer assertion failed: {assertion}")

        changes = state_diff(initial_state, final_state)
        forbidden_ok = True
        for assertion in task["forbidden_assertions"]:
            allowed = assertion["allowed_paths"]
            disallowed = [
                change["path"]
                for change in changes
                if not any(path_is_allowed(change["path"], pattern) for pattern in allowed)
            ]
            if disallowed:
                forbidden_ok = False
                failures.append(f"forbidden side effects: {disallowed}")

        within_budget = actual_tool_calls <= task["max_tool_calls"]
        if not within_budget:
            failures.append(
                f"tool budget exceeded: {actual_tool_calls} > {task['max_tool_calls']}"
            )
        task_success = not failures and state_ok and answer_ok and forbidden_ok and within_budget
        return EvaluationResult(
            task_id=task["task_id"],
            task_success=task_success,
            state_assertions_passed=state_ok,
            answer_assertions_passed=answer_ok,
            forbidden_assertions_passed=forbidden_ok,
            within_tool_budget=within_budget,
            actual_tool_calls=actual_tool_calls,
            oracle_tool_calls=oracle_tool_calls,
            failures=tuple(failures),
        )
