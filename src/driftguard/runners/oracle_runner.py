from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from driftguard.contracts.loader import DEFAULT_TASKS_PATH
from driftguard.runners.bindings import BindingError, resolve_bindings
from driftguard.runners.conditions import evaluate_condition
from driftguard.sandbox.diff import state_diff
from driftguard.sandbox.evaluator import TaskEvaluator
from driftguard.sandbox.service import SandboxService


class OracleRunner:
    def __init__(
        self,
        service: SandboxService | None = None,
        tasks_path: Path | str = DEFAULT_TASKS_PATH,
    ):
        self.service = service or SandboxService()
        self.evaluator = TaskEvaluator()
        with Path(tasks_path).open(encoding="utf-8") as stream:
            document = json.load(stream)
        self.task_document = document
        self.tasks = document["tasks"]
        self._by_id = {task["task_id"]: task for task in self.tasks}

    def task(self, task_id: str) -> dict[str, Any]:
        try:
            return self._by_id[task_id]
        except KeyError as exc:
            raise ValueError(f"unknown task: {task_id}") from exc

    @staticmethod
    def _build_answer(task: dict[str, Any], successful_payloads: list[dict[str, Any]]) -> dict[str, Any]:
        answer: dict[str, Any] = {}
        for assertion in task["answer_assertions"]:
            key = assertion["key"]
            for payload in reversed(successful_payloads):
                data = payload.get("data", {})
                if key in data:
                    answer[key] = deepcopy(data[key])
                    break
        return answer

    def run_once(self, task: dict[str, Any]) -> dict[str, Any]:
        self.service.reset()
        initial_state = self.service.store.snapshot()
        completed_steps: dict[str, dict[str, Any]] = {}
        successful_payloads: list[dict[str, Any]] = []
        step_results: list[dict[str, Any]] = []
        execution_failures: list[str] = []
        actual_calls = 0

        for step in task["oracle_plan"]:
            step_id = step["step_id"]
            try:
                if "when" in step and not evaluate_condition(step["when"], completed_steps):
                    step_results.append({"step_id": step_id, "tool": step["tool"], "skipped": True})
                    continue
                arguments = resolve_bindings(step["arguments"], completed_steps)
            except (BindingError, ValueError, KeyError) as exc:
                execution_failures.append(f"{step_id}: {exc}")
                step_results.append({"step_id": step_id, "tool": step["tool"], "skipped": False, "error": str(exc)})
                break

            result = self.service.call_tool(step["tool"], arguments, task["actor_id"])
            actual_calls += 1
            record = {
                "step_id": step_id,
                "tool": step["tool"],
                "arguments": deepcopy(arguments),
                "skipped": False,
                "status_code": result.status_code,
                "payload": result.to_dict()["payload"],
            }
            step_results.append(record)
            if not result.ok:
                execution_failures.append(f"{step_id}: {result.payload['error']}")
                break
            completed_steps[step_id] = result.to_dict()["payload"]
            successful_payloads.append(result.to_dict()["payload"])

        final_state = self.service.store.snapshot()
        answer = self._build_answer(task, successful_payloads)
        evaluation = self.evaluator.evaluate(
            task=task,
            initial_state=initial_state,
            final_state=final_state,
            answer=answer,
            actual_tool_calls=actual_calls,
            oracle_tool_calls=sum(not item.get("skipped", False) for item in step_results),
            execution_failures=execution_failures,
        )
        result = {
            "task_id": task["task_id"],
            "task_success": evaluation.task_success,
            "actual_tool_calls": actual_calls,
            "oracle_steps": step_results,
            "skipped_steps": [item["step_id"] for item in step_results if item.get("skipped")],
            "answer": answer,
            "final_diff": state_diff(initial_state, final_state),
            "assertion_results": evaluation.to_dict(),
            "failure_details": list(evaluation.failures),
        }
        self.service.reset()
        return result

    def run_task(self, task_id: str, replay: int = 2) -> dict[str, Any]:
        if replay < 2:
            raise ValueError("replay must be at least 2")
        task = self.task(task_id)
        runs = [self.run_once(task) for _ in range(replay)]
        consistent = all(run == runs[0] for run in runs[1:])
        result = deepcopy(runs[0])
        result["replay_consistent"] = consistent
        if not consistent:
            result["task_success"] = False
            result["failure_details"].append("deterministic replay mismatch")
        return result

    def run_all(self, replay: int = 2, task_id: str | None = None) -> dict[str, Any]:
        selected = [self.task(task_id)] if task_id else self.tasks
        results = [self.run_task(task["task_id"], replay=replay) for task in selected]
        passed = sum(result["task_success"] for result in results)
        forbidden_failures = sum(
            not result["assertion_results"]["forbidden_assertions_passed"]
            for result in results
        )
        replay_passed = sum(result["replay_consistent"] for result in results)
        return {
            "run_version": "1.0",
            "fixture_id": self.task_document["fixture_id"],
            "canonical_contract_version": self.task_document["canonical_contract_version"],
            "summary": {
                "tasks": len(results),
                "passed": passed,
                "failed": len(results) - passed,
                "forbidden_side_effects": forbidden_failures,
                "deterministic_replay_passed": replay_passed,
            },
            "tasks": results,
        }
