from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from driftguard.runtime.context import ExecutionContext
from driftguard.sandbox.diff import state_diff
from driftguard.sandbox.evaluator import TaskEvaluator
from driftguard.sandbox.service import SandboxService
from driftguard.sandbox.state_store import StateStore

from .models import ToolSpecPatch, ValidationResult


@dataclass(frozen=True)
class RepairRun:
    passed: bool
    task_evaluator_passed: bool
    forbidden_side_effects: int
    call_count: int
    parsed_data: dict[str, Any]
    initial_state: dict[str, Any]
    final_state: dict[str, Any]
    call_log: tuple[dict[str, Any], ...]
    failure: str | None = None

    def public_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "task_evaluator_passed": self.task_evaluator_passed,
            "forbidden_side_effects": self.forbidden_side_effects,
            "call_count": self.call_count,
            "state_change_count": len(state_diff(self.initial_state, self.final_state)),
            "failure": self.failure,
        }


class RepairExecutor:
    def __init__(self, max_calls: int = 8):
        self.max_calls = max_calls

    def execute(
        self,
        patch: ToolSpecPatch,
        context: ExecutionContext,
        arguments: dict[str, Any],
        initial_state: dict[str, Any] | None = None,
        episode: int = 5,
    ) -> RepairRun:
        context.set_episode(episode)
        context.reset_runtime()
        store = StateStore.from_state(initial_state) if initial_state is not None else StateStore.from_fixture()
        service = SandboxService(store=store, execution_context=context)
        before = service.store.snapshot()
        semantics = patch.semantic_extensions["x-driftguard-patch-semantics"]
        parsed: dict[str, Any] = {}
        failure: str | None = None
        try:
            parsed = self._execute_behavior(service, patch, semantics, deepcopy(arguments))
        except (KeyError, TypeError, ValueError) as exc:
            failure = f"{type(exc).__name__}: {exc}"
        after = service.store.snapshot()
        calls = service.call_log()
        behavior_ok = failure is None and bool(parsed.get("repair_ok"))
        task = {
            "task_id": "phase8-repair", "success_assertions": [],
            "answer_assertions": [{"key": "repair_ok", "operator": "eq", "value": True}],
            "forbidden_assertions": [{"allowed_paths": _allowed_paths(arguments)}],
            "max_tool_calls": self.max_calls,
        }
        evaluation = TaskEvaluator().evaluate(
            task, before, after, {"repair_ok": behavior_ok}, len(calls), len(calls),
            [] if behavior_ok else [failure or "patch-aware task behavior failed"],
        )
        forbidden = 0 if evaluation.forbidden_assertions_passed else 1
        return RepairRun(
            evaluation.task_success, evaluation.task_success, forbidden, len(calls), parsed,
            before, after, tuple(calls), failure,
        )

    def _execute_behavior(
        self, service: SandboxService, patch: ToolSpecPatch, semantics: dict[str, Any], arguments: dict[str, Any]
    ) -> dict[str, Any]:
        category, tool = patch.drift_category, patch.target_tool_id
        if category == "ICD":
            transform = semantics["request_transform"]
            if transform["kind"] == "rename":
                arguments[transform["to"]] = arguments.pop(transform["from"])
            elif transform["kind"] == "supply_required":
                arguments.setdefault(transform["field"], transform["value"])
            elif transform["kind"] == "enum_alias" and arguments.get(transform["field"]) == transform["from"]:
                arguments[transform["field"]] = transform["to"]
            result = service.call_tool(tool, arguments, "agent_admin")
            return {"repair_ok": result.ok, "data": result.payload.get("data", {})}
        if category == "RSD":
            result = service.call_tool(tool, arguments, "agent_admin")
            if not result.ok:
                return {"repair_ok": False}
            raw = result.payload["data"]
            parsed = deepcopy(raw)
            for canonical, runtime_path in semantics["response_mapping"].items():
                parsed[canonical] = _read_dotted(raw, runtime_path)
            expected = tuple(semantics["response_mapping"])
            return {"repair_ok": all(name in parsed for name in expected), "data": parsed}
        if category == "WPD":
            workflow = semantics["workflow"]
            kind = workflow["kind"]
            if kind == "active_assignee":
                issue = service.call_tool("get_issue", {"repo_id": arguments["repo_id"], "issue_id": arguments["issue_id"]}, "agent_admin")
                if not issue.ok:
                    return {"repair_ok": False}
                if not issue.payload["data"].get("assignee"):
                    assign = service.call_tool("assign_issue", {**arguments, "assignee": "bob"}, "agent_admin")
                    if not assign.ok:
                        return {"repair_ok": False}
            elif kind == "verification_binding":
                if workflow["read"] == "get_pipeline_status":
                    read_args = {"repo_id": arguments["repo_id"], "run_id": arguments["run_id"]}
                else:
                    read_args = {"repo_id": arguments["repo_id"], "username": arguments["assignee"]}
                read = service.call_tool(workflow["read"], read_args, "agent_admin")
                if not read.ok:
                    return {"repair_ok": False}
                arguments[workflow["binding"]] = read.payload["data"][workflow["binding"]]
            elif kind == "ordered_transition":
                intermediate = service.call_tool(tool, {**arguments, "role": workflow["intermediate"]}, "agent_admin")
                if not intermediate.ok:
                    return {"repair_ok": False}
            elif kind == "initial_then_promote":
                target_role = arguments.get("role", "read")
                added = service.call_tool(tool, {**arguments, "role": workflow["initial_role"]}, "agent_admin")
                if not added.ok:
                    return {"repair_ok": False}
                if target_role != workflow["initial_role"]:
                    promoted = service.call_tool(workflow["promotion_tool"], {**arguments, "role": target_role}, "agent_admin")
                    return {"repair_ok": promoted.ok, "data": promoted.payload.get("data", {})}
                return {"repair_ok": True, "data": added.payload.get("data", {})}
            result = service.call_tool(tool, arguments, "agent_admin")
            return {"repair_ok": result.ok, "data": result.payload.get("data", {})}
        policy = semantics["observation_policy"]
        result = service.call_tool(tool, arguments, "agent_admin")
        if not result.ok:
            return {"repair_ok": False}
        if policy["kind"] == "new_resource":
            read_args = {"repo_id": arguments["repo_id"], policy["identity_field"]: result.payload["data"][policy["identity_field"]]}
        else:
            read_args = _confirmation_arguments(policy["confirmation_tool"], arguments)
        confirmation = service.call_tool(policy["confirmation_tool"], read_args, "agent_admin")
        return {"repair_ok": confirmation.ok, "data": confirmation.payload.get("data", {})}


def _read_dotted(document: dict[str, Any], path: str) -> Any:
    current: Any = document
    for token in path.split("."):
        current = current[token]
    return current


def _confirmation_arguments(tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
    keys = {
        "get_issue": ("repo_id", "issue_id"),
        "get_member": ("repo_id", "username"),
        "get_repository": ("repo_id",),
    }[tool]
    return {key: arguments[key] for key in keys}


def _allowed_paths(arguments: dict[str, Any]) -> list[str]:
    repo_id = arguments.get("repo_id", "R1")
    return [f"/repositories/{repo_id}/**", f"/next_ids/{repo_id}/**"]


def repair_validation(run: RepairRun) -> ValidationResult:
    reasons = ("TASK_EVALUATOR_PASSED", "FORBIDDEN_SIDE_EFFECTS_ZERO") if run.passed else (run.failure or "REPAIR_FAILED",)
    return ValidationResult(run.passed, "immediate_repair", reasons, run.public_dict())
