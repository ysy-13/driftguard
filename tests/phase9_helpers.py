from __future__ import annotations

import json

from driftguard.agents import ToolAgentController
from driftguard.agents.policies import POLICIES
from driftguard.contracts.loader import PROJECT_ROOT
from driftguard.experiments import BudgetTracker, ExperimentBudget
from driftguard.llm import MockProvider, ModelConfig
from driftguard.runners.injection_conformance_runner import CASE_ARGUMENTS, InjectionConformanceRunner
from driftguard.runtime import ExecutionContext, ExecutionMode, ExecutionProfile
from driftguard.sandbox import SandboxService


ACTION_SCHEMA = json.loads((PROJECT_ROOT / "benchmark/schemas/agent_action_schema_v1.json").read_text())
BASE_PROMPT = (PROJECT_ROOT / "benchmark/prompts/base_tool_agent_v1.txt").read_text()


def context_for(drift_id: str, mode: str):
    runner = InjectionConformanceRunner()
    family = next(item for item in runner.families if item["source_drift_id"] == drift_id)
    case = runner._drift_by_id[drift_id]
    return ExecutionContext(ExecutionProfile(ExecutionMode(mode), case, family, change_point=int(case["change_point"])))


def smoke_task(repo_id="R1"):
    return {
        "task_id": "test", "instruction": "Complete the visible tool task.", "actor_id": "agent_admin",
        "success_assertions": [],
        "answer_assertions": [{"key": "tool_succeeded", "operator": "eq", "value": True}],
        "forbidden_assertions": [{"allowed_paths": [f"/repositories/{repo_id}/**", f"/next_ids/{repo_id}/**"]}],
        "max_tool_calls": 8,
    }


def controller(outputs, policy="standard", context=None, budget=None, cache=None):
    service = SandboxService(execution_context=context)
    model = ModelConfig("mock", "mock-deterministic")
    tracker = BudgetTracker(budget or ExperimentBudget())
    return ToolAgentController(
        MockProvider(outputs), model, POLICIES[policy](), service, tracker,
        ACTION_SCHEMA, BASE_PROMPT, cache=cache,
    ), service

