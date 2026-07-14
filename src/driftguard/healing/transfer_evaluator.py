from __future__ import annotations

from copy import deepcopy
from typing import Callable

from driftguard.runtime.context import ExecutionContext
from driftguard.sandbox.service import SandboxService
from driftguard.sandbox.state_store import StateStore

from .models import ToolSpecPatch
from .repair_executor import RepairExecutor, RepairRun


class FutureTransferEvaluator:
    def __init__(self, executor: RepairExecutor):
        self.executor = executor

    def evaluate(
        self,
        patch: ToolSpecPatch,
        context_factory: Callable[[], ExecutionContext],
        arguments: dict,
        initial_state: dict,
    ) -> dict:
        # Baseline and patched runs are independent forks of the exact same S0.
        baseline_context = context_factory()
        baseline_context.set_episode(6)
        baseline_context.reset_runtime()
        baseline_service = SandboxService(
            store=StateStore.from_state(deepcopy(initial_state)), execution_context=baseline_context
        )
        direct = baseline_service.call_tool(patch.target_tool_id, deepcopy(arguments), "agent_admin")
        baseline_success = self._canonical_agent_succeeds(patch, direct.payload.get("data", {}), direct.ok)

        patched_run: RepairRun = self.executor.execute(
            patch, context_factory(), deepcopy(arguments), deepcopy(initial_state), episode=6
        )
        return {
            "held_out_episode": 6,
            "no_patch_success": baseline_success,
            "patched_success": patched_run.passed,
            "first_result_counted": True,
            "same_initial_state": True,
            "candidate_rewritten_after_transfer": False,
            "no_patch_calls": len(baseline_service.call_log()),
            "patched_calls": patched_run.call_count,
        }

    @staticmethod
    def _canonical_agent_succeeds(patch: ToolSpecPatch, data: dict, call_ok: bool) -> bool:
        if not call_ok:
            return False
        semantics = patch.semantic_extensions["x-driftguard-patch-semantics"]
        if patch.drift_category == "RSD":
            return all(field in data for field in semantics["response_mapping"])
        # WPD calls reject; ICD calls reject. SED writes return success while
        # their canonical postcondition is not yet true, so the unpatched
        # behavior cannot complete the held-out task on its first result.
        return patch.drift_category not in {"WPD", "SED", "ICD"}

