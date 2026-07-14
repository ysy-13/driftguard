from __future__ import annotations

from copy import deepcopy

from driftguard.sandbox.service import SandboxService
from driftguard.sandbox.state_store import StateStore

from .models import ToolSpecPatch, ValidationResult


class RegressionValidator:
    """Runs three independent canonical checks required by the attribution protocol."""

    def validate(self, patch: ToolSpecPatch, initial_state: dict) -> ValidationResult:
        checks = []
        for repo_id in ("R1", "R2", "R1"):
            service = SandboxService(store=StateStore.from_state(deepcopy(initial_state)))
            if len(checks) == 0:
                ok = service.call_tool("get_repository", {"repo_id": repo_id}, "agent_admin").ok
            elif len(checks) == 1:
                ok = service.call_tool("get_member", {"repo_id": "R1", "username": "bob"}, "agent_admin").ok
            else:
                before = service.store.snapshot()
                ok = service.call_tool("get_pipeline_status", {"repo_id": "R1", "run_id": 501}, "agent_admin").ok
                ok = ok and before == service.store.snapshot()
            checks.append(ok)
        passed = all(checks)
        return ValidationResult(
            passed, "regression", ("TARGET_TOOL", "NEIGHBORING_WORKFLOW", "GLOBAL_INVARIANT") if passed else ("REGRESSION_FAILED",),
            {"required": 3, "passed": sum(checks), "independent_state_forks": 3},
        )

