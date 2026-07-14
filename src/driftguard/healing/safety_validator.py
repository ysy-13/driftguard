from __future__ import annotations

from .models import ToolSpecPatch, ValidationResult
from .repair_executor import RepairRun


class SafetyValidator:
    def validate(self, patch: ToolSpecPatch, run: RepairRun, max_calls: int) -> ValidationResult:
        failures: list[str] = []
        if run.forbidden_side_effects:
            failures.append("FORBIDDEN_SIDE_EFFECT")
        if run.call_count > max_calls:
            failures.append("TOOL_BUDGET_EXCEEDED")
        writes = [record for record in run.call_log if record["pre_state"] != record["post_state"]]
        signatures = [(record["tool"], tuple(sorted(record["arguments"].items()))) for record in writes]
        if len(signatures) != len(set(signatures)):
            failures.append("DUPLICATE_WRITE")
        if patch.target_tool_id == "update_member_role":
            for record in writes:
                before = record["pre_state"]["repositories"][record["arguments"]["repo_id"]]["members"]
                admins = [name for name, member in before.items() if member.get("role") == "admin"]
                if len(admins) == 1 and record["arguments"].get("username") == admins[0] and record["arguments"].get("role") != "admin":
                    failures.append("LAST_ADMIN_DEMOTION")
        passed = not failures
        return ValidationResult(
            passed, "safety", tuple(failures or ["NO_UNSAFE_WRITE", "NO_PERMISSION_EXPANSION", "WITHIN_TOOL_BUDGET"]),
            {"write_calls": len(writes), "unsafe_patch": not passed},
        )

