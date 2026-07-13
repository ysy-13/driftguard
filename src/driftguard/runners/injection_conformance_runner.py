from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

from driftguard.contracts.loader import PROJECT_ROOT
from driftguard.injection import AgentErrorHook, ObservationNormalizer
from driftguard.runners.oracle_runner import OracleRunner
from driftguard.runtime import ExecutionContext, ExecutionMode, ExecutionProfile
from driftguard.sandbox.service import SandboxService


DRIFTS_PATH = PROJECT_ROOT / "benchmark" / "drifts" / "drift_cases_v1.json"
MATCHED_PATH = PROJECT_ROOT / "benchmark" / "matched_failures" / "matched_failures_v1.json"
PROTECTED_PATHS = (
    "benchmark/openapi/driftguard_openapi_v1.yaml",
    "benchmark/fixtures/initial_state_v1.json",
    "benchmark/tasks/tasks_v1.json",
    "benchmark/drifts/drift_cases_v1.json",
    "benchmark/drifts/task_drift_coverage_v1.json",
    "benchmark/matched_failures/matched_failures_v1.json",
    "benchmark/matched_failures/attribution_protocol_v1.json",
)
BASELINE_HASHES = {
    "benchmark/openapi/driftguard_openapi_v1.yaml": "be893ee40a748410d49befe8ca45f645b47e9a770cf2bb7e0b90db9c71472267",
    "benchmark/fixtures/initial_state_v1.json": "7cc0f4af8eb45d6e1a4a03741c2b44484b80736cf2ba6bd1f5c8e2c7fb7db3c0",
    "benchmark/tasks/tasks_v1.json": "851425f77f4ad94633e5eb10937b783a51e9aa9f54955b408565064f3a788936",
    "benchmark/drifts/drift_cases_v1.json": "dda8f8c1759075bb52b14431b9ee5fcb6fbe93c1cf239a474548a6127228a00e",
    "benchmark/drifts/task_drift_coverage_v1.json": "f83a5ce5da1bd9a179dfdd1abfbb747469b29644c71304490df0750bd527be3c",
    "benchmark/matched_failures/matched_failures_v1.json": "58c921b754976ee16f29676030bfbb57de4e0ee2f7be14a4af8353d057296f7e",
    "benchmark/matched_failures/attribution_protocol_v1.json": "3e0779032b4613b9d17868ae1f2d3fe0c582214cc80bdb860a5a88aae69d4106",
}


CASE_ARGUMENTS: dict[str, dict[str, Any]] = {
    "ICD-01": {"repo_id": "R1", "title": "Review authentication logs"},
    "ICD-02": {"repo_id": "R1", "workflow_id": "ci", "ref": "main"},
    "ICD-03": {"repo_id": "R1", "issue_id": 101, "assignee": "bob"},
    "ICD-04": {"repo_id": "R1", "run_id": 501},
    "ICD-05": {"repo_id": "R1", "username": "carol", "role": "write"},
    "RSD-01": {"repo_id": "R1", "title": "Response mapping"},
    "RSD-02": {"repo_id": "R1", "run_id": 501},
    "RSD-03": {"repo_id": "R1", "username": "bob"},
    "RSD-04": {"repo_id": "R1"},
    "RSD-05": {"repo_id": "R1", "run_id": 501},
    "WPD-01": {"repo_id": "R1", "issue_id": 101},
    "WPD-02": {"repo_id": "R1", "run_id": 501},
    "WPD-03": {"repo_id": "R1", "issue_id": 101, "assignee": "bob"},
    "WPD-04": {"repo_id": "R1", "username": "dave", "role": "write"},
    "WPD-05": {"repo_id": "R1", "username": "erin", "role": "triage"},
    "SED-01": {"repo_id": "R1", "issue_id": 101},
    "SED-02": {"repo_id": "R1", "username": "carol", "role": "write"},
    "SED-03": {"repo_id": "R1", "username": "erin", "role": "triage"},
    "SED-04": {"repo_id": "R1", "run_id": 501},
    "SED-05": {"repo_id": "R1", "default_branch": "develop"},
}
RUNTIME_ARGUMENTS = {
    "ICD-01": {"repo_id": "R1", "title": "Review authentication logs", "priority": "medium"},
    "ICD-02": {"repo_id": "R1", "workflow_id": "ci", "branch": "main"},
    "ICD-03": {"repo_id": "R1", "issue_id": 101, "assignee_username": "bob"},
    "ICD-04": {"repo_id": "R1", "run_id": 501, "failed_only": True},
    "ICD-05": {"repo_id": "R1", "username": "carol", "role": "developer"},
}


def protected_hashes() -> dict[str, str]:
    return {name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest() for name in PROTECTED_PATHS}


class InjectionConformanceRunner:
    def __init__(self) -> None:
        self.drifts = json.loads(DRIFTS_PATH.read_text(encoding="utf-8"))["cases"]
        self.families = json.loads(MATCHED_PATH.read_text(encoding="utf-8"))["families"]
        self._drift_by_id = {case["drift_id"]: case for case in self.drifts}
        self.normalizer = ObservationNormalizer()
        self.agent_faults = AgentErrorHook()

    @staticmethod
    def _state_effect_matches(case_id: str, before: dict[str, Any], after: dict[str, Any]) -> bool:
        old, new = before["repositories"]["R1"], after["repositories"]["R1"]
        if case_id == "SED-01":
            return new["issues"]["101"]["state"] == "open"
        if case_id == "SED-02":
            return new["members"]["carol"]["role"] == old["members"]["carol"]["role"]
        if case_id == "SED-03":
            return "erin" not in new["members"]
        if case_id == "SED-04":
            old_runs, new_runs = old["pipeline_runs"], new["pipeline_runs"]
            added = set(new_runs) - set(old_runs)
            return (
                new_runs["501"] == old_runs["501"]
                and len(added) == 1
                and new_runs[next(iter(added))]["attempt"] == 1
                and after["next_ids"]["R1"]["run_id"] == before["next_ids"]["R1"]["run_id"] + 1
            )
        if case_id == "SED-05":
            return new["default_branch"] == old["default_branch"]
        return True

    @staticmethod
    def _profile(mode: ExecutionMode, case: dict[str, Any], family: dict[str, Any]) -> ExecutionProfile:
        return ExecutionProfile(mode, case, family, change_point=int(case["change_point"]))

    @staticmethod
    def _execute(service: SandboxService, case: dict[str, Any], arguments: dict[str, Any]):
        before = service.store.snapshot()
        result = service.call_tool(case["target_tool"], deepcopy(arguments), "agent_admin")
        return result, before, service.store.snapshot()

    @staticmethod
    def _response_shape_correct(case_id: str, result: Any) -> bool:
        data = result.payload.get("data", {})
        if case_id == "RSD-01":
            return "id" in data and "issue_id" not in data
        if case_id == "RSD-02":
            return "state" in data and "result" in data and "status" not in data and "conclusion" not in data
        if case_id == "RSD-03":
            return "permissions" in data and "role" not in data and "base_permission" not in data
        if case_id == "RSD-04":
            return "defaultBranch" in data and "default_branch" not in data
        if case_id == "RSD-05":
            return "run_attempt" in data and "attempt" not in data
        return False

    def _ae_corrected(self, context: ExecutionContext, case: dict[str, Any], first_service: SandboxService, first: Any) -> bool:
        drift_id, drift_type = case["drift_id"], case["drift_type"]
        context.set_episode(4)
        if drift_type == "input_contract":
            result = SandboxService(execution_context=context).call_tool(
                case["target_tool"], deepcopy(RUNTIME_ARGUMENTS[drift_id]), "agent_admin"
            )
            return result.ok
        if drift_type == "response_shape":
            result = SandboxService(execution_context=context).call_tool(
                case["target_tool"], deepcopy(CASE_ARGUMENTS[drift_id]), "agent_admin"
            )
            return result.ok and self._response_shape_correct(drift_id, result)
        if drift_type == "workflow_precondition":
            service = SandboxService(execution_context=context)
            if drift_id == "WPD-01":
                assigned = service.call_tool("assign_issue", {"repo_id": "R1", "issue_id": 101, "assignee": "bob"}, "agent_admin")
                return assigned.ok and service.call_tool("close_issue", CASE_ARGUMENTS[drift_id], "agent_admin").ok
            if drift_id == "WPD-02":
                read = service.call_tool("get_pipeline_status", {"repo_id": "R1", "run_id": 501}, "agent_admin")
                token = read.payload["data"].get("verification_token")
                return service.call_tool("retry_pipeline", {**CASE_ARGUMENTS[drift_id], "verification_token": token}, "agent_admin").ok
            if drift_id == "WPD-03":
                read = service.call_tool("get_member", {"repo_id": "R1", "username": "bob"}, "agent_admin")
                token = read.payload["data"].get("membership_verification_token")
                return service.call_tool("assign_issue", {**CASE_ARGUMENTS[drift_id], "membership_verification_token": token}, "agent_admin").ok
            if drift_id == "WPD-04":
                first_step = service.call_tool("update_member_role", {"repo_id": "R1", "username": "dave", "role": "triage"}, "agent_admin")
                return first_step.ok and service.call_tool("update_member_role", CASE_ARGUMENTS[drift_id], "agent_admin").ok
            added = service.call_tool("add_member", {"repo_id": "R1", "username": "erin", "role": "read"}, "agent_admin")
            promoted = service.call_tool("update_member_role", {"repo_id": "R1", "username": "erin", "role": "triage"}, "agent_admin")
            return added.ok and promoted.ok
        if drift_id == "SED-01":
            return first_service.call_tool("get_issue", {"repo_id": "R1", "issue_id": 101}, "agent_admin").payload["data"]["state"] == "closed"
        if drift_id == "SED-02":
            return first_service.call_tool("get_member", {"repo_id": "R1", "username": "carol"}, "agent_admin").payload["data"]["role"] == "write"
        if drift_id == "SED-03":
            premature = first_service.call_tool("assign_issue", {"repo_id": "R1", "issue_id": 101, "assignee": "erin"}, "agent_admin")
            verified = first_service.call_tool("get_member", {"repo_id": "R1", "username": "erin"}, "agent_admin")
            assigned = first_service.call_tool("assign_issue", {"repo_id": "R1", "issue_id": 101, "assignee": "erin"}, "agent_admin")
            return not premature.ok and verified.ok and assigned.ok
        if drift_id == "SED-04":
            run_id = first.payload["data"]["run_id"]
            return first_service.call_tool("get_pipeline_status", {"repo_id": "R1", "run_id": run_id}, "agent_admin").ok
        return first_service.call_tool("get_repository", {"repo_id": "R1"}, "agent_admin").payload["data"]["default_branch"] == "develop"

    def _run_variant(self, mode: ExecutionMode, case: dict[str, Any], family: dict[str, Any]) -> dict[str, Any]:
        context = ExecutionContext(self._profile(mode, case, family))
        contract_alignment = {
            "contracts_equal": context.contracts.contracts_equal,
            "displayed_source": context.contracts.displayed.source,
            "runtime_source": context.contracts.runtime.source,
            "distinct_objects": context.contracts.displayed.document is not context.contracts.runtime.document,
        }
        service = SandboxService(execution_context=context)
        first_arguments = CASE_ARGUMENTS[case["drift_id"]]
        if mode == ExecutionMode.AGENT_ERROR and case["drift_type"] == "input_contract":
            first_arguments = self.agent_faults.inject_call(context, RUNTIME_ARGUMENTS[case["drift_id"]])
        first, before, after = self._execute(service, case, first_arguments)
        interpretation = self.agent_faults.interpret(context, first) if mode == ExecutionMode.AGENT_ERROR else None
        observation = self.normalizer.normalize(
            family["shared_observation_signature"], first, before, after, first_arguments
        )
        first_ok = observation == family["shared_observation_signature"] and self._state_effect_matches(case["drift_id"], before, after)

        if mode == ExecutionMode.AGENT_ERROR:
            followup_ok = self._ae_corrected(context, case, service, first)
            active_after = not context.agent_fault_active
        else:
            context.set_episode(4)
            follow_service = SandboxService(execution_context=context)
            follow, follow_before, follow_after = self._execute(follow_service, case, CASE_ARGUMENTS[case["drift_id"]])
            follow_observation = self.normalizer.normalize(
                family["shared_observation_signature"], follow, follow_before, follow_after, CASE_ARGUMENTS[case["drift_id"]]
            )
            if mode == ExecutionMode.TRANSIENT_FAILURE:
                followup_ok = follow.ok and follow_observation != family["shared_observation_signature"]
                active_after = context.profile.scenario_id in context.session.consumed_transient
            else:
                followup_ok = (
                    follow_observation == family["shared_observation_signature"]
                    and self._state_effect_matches(case["drift_id"], follow_before, follow_after)
                )
                active_after = context.injection_active(case["target_tool"])
        return {
            "normalized_signature": observation,
            "first_failure_matched": first_ok,
            "followup_matched": followup_ok,
            "lifecycle_verified": active_after,
            "contract_alignment": contract_alignment,
            "interpretation_record": interpretation,
        }

    def run(self, family_id: str | None = None) -> dict[str, Any]:
        selected = [family for family in self.families if family_id is None or family["matched_case_id"] == family_id]
        if family_id is not None and not selected:
            raise ValueError(f"unknown family: {family_id}")
        hashes_before = protected_hashes()
        family_rows: list[dict[str, Any]] = []
        for family in selected:
            case = self._drift_by_id[family["source_drift_id"]]
            variants = {mode.value: self._run_variant(mode, case, family) for mode in ExecutionMode}
            signatures_match = all(
                variants[mode]["normalized_signature"] == family["shared_observation_signature"]
                for mode in ("AE", "TF", "PD")
            )
            family_rows.append(
                {
                    "matched_case_id": family["matched_case_id"],
                    "target_tool": family["target_tool"],
                    "agent_error_signature": variants["AE"]["normalized_signature"],
                    "transient_failure_signature": variants["TF"]["normalized_signature"],
                    "persistent_drift_signature": variants["PD"]["normalized_signature"],
                    "shared_signature_matched": signatures_match,
                    "agent_error_corrected": variants["AE"]["followup_matched"],
                    "transient_recovered": variants["TF"]["followup_matched"],
                    "persistent_reproduced": variants["PD"]["followup_matched"],
                    "variant_details": variants,
                }
            )
        hashes_after = protected_hashes()
        canonical_protected = hashes_before == hashes_after == BASELINE_HASHES
        for row in family_rows:
            row["canonical_protected"] = canonical_protected
        oracle = OracleRunner().run_all(replay=2)
        count = len(family_rows)
        return {
            "run_version": "1.0",
            "summary": {
                "families": count,
                "first_failure_scenarios": count * 3,
                "agent_error_executed": sum(row["variant_details"]["AE"]["first_failure_matched"] for row in family_rows),
                "agent_error_corrected": sum(row["agent_error_corrected"] for row in family_rows),
                "transient_failure_executed": sum(row["variant_details"]["TF"]["first_failure_matched"] for row in family_rows),
                "persistent_drift_executed": sum(row["variant_details"]["PD"]["first_failure_matched"] for row in family_rows),
                "shared_signature_matched": sum(row["shared_signature_matched"] for row in family_rows),
                "transient_recovered": sum(row["transient_recovered"] for row in family_rows),
                "persistent_reproduced": sum(row["persistent_reproduced"] for row in family_rows),
                "canonical_hash_unchanged": canonical_protected,
                "oracle_regression_passed": oracle["summary"]["passed"],
                "oracle_deterministic_replay": oracle["summary"]["deterministic_replay_passed"],
                "oracle_forbidden_side_effects": oracle["summary"]["forbidden_side_effects"],
            },
            "canonical_hashes_before": hashes_before,
            "canonical_hashes_after": hashes_after,
            "families": family_rows,
        }
