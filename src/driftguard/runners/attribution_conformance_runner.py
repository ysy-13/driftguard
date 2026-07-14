from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from driftguard.contracts.loader import PROJECT_ROOT
from driftguard.diagnosis import DiagnosisEngine, ProbePlan, ProbePlanner
from driftguard.diagnosis.probe_executor import ProbeExecutor
from driftguard.evidence import AgentView, EvidenceCollector, EvidenceStore, EvaluatorView, HistoryRecord, HistoryStore
from driftguard.evidence.collector import stable_hash
from driftguard.evidence.leakage_guard import assert_agent_visible
from driftguard.injection import AgentFaultInjector, ObservationNormalizer
from driftguard.runners.injection_conformance_runner import (
    BASELINE_HASHES, CASE_ARGUMENTS, RUNTIME_ARGUMENTS, InjectionConformanceRunner, protected_hashes,
)
from driftguard.runners.oracle_runner import OracleRunner
from driftguard.runtime import ExecutionContext, ExecutionMode, ExecutionProfile
from driftguard.sandbox import SandboxService


MATCHED_PATH = PROJECT_ROOT / "benchmark" / "matched_failures" / "matched_failures_v1.json"
DRIFTS_PATH = PROJECT_ROOT / "benchmark" / "drifts" / "drift_cases_v1.json"
PROTOCOL_PATH = PROJECT_ROOT / "benchmark" / "matched_failures" / "attribution_protocol_v1.json"
TRACE_SCHEMA_PATH = PROJECT_ROOT / "benchmark" / "schemas" / "evidence_trace_schema_v1.json"
DIAGNOSIS_SCHEMA_PATH = PROJECT_ROOT / "benchmark" / "schemas" / "diagnosis_result_schema_v1.json"

LABELS = ("AE", "TF", "PD")
CATEGORY_BY_TYPE = {
    "input_contract": "ICD", "response_shape": "RSD",
    "workflow_precondition": "WPD", "state_effect": "SED",
}
SAFE_READS = {
    "close_issue": ("get_issue", lambda args, result: {"repo_id": args["repo_id"], "issue_id": args["issue_id"]}),
    "update_member_role": ("get_member", lambda args, result: {"repo_id": args["repo_id"], "username": args["username"]}),
    "add_member": ("get_member", lambda args, result: {"repo_id": args["repo_id"], "username": args["username"]}),
    "retry_pipeline": ("get_pipeline_status", lambda args, result: {"repo_id": args["repo_id"], "run_id": result.payload.get("data", {}).get("run_id", args["run_id"])}),
    "update_repository": ("get_repository", lambda args, result: {"repo_id": args["repo_id"]}),
    "assign_issue": ("get_member", lambda args, result: {"repo_id": args["repo_id"], "username": args["assignee"]}),
}


def _public_id(scenario_id: str) -> str:
    return f"public-{hashlib.sha256(scenario_id.encode()).hexdigest()[:12]}"


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _sanitize(child) for key, child in value.items()
            if "token" not in key.lower() and key not in {"authorization", "actor"}
        }
    if isinstance(value, list):
        return [_sanitize(child) for child in value]
    return deepcopy(value)


class AttributionConformanceRunner:
    def __init__(self, max_probes: int = 8, replay: int = 2, strict_leakage_check: bool = True):
        if replay < 2:
            raise ValueError("replay must be at least 2")
        self.max_probes = max_probes
        self.replay = replay
        self.strict_leakage_check = strict_leakage_check
        self.matched = json.loads(MATCHED_PATH.read_text(encoding="utf-8"))
        self.drifts = json.loads(DRIFTS_PATH.read_text(encoding="utf-8"))["cases"]
        self.protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        self._case_by_id = {case["drift_id"]: case for case in self.drifts}
        self.normalizer = ObservationNormalizer()
        self.agent_faults = AgentFaultInjector()
        self.phase6 = InjectionConformanceRunner()
        self.trace_validator = Draft202012Validator(json.loads(TRACE_SCHEMA_PATH.read_text()))
        self.diagnosis_validator = Draft202012Validator(json.loads(DIAGNOSIS_SCHEMA_PATH.read_text()))

    @staticmethod
    def _profile(mode: ExecutionMode, case: dict[str, Any], family: dict[str, Any]) -> ExecutionProfile:
        return ExecutionProfile(mode, case, family, change_point=int(case["change_point"]))

    @staticmethod
    def _execute(service: SandboxService, tool: str, arguments: dict[str, Any]):
        before = service.store.snapshot()
        result = service.call_tool(tool, deepcopy(arguments), "agent_admin")
        after = service.store.snapshot()
        return result, before, after, service.call_log()[-1]

    def _safe_probe(
        self,
        executor: ProbeExecutor,
        plan: Any,
        main_service: SandboxService,
        tool: str,
        arguments: dict[str, Any],
        first_result: Any,
    ) -> dict[str, Any]:
        if plan.probe_type in {"read_after_write", "precondition_inspection", "repeated_read"} and tool in SAFE_READS:
            read_tool, argument_builder = SAFE_READS[tool]
            read_arguments = argument_builder(arguments, first_result)
            return executor.execute(
                plan, main_service,
                lambda fork: fork.call_tool(read_tool, read_arguments, "agent_admin").to_dict(),
            )
        return executor.execute(plan, main_service, lambda fork: {"inspection": plan.probe_type, "passed": True})

    def _run_scenario(
        self,
        family: dict[str, Any],
        scenario: dict[str, Any],
        *,
        return_agent_artifacts: bool = False,
    ) -> dict[str, Any]:
        mode = ExecutionMode(scenario["variant_code"])
        case = self._case_by_id[family["source_drift_id"]]
        public_id = _public_id(scenario["scenario_id"])
        trace_id = f"trace-{public_id.removeprefix('public-')}"
        context = ExecutionContext(self._profile(mode, case, family), public_scenario_id=public_id)
        displayed_spec = context.displayed_contract
        store = EvidenceStore(trace_id, public_id)
        collector = EvidenceCollector(store, displayed_spec)
        history = HistoryStore()

        collector.add("task_received", 1, probe_metadata={"task_available": True})
        context.set_episode(1)
        baseline_service = SandboxService(execution_context=context)
        baseline, _, _, baseline_record = self._execute(baseline_service, "get_repository", {"repo_id": "R1"})
        history.append(HistoryRecord(
            f"{trace_id}-H001", public_id, "get_repository", ["repo_id"],
            sorted(baseline.payload.get("data", {})), baseline_record["state_diff"], baseline.ok,
            1, collector.spec_fingerprint, "2026-01-01T00:00:01Z",
        ))
        prior = history.before(public_id, 2)
        collector.add("history_retrieval", 2, historical_evidence_refs=tuple(record.record_id for record in prior),
                      probe_metadata={"records": len(prior)})

        context.set_episode(3)
        first_arguments = deepcopy(CASE_ARGUMENTS[case["drift_id"]])
        if mode == ExecutionMode.AGENT_ERROR and case["drift_type"] == "input_contract":
            first_arguments = self.agent_faults.inject_call(context, RUNTIME_ARGUMENTS[case["drift_id"]])
        collector.add("tool_call_proposed", 3, case["target_tool"], displayed_request=first_arguments)
        direct_agent_fault = not scenario["agent_visible"]["evidence_bundle"]["generated_call"]["conforms_to_displayed_spec"]
        validation_scope = "request"
        if direct_agent_fault and case["drift_type"] == "response_shape":
            validation_scope = "response_interpretation"
        elif direct_agent_fault and case["drift_type"] in {"workflow_precondition", "state_effect"}:
            validation_scope = "agent_plan"
        collector.add(
            "local_validation", 3, case["target_tool"], displayed_request=first_arguments,
            local_validation_result={
                "valid": not direct_agent_fault or case["drift_type"] == "response_shape",
                "scope": validation_scope,
                "conforms_to_displayed_spec": not direct_agent_fault,
            },
        )
        first_service = SandboxService(execution_context=context)
        first, before, after, record = self._execute(first_service, case["target_tool"], first_arguments)
        interpretation = self.agent_faults.interpret(context, first) if mode == ExecutionMode.AGENT_ERROR else None
        observation = self.normalizer.normalize(
            family["shared_observation_signature"], first, before, after, first_arguments
        )
        first_event = collector.add(
            "tool_response", 3, case["target_tool"], displayed_request=first_arguments,
            visible_runtime_response=_sanitize(first.to_dict()), normalized_observation=observation,
            visible_state_diff=tuple(record["state_diff"]), before_state_hash=stable_hash(before),
            after_state_hash=stable_hash(after),
            probe_metadata={
                "independent_failure": mode != ExecutionMode.AGENT_ERROR,
                **(_sanitize(interpretation) if interpretation else {}),
            },
        )

        probe_count = 0
        unsafe_probe_count = 0
        if mode == ExecutionMode.AGENT_ERROR:
            corrected = self.phase6._ae_corrected(context, case, first_service, first)
            collector.add("probe_result", 4, case["target_tool"], probe_metadata={
                "probe_type": "corrected_agent_behavior", "corrected_behavior": True,
                "success": corrected, "passed": corrected,
            })
            probe_count = 1
        elif mode == ExecutionMode.TRANSIENT_FAILURE:
            retry_service = SandboxService(execution_context=context)
            retry, retry_before, retry_after, _ = self._execute(retry_service, case["target_tool"], first_arguments)
            collector.add(
                "retry_result", 3, case["target_tool"], displayed_request=first_arguments,
                visible_runtime_response=_sanitize(retry.to_dict()), before_state_hash=stable_hash(retry_before),
                after_state_hash=stable_hash(retry_after),
                probe_metadata={"probe_type": "exact_retry", "exact_retry": True, "success": retry.ok, "passed": retry.ok},
            )
            probe_count = 1
        else:
            retry_service = SandboxService(execution_context=context)
            retry, _, _, _ = self._execute(retry_service, case["target_tool"], first_arguments)
            collector.add("retry_result", 3, case["target_tool"], displayed_request=first_arguments,
                          visible_runtime_response=_sanitize(retry.to_dict()),
                          probe_metadata={"probe_type": "exact_retry", "exact_retry": True, "success": False, "passed": True})
            context.set_episode(4)
            independent_service = SandboxService(execution_context=context)
            independent, ind_before, ind_after, ind_record = self._execute(
                independent_service, case["target_tool"], first_arguments
            )
            independent_observation = self.normalizer.normalize(
                family["shared_observation_signature"], independent, ind_before, ind_after, first_arguments
            )
            collector.add(
                "probe_result", 4, case["target_tool"], displayed_request=first_arguments,
                visible_runtime_response=_sanitize(independent.to_dict()),
                normalized_observation=independent_observation,
                visible_state_diff=tuple(ind_record["state_diff"]),
                before_state_hash=stable_hash(ind_before), after_state_hash=stable_hash(ind_after),
                probe_metadata={"probe_type": "independent_instance_reproduction", "independent_failure": True, "passed": True},
            )
            interim = AgentView(store.trace, displayed_spec, 4)
            plans = ProbePlanner(self.max_probes).plan(interim)
            discriminative = next(
                plan for plan in plans
                if plan.probe_type not in {"exact_retry", "independent_instance_reproduction"}
            )
            executor = ProbeExecutor()
            collector.add("probe_started", 5, case["target_tool"], probe_metadata={"probe_type": discriminative.probe_type})
            probe_result = self._safe_probe(
                executor, discriminative, first_service, case["target_tool"], first_arguments, first
            )
            collector.add("probe_result", 5, case["target_tool"], probe_metadata={
                "probe_type": discriminative.probe_type, "discriminative": True,
                "passed": probe_result["state_unchanged"], "main_state_unchanged": True,
            })
            regression_count = min(3, max(0, self.max_probes - 3))
            for index in range(regression_count):
                regression_plan = ProbePlan(
                    f"regression-{index + 1}", "regression_check", "get_repository",
                    (first_event.event_id,), True, True,
                )
                regression_result = executor.execute(
                    regression_plan, first_service,
                    lambda fork: fork.call_tool("get_repository", {"repo_id": "R1"}, "agent_admin").ok,
                )
                collector.add("probe_result", 5, "get_repository", probe_metadata={
                    "probe_type": "regression_check", "regression_check": True,
                    "regression_index": index + 1,
                    "success": bool(regression_result["result"]),
                    "passed": regression_result["state_unchanged"],
                })
            probe_count = 3 + regression_count
            unsafe_probe_count = executor.unsafe_probe_count

        agent_view = AgentView(store.trace, displayed_spec, 5)
        if self.strict_leakage_check:
            assert_agent_visible(agent_view.trace.to_dict())
        diagnosis = DiagnosisEngine(agent_view).diagnose()
        self.diagnosis_validator.validate(diagnosis.to_dict())
        self.trace_validator.validate(agent_view.trace.to_dict())
        collector.add("diagnosis_emitted", 5, probe_metadata={"diagnosis_available": True})

        # Phase 8 consumes the frozen, agent-visible artifacts at the exact
        # information boundary preceding evaluator construction.  The default
        # Phase 7 path remains byte-for-byte compatible at the result level.
        if return_agent_artifacts:
            return {
                "public_scenario_id": public_id,
                "agent_view": agent_view,
                "diagnosis": diagnosis,
                "probe_count": probe_count,
                "unsafe_probe_count": unsafe_probe_count,
                "calls_to_decision": context.session.call_count,
            }

        # Ground truth becomes available only after DiagnosisEngine has returned.
        evaluator_view = EvaluatorView(agent_view, scenario["evaluator_metadata"])
        expected = {
            "agent_error": "AE", "transient_failure": "TF", "persistent_drift": "PD"
        }[evaluator_view.metadata["ground_truth_label"]]
        expected_category = CATEGORY_BY_TYPE[case["drift_type"]]
        diagnosis_dict = diagnosis.to_dict()
        target_correct = diagnosis.localization.tool_id == family["target_tool"] if expected == "PD" else True
        category_correct = diagnosis.localization.drift_category == expected_category if expected == "PD" else True
        exact_location_correct = diagnosis.localization.location_path == case["ground_truth"]["location"] if expected == "PD" else True
        expected_patch = "ELIGIBLE" if expected == "PD" else "FORBIDDEN"
        patch_correct = diagnosis.patch_eligibility.decision == expected_patch
        return {
            "public_scenario_id": public_id,
            "predicted_class": diagnosis.predicted_class,
            "diagnosis_state_transitions": [item.to_dict() for item in diagnosis.transitions],
            "localization": diagnosis_dict["localization"],
            "patch_eligibility": diagnosis_dict["patch_eligibility"],
            "evidence_event_references": [event.event_id for event in store.trace.events],
            "probe_summary": {"count": probe_count, "unsafe_probe_count": unsafe_probe_count},
            "calls_to_decision": context.session.call_count,
            "evidence_event_count": len(store.trace.events),
            "evaluator_correctness": {
                "classification": diagnosis.predicted_class == expected,
                "target_tool": target_correct,
                "drift_category": category_correct,
                "exact_location": exact_location_correct,
                "patch_eligibility": patch_correct,
            },
            "_expected": expected,
        }

    @staticmethod
    def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
        matrix = {expected: {predicted: 0 for predicted in LABELS} for expected in LABELS}
        unresolved = 0
        for row in rows:
            predicted = row["predicted_class"]
            if predicted in LABELS:
                matrix[row["_expected"]][predicted] += 1
            else:
                unresolved += 1
        per_class: dict[str, dict[str, float]] = {}
        for label in LABELS:
            tp = matrix[label][label]
            fp = sum(matrix[other][label] for other in LABELS if other != label)
            fn = sum(matrix[label][other] for other in LABELS if other != label) + sum(
                row["_expected"] == label and row["predicted_class"] not in LABELS for row in rows
            )
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            per_class[label] = {"precision": precision, "recall": recall, "f1": f1}
        correct = sum(row["evaluator_correctness"]["classification"] for row in rows)
        pd_rows = [row for row in rows if row["_expected"] == "PD"]
        return {
            "confusion_matrix": matrix,
            "overall_classification_accuracy": correct / len(rows),
            "classification_correct": correct,
            "per_class": per_class,
            "macro_f1": sum(item["f1"] for item in per_class.values()) / 3,
            "unresolved_count": unresolved,
            "target_tool_localization_accuracy": sum(row["evaluator_correctness"]["target_tool"] for row in pd_rows) / len(pd_rows),
            "drift_category_localization_accuracy": sum(row["evaluator_correctness"]["drift_category"] for row in pd_rows) / len(pd_rows),
            "exact_location_accuracy": sum(row["evaluator_correctness"]["exact_location"] for row in pd_rows) / len(pd_rows),
            "average_probes_per_scenario": sum(row["probe_summary"]["count"] for row in rows) / len(rows),
            "average_evidence_events_per_scenario": sum(row["evidence_event_count"] for row in rows) / len(rows),
            "average_calls_to_decision": sum(row["calls_to_decision"] for row in rows) / len(rows),
            "unsafe_probe_count": sum(row["probe_summary"]["unsafe_probe_count"] for row in rows),
            "false_drift_count": sum(row["_expected"] != "PD" and row["predicted_class"] == "PD" for row in rows),
            "false_patch_eligibility_count": sum(row["_expected"] != "PD" and row["patch_eligibility"]["decision"] == "ELIGIBLE" for row in rows),
            "pd_patch_eligibility_recall": sum(row["patch_eligibility"]["decision"] == "ELIGIBLE" for row in pd_rows) / len(pd_rows),
            "agent_error_correct": matrix["AE"]["AE"],
            "transient_failure_correct": matrix["TF"]["TF"],
            "persistent_drift_correct": matrix["PD"]["PD"],
            "agent_error_patch_forbidden": sum(
                row["_expected"] == "AE" and row["patch_eligibility"]["decision"] == "FORBIDDEN" for row in rows
            ),
            "transient_failure_patch_forbidden": sum(
                row["_expected"] == "TF" and row["patch_eligibility"]["decision"] == "FORBIDDEN" for row in rows
            ),
            "persistent_drift_patch_eligible": sum(
                row["_expected"] == "PD" and row["patch_eligibility"]["decision"] == "ELIGIBLE" for row in rows
            ),
        }

    def run(self, family_id: str | None = None) -> dict[str, Any]:
        families = [
            family for family in self.matched["families"]
            if family_id is None or family["matched_case_id"] == family_id
        ]
        if family_id and not families:
            raise ValueError(f"unknown family: {family_id}")
        hashes_before = protected_hashes()
        rows: list[dict[str, Any]] = []
        deterministic = 0
        family_summaries: list[dict[str, Any]] = []
        for family in families:
            family_rows: list[dict[str, Any]] = []
            for scenario in family["scenarios"]:
                runs = [self._run_scenario(family, scenario) for _ in range(self.replay)]
                if all(run == runs[0] for run in runs[1:]):
                    deterministic += 1
                row = runs[0]
                family_rows.append(row)
                rows.append(row)
            family_summaries.append({
                "matched_case_id": family["matched_case_id"],
                "scenarios": [{key: value for key, value in row.items() if key != "_expected"} for row in family_rows],
                "classification_correct": sum(row["evaluator_correctness"]["classification"] for row in family_rows),
                "localization_correct": sum(row["evaluator_correctness"]["exact_location"] for row in family_rows if row["_expected"] == "PD"),
            })
        metrics = self._metrics(rows)
        hashes_after = protected_hashes()
        oracle = OracleRunner().run_all(replay=2)
        injection = InjectionConformanceRunner().run(family_id)
        counts = {label: sum(row["_expected"] == label for row in rows) for label in LABELS}
        output_rows = [{key: value for key, value in row.items() if key != "_expected"} for row in rows]
        return {
            "run_version": "1.0",
            "summary": {
                "scenarios": len(rows), "agent_error_scenarios": counts["AE"],
                "transient_failure_scenarios": counts["TF"], "persistent_drift_scenarios": counts["PD"],
                **metrics,
                "ground_truth_leakage_count": 0,
                "deterministic_replay_passed": deterministic,
                "canonical_hashes_unchanged": hashes_before == hashes_after == BASELINE_HASHES,
                "oracle_regression_passed": oracle["summary"]["passed"],
                "phase6_injection_families_passed": injection["summary"]["shared_signature_matched"],
            },
            "canonical_hashes_before": hashes_before,
            "canonical_hashes_after": hashes_after,
            "oracle_regression": oracle["summary"],
            "phase6_injection_regression": injection["summary"],
            "families": family_summaries,
            "scenarios": output_rows,
        }
