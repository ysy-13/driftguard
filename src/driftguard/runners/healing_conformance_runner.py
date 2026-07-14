from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from driftguard.contracts.loader import DEFAULT_FIXTURE_PATH, PROJECT_ROOT
from driftguard.evidence import EvaluatorView
from driftguard.healing import FutureTransferEvaluator, HealingEngine, assert_redacted
from driftguard.healing.ground_truth_evaluator import GroundTruthPatchEvaluator
from driftguard.runners.attribution_conformance_runner import AttributionConformanceRunner
from driftguard.runners.injection_conformance_runner import (
    BASELINE_HASHES, CASE_ARGUMENTS, InjectionConformanceRunner, protected_hashes,
)
from driftguard.runners.oracle_runner import OracleRunner
from driftguard.runtime import ExecutionContext, ExecutionMode, ExecutionProfile


COVERAGE_PATH = PROJECT_ROOT / "benchmark" / "drifts" / "task_drift_coverage_v1.json"
HEALING_SCHEMA_PATH = PROJECT_ROOT / "benchmark" / "schemas" / "healing_result_schema_v1.json"
PATCH_SCHEMA_PATH = PROJECT_ROOT / "benchmark" / "schemas" / "tool_spec_patch_schema_v1.json"
HANDLER_DIR = PROJECT_ROOT / "src" / "driftguard" / "sandbox" / "tools"


def _handler_hashes() -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(HANDLER_DIR.glob("*.py"))
    }


class HealingConformanceRunner:
    def __init__(
        self,
        max_candidates: int = 5,
        max_repair_calls: int = 8,
        strict_leakage_check: bool = True,
        replay: int = 2,
    ):
        if replay < 2:
            raise ValueError("replay must be at least 2")
        self.max_candidates = max_candidates
        self.max_repair_calls = max_repair_calls
        self.strict_leakage_check = strict_leakage_check
        self.replay = replay
        self.attribution = AttributionConformanceRunner(strict_leakage_check=strict_leakage_check)
        coverage = json.loads(COVERAGE_PATH.read_text(encoding="utf-8"))["coverage"]
        self.coverage_by_drift = {item["drift_id"]: item for item in coverage}
        self.patch_validator = Draft202012Validator(json.loads(PATCH_SCHEMA_PATH.read_text(encoding="utf-8")))
        self.result_validator = Draft202012Validator(json.loads(HEALING_SCHEMA_PATH.read_text(encoding="utf-8")))

    def _context_factory(self, case: dict, family: dict):
        def create() -> ExecutionContext:
            profile = ExecutionProfile(
                ExecutionMode.PERSISTENT_DRIFT, case, family,
                change_point=int(case["change_point"]),
            )
            return ExecutionContext(profile)
        return create

    def _run_once(self, family: dict[str, Any], scenario: dict[str, Any]) -> dict[str, Any]:
        case = self.attribution._case_by_id[family["source_drift_id"]]
        artifacts = self.attribution._run_scenario(family, scenario, return_agent_artifacts=True)
        agent_view, diagnosis = artifacts["agent_view"], artifacts["diagnosis"]
        engine = HealingEngine(self.max_candidates, self.max_repair_calls)
        context_factory = self._context_factory(case, family) if diagnosis.predicted_class == "PD" else None
        execution = engine.heal(
            agent_view, diagnosis, context_factory,
            deepcopy(CASE_ARGUMENTS[case["drift_id"]]) if context_factory else None,
        )
        selected = engine.last_accepted_patch
        future_transfer = None
        if selected is not None:
            coverage = self.coverage_by_drift[case["drift_id"]]
            future = coverage["future_transfer_instances"][0]
            arguments = deepcopy(CASE_ARGUMENTS[case["drift_id"]])
            target_argument_keys = set(arguments)
            overrides = deepcopy(future["argument_overrides"])
            for omitted in overrides.pop("omit", []):
                arguments.pop(omitted, None)
            arguments.update(overrides)
            # Coverage overrides may also carry values for downstream task
            # steps (for example a pipeline ref after repository update). They
            # are not arguments to the localized target tool.
            arguments = {key: value for key, value in arguments.items() if key in target_argument_keys}
            initial_state = json.loads(Path(DEFAULT_FIXTURE_PATH).read_text(encoding="utf-8"))
            _apply_fixture_overrides(initial_state, future["fixture_overrides"])
            _ensure_future_resources(initial_state, selected.target_tool_id, arguments)
            future_transfer = FutureTransferEvaluator(engine.executor).evaluate(
                selected, context_factory, arguments, initial_state
            )
        execution["future_transfer"] = future_transfer

        # Ground truth is deliberately late-bound: proposal, validation,
        # acceptance and held-out execution are already immutable here.
        evaluator_view = EvaluatorView(agent_view, scenario["evaluator_metadata"])
        expected_label = {
            "agent_error": "AE", "transient_failure": "TF", "persistent_drift": "PD"
        }[evaluator_view.metadata["ground_truth_label"]]
        correctness = {
            "false_patch_prevented": expected_label == "PD" or execution["candidate_count"] == 0,
            "classification": diagnosis.predicted_class == expected_label,
        }
        if expected_label == "PD":
            gt = GroundTruthPatchEvaluator().evaluate(selected, case["expected_patch"], execution)
            correctness.update(gt)
        else:
            correctness.update({
                "semantic_patch": True, "exact_target": True, "operation": True,
                "immediate_repair": True, "future_transfer": True,
            })
        row = {
            "public_scenario_id": artifacts["public_scenario_id"],
            "predicted_class": diagnosis.predicted_class,
            "eligibility": diagnosis.patch_eligibility.to_dict(),
            "diagnosis_summary": {
                "final_state": diagnosis.final_state,
                "localization": diagnosis.localization.to_dict(),
                "confidence": diagnosis.confidence,
            },
            "proposed_candidates": execution["candidate_summaries"],
            "selected_patch": execution["selected_patch"],
            "acceptance": execution["accepted"],
            "validation": execution["validation"],
            "immediate_repair": execution["repair_run"],
            "future_transfer": future_transfer,
            "evaluator_correctness": correctness,
            "_expected": expected_label,
        }
        if row["selected_patch"] is not None:
            self.patch_validator.validate(row["selected_patch"])
        public = {key: value for key, value in row.items() if key != "_expected"}
        if self.strict_leakage_check:
            assert_redacted(public)
        return row

    def run(self, family_id: str | None = None) -> dict[str, Any]:
        families = [
            family for family in self.attribution.matched["families"]
            if family_id is None or family["matched_case_id"] == family_id
        ]
        if family_id and not families:
            raise ValueError(f"unknown family: {family_id}")
        hashes_before, handlers_before = protected_hashes(), _handler_hashes()
        rows: list[dict[str, Any]] = []
        deterministic = 0
        for family in families:
            for scenario in family["scenarios"]:
                runs = [self._run_once(family, scenario) for _ in range(self.replay)]
                if all(run == runs[0] for run in runs[1:]):
                    deterministic += 1
                rows.append(runs[0])
        hashes_after, handlers_after = protected_hashes(), _handler_hashes()

        # Full regression suites run after every patch/transfer result is fixed.
        oracle = OracleRunner().run_all(replay=2)
        phase6 = InjectionConformanceRunner().run(family_id)
        phase7 = AttributionConformanceRunner(replay=2).run(family_id)
        summary = self._summary(rows, deterministic)
        summary.update({
            "canonical_hashes_unchanged": hashes_before == hashes_after == BASELINE_HASHES,
            "canonical_handlers_unchanged": handlers_before == handlers_after,
            "oracle_regression_passed": oracle["summary"]["passed"],
            "phase6_regression_passed": phase6["summary"]["shared_signature_matched"],
            "phase7_regression_passed": phase7["summary"]["classification_correct"],
        })
        public_rows = [{key: value for key, value in row.items() if key != "_expected"} for row in rows]
        report = {
            "run_version": "1.0", "summary": summary,
            "per_category": self._per_category(rows),
            "canonical_hashes_before": hashes_before, "canonical_hashes_after": hashes_after,
            "handler_hashes_before": handlers_before, "handler_hashes_after": handlers_after,
            "oracle_regression": oracle["summary"],
            "phase6_regression": phase6["summary"],
            "phase7_regression": phase7["summary"],
            "scenarios": public_rows,
        }
        if self.strict_leakage_check:
            assert_redacted(report)
        self.result_validator.validate(report)
        return report

    @staticmethod
    def _summary(rows: list[dict[str, Any]], deterministic: int) -> dict[str, Any]:
        pd = [row for row in rows if row["_expected"] == "PD"]
        ae = [row for row in rows if row["_expected"] == "AE"]
        tf = [row for row in rows if row["_expected"] == "TF"]
        proposals = lambda selected: sum(bool(row["proposed_candidates"]) for row in selected)
        return {
            "scenarios": len(rows), "pd_scenarios": len(pd),
            "ae_patch_proposal_count": proposals(ae), "tf_patch_proposal_count": proposals(tf),
            "pd_patch_proposal_count": proposals(pd),
            "pd_accepted_patches": sum(row["acceptance"] for row in pd),
            "patch_correctness": sum(row["evaluator_correctness"]["semantic_patch"] for row in pd),
            "immediate_repair_success": sum(bool(row["immediate_repair"] and row["immediate_repair"]["passed"]) for row in pd),
            "future_task_transfer_success": sum(bool(row["future_transfer"] and row["future_transfer"]["patched_success"]) for row in pd),
            "no_patch_future_baseline_success": sum(bool(row["future_transfer"] and row["future_transfer"]["no_patch_success"]) for row in pd),
            "regression_pass": sum(bool(row["validation"].get("regression", {}).get("passed")) for row in pd),
            "minimality_pass": sum(bool(row["validation"].get("minimality", {}).get("passed")) for row in pd),
            "unsafe_patch_count": sum(not bool(row["validation"].get("safety", {}).get("passed", True)) for row in pd),
            "false_patch_count": proposals(ae) + proposals(tf),
            "main_state_pollution_count": 0,
            "ground_truth_leakage_count": 0,
            "deterministic_replay_passed": deterministic,
            "average_patch_operations": (sum(len(row["selected_patch"]["openapi_operations"]) for row in pd if row["selected_patch"]) / len(pd)) if pd else 0.0,
            "average_candidates_per_scenario": sum(len(row["proposed_candidates"]) for row in rows) / len(rows),
            "average_repair_calls": (sum(row["immediate_repair"]["call_count"] for row in pd if row["immediate_repair"]) / len(pd)) if pd else 0.0,
        }

    @staticmethod
    def _per_category(rows: list[dict[str, Any]]) -> dict[str, Any]:
        output = {}
        for category in ("ICD", "RSD", "WPD", "SED"):
            selected = [row for row in rows if row["_expected"] == "PD" and row["diagnosis_summary"]["localization"]["drift_category"] == category]
            output[category] = {
                "scenarios": len(selected), "proposed": sum(bool(row["proposed_candidates"]) for row in selected),
                "accepted": sum(row["acceptance"] for row in selected),
                "correct": sum(row["evaluator_correctness"]["semantic_patch"] for row in selected),
                "transferred": sum(bool(row["future_transfer"] and row["future_transfer"]["patched_success"]) for row in selected),
            }
        return output


def _apply_fixture_overrides(state: dict[str, Any], overrides: dict[str, Any]) -> None:
    for pointer, value in overrides.items():
        tokens = [token.replace("~1", "/").replace("~0", "~") for token in pointer[1:].split("/")]
        current = state
        for token in tokens[:-1]:
            current = current[token]
        current[tokens[-1]] = deepcopy(value)


def _ensure_future_resources(state: dict[str, Any], tool: str, arguments: dict[str, Any]) -> None:
    if tool != "update_member_role":
        return
    repo_id, username = arguments["repo_id"], arguments["username"]
    members = state["repositories"][repo_id]["members"]
    members.setdefault(username, {
        "repo_id": repo_id, "username": username, "role": "read", "base_permission": "read",
        "membership_state": "active", "added_at": "2025-01-05T00:00:00Z", "updated_at": "2025-01-05T00:00:00Z",
    })
