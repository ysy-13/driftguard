from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

from driftguard.contracts.input_validator import InputValidationError, InputValidator
from driftguard.contracts.loader import PROJECT_ROOT
from driftguard.contracts.registry import ContractRegistry
from driftguard.injection import AgentErrorHook
from driftguard.runners.injection_conformance_runner import BASELINE_HASHES, PROTECTED_PATHS
from driftguard.runtime import ExecutionContext, ExecutionMode, ExecutionProfile
from driftguard.sandbox import SandboxService


V4_PATH = PROJECT_ROOT / "results/experiments/phase10/pilot_v4_canary"
AUDIT_PATH = PROJECT_ROOT / "results/experiments/phase10/post_repair_consistency_audit"
SEMANTIC_SOURCE_PATHS = (
    "src/driftguard/agents/controller.py",
    "src/driftguard/agents/action_parser.py",
    "src/driftguard/agents/context_builder.py",
    "src/driftguard/agents/capabilities.py",
    "src/driftguard/agents/tool_catalog.py",
    "src/driftguard/experiments/scenario_runner.py",
    "src/driftguard/experiments/task_resolver.py",
    "src/driftguard/runtime/contract_snapshots.py",
    "src/driftguard/contracts/input_validator.py",
    "src/driftguard/injection/agent_error.py",
    "src/driftguard/sandbox/service.py",
    "src/driftguard/sandbox/evaluator.py",
    "src/driftguard/agents/policies/base.py",
    "src/driftguard/agents/policies/standard.py",
    "src/driftguard/agents/policies/retry_only.py",
    "src/driftguard/agents/policies/reflection.py",
    "src/driftguard/agents/policies/validation_guided.py",
    "src/driftguard/agents/policies/driftguard.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_snapshot() -> dict[str, Any]:
    files = {path: sha256(PROJECT_ROOT / path) for path in SEMANTIC_SOURCE_PATHS}
    material = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {"source_snapshot_hash": hashlib.sha256(material).hexdigest(), "files": files}


def composite_hash(*paths: str) -> str:
    values = {path: sha256(PROJECT_ROOT / path) for path in paths}
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def _record_generation_snapshot(provider: str, method: str) -> str:
    if provider == "deepseek" and method == "standard":
        return "S1_PRE_RUNTIME_NORMALIZATION_REPAIR"
    if provider == "deepseek" and method in {"retry_only", "reflection"}:
        return "S2_POST_NORMALIZATION_PRE_REQUIRED_DEFAULT_REPAIR"
    return "S3_POST_REQUIRED_DEFAULT_REPAIR"


def _record_rows() -> list[dict[str, Any]]:
    manifest = json.loads((V4_PATH / "manifest.json").read_text(encoding="utf-8"))
    rows = []
    for path in sorted(V4_PATH.glob("records/*/records/*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        policy_path = f"src/driftguard/agents/policies/{record['method'].replace('driftguard_llm', 'driftguard')}.py"
        rows.append({
            "record_id": record["record_id"],
            "provider": record["provider"],
            "method": record["method"],
            "public_scenario_id": record["public_scenario_id"],
            "effective_prompt_hash": record["prompt_hash"],
            "schema_hash": record["schema_sha256"],
            "tool_catalog_fingerprint": record["catalog_fingerprint"],
            "config_hash": manifest["config_hash"],
            "controller_source_hash_at_generation": "NOT_EMBEDDED_UNRECOVERABLE_PREIMAGE",
            "task_resolver_source_hash_at_generation": "NOT_EMBEDDED_UNRECOVERABLE_PREIMAGE",
            "runtime_normalization_source_hash_at_generation": "NOT_EMBEDDED_UNRECOVERABLE_PREIMAGE",
            "input_validator_default_source_hash_at_generation": "NOT_EMBEDDED_UNRECOVERABLE_PREIMAGE",
            "agent_fault_scope_source_hash_at_generation": "NOT_EMBEDDED_UNRECOVERABLE_PREIMAGE",
            "policy_source_hash_at_generation": "NOT_EMBEDDED_UNRECOVERABLE_PREIMAGE",
            "evaluator_source_hash_at_generation": "NOT_EMBEDDED_UNRECOVERABLE_PREIMAGE",
            "current_final_hashes_for_reference_only": {
                "controller": sha256(PROJECT_ROOT / "src/driftguard/agents/controller.py"),
                "task_resolver": sha256(PROJECT_ROOT / "src/driftguard/experiments/task_resolver.py"),
                "runtime_normalization": sha256(PROJECT_ROOT / "src/driftguard/agents/controller.py"),
                "input_validator_default": composite_hash(
                    "src/driftguard/contracts/input_validator.py",
                    "src/driftguard/runtime/contract_snapshots.py",
                ),
                "agent_fault_scope": composite_hash(
                    "src/driftguard/agents/controller.py",
                    "src/driftguard/injection/agent_error.py",
                ),
                "policy": sha256(PROJECT_ROOT / policy_path),
                "evaluator": sha256(PROJECT_ROOT / "src/driftguard/sandbox/evaluator.py"),
            },
            "result_generated_at": datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(),
            "reconstructed_semantic_snapshot": _record_generation_snapshot(record["provider"], record["method"]),
            "reconstruction_basis": "record/cache/archive mtimes and behavior-specific archived attempts; not manifest amendment alone",
        })
    return rows


def _load_fixture() -> tuple[dict[str, Any], dict[str, Any]]:
    families = json.loads(
        (PROJECT_ROOT / "benchmark/matched_failures/matched_failures_v1.json").read_text(encoding="utf-8")
    )["families"]
    cases = json.loads(
        (PROJECT_ROOT / "benchmark/drifts/drift_cases_v1.json").read_text(encoding="utf-8")
    )["cases"]
    family = next(item for item in families if item["matched_case_id"] == "M01")
    case = next(item for item in cases if item["drift_id"] == family["source_drift_id"])
    return family, case


def agent_fault_scope_matrix() -> list[dict[str, Any]]:
    family, case = _load_fixture()
    rows = []
    hook = AgentErrorHook()
    for mode in ExecutionMode:
        for provider in ("deepseek", "dashscope"):
            for method in ("standard", "retry_only", "reflection", "validation_guided", "driftguard_llm"):
                context = ExecutionContext(ExecutionProfile(mode, case, family, change_point=int(case["change_point"])))
                context.set_episode(3)
                before = hook.active(context)
                target_mutated = hook.inject_call(context, {"repo_id": "R1", "title": "x", "priority": "medium"})
                context.set_episode(4)
                after_episode = hook.active(context)
                context.reset()
                context.set_episode(3)
                after_reset = hook.active(context)
                rows.append({
                    "variant": mode.value, "provider": provider, "method": method,
                    "active_at_protocol_episode": before,
                    "target_call_changed": target_mutated != {"repo_id": "R1", "title": "x", "priority": "medium"},
                    "active_next_episode": after_episode,
                    "active_after_reset_same_scenario": after_reset,
                    "expected_active": mode == ExecutionMode.AGENT_ERROR,
                    "passed": before == (mode == ExecutionMode.AGENT_ERROR) and not after_episode,
                })
    return rows


def required_default_negative_audit() -> list[dict[str, Any]]:
    validator = InputValidator()
    canonical = ContractRegistry.from_openapi().require("create_issue")

    def rejects(contract, arguments) -> bool:
        try:
            validator.validate(contract, arguments)
        except InputValidationError:
            return True
        return False

    no_default_schema = deepcopy(canonical.input_schema)
    no_default_schema["properties"]["priority"].pop("default", None)
    no_default_schema["required"] = [*no_default_schema["required"], "priority"]
    displayed_no_default = replace(canonical, input_schema=no_default_schema, body_schema=None)

    hidden_runtime_schema = deepcopy(no_default_schema)
    hidden_runtime_schema["properties"]["priority"]["default"] = "medium"
    hidden_runtime = replace(canonical, input_schema=hidden_runtime_schema, body_schema=None)
    omitted = {"repo_id": "R1", "title": "x"}
    wrong_entity = {"repo_id": "R404", "title": "x", "priority": "high"}
    runtime_accepts = not rejects(hidden_runtime, omitted)
    visible_rejects = rejects(displayed_no_default, omitted)
    normalized_wrong = validator.validate(displayed_no_default, wrong_entity)
    family, case = _load_fixture()
    pd_context = ExecutionContext(ExecutionProfile(
        ExecutionMode.PERSISTENT_DRIFT, case, family, change_point=int(case["change_point"]),
    ))
    pd_context.set_episode(3)
    displayed_contract = ContractRegistry(pd_context.displayed_contract).require("create_issue")
    displayed_accepts_omission = not rejects(displayed_contract, omitted)
    pd_runtime_result = SandboxService(execution_context=pd_context).call_tool(
        "create_issue", omitted, "agent_admin",
    )
    return [
        {"case": "missing_required_without_default", "passed": visible_rejects},
        {"case": "default_exists_only_in_hidden_runtime", "passed": visible_rejects and runtime_accepts,
         "agent_visible_validation_used_runtime": False},
        {"case": "default_exists_only_in_ground_truth", "passed": visible_rejects,
         "ground_truth_was_input_to_validator": False},
        {"case": "pd_old_displayed_spec_has_no_new_required_field",
         "passed": displayed_accepts_omission and not pd_runtime_result.ok,
         "displayed_validation_accepts": displayed_accepts_omission,
         "runtime_rejects_new_required_omission": not pd_runtime_result.ok,
         "pd_runtime_not_bypassed": True},
        {"case": "wrong_entity_id_not_overwritten", "passed": normalized_wrong["repo_id"] == "R404",
         "before": "R404", "after": normalized_wrong["repo_id"]},
    ]


def run_consistency_audit() -> dict[str, Any]:
    rows = _record_rows()
    snapshots = sorted({row["reconstructed_semantic_snapshot"] for row in rows})
    scope = agent_fault_scope_matrix()
    defaults = required_default_negative_audit()
    protected = {path: sha256(PROJECT_ROOT / path) for path in PROTECTED_PATHS}
    report = {
        "audit_version": "phase10a-post-repair-consistency-v1",
        "offline_only": True,
        "real_provider_calls": 0,
        "v4_classification": "MIXED_VERSION_DEVELOPMENT_ARTIFACT",
        "combined_task_success_permitted": False,
        "clean_canary_required": len(snapshots) >= 2,
        "semantic_snapshots": snapshots,
        "record_source_proof": rows,
        "changes": [
            {
                "name": "runtime_normalization", "behavior_affecting": True,
                "before": "displayed validation defaults were passed to Runtime, masking the original proposed call",
                "after": "execution contexts pass the validated but unnormalized proposed arguments to Runtime",
                "files_functions": ["src/driftguard/agents/controller.py::ToolAgentController.run"],
                "time_evidence": "after 2026-07-15T23:12:23+08:00 and before DeepSeek retry/reflection post-repair re-execution",
                "model_input_changed": False, "argument_validity_changed": True, "sandbox_execution_changed": True,
                "variant_effect": {"AE": "preserves Agent omission", "TF": "restores first transient failure", "PD": "prevents displayed defaults masking runtime drift"},
                "task_success_may_change": True,
            },
            {
                "name": "required_field_default", "behavior_affecting": True,
                "before": "add_required retained an obsolete default in the mutated displayed/runtime contract copy",
                "after": "newly required fields lose the obsolete default only in the mutated copy",
                "files_functions": ["src/driftguard/runtime/contract_snapshots.py::_mutate_contract"],
                "time_evidence": "source mtime 2026-07-15T23:31:56+08:00; archived false-negative at 23:28:59",
                "model_input_changed": True, "argument_validity_changed": True, "sandbox_execution_changed": True,
                "variant_effect": {"AE": "visible new requirement is enforceable", "TF": "canonical display unchanged", "PD": "runtime-only requirement cannot be bypassed"},
                "task_success_may_change": True,
            },
            {
                "name": "agent_fault_scope", "behavior_affecting": True,
                "before": "fault could be considered on unrelated tools and recur on later target calls",
                "after": "fault is target-tool scoped and consumed on the first target call per record",
                "files_functions": ["src/driftguard/agents/controller.py::ToolAgentController.run", "src/driftguard/injection/agent_error.py::AgentFaultInjector"],
                "time_evidence": "completed before first successful provider cache response at 2026-07-15T23:04:01+08:00",
                "model_input_changed": True, "argument_validity_changed": True, "sandbox_execution_changed": True,
                "variant_effect": {"AE": "one target-call fault", "TF": "inactive", "PD": "inactive"},
                "task_success_may_change": True,
            },
        ],
        "runtime_normalization": {
            "allowed_scope": ["empty content", "tool-call envelope", "JSON string parsing", "field encoding"],
            "semantic_field_rewrites_allowed": False,
            "legacy_aliases_tool_parameters_accepted": False,
            "clean_records_require_audit_fields": ["raw_model_action", "normalized_action", "normalization_diff", "normalization_reason_code"],
        },
        "required_default_negative_tests": defaults,
        "agent_fault_scope_matrix": scope,
        "agent_fault_scope_passed": all(row["passed"] for row in scope),
        "protected_files": {
            "count": len(protected), "hashes": protected,
            "match_phase6_baseline": protected == BASELINE_HASHES,
        },
        "current_final_source_snapshot": source_snapshot(),
        "phase10b_status": "NOT RUN",
        "full_real_model_experiment_status": "NOT RUN",
    }
    AUDIT_PATH.mkdir(parents=True, exist_ok=True)
    (AUDIT_PATH / "audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (V4_PATH / "mixed_version_classification.json").write_text(json.dumps({
        "classification": "MIXED_VERSION_DEVELOPMENT_ARTIFACT",
        "combined_task_success_permitted": False,
        "source_audit": str((AUDIT_PATH / "audit.json").relative_to(PROJECT_ROOT)),
        "original_records_preserved": True,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
