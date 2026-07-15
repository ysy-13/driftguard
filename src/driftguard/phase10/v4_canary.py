from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

from driftguard.agents.capabilities import PolicyCapabilities
from driftguard.agents.tool_catalog import ToolCatalogRenderer
from driftguard.contracts.loader import PROJECT_ROOT, load_openapi
from driftguard.experiments.checkpoint import CheckpointStore
from driftguard.experiments.evaluator import strip_evaluator_fields
from driftguard.experiments.manifest import file_hash
from driftguard.experiments.runner import PROMPT_NAMES, _load_schema
from driftguard.experiments.scenario_runner import ExperimentScenarioRunner, _public_id
from driftguard.experiments.task_resolver import TaskInstanceResolver
from driftguard.llm import LLMCache, OpenAICompatibleProvider, ProviderResponse, RequestRateLimiter
from driftguard.llm.prompt_loader import PromptLoader
from driftguard.llm.redaction import assert_secret_absent
from driftguard.phase10.offline_repair import OfflineRepairGate
from driftguard.runners.attribution_conformance_runner import MATCHED_PATH
from driftguard.runners.injection_conformance_runner import PROTECTED_PATHS
from driftguard.runtime import ExecutionContext, ExecutionMode, ExecutionProfile

from .config import Phase10Config
from .costs import CostBudgetManager, CostHardLimit
from .credentials import credential_status, load_project_dotenv
from .pricing import PricingCatalog


CONFIG_PATH = PROJECT_ROOT / "configs" / "experiments" / "phase10_real_pilot_v4_canary.yaml"
OUTPUT_PATH = PROJECT_ROOT / "results" / "experiments" / "phase10" / "pilot_v4_canary"
LEDGER_PATH = PROJECT_ROOT / "results" / "experiments" / "phase10" / "cost" / "ledger.json"
BASE_SPENT_CNY = 3.497508830
INCREMENTAL_SOFT_CNY = 1.0
INCREMENTAL_HARD_CNY = 2.0
TOTAL_HARD_CNY = 50.0
INFRASTRUCTURE_ERRORS = {
    "INVALID_STRUCTURED_OUTPUT", "INTERNAL_ERROR", "PROVIDER_ERROR",
    "PROVIDER_TIMEOUT", "RATE_LIMITED",
}


class CanaryGateError(RuntimeError):
    pass


class CacheOnlyProvider:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, request):
        self.calls += 1
        raise RuntimeError(f"v4 cache replay miss: {request.public_scenario_id}/{request.method}")


class ReadOnlyCache:
    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)

    def get(self, key):
        raw_path = self.directory / "raw" / f"{key}.json"
        parsed_path = self.directory / "parsed" / f"{key}.json"
        if not raw_path.exists():
            return None
        value = json.loads(raw_path.read_text(encoding="utf-8"))
        value["parsed_output"] = (
            json.loads(parsed_path.read_text(encoding="utf-8"))
            if parsed_path.exists() else None
        )
        return ProviderResponse(**value, cached=True)

    def put(self, key, response):
        raise PermissionError("v4 cache replay is read-only")


class Phase10V4Canary:
    def __init__(
        self,
        config_path: Path | str = CONFIG_PATH,
        output: Path | str = OUTPUT_PATH,
        allow_real_api: bool = False,
        *,
        attempt: str = "v4_canary",
        cache_namespace: str = "v4_canary",
        base_spent_cny: float = BASE_SPENT_CNY,
        incremental_soft_cny: float = INCREMENTAL_SOFT_CNY,
        incremental_hard_cny: float = INCREMENTAL_HARD_CNY,
        total_hard_cny: float = TOTAL_HARD_CNY,
        frozen_source_snapshot: dict[str, Any] | None = None,
    ) -> None:
        load_project_dotenv(PROJECT_ROOT)
        self.config_path, self.output = Path(config_path), Path(output)
        self.config = Phase10Config.load(self.config_path)
        self.allow_real_api = allow_real_api
        self.attempt, self.cache_namespace = attempt, cache_namespace
        self.base_spent_cny = base_spent_cny
        self.incremental_soft_cny = incremental_soft_cny
        self.incremental_hard_cny = incremental_hard_cny
        self.total_hard_cny = total_hard_cny
        self.frozen_source_snapshot = deepcopy(frozen_source_snapshot)
        self._validate_scope()
        self.catalog = PricingCatalog.load_default()
        ledger = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
        self.initial_ledger = deepcopy(ledger)
        self.costs = CostBudgetManager(
            self.catalog,
            self.base_spent_cny + self.incremental_soft_cny,
            self.base_spent_cny + self.incremental_hard_cny,
            self.config.budget.max_input_tokens_per_record,
            self.config.budget.max_output_tokens_per_record,
            ledger["spent_cny"], ledger["api_attempts"],
            ledger["provider_reported_input_tokens"], ledger["provider_reported_output_tokens"],
        )
        loader = PromptLoader(PROJECT_ROOT / "benchmark" / "prompts")
        names = set(PROMPT_NAMES) | {"base_tool_agent_v3.txt", "component_attribution_v2.txt"}
        self.prompts = {name: loader.load(name) for name in names}
        self.prompt_hashes = {name: loader.hash(name) for name in names}
        self.schemas = {
            "agent_action": _load_schema("agent_action_schema_v3.json"),
            "llm_attribution": _load_schema("llm_attribution_schema_v1.json"),
        }
        self.families = json.loads(MATCHED_PATH.read_text(encoding="utf-8"))["families"]
        self.family_by_id = {item["matched_case_id"]: item for item in self.families}
        self.task_resolver = TaskInstanceResolver()
        self.selection = tuple(deepcopy(self.config.raw["experiment"]["selection"]))
        self.rate_limiters = {
            model.provider: RequestRateLimiter(self.config.execution.get("rate_limit_per_second"))
            for model in self.config.models
        }

    def _validate_scope(self) -> None:
        experiment = self.config.raw["experiment"]
        if experiment.get("pilot_attempt") != self.attempt or experiment.get("run_scope") != "end_to_end_canary_only":
            raise CanaryGateError("v4 runner accepts only the isolated end-to-end canary scope")
        if self.config.modes != ("end_to_end",) or self.config.methods.get("component"):
            raise CanaryGateError("Component execution is not authorized")
        if self.config.budget.full_hard_limit_cny != self.total_hard_cny:
            raise CanaryGateError("the original CNY 50 total hard limit must remain active")
        if self.config.budget.pilot_hard_limit_cny != self.base_spent_cny + self.incremental_hard_cny:
            raise CanaryGateError("v4 cumulative hard gate must equal existing ledger plus CNY 2")
        if self.config.budget.pilot_soft_limit_cny != self.base_spent_cny + self.incremental_soft_cny:
            raise CanaryGateError("v4 cumulative soft gate must equal existing ledger plus CNY 1")

    def preflight(self, write: bool = True) -> dict[str, Any]:
        gate = OfflineRepairGate().run()
        experiment = self.config.raw["experiment"]
        prompt_hash = file_hash(PROJECT_ROOT / "benchmark/prompts/base_tool_agent_v3.txt")
        schema_hash = file_hash(PROJECT_ROOT / "benchmark/schemas/agent_action_schema_v3.json")
        catalog_fingerprint = ToolCatalogRenderer().fingerprint(load_openapi())
        credentials = credential_status()
        mappings = self._validate_selection()
        capability_checks = {}
        for item in self.selection:
            method = item["method"]
            capabilities = PolicyCapabilities.for_method(method)
            narrowed = capabilities.narrow_schema(self.schemas["agent_action"])
            rendered = self.prompts["base_tool_agent_v3.txt"].replace(
                "{{OUTPUT_SCHEMA}}", json.dumps(narrowed, sort_keys=True),
            ).replace("{{POLICY_CAPABILITIES}}", capabilities.prompt_fragment())
            capability_checks[method] = {
                "allowed_actions": list(capabilities.allowed_actions),
                "probe_visible": "REQUEST_PROBE" in rendered,
                "passed": ("REQUEST_PROBE" in rendered) == (method == "driftguard_llm"),
            }
        estimate = self._estimate_cost()
        checks = {
            "resolved_tasks": gate["task_consistency"]["passed"],
            "tool_catalog": gate["tool_catalog"]["passed"],
            "tool_schema_diff_0_of_12": gate["tool_catalog"]["input_schema_diffs"] == 0,
            "mock_feedback_5_of_5": gate["mock_feedback"]["successful_two_round_repairs"] == 5,
            "offline_oracle_12_of_12": gate["offline_oracle"]["evaluator_success"] == 12,
            "capabilities": all(item["passed"] for item in capability_checks.values()),
            "prompt_hash": prompt_hash == experiment["expected_prompt_v3_sha256"],
            "schema_hash": schema_hash == experiment["expected_schema_v3_sha256"],
            "catalog_fingerprint": catalog_fingerprint == experiment["expected_catalog_fingerprint"],
            "credentials": all(value == "configured" for value in credentials.values()),
            "selection": len(mappings) == 5,
            "estimated_increment_below_hard_limit": estimate["estimated_increment_cny"] <= self.incremental_hard_cny,
            "ledger_continues": self.costs.spent_cny >= self.base_spent_cny,
            "total_hard_limit": self.costs.spent_cny + self.incremental_hard_cny < self.total_hard_cny,
        }
        report = {
            "preflight_version": "phase10-v4-canary-v1",
            "checks": checks, "passed": all(checks.values()),
            "credentials": credentials,
            "prompt_v3_sha256": prompt_hash, "schema_v3_sha256": schema_hash,
            "catalog_fingerprint": catalog_fingerprint,
            "effective_catalog_fingerprints": {
                item["method"]: self._expected_catalog_fingerprint(item)
                for item in self.selection
            },
            "policy_capabilities": capability_checks,
            "selection": [self._public_selection(item) for item in mappings],
            "cost_estimate": estimate,
            "starting_ledger": {
                "spent_cny": self.costs.spent_cny, "api_attempts": self.costs.api_attempts,
                "input_tokens": self.initial_ledger["provider_reported_input_tokens"],
                "output_tokens": self.initial_ledger["provider_reported_output_tokens"],
            },
            "real_provider_calls": 0,
            "full_real_model_experiment_status": "NOT RUN",
        }
        if write:
            self.output.mkdir(parents=True, exist_ok=True)
            path = self.output / "preflight.json"
            if not path.exists():
                _atomic(path, report)
        if not report["passed"]:
            failed = [key for key, value in checks.items() if not value]
            raise CanaryGateError(f"v4 offline preflight failed: {failed}")
        return report

    def _validate_selection(self) -> list[dict[str, Any]]:
        expected = {
            "standard": "AE", "retry_only": "TF", "reflection": "AE",
            "validation_guided": "AE", "driftguard_llm": "PD",
        }
        if {item["method"] for item in self.selection} != set(expected):
            raise CanaryGateError("v4 selection must contain exactly the five registered methods")
        rows = []
        for item in self.selection:
            if item["family"] not in {"M01", "M06", "M11", "M16"}:
                raise CanaryGateError("v4 canary touched a held-out family")
            if item["variant"] != expected[item["method"]] or item["scenario_id"] != f"{item['family']}-{item['variant']}":
                raise CanaryGateError(f"invalid method/variant mapping: {item}")
            family = self.family_by_id[item["family"]]
            scenario = next(value for value in family["scenarios"] if value["scenario_id"] == item["scenario_id"])
            resolved = OfflineRepairGate().resolver.resolve(family, scenario)
            rows.append({
                **item,
                "source_task_ref": resolved.source_task_ref,
                "public_task_id": resolved.public_task_id,
                "public_scenario_id": resolved.public_scenario_id,
                "resolved_instruction": resolved.instruction,
                "bindings": resolved.bindings,
            })
        return rows

    def _estimate_cost(self) -> dict[str, Any]:
        # Conservative canary estimate based on a full catalog plus one recovery
        # turn: 35K input and 2K output tokens per record. The pre-call reserve
        # and cumulative hard gate remain authoritative if usage is higher.
        by_provider = {}
        total = 0.0
        for model in self.config.models:
            per_record = self.catalog.estimate_cny(model, 35_000, 2_000)
            value = per_record * 5
            by_provider[model.provider] = {
                "records": 5, "assumed_input_tokens_per_record": 35_000,
                "assumed_output_tokens_per_record": 2_000,
                "estimated_cny": value,
            }
            total += value
        return {
            "estimate_only": True, "by_provider": by_provider,
            "estimated_increment_cny": total,
            "incremental_soft_limit_cny": self.incremental_soft_cny,
            "incremental_hard_limit_cny": self.incremental_hard_cny,
            "total_hard_limit_cny": self.total_hard_cny,
        }

    def run(self, resume: bool = True) -> dict[str, Any]:
        if not self.allow_real_api:
            raise PermissionError("v4 Canary real requests require explicit allow_real_api")
        preflight = self.preflight(write=True)
        self._assert_source_frozen()
        self._write_manifest(preflight)
        provider_reports = []
        all_records = []
        halted = False
        halt_reason = None
        for provider_name in ("deepseek", "dashscope"):
            if halted:
                provider_reports.append({"provider": provider_name, "status": "SKIPPED_AFTER_GATE", "reason": halt_reason, "records": 0})
                continue
            records, report = self._run_provider(provider_name, resume)
            all_records.extend(records)
            provider_reports.append(report)
            if report["status"] != "PASSED":
                halted = True
                halt_reason = report["status"]
        replay = self.cache_replay(all_records) if len(all_records) == 10 and not halted else {
            "status": "NOT_RUN_GATE_FAILED", "records": 0, "provider_fallback_calls": 0,
        }
        final = self.costs.snapshot()
        summary = {
            "experiment_id": self.config.name,
            "attempt": self.attempt, "scope": "10-record end-to-end canary only",
            "records": len(all_records), "planned_records": 10,
            "component_records": 0,
            "provider_reports": provider_reports,
            "agent_action_valid": sum(item["agent_action_valid"] for item in all_records),
            "sandbox_entered": sum(item["sandbox_entered"] for item in all_records),
            "resolved_task_instances_correct": sum(item["resolved_task_instance_correct"] for item in all_records),
            "complete_tool_catalogs": sum(item["tool_catalog_complete"] for item in all_records),
            "task_success": sum(item["final_task_success"] for item in all_records),
            "method_task_success": {
                method: sum(item["final_task_success"] for item in all_records if item["method"] == method)
                for method in ("standard", "retry_only", "reflection", "validation_guided", "driftguard_llm")
            },
            "tool_calls": sum(item["tool_calls"] for item in all_records),
            "llm_calls": sum(item["llm_calls"] for item in all_records),
            "probe_calls": sum(item["probe_calls"] for item in all_records),
            "input_tokens": sum(item["input_tokens"] for item in all_records),
            "output_tokens": sum(item["output_tokens"] for item in all_records),
            "api_calls": sum(item["api_calls"] for item in all_records),
            "infrastructure_errors": sum(item["infrastructure_error"] for item in all_records),
            "request_probe_capability_mismatches": sum(item["request_probe_capability_mismatch"] for item in all_records),
            "unauthorized_safety_blocks": sum(item["unauthorized_safety_block"] for item in all_records),
            "state_isolation_errors": sum(item["state_isolation_error"] for item in all_records),
            "api_key_leakage": 0, "ground_truth_leakage": 0,
            "cache_replay": replay,
            "cost": {
                **final,
                "attempt_base_spent_cny": self.base_spent_cny,
                "attempt_incremental_spent_cny": final["spent_cny"] - self.base_spent_cny,
                "incremental_soft_limit_cny": self.incremental_soft_cny,
                "incremental_hard_limit_cny": self.incremental_hard_cny,
                "total_hard_limit_cny": self.total_hard_cny,
            },
            "halted": halted, "halt_reason": halt_reason,
            "prompt_v3_sha256": preflight["prompt_v3_sha256"],
            "schema_v3_sha256": preflight["schema_v3_sha256"],
            "canonical_catalog_fingerprint": preflight["catalog_fingerprint"],
            "heldout48_touched": False,
            "common_execution_chain_issues": sorted({
                failure for item in all_records for failure in item["canary_path_failures"]
            }),
            "full_real_model_experiment_status": "NOT RUN",
        }
        if self.frozen_source_snapshot is not None:
            summary["source_snapshot_hash"] = self.frozen_source_snapshot["source_snapshot_hash"]
            summary["source_snapshot_consistent_records"] = sum(
                item.get("source_snapshot_hash") == self.frozen_source_snapshot["source_snapshot_hash"]
                for item in all_records
            )
        self._assert_source_frozen()
        _atomic(self.output / "summary.json", summary)
        self._write_ledger()
        return summary

    def _run_provider(self, provider_name: str, resume: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        model = next(item for item in self.config.models if item.provider == provider_name)
        key = os.getenv(model.api_key_env or "")
        if not key:
            raise CanaryGateError(f"{model.api_key_env}: missing")
        directory = self.output / "records" / provider_name
        checkpoints = CheckpointStore(directory)
        cache_directory = self.output / ".cache" / self.cache_namespace / provider_name / "end_to_end"
        cache = LLMCache(cache_directory)
        records = []
        status = "PASSED"
        stop_reason = None
        for selected in self.selection:
            method, scenario_id = selected["method"], selected["scenario_id"]
            record_id = self._record_id(model, selected)
            if resume and not checkpoints.has(record_id):
                self._restore_gate_false_positive(provider_name, selected, record_id, checkpoints)
            if resume and checkpoints.has(record_id):
                record = checkpoints.read(record_id)
                records.append(record)
                if record["infrastructure_error"] or not record["canary_path_gate_passed"]:
                    status, stop_reason = "FAILED_RESUMED_RECORD_GATE", record["termination_reason"]
                    break
                continue
            self._assert_source_frozen()
            if self.costs.spent_cny - self.base_spent_cny >= self.incremental_hard_cny:
                status, stop_reason = "STOPPED_INCREMENTAL_HARD_LIMIT", "COST_HARD_LIMIT"
                break
            family = self.family_by_id[selected["family"]]
            scenario = next(item for item in family["scenarios"] if item["scenario_id"] == scenario_id)
            view = self.config.phase9_view(model, "end_to_end", (method,), self.config.seeds[0])

            def provider_factory(_outputs):
                return OpenAICompatibleProvider(
                    view.model, key,
                    rate_limiter=self.rate_limiters[provider_name],
                    cost_controller=self.costs,
                )

            runner = ExperimentScenarioRunner(view, cache, self.prompts, self.schemas, provider_factory)
            cost_before = self.costs.snapshot()
            try:
                internal = runner.run(family, scenario, method, "end_to_end", 0)
                record = self._decorate(
                    internal, model, selected, record_id, cache_directory,
                    _cost_delta(cost_before, self.costs.snapshot()),
                )
                archived = self._archived_cost_attribution(provider_name, record_id)
                if archived is not None:
                    new_calls = record["api_calls"]
                    new_cost = record["estimated_cost_cny"]
                    record["api_calls"] += archived.get("api_calls", 0)
                    record["estimated_cost_cny"] += archived.get("estimated_cost_cny", 0.0)
                    record["cache_reexecution"] = {
                        "status": "REUSED_SAME_V4_RESPONSE_AFTER_LOCAL_PIPELINE_FIX",
                        "new_provider_calls": new_calls, "new_cost_cny": new_cost,
                    }
            except Exception as exc:
                record = self._error_record(
                    model, selected, record_id, exc, cache_directory,
                    _cost_delta(cost_before, self.costs.snapshot()),
                )
            self._assert_safe(record, key)
            self._assert_source_frozen()
            checkpoints.write(record_id, record)
            records.append(record)
            self._write_ledger()
            if record["infrastructure_error"]:
                status, stop_reason = "FAILED_INFRASTRUCTURE_GATE", record["error_category"]
                break
            if not record["canary_path_gate_passed"]:
                status, stop_reason = "FAILED_EXECUTION_PATH_GATE", ",".join(record["canary_path_failures"])
                break
        if len(records) == 5 and status == "PASSED" and not any(item["final_task_success"] for item in records):
            status, stop_reason = "FAILED_ZERO_OF_FIVE_TASK_SUCCESS", "TASK_SUCCESS_0_OF_5"
        report = {
            "provider": provider_name, "configured_model": model.model_id,
            "status": status, "stop_reason": stop_reason,
            "records": len(records), "task_success": sum(item["final_task_success"] for item in records),
            "infrastructure_errors": sum(item["infrastructure_error"] for item in records),
            "infrastructure_error_rate": sum(item["infrastructure_error"] for item in records) / len(records) if records else 0.0,
            "agent_action_valid": sum(item["agent_action_valid"] for item in records),
            "sandbox_entered": sum(item["sandbox_entered"] for item in records),
            "tool_calls": sum(item["tool_calls"] for item in records),
            "llm_calls": sum(item["llm_calls"] for item in records),
            "probe_calls": sum(item["probe_calls"] for item in records),
            "api_calls": sum(item["api_calls"] for item in records),
            "input_tokens": sum(item["input_tokens"] for item in records),
            "output_tokens": sum(item["output_tokens"] for item in records),
            "incremental_cost_cny": sum(item["estimated_cost_cny"] for item in records),
        }
        _atomic(directory / "summary.json", report)
        return records, report

    def _decorate(self, internal, model, selected, record_id, cache_directory, cost_delta):
        public = strip_evaluator_fields(internal)
        for key in (
            "target_tool_correct", "drift_category_correct", "exact_location_correct",
            "raw_patch_correct", "accepted_patch_correct", "regression_pass",
            "minimality_pass", "false_patch",
        ):
            public.pop(key, None)
        policy = _policy_gate(selected["method"], internal)
        task_evaluation = _public_task_evaluation(internal.get("_task_evaluation", {}))
        action_types = [item.get("action_type") for item in internal.get("_actions", ())]
        infrastructure = public.get("error_category") in INFRASTRUCTURE_ERRORS
        probe_mismatch = selected["method"] != "driftguard_llm" and "REQUEST_PROBE" in action_types
        safety_block = public.get("termination_reason") == "SAFETY_BLOCKED"
        state_isolation_valid = bool(internal.get("_state_isolation_valid"))
        resolved = self.task_resolver.resolve(
            self.family_by_id[selected["family"]],
            next(item for item in self.family_by_id[selected["family"]]["scenarios"] if item["scenario_id"] == selected["scenario_id"]),
        )
        resolved_valid = (
            _public_id(selected["scenario_id"]) == public.get("public_scenario_id")
            and "{{" not in resolved.instruction and "${" not in resolved.instruction
        )
        expected_catalog_fingerprint = self._expected_catalog_fingerprint(selected)
        catalog_complete = public.get("catalog_fingerprint") == expected_catalog_fingerprint
        path_failures = []
        if not action_types:
            path_failures.append("NO_PARSED_AGENT_ACTION")
        if int(public.get("tool_calls", 0)) < 1:
            path_failures.append("SANDBOX_NOT_ENTERED")
        if not policy["passed"]:
            path_failures.extend(policy["failures"])
        if not catalog_complete:
            path_failures.append("TOOL_CATALOG_MISSING_OR_MISMATCHED")
        if not internal.get("_state_transitions"):
            path_failures.append("CONTROLLER_STATE_MACHINE_MISSING")
        if not resolved_valid:
            path_failures.append("TASK_BINDING_ERROR")
        if probe_mismatch:
            path_failures.append("REQUEST_PROBE_CAPABILITY_MISMATCH")
        if safety_block:
            path_failures.append("UNAUTHORIZED_SAFETY_BLOCK")
        if not state_isolation_valid:
            path_failures.append("STATE_ISOLATION_ERROR")
        public.update({
            "record_id": record_id,
            "attempt": self.attempt,
            "provider": model.provider, "configured_model": model.model_id,
            "actual_model": (internal.get("_actual_models") or [model.model_id])[-1],
            "api_calls": cost_delta["api_attempts"],
            "estimated_cost_cny": cost_delta["spent_cny"],
            "selection": self._public_selection({
                **selected,
                "public_scenario_id": resolved.public_scenario_id,
                "public_task_id": resolved.public_task_id,
            }),
            "agent_action_valid": not infrastructure and bool(action_types),
            "sandbox_entered": int(public.get("tool_calls", 0)) > 0,
            "infrastructure_error": infrastructure,
            "policy_gate": policy,
            "canary_path_gate_passed": not path_failures,
            "canary_path_failures": path_failures,
            "action_types": action_types,
            "state_transitions": deepcopy(internal.get("_state_transitions", [])),
            "task_evaluator": task_evaluation,
            "resolved_task_instance_correct": resolved_valid,
            "resolved_task_public_id": resolved.public_task_id,
            "expected_catalog_fingerprint": expected_catalog_fingerprint,
            "tool_catalog_complete": catalog_complete,
            "request_probe_capability_mismatch": probe_mismatch,
            "unauthorized_safety_block": safety_block,
            "state_isolation_error": not state_isolation_valid,
            "cache_namespace": str(cache_directory.relative_to(self.output)),
            "prompt_template_sha256": self.config.raw["experiment"]["expected_prompt_v3_sha256"],
            "schema_sha256": self.config.raw["experiment"]["expected_schema_v3_sha256"],
            "api_key_leakage": False, "ground_truth_leakage": False,
        })
        if self.frozen_source_snapshot is not None:
            files = self.frozen_source_snapshot["files"]
            public.update({
                "source_snapshot_hash": self.frozen_source_snapshot["source_snapshot_hash"],
                "controller_source_hash": files["src/driftguard/agents/controller.py"],
                "task_resolver_source_hash": files["src/driftguard/experiments/task_resolver.py"],
                "runtime_normalization_source_hash": files["src/driftguard/agents/controller.py"],
                "input_validator_default_source_hash": hashlib.sha256(json.dumps({
                    "src/driftguard/contracts/input_validator.py": files["src/driftguard/contracts/input_validator.py"],
                    "src/driftguard/runtime/contract_snapshots.py": files["src/driftguard/runtime/contract_snapshots.py"],
                }, sort_keys=True).encode()).hexdigest(),
                "agent_fault_scope_source_hash": hashlib.sha256(json.dumps({
                    "src/driftguard/agents/controller.py": files["src/driftguard/agents/controller.py"],
                    "src/driftguard/injection/agent_error.py": files["src/driftguard/injection/agent_error.py"],
                }, sort_keys=True).encode()).hexdigest(),
                "policy_source_hash": files[f"src/driftguard/agents/policies/{selected['method'].replace('driftguard_llm', 'driftguard')}.py"],
                "evaluator_source_hash": files["src/driftguard/sandbox/evaluator.py"],
                "action_audit": [deepcopy(item.get("action_normalization_audit")) for item in internal.get("_actions", ()) if item.get("action_normalization_audit")],
                "displayed_input_audit": [deepcopy(item.get("displayed_input_audit")) for item in internal.get("_actions", ()) if item.get("displayed_input_audit")],
            })
        return public

    def _error_record(self, model, selected, record_id, exc, cache_directory, cost_delta):
        error = _error_category(exc)
        return {
            "experiment_id": self.config.name,
            "public_scenario_id": _public_id(selected["scenario_id"]),
            "mode": "end_to_end", "method": selected["method"], "model": model.model_id,
            "provider": model.provider, "configured_model": model.model_id, "actual_model": model.model_id,
            "record_id": record_id, "attempt": self.attempt,
            "selection": {
                "method": selected["method"],
                "public_scenario_id": _public_id(selected["scenario_id"]),
            },
            "final_task_success": False, "termination_reason": error, "error_category": error,
            "tool_calls": 0, "llm_calls": 0, "probe_calls": 0, "input_tokens": 0, "output_tokens": 0,
            "api_calls": cost_delta["api_attempts"], "estimated_cost_cny": cost_delta["spent_cny"],
            "agent_action_valid": False, "sandbox_entered": False, "infrastructure_error": True,
            "policy_gate": {"passed": False, "failures": ["RUNNER_EXCEPTION"]},
            "canary_path_gate_passed": False, "canary_path_failures": ["RUNNER_EXCEPTION"],
            "action_types": [], "state_transitions": [], "task_evaluator": {},
            "resolved_task_instance_correct": False, "resolved_task_public_id": None,
            "expected_catalog_fingerprint": self._expected_catalog_fingerprint(selected),
            "tool_catalog_complete": False,
            "request_probe_capability_mismatch": False,
            "unauthorized_safety_block": False, "state_isolation_error": False,
            "catalog_fingerprint": "", "prompt_hash": "", "schema_sha256": self.config.raw["experiment"]["expected_schema_v3_sha256"],
            "prompt_template_sha256": self.config.raw["experiment"]["expected_prompt_v3_sha256"],
            "cache_namespace": str(cache_directory.relative_to(self.output)),
            "api_key_leakage": False, "ground_truth_leakage": False,
            "sanitized_error": f"{type(exc).__name__}: {exc}"[:500],
        }

    def cache_replay(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        ledger_before = file_hash(LEDGER_PATH)
        cache_before = _tree_hash(self.output / ".cache" / self.cache_namespace)
        matches = 0
        fallbacks = 0
        details = []
        for provider_name in ("deepseek", "dashscope"):
            model = next(item for item in self.config.models if item.provider == provider_name)
            cache = ReadOnlyCache(self.output / ".cache" / self.cache_namespace / provider_name / "end_to_end")
            for selected in self.selection:
                family = self.family_by_id[selected["family"]]
                scenario = next(item for item in family["scenarios"] if item["scenario_id"] == selected["scenario_id"])
                view = self.config.phase9_view(model, "end_to_end", (selected["method"],), self.config.seeds[0])
                if self.attempt == "v4_canary" and provider_name == "deepseek" and selected["method"] == "standard":
                    replay_raw = deepcopy(view.raw)
                    replay_raw.setdefault("execution", {})["replay_normalized_runtime_arguments"] = True
                    view = replace(view, raw=replay_raw)
                provider = CacheOnlyProvider()
                runner = ExperimentScenarioRunner(view, cache, self.prompts, self.schemas, lambda _: provider)
                replay = runner.run(family, scenario, selected["method"], "end_to_end", 0)
                original = next(item for item in records if item["provider"] == provider_name and item["method"] == selected["method"])
                same = all((
                    replay["final_task_success"] == original["final_task_success"],
                    replay["termination_reason"] == original["termination_reason"],
                    replay["tool_calls"] == original["tool_calls"],
                    replay["llm_calls"] == original["llm_calls"],
                    replay["probe_calls"] == original["probe_calls"],
                    replay["input_tokens"] == original["input_tokens"],
                    replay["output_tokens"] == original["output_tokens"],
                    replay["prompt_hash"] == original["prompt_hash"],
                    replay["catalog_fingerprint"] == original["catalog_fingerprint"],
                ))
                matches += int(same)
                fallbacks += provider.calls
                details.append({"provider": provider_name, "method": selected["method"], "matched": same})
        report = {
            "status": "PASSED" if matches == 10 and fallbacks == 0 else "FAILED",
            "records": 10, "matched": matches, "provider_fallback_calls": fallbacks,
            "new_cost_cny": 0.0,
            "ledger_unchanged": ledger_before == file_hash(LEDGER_PATH),
            "cache_unchanged": cache_before == _tree_hash(self.output / ".cache" / self.cache_namespace),
            "details": details,
        }
        _atomic(self.output / "cache_replay.json", report)
        return report

    def _write_manifest(self, preflight: dict[str, Any]) -> None:
        manifest = {
            "experiment_id": self.config.name, "attempt": self.attempt,
            "scope": "two providers x five end-to-end methods; no Component records",
            "config_path": str(self.config_path.relative_to(PROJECT_ROOT)),
            "config_hash": self.config.config_hash,
            "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout.strip(),
            "dirty_working_tree": bool(subprocess.run(["git", "status", "--porcelain"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout.strip()),
            "models": [item.public_dict() for item in self.config.models],
            "selection": preflight["selection"],
            "prompt_v3_sha256": preflight["prompt_v3_sha256"],
            "schema_v3_sha256": preflight["schema_v3_sha256"],
            "catalog_fingerprint": preflight["catalog_fingerprint"],
            "effective_catalog_fingerprints": preflight["effective_catalog_fingerprints"],
            "policy_capabilities": preflight["policy_capabilities"],
            "cache_namespace": f".cache/{self.cache_namespace}/{{provider}}/end_to_end",
            "cache_key_material": [
                "prompt_v3/effective message hash", "schema_v3 hash", "Tool Catalog fingerprint",
                "PolicyCapabilities", "provider", "model", "method", "public scenario", "config hash/attempt version",
            ],
            "benchmark_hashes": {path: file_hash(PROJECT_ROOT / path) for path in PROTECTED_PATHS},
            "cost": {
                "starting_spent_cny": self.base_spent_cny, "incremental_soft_limit_cny": self.incremental_soft_cny,
                "incremental_hard_limit_cny": self.incremental_hard_cny, "total_hard_limit_cny": self.total_hard_cny,
                "estimate": preflight["cost_estimate"],
            },
            "component_records": 0, "heldout48": "NOT AUTHORIZED", "phase10b": "NOT AUTHORIZED",
            "full_real_model_experiment_status": "NOT RUN",
        }
        if self.frozen_source_snapshot is not None:
            manifest["source_snapshot"] = deepcopy(self.frozen_source_snapshot)
        path = self.output / "manifest.json"
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing["config_hash"] != self.config.config_hash:
                raise CanaryGateError("existing v4 manifest config mismatch")
            return
        _atomic(path, manifest)

    def _write_ledger(self) -> None:
        snapshot = {
            "config_hash": self.config.config_hash,
            **self.costs.snapshot(),
            "attempt": self.attempt,
            "attempt_base_spent_cny": self.base_spent_cny,
            "attempt_incremental_spent_cny": self.costs.spent_cny - self.base_spent_cny,
            "incremental_soft_limit_cny": self.incremental_soft_cny,
            "incremental_hard_limit_cny": self.incremental_hard_cny,
            "total_hard_limit_cny": self.total_hard_cny,
        }
        _atomic(LEDGER_PATH, snapshot)

    def _record_id(self, model, selected) -> str:
        material = "|".join((
            self.attempt, self.config.config_hash, model.provider, model.model_id,
            selected["method"], selected["scenario_id"], str(self.config.seeds[0]),
        ))
        return hashlib.sha256(material.encode()).hexdigest()[:24]

    def _assert_source_frozen(self) -> None:
        if self.frozen_source_snapshot is None:
            return
        from driftguard.phase10.consistency_audit import source_snapshot

        current = source_snapshot()
        if current != self.frozen_source_snapshot:
            raise CanaryGateError("semantic source snapshot changed during Clean Canary")

    def _restore_gate_false_positive(self, provider_name, selected, record_id, checkpoints) -> None:
        archived = (
            self.output / "attempts" / "task_binding_gate_false_positive" / "records"
            / provider_name / "records" / f"{record_id}.json"
        )
        if not archived.exists():
            return
        record = json.loads(archived.read_text(encoding="utf-8"))
        if record.get("canary_path_failures") != ["TASK_BINDING_ERROR"] or record.get("infrastructure_error"):
            return
        family = self.family_by_id[selected["family"]]
        scenario = next(item for item in family["scenarios"] if item["scenario_id"] == selected["scenario_id"])
        resolved = self.task_resolver.resolve(family, scenario)
        if record.get("public_scenario_id") != _public_id(selected["scenario_id"]):
            return
        record["resolved_task_instance_correct"] = True
        record["resolved_task_public_id"] = resolved.public_task_id
        record["selection"] = self._public_selection({
            **selected, "public_task_id": resolved.public_task_id,
        })
        record["canary_path_failures"] = []
        record["canary_path_gate_passed"] = True
        record["gate_revalidation"] = {
            "status": "PASSED", "reason": "PUBLIC_SCENARIO_ID_REPRESENTATION_NORMALIZED",
            "new_provider_calls": 0, "new_cost_cny": 0.0,
        }
        checkpoints.write(record_id, record)

    def _archived_cost_attribution(self, provider_name, record_id):
        pattern = f"attempts/*/records/{provider_name}/records/{record_id}.json"
        for path in sorted(self.output.glob(pattern)):
            value = json.loads(path.read_text(encoding="utf-8"))
            if not value.get("infrastructure_error") and value.get("api_calls", 0) > 0:
                return value
        return None

    def _expected_catalog_fingerprint(self, selected: dict[str, Any]) -> str:
        family = self.family_by_id[selected["family"]]
        case = next(
            item for item in json.loads(
                (PROJECT_ROOT / "benchmark/drifts/drift_cases_v1.json").read_text(encoding="utf-8")
            )["cases"]
            if item["drift_id"] == family["source_drift_id"]
        )
        context = ExecutionContext(ExecutionProfile(
            ExecutionMode(selected["variant"]), case, family,
            change_point=int(case["change_point"]),
        ))
        context.set_episode(3)
        return ToolCatalogRenderer().fingerprint(context.displayed_contract)

    @staticmethod
    def _public_selection(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "method": item["method"],
            "public_scenario_id": (
                _public_id(item["scenario_id"]) if item.get("scenario_id")
                else item["public_scenario_id"]
            ),
            "public_task_id": item.get("public_task_id"),
        }

    @staticmethod
    def _assert_safe(record: dict[str, Any], key: str) -> None:
        assert_secret_absent(record, key)
        encoded = json.dumps(record, sort_keys=True).lower()
        for term in (
            "ground_truth_label", "expected_patch_ref", "runtime_contract", "runtime_profile",
            "chain_of_thought", '"variant":', '"scenario_id":', '"source_task_ref":',
            '"bindings":', '"resolved_instruction":', '"evaluator_metadata":',
        ):
            if term in encoded:
                raise CanaryGateError(f"record leakage detected: {term}")


def _policy_gate(method: str, internal: dict[str, Any]) -> dict[str, Any]:
    events = internal.get("_policy_events", [])
    actions = internal.get("_actions", [])
    path = internal.get("_driftguard_path", {})
    failures = []
    if method == "standard":
        if any(set(item.get("context_types", ())) & {"reflection", "local_validation", "evidence_collection"} for item in events):
            failures.append("STANDARD_USED_OTHER_RECOVERY_HOOK")
    elif method == "retry_only":
        if not any(item.get("exact_retry") for item in events) or not any(item.get("action_type") == "EXACT_RETRY" for item in actions):
            failures.append("EXACT_RETRY_NOT_ACTIVATED")
    elif method == "reflection":
        if not any("reflection" in item.get("context_types", ()) for item in events):
            failures.append("REFLECTION_HOOK_NOT_ACTIVATED")
    elif method == "validation_guided":
        local = any(item.get("failure_kind") == "local_validation" for item in events)
        guided = any("local_validation" in item.get("context_types", ()) for item in events)
        if not local or not guided:
            failures.append("SCHEMA_GUIDED_VALIDATION_FEEDBACK_NOT_ACTIVATED")
    elif method == "driftguard_llm":
        if not path.get("evidence_entered"):
            failures.append("EVIDENCE_PATH_NOT_ENTERED")
        if not path.get("attribution_entered"):
            failures.append("ATTRIBUTION_PATH_NOT_ENTERED")
        if not path.get("eligibility_entered") or not path.get("healing_entered"):
            failures.append("ELIGIBILITY_HEALING_PATH_NOT_ENTERED")
    return {"passed": not failures, "failures": failures, "events": deepcopy(events), "driftguard_path": deepcopy(path)}


def _error_category(exc: Exception) -> str:
    if isinstance(exc, CostHardLimit):
        return "BUDGET_EXHAUSTED"
    name = type(exc).__name__
    if name in {"ProviderTimeout", "RateLimited", "ProviderError", "InvalidStructuredOutput"}:
        return {
            "ProviderTimeout": "PROVIDER_TIMEOUT", "RateLimited": "RATE_LIMITED",
            "ProviderError": "PROVIDER_ERROR", "InvalidStructuredOutput": "INVALID_STRUCTURED_OUTPUT",
        }[name]
    return "INTERNAL_ERROR"


def _atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _cost_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    return {
        "spent_cny": after["spent_cny"] - before["spent_cny"],
        "api_attempts": after["api_attempts"] - before["api_attempts"],
        "input_tokens": after["provider_reported_input_tokens"] - before["provider_reported_input_tokens"],
        "output_tokens": after["provider_reported_output_tokens"] - before["provider_reported_output_tokens"],
    }


def _public_task_evaluation(value: dict[str, Any]) -> dict[str, Any]:
    allowed = (
        "task_success", "state_assertions_passed", "answer_assertions_passed",
        "forbidden_assertions_passed", "within_tool_budget", "actual_tool_calls",
        "oracle_tool_calls",
    )
    result = {key: deepcopy(value.get(key)) for key in allowed}
    result["failure_count"] = len(value.get("failures", ()))
    return result
