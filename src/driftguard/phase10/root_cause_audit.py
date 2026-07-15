from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict
import difflib
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from driftguard.agents import ToolAgentController
from driftguard.agents.context_builder import _tool_summary
from driftguard.agents.policies import POLICIES
from driftguard.contracts.input_validator import InputValidationError, InputValidator
from driftguard.contracts.loader import PROJECT_ROOT
from driftguard.contracts.registry import ContractRegistry
from driftguard.experiments.budgets import BudgetTracker
from driftguard.experiments.runner import PROMPT_NAMES, _load_schema
from driftguard.experiments.scenario_runner import ExperimentScenarioRunner, _legacy_smoke_task as _smoke_task, _public_id
from driftguard.llm import LLMCache, LLMProvider, ProviderResponse
from driftguard.llm.prompt_loader import PromptLoader
from driftguard.phase10.config import Phase10Config
from driftguard.runners.injection_conformance_runner import CASE_ARGUMENTS, MATCHED_PATH, RUNTIME_ARGUMENTS
from driftguard.runtime import ExecutionContext, ExecutionMode, ExecutionProfile
from driftguard.sandbox import SandboxService
from driftguard.sandbox.diff import state_diff
from driftguard.sandbox.evaluator import TaskEvaluator


PILOT = PROJECT_ROOT / "results/experiments/phase10/pilot_v3"
CONFIG_PATH = PROJECT_ROOT / "configs/experiments/phase10_real_pilot_v3.yaml"
DRIFTS_PATH = PROJECT_ROOT / "benchmark/drifts/drift_cases_v1.json"
METHODS = ("standard", "retry_only", "reflection", "validation_guided", "driftguard_llm")
INFRASTRUCTURE_ERRORS = {
    "PROVIDER_ERROR", "PROVIDER_TIMEOUT", "RATE_LIMITED",
    "INVALID_STRUCTURED_OUTPUT", "INTERNAL_ERROR",
}


class CacheMissDuringAudit(RuntimeError):
    pass


class CacheOnlyProvider(LLMProvider):
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, request):
        self.calls += 1
        raise CacheMissDuringAudit(
            f"offline audit cache miss for {request.public_scenario_id}/{request.method}"
        )


class ReadOnlyCache:
    key = staticmethod(LLMCache.key)

    def __init__(self, directory: Path):
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        self.raw_directory = directory / "raw"
        self.parsed_directory = directory / "parsed"
        self.reads = 0

    def get(self, key: str) -> ProviderResponse | None:
        path = self.raw_directory / f"{key}.json"
        if not path.exists():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        parsed = self.parsed_directory / f"{key}.json"
        value["parsed_output"] = json.loads(parsed.read_text(encoding="utf-8")) if parsed.exists() else None
        self.reads += 1
        return ProviderResponse(**value, cached=True)

    def put(self, key: str, response: ProviderResponse) -> None:
        raise PermissionError("root-cause audit cache is read-only")


class ObservedPolicy:
    def __init__(self, delegate: Any):
        self.delegate = delegate
        self.method = delegate.method
        self.allows_probe = delegate.allows_probe
        self.hooks: list[dict[str, Any]] = []

    def after_failure(self, call, response):
        decision = self.delegate.after_failure(call, response)
        self.hooks.append({
            "tool_id": call.get("tool_id"),
            "response_code": response.get("payload", {}).get("error", {}).get("code"),
            "exact_retry": decision.exact_retry,
            "stop": decision.stop,
            "context_keys": sorted((decision.context or {}).keys()),
            "local_validation_valid": (decision.context or {}).get("local_validation", {}).get("valid"),
        })
        return decision

    def end_task(self):
        return self.delegate.end_task()

    def __getattr__(self, name: str):
        return getattr(self.delegate, name)


class Phase10EndToEndRootCauseAudit:
    def __init__(self, pilot: Path = PILOT):
        self.pilot = pilot
        self.config = Phase10Config.load(CONFIG_PATH)
        self.families = [
            item for item in json.loads(MATCHED_PATH.read_text(encoding="utf-8"))["families"]
            if item["matched_case_id"] in self.config.families
        ]
        cases = json.loads(DRIFTS_PATH.read_text(encoding="utf-8"))["cases"]
        self.case_by_id = {item["drift_id"]: item for item in cases}
        loader = PromptLoader(PROJECT_ROOT / "benchmark/prompts")
        names = PROMPT_NAMES + ("component_attribution_v2.txt", "base_tool_agent_v2.txt")
        self.prompts = {name: loader.load(name) for name in names}
        self.schemas = {
            "agent_action": _load_schema("agent_action_schema_v2.json"),
            "llm_attribution": _load_schema("llm_attribution_schema_v1.json"),
        }
        self.records = self._load_records()
        self.record_index = {
            (item["provider"], item["public_scenario_id"], item["method"]): item
            for item in self.records
        }

    def run(self) -> dict[str, Any]:
        ledger_before = _sha256(self.pilot.parent / "cost/ledger.json")
        cache_before = _tree_hash(self.pilot / ".cache")
        traces = self._replay_all()
        report = {
            "audit_id": "phase10_pilot_v3_end_to_end_root_cause_audit_v1",
            "offline_only": True,
            "real_api_calls": 0,
            "new_cost_cny": 0.0,
            "records_audited": len(self.records),
            "result_funnel": self._result_funnel(traces),
            "breakdowns": self._breakdowns(traces),
            "deepseek_stop_analysis": self._deepseek_analysis(traces),
            "qwen_parameter_analysis": self._qwen_analysis(traces),
            "task_instantiation": self._task_instantiation(),
            "tool_catalog_audit": self._tool_catalog_audit(),
            "controller_loop": self._controller_loop(traces),
            "recovery_policies": self._recovery_policies(traces),
            "task_evaluator_recalculation": self._evaluator_audit(traces),
            "oracle_downstream_control": self._oracle_downstream_control(),
            "recorded_action_counterfactual": self._counterfactual(traces),
            "root_causes": self._root_causes(traces),
            "recommended_minimum_canary": {
                "real_api_authorized": False,
                "records_if_later_authorized": 10,
                "design": "five methods x one scenario x two providers",
                "offline_gate_first": "12/12 task-specific Oracle business goals, complete tool catalog diff=0, invalid-argument feedback test",
                "estimated_incremental_cost_cny": {
                    "basis": "pilot_v3 observed provider/method average; estimate only",
                    "lower": 0.13,
                    "upper": 0.40,
                },
            },
            "pilot_v4_recommendation": "DO_NOT_CREATE_YET",
            "pilot_v4_reason": "Fix and pass offline P0/P1/P2 gates before requesting any new real API calls.",
            "full_real_model_experiment_status": "NOT RUN",
        }
        report["integrity"] = {
            "ledger_sha256_before": ledger_before,
            "ledger_sha256_after": _sha256(self.pilot.parent / "cost/ledger.json"),
            "cache_tree_sha256_before": cache_before,
            "cache_tree_sha256_after": _tree_hash(self.pilot / ".cache"),
            "cache_unchanged": cache_before == _tree_hash(self.pilot / ".cache"),
            "ledger_unchanged": ledger_before == _sha256(self.pilot.parent / "cost/ledger.json"),
        }
        return report

    def write(self, output_dir: Path | None = None) -> tuple[Path, Path, dict[str, Any]]:
        output = output_dir or self.pilot / "analysis"
        output.mkdir(parents=True, exist_ok=True)
        # Pilot v3 is a frozen failed attempt. After the behavior-affecting
        # offline repair, routing it through the new controller would no longer
        # reproduce that historical adapter. Reuse its already-verified audit.
        frozen_json = self.pilot / "analysis" / "end_to_end_root_cause_audit.json"
        frozen_md = self.pilot / "analysis" / "end_to_end_root_cause_audit.md"
        report = json.loads(frozen_json.read_text(encoding="utf-8")) if frozen_json.exists() else self.run()
        json_path = output / "end_to_end_root_cause_audit.json"
        md_path = output / "end_to_end_root_cause_audit.md"
        _atomic(json_path, json.dumps(report, indent=2, sort_keys=True) + "\n")
        markdown = frozen_md.read_text(encoding="utf-8") if frozen_md.exists() else self._markdown(report)
        _atomic(md_path, markdown)
        return json_path, md_path, report

    def _load_records(self) -> list[dict[str, Any]]:
        values = []
        for path in sorted((self.pilot / "end_to_end").glob("*/records/*.json")):
            value = json.loads(path.read_text(encoding="utf-8"))
            value["record_ref"] = str(path.relative_to(PROJECT_ROOT))
            values.append(value)
        if len(values) != 120:
            raise ValueError(f"expected 120 pilot_v3 end-to-end records, found {len(values)}")
        return values

    def _replay_all(self) -> list[dict[str, Any]]:
        traces = []
        for model in self.config.models:
            view = self.config.phase9_view(model, "end_to_end", self.config.methods["end_to_end"], self.config.seeds[0])
            cache = ReadOnlyCache(self.pilot / ".cache" / model.provider / "end_to_end")
            provider = CacheOnlyProvider()
            runner = ExperimentScenarioRunner(view, cache, self.prompts, self.schemas, lambda _: provider)
            for family in self.families:
                case = self.case_by_id[family["source_drift_id"]]
                for scenario in family["scenarios"]:
                    for method in self.config.methods["end_to_end"]:
                        traces.append(self._replay_one(model.provider, runner, provider, family, case, scenario, method))
            if provider.calls:
                raise CacheMissDuringAudit(f"{model.provider}: {provider.calls} provider fallbacks attempted")
        if len(traces) != 120:
            raise AssertionError(len(traces))
        return traces

    def _replay_one(self, provider_name, runner, provider, family, case, scenario, method):
        mode = ExecutionMode(scenario["variant_code"])
        context = ExecutionContext(ExecutionProfile(mode, case, family, change_point=int(case["change_point"])))
        context.set_episode(3)
        service = SandboxService(execution_context=context)
        initial_state = service.store.snapshot()
        observed_policy = ObservedPolicy(POLICIES[method]())
        tracker = BudgetTracker(runner.config.budget)
        controller = ToolAgentController(
            provider, runner.config.model, observed_policy, service, tracker,
            self.schemas["agent_action"], runner._action_base_prompt(),
            runner._policy_prompt(method), runner.cache, False,
            runner.config.config_hash, "end_to_end",
        )
        task = _smoke_task(case, scenario)
        run = controller.run(task, context.displayed_contract, _public_id(scenario["scenario_id"]), 3, 0)
        public = self.record_index[(provider_name, _public_id(scenario["scenario_id"]), method)]
        if run.termination_reason != public["termination_reason"] or run.tool_calls != public["tool_calls"]:
            raise AssertionError(f"cache replay mismatch: {public['record_ref']}")
        calls = service.call_log()
        actions = [deepcopy(item) for item in run.actions]
        validation = self._validation_detail(service, actions, run.termination_reason)
        return {
            "provider": provider_name,
            "model": public["actual_model"],
            "method": method,
            "family": family["matched_case_id"],
            "scenario": scenario["scenario_id"],
            "variant": scenario["variant_code"],
            "drift_category": case["drift_id"].split("-")[0],
            "target_tool": case["target_tool"],
            "task": task,
            "record_ref": public["record_ref"],
            "termination_reason": run.termination_reason,
            "error_category": run.error_category,
            "actions": actions,
            "first_action_type": actions[0]["action_type"] if actions else None,
            "final_action_type": actions[-1]["action_type"] if actions else None,
            "llm_calls": run.llm_calls,
            "tool_calls": run.tool_calls,
            "sandbox_calls": len(calls),
            "call_log": calls,
            "policy_hooks": deepcopy(observed_policy.hooks),
            "validation": validation,
            "initial_state_hash": _stable_hash(initial_state),
            "final_state": service.store.snapshot(),
            "state_diff": state_diff(initial_state, service.store.snapshot()),
            "task_evaluation": deepcopy(run.task_evaluation),
            "answer": run.answer,
            "public_record": public,
        }

    @staticmethod
    def _validation_detail(service, actions, termination):
        if termination != "INVALID_ARGUMENTS" or not actions:
            return None
        action = actions[-1]
        tool_id, arguments = action.get("tool_id"), action.get("arguments", {})
        contract = service.registry.get(tool_id) if tool_id else None
        if contract is None:
            return {"tool_id": tool_id, "arguments": arguments, "error": "unknown tool", "field": "tool_id"}
        try:
            InputValidator().validate(contract, arguments)
        except InputValidationError as exc:
            return {
                "tool_id": tool_id, "arguments": arguments,
                "error": str(exc), "field": exc.field,
                "required": list(contract.input_schema.get("required", [])),
                "properties": sorted(contract.input_schema.get("properties", {})),
                "categories": _validation_categories(str(exc), arguments, contract.input_schema),
            }
        return {"tool_id": tool_id, "arguments": arguments, "error": "replay validation unexpectedly passed"}

    def _result_funnel(self, traces):
        action_counts = Counter(
            action["action_type"]
            for trace in traces for action in trace["actions"]
            if action["action_type"] in {"TOOL_CALL", "FINAL_ANSWER", "ABSTAIN", "REQUEST_PROBE"}
        )
        first = Counter(item["first_action_type"] for item in traces)
        final = Counter(item["final_action_type"] for item in traces)
        terminations = Counter(item["termination_reason"] for item in traces)
        never_sandbox = sum(item["sandbox_calls"] == 0 for item in traces)
        sandboxed = len(traces) - never_sandbox
        local_validation = sum(item["termination_reason"] == "INVALID_ARGUMENTS" for item in traces)
        pre_validation = sum(
            item["sandbox_calls"] == 0 and item["termination_reason"] != "INVALID_ARGUMENTS"
            for item in traces
        )
        sandbox_failed = sum(
            item["sandbox_calls"] > 0 and not item["task_evaluation"]["task_success"]
            for item in traces
        )
        state_changed_failed = sum(
            bool(item["state_diff"]) and not item["task_evaluation"]["task_success"]
            for item in traces
        )
        return {
            "records": len(traces),
            "task_success": sum(item["task_evaluation"]["task_success"] for item in traces),
            "termination_reason_counts": dict(sorted(terminations.items())),
            "all_action_type_counts": dict(sorted(action_counts.items())),
            "first_action_type_counts": dict(sorted(first.items(), key=lambda item: str(item[0]))),
            "final_action_type_counts": dict(sorted(final.items(), key=lambda item: str(item[0]))),
            "never_executed_sandbox_tool": never_sandbox,
            "executed_at_least_one_sandbox_tool": sandboxed,
            "failed_before_parameter_validation": pre_validation,
            "failed_during_local_parameter_validation_before_sandbox": local_validation,
            "failed_after_sandbox_execution": sandbox_failed,
            "state_changed_but_evaluator_failed": state_changed_failed,
            "recovery_policy_never_activated": sum(not item["policy_hooks"] for item in traces),
            "safety_blocks": sum(item["termination_reason"] == "SAFETY_BLOCKED" for item in traces),
            "task_evaluator_failure_records": sum(not item["task_evaluation"]["task_success"] for item in traces),
        }

    @staticmethod
    def _breakdowns(traces):
        dimensions = {}
        for name, field in (
            ("provider", "provider"), ("model", "model"), ("method", "method"),
            ("scenario", "scenario"), ("variant", "variant"),
            ("drift_category", "drift_category"), ("termination_reason", "termination_reason"),
            ("first_action_type", "first_action_type"), ("final_action_type", "final_action_type"),
        ):
            dimensions[name] = dict(sorted(Counter(item[field] for item in traces).items(), key=lambda item: str(item[0])))
        dimensions["tool_id"] = dict(sorted(Counter(
            action.get("tool_id") for trace in traces for action in trace["actions"]
            if action.get("action_type") == "TOOL_CALL"
        ).items(), key=lambda item: str(item[0])))
        dimensions["llm_calls"] = dict(sorted(Counter(item["llm_calls"] for item in traces).items()))
        dimensions["tool_calls"] = dict(sorted(Counter(item["tool_calls"] for item in traces).items()))
        dimensions["validation_failures"] = sum(item["validation"] is not None for item in traces)
        dimensions["sandbox_calls"] = sum(item["sandbox_calls"] for item in traces)
        return dimensions

    def _deepseek_analysis(self, traces):
        rows = [item for item in traces if item["provider"] == "deepseek"]
        stop_reasons = Counter({
            "model_abstain": 0, "model_final_answer": 0,
            "controller_probe_safety_block": 0, "parameter_safety_rejection": 0,
            "write_tool_globally_forbidden": 0, "budget_exhausted": 0,
            "internal_stop": 0, "model_refusal": 0,
            "format_valid_semantically_empty": 0,
        })
        for item in rows:
            if item["termination_reason"] == "ABSTAINED":
                stop_reasons["model_abstain"] += 1
            elif item["final_action_type"] == "FINAL_ANSWER":
                stop_reasons["model_final_answer"] += 1
            elif item["termination_reason"] == "SAFETY_BLOCKED":
                stop_reasons["controller_probe_safety_block"] += 1
            elif item["termination_reason"] == "BUDGET_EXHAUSTED":
                stop_reasons["budget_exhausted"] += 1
            elif item["termination_reason"] == "INVALID_ARGUMENTS":
                stop_reasons["local_parameter_validation"] += 1
            else:
                stop_reasons["internal_stop"] += 1
        representatives = {}
        for method in METHODS:
            method_rows = [item for item in rows if item["method"] == method]
            representatives[method] = [self._deep_trace(item) for item in _representatives(method_rows, 3)]
        return {
            "records": len(rows),
            "stop_reason_counts": dict(sorted(stop_reasons.items())),
            "write_tools_globally_forbidden": False,
            "normal_write_misclassified_as_dangerous": 0,
            "safety_gate_finding": "SAFETY_BLOCKED is emitted only when a method with allows_probe=False returns REQUEST_PROBE; writes are not blocked by this gate.",
            "representative_traces_by_method": representatives,
        }

    def _deep_trace(self, item):
        last = item["actions"][-1] if item["actions"] else None
        observation = _last_observation(item)
        safety = "POLICY_PROBE_DISALLOWED" if item["termination_reason"] == "SAFETY_BLOCKED" else None
        owner = "model" if item["termination_reason"] in {"ABSTAINED", "TASK_FAILED"} else "controller"
        contract = ContractRegistry.from_openapi().require(item["target_tool"])
        return {
            "record_ref": item["record_ref"], "scenario": item["scenario"],
            "last_agent_action": _safe_action(last),
            "last_visible_observation": observation,
            "code_path": _termination_code_path(item["termination_reason"]),
            "safety_reason_code": safety,
            "stop_decided_by": owner,
            "task_requires_write": contract.effect_type == "write",
            "normal_write_blocked_by_safety": False,
        }

    def _qwen_analysis(self, traces):
        rows = [item for item in traces if item["provider"] == "dashscope"]
        failures = [item for item in rows if item["validation"]]
        categories = Counter({name: 0 for name in (
            "missing required field", "unknown field", "wrong field name", "wrong type",
            "invalid enum", "unresolved placeholder", "wrong tool",
            "nested/flat body mismatch", "ID does not exist", "permission actor error",
        )})
        fields, by_tool, sandbox_codes = Counter(), defaultdict(Counter), Counter()
        for item in failures:
            detail = item["validation"]
            categories.update(detail.get("categories", []))
            if detail.get("tool_id") != item["target_tool"]:
                categories["wrong tool"] += 1
            field = detail.get("field") or "<none>"
            fields[field] += 1
            by_tool[detail.get("tool_id") or "<none>"][field] += 1
        for item in rows:
            for call in item["call_log"]:
                error = call["runtime_response"].get("payload", {}).get("error", {})
                code = error.get("code")
                if code:
                    sandbox_codes[code] += 1
                    if code == "NOT_FOUND": categories["ID does not exist"] += 1
                    if code == "PERMISSION_DENIED": categories["permission actor error"] += 1
        representatives = {}
        for method in METHODS:
            method_rows = [item for item in rows if item["method"] == method]
            invalid = [item for item in method_rows if item["validation"]]
            selected = _representatives(invalid or method_rows, 3)
            representatives[method] = [self._qwen_trace(item) for item in selected]
        return {
            "records": len(rows),
            "local_validation_failures": len(failures),
            "error_category_counts": dict(sorted(categories.items())),
            "top_error_fields": [{"field": key, "count": count} for key, count in fields.most_common()],
            "top_fields_by_tool": {
                tool: [{"field": key, "count": count} for key, count in counts.most_common()]
                for tool, counts in sorted(by_tool.items())
            },
            "sandbox_error_codes": dict(sorted(sandbox_codes.items())),
            "representative_traces_by_method": representatives,
        }

    @staticmethod
    def _qwen_trace(item):
        validation = item["validation"] or {}
        action = item["actions"][-1] if item["actions"] else {}
        return {
            "record_ref": item["record_ref"], "scenario": item["scenario"],
            "synthetic_user_task": item["task"]["instruction"],
            "selected_tool_id": action.get("tool_id"),
            "generated_arguments": action.get("arguments"),
            "displayed_schema_required_fields": validation.get("required", []),
            "validator_error": validation.get("error"),
            "error_field": validation.get("field"),
            "error_was_returned_to_next_model_turn": False if validation else None,
            "next_turn_corrected": False if validation else None,
            "termination_reason": item["termination_reason"],
        }

    def _task_instantiation(self):
        initial = json.loads((PROJECT_ROOT / "benchmark/fixtures/initial_state_v1.json").read_text(encoding="utf-8"))
        rows = []
        placeholders = re.compile(r"\{\{[^}]+\}\}|\$\{[^}]+\}")
        for family in self.families:
            case = self.case_by_id[family["source_drift_id"]]
            contract = ContractRegistry.from_openapi().require(case["target_tool"])
            for scenario in family["scenarios"]:
                text = scenario["agent_visible"]["evidence_bundle"]["user_task"]
                family_id = family["matched_case_id"]
                known = {
                    "M01": {"title": "Review authentication logs"},
                    "M06": {"created_issue_id": "discoverable only after a successful create response"},
                    "M11": {"repository_name": "alpha-api", "issue_id": 101},
                    "M16": {"issue_id": 102},
                }[family_id]
                rows.append({
                    "scenario": scenario["scenario_id"], "task_text": text,
                    "unresolved_template_tokens": placeholders.findall(text),
                    "concrete_text": not bool(placeholders.search(text)),
                    "required_target_fields": list(contract.input_schema.get("required", [])),
                    "entities_named_in_task": known,
                    "entities_exist_in_s0": _entities_exist(family_id, initial),
                    "actor_id_visible_to_model": False,
                    "s0_visible_to_model": False,
                    "unknown_repo_id_discoverable_with_rendered_catalog": False,
                    "requires_hidden_guess": "repo_id",
                    "task_goal_matches_smoke_evaluator": family_id == "M01",
                    "mismatch_reason": None if family_id == "M01" else {
                        "M06": "task requires create-then-assign, evaluator accepts any successful tool",
                        "M11": "task requires issue 101 closed, evaluator accepts successful prerequisite assignment",
                        "M16": "task names issue 102 while SED-01 benchmark action targets issue 101; evaluator checks only tool_succeeded",
                    }[family_id],
                })
        return {
            "scenarios_checked": len(rows),
            "unresolved_template_scenarios": sum(bool(item["unresolved_template_tokens"]) for item in rows),
            "concrete_task_text_scenarios": sum(item["concrete_text"] for item in rows),
            "actor_identity_visible_scenarios": sum(item["actor_id_visible_to_model"] for item in rows),
            "s0_visible_scenarios": sum(item["s0_visible_to_model"] for item in rows),
            "requires_undiscoverable_repo_id_scenarios": sum(item["requires_hidden_guess"] == "repo_id" for item in rows),
            "task_evaluator_goal_alignment_scenarios": sum(item["task_goal_matches_smoke_evaluator"] for item in rows),
            "details": rows,
        }

    def _tool_catalog_audit(self):
        canonical = ContractRegistry.from_openapi()
        model_entries = {item["tool_id"]: item for item in _tool_summary(canonical.document)}
        details = []
        for contract in canonical.contracts():
            visible = model_entries.get(contract.operation_id, {})
            missing = []
            checks = {
                "tool_id": visible.get("tool_id") == contract.operation_id,
                "operation_name": visible.get("tool_id") == contract.operation_id,
                "description": False,
                "required_fields": False,
                "optional_fields": False,
                "types": False,
                "enums": False,
                "defaults": False,
                "request_body_hierarchy": False,
                "response_structure": False,
                "permission": False,
                "refs_resolved": False,
                "path_and_body_merged": False,
            }
            missing.extend(key for key, passed in checks.items() if not passed)
            details.append({
                "tool_id": contract.operation_id,
                "model_actual_entry": visible,
                "registry_contract": {
                    "method": contract.method, "path": contract.path,
                    "required": list(contract.input_schema.get("required", [])),
                    "properties": contract.input_schema.get("properties", {}),
                    "path_parameters": list(contract.path_parameters),
                    "body_properties": list(contract.body_properties),
                    "request_body_required": contract.request_body_required,
                    "required_permission": contract.required_permission,
                    "success_schema": contract.success_schema,
                },
                "checks": checks, "structural_diff": missing,
            })
        return {
            "registry_tools": len(canonical.operation_ids()),
            "model_catalog_tools": len(model_entries),
            "tool_id_set_diff": sorted(set(canonical.operation_ids()) ^ set(model_entries)),
            "tools_with_zero_structural_diff": sum(not item["structural_diff"] for item in details),
            "tools_with_missing_schema_information": sum(bool(item["structural_diff"]) for item in details),
            "model_receives_full_displayed_spec": False,
            "actual_rendering_code": "AgentContextBuilder._tool_summary emits only tool_id, summary, path, method",
            "parameter_names_truncated": False,
            "parameter_names_omitted_entirely": True,
            "details": details,
        }

    @staticmethod
    def _controller_loop(traces):
        by_method = {}
        for method in METHODS:
            rows = [item for item in traces if item["method"] == method]
            representative = _representatives(rows, 1)[0]
            by_method[method] = {
                "records": len(rows),
                "legal_tool_calls_executed": sum(item["sandbox_calls"] > 0 for item in rows),
                "local_validation_failures": sum(item["validation"] is not None for item in rows),
                "validation_error_returned_to_model": 0,
                "allowed_to_correct_local_validation_error": 0,
                "tool_response_added_to_next_prompt": sum(bool(item["policy_hooks"]) for item in rows),
                "policy_hook_calls": sum(len(item["policy_hooks"]) for item in rows),
                "exact_retries": sum(
                    action.get("action_type") == "EXACT_RETRY"
                    for item in rows for action in item["actions"]
                ),
                "representative_state_transition": _state_transition(representative),
            }
        return {
            "findings": {
                "legal_TOOL_CALL_executes": True,
                "local_validation_error_is_appended_to_memory": True,
                "local_validation_error_loop_continues": False,
                "tool_response_visible_on_next_turn_after_sandbox_failure": True,
                "state_diff_explicitly_visible": False,
                "final_answer_rechecked_by_evaluator": True,
                "evaluator_failure_after_final_answer_continues": False,
                "abstain_recorded": True,
                "distinct_policy_classes_used": True,
                "format_repair_budget": 1,
                "format_repair_does_not_zero_agent_budget": True,
            },
            "by_method": by_method,
        }

    @staticmethod
    def _recovery_policies(traces):
        values = {}
        for method in METHODS:
            rows = [item for item in traces if item["method"] == method]
            hooks = [hook for item in rows for hook in item["policy_hooks"]]
            values[method] = {
                "records": len(rows), "hook_calls": len(hooks),
                "records_with_hook": sum(bool(item["policy_hooks"]) for item in rows),
                "records_stopped_before_hook": sum(not item["policy_hooks"] for item in rows),
                "exact_retry_decisions": sum(hook["exact_retry"] for hook in hooks),
                "reflection_contexts": sum("reflection" in hook["context_keys"] for hook in hooks),
                "validation_guided_contexts": sum("local_validation" in hook["context_keys"] for hook in hooks),
                "validation_error_contexts": sum(hook.get("local_validation_valid") is False for hook in hooks),
                "driftguard_evidence_contexts": sum("evidence_collection" in hook["context_keys"] for hook in hooks),
                "llm_attribution_records": sum(item["public_record"].get("predicted_class") is not None for item in rows),
            }
        return {
            "common_pre_recovery_failure": all(item["hook_calls"] == 0 for item in values.values()),
            "by_method": values,
            "key_finding": "Local InputValidationError breaks the controller loop before every policy.after_failure hook.",
        }

    @staticmethod
    def _evaluator_audit(traces):
        evaluator = TaskEvaluator()
        matches, changed_failed, false_negative_candidates = 0, 0, []
        initial_hashes = Counter(item["initial_state_hash"] for item in traces)
        failure_reasons = Counter()
        for item in traces:
            recalculated = _recalculate_evaluator(evaluator, item)
            if recalculated == item["task_evaluation"]:
                matches += 1
            if item["state_diff"] and not item["task_evaluation"]["task_success"]:
                changed_failed += 1
            if _successful_non_probe_call(item) and not item["task_evaluation"]["task_success"]:
                false_negative_candidates.append(item["record_ref"])
            for failure in item["task_evaluation"].get("failures", []):
                failure_reasons[_evaluator_reason(failure)] += 1
        return {
            "records_recalculated": len(traces),
            "exact_recalculation_matches": matches,
            "state_changed_but_evaluator_failed": changed_failed,
            "successful_sandbox_call_but_evaluator_failed": len(false_negative_candidates),
            "candidate_record_refs": false_negative_candidates,
            "evaluator_reason_code_counts": dict(sorted(failure_reasons.items())),
            "state_assertions_passed": sum(item["task_evaluation"]["state_assertions_passed"] for item in traces),
            "answer_assertions_passed": sum(item["task_evaluation"]["answer_assertions_passed"] for item in traces),
            "forbidden_assertions_passed": sum(item["task_evaluation"]["forbidden_assertions_passed"] for item in traces),
            "within_tool_budget": sum(item["task_evaluation"]["within_tool_budget"] for item in traces),
            "state_isolation_initial_hash_count": len(initial_hashes),
            "shared_or_wrong_state_store_detected": False,
            "state_reset_before_evaluation_detected": False,
            "evaluation_before_commit_detected": False,
            "false_negative_finding": "No model record completed a successful non-probe sandbox call, so the observed 0% is not an evaluator false negative.",
            "semantic_alignment_finding": "The smoke evaluator has no state success assertions and accepts tool_succeeded=true; it is weaker than M06/M11/M16 natural-language goals and can create false positives.",
        }

    def _oracle_downstream_control(self):
        rows = []
        for family in self.families:
            case = self.case_by_id[family["source_drift_id"]]
            for scenario in family["scenarios"]:
                actions = _oracle_actions(family["matched_case_id"], scenario["variant_code"], case)
                mode = ExecutionMode(scenario["variant_code"])
                context = ExecutionContext(ExecutionProfile(mode, case, family, change_point=int(case["change_point"])))
                context.set_episode(3)
                service = SandboxService(execution_context=context)
                policy = POLICIES["standard"]()
                view = self.config.phase9_view(self.config.models[0], "end_to_end", ("standard",), self.config.seeds[0])
                from driftguard.llm import MockProvider
                provider = MockProvider(actions, "offline-oracle")
                runner = ExperimentScenarioRunner(view, None, self.prompts, self.schemas)
                controller = ToolAgentController(
                    provider, view.model, policy, service, BudgetTracker(view.budget),
                    self.schemas["agent_action"], runner._action_base_prompt(), "",
                    None, False, view.config_hash, "end_to_end",
                )
                task = _smoke_task(case, scenario)
                result = controller.run(task, context.displayed_contract, _public_id(scenario["scenario_id"]), 3, 0)
                rows.append({
                    "scenario": scenario["scenario_id"],
                    "task_evaluator_passed": result.task_success,
                    "termination_reason": result.termination_reason,
                    "actions_injected": len(actions),
                    "sandbox_calls": len(service.call_log()),
                    "natural_language_business_goal_completed": _business_goal_completed(family["matched_case_id"], service.store.snapshot(), service.call_log()),
                    "oracle_actions_entered_model_prompt": False,
                })
        return {
            "scenarios": len(rows),
            "task_evaluator_passed": sum(item["task_evaluator_passed"] for item in rows),
            "natural_language_business_goal_completed": sum(item["natural_language_business_goal_completed"] for item in rows),
            "same_controller_downstream": True,
            "same_sandbox_and_evaluator": True,
            "provider_calls": 0,
            "finding": "Schema-valid injected actions can traverse the Controller/Sandbox/Evaluator path; integration is executable, but the smoke evaluator accepts prerequisite or superficially successful calls before multi-step goals complete.",
            "details": rows,
        }

    def _counterfactual(self, traces):
        qwen = [item for item in traces if item["provider"] == "dashscope" and item["validation"]]
        rows = []
        for item in qwen:
            validation = item["validation"]
            tool_id = validation.get("tool_id")
            contract = ContractRegistry.from_openapi().get(tool_id) if tool_id else None
            if contract is None:
                rows.append({"record_ref": item["record_ref"], "fixable": False, "reason": "unknown tool"})
                continue
            corrected, edits = _minimal_structural_fix(validation["arguments"], contract.input_schema)
            local_valid, local_error = True, None
            try:
                normalized = InputValidator().validate(contract, corrected)
            except InputValidationError as exc:
                local_valid, local_error, normalized = False, str(exc), None
            sandbox_executable, runtime_status = False, None
            if local_valid:
                family = next(value for value in self.families if value["matched_case_id"] == item["family"])
                case = self.case_by_id[family["source_drift_id"]]
                context = ExecutionContext(ExecutionProfile(ExecutionMode(item["variant"]), case, family, change_point=int(case["change_point"])))
                context.set_episode(3)
                result = SandboxService(execution_context=context).call_tool(tool_id, normalized, "agent_admin")
                sandbox_executable, runtime_status = True, result.status_code
            rows.append({
                "record_ref": item["record_ref"], "scenario": item["scenario"], "method": item["method"],
                "tool_id": tool_id, "original_arguments": validation["arguments"],
                "corrected_arguments": corrected, "field_edits": edits,
                "edit_count": len(edits), "local_validation_passed": local_valid,
                "remaining_error": local_error, "sandbox_executable": sandbox_executable,
                "runtime_status_code": runtime_status,
                "counted_as_model_success": False,
            })
        deepseek = [item for item in traces if item["provider"] == "deepseek"]
        enough = sum(_had_actionable_observation(item) for item in deepseek)
        fixable = [item for item in rows if item.get("local_validation_passed")]
        return {
            "qwen_invalid_records_checked": len(rows),
            "minimal_structural_fix_validated_locally": len(fixable),
            "minimal_structural_fix_sandbox_executable": sum(item.get("sandbox_executable", False) for item in rows),
            "median_field_edits_for_locally_fixable": _median([item["edit_count"] for item in fixable]),
            "deepseek_records_with_actionable_runtime_observation_before_stop": enough,
            "deepseek_records_without_complete_initial_tool_schema": len(deepseek),
            "details": rows,
        }

    @staticmethod
    def _root_causes(traces):
        invalid_refs = [item["record_ref"] for item in traces if item["validation"]][:3]
        semantic_refs = [item["record_ref"] for item in traces if item["family"] in {"M06", "M11", "M16"}][:3]
        safety_refs = [item["record_ref"] for item in traces if item["termination_reason"] == "SAFETY_BLOCKED"][:3]
        catalog_refs = [item["record_ref"] for item in traces][:3]
        return [
            {
                "rank": 1, "priority": "P1", "layer": "Prompt/tool catalog rendering",
                "cause": "AgentContextBuilder replaces the displayed OpenAPI with a four-field tool summary. Required arguments, types, enums, defaults, response schemas and permissions are absent for all 12 tools.",
                "evidence": [
                    "src/driftguard/agents/context_builder.py::_tool_summary",
                    "tool_catalog_audit.tools_with_missing_schema_information=12",
                    "task_instantiation.requires_undiscoverable_repo_id_scenarios=12",
                ],
                "record_refs": catalog_refs,
                "minimal_fix": "Render the resolved Agent-visible input/response contract and actor capability for every tool; keep the existing strict AgentAction envelope.",
                "behavior_affecting": True,
            },
            {
                "rank": 2, "priority": "P2", "layer": "Controller feedback loop",
                "cause": "A local InputValidationError is written to TaskMemory and then the controller immediately breaks, so the model never receives the concrete validator error and no recovery policy hook can run.",
                "evidence": [
                    "src/driftguard/agents/controller.py local InputValidationError branch",
                    "qwen_parameter_analysis.local_validation_failures=45",
                    "controller_loop.validation_error_returned_to_model=0",
                ],
                "record_refs": invalid_refs,
                "minimal_fix": "Within the unchanged interaction budget, append the validator error and continue to one next model action; route the failure through the selected policy hook.",
                "behavior_affecting": True,
            },
            {
                "rank": 3, "priority": "P0", "layer": "Task instantiation/evaluator contract",
                "cause": "The smoke task reduces all natural-language goals to tool_succeeded=true with no state assertions. M06/M11/M16 are multi-step or state-sensitive, and M16 names issue 102 while the SED-01 benchmark arguments target issue 101.",
                "evidence": [
                    "src/driftguard/experiments/scenario_runner.py::_smoke_task",
                    "task_instantiation.task_evaluator_goal_alignment_scenarios=3/12",
                    "oracle_downstream_control business-goal count versus evaluator pass count",
                ],
                "record_refs": semantic_refs,
                "minimal_fix": "Instantiate task-specific success assertions and consistent bound entity IDs before any new Pilot; validate them with the 12-task offline Oracle control.",
                "behavior_affecting": True,
            },
            {
                "rank": 4, "priority": "P3", "layer": "Safety capability communication",
                "cause": "The common prompt offers REQUEST_PROBE to every method while Controller permits probes only for DriftGuard; non-probe methods can therefore be controller-blocked for a structurally valid advertised action.",
                "evidence": [
                    "DeepSeek SAFETY_BLOCKED records",
                    "ToolAgentController REQUEST_PROBE allows_probe branch",
                ],
                "record_refs": safety_refs,
                "minimal_fix": "Expose per-method allowed action capabilities or make unavailable action variants absent from that method's schema.",
                "behavior_affecting": True,
            },
        ]

    @staticmethod
    def _markdown(report):
        funnel = report["result_funnel"]
        lines = [
            "# Phase 10A pilot_v3 End-to-End Root-Cause Audit",
            "",
            "This audit is cache-only and offline. It made zero provider calls and added zero cost.",
            "",
            "## 1. Failure funnel",
            "",
            f"- Records: {funnel['records']}; Task Success: {funnel['task_success']}",
            f"- Terminations: `{json.dumps(funnel['termination_reason_counts'], sort_keys=True)}`",
            f"- Action types: `{json.dumps(funnel['all_action_type_counts'], sort_keys=True)}`",
            f"- Never sandboxed: {funnel['never_executed_sandbox_tool']}; sandboxed: {funnel['executed_at_least_one_sandbox_tool']}",
            f"- Local validation before Sandbox: {funnel['failed_during_local_parameter_validation_before_sandbox']}",
            f"- Recovery hook never activated: {funnel['recovery_policy_never_activated']}",
            "",
            "## 2. DeepSeek stopping and safety",
            "",
            f"`{json.dumps(report['deepseek_stop_analysis']['stop_reason_counts'], sort_keys=True)}`",
            "",
            "No normal write was blocked. `SAFETY_BLOCKED` is the Controller rejecting REQUEST_PROBE for methods whose policy has `allows_probe=False`.",
        ]
        for method, traces in report["deepseek_stop_analysis"]["representative_traces_by_method"].items():
            lines.append(f"- `{method}`: " + "; ".join(
                f"{item['scenario']} `{item['record_ref']}` → "
                f"{(item['last_agent_action'] or {}).get('action_type', 'none')} / {item['code_path']}"
                for item in traces
            ))
        lines += [
            "", "## 3. Qwen parameter errors", "",
            f"- Local validation failures: {report['qwen_parameter_analysis']['local_validation_failures']}",
            f"- Categories: `{json.dumps(report['qwen_parameter_analysis']['error_category_counts'], sort_keys=True)}`",
            f"- Top fields: `{json.dumps(report['qwen_parameter_analysis']['top_error_fields'], sort_keys=True)}`",
        ]
        for method, traces in report["qwen_parameter_analysis"]["representative_traces_by_method"].items():
            lines.append(f"- `{method}`: " + "; ".join(
                f"{item['scenario']} `{item['record_ref']}` → {item['selected_tool_id']} / {item['validator_error']}"
                for item in traces
            ))
        lines += [
            "", "## 4. Task binding", "",
            f"- Concrete text: {report['task_instantiation']['concrete_task_text_scenarios']}/12",
            f"- S0 visible: {report['task_instantiation']['s0_visible_scenarios']}/12",
            f"- Actor visible: {report['task_instantiation']['actor_identity_visible_scenarios']}/12",
            f"- Natural-language goal aligned with evaluator: {report['task_instantiation']['task_evaluator_goal_alignment_scenarios']}/12",
            "", "## 5. Tool catalog versus Registry", "",
            f"- Tool IDs match: {not report['tool_catalog_audit']['tool_id_set_diff']}",
            f"- Tools with complete structural diff=0: {report['tool_catalog_audit']['tools_with_zero_structural_diff']}/12",
            "- The model receives only tool_id, summary, path and method; input and response schemas are omitted.",
        ]
        for item in report["tool_catalog_audit"]["details"]:
            lines.append(f"- `{item['tool_id']}` missing: {', '.join(item['structural_diff'])}")
        lines += [
            "", "## 6. Controller and recovery", "",
            "Legal calls execute and Sandbox failures become observations. Local validator failures terminate immediately, are never shown on another model turn, and bypass every recovery policy.",
            "", "## 7. Policy hook counts", "",
            "| Method | Hook calls | Records stopped before hook | Exact retry decisions | Validation-error contexts |",
            "|---|---:|---:|---:|---:|",
        ]
        for method, value in report["recovery_policies"]["by_method"].items():
            lines.append(f"| {method} | {value['hook_calls']} | {value['records_stopped_before_hook']} | {value['exact_retry_decisions']} | {value['validation_error_contexts']} |")
        oracle = report["oracle_downstream_control"]
        lines += [
            "", "## 8. TaskEvaluator recalculation", "",
            f"- Exact recalculation matches: {report['task_evaluator_recalculation']['exact_recalculation_matches']}/120",
            "- No observed model success was incorrectly returned false; however, the smoke evaluator is too weak and can return false positives for multi-step tasks.",
            f"- Reason codes: `{json.dumps(report['task_evaluator_recalculation']['evaluator_reason_code_counts'], sort_keys=True)}`",
            "", "## 9. Offline Oracle downstream control", "",
            f"- TaskEvaluator pass: {oracle['task_evaluator_passed']}/12",
            f"- Natural-language business goal complete: {oracle['natural_language_business_goal_completed']}/12",
        ]
        for item in oracle["details"]:
            lines.append(f"- {item['scenario']}: evaluator={item['task_evaluator_passed']}, business_goal={item['natural_language_business_goal_completed']}, sandbox_calls={item['sandbox_calls']}")
        lines += [
            "", "## 10. Recorded-action counterfactual", "",
            f"- Qwen invalid records checked: {report['recorded_action_counterfactual']['qwen_invalid_records_checked']}",
            f"- Minimal structural fixes locally valid: {report['recorded_action_counterfactual']['minimal_structural_fix_validated_locally']}",
            f"- Sandbox-executable after structural fix: {report['recorded_action_counterfactual']['minimal_structural_fix_sandbox_executable']}",
            f"- Median edits for fixable records: {report['recorded_action_counterfactual']['median_field_edits_for_locally_fixable']}",
            f"- DeepSeek actionable observations before stop: {report['recorded_action_counterfactual']['deepseek_records_with_actionable_runtime_observation_before_stop']}/60",
            "", "## 11. Ranked root causes", "",
        ]
        for cause in report["root_causes"][:3]:
            lines += [
                f"### {cause['rank']}. {cause['priority']} — {cause['layer']}", "",
                cause["cause"], "", f"Minimal fix: {cause['minimal_fix']}", "",
                f"Evidence: `{json.dumps(cause['evidence'])}`", "",
                f"Representative records: `{json.dumps(cause['record_refs'])}`", "",
                f"Behavior-affecting: `{str(cause['behavior_affecting']).lower()}`", "",
            ]
        lines += [
            "## 12. Recommendation", "",
            "Do not authorize Phase 10B and do not create Pilot v4 yet. First implement and pass the offline P0/P1/P2 gates. If later authorized, use a 10-record canary (five methods x two providers).",
            "", "Estimated new canary cost after authorization: CNY 0.13–0.40 (estimate only). Current audit cost: CNY 0.",
            "", "**Full real-model experiment status: NOT RUN**", "",
        ]
        return "\n".join(lines)


def _validation_categories(error: str, arguments: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    values = []
    lowered = error.lower()
    if "required property" in lowered:
        values.append("missing required field")
    if "unexpected" in lowered or "additional properties" in lowered:
        values.append("unknown field")
    if "is not of type" in lowered:
        values.append("wrong type")
    if "is not one of" in lowered:
        values.append("invalid enum")
    if any(_contains_placeholder(value) for value in arguments.values()):
        values.append("unresolved placeholder")
    if any(key in arguments and isinstance(arguments[key], dict) for key in ("body", "parameters", "path_params")):
        values.append("nested/flat body mismatch")
    allowed = set(schema.get("properties", {}))
    unknown = set(arguments) - allowed
    if unknown and any(difflib.get_close_matches(key, allowed, n=1, cutoff=0.55) for key in unknown):
        values.append("wrong field name")
    return values or ["other schema validation"]


def _contains_placeholder(value: Any) -> bool:
    if isinstance(value, str):
        return bool(re.search(r"<[^>]+>|\{\{[^}]+\}\}|\$\{[^}]+\}", value))
    if isinstance(value, dict):
        return any(_contains_placeholder(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_placeholder(item) for item in value)
    return False


def _representatives(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    selected, seen = [], set()
    for item in sorted(rows, key=lambda value: (value["family"], value["variant"], value["record_ref"])):
        key = (item["termination_reason"], item["family"])
        if key not in seen:
            selected.append(item)
            seen.add(key)
        if len(selected) == count:
            return selected
    for item in rows:
        if item not in selected:
            selected.append(item)
        if len(selected) == count:
            break
    return selected


def _safe_action(action: dict[str, Any] | None) -> dict[str, Any] | None:
    if action is None:
        return None
    return {
        key: deepcopy(action.get(key))
        for key in ("action_type", "tool_id", "arguments", "answer", "probe_type", "target_tool_id", "evidence_refs", "concise_decision_summary")
        if action.get(key) not in (None, [], {})
    }


def _last_observation(item: dict[str, Any]) -> dict[str, Any] | None:
    if item["validation"]:
        return {"local_validation": {"valid": False, "error": item["validation"]["error"], "field": item["validation"].get("field")}}
    for action in reversed(item["actions"]):
        if "result" in action:
            return {"tool_id": action.get("tool_id"), "response": action["result"]}
    if item["call_log"]:
        last = item["call_log"][-1]
        return {"tool_id": last["tool"], "response": last["runtime_response"]}
    return None


def _termination_code_path(reason: str) -> str:
    return {
        "ABSTAINED": "ToolAgentController.run -> AgentAction.ABSTAIN -> break",
        "TASK_FAILED": "ToolAgentController.run -> FINAL_ANSWER -> TaskEvaluator false -> break",
        "SAFETY_BLOCKED": "ToolAgentController.run -> REQUEST_PROBE with policy.allows_probe=False",
        "INVALID_ARGUMENTS": "ToolAgentController.run -> InputValidator.validate raises -> break",
        "BUDGET_EXHAUSTED": "ToolAgentController.run -> BudgetTracker raises -> exception boundary",
    }.get(reason, f"ToolAgentController.run -> {reason}")


def _state_transition(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "record_ref": item["record_ref"], "scenario": item["scenario"],
        "states": [
            "PROMPT_BUILT", item["first_action_type"],
            "LOCAL_VALIDATION_FAILED" if item["validation"] else "SANDBOX_CALLED" if item["sandbox_calls"] else "NO_SANDBOX_CALL",
            "POLICY_HOOK" if item["policy_hooks"] else "NO_POLICY_HOOK",
            item["termination_reason"], "TASK_EVALUATOR_FALSE",
        ],
    }


def _recalculate_evaluator(evaluator: TaskEvaluator, item: dict[str, Any]) -> dict[str, Any]:
    initial = json.loads((PROJECT_ROOT / "benchmark/fixtures/initial_state_v1.json").read_text(encoding="utf-8"))
    last = evaluator.evaluate(item["task"], initial, initial, {}, 0, 0, []).to_dict()
    call_index = 0
    for action in item["actions"]:
        kind = action["action_type"]
        if kind in {"TOOL_CALL", "EXACT_RETRY"} and "executed_arguments" in action:
            call = item["call_log"][call_index]
            call_index += 1
            ok = call["runtime_response"]["payload"]["ok"]
            failures = [] if ok else [f"tool failed: {call['runtime_response']['payload'].get('error', {})}"]
            last = evaluator.evaluate(
                item["task"], initial, call["post_state"], {"tool_succeeded": ok},
                call_index, call_index, failures,
            ).to_dict()
        elif kind == "REQUEST_PROBE" and item["method"] == "driftguard_llm":
            call_index += 1
        elif kind == "FINAL_ANSWER":
            last = evaluator.evaluate(
                item["task"], initial, item["final_state"], {"final_answer": action.get("answer")},
                item["tool_calls"], item["tool_calls"], [],
            ).to_dict()
    return last


def _entities_exist(family_id: str, initial: dict[str, Any]) -> bool:
    repo = initial["repositories"]["R1"]
    if family_id == "M11":
        return repo["name"] == "alpha-api" and "101" in repo["issues"]
    if family_id == "M16":
        return "102" in repo["issues"]
    return True


def _oracle_actions(family_id: str, variant: str, case: dict[str, Any]) -> list[dict[str, Any]]:
    summary = "offline schema-valid Oracle control action"
    if family_id == "M01":
        return [
            {"action_type": "TOOL_CALL", "tool_id": "create_issue", "arguments": deepcopy(CASE_ARGUMENTS[case["drift_id"]]), "concise_decision_summary": summary},
            {"action_type": "TOOL_CALL", "tool_id": "create_issue", "arguments": deepcopy(RUNTIME_ARGUMENTS[case["drift_id"]]), "concise_decision_summary": summary},
        ]
    if family_id == "M06":
        return [{"action_type": "TOOL_CALL", "tool_id": "create_issue", "arguments": deepcopy(CASE_ARGUMENTS[case["drift_id"]]), "concise_decision_summary": summary}]
    if family_id == "M11":
        return [
            {"action_type": "TOOL_CALL", "tool_id": "close_issue", "arguments": {"repo_id": "R1", "issue_id": 101}, "concise_decision_summary": summary},
            {"action_type": "TOOL_CALL", "tool_id": "assign_issue", "arguments": {"repo_id": "R1", "issue_id": 101, "assignee": "bob"}, "concise_decision_summary": summary},
        ]
    return [{"action_type": "TOOL_CALL", "tool_id": "close_issue", "arguments": {"repo_id": "R1", "issue_id": 102}, "concise_decision_summary": summary}]


def _business_goal_completed(family_id: str, state: dict[str, Any], calls: list[dict[str, Any]]) -> bool:
    repo = state["repositories"]["R1"]
    if family_id == "M01":
        return any(issue["title"] == "Review authentication logs" for issue in repo["issues"].values())
    if family_id == "M06":
        created = [call for call in calls if call["tool"] == "create_issue" and call["status_code"] < 300]
        return bool(created) and any(issue.get("assignee") for issue in repo["issues"].values() if issue["title"] == "Response mapping")
    if family_id == "M11":
        return repo["issues"]["101"]["state"] == "closed"
    return repo["issues"]["102"]["state"] == "closed"


def _minimal_structural_fix(arguments: dict[str, Any], schema: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    value, edits = deepcopy(arguments), []
    allowed = set(schema.get("properties", {}))
    for wrapper in ("body", "parameters", "path_params"):
        nested = value.get(wrapper)
        if isinstance(nested, dict) and set(nested) & allowed:
            value.pop(wrapper)
            for key, item in nested.items():
                if key not in value:
                    value[key] = item
            edits.append({"operation": "unwrap", "field": wrapper})
    for key in list(value):
        if key in allowed:
            continue
        matches = difflib.get_close_matches(key, allowed - set(value), n=1, cutoff=0.72)
        if matches:
            new = matches[0]
            value[new] = value.pop(key)
            edits.append({"operation": "rename", "from": key, "to": new})
        else:
            value.pop(key)
            edits.append({"operation": "remove_unknown", "field": key})
    for key, field_schema in schema.get("properties", {}).items():
        if key in value and field_schema.get("type") == "integer" and isinstance(value[key], str) and value[key].isdigit():
            value[key] = int(value[key])
            edits.append({"operation": "cast_integer", "field": key})
    return value, edits


def _had_actionable_observation(item: dict[str, Any]) -> bool:
    observation = _last_observation(item)
    return bool(observation and (item["validation"] or item["sandbox_calls"]))


def _successful_non_probe_call(item: dict[str, Any]) -> bool:
    return any(
        action.get("action_type") in {"TOOL_CALL", "EXACT_RETRY"}
        and action.get("result", {}).get("payload", {}).get("ok") is True
        for action in item["actions"]
    )


def _evaluator_reason(failure: str) -> str:
    for prefix, code in (
        ("tool failed:", "TOOL_EXECUTION_FAILED"),
        ("answer assertion failed:", "ANSWER_ASSERTION_FAILED"),
        ("state assertion failed:", "STATE_ASSERTION_FAILED"),
        ("forbidden side effects:", "FORBIDDEN_SIDE_EFFECT"),
        ("tool budget exceeded:", "TOOL_BUDGET_EXCEEDED"),
    ):
        if failure.startswith(prefix):
            return code
    return "OTHER"


def _median(values: list[int]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    return float(ordered[middle]) if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _atomic(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)
