from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import socket
from typing import Any, Iterator

from driftguard.agents import ToolAgentController
from driftguard.agents.capabilities import PolicyCapabilities
from driftguard.agents.policies import POLICIES
from driftguard.agents.tool_catalog import ToolCatalogRenderer
from driftguard.contracts.loader import DEFAULT_OPENAPI_PATH, PROJECT_ROOT, load_openapi
from driftguard.contracts.registry import ContractRegistry
from driftguard.experiments.budgets import BudgetTracker, ExperimentBudget
from driftguard.experiments.task_resolver import MATCHED_PATH, ResolvedTaskInstance, TaskInstanceResolver
from driftguard.llm import MockProvider, ModelConfig
from driftguard.runners.injection_conformance_runner import CASE_ARGUMENTS, PROTECTED_PATHS, protected_hashes
from driftguard.runtime import ExecutionContext, ExecutionMode, ExecutionProfile
from driftguard.sandbox import SandboxService


DRIFTS_PATH = PROJECT_ROOT / "benchmark" / "drifts" / "drift_cases_v1.json"
PROMPT_PATH = PROJECT_ROOT / "benchmark" / "prompts" / "base_tool_agent_v3.txt"
SCHEMA_PATH = PROJECT_ROOT / "benchmark" / "schemas" / "agent_action_schema_v3.json"
LEDGER_PATH = PROJECT_ROOT / "results" / "experiments" / "phase10" / "cost" / "ledger.json"
EXPECTED_SPENT_CNY = 3.497508830
EXPECTED_API_ATTEMPTS = 736


class NetworkAccessBlocked(RuntimeError):
    pass


@contextmanager
def blocked_network() -> Iterator[None]:
    original_connect = socket.socket.connect
    original_create = socket.create_connection

    def reject(*_args, **_kwargs):
        raise NetworkAccessBlocked("offline Phase 10 repair forbids network access")

    socket.socket.connect = reject  # type: ignore[method-assign]
    socket.create_connection = reject  # type: ignore[assignment]
    try:
        yield
    finally:
        socket.socket.connect = original_connect  # type: ignore[method-assign]
        socket.create_connection = original_create  # type: ignore[assignment]


class RecordingMockProvider(MockProvider):
    def __init__(self, outputs, model="phase10-offline-oracle"):
        super().__init__(outputs, model)
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        return super().complete(request)


class OfflineRepairGate:
    def __init__(self) -> None:
        self.resolver = TaskInstanceResolver()
        self.families = json.loads(MATCHED_PATH.read_text(encoding="utf-8"))["families"]
        self.cases = {item["drift_id"]: item for item in json.loads(DRIFTS_PATH.read_text(encoding="utf-8"))["cases"]}
        self.schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        self.prompt_template = PROMPT_PATH.read_text(encoding="utf-8")
        self.policy_prompts = {
            "reflection": (PROJECT_ROOT / "benchmark/prompts/reflection_v1.txt").read_text(encoding="utf-8"),
            "validation_guided": (PROJECT_ROOT / "benchmark/prompts/validation_guided_v1.txt").read_text(encoding="utf-8"),
            "driftguard_llm": (PROJECT_ROOT / "benchmark/prompts/driftguard_attribution_v1.txt").read_text(encoding="utf-8"),
        }
        self.model = ModelConfig(provider="mock", model_id="phase10-offline", temperature=0.0, seed=20260715)
        self.budget = ExperimentBudget(
            max_llm_calls=6, max_tool_calls=8, max_probe_calls=3, max_total_interactions=12,
            max_input_tokens=200_000, max_output_tokens=20_000,
        )

    def run(self) -> dict[str, Any]:
        hashes_before = protected_hashes()
        ledger_before = _sha256(LEDGER_PATH)
        ledger_value = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
        with blocked_network():
            provenance, consistency = self.task_consistency()
            catalog = self.catalog_gate()
            oracle = self.oracle_gate()
            mock = self.mock_feedback_gate()
        hashes_after = protected_hashes()
        ledger_after = _sha256(LEDGER_PATH)
        return {
            "report_version": "phase10a-end-to-end-offline-repair-v1",
            "behavior_affecting_prompt_format_change": True,
            "real_provider_calls": 0,
            "new_cost_cny": 0.0,
            "cost_ledger": {
                "spent_cny": ledger_value["spent_cny"],
                "api_attempts": ledger_value["api_attempts"],
                "expected_spent_cny": EXPECTED_SPENT_CNY,
                "expected_api_attempts": EXPECTED_API_ATTEMPTS,
                "baseline_or_later": (
                    ledger_value["spent_cny"] + 1e-12 >= EXPECTED_SPENT_CNY
                    and ledger_value["api_attempts"] >= EXPECTED_API_ATTEMPTS
                ),
                "sha256_before": ledger_before,
                "sha256_after": ledger_after,
                "unchanged": ledger_before == ledger_after,
            },
            "m16_root_cause": {
                "protected_benchmark_conflict": False,
                "canonical_source_task": "A07",
                "canonical_issue_id": 102,
                "adapter_error_location": "src/driftguard/experiments/scenario_runner.py::_legacy_smoke_task (historical only)",
                "cross_layer_source": "src/driftguard/runners/injection_conformance_runner.py::CASE_ARGUMENTS['SED-01'].issue_id",
                "cross_layer_value": CASE_ARGUMENTS["SED-01"]["issue_id"],
                "repair": "active end-to-end path resolves SED-01-D1 -> A07 and never uses conformance CASE_ARGUMENTS",
            },
            "task_consistency": consistency,
            "provenance": provenance,
            "tool_catalog": catalog,
            "offline_oracle": oracle,
            "mock_feedback": mock,
            "prompt": {
                "path": str(PROMPT_PATH.relative_to(PROJECT_ROOT)),
                "sha256": _sha256(PROMPT_PATH),
                "schema_path": str(SCHEMA_PATH.relative_to(PROJECT_ROOT)),
                "schema_sha256": _sha256(SCHEMA_PATH),
                "effective_prompt_hashes": {
                    method: self._effective_prompt_hash(method, load_openapi())
                    for method in ("standard", "retry_only", "reflection", "validation_guided", "driftguard_llm")
                },
            },
            "protection": {
                "canonical_hashes_before": hashes_before,
                "canonical_hashes_after": hashes_after,
                "canonical_unchanged": hashes_before == hashes_after,
                "protected_paths": list(PROTECTED_PATHS),
            },
            "pilot_v4_created_or_run": (PROJECT_ROOT / "results/experiments/phase10/pilot_v4_canary/manifest.json").exists(),
            "full_real_model_experiment_status": "NOT RUN",
        }

    def task_consistency(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        by_family: dict[str, list[ResolvedTaskInstance]] = {}
        unresolved = nl_mismatch = hidden = 0
        for family in self.families:
            for scenario in family["scenarios"]:
                resolved = self.resolver.resolve(family, scenario)
                by_family.setdefault(family["matched_case_id"], []).append(resolved)
                placeholders = bool(re.search(r"\{\{[^}]+\}\}|\$\{[^}]+\}", resolved.instruction))
                unresolved += int(placeholders)
                issue_ids = [int(value) for value in re.findall(r"\bissue\s+(\d+)\b", resolved.instruction, re.I)]
                evaluator_issue_ids = sorted({
                    int(match.group(1)) for assertion in resolved.expected_state
                    if (match := re.search(r"/issues/(\d+)(?:/|$)", assertion["path"]))
                })
                mismatch = bool(issue_ids and not set(issue_ids).issubset(evaluator_issue_ids))
                nl_mismatch += int(mismatch)
                repo_id = resolved.bindings.get("repo_id")
                hidden_requirement = bool(repo_id and str(repo_id) not in resolved.instruction)
                hidden += int(hidden_requirement)
                case = self.cases[family["source_drift_id"]]
                initial = self.resolver.fixture
                initial_entity = {
                    "repository_exists": repo_id in initial.get("repositories", {}),
                    "issue_ids_exist": {
                        str(issue_id): str(issue_id) in initial["repositories"][repo_id]["issues"]
                        for issue_id in issue_ids if repo_id in initial.get("repositories", {})
                    },
                }
                rows.append({
                    "family": family["matched_case_id"],
                    "scenario_public_id": resolved.public_scenario_id,
                    "source_task_ref": resolved.source_task_ref,
                    "task_instance_ref": resolved.provenance["task_instance_ref"],
                    "natural_language_entity": {"issue_ids": issue_ids, "repository_name": resolved.bindings.get("repository_name")},
                    "resolved_binding": deepcopy(resolved.bindings),
                    "evaluator_target_entity": {"issue_ids": evaluator_issue_ids, "state_paths": [item["path"] for item in resolved.expected_state]},
                    "drift_target_tool": case["target_tool"],
                    "benchmark_arguments_entity": deepcopy(CASE_ARGUMENTS[case["drift_id"]]),
                    "initial_state_entity": initial_entity,
                    "unresolved_binding": placeholders,
                    "natural_language_evaluator_mismatch": mismatch,
                    "hidden_entity_requirement": hidden_requirement,
                })
        variant_mismatch = 0
        for values in by_family.values():
            goals = {
                json.dumps({
                    "instruction": item.instruction,
                    "state": item.expected_state,
                    "answer": item.expected_answer,
                    "forbidden": item.forbidden_side_effects,
                }, sort_keys=True)
                for item in values
            }
            variant_mismatch += int(len(goals) != 1)
        return rows, {
            "scenarios": len(rows), "families": len(by_family),
            "unresolved_bindings": unresolved,
            "natural_language_evaluator_target_mismatches": nl_mismatch,
            "variant_goal_mismatches": variant_mismatch,
            "hidden_entity_requirements": hidden,
            "passed": len(rows) == 60 and not any((unresolved, nl_mismatch, variant_mismatch, hidden)),
        }

    def catalog_gate(self) -> dict[str, Any]:
        renderer = ToolCatalogRenderer()
        spec = load_openapi()
        catalog = renderer.render(spec)
        registry = ContractRegistry(spec)
        diffs = []
        for tool in catalog["tools"]:
            expected = registry.require(tool["tool_id"]).input_schema
            if tool["input_schema"] != expected:
                diffs.append({"tool_id": tool["tool_id"], "rendered": tool["input_schema"], "registry": expected})
        encoded = json.dumps(catalog, sort_keys=True).lower()
        leakage = [term for term in ("runtime_profile", "runtime_contract", "source_drift_id", "expected_patch", "ground_truth") if term in encoded]
        return {
            "tools": len(catalog["tools"]),
            "input_schema_diffs": len(diffs),
            "diff_details": diffs,
            "catalog_fingerprint": renderer.fingerprint(spec),
            "stable_serialization": renderer.stable_json(spec) == renderer.stable_json(deepcopy(spec)),
            "unresolved_refs": "$ref" in encoded,
            "leakage_terms": leakage,
            "passed": len(catalog["tools"]) == 12 and not diffs and "$ref" not in encoded and not leakage,
        }

    def oracle_gate(self) -> dict[str, Any]:
        selected = [family for family in self.families if family["matched_case_id"] in {"M01", "M06", "M11", "M16"}]
        rows = []
        for family in selected:
            case = self.cases[family["source_drift_id"]]
            for scenario in family["scenarios"]:
                resolved = self.resolver.resolve(family, scenario)
                actions = self._oracle_actions(resolved, case, scenario["variant_code"])
                context = ExecutionContext(ExecutionProfile(ExecutionMode(scenario["variant_code"]), case, family, change_point=int(case["change_point"])))
                context.set_episode(3)
                service = SandboxService(execution_context=context)
                capabilities = PolicyCapabilities.for_method("standard")
                schema = capabilities.narrow_schema(self.schema)
                provider = RecordingMockProvider(actions)
                prompt = self._prompt("standard", schema)
                controller = ToolAgentController(
                    provider, self.model, POLICIES["standard"](), service, BudgetTracker(self.budget),
                    schema, prompt, "", None, False, "offline-repair", "end_to_end_offline_oracle",
                )
                result = controller.run(resolved.evaluator_task(), context.displayed_contract, resolved.public_scenario_id, 3, 0)
                natural_goal = self._natural_goal(resolved, service.call_log(), result.task_evaluation)
                rows.append({
                    "scenario": scenario["scenario_id"], "source_task_ref": resolved.source_task_ref,
                    "task_evaluator_success": result.task_success,
                    "natural_language_business_goal": natural_goal,
                    "forbidden_side_effects_passed": result.task_evaluation["forbidden_assertions_passed"],
                    "binding_consistent": self._binding_consistent(resolved, service.call_log()),
                    "tool_budget_passed": result.task_evaluation["within_tool_budget"],
                    "tool_calls": result.tool_calls, "termination": result.termination_reason,
                    "oracle_actions_in_initial_prompt": self._oracle_leaked(provider, actions),
                })
        return {
            "scenarios": len(rows),
            "evaluator_success": sum(item["task_evaluator_success"] for item in rows),
            "natural_language_business_goal": sum(item["natural_language_business_goal"] for item in rows),
            "forbidden_side_effect_failures": sum(not item["forbidden_side_effects_passed"] for item in rows),
            "binding_consistency": sum(item["binding_consistent"] for item in rows),
            "tool_budget_pass": sum(item["tool_budget_passed"] for item in rows),
            "oracle_prompt_leaks": sum(item["oracle_actions_in_initial_prompt"] for item in rows),
            "provider_calls": 0, "details": rows,
        }

    def mock_feedback_gate(self) -> dict[str, Any]:
        family = next(item for item in self.families if item["matched_case_id"] == "M16")
        scenario = family["scenarios"][0]
        resolved = self.resolver.resolve(family, scenario)
        rows = []
        for method in ("standard", "retry_only", "reflection", "validation_guided", "driftguard_llm"):
            capabilities = PolicyCapabilities.for_method(method)
            schema = capabilities.narrow_schema(self.schema)
            bad = _action("close_issue", {"repo_id": "R1", "issue_id": "102", "resolution": "completed"}, "first schema-valid action has the wrong argument type")
            corrected = _action("close_issue", {"repo_id": "R1", "issue_id": 102, "resolution": "completed"}, "correct the visible field type after validation feedback")
            provider = RecordingMockProvider([bad, corrected], f"offline-{method}")
            service = SandboxService()
            controller = ToolAgentController(
                provider, self.model, POLICIES[method](), service, BudgetTracker(self.budget), schema,
                self._prompt(method, schema), "", None, False, "offline-feedback", "end_to_end_offline_mock",
            )
            result = controller.run(resolved.evaluator_task(), load_openapi(), resolved.public_scenario_id, 3, 0)
            second_prompt = provider.requests[1].messages[-1]["content"] if len(provider.requests) > 1 else ""
            rows.append({
                "method": method, "rounds": len(provider.requests), "task_success": result.task_success,
                "sandbox_calls": len(service.call_log()),
                "exact_retry_after_local_validation": any(item["action_type"] == "EXACT_RETRY" for item in result.actions),
                "reflection_hook_visible": '"reflection"' in second_prompt,
                "field_error_visible": "invalid_field_path" in second_prompt,
                "schema_fragment_visible": "displayed_schema_fragment" in second_prompt,
                "driftguard_evidence_visible": '"evidence"' in second_prompt,
                "driftguard_evidence_events": len(controller.last_evidence_trace.events) if controller.last_evidence_trace else 0,
                "probe_capability_visible": "REQUEST_PROBE" in self._prompt(method, schema),
                "validation_feedback_transition": any(item["to_state"] == "VALIDATION_FEEDBACK" for item in result.state_transitions),
                "next_model_transition": any(item["to_state"] == "NEXT_MODEL_ACTION" for item in result.state_transitions),
            })
        return {
            "methods": len(rows), "successful_two_round_repairs": sum(item["task_success"] for item in rows),
            "only_driftguard_sees_probe": [item["method"] for item in rows if item["probe_capability_visible"]] == ["driftguard_llm"],
            "network_calls": 0, "details": rows,
        }

    def _prompt(self, method: str, schema: dict[str, Any]) -> str:
        capabilities = PolicyCapabilities.for_method(method)
        base = self.prompt_template.replace("{{OUTPUT_SCHEMA}}", json.dumps(schema, indent=2, sort_keys=True)).replace(
            "{{POLICY_CAPABILITIES}}", capabilities.prompt_fragment()
        )
        return base + "\n" + self.policy_prompts.get(method, "")

    def _effective_prompt_hash(self, method: str, displayed_spec: dict[str, Any]) -> str:
        capabilities = PolicyCapabilities.for_method(method)
        schema = capabilities.narrow_schema(self.schema)
        material = {
            "system_prompt": self._prompt(method, schema),
            "action_schema": schema,
            "policy_capabilities": {"method": method, "allowed_actions": capabilities.allowed_actions},
            "method_policy": self.policy_prompts.get(method, ""),
            "tool_catalog": ToolCatalogRenderer().render(displayed_spec),
        }
        return hashlib.sha256(json.dumps(material, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def _oracle_actions(self, resolved: ResolvedTaskInstance, case: dict[str, Any], variant_code: str) -> list[dict[str, Any]]:
        steps = [deepcopy(step) for step in resolved.oracle_plan]
        target = case["target_tool"]
        if case["drift_type"] == "workflow_precondition":
            target_step = next(step for step in steps if step["tool"] == target)
            issue_id = target_step["arguments"]["issue_id"]
            repo_id = target_step["arguments"]["repo_id"]
            assignee = next(
                item["value"] for item in resolved.expected_state
                if item["path"].endswith(f"/issues/{issue_id}/assignee") and item["operator"] == "eq"
            )
            steps.insert(0, {"tool": "assign_issue", "arguments": {"repo_id": repo_id, "issue_id": issue_id, "assignee": assignee}})
        if case["drift_type"] == "state_effect":
            target_step = next(step for step in steps if step["tool"] == target)
            if target == "close_issue":
                steps.append({"tool": "get_issue", "arguments": {key: target_step["arguments"][key] for key in ("repo_id", "issue_id")}})
        actions = []
        canonical = ContractRegistry(load_openapi())
        displayed = ExecutionContext(ExecutionProfile(ExecutionMode.AGENT_ERROR, case, next(f for f in self.families if f["source_drift_id"] == case["drift_id"]))).displayed_contract
        displayed_registry = ContractRegistry(displayed)
        for step in steps:
            arguments = self._resolve_oracle_bindings(step["arguments"], resolved.bindings)
            if case["drift_type"] == "input_contract" and step["tool"] == target:
                displayed_contract = displayed_registry.require(target)
                canonical_contract = canonical.require(target)
                for field in displayed_contract.input_schema["required"]:
                    if field not in arguments:
                        field_schema = canonical_contract.input_schema["properties"][field]
                        if "default" not in field_schema:
                            raise ValueError(f"offline Oracle cannot derive required field {field}")
                        arguments[field] = deepcopy(field_schema["default"])
            actions.append(_action(step["tool"], arguments, "offline Oracle executes the resolved formal task"))
            if variant_code == "AE" and case["drift_type"] == "input_contract" and step["tool"] == target:
                actions.append(_action(
                    step["tool"], deepcopy(arguments),
                    "offline Oracle corrects the one-shot Agent fault using the displayed contract",
                ))
        return actions

    @staticmethod
    def _resolve_oracle_bindings(value: Any, bindings: dict[str, Any]) -> Any:
        if isinstance(value, str) and value.startswith("$steps."):
            if value.endswith(".issue_id") and "created_issue_id" in bindings:
                return bindings["created_issue_id"]
            raise ValueError(f"unresolved offline Oracle binding: {value}")
        if isinstance(value, dict):
            return {key: OfflineRepairGate._resolve_oracle_bindings(child, bindings) for key, child in value.items()}
        return deepcopy(value)

    @staticmethod
    def _binding_consistent(resolved: ResolvedTaskInstance, calls: list[dict[str, Any]]) -> bool:
        dynamic = any(
            isinstance(value, str) and value.startswith("$steps.")
            for step in resolved.oracle_plan for value in step["arguments"].values()
        )
        if not dynamic:
            return True
        created = [call for call in calls if call["tool"] == "create_issue" and call["runtime_response"]["payload"]["ok"]]
        assigned = [call for call in calls if call["tool"] == "assign_issue" and call["runtime_response"]["payload"]["ok"]]
        if not created or not assigned:
            return False
        created_id = created[-1]["canonical_response"]["payload"]["data"]["issue_id"]
        return assigned[-1]["arguments"]["issue_id"] == created_id

    @staticmethod
    def _natural_goal(resolved: ResolvedTaskInstance, calls: list[dict[str, Any]], evaluation: dict[str, Any]) -> bool:
        if not evaluation["state_assertions_passed"] or not evaluation["answer_assertions_passed"]:
            return False
        planned = [step["tool"] for step in resolved.oracle_plan]
        executed = [call["tool"] for call in calls if call["runtime_response"]["payload"]["ok"]]
        return all(tool in executed for tool in planned) and OfflineRepairGate._binding_consistent(resolved, calls)

    @staticmethod
    def _oracle_leaked(provider: RecordingMockProvider, actions: list[dict[str, Any]]) -> bool:
        if not provider.requests:
            return False
        initial = json.dumps(provider.requests[0].messages, sort_keys=True)
        return any(json.dumps(action, sort_keys=True) in initial for action in actions)


def _action(tool_id: str, arguments: dict[str, Any], summary: str) -> dict[str, Any]:
    return {"action_type": "TOOL_CALL", "tool_id": tool_id, "arguments": arguments, "concise_decision_summary": summary}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_offline_repair_report(output: Path) -> tuple[Path, dict[str, Any]]:
    report = OfflineRepairGate().run()
    output.mkdir(parents=True, exist_ok=True)
    path = output / "end_to_end_offline_repair_v1.json"
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    provenance_path = output / "scenario_task_provenance_v1.json"
    temporary = provenance_path.with_name("." + provenance_path.name + ".tmp")
    temporary.write_text(json.dumps(report["provenance"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(provenance_path)
    return path, report
