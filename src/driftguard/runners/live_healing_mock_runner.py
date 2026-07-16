from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from driftguard.agents import ToolAgentController
from driftguard.agents.capabilities import PolicyCapabilities
from driftguard.agents.policies import POLICIES
from driftguard.contracts.loader import DEFAULT_FIXTURE_PATH, PROJECT_ROOT
from driftguard.experiments.budgets import BudgetTracker, ExperimentBudget
from driftguard.experiments.task_resolver import TaskInstanceResolver
from driftguard.healing.category_generators import operation_path_for_tool
from driftguard.live import LiveEvidenceBridge, LiveHistoryStore, LiveScope
from driftguard.live.orchestrator import LiveHealingOrchestrator
from driftguard.live.provider_adapter import EvidenceRefMockProvider, FocusedProviderAdapter
from driftguard.live.stages import LiveStructuredStages
from driftguard.llm import LLMCache, MockProvider, ModelConfig
from driftguard.runtime import ExecutionContext, ExecutionMode, ExecutionProfile
from driftguard.sandbox import SandboxService


MATCHED = PROJECT_ROOT / "benchmark/matched_failures/matched_failures_v1.json"
DRIFTS = PROJECT_ROOT / "benchmark/drifts/drift_cases_v1.json"
SCHEMA = PROJECT_ROOT / "benchmark/schemas/agent_action_schema_v3.json"
PROMPT = PROJECT_ROOT / "benchmark/prompts/base_tool_agent_v3.txt"
POLICY_PROMPT = PROJECT_ROOT / "benchmark/prompts/driftguard_attribution_v1.txt"


class LiveHealingMockRunner:
    FAMILIES = ("M01", "M06", "M11", "M16")

    def __init__(self) -> None:
        matched = json.loads(MATCHED.read_text(encoding="utf-8"))["families"]
        cases = json.loads(DRIFTS.read_text(encoding="utf-8"))["cases"]
        self.families = {item["matched_case_id"]: item for item in matched}
        self.cases = {item["drift_id"]: item for item in cases}
        self.resolver = TaskInstanceResolver()
        self.action_schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
        self.base_prompt = PROMPT.read_text(encoding="utf-8")
        self.policy_prompt = POLICY_PROMPT.read_text(encoding="utf-8")
        self.initial_state = json.loads(Path(DEFAULT_FIXTURE_PATH).read_text(encoding="utf-8"))

    def run(self) -> dict[str, Any]:
        rows = [self.run_pd(family_id) for family_id in self.FAMILIES]
        return {
            "run_version": "phase10a-live-healing-mock-v1",
            "real_provider_calls": 0,
            "symbolic_fallback_count": sum(row["symbolic_fallback_used"] for row in rows),
            "summary": {
                "pd_chains": len(rows),
                "accepted": sum(row["patch_registry_state"]["accepted_count"] == 1 for row in rows),
                "immediate_repair": sum(bool(row["immediate_repair"] and row["immediate_repair"]["passed"]) for row in rows),
                "future_transfer": sum(bool(row["future_transfer"] and row["future_transfer"]["patched_success"]) for row in rows),
                "safe_probes": sum(bool(row["probe_result"] and row["probe_result"]["state_unchanged"]) for row in rows),
            },
            "records": rows,
        }

    def run_pd(
        self, family_id: str, *, stage_outputs: list[dict[str, Any]] | None = None,
        validator=None, model_config: ModelConfig | None = None,
        provider_adapter: FocusedProviderAdapter | None = None,
        cache: LLMCache | None = None, config_hash: str = "live-healing-v1",
        repetition: int = 0, seed: int = 20260715,
    ) -> dict[str, Any]:
        family = self.families[family_id]
        scenario = next(item for item in family["scenarios"] if item["variant_code"] == "PD")
        case = self.cases[family["source_drift_id"]]
        resolved = self.resolver.resolve(family, scenario)
        task = resolved.evaluator_task()
        action_step = next(step for step in resolved.oracle_plan if step["tool"] == case["target_tool"])
        arguments = deepcopy(action_step["arguments"])
        public_id = resolved.public_scenario_id
        first_context = self._context(case, family, 3)
        displayed = first_context.displayed_contract
        configured_model = model_config or ModelConfig(provider="mock", model_id="mock-live-agent")
        scope = LiveScope(public_id, configured_model.provider, "driftguard_llm", repetition)
        bridge = LiveEvidenceBridge(scope, displayed)
        tracker = BudgetTracker(ExperimentBudget(
            max_llm_calls=10, max_tool_calls=10, max_probe_calls=2,
            max_patch_proposal_calls=2, max_total_interactions=20,
            max_input_tokens=240_000, max_output_tokens=20_000,
            max_format_repairs=1,
        ))
        latest_service = None
        agent_provider_calls = 0
        controller_runs = []
        for episode in (3, 4):
            context = self._context(case, family, episode)
            service = SandboxService(execution_context=context)
            action_outputs = [
                {"action_type": "TOOL_CALL", "tool_id": case["target_tool"], "arguments": arguments,
                 "concise_decision_summary": "execute the visible task"},
                {"action_type": "ABSTAIN",
                 "concise_decision_summary": "stop this independent execution instance"},
            ]
            provider = (
                provider_adapter.create(action_outputs) if provider_adapter
                else MockProvider(action_outputs, model="mock-live-agent")
            )
            controller = ToolAgentController(
                provider, configured_model,
                POLICIES["driftguard_llm"](), service, tracker,
                PolicyCapabilities.for_method("driftguard_llm").narrow_schema(self.action_schema),
                self._render_action_prompt(), self.policy_prompt,
                cache=cache, config_hash=config_hash, run_mode="focused_live_action",
                live_evidence_bridge=bridge,
                execution_context_id=f"{public_id}:independent-episode-{episode}",
            )
            controller_result = controller.run(task, displayed, public_id, episode, repetition)
            controller_runs.append({
                "episode": episode,
                "task_success": controller_result.task_success,
                "termination_reason": controller_result.termination_reason,
                "error_category": controller_result.error_category,
            })
            agent_provider_calls += int(getattr(provider, "calls", 0))
            latest_service = service
        outputs = stage_outputs or self._stage_outputs(case, displayed)
        stage_provider = (
            provider_adapter.create(outputs, resolve_evidence_refs=True) if provider_adapter
            else EvidenceRefMockProvider(outputs, model="mock-live-diagnosis-and-patch")
        )
        stages = LiveStructuredStages(
            stage_provider, tracker, seed=seed, repetition=repetition, cache=cache,
            model_config=configured_model, config_hash=config_hash,
        )
        orchestrator = LiveHealingOrchestrator(
            stages, tracker, history=LiveHistoryStore(), validator=validator,
        )

        def context_factory():
            return self._context(case, family, 5)

        def probe_operation(service):
            return service.call_tool(case["target_tool"], deepcopy(arguments), resolved.actor_id)

        row = orchestrator.run(
            bridge, latest_service, context_factory, arguments, self.initial_state,
            probe_operation=probe_operation, future_arguments=arguments,
        )
        row.update({
            "family": family_id, "public_scenario_id": public_id,
            "drift_category": self._category(case["drift_type"]),
            "live_evidence_events": len(bridge.trace.events),
            "live_evidence_trace": bridge.trace.to_dict(),
            "llm_calls": tracker.llm_calls, "tool_calls": tracker.tool_calls,
            "probe_calls": tracker.probe_calls, "patch_proposal_calls": tracker.patch_proposal_calls,
            "input_tokens": tracker.input_tokens, "output_tokens": tracker.output_tokens,
            "provider_calls": int(getattr(stage_provider, "calls", 0)) + agent_provider_calls,
            "llm_stage_calls": deepcopy(stages.call_records),
            "controller_runs": controller_runs,
        })
        return row

    @staticmethod
    def _context(case, family, episode):
        context = ExecutionContext(ExecutionProfile(
            ExecutionMode.PERSISTENT_DRIFT, case, family, change_point=int(case["change_point"]),
        ))
        context.set_episode(episode)
        return context

    def _render_action_prompt(self):
        capabilities = PolicyCapabilities.for_method("driftguard_llm")
        schema = capabilities.narrow_schema(self.action_schema)
        return self.base_prompt.replace("{{OUTPUT_SCHEMA}}", json.dumps(schema, sort_keys=True)).replace(
            "{{POLICY_CAPABILITIES}}", capabilities.prompt_fragment()
        )

    def _stage_outputs(self, case, displayed):
        category = self._category(case["drift_type"])
        location = case["runtime_mutation"]["target"]
        target = case["target_tool"]
        refs = ["@FAILURE_1", "@FAILURE_2"]
        preliminary = self._attribution(target, category, location, refs, requested={"needed": True})
        probe_type = {"ICD": "local_schema_check", "RSD": "response_shape_inspection", "WPD": "precondition_inspection", "SED": "read_after_write"}[category]
        selection = {
            "decision": "SELECT_PROBE", "probe_type": probe_type, "target_tool_id": target,
            "evidence_refs": refs, "concise_reason": "distinguish a persistent visible contract mismatch",
        }
        final = self._attribution(target, category, location, [*refs, "@PROBE"], requested=None)
        patch = self._patch_output(category, target, location, displayed, [*refs, "@PROBE"])
        return [preliminary, selection, final, patch]

    @staticmethod
    def _attribution(target, category, location, refs, requested):
        return {
            "predicted_class": "PERSISTENT_DRIFT", "target_tool_id": target,
            "drift_category": category, "location_type": {
                "ICD": "request_schema", "RSD": "response_schema",
                "WPD": "workflow_precondition", "SED": "state_effect",
            }[category],
            "location_path": location, "evidence_refs": refs,
            "requested_probe": requested, "confidence": 0.95,
            "concise_reason": "two independent live failures conflict with the displayed contract",
        }

    @staticmethod
    def _patch_output(category, target, location, displayed, refs):
        if category == "ICD":
            path, value = "/components/schemas/CreateIssueRequest/required/-", "priority"
            semantics = {"operation": "add_required", "target": "/adapter/create_issue/required", "before": ["title"], "after": ["title", "priority"], "request_transform": {"kind": "supply_required", "field": "priority", "value": "medium"}}
        elif category == "RSD":
            path, value = "/components/schemas/Issue/x-driftguard-runtime-path", {"issue_id": "id"}
            semantics = {"operation": "rename_output_field", "target": "/adapter/create_issue/output", "before": "id", "after": "issue_id", "response_mapping": {"issue_id": "id"}}
        elif category == "WPD":
            path = operation_path_for_tool(displayed, target) + "/x-driftguard-preconditions-v2"
            value = {"kind": "active_assignee", "read": "get_issue", "write": "assign_issue"}
            semantics = {"operation": "add_prerequisite", "target": "/adapter/close_issue/workflow", "before": ["issue exists and is open"], "after": ["get_issue", "assign_issue_if_null", "close_issue"], "workflow": value}
        else:
            path = operation_path_for_tool(displayed, target) + "/x-driftguard-observation-policy"
            value = {"kind": "read_after_write", "confirmation_tool": "get_issue", "condition": "state=closed"}
            semantics = {"operation": "add_postcondition_verification", "target": "/adapter/close_issue/postcondition", "before": ["close_issue"], "after": ["close_issue", "get_issue", "verify state=closed"], "observation_policy": value}
        return {
            "target_tool_id": target, "drift_category": category, "location_path": location,
            "openapi_operations": [{"op": "add", "path": path, "value": value, "evidence_refs": refs, "reason_code": "LIVE_EVIDENCE_MINIMAL_CHANGE"}],
            "semantic_extensions": {"x-driftguard-patch-semantics": semantics},
            "evidence_refs": refs,
            "expected_agent_behavior_change": "adapt the displayed tool contract using only the cited live evidence",
            "confidence": 0.95, "concise_reason": "minimal target-scoped live-evidence patch",
        }

    @staticmethod
    def _category(drift_type):
        return {"input_contract": "ICD", "response_shape": "RSD", "workflow_precondition": "WPD", "state_effect": "SED"}[drift_type]
