from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from driftguard.agents.capabilities import PolicyCapabilities
from driftguard.agents.tool_catalog import ToolCatalogRenderer
from driftguard.contracts.loader import PROJECT_ROOT, load_openapi
from driftguard.contracts.registry import ContractRegistry
from driftguard.evidence.leakage_guard import EvidenceLeakageError
from driftguard.experiments.runner import _load_schema
from driftguard.experiments.scenario_runner import ExperimentScenarioRunner
from driftguard.experiments.task_resolver import MATCHED_PATH, TaskInstanceResolver
from driftguard.llm.prompt_loader import PromptLoader
from driftguard.phase10.config import Phase10Config
from driftguard.phase10.offline_repair import EXPECTED_API_ATTEMPTS, EXPECTED_SPENT_CNY, OfflineRepairGate
from driftguard.phase10.offline_repair import NetworkAccessBlocked, blocked_network
from driftguard.runtime import ExecutionContext, ExecutionMode, ExecutionProfile
from driftguard.sandbox import SandboxService
from driftguard.sandbox.evaluator import TaskEvaluator
from jsonschema import Draft202012Validator, ValidationError


def _families():
    return json.loads(MATCHED_PATH.read_text(encoding="utf-8"))["families"]


def test_all_60_tasks_resolve_without_hidden_entities_or_variant_goal_changes():
    gate = OfflineRepairGate()
    rows, summary = gate.task_consistency()
    assert len(rows) == 60 and summary["passed"]
    assert summary["unresolved_bindings"] == 0
    assert summary["natural_language_evaluator_target_mismatches"] == 0
    assert summary["variant_goal_mismatches"] == 0
    assert summary["hidden_entity_requirements"] == 0


def test_m16_conflict_is_adapter_level_and_resolves_to_canonical_issue_102():
    resolver = TaskInstanceResolver()
    family = next(item for item in _families() if item["matched_case_id"] == "M16")
    resolved = [resolver.resolve(family, scenario) for scenario in family["scenarios"]]
    assert {item.source_task_ref for item in resolved} == {"A07"}
    assert {item.bindings["issue_id"] for item in resolved} == {102}
    assert all("issue 102" in item.instruction.lower() for item in resolved)
    assert all(any("/issues/102/" in assertion["path"] for assertion in item.expected_state) for item in resolved)
    assert all(item.bindings["repo_id"] in item.instruction for item in resolved)


def test_multistep_and_created_id_goals_are_real_state_goals_not_tool_succeeded():
    resolver = TaskInstanceResolver()
    for family_id in ("M01", "M06", "M11", "M16"):
        family = next(item for item in _families() if item["matched_case_id"] == family_id)
        task = resolver.resolve(family, family["scenarios"][0]).evaluator_task()
        assert task["success_assertions"]
        assert not any(item.get("key") == "tool_succeeded" for item in task["answer_assertions"])
    m06 = next(item for item in _families() if item["matched_case_id"] == "M06")
    resolved = resolver.resolve(m06, m06["scenarios"][0])
    assert resolved.bindings["created_issue_id"] == 105
    assert any(step["arguments"].get("issue_id") == "$steps.s1.data.issue_id" for step in resolved.oracle_plan)


def test_catalog_has_all_public_contract_fields_and_zero_registry_input_diff():
    spec = load_openapi()
    renderer = ToolCatalogRenderer()
    catalog = renderer.render(spec)
    registry = ContractRegistry(spec)
    assert len(catalog["tools"]) == 12
    for tool in catalog["tools"]:
        assert tool["input_schema"] == registry.require(tool["tool_id"]).input_schema
        assert tool["operation_id"] == tool["tool_id"]
        assert tool["method"] and tool["path"] and tool["permission_requirement"]
        assert tool["success_response"] and tool["public_error_responses"]
        assert "$ref" not in json.dumps(tool)
        for name, schema in tool["input_schema"]["properties"].items():
            assert "type" in schema or "anyOf" in schema, (tool["tool_id"], name)
    assert renderer.stable_json(spec) == renderer.stable_json(deepcopy(spec))


def test_catalog_negative_missing_required_and_leakage_are_detected():
    spec = load_openapi()
    mutated = deepcopy(spec)
    mutated["components"]["schemas"]["CreateIssueRequest"]["required"] = []
    rendered = ToolCatalogRenderer().render(mutated)
    create = next(item for item in rendered["tools"] if item["tool_id"] == "create_issue")
    assert create["input_schema"] != ContractRegistry(spec).require("create_issue").input_schema
    leaked = deepcopy(spec)
    leaked["paths"]["/repositories/{repo_id}"]["get"]["description"] = "runtime_contract hidden rule"
    with pytest.raises((EvidenceLeakageError, ValueError)):
        ToolCatalogRenderer().render(leaked)


def test_policy_capabilities_narrow_prompt_schema_before_execution():
    schema = _load_schema("agent_action_schema_v3.json")
    for method in ("standard", "retry_only", "reflection", "validation_guided"):
        capabilities = PolicyCapabilities.for_method(method)
        narrowed = capabilities.narrow_schema(schema)
        actions = {branch["properties"]["action_type"]["const"] for branch in narrowed["oneOf"]}
        assert actions == {"TOOL_CALL", "FINAL_ANSWER", "ABSTAIN"}
        assert "REQUEST_PROBE" not in capabilities.prompt_fragment()
    driftguard = PolicyCapabilities.for_method("driftguard_llm")
    assert "REQUEST_PROBE" in driftguard.allowed_actions
    assert "REQUEST_PROBE" in driftguard.prompt_fragment()


def test_effective_v3_prompt_hash_includes_catalog_capabilities_schema_and_policy():
    config = Phase10Config.load(PROJECT_ROOT / "configs/experiments/phase10_real_pilot_v3.yaml")
    view = config.phase9_view(config.models[0], "end_to_end", ("standard", "driftguard_llm"), config.seeds[0])
    raw = deepcopy(view.raw)
    raw.setdefault("experiment", {})["action_prompt_version"] = "base_tool_agent_v3"
    view = replace(view, raw=raw)
    loader = PromptLoader(PROJECT_ROOT / "benchmark/prompts")
    prompts = {
        name: loader.load(name) for name in (
            "base_tool_agent_v1.txt", "base_tool_agent_v3.txt", "reflection_v1.txt",
            "validation_guided_v1.txt", "driftguard_attribution_v1.txt",
        )
    }
    runner = ExperimentScenarioRunner(view, None, prompts, {
        "agent_action": _load_schema("agent_action_schema_v3.json"),
        "llm_attribution": _load_schema("llm_attribution_schema_v1.json"),
    })
    canonical = load_openapi()
    family = next(item for item in _families() if item["matched_case_id"] == "M01")
    case = runner.attribution._case_by_id[family["source_drift_id"]]
    changed = ExecutionContext(ExecutionProfile(ExecutionMode.AGENT_ERROR, case, family)).displayed_contract
    assert runner._prompt_hash("standard", canonical) != runner._prompt_hash("standard", changed)
    assert runner._prompt_hash("standard", canonical) != runner._prompt_hash("driftguard_llm", canonical)


def test_offline_oracle_and_two_round_feedback_gates_pass_without_network():
    gate = OfflineRepairGate()
    oracle = gate.oracle_gate()
    assert oracle["scenarios"] == oracle["evaluator_success"] == 12
    assert oracle["natural_language_business_goal"] == 12
    assert oracle["forbidden_side_effect_failures"] == 0
    assert oracle["binding_consistency"] == oracle["tool_budget_pass"] == 12
    assert oracle["oracle_prompt_leaks"] == oracle["provider_calls"] == 0
    mock = gate.mock_feedback_gate()
    assert mock["methods"] == mock["successful_two_round_repairs"] == 5
    assert mock["only_driftguard_sees_probe"] and mock["network_calls"] == 0
    for item in mock["details"]:
        assert item["rounds"] == 2 and item["sandbox_calls"] == 1
        assert item["validation_feedback_transition"] and item["next_model_transition"]
        assert not item["exact_retry_after_local_validation"]
    assert next(item for item in mock["details"] if item["method"] == "reflection")["reflection_hook_visible"]
    assert next(item for item in mock["details"] if item["method"] == "validation_guided")["schema_fragment_visible"]
    driftguard = next(item for item in mock["details"] if item["method"] == "driftguard_llm")
    assert driftguard["driftguard_evidence_visible"] and driftguard["driftguard_evidence_events"] == 1


def test_full_offline_report_preserves_ledger_and_protected_hashes():
    report = OfflineRepairGate().run()
    assert report["real_provider_calls"] == 0 and report["new_cost_cny"] == 0
    assert report["cost_ledger"]["spent_cny"] + 1e-12 >= EXPECTED_SPENT_CNY
    assert report["cost_ledger"]["api_attempts"] >= EXPECTED_API_ATTEMPTS
    assert report["cost_ledger"]["unchanged"] and report["cost_ledger"]["baseline_or_later"]
    assert report["protection"]["canonical_unchanged"]
    assert report["pilot_v4_created_or_run"] == (
        PROJECT_ROOT / "results/experiments/phase10/pilot_v4_canary/manifest.json"
    ).exists()
    assert report["full_real_model_experiment_status"] == "NOT RUN"


def test_negative_unknown_task_binding_and_disallowed_probe_fail_before_execution():
    resolver = TaskInstanceResolver()
    family = deepcopy(next(item for item in _families() if item["matched_case_id"] == "M16"))
    family["scenarios"][0]["episode_schedule"][2]["task_instance_ref"] = "UNKNOWN-D1"
    with pytest.raises(ValueError, match="unresolvable first-failure task instance"):
        resolver.resolve(family, family["scenarios"][0])
    baseline_schema = PolicyCapabilities.for_method("standard").narrow_schema(_load_schema("agent_action_schema_v3.json"))
    with pytest.raises(ValidationError):
        Draft202012Validator(baseline_schema).validate({
            "action_type": "REQUEST_PROBE", "probe_type": "read", "target_tool_id": "get_issue",
            "hypothesis": "neutral", "evidence_refs": [], "concise_decision_summary": "neutral",
        })


def test_negative_partial_multistep_work_never_counts_as_task_success():
    resolver = TaskInstanceResolver()
    evaluator = TaskEvaluator()
    initial_service = SandboxService()
    initial = initial_service.store.snapshot()

    m06 = next(item for item in _families() if item["matched_case_id"] == "M06")
    task = resolver.resolve(m06, m06["scenarios"][0]).evaluator_task()
    service = SandboxService()
    service.call_tool("create_issue", {"repo_id": "R1", "title": "Add refresh-token tests", "priority": "high"}, "agent_admin")
    created_only = evaluator.evaluate(task, initial, service.store.snapshot(), {"tool_succeeded": True}, 1, 1)
    assert not created_only.task_success and not created_only.state_assertions_passed

    m11 = next(item for item in _families() if item["matched_case_id"] == "M11")
    task = resolver.resolve(m11, m11["scenarios"][0]).evaluator_task()
    service = SandboxService()
    service.call_tool("assign_issue", {"repo_id": "R1", "issue_id": 101, "assignee": "bob"}, "agent_admin")
    prerequisite_only = evaluator.evaluate(task, initial, service.store.snapshot(), {"tool_succeeded": True}, 1, 1)
    assert not prerequisite_only.task_success and not prerequisite_only.state_assertions_passed


def test_negative_arbitrary_tool_success_and_network_access_are_blocked():
    resolver = TaskInstanceResolver()
    family = next(item for item in _families() if item["matched_case_id"] == "M16")
    task = resolver.resolve(family, family["scenarios"][0]).evaluator_task()
    initial = SandboxService().store.snapshot()
    evaluation = TaskEvaluator().evaluate(task, initial, initial, {"tool_succeeded": True}, 1, 1)
    assert not evaluation.task_success
    import socket
    with blocked_network(), pytest.raises(NetworkAccessBlocked):
        socket.create_connection(("example.com", 443))
