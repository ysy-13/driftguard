import pytest

from driftguard.agents.policies import DriftGuardPolicy, OracleSymbolicPolicy
from driftguard.evidence.models import EvaluatorView
from phase8_helpers import artifacts
from driftguard.experiments import ExperimentBudget
from driftguard.runners.injection_conformance_runner import CASE_ARGUMENTS
from phase9_helpers import context_for, controller, smoke_task


def tool_action(tool, arguments):
    return {"action_type": "TOOL_CALL", "tool_id": tool, "arguments": arguments, "concise_decision_summary": "call"}


def final_action(answer="done"):
    return {"action_type": "FINAL_ANSWER", "answer": answer, "concise_decision_summary": "final"}


def test_standard_agent_executes_one_valid_tool_call():
    agent, service = controller([tool_action("get_repository", {"repo_id": "R1"})])
    result = agent.run(smoke_task(), service.registry.document, "public-0123456789ab", 1, 0)
    assert result.task_success and result.tool_calls == 1 and result.llm_calls == 1


def test_unknown_tool_is_rejected_without_execution():
    agent, service = controller([tool_action("shell_exec", {})])
    result = agent.run(smoke_task(), service.registry.document, "public-0123456789ab", 1, 0)
    assert result.error_category == "UNKNOWN_TOOL" and result.tool_calls == 0


def test_invalid_arguments_are_rejected_locally():
    agent, service = controller([tool_action("get_repository", {})])
    result = agent.run(smoke_task(), service.registry.document, "public-0123456789ab", 1, 0)
    assert result.error_category == "INVALID_ARGUMENTS" and result.tool_calls == 0


def test_llm_final_answer_cannot_bypass_task_evaluator():
    agent, service = controller([final_action("I succeeded")])
    result = agent.run(smoke_task(), service.registry.document, "public-0123456789ab", 1, 0)
    assert not result.task_success and result.termination_reason == "TASK_FAILED"


def test_retry_only_performs_exactly_one_tool_retry_without_extra_llm_call():
    context = context_for("WPD-01", "TF")
    action = tool_action("close_issue", CASE_ARGUMENTS["WPD-01"])
    agent, service = controller([action], "retry_only", context)
    result = agent.run(smoke_task(), context.displayed_contract, "public-0123456789ab", 3, 0)
    assert result.task_success and result.tool_calls == 2 and result.llm_calls == 1
    assert result.actions[1]["executed_arguments"] == result.actions[0]["executed_arguments"]


def test_reflection_extra_call_counts_against_llm_budget_and_memory_is_ephemeral():
    context = context_for("WPD-01", "PD")
    agent, service = controller([tool_action("close_issue", CASE_ARGUMENTS["WPD-01"]), final_action()], "reflection", context)
    result = agent.run(smoke_task(), context.displayed_contract, "public-0123456789ab", 3, 0)
    assert result.llm_calls == 2 and not result.task_success
    assert agent.memory.visible() == []


def test_validation_guided_does_not_persist_patch_registry():
    context = context_for("WPD-01", "PD")
    agent, service = controller([tool_action("close_issue", CASE_ARGUMENTS["WPD-01"]), final_action()], "validation_guided", context)
    agent.run(smoke_task(), context.displayed_contract, "public-0123456789ab", 3, 0)
    assert not hasattr(agent.policy, "registry") and not agent.policy.persistent_patch


def test_invalid_json_uses_bounded_format_repair_and_counts_it():
    agent, service = controller(["not json", tool_action("get_repository", {"repo_id": "R1"})])
    result = agent.run(smoke_task(), service.registry.document, "public-0123456789ab", 1, 0)
    assert result.task_success and result.llm_calls == 2 and agent.tracker.format_repairs == 1


def test_budget_exhaustion_stops_further_calls():
    budget = ExperimentBudget(max_llm_calls=0)
    agent, service = controller([final_action()], budget=budget)
    result = agent.run(smoke_task(), service.registry.document, "public-0123456789ab", 1, 0)
    assert result.termination_reason == "BUDGET_EXHAUSTED" and result.llm_calls == 0 and result.tool_calls == 0


def test_probe_counts_as_tool_interaction_and_is_policy_gated():
    probe = {"action_type": "REQUEST_PROBE", "probe_type": "safe_read", "target_tool_id": "get_repository", "hypothesis": "state exists", "evidence_refs": [], "concise_decision_summary": "probe"}
    agent, service = controller([probe], "standard")
    blocked = agent.run(smoke_task(), service.registry.document, "public-0123456789ab", 1, 0)
    assert blocked.error_category == "SAFETY_BLOCKED" and blocked.probe_calls == 0
    agent, service = controller([probe, final_action()], "driftguard_llm")
    allowed = agent.run(smoke_task(), service.registry.document, "public-0123456789ab", 1, 0)
    assert allowed.probe_calls == 1 and allowed.tool_calls == 1


def test_driftguard_llm_rejected_patch_is_not_symbolically_replaced():
    policy = DriftGuardPolicy("llm")
    bad = {"bad": "proposal"}
    assert policy.guarded_llm_patch(bad, lambda value: False) is None
    assert policy.raw_llm_prediction is None


def test_symbolic_upper_bound_is_explicitly_marked():
    policy = OracleSymbolicPolicy()
    assert policy.is_upper_bound and policy.method == "oracle_symbolic_upper_bound"


def test_agent_rejects_evaluator_view():
    value = artifacts("M01")
    agent, service = controller([final_action()])
    with pytest.raises(TypeError, match="EvaluatorView"):
        agent.run(smoke_task(), EvaluatorView(value["agent_view"], {}), "public-0123456789ab", 1, 0)
