from __future__ import annotations

from copy import deepcopy
import json

import pytest
from jsonschema import Draft202012Validator

from driftguard.contracts.loader import PROJECT_ROOT, load_openapi
from driftguard.diagnosis.probes import UnsafeProbeError
from driftguard.experiments.budgets import BudgetExhausted, BudgetTracker, ExperimentBudget
from driftguard.live import LiveEvidenceBridge, LivePatchEligibilityGate, LiveScope
from driftguard.live.focused_config import validate_focused_canary_config
from driftguard.live.stages import LLMPatchAdapter, LiveProbeExecutor, LiveStructuredStages
from driftguard.live.state_machine import LiveHealingState, LiveHealingStateMachine
from driftguard.llm import InvalidStructuredOutput, LLMCache, MockProvider, ModelConfig
from driftguard.runners.live_healing_mock_runner import LiveHealingMockRunner
from driftguard.sandbox import SandboxService


def _live_view(*, failures: int = 2, recovery: str | None = None, probe: bool = True):
    bridge = LiveEvidenceBridge(LiveScope("public-components", "mock", "driftguard_llm", 0), load_openapi())
    for index in range(failures):
        metadata = {"independent_failure": True, "regression_requirements_identified": True}
        if recovery:
            metadata[recovery] = True
        bridge.emit(
            "tool_response", 3 + index, source_event="test.controller.failure",
            execution_context_id=f"independent-{index}", tool_id="create_issue",
            normalized_observation={"channel": "response_error", "task_complete": False},
            visible_runtime_response={"status_code": 422}, probe_metadata=metadata,
        )
    if probe:
        bridge.emit(
            "probe_executed", 5, source_event="test.probe", execution_context_id="probe-fork",
            tool_id="create_issue", normalized_observation={"executed": True},
            probe_metadata={"discriminative": True, "passed": True, "unsafe_write": False,
                            "regression_requirements_identified": True},
        )
    return bridge, bridge.agent_view()


def _attribution(label: str = "PERSISTENT_DRIFT", refs=None):
    return {
        "predicted_class": label, "target_tool_id": "create_issue", "drift_category": "ICD",
        "location_type": "request_schema",
        "location_path": "/paths/~1repositories~1{repo_id}~1issues/post/requestBody",
        "evidence_refs": list(refs or []), "requested_probe": None, "confidence": 0.9,
        "concise_reason": "live evidence only",
    }


@pytest.mark.parametrize("label", ["AGENT_ERROR", "TRANSIENT_FAILURE"])
def test_ae_and_tf_attributions_are_hard_forbidden_from_patching(label: str):
    _, view = _live_view()
    decision = LivePatchEligibilityGate().decide(view, _attribution(label))
    assert decision.decision == "FORBIDDEN"


def test_one_failure_cannot_satisfy_persistent_patch_eligibility():
    _, view = _live_view(failures=1)
    decision = LivePatchEligibilityGate().decide(view, _attribution())
    assert decision.decision == "INSUFFICIENT_EVIDENCE"
    assert "TWO_INDEPENDENT_FAILURES_REQUIRED" in decision.reason_codes


@pytest.mark.parametrize("recovery", ["agent_correction_succeeded", "transient_retry_recovered"])
def test_llm_pd_misclassification_is_rejected_without_rewriting_prediction(recovery: str):
    _, view = _live_view(recovery=recovery)
    attribution = _attribution()
    decision = LivePatchEligibilityGate().decide(view, attribution)
    assert attribution["predicted_class"] == "PERSISTENT_DRIFT"
    assert decision.decision == "FORBIDDEN"


@pytest.mark.parametrize(
    "failures,probe,budget,reason",
    [(2, False, True, "DISCRIMINATIVE_PROBE_REQUIRED"), (2, True, False, "BUDGET_EXHAUSTED")],
)
def test_hard_gate_requires_probe_and_remaining_budget(failures, probe, budget, reason):
    _, view = _live_view(failures=failures, probe=probe)
    decision = LivePatchEligibilityGate().decide(view, _attribution(), budget_remaining=budget)
    assert decision.decision == "INSUFFICIENT_EVIDENCE" and reason in decision.reason_codes


def test_two_independent_failures_and_probe_make_pd_eligible():
    _, view = _live_view()
    decision = LivePatchEligibilityGate().decide(view, _attribution())
    assert decision.decision == "ELIGIBLE" and decision.independent_failure_count == 2


def test_two_contexts_with_different_symptoms_do_not_count_as_reproduction():
    bridge, _ = _live_view(failures=1, probe=False)
    bridge.emit(
        "tool_response", 4, source_event="test.controller.different_failure",
        execution_context_id="independent-other", tool_id="create_issue",
        normalized_observation={"channel": "task_incomplete", "task_complete": False},
        visible_runtime_response={"status_code": 200},
        probe_metadata={"independent_failure": True, "regression_requirements_identified": True},
    )
    bridge.emit(
        "probe_executed", 5, source_event="test.probe", execution_context_id="probe-fork",
        tool_id="create_issue", probe_metadata={"discriminative": True, "passed": True,
        "regression_requirements_identified": True},
    )
    decision = LivePatchEligibilityGate().decide(bridge.agent_view(), _attribution())
    assert decision.independent_failure_count == 1
    assert decision.decision == "INSUFFICIENT_EVIDENCE"


def test_invalid_attribution_evidence_reference_is_rejected():
    bridge, _ = _live_view()
    provider = MockProvider([_attribution(refs=["does-not-exist"])])
    stages = LiveStructuredStages(provider, BudgetTracker(ExperimentBudget(max_llm_calls=4)))
    with pytest.raises(InvalidStructuredOutput, match="nonexistent"):
        stages.attribution(bridge.agent_view(), "FINAL")


def test_live_structured_stage_cache_replays_without_a_provider_call(tmp_path):
    bridge, view = _live_view()
    output = _attribution(refs=[view.trace.events[0].event_id])
    cache = LLMCache(tmp_path / "isolated-live-cache")
    model = ModelConfig(provider="mock", model_id="mock-live")
    first_provider = MockProvider([output], model="mock-live")
    first = LiveStructuredStages(
        first_provider, BudgetTracker(ExperimentBudget(max_llm_calls=2)),
        cache=cache, model_config=model, config_hash="prompt-v3-schema-v3-catalog-policy",
    )
    first.attribution(bridge.agent_view(), "FINAL")
    second_provider = MockProvider([], model="mock-live")
    second = LiveStructuredStages(
        second_provider, BudgetTracker(ExperimentBudget(max_llm_calls=2)),
        cache=cache, model_config=model, config_hash="prompt-v3-schema-v3-catalog-policy",
    )
    second.attribution(bridge.agent_view(), "FINAL")
    assert first_provider.calls == 1 and second_provider.calls == 0


def test_dedicated_probe_selection_rejects_probe_outside_runtime_allowlist():
    bridge, view = _live_view()
    ref = view.trace.events[0].event_id
    output = {"decision": "SELECT_PROBE", "probe_type": "exact_retry", "target_tool_id": "create_issue",
              "evidence_refs": [ref], "concise_reason": "bounded discrimination"}
    stages = LiveStructuredStages(MockProvider([output]), BudgetTracker(ExperimentBudget(max_llm_calls=4)))
    with pytest.raises(UnsafeProbeError):
        stages.probe_selection(bridge.agent_view(), ("local_schema_check",))


def test_dedicated_probe_refusal_remains_a_model_refusal():
    bridge, view = _live_view()
    output = {"decision": "REFUSE_PROBE", "probe_type": None, "target_tool_id": None,
              "evidence_refs": [view.trace.events[0].event_id], "concise_reason": "insufficient basis"}
    stages = LiveStructuredStages(MockProvider([output]), BudgetTracker(ExperimentBudget(max_llm_calls=4)))
    parsed, _ = stages.probe_selection(bridge.agent_view(), ("local_schema_check",))
    assert parsed["decision"] == "REFUSE_PROBE"


def test_probe_selection_allows_exactly_one_budgeted_schema_format_repair():
    bridge, view = _live_view()
    ref = view.trace.events[0].event_id
    valid = {"decision": "SELECT_PROBE", "probe_type": "local_schema_check",
             "target_tool_id": "create_issue", "evidence_refs": [ref],
             "concise_reason": "bounded discrimination"}
    provider = MockProvider(["not-json", valid])
    tracker = BudgetTracker(ExperimentBudget(max_llm_calls=3, max_format_repairs=1))
    parsed, _ = LiveStructuredStages(provider, tracker).probe_selection(
        bridge.agent_view(), ("local_schema_check",),
    )
    assert parsed == valid and provider.calls == 2
    assert tracker.llm_calls == 2 and tracker.format_repairs == 1


def test_probe_selection_rejects_second_invalid_format_without_extra_call():
    bridge, _ = _live_view()
    provider = MockProvider(["not-json", "still-not-json", {"unused": True}])
    tracker = BudgetTracker(ExperimentBudget(max_llm_calls=3, max_format_repairs=1))
    with pytest.raises(InvalidStructuredOutput):
        LiveStructuredStages(provider, tracker).probe_selection(
            bridge.agent_view(), ("local_schema_check",),
        )
    assert provider.calls == 2 and tracker.format_repairs == 1


def test_probe_executes_in_fork_and_does_not_pollute_main_state():
    service = SandboxService()
    before = service.store.snapshot()
    selection = {"decision": "SELECT_PROBE", "probe_type": "repeated_read"}
    result = LiveProbeExecutor().execute(
        selection, service, lambda: None,
        lambda fork: fork.call_tool("get_repository", {"repo_id": "R1"}, "agent_admin"),
    )
    assert result["executed"] and result["state_unchanged"]
    assert service.store.snapshot() == before


def test_patch_proposal_is_independent_from_agent_action_schema():
    schema = json.loads((PROJECT_ROOT / "benchmark/schemas/agent_action_schema_v3.json").read_text())
    encoded = json.dumps(schema)
    assert "openapi_operations" not in encoded and "semantic_extensions" not in encoded


def test_llm_patch_adapter_rejects_nonexistent_live_evidence_refs():
    runner = LiveHealingMockRunner()
    bridge, view = _live_view()
    case = runner.cases["ICD-01"]
    output = runner._patch_output("ICD", "create_issue", case["runtime_mutation"]["target"], load_openapi(), ["missing"])
    with pytest.raises(ValueError, match="nonexistent"):
        LLMPatchAdapter().from_output(output, view, _attribution())


def test_patch_eligibility_cannot_be_bypassed_in_state_machine():
    machine = LiveHealingStateMachine()
    machine.move(LiveHealingState.PATCH_ELIGIBILITY, "TEST")
    with pytest.raises(ValueError, match="cannot be bypassed"):
        machine.move(LiveHealingState.PATCH_VALIDATION, "TEST")


def test_patch_proposal_and_revision_share_a_hard_budget_of_two():
    tracker = BudgetTracker(ExperimentBudget(max_llm_calls=4, max_patch_proposal_calls=2, max_total_interactions=6))
    tracker.consume_patch_proposal()
    tracker.consume_patch_proposal()
    assert tracker.llm_calls == tracker.patch_proposal_calls == 2
    with pytest.raises(BudgetExhausted):
        tracker.consume_patch_proposal()


def test_experiment_record_v2_requires_live_fields_and_forbids_symbolic_fallback():
    schema = json.loads((PROJECT_ROOT / "benchmark/schemas/experiment_record_schema_v2.json").read_text())
    record = {key: None for key in schema["required"]}
    record.update({
        "live_evidence_trace_ref": "live-x", "independent_failure_count": 0,
        "patch_eligibility": {}, "patch_registry_state": {}, "symbolic_fallback_used": False,
    })
    Draft202012Validator(schema).validate(record)
    record["symbolic_fallback_used"] = True
    assert list(Draft202012Validator(schema).iter_errors(record))


def test_focused_canary_config_is_exactly_eight_development_pd_records():
    report = validate_focused_canary_config()
    assert report["passed"] and report["planned_records"] == 8 and report["heldout48_overlap"] == 0


def test_focused_canary_is_created_but_explicitly_not_authorized():
    raw = (PROJECT_ROOT / "configs/experiments/phase10_driftguard_focused_canary.yaml").read_text()
    assert "run_authorized: false" in raw and "live_healing_enabled: true" in raw


def test_live_patch_prompt_has_no_symbolic_or_benchmark_answer_channel():
    prompt = (PROJECT_ROOT / "benchmark/prompts/live/driftguard_patch_v2.txt").read_text().lower()
    for forbidden in ("expected_patch_ref", "source drift id", "evaluatorview"):
        assert forbidden not in prompt
