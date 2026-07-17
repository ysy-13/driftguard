from __future__ import annotations

from copy import deepcopy
import inspect
import json

import pytest
from jsonschema import Draft202012Validator

from driftguard.contracts.loader import load_openapi
from driftguard.contracts.loader import PROJECT_ROOT
from driftguard.experiments.budgets import BudgetTracker, ExperimentBudget
from driftguard.healing.models import ValidationResult
from driftguard.live import LiveEvidenceBridge, LivePatchEligibilityGate, LiveScope
from driftguard.live.stages import LivePatchValidator, LiveStructuredStages
from driftguard.llm import MockProvider
from driftguard.runners.live_healing_mock_runner import LiveHealingMockRunner


@pytest.fixture(scope="module")
def live_report():
    return LiveHealingMockRunner().run()


@pytest.mark.parametrize("family,category", [("M01", "ICD"), ("M06", "RSD"), ("M11", "WPD"), ("M16", "SED")])
def test_four_pd_mock_chains_complete_live_probe_patch_repair_and_transfer(live_report, family, category):
    row = next(item for item in live_report["records"] if item["family"] == family)
    assert row["drift_category"] == category
    assert row["independent_failure_count"] == 2
    assert row["patch_eligibility"]["decision"] == "ELIGIBLE"
    assert row["probe_result"]["state_unchanged"]
    assert row["static_validation"]["passed"]
    assert row["regression_validation"]["passed"]
    assert row["safety_validation"]["passed"]
    assert row["minimality_validation"]["passed"]
    assert row["immediate_repair"]["passed"]
    assert row["future_transfer"]["patched_success"]
    assert row["patch_registry_state"]["accepted_count"] == 1
    assert row["symbolic_fallback_used"] is False


def test_live_mock_uses_controller_trace_not_phase7_artifact(live_report):
    for row in live_report["records"]:
        events = row["live_evidence_trace"]["events"]
        assert events and all(event["provenance"]["source"] == "controller_event" for event in events)
        assert any(event["event_type"] == "tool_response" for event in events)
        assert all("diagnosis_emitted" != event["event_type"] for event in events)


def test_live_trace_and_experiment_record_validate_against_versioned_v2_schemas(live_report):
    trace_schema = json.loads((PROJECT_ROOT / "benchmark/schemas/evidence_trace_schema_v2.json").read_text())
    record_schema = json.loads((PROJECT_ROOT / "benchmark/schemas/experiment_record_schema_v2.json").read_text())
    for row in live_report["records"]:
        Draft202012Validator(trace_schema).validate(row["live_evidence_trace"])
        Draft202012Validator(record_schema).validate(row)


def test_state_machine_records_all_live_pd_states_and_budgets(live_report):
    expected = {
        "TASK_EXECUTION", "FAILURE_OBSERVED", "PRELIMINARY_ATTRIBUTION",
        "RETRY_OR_EVIDENCE_COLLECTION", "REPRODUCTION_CHECK", "PROBE_SELECTION",
        "PROBE_EXECUTION", "FINAL_ATTRIBUTION", "PATCH_ELIGIBILITY",
        "PATCH_PROPOSAL", "PATCH_VALIDATION", "IMMEDIATE_REPAIR", "PATCH_ACCEPTED",
        "FUTURE_TRANSFER", "FINAL_EVALUATION",
    }
    for row in live_report["records"]:
        transitions = row["state_transitions"]
        assert {item["new_state"] for item in transitions} == expected
        assert all("remaining_budget" in item for item in transitions)


def test_budget_counts_attribution_probe_selection_patch_and_deterministic_calls(live_report):
    for row in live_report["records"]:
        assert row["llm_calls"] == 6  # two AgentAction + four structured-stage calls
        assert row["patch_proposal_calls"] == 1
        assert row["probe_calls"] == 1
        assert row["tool_calls"] >= 5
        assert row["provider_calls"] == 6  # two AgentAction + four independent structured calls


@pytest.mark.parametrize(
    "label,recovery_key",
    [("AGENT_ERROR", "agent_correction_succeeded"), ("TRANSIENT_FAILURE", "transient_retry_recovered")],
)
def test_ae_tf_negative_mock_attribution_never_opens_patch_channel(label, recovery_key):
    public_id = "public-negative-a" if label == "AGENT_ERROR" else "public-negative-t"
    bridge = LiveEvidenceBridge(LiveScope(public_id, "mock", "driftguard_llm", 0), load_openapi())
    event = bridge.emit(
        "retry_result", 3, source_event="test.controller.recovery", execution_context_id="independent-a",
        tool_id="create_issue", normalized_observation={"task_complete": True},
        probe_metadata={recovery_key: True, "regression_requirements_identified": True},
    )
    attribution = {
        "predicted_class": label, "target_tool_id": "create_issue", "drift_category": "ICD",
        "location": {"tool_id": "create_issue", "layer": "input", "spec_pointer": "/paths/x",
                     "runtime_path": "request", "adapter_operation_type": "add_required"},
        "evidence_refs": [event.event_id],
        "requested_probe": None, "confidence": 0.9, "concise_reason": "mock recovery evidence",
    }
    tracker = BudgetTracker(ExperimentBudget(max_llm_calls=2, max_input_tokens=100_000))
    parsed, _ = LiveStructuredStages(MockProvider([attribution]), tracker).attribution(bridge.agent_view(), "FINAL")
    decision = LivePatchEligibilityGate().decide(bridge.agent_view(), parsed)
    assert decision.decision == "FORBIDDEN" and tracker.patch_proposal_calls == 0


def _custom_outputs(runner: LiveHealingMockRunner, family_id: str):
    family = runner.families[family_id]
    scenario = next(item for item in family["scenarios"] if item["variant_code"] == "PD")
    case = runner.cases[family["source_drift_id"]]
    displayed = runner._context(case, family, 3).displayed_contract
    resolved = runner.resolver.resolve(family, scenario)
    action = next(step for step in resolved.oracle_plan if step["tool"] == case["target_tool"])
    return runner._stage_outputs(case, displayed, action["arguments"])


def test_invalid_patch_gets_exactly_one_revision_then_succeeds_or_rejects():
    runner = LiveHealingMockRunner()
    outputs = _custom_outputs(runner, "M01")
    invalid = deepcopy(outputs[3])
    invalid["evidence_refs"] = ["invalid-live-ref"]
    invalid["openapi_operations"][0]["evidence_refs"] = ["invalid-live-ref"]
    success = runner.run_pd("M01", stage_outputs=[*outputs[:3], invalid, outputs[3]])
    assert success["patch_revision"]["used"] and success["termination_reason"] == "LIVE_HEALING_COMPLETE"
    assert success["patch_proposal_calls"] == 2
    failed = runner.run_pd("M01", stage_outputs=[*outputs[:3], invalid, invalid])
    assert failed["patch_revision"]["used"]
    assert failed["termination_reason"] == "PATCH_REJECTED_STATIC_VALIDATION"
    assert failed["patch_registry_state"]["accepted_count"] == 0


class _FailRegression:
    def validate(self, patch, initial_state):
        return ValidationResult(False, "regression", ("MOCK_REGRESSION_FAILURE",), {})


class _FailSafety:
    def validate(self, patch, repair_run, max_calls):
        return ValidationResult(False, "safety", ("MOCK_SAFETY_FAILURE",), {})


class _FutureFailureValidator(LivePatchValidator):
    def validate_after_static(self, *args, **kwargs):
        result = super().validate_after_static(*args, **kwargs)
        if result.get("future_transfer"):
            result["future_transfer"]["patched_success"] = False
        return result


@pytest.mark.parametrize(
    "validator,failed_stage",
    [
        (LivePatchValidator(regression=_FailRegression()), "regression_validation"),
        (LivePatchValidator(safety=_FailSafety()), "safety_validation"),
    ],
)
def test_regression_and_safety_failures_reject_llm_patch(validator, failed_stage):
    row = LiveHealingMockRunner().run_pd("M01", validator=validator)
    assert row[failed_stage]["passed"] is False
    assert row["termination_reason"] == "PATCH_REJECTED_DETERMINISTIC_VALIDATION"
    assert row["patch_registry_state"]["accepted_count"] == 0


def test_future_transfer_failure_is_frozen_and_reported_without_rewriting_patch():
    row = LiveHealingMockRunner().run_pd("M01", validator=_FutureFailureValidator())
    assert row["future_transfer"]["patched_success"] is False
    assert row["future_transfer"]["first_result_counted"] is True
    assert row["future_transfer"]["candidate_rewritten_after_transfer"] is False


def test_live_validator_never_invokes_symbolic_patch_candidate_generator():
    source = inspect.getsource(LivePatchValidator)
    assert "import PatchCandidateGenerator" not in source
    assert "HealingEngine(" not in source and "self.generator" not in source


def test_episode_six_appears_only_after_patch_acceptance(live_report):
    for row in live_report["records"]:
        events = row["live_evidence_trace"]["events"]
        episode_six = [event for event in events if event["episode_number"] == 6]
        assert [event["event_type"] for event in episode_six] == ["future_transfer"]
        accepted_index = next(i for i, transition in enumerate(row["state_transitions"])
                              if transition["new_state"] == "PATCH_ACCEPTED")
        transfer_index = next(i for i, transition in enumerate(row["state_transitions"])
                              if transition["new_state"] == "FUTURE_TRANSFER")
        assert accepted_index < transfer_index


def test_no_patch_future_baseline_is_isolated_and_not_model_evidence(live_report):
    for row in live_report["records"]:
        assert row["future_transfer"]["no_patch_success"] is False
        assert row["future_transfer"]["same_initial_state"] is True
        encoded = str(row["live_evidence_trace"]).lower()
        assert "no_patch_success" in encoded  # emitted only after the model-facing stages are over
