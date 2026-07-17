from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import pytest
from jsonschema import Draft202012Validator

from driftguard.contracts.loader import PROJECT_ROOT, load_openapi
from driftguard.evidence.collector import stable_hash
from driftguard.experiments.budgets import BudgetExhausted, BudgetTracker, ExperimentBudget
from driftguard.healing.models import PatchOperation, ToolSpecPatch
from driftguard.healing.static_validator import StaticPatchValidator
from driftguard.live import LiveEvidenceBridge, LivePatchEligibilityGate, LiveScope, evaluate_drift_exposure
from driftguard.live.adapter_contract import (
    ADAPTER_OPERATIONS, neutral_example, operation_types, rendered_contract, validate_semantics,
)
from driftguard.live.location import locations_match, normalize_location
from driftguard.live.stages import (
    ControllerProbeError, LLMPatchAdapter, LiveProbeExecutor, LiveStructuredStages,
)
from driftguard.llm import LLMProvider, MockProvider, ProviderResponse
from driftguard.runners.live_healing_mock_runner import LiveHealingMockRunner
from driftguard.live.focused_config import FOCUSED_CONFIG
from driftguard.runners.focused_live_healing_runner import FocusedLiveHealingRunner
from driftguard.sandbox import SandboxService


def _selection(probe_type: str, target: str, arguments: dict) -> dict:
    return {
        "decision": "SELECT_PROBE", "probe_type": probe_type,
        "target_tool": target, "arguments": arguments,
        "expected_observation_type": "visible neutral observation",
        "rationale": "bounded safe discrimination", "evidence_refs": [],
    }


def test_selected_get_tool_is_the_get_tool_actually_executed():
    result = LiveProbeExecutor().execute(
        _selection("response_shape_inspection", "get_repository", {"repo_id": "R1"}),
        SandboxService(), lambda: None, load_openapi(),
    )
    assert result["requested_target"] == result["executed_target"] == "get_repository"
    assert result["executed_probe"]["executed_tools"] == ["get_repository"]


def test_response_shape_inspection_cannot_dispatch_a_write():
    service = SandboxService()
    before = stable_hash(service.store.snapshot())
    with pytest.raises(ControllerProbeError, match="read-only"):
        LiveProbeExecutor().execute(
            _selection("response_shape_inspection", "create_issue", {
                "repo_id": "R1", "title": "Neutral title", "priority": "low",
            }), service, lambda: None, load_openapi(),
        )
    assert stable_hash(service.store.snapshot()) == before and service.call_log() == []


def test_repeated_read_uses_the_selected_safe_read_and_main_state_is_clean():
    service = SandboxService()
    before = stable_hash(service.store.snapshot())
    result = LiveProbeExecutor().execute(
        _selection("repeated_read", "get_repository", {"repo_id": "R1"}),
        service, lambda: None, load_openapi(),
    )
    assert result["executed_probe"]["executed_tools"] == ["get_repository", "get_repository"]
    assert result["state_unchanged"] and not result["main_state_pollution"]
    assert stable_hash(service.store.snapshot()) == before


def test_read_after_write_records_real_post_state_evidence_in_agent_view():
    row = LiveHealingMockRunner().run_pd("M16")
    probe = row["probe_result"]
    assert probe["executed_probe"]["executed_tools"] == ["close_issue", "get_issue"]
    assert "read_after_write_response" in probe["result"]
    event = next(item for item in row["live_evidence_trace"]["events"] if item["event_type"] == "probe_executed")
    assert event["before_state_hash"] and event["after_state_hash"]
    assert event["normalized_observation"]["requested_target"] == event["normalized_observation"]["executed_target"]
    assert event["normalized_observation"]["tool_responses"]


def test_unexposed_target_is_not_attribution_evaluable_and_cannot_patch():
    bridge = LiveEvidenceBridge(LiveScope("public-neutral", "mock", "driftguard_llm", 0), load_openapi())
    bridge.emit(
        "tool_response", 3, source_event="test.controller", execution_context_id="natural-1",
        tool_id="get_repository", visible_runtime_response={"status_code": 200},
        normalized_observation={"channel": "task_incomplete", "task_complete": False},
        probe_metadata={"independent_failure": True, "regression_requirements_identified": True},
    )
    exposure = evaluate_drift_exposure(bridge.trace, "create_issue")
    assert exposure == {
        "drift_exposed": False, "observed_target_tool": None,
        "observed_failure_signature": None, "attribution_evaluable": False,
        "pre_drift_agent_failure": True,
    }
    attribution = {
        "predicted_class": "PERSISTENT_DRIFT", "target_tool_id": "get_repository",
        "drift_category": "RSD", "location": {
            "tool_id": "get_repository", "layer": "response",
            "spec_pointer": "/paths/~1repositories~1{repo_id}/get/responses/200",
            "runtime_path": "payload", "adapter_operation_type": "rename_output_field",
        },
    }
    decision = LivePatchEligibilityGate().decide(
        bridge.agent_view(), attribution, drift_exposed=False,
    )
    assert decision.decision == "INSUFFICIENT_EVIDENCE"
    assert "TARGET_DRIFT_NOT_EXPOSED" in decision.reason_codes


@pytest.mark.parametrize("category,layer,operation", [
    ("ICD", "input", "add_required"),
    ("RSD", "response", "rename_output_field"),
    ("WPD", "workflow", "add_prerequisite"),
    ("SED", "state_effect", "add_postcondition_verification"),
])
def test_attribution_v3_taxonomy_accepts_all_four_neutral_classes(category, layer, operation):
    schema = json.loads((PROJECT_ROOT / "benchmark/schemas/llm_attribution_schema_v3.json").read_text())
    Draft202012Validator(schema).validate({
        "predicted_class": "PERSISTENT_DRIFT", "target_tool_id": "set_thermostat",
        "drift_category": category,
        "location": {"tool_id": "set_thermostat", "layer": layer,
                     "spec_pointer": "/paths/~1devices/put", "runtime_path": "device.value",
                     "adapter_operation_type": operation},
        "evidence_refs": ["visible-a"], "requested_probe": None,
        "confidence": 0.8, "concise_reason": "Neutral visible evidence supports this category.",
    })
    prompt = (PROJECT_ROOT / "benchmark/prompts/component_attribution_v3.txt").read_text()
    assert category + ":" in prompt


def _neutral_spec() -> dict:
    operation = {
        "operationId": "set_thermostat", "responses": {"200": {"description": "ok"}},
    }
    return {
        "openapi": "3.1.0", "info": {"title": "Neutral", "version": "1"},
        "paths": {"/devices/temperature": {"post": operation}},
    }


@pytest.mark.parametrize("category,operation,extra", [
    ("ICD", "add_required", {"request_transform": {"kind": "supply_required", "field": "setting", "value": 1}}),
    ("RSD", "rename_output_field", {"response_mapping": {"reading": "value"}}),
    ("WPD", "add_prerequisite", {"workflow": {"kind": "ordered_transition", "intermediate": "ready"}}),
    ("SED", "add_postcondition_verification", {"observation_policy": {"kind": "read_after_write", "confirmation_tool": "read_thermostat", "condition": "matches"}}),
])
def test_four_neutral_adapter_patches_match_allowlist_and_pass_static(category, operation, extra):
    spec = _neutral_spec()
    location = {"tool_id": "set_thermostat", "layer": {
        "ICD": "input", "RSD": "response", "WPD": "workflow", "SED": "state_effect",
    }[category], "spec_pointer": "/paths/~1devices~1temperature/post/x-contract-v3",
        "runtime_path": "device.value", "adapter_operation_type": operation}
    semantics = {"operation": operation, "target": "/adapter/set_thermostat/contract",
                 "before": [], "after": ["visible-change"], **extra}
    validate_semantics(category, location, {"x-driftguard-patch-semantics": semantics})
    patch = ToolSpecPatch(
        "patch-neutral", "3.0", "2026-01-01T00:00:00Z", "set_thermostat",
        stable_hash(spec), category, location["layer"], location["spec_pointer"],
        ("visible-a",), ("VISIBLE_ONLY",),
        (PatchOperation("add", location["spec_pointer"], {"enabled": True}, ("visible-a",), "VISIBLE"),),
        {"x-driftguard-patch-semantics": semantics}, "Use the visible contract.", 0.8,
        normalized_location=location,
    )
    result, _ = StaticPatchValidator().validate(patch, spec)
    assert result.passed, result.to_dict()


def test_patch_prompt_contract_is_rendered_from_same_allowlist_and_example_is_neutral():
    rendered = rendered_contract()
    assert set(operation_types()) == {
        operation for rules in ADAPTER_OPERATIONS.values() for operation in rules
    }
    prompt = (PROJECT_ROOT / "benchmark/prompts/live/driftguard_patch_v3.txt").read_text()
    assert "{{ADAPTER_CONTRACT}}" in prompt and "{{NEUTRAL_EXAMPLE}}" in prompt
    encoded = json.dumps(neutral_example()).lower()
    for forbidden in ("issue", "repository", "pipeline", "member", "priority", "issue_id"):
        assert forbidden not in encoded
    assert rendered["source_of_truth"].startswith("Phase 8")


def test_normalized_locations_match_only_in_one_coordinate_system():
    location = {"tool_id": "read_thermostat", "layer": "response",
                "spec_pointer": "/paths/~1devices/get/responses/200", "runtime_path": "payload.value",
                "adapter_operation_type": "rename_output_field"}
    assert locations_match(location, deepcopy(location))
    changed = deepcopy(location); changed["runtime_path"] = "payload.other"
    assert not locations_match(location, changed)
    assert normalize_location(location) == location


class _CaptureProvider(LLMProvider):
    def __init__(self, output):
        self.output, self.request = output, None

    def complete(self, request):
        self.request = request
        raw = json.dumps(self.output)
        return ProviderResponse("capture", "mock", self.output, raw, 10, 10, 20, 0.0, 1, "stop")


def test_revision_receives_validator_details_and_no_hidden_answer_material():
    runner = LiveHealingMockRunner()
    family = runner.families["M01"]
    case = runner.cases[family["source_drift_id"]]
    displayed = runner._context(case, family, 3).displayed_contract
    patch = runner._patch_output("ICD", "create_issue", "visible", displayed, ["visible-a"])
    attribution = runner._attribution("create_issue", "ICD", patch["location"], ["visible-a"], None)
    bridge = LiveEvidenceBridge(LiveScope("public-revision", "mock", "driftguard_llm", 0), displayed)
    bridge.emit("tool_response", 3, source_event="test", tool_id="create_issue",
                normalized_observation={"task_complete": False}, probe_metadata={},)
    ref = bridge.trace.events[0].event_id
    patch["evidence_refs"] = [ref]; patch["openapi_operations"][0]["evidence_refs"] = [ref]
    attribution["evidence_refs"] = [ref]
    provider = _CaptureProvider(patch)
    stages = LiveStructuredStages(provider, BudgetTracker(ExperimentBudget(
        max_llm_calls=2, max_patch_proposal_calls=2, max_input_tokens=100_000,
    )))
    stages.patch_proposal(
        bridge.agent_view(), attribution, {"decision": "ELIGIBLE"}, previous=patch,
        validator_error={"stage": "adapter", "passed": False,
                         "reason_codes": ["LOCATION_MISMATCH"],
                         "details": {"actual": "/visible/a", "allowed": "/visible/b"}},
    )
    payload = json.loads(provider.request.messages[-1]["content"])
    assert payload["deterministic_validator_error"]["details"]["actual"] == "/visible/a"
    assert payload["allowed_adapter_operations"] and payload["allowed_target_scope"]
    assert payload["adapter_schema_fragment"] and payload["normalized_attributed_location"]
    encoded = json.dumps(payload).lower()
    for forbidden in ("expected_patch", "source_drift_id", "evaluator_metadata"):
        assert forbidden not in encoded


def test_patch_budget_is_reserved_before_a_provider_could_be_called():
    tracker = BudgetTracker(ExperimentBudget(max_llm_calls=3, max_patch_proposal_calls=1))
    reservation = tracker.reserve_llm_call(patch_proposal=True)
    tracker.settle_llm_call(reservation, 1, 1)
    with pytest.raises(BudgetExhausted, match="patch proposal"):
        tracker.reserve_llm_call(patch_proposal=True)
    assert tracker.llm_calls == tracker.patch_proposal_calls == 1


def test_budget_exception_carries_completed_partial_stages():
    runner = LiveHealingMockRunner()
    family = runner.families["M01"]
    scenario = next(item for item in family["scenarios"] if item["variant_code"] == "PD")
    case = runner.cases[family["source_drift_id"]]
    displayed = runner._context(case, family, 3).displayed_contract
    resolved = runner.resolver.resolve(family, scenario)
    action = next(step for step in resolved.oracle_plan if step["tool"] == case["target_tool"])
    outputs = runner._stage_outputs(case, displayed, action["arguments"])
    malformed = {"not_a_patch": True}
    invalid = deepcopy(outputs[3]); invalid["location"]["runtime_path"] = "different"
    with pytest.raises(BudgetExhausted) as caught:
        runner.run_pd("M01", stage_outputs=[*outputs[:3], malformed, invalid, invalid])
    partial = caught.value.partial_result
    assert partial["preliminary_attribution"] and partial["final_attribution"]
    assert partial["probe_result"] and partial["raw_llm_patch"]
    assert partial["llm_stage_calls"] and partial["termination_reason"] == "BudgetExhausted"


def test_focused_failure_record_persists_partial_stage_results(tmp_path):
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps({
        "api_attempts": 863, "provider_reported_input_tokens": 3_580_832,
        "provider_reported_output_tokens": 84_182, "spent_cny": 5.202644120,
        "reserved_cny": 0.0, "soft_warning": False,
    }))
    runner = FocusedLiveHealingRunner(
        FOCUSED_CONFIG, mode="mock", output=tmp_path / "results",
        cache_root=tmp_path / "cache", ledger_path=ledger,
    )
    partial = {
        "live_evidence_trace": {"trace_id": "live-partial", "public_scenario_id": "public-M01", "events": []},
        "state_transitions": [{"new_state": "PRELIMINARY_ATTRIBUTION"}],
        "llm_stage_calls": [{"stage": "live_preliminary_attribution", "response_id": "raw-ref",
                             "raw_response_sha256": "a" * 64, "input_tokens": 10,
                             "output_tokens": 5, "latency_ms": 1.0}],
        "llm_calls": 1, "tool_calls": 2, "probe_calls": 0, "patch_proposal_calls": 0,
        "preliminary_attribution": {"predicted_class": "INSUFFICIENT_EVIDENCE"},
        "patch_eligibility": {"decision": "NOT_EVALUATED"},
        "controller_runs": [], "termination_reason": "BudgetExhausted",
        "drift_exposed": False, "observed_target_tool": None,
        "observed_failure_signature": None, "attribution_evaluable": False,
        "pre_drift_agent_failure": True,
    }

    def fail(*args, **kwargs):
        error = BudgetExhausted("interaction budget exhausted")
        error.partial_result = partial
        raise error

    runner.mock_runner.run_pd = fail
    record = runner._run_record(runner.config.plan[0])
    assert record["infrastructure_or_method_failure"] == "METHOD_FAILURE"
    assert record["preliminary_attribution"] == partial["preliminary_attribution"]
    assert record["llm_stage_calls"][0]["raw_response_sha256"] == "a" * 64
    assert record["pre_drift_agent_failure"] and not record["drift_exposed"]


def test_v1_v2_prompt_and_schema_artifacts_remain_byte_stable_and_no_secret_channels():
    expected = {
        "benchmark/prompts/component_attribution_v2.txt": "41255c4096399f8ded854f64d84a4f5de02174d1fb8a9825828e5c91c3ab2d5e",
        "benchmark/prompts/live/driftguard_probe_selection_v1.txt": "0b93fcbf4fdf56f52d3fcfe1550dd111e11e5e164d8f294219dcc0346d77f4eb",
        "benchmark/prompts/live/driftguard_patch_v2.txt": "4c325cc684341e91f405c13c05a3a8d0a751aa4fe83a8ee3e956bb75262953df",
    }
    for relative, digest in expected.items():
        assert hashlib.sha256((PROJECT_ROOT / relative).read_bytes()).hexdigest() == digest
    new_material = "".join((PROJECT_ROOT / relative).read_text().lower() for relative in (
        "benchmark/prompts/component_attribution_v3.txt",
        "benchmark/prompts/live/driftguard_probe_selection_v3.txt",
        "benchmark/prompts/live/driftguard_patch_v3.txt",
    ))
    for forbidden in ("api_key", "expected_patch", "source_drift_id", "runtime_contract"):
        assert forbidden not in new_material
