import hashlib
import json

import pytest
from jsonschema import Draft202012Validator, ValidationError

from driftguard.agents import ToolAgentController
from driftguard.agents.policies import POLICIES
from driftguard.contracts.loader import PROJECT_ROOT
from driftguard.experiments.budgets import BudgetTracker
from driftguard.experiments.runner import PROMPT_NAMES, _load_schema
from driftguard.experiments.scenario_runner import ExperimentScenarioRunner
from driftguard.llm import MockProvider
from driftguard.llm import LLMCache, ProviderRequest
from driftguard.llm.prompt_loader import PromptLoader
from driftguard.phase10.config import Phase10Config
from driftguard.phase10.costs import CostBudgetManager
from driftguard.phase10.pilot import Phase10PilotHarness
from driftguard.phase10.pricing import PricingCatalog
from driftguard.phase10.runner import Phase10Execution
from driftguard.sandbox import SandboxService


CONFIG_PATH = PROJECT_ROOT / "configs/experiments/phase10_real_pilot_v3.yaml"
CONFIG = Phase10Config.load(CONFIG_PATH)
ACTION_SCHEMA = _load_schema("agent_action_schema_v2.json")
METHODS = ("standard", "retry_only", "reflection", "validation_guided", "driftguard_llm")


def _prompts():
    loader = PromptLoader(PROJECT_ROOT / "benchmark/prompts")
    names = PROMPT_NAMES + ("component_attribution_v2.txt", "base_tool_agent_v2.txt")
    return {name: loader.load(name) for name in names}


def _runner(model_index=0):
    view = CONFIG.phase9_view(CONFIG.models[model_index], "end_to_end", METHODS, CONFIG.seeds[0])
    return ExperimentScenarioRunner(
        view, None, _prompts(),
        {"agent_action": ACTION_SCHEMA, "llm_attribution": _load_schema("llm_attribution_schema_v1.json")},
    )


def test_action_prompt_v2_is_new_neutral_and_v1_is_preserved():
    prompt_root = PROJECT_ROOT / "benchmark/prompts"
    old = (prompt_root / "base_tool_agent_v1.txt").read_bytes()
    new = (prompt_root / "base_tool_agent_v2.txt").read_bytes()
    assert old and new and old != new
    assert hashlib.sha256(old).hexdigest() == "8773e647effcb28e7e070750baca8d03466e295dadc753b8887c3b2df564214f"
    text = new.decode()
    assert '"tool_id": "<DISPLAYED_TOOL_ID>"' in text
    assert '"<ARGUMENT_NAME>": "<ARGUMENT_VALUE>"' in text
    assert not any(tool_id in text for tool_id in SandboxService().registry.operation_ids())
    assert not any(value in text for value in ("M01", "M06", "M11", "M16", "ground_truth", "runtime_contract"))


def test_action_schema_v2_covers_four_strict_action_shapes():
    Draft202012Validator.check_schema(ACTION_SCHEMA)
    valid = [
        {"action_type": "TOOL_CALL", "tool_id": "<DISPLAYED_TOOL_ID>", "arguments": {}, "concise_decision_summary": "brief"},
        {"action_type": "FINAL_ANSWER", "answer": "<ANSWER>", "concise_decision_summary": "brief"},
        {"action_type": "REQUEST_PROBE", "probe_type": "<PROBE_TYPE>", "target_tool_id": "<DISPLAYED_TOOL_ID>", "hypothesis": "<HYPOTHESIS>", "evidence_refs": [], "concise_decision_summary": "brief"},
        {"action_type": "ABSTAIN", "concise_decision_summary": "brief"},
    ]
    validator = Draft202012Validator(ACTION_SCHEMA)
    for value in valid:
        validator.validate(value)
    with pytest.raises(ValidationError):
        validator.validate({"tool": "<DISPLAYED_TOOL_ID>", "parameters": {}})
    with pytest.raises(ValidationError):
        validator.validate({**valid[0], "answer": "forbidden for TOOL_CALL"})


def test_every_end_to_end_method_and_model_receives_the_same_complete_action_schema():
    schema_text = json.dumps(ACTION_SCHEMA, indent=2, sort_keys=True)
    first, second = _runner(0), _runner(1)
    assert first._action_base_prompt() == second._action_base_prompt()
    for method in METHODS:
        for runner in (first, second):
            rendered = runner._action_system_prompt(method)
            assert schema_text in rendered
            assert "TOOL_CALL" in rendered and "FINAL_ANSWER" in rendered
            assert "REQUEST_PROBE" in rendered and "ABSTAIN" in rendered
            assert "Required:" in rendered and "Optional:" in rendered and "Forbidden:" in rendered
        assert first._prompt_hash(method) == second._prompt_hash(method)


def test_v2_action_prompt_hash_and_cache_key_differ_from_the_failed_attempt():
    old_config = Phase10Config.load(PROJECT_ROOT / "configs/experiments/phase10_real_pilot.yaml")
    old_view = old_config.phase9_view(old_config.models[0], "end_to_end", METHODS, old_config.seeds[0])
    old_prompts = _prompts()
    old_prompts.pop("base_tool_agent_v2.txt")
    old_runner = ExperimentScenarioRunner(
        old_view, None, old_prompts,
        {"agent_action": _load_schema("agent_action_schema_v1.json"), "llm_attribution": _load_schema("llm_attribution_schema_v1.json")},
    )
    new_runner = _runner()
    old_hash = old_runner._prompt_hash("standard")
    new_hash = new_runner._prompt_hash("standard")
    assert old_hash != new_hash
    messages = ({"role": "system", "content": "neutral"},)
    old_request = ProviderRequest(messages, old_runner.schemas["agent_action"], old_hash, "public-neutral", 3, "standard", 0, 20260715, "end_to_end", old_config.config_hash)
    new_request = ProviderRequest(messages, new_runner.schemas["agent_action"], new_hash, "public-neutral", 3, "standard", 0, 20260715, "end_to_end", CONFIG.config_hash)
    old_schema_hash = hashlib.sha256(json.dumps(old_runner.schemas["agent_action"], sort_keys=True).encode()).hexdigest()
    new_schema_hash = hashlib.sha256(json.dumps(new_runner.schemas["agent_action"], sort_keys=True).encode()).hexdigest()
    assert LLMCache.key(old_view.model, old_request, old_schema_hash) != LLMCache.key(CONFIG.phase9_view(CONFIG.models[0], "end_to_end", METHODS, CONFIG.seeds[0]).model, new_request, new_schema_hash)


def test_legacy_tool_parameters_are_rejected_and_repair_contains_error_fields_and_schema():
    requests = []

    class RecordingProvider(MockProvider):
        def complete(self, request):
            requests.append(request)
            return super().complete(request)

    provider = RecordingProvider([
        {"tool": "<DISPLAYED_TOOL_ID>", "parameters": {}},
        {"action_type": "TOOL_CALL", "tool_id": "<DISPLAYED_TOOL_ID>", "arguments": {}, "concise_decision_summary": "brief"},
    ], CONFIG.models[0].model_id)
    view = CONFIG.phase9_view(CONFIG.models[0], "end_to_end", ("standard",), CONFIG.seeds[0])
    tracker = BudgetTracker(view.budget)
    runner = _runner()
    controller = ToolAgentController(
        provider, view.model, POLICIES["standard"](), SandboxService(), tracker,
        ACTION_SCHEMA, runner._action_base_prompt(), "",
    )
    action = controller._next_action(
        ({"role": "system", "content": runner._action_base_prompt()},),
        "public-neutral", 3, 0,
    )
    assert action.action_type.value == "TOOL_CALL"
    assert tracker.llm_calls == 2 and tracker.format_repairs == 1
    repair = requests[1].messages[-1]["content"]
    assert "Schema validation error:" in repair
    assert "Missing or illegal fields:" in repair
    assert "missing=action_type, concise_decision_summary" in repair
    assert "illegal=parameters, tool" in repair
    assert json.dumps(ACTION_SCHEMA, indent=2, sort_keys=True) in repair
    assert "Return only one JSON object" in repair


def test_v3_is_end_to_end_only_and_cannot_rerun_component(tmp_path):
    assert CONFIG.raw["experiment"]["pilot_attempt"] == "v3"
    assert CONFIG.raw["experiment"]["run_scope"] == "end_to_end_only"
    assert CONFIG.modes == ("end_to_end",) and CONFIG.planned_real_records() == 120
    execution = Phase10Execution(CONFIG_PATH, tmp_path, True)
    with pytest.raises(PermissionError, match="component records must not be rerun"):
        execution.run_component_canary("deepseek")


def test_offline_v3_end_to_end_canary_has_five_valid_methods_and_a_real_tool_call(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-only")
    costs = CostBudgetManager(PricingCatalog.load_default(), 35, 50, 60_000, 12_000)
    harness = Phase10PilotHarness(
        CONFIG, tmp_path / "pilot_v3", costs,
        lambda model, outputs: MockProvider(outputs, model.model_id),
    )
    records, report = harness._run_stage(CONFIG.models[0], "end_to_end", False, canary=True)
    assert len(records) == report["records"] == 5
    assert report["structured_valid_records"] == 5
    assert report["infrastructure_errors"] == 0
    assert report["tool_call_interface_verified"]
    assert report["parsed_tool_call_records"] >= 1
    assert costs.api_attempts == 0 and costs.spent_cny == 0
