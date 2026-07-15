import hashlib
import json

from driftguard.contracts.loader import PROJECT_ROOT
from driftguard.experiments.runner import PROMPT_NAMES, _load_schema
from driftguard.experiments.scenario_runner import ExperimentScenarioRunner
from driftguard.llm import LLMCache, ProviderResponse
from driftguard.llm.prompt_loader import PromptLoader
from driftguard.phase10.config import Phase10Config
from driftguard.runners.attribution_conformance_runner import MATCHED_PATH


CONFIG = Phase10Config.load(PROJECT_ROOT / "configs/experiments/phase10_real_pilot.yaml")


def _prompts():
    loader = PromptLoader(PROJECT_ROOT / "benchmark/prompts")
    names = PROMPT_NAMES + ("component_attribution_v2.txt",)
    return {name: loader.load(name) for name in names}


def _runner(model_index=0, provider_factory=None):
    view = CONFIG.phase9_view(CONFIG.models[model_index], "component", ("reflection",), CONFIG.seeds[0])
    schemas = {
        "agent_action": _load_schema("agent_action_schema_v1.json"),
        "llm_attribution": _load_schema("llm_attribution_schema_v1.json"),
    }
    return ExperimentScenarioRunner(view, None, _prompts(), schemas, provider_factory)


def test_component_prompt_v2_is_new_and_v1_files_remain_unchanged():
    prompt = (PROJECT_ROOT / "benchmark/prompts/component_attribution_v2.txt").read_text()
    assert "{{OUTPUT_SCHEMA}}" in prompt
    assert (PROJECT_ROOT / "benchmark/prompts/base_tool_agent_v1.txt").exists()
    assert (PROJECT_ROOT / "benchmark/prompts/reflection_v1.txt").exists()
    assert hashlib.sha256(prompt.encode()).hexdigest() != hashlib.sha256(
        (PROJECT_ROOT / "benchmark/prompts/base_tool_agent_v1.txt").read_bytes()
    ).hexdigest()


def test_v2_example_is_neutral_and_contains_no_benchmark_answer_hint():
    text = (PROJECT_ROOT / "benchmark/prompts/component_attribution_v2.txt").read_text()
    assert '"predicted_class": "INSUFFICIENT_EVIDENCE"' in text
    assert "Do not copy its values" in text
    forbidden = ("M01", "M06", "M11", "M16", "source_drift_id", "expected_patch_ref", "runtime_contract")
    assert not any(item in text for item in forbidden)


def test_rendered_v2_contains_complete_attribution_schema_and_matches_both_models():
    first, second = _runner(0), _runner(1)
    schema_text = json.dumps(first.schemas["llm_attribution"], indent=2, sort_keys=True)
    rendered = first.prompts["component_attribution_v2.txt"].replace("{{OUTPUT_SCHEMA}}", schema_text)
    assert schema_text in rendered
    assert first._component_prompt_hash("reflection") == second._component_prompt_hash("reflection")


def test_v1_and_v2_prompt_and_cache_keys_cannot_collide():
    runner = _runner()
    assert runner._prompt_hash("reflection") != runner._component_prompt_hash("reflection")


def test_format_repair_contains_validation_error_and_full_schema():
    requests = []
    valid = {
        "predicted_class": "INSUFFICIENT_EVIDENCE", "target_tool_id": None,
        "drift_category": "UNKNOWN", "location_type": "unknown", "location_path": None,
        "evidence_refs": [], "requested_probe": None, "confidence": 0.0,
        "concise_reason": "Visible evidence is not sufficient for a classification.",
    }
    class RecordingProvider:
        def __init__(self): self.calls = 0
        def complete(self, request):
            requests.append(request)
            self.calls += 1
            raw = '{"attribution":"free-form"}' if self.calls == 1 else json.dumps(valid)
            return ProviderResponse(f"r{self.calls}", CONFIG.models[0].model_id, None, raw, 10, 5, 15, 1, 1, "stop")
    provider = RecordingProvider()
    runner = _runner(provider_factory=lambda outputs: provider)
    families = json.loads(MATCHED_PATH.read_text())["families"]
    family = next(item for item in families if item["matched_case_id"] == "M01")
    record = runner.run(family, family["scenarios"][0], "reflection", "component", 0)
    assert record["llm_calls"] == 2 and len(requests) == 2
    repair = requests[1].messages[-1]["content"]
    assert "Schema validation error:" in repair
    assert "<OUTPUT_SCHEMA>" in repair
    assert json.dumps(runner.schemas["llm_attribution"], indent=2, sort_keys=True) in repair


def test_heldout48_permanently_excludes_all_development_families():
    heldout = Phase10Config.load(PROJECT_ROOT / "configs/experiments/phase10_main_heldout48.yaml")
    assert not ({"M01", "M06", "M11", "M16"} & set(heldout.families))
