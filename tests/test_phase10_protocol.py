import json

import pytest

from driftguard.contracts.loader import PROJECT_ROOT
from driftguard.phase10.config import Phase10Config
from driftguard.phase10.runner import Phase10Execution
from driftguard.phase10.costs import CostBudgetManager
from driftguard.phase10.pilot import Phase10PilotHarness
from driftguard.phase10.pricing import PricingCatalog
from driftguard.llm import MockProvider
from driftguard.llm import ProviderResponse
from driftguard.phase10.preflight import PreflightRunner


CONFIGS = PROJECT_ROOT / "configs/experiments"


def test_real_api_requires_explicit_allow_flag(tmp_path):
    with pytest.raises(PermissionError, match="allow-real-api"):
        Phase10Execution(CONFIGS / "phase10_real_pilot.yaml", tmp_path, False)


@pytest.mark.parametrize("name", ["phase10_main_all60.yaml", "phase10_main_heldout48.yaml", "phase10_ablation.yaml"])
def test_pilot_entry_cannot_trigger_main_or_ablation(name, tmp_path):
    with pytest.raises(PermissionError, match="confirm-full-run"):
        Phase10Execution(CONFIGS / name, tmp_path, True)
    with pytest.raises(PermissionError, match="cannot execute"):
        Phase10Execution(CONFIGS / name, tmp_path, True, True)


def test_component_end_to_end_and_symbolic_are_separate():
    config = Phase10Config.load(CONFIGS / "phase10_real_pilot.yaml")
    assert set(config.modes) == {"component", "end_to_end"}
    assert config.methods["deterministic"] == ("oracle_symbolic_upper_bound",)
    assert "oracle_symbolic_upper_bound" not in config.methods["component"] + config.methods["end_to_end"]


def test_pilot_manifest_and_record_schemas_require_development_labels():
    manifest = json.loads((PROJECT_ROOT / "benchmark/schemas/phase10_manifest_schema_v1.json").read_text())
    record = json.loads((PROJECT_ROOT / "benchmark/schemas/phase10_record_schema_v1.json").read_text())
    assert manifest["properties"]["full_real_model_experiment_status"]["const"] == "NOT RUN"
    assert record["properties"]["experiment_stage"]["const"] == "PILOT"
    assert record["properties"]["publication_status"]["const"] == "DEVELOPMENT_ONLY"


def test_prompt_freeze_inputs_are_hashable_and_stable():
    prompt_dir = PROJECT_ROOT / "benchmark/prompts"
    first = {path.name: __import__("hashlib").sha256(path.read_bytes()).hexdigest() for path in prompt_dir.glob("*.txt")}
    second = {path.name: __import__("hashlib").sha256(path.read_bytes()).hexdigest() for path in prompt_dir.glob("*.txt")}
    assert first == second
    assert set(first) == {
        "base_tool_agent_v1.txt",
        "base_tool_agent_v2.txt",
        "base_tool_agent_v3.txt",
        "component_attribution_v2.txt",
        "component_attribution_v3.txt",
        "driftguard_attribution_v1.txt",
        "driftguard_patch_v1.txt",
        "reflection_v1.txt",
        "specdriftbench_component_v1.txt",
        "validation_guided_v1.txt",
    }


def test_repetition_seed_pairing_and_cache_isolation_are_explicit():
    config = Phase10Config.load(CONFIGS / "phase10_main_all60.yaml")
    assert list(enumerate(config.seeds)) == [(0, 20260715), (1, 20260716), (2, 20260717)]
    assert config.repetitions == len(config.seeds)


def test_offline_component_stage_and_resume_do_not_charge_or_repeat_provider(tmp_path, monkeypatch):
    config = Phase10Config.load(CONFIGS / "phase10_real_pilot.yaml")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-only")
    calls = {"count": 0}
    def builder(model, outputs):
        calls["count"] += 1
        return MockProvider(outputs, model.model_id)
    costs = CostBudgetManager(PricingCatalog.load_default(), 35, 50, 60_000, 12_000)
    harness = Phase10PilotHarness(config, tmp_path / "pilot", costs, builder)
    harness._write_manifest()
    first, report = harness._run_stage(config.models[0], "component", False)
    before = calls["count"]
    second, resumed = harness._run_stage(config.models[0], "component", True)
    assert len(first) == len(second) == 36 and report["created"] == 36
    assert resumed["resumed"] == 36 and calls["count"] == before
    assert costs.api_attempts == 0 and costs.spent_cny == 0
    manifest_text = (tmp_path / "pilot/manifest.json").read_text()
    assert "test-only" not in manifest_text and "pricing_snapshot_id" in manifest_text
    assert "https://api.deepseek.com" in manifest_text


def test_offline_end_to_end_stage_uses_real_task_evaluator_boundary(tmp_path, monkeypatch):
    config = Phase10Config.load(CONFIGS / "phase10_real_pilot.yaml")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-only")
    costs = CostBudgetManager(PricingCatalog.load_default(), 35, 50, 60_000, 12_000)
    harness = Phase10PilotHarness(
        config, tmp_path / "pilot", costs,
        lambda model, outputs: MockProvider(outputs, model.model_id),
    )
    harness.families = harness.families[:1]
    records, report = harness._run_stage(config.models[0], "end_to_end", False)
    assert len(records) == 15 and report["records"] == 15
    assert all(record["experiment_stage"] == "PILOT" for record in records)
    assert all("final_task_success" in record and record["publication_status"] == "DEVELOPMENT_ONLY" for record in records)
    assert costs.api_attempts == 0


def test_offline_preflight_exercises_text_json_and_tool_checks(tmp_path, monkeypatch):
    config = Phase10Config.load(CONFIGS / "phase10_real_pilot.yaml")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-only-a")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-only-b")
    class FakeProvider:
        def __init__(self, model): self.model = model
        def complete(self, request):
            tool_calls = ({"id": "t", "type": "function", "function": {"name": "phase10_echo", "arguments": "{\"text\":\"ok\"}"}},) if request.tools else ()
            raw = '{"status":"ok"}' if request.response_schema else ("OK" if not request.tools else '{"tool_calls":[]}')
            return ProviderResponse("r", self.model, None, raw, 2, 1, 3, 1, 1, "stop", tool_calls=tool_calls)
    costs = CostBudgetManager(PricingCatalog.load_default(), 35, 50, 60_000, 12_000)
    result = PreflightRunner(config, tmp_path, costs, lambda model, key: FakeProvider(model.model_id)).run()
    assert result["passed_models"] == ["deepseek", "dashscope"]
    assert all(len(item["checks"]) == 3 and item["passed"] for item in result["reports"])
