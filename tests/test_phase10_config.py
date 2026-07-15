import json

import pytest
from jsonschema import Draft202012Validator

from driftguard.contracts.loader import PROJECT_ROOT
from driftguard.phase10.config import Phase10Config


CONFIGS = PROJECT_ROOT / "configs/experiments"


def test_pilot_has_exact_models_endpoints_keys_and_temperature():
    config = Phase10Config.load(CONFIGS / "phase10_real_pilot.yaml")
    assert [(m.provider, m.base_url, m.model_id, m.api_key_env) for m in config.models] == [
        ("deepseek", "https://api.deepseek.com", "deepseek-v4-flash", "DEEPSEEK_API_KEY"),
        ("dashscope", "https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen3.7-plus", "DASHSCOPE_API_KEY"),
    ]
    assert all(m.temperature == 0.1 and m.thinking_mode == "disabled" for m in config.models)


def test_pilot_selection_methods_and_size_are_fixed():
    config = Phase10Config.load(CONFIGS / "phase10_real_pilot.yaml")
    assert set(config.families) == {"M01", "M06", "M11", "M16"}
    assert config.repetitions == 1 and config.seeds == (20260715,)
    assert len(config.methods["end_to_end"]) == 5
    assert len(config.methods["component"]) == 3
    assert config.planned_real_records() == 192


def test_v3_pilot_is_strictly_end_to_end_only_and_keeps_the_same_development_set():
    config = Phase10Config.load(CONFIGS / "phase10_real_pilot_v3.yaml")
    assert config.modes == ("end_to_end",)
    assert config.raw["experiment"]["pilot_attempt"] == "v3"
    assert config.raw["experiment"]["prior_component_attempt"] == "pilot_v2"
    assert set(config.families) == {"M01", "M06", "M11", "M16"}
    assert config.planned_real_records() == 120


def test_v4_canary_is_isolated_fixed_and_exactly_ten_records():
    config = Phase10Config.load(CONFIGS / "phase10_real_pilot_v4_canary.yaml")
    experiment = config.raw["experiment"]
    assert experiment["pilot_attempt"] == "v4_canary"
    assert experiment["run_scope"] == "end_to_end_canary_only"
    assert experiment["development_only"] and not experiment["heldout48_allowed"]
    assert config.modes == ("end_to_end",) and config.methods["component"] == ()
    assert [(item.provider, item.model_id) for item in config.models] == [
        ("deepseek", "deepseek-v4-flash"), ("dashscope", "qwen3.7-plus"),
    ]
    selection = experiment["selection"]
    assert len(selection) == 5
    assert [(item["method"], item["variant"]) for item in selection] == [
        ("standard", "AE"), ("retry_only", "TF"), ("reflection", "AE"),
        ("validation_guided", "AE"), ("driftguard_llm", "PD"),
    ]
    assert {item["family"] for item in selection} <= {"M01", "M06", "M11", "M16"}
    assert config.budget.pilot_soft_limit_cny == pytest.approx(4.497508830)
    assert config.budget.pilot_hard_limit_cny == pytest.approx(5.497508830)


def test_main_all60_and_heldout48_sizes_and_split():
    all60 = Phase10Config.load(CONFIGS / "phase10_main_all60.yaml")
    heldout = Phase10Config.load(CONFIGS / "phase10_main_heldout48.yaml")
    assert len(all60.families) == 20 and len(heldout.families) == 16
    assert set(all60.families) - set(heldout.families) == {"M01", "M06", "M11", "M16"}
    assert len(all60.models) * 5 * 60 * all60.repetitions == 1800
    assert len(heldout.models) * 5 * 48 * heldout.repetitions == 1440
    assert all60.modes == heldout.modes == ("end_to_end",)
    assert all60.planned_real_records() == 1800
    assert heldout.planned_real_records() == 1440
    assert all60.seeds == heldout.seeds == (20260715, 20260716, 20260717)


def test_ablation_is_deepseek_only_and_sandbox_guarded():
    config = Phase10Config.load(CONFIGS / "phase10_ablation.yaml")
    assert [model.provider for model in config.models] == ["deepseek"]
    assert len(config.methods["end_to_end"]) == 6
    assert config.execution["sandbox_fork_required"] and config.execution["unsafe_action_block"]


def test_configs_never_contain_credential_values():
    for path in CONFIGS.glob("phase10_*.yaml"):
        text = path.read_text()
        assert "sk-" not in text and "api_key:" not in text.lower()


def test_pricing_snapshot_schema_and_conservative_flags():
    document = json.loads((PROJECT_ROOT / "benchmark/pricing/phase10_pricing_v1.json").read_text())
    schema = json.loads((PROJECT_ROOT / "benchmark/schemas/phase10_pricing_schema_v1.json").read_text())
    Draft202012Validator(schema).validate(document)
    assert all(not item["discount_used"] for item in document["models"])
    assert document["budget_estimate_only"]


def test_fixed_model_substitution_is_rejected(tmp_path):
    text = (CONFIGS / "phase10_real_pilot.yaml").read_text().replace("deepseek-v4-flash", "deepseek-chat")
    path = tmp_path / "bad.yaml"
    path.write_text(text)
    with pytest.raises(ValueError, match="fixed Phase 10 model mismatch"):
        Phase10Config.load(path)
