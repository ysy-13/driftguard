from __future__ import annotations

from collections import Counter
import hashlib
import json

import httpx
import pytest
from jsonschema import Draft202012Validator
import yaml

from driftguard.contracts.loader import PROJECT_ROOT
from driftguard.evidence.leakage_guard import EvidenceLeakageError
from driftguard.llm import (
    LLMCache, ModelCapabilities, ModelConfig, OpenAICompatibleProvider,
    ProviderCapabilityAdapter, ProviderRequest,
)
from driftguard.llm.leakage import assert_provider_request_visible
from driftguard.llm.models import ProviderResponse
from driftguard.phase10.pricing import PricingCatalog
from driftguard.runners.focused_live_healing_runner import AtomicFocusedLedger
from driftguard.specdriftbench.protocol import (
    BenchmarkErrorClass, CanonicalToolRegistry, EvidenceViewBuilder,
    OutputTruncated, TrueEvidenceLeakage, assert_benchmark_visible, balanced_evidence_view,
    classify_error, normalize_prediction,
)
from driftguard.specdriftbench.runner import (
    CONFIG_PATH, CONFIRM_36, OUTPUT_SCHEMA_PATH, PRICING_PATH,
    SpecDriftBenchConfig, SpecDriftBenchRunner,
    _LedgerConfigAdapter, capture_ledger_baseline, offline_rescore_v1,
)


KIMI_CN_PROFILE_CONFIG = PROJECT_ROOT / "configs/experiments/specdriftbench_component_canary_v1_moonshot_cn_k26_v1.yaml"
KIMI_CN_NONTHINKING_CONFIG = PROJECT_ROOT / "configs/experiments/specdriftbench_component_canary_v1_moonshot_cn_k26_nonthinking_v1.yaml"


def test_balanced_mapping_is_four_per_view_and_variant_for_every_provider():
    config = SpecDriftBenchConfig.load()
    assert len(config.plan) == 36
    for provider in ("deepseek", "dashscope", "moonshot"):
        rows = [item for item in config.plan if item.provider == provider]
        assert Counter(item.variant for item in rows) == {"AE": 4, "TF": 4, "PD": 4}
        assert Counter(item.evidence_view for item in rows) == {
            "FIRST_FAILURE": 4, "RETRY_HISTORY": 4, "FULL_EVIDENCE": 4,
        }
    assert balanced_evidence_view("M01", "AE") == "FIRST_FAILURE"
    assert balanced_evidence_view("M16", "PD") == "FULL_EVIDENCE"


def test_evidence_views_are_monotone_and_exclude_future_diagnosis():
    builder = EvidenceViewBuilder()
    first = builder.build("M01", "PD", "FIRST_FAILURE")
    retry = builder.build("M01", "PD", "RETRY_HISTORY")
    full = builder.build("M01", "PD", "FULL_EVIDENCE")
    first_types = {item["event_type"] for item in first["evidence_trace"]["events"]}
    retry_types = {item["event_type"] for item in retry["evidence_trace"]["events"]}
    full_types = {item["event_type"] for item in full["evidence_trace"]["events"]}
    assert not first_types & {"history_retrieval", "retry_result", "probe_result", "probe_started"}
    assert "retry_result" in retry_types and "probe_result" not in retry_types
    assert "probe_result" in full_types
    assert "diagnosis_emitted" not in first_types | retry_types | full_types
    assert first["displayed_specification"] == retry["displayed_specification"] == full["displayed_specification"]


def test_tool_registry_is_bijective_and_normalizes_public_alias():
    registry = CanonicalToolRegistry.from_displayed_spec()
    assert registry.canonicalize("T03") == "create_issue"
    assert registry.alias("create_issue") == "T03"
    value = normalize_prediction({
        "target_tool": "T03",
        "normalized_location": {"tool_id": "T03", "layer": "response", "spec_pointer": "/x", "runtime_path": "data.id"},
    }, registry)
    assert value["target_tool"] == value["normalized_location"]["tool_id"] == "create_issue"


@pytest.mark.parametrize("label", [
    "AGENT_ERROR", "TRANSIENT_FAILURE", "PERSISTENT_DRIFT",
    "ICD", "RSD", "WPD", "SED", "NONE",
])
def test_legal_prediction_labels_are_not_leakage(label):
    assert_benchmark_visible({"prediction": label, "reason": f"visible evidence supports {label}"})


@pytest.mark.parametrize("hidden", [
    {"ground_truth_label": "agent_error"},
    {"evaluator_metadata": {}},
    {"content": "expected_patch is hidden"},
    {"content": "lookup M01"},
])
def test_hidden_evaluator_material_is_true_leakage(hidden):
    with pytest.raises(TrueEvidenceLeakage):
        assert_benchmark_visible(hidden)
    assert classify_error(TrueEvidenceLeakage("hidden")) == BenchmarkErrorClass.TRUE_EVIDENCE_LEAKAGE


def test_provider_request_guard_allows_legal_labels_but_not_hidden_truth():
    request = ProviderRequest(
        ({"role": "system", "content": "Choose AGENT_ERROR, TRANSIENT_FAILURE, or PERSISTENT_DRIFT; categories ICD RSD WPD SED NONE."},),
        {}, "0" * 64, "public-neutral", 1, "component", 0,
    )
    assert_provider_request_visible(request)
    hidden = ProviderRequest(
        ({"role": "user", "content": "ground_truth_label=agent_error"},),
        {}, "0" * 64, "public-neutral", 1, "component", 0,
    )
    with pytest.raises((EvidenceLeakageError, ValueError)):
        assert_provider_request_visible(hidden)


def test_three_provider_configuration_and_heldout_gate():
    config = SpecDriftBenchConfig.load()
    assert [(item.provider, item.model_id) for item in config.models] == [
        ("deepseek", "deepseek-v4-flash"),
        ("dashscope", "qwen3.7-plus"),
        ("moonshot", "kimi-k2.6"),
    ]
    assert config.models[2].base_url == "https://api.moonshot.ai/v1"
    assert config.raw["experiment"]["heldout48_allowed"] is False


def test_kimi_k26_profile_uses_fixed_sampling_json_object_and_no_tools():
    captured = {}
    def handler(request):
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "kimi-test", "model": "kimi-k2.6",
            "choices": [{"message": {"content": '{"ok":true}'}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
        })
    model = ModelConfig(
        "moonshot", "kimi-k2.6", "https://api.moonshot.cn/v1",
        temperature=1.0, top_p=0.95, thinking_mode="enabled",
        structured_output_mode="strict_json",
        capabilities=ModelCapabilities(strict_json=True, tool_calling=False),
    )
    request = ProviderRequest(
        ({"role": "user", "content": "return JSON"},), {"type": "object"},
        "0" * 64, "public-kimi", 1, "preflight", 0,
    )
    provider = OpenAICompatibleProvider(model, "private", httpx.Client(transport=httpx.MockTransport(handler)))
    provider.complete(request)
    assert ProviderCapabilityAdapter.for_model(model).profile_id == ProviderCapabilityAdapter.KIMI_K26_PROFILE
    assert captured["temperature"] == 1.0
    assert captured["top_p"] == 0.95
    assert captured["n"] == 1
    assert captured["presence_penalty"] == 0.0
    assert captured["frequency_penalty"] == 0.0
    assert captured["response_format"] == {"type": "json_object"}
    assert "tools" not in captured and "tool_choice" not in captured


def test_kimi_k26_nonthinking_profile_uses_fixed_short_json_parameters():
    captured = {}
    def handler(request):
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "kimi-nonthinking-test", "model": "kimi-k2.6",
            "choices": [{"message": {"content": '{"ok":true}'}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
        })
    model = SpecDriftBenchConfig.load(KIMI_CN_NONTHINKING_CONFIG).model("moonshot")
    request = ProviderRequest(
        ({"role": "user", "content": "return JSON"},), {"type": "object"},
        "0" * 64, "public-kimi", 1, "preflight", 0,
    )
    provider = OpenAICompatibleProvider(model, "private", httpx.Client(transport=httpx.MockTransport(handler)))
    provider.complete(request)
    assert ProviderCapabilityAdapter.for_model(model).profile_id == ProviderCapabilityAdapter.KIMI_K26_NONTHINKING_PROFILE
    assert captured["thinking"] == {"type": "disabled"}
    assert captured["temperature"] == 0.6
    assert captured["top_p"] == 0.95
    assert captured["max_tokens"] == 1024
    assert captured["n"] == 1
    assert captured["presence_penalty"] == captured["frequency_penalty"] == 0.0
    assert captured["response_format"] == {"type": "json_object"}
    assert captured["stream"] is False


@pytest.mark.parametrize("field,value", [
    ("temperature", 0.1), ("top_p", 1.0), ("thinking_mode", "disabled"),
])
def test_kimi_k26_profile_rejects_illegal_config_before_network(field, value):
    called = {"network": 0}
    def handler(request):
        called["network"] += 1
        raise AssertionError("network boundary must not be reached")
    values = {
        "provider": "moonshot", "model_id": "kimi-k2.6",
        "base_url": "https://api.moonshot.cn/v1", "temperature": 1.0,
        "top_p": 0.95, "thinking_mode": "enabled",
        "structured_output_mode": "strict_json",
        "capabilities": ModelCapabilities(strict_json=True, tool_calling=False),
    }
    values[field] = value
    model = ModelConfig(**values)
    request = ProviderRequest(
        ({"role": "user", "content": "return JSON"},), {"type": "object"},
        "0" * 64, "public-kimi", 1, "preflight", 0,
    )
    provider = OpenAICompatibleProvider(
        model, "private", httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(ValueError, match="kimi-k2.6 capability profile"):
        provider.complete(request)
    assert called["network"] == 0


@pytest.mark.parametrize("field,value", [
    ("temperature", 1.0), ("top_p", 1.0), ("max_output_tokens", 128),
])
def test_kimi_k26_nonthinking_profile_rejects_illegal_config_before_network(field, value):
    called = {"network": 0}
    def handler(request):
        called["network"] += 1
        raise AssertionError("network boundary must not be reached")
    model = SpecDriftBenchConfig.load(KIMI_CN_NONTHINKING_CONFIG).model("moonshot")
    values = dict(model.__dict__)
    values[field] = value
    provider = OpenAICompatibleProvider(
        ModelConfig(**values), "private", httpx.Client(transport=httpx.MockTransport(handler)),
    )
    request = ProviderRequest(
        ({"role": "user", "content": "return JSON"},), {"type": "object"},
        "0" * 64, "public-kimi", 1, "preflight", 0,
    )
    with pytest.raises(ValueError, match="kimi-k2.6"):
        provider.complete(request)
    assert called["network"] == 0


def test_deepseek_and_qwen_sampling_parameters_remain_unchanged():
    bodies = {}
    def handler(request):
        body = json.loads(request.content)
        bodies[body["model"]] = body
        return httpx.Response(200, json={
            "id": "provider-test", "model": body["model"],
            "choices": [{"message": {"content": '{}'}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })
    request = ProviderRequest(
        ({"role": "user", "content": "return JSON"},), {"type": "object"},
        "0" * 64, "public-provider", 1, "preflight", 0,
    )
    for model in (
        ModelConfig("deepseek", "deepseek-v4-flash", "https://api.deepseek.com", temperature=0.1, top_p=1.0, thinking_mode="disabled", structured_output_mode="strict_json"),
        ModelConfig("dashscope", "qwen3.7-plus", "https://dashscope.aliyuncs.com/compatible-mode/v1", temperature=0.1, top_p=1.0, thinking_mode="disabled", structured_output_mode="strict_json"),
    ):
        OpenAICompatibleProvider(model, "private", httpx.Client(transport=httpx.MockTransport(handler))).complete(request)
    deepseek, qwen = bodies["deepseek-v4-flash"], bodies["qwen3.7-plus"]
    assert (deepseek["temperature"], deepseek["top_p"], deepseek["thinking"]) == (0.1, 1.0, {"type": "disabled"})
    assert (qwen["temperature"], qwen["top_p"], qwen["enable_thinking"]) == (0.1, 1.0, False)
    assert not {"n", "presence_penalty", "frequency_penalty"} & deepseek.keys()
    assert not {"n", "presence_penalty", "frequency_penalty"} & qwen.keys()


def test_kimi_profile_enters_manifest_and_cache_key(tmp_path, monkeypatch):
    raw = yaml.safe_load(KIMI_CN_NONTHINKING_CONFIG.read_text())
    raw["experiment"]["run_authorized"] = True
    config_path = tmp_path / "authorized.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False))
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps({
        "api_attempts": 10, "provider_reported_input_tokens": 100,
        "provider_reported_output_tokens": 20, "spent_cny": 1.0,
        "reserved_cny": 0.0, "soft_warning": False,
    }))
    for name in ("DEEPSEEK_API_KEY", "DASHSCOPE_API_KEY", "MOONSHOT_API_KEY"):
        monkeypatch.setenv(name, "test-only")
    output = tmp_path / "attempt"
    SpecDriftBenchRunner(
        config_path, attempt_id="profile-manifest", mode="real",
        allow_real_api=True, confirmation=CONFIRM_36, output_root=output,
        ledger_path=ledger,
    )
    manifest = json.loads((output / "results/manifest.json").read_text())
    profile = manifest["provider_capability_profiles"]["moonshot"]
    assert profile["profile_id"] == ProviderCapabilityAdapter.KIMI_K26_NONTHINKING_PROFILE
    assert profile["sampling_parameters"] == {
        "temperature": 0.6, "top_p": 0.95, "n": 1,
        "presence_penalty": 0.0, "frequency_penalty": 0.0,
    }
    assert profile["max_tokens"] == 1024

    config = SpecDriftBenchConfig.load(KIMI_CN_NONTHINKING_CONFIG)
    model = config.model("moonshot")
    request = ProviderRequest(
        ({"role": "user", "content": "visible"},), {}, "0" * 64,
        "public-neutral", 1, "component", 0, config_hash=config.config_hash,
    )
    first = LLMCache.key(model, request, "1" * 64)
    class AlternateProfile:
        def public_profile(self):
            return {"schema_version": "provider-capability-profile-v1", "profile_id": "alternate"}
    monkeypatch.setattr(
        "driftguard.llm.cache.ProviderCapabilityAdapter.for_model",
        lambda model: AlternateProfile(),
    )
    assert LLMCache.key(model, request, "1" * 64) != first


def test_old_thinking_cache_key_cannot_hit_nonthinking_profile():
    thinking = SpecDriftBenchConfig.load(KIMI_CN_PROFILE_CONFIG).model("moonshot")
    nonthinking = SpecDriftBenchConfig.load(KIMI_CN_NONTHINKING_CONFIG).model("moonshot")
    request = ProviderRequest(
        ({"role": "user", "content": "visible"},), {}, "0" * 64,
        "public-neutral", 1, "component", 0,
    )
    assert LLMCache.key(thinking, request, "1" * 64) != LLMCache.key(nonthinking, request, "1" * 64)


def test_length_output_is_truncated_and_reasoning_content_is_not_persisted():
    secret_reasoning = "private chain of thought must not persist"
    def handler(request):
        return httpx.Response(200, json={
            "id": "kimi-length", "model": "kimi-k2.6",
            "choices": [{
                "message": {"content": "", "reasoning_content": secret_reasoning},
                "finish_reason": "length",
            }],
            "usage": {
                "prompt_tokens": 40, "completion_tokens": 128, "total_tokens": 168,
                "completion_tokens_details": {"reasoning_tokens": 128},
            },
        })
    model = SpecDriftBenchConfig.load(KIMI_CN_NONTHINKING_CONFIG).model("moonshot")
    response = OpenAICompatibleProvider(
        model, "private", httpx.Client(transport=httpx.MockTransport(handler)),
    ).complete(ProviderRequest(
        ({"role": "user", "content": "return JSON"},), {"type": "object"},
        "0" * 64, "public-kimi", 1, "preflight", 0,
    ))
    error = OutputTruncated(response.finish_reason, len(response.raw_text))
    assert classify_error(error) == BenchmarkErrorClass.OUTPUT_TRUNCATED
    assert response.reasoning_tokens == 128
    assert secret_reasoning not in json.dumps(response.public_dict())


def test_old_kimi_preflight_failures_remain_immutable():
    expected = {
        "specdriftbench-v1-20260717-kimi-preflight-7f3c9a21": "db9c188aa12ebb709b4893819ca465cb9e91dbf38eb6f3a1d0875126f297c88d",
        "specdriftbench-v1-20260717-kimi-preflight-2-91c84f62": "60cc6973ca9b44fb2d56958e0badfb67fb71a615a63e67493e994ac31058cb0b",
        "specdriftbench-v1-cn-20260717-kimi-preflight-3-b6a407de": "008fcd5c1c4ff547997cd72f2cad97c19b1dab19e044a013b19354a95489e244",
        "specdriftbench-k26p1-20260717-preflight-4-e31a92bf": "f3c36fcd4d637d3eb5385acb4c5071b26290c1d41d78258638e0446da958f750",
    }
    root = PROJECT_ROOT / "results/experiments/phase10/specdriftbench_component_canary/attempts"
    for attempt, digest in expected.items():
        path = root / attempt / "preflight/result.json"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_benchmark_schema_is_strict():
    schema = json.loads(OUTPUT_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)
    valid = {
        "predicted_class": "PERSISTENT_DRIFT", "drift_category": "RSD",
        "target_tool": "T03", "normalized_location": {
            "tool_id": "T03", "layer": "response", "spec_pointer": "/components/schemas/X",
            "runtime_path": "data.id",
        }, "confidence": 0.8, "abstain": False, "concise_reason": "visible mismatch",
    }
    validator.validate(valid)
    with pytest.raises(Exception):
        validator.validate({**valid, "ground_truth_label": "persistent_drift"})


def test_cache_isolated_by_provider_and_replay_read_is_free(tmp_path):
    config = SpecDriftBenchConfig.load()
    request = ProviderRequest(
        ({"role": "user", "content": "visible"},), {}, "0" * 64,
        "public-neutral", 1, "component", 0, config_hash=config.config_hash,
    )
    keys = {LLMCache.key(model, request, "1" * 64) for model in config.models}
    assert len(keys) == 3
    cache = LLMCache(tmp_path / "deepseek")
    key = next(iter(keys))
    response = ProviderResponse("r", "m", None, "{}", 1, 1, 2, 1.0, 1, "stop")
    cache.put(key, response)
    assert cache.get(key).cached and cache.get("missing") is None


def test_atomic_ledger_reservation_settlement_resume_and_replay_are_monotone(tmp_path):
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps({
        "api_attempts": 10, "provider_reported_input_tokens": 100,
        "provider_reported_output_tokens": 20, "spent_cny": 1.0,
        "reserved_cny": 0.0, "soft_warning": False,
    }))
    config = SpecDriftBenchConfig.load()
    baseline, before_hash = capture_ledger_baseline(ledger)
    pricing = PricingCatalog(json.loads(PRICING_PATH.read_text()))
    manager = AtomicFocusedLedger(ledger, _LedgerConfigAdapter(config), "test-attempt", baseline, pricing_catalog=pricing)
    model = config.models[0]
    request = ProviderRequest(({"role": "user", "content": "visible"},), {}, "0" * 64, "public-x", 1, "component", 0)
    reservation = manager.reserve(model, request)
    assert json.loads(ledger.read_text())["reserved_cny"] > 0
    manager.settle(reservation, ProviderResponse("r", model.model_id, None, "{}", 10, 2, 12, 1.0, 1, "stop"))
    settled = manager.snapshot()
    assert settled["reserved_cny"] == 0 and settled["api_attempts"] == 11
    resumed = AtomicFocusedLedger(ledger, _LedgerConfigAdapter(config), "test-attempt", baseline, settled["spent_cny"] - 1.0, pricing_catalog=pricing)
    assert resumed.snapshot()["api_attempts"] == 11
    _, replay_hash = capture_ledger_baseline(ledger)
    assert replay_hash == hashlib.sha256(ledger.read_bytes()).hexdigest() and replay_hash != before_hash


def test_offline_rescore_preserves_old_records_and_fixes_known_evaluator_issues(tmp_path):
    records = PROJECT_ROOT / "results/experiments/phase10/driftguard_focused_canary/real_attempts/phase10a-v3-20260717-eb2efa85682d/results/records"
    before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in records.glob("*.json")}
    report = offline_rescore_v1(output=tmp_path / "offline_rescore_v1.json")
    after = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in records.glob("*.json")}
    assert before == after and report["real_api_calls"] == 0
    assert report["counts"]["evaluator_errors"] == 2
    assert report["counts"]["budget_errors"] == 2
    qwen_m06 = next(item for item in report["records"] if item["provider"] == "dashscope" and item["family"] == "M06")
    assert qwen_m06["canonical_target_tool"] == "create_issue"
    assert qwen_m06["usable_attribution_before_budget_exhaustion"]


def test_exact_36_runner_resume_and_cache_only_replay_use_no_network(tmp_path, monkeypatch):
    raw = yaml.safe_load(CONFIG_PATH.read_text())
    raw["experiment"]["run_authorized"] = True
    config_path = tmp_path / "authorized.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False))
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps({
        "api_attempts": 10, "provider_reported_input_tokens": 100,
        "provider_reported_output_tokens": 20, "spent_cny": 1.0,
        "reserved_cny": 0.0, "soft_warning": False,
    }))
    for name in ("DEEPSEEK_API_KEY", "DASHSCOPE_API_KEY", "MOONSHOT_API_KEY"):
        monkeypatch.setenv(name, "test-only")

    class FakeProvider:
        def __init__(self, model): self.model = model
        def complete(self, request):
            output = {
                "predicted_class": "AGENT_ERROR", "drift_category": "NONE",
                "target_tool": None, "normalized_location": None,
                "confidence": 0.5, "abstain": False,
                "concise_reason": "Only visible evidence was used.",
            }
            text = json.dumps(output)
            return ProviderResponse(
                "fake-" + request.public_scenario_id, self.model.model_id, None,
                text, 10, 5, 15, 1.0, 1, "stop",
            )

    monkeypatch.setattr(SpecDriftBenchRunner, "_provider", lambda self, model: FakeProvider(model))
    output = tmp_path / "attempt"
    real = SpecDriftBenchRunner(
        config_path, attempt_id="offline-real-path", mode="real",
        allow_real_api=True, confirmation=CONFIRM_36,
        output_root=output, ledger_path=ledger,
    ).run()
    assert real["summary"]["records_completed"] == 36
    resumed = SpecDriftBenchRunner(
        config_path, attempt_id="offline-real-path", mode="real",
        allow_real_api=True, confirmation=CONFIRM_36,
        output_root=output, ledger_path=ledger,
    ).run(resume=True)
    assert resumed["summary"]["records_completed"] == 36
    before = hashlib.sha256(ledger.read_bytes()).hexdigest()
    replay = SpecDriftBenchRunner(
        config_path, attempt_id="offline-real-path", mode="replay",
        output_root=output, ledger_path=ledger,
    ).run()
    assert replay["summary"]["records_completed"] == 36
    assert replay["summary"]["provider_instances"] == replay["summary"]["network_calls"] == 0
    assert replay["summary"]["attempts_delta"] == replay["summary"]["input_tokens_delta"] == replay["summary"]["output_tokens_delta"] == 0
    assert replay["summary"]["cost_cny_delta"] == 0
    assert hashlib.sha256(ledger.read_bytes()).hexdigest() == before
