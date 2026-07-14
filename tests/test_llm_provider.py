import json

import httpx
import pytest

from driftguard.contracts.loader import PROJECT_ROOT
from driftguard.llm import (
    LLMCache, MockProvider, ModelCapabilities, ModelConfig, OpenAICompatibleProvider,
    ProviderError, ProviderRequest, ProviderTimeout, RateLimited, RequestRateLimiter,
    parse_structured_output,
)
from driftguard.runtime import RuntimeProfile


SCHEMA = json.loads((PROJECT_ROOT / "benchmark/schemas/agent_action_schema_v1.json").read_text())
REQUEST = ProviderRequest(
    ({"role": "user", "content": "test"},), SCHEMA, "0" * 64,
    "public-0123456789ab", 1, "standard", 0,
)


def test_mock_provider_is_deterministic_and_offline(monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network")))
    output = {"action_type": "ABSTAIN", "concise_decision_summary": "done"}
    first = MockProvider([output]).complete(REQUEST)
    second = MockProvider([output]).complete(REQUEST)
    assert first.raw_text == second.raw_text and first.response_id == second.response_id


@pytest.mark.parametrize("script,exception", [("TIMEOUT", ProviderTimeout), ("429", RateLimited), ("500", ProviderError), ("503", ProviderError)])
def test_mock_provider_fault_scripts(script, exception):
    with pytest.raises(exception):
        MockProvider([script]).complete(REQUEST)


def test_openai_compatible_config_reads_environment_without_key():
    config = ModelConfig.from_mapping({}, {
        "DRIFTGUARD_LLM_MODEL": "test-model", "DRIFTGUARD_LLM_BASE_URL": "https://example.invalid/v1",
    })
    assert config.model_id == "test-model" and config.base_url.endswith("/v1")
    assert "api_key" not in config.public_dict()


def test_openai_provider_retries_5xx_separately_from_tools():
    attempts = {"count": 0}
    def handler(request):
        attempts["count"] += 1
        if attempts["count"] < 3:
            return httpx.Response(503)
        return httpx.Response(200, json={
            "id": "r1", "choices": [{"message": {"content": '{"action_type":"ABSTAIN","concise_decision_summary":"ok"}'}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        })
    config = ModelConfig(
        "openai_compatible", "test", "https://example.invalid/v1", max_provider_retries=2,
        capabilities=ModelCapabilities(json_schema=True),
    )
    provider = OpenAICompatibleProvider(config, "secret-key", httpx.Client(transport=httpx.MockTransport(handler)))
    response = provider.complete(REQUEST)
    assert response.provider_attempts == 3 and attempts["count"] == 3
    assert "secret-key" not in repr(response.public_dict())


def test_openai_provider_timeout_is_classified():
    def handler(request):
        raise httpx.ReadTimeout("timeout", request=request)
    config = ModelConfig("openai_compatible", "test", "https://example.invalid", max_provider_retries=0)
    with pytest.raises(ProviderTimeout):
        OpenAICompatibleProvider(config, "secret", httpx.Client(transport=httpx.MockTransport(handler))).complete(REQUEST)


def test_provider_rate_limiter_is_applied_to_every_provider_attempt():
    class CountingLimiter:
        def __init__(self):
            self.calls = 0

        def acquire(self):
            self.calls += 1

    attempts = {"count": 0}
    def handler(request):
        attempts["count"] += 1
        if attempts["count"] == 1:
            return httpx.Response(503)
        return httpx.Response(200, json={
            "id": "r", "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
            "usage": {},
        })
    limiter = CountingLimiter()
    config = ModelConfig("openai_compatible", "test", "https://example.invalid", max_provider_retries=1)
    provider = OpenAICompatibleProvider(
        config, "secret", httpx.Client(transport=httpx.MockTransport(handler)), limiter,
    )
    provider.complete(REQUEST)
    assert limiter.calls == attempts["count"] == 2


def test_request_rate_limiter_is_deterministic_with_injected_clock():
    now = [0.0]
    sleeps = []
    def sleep(delay):
        sleeps.append(delay)
        now[0] += delay
    limiter = RequestRateLimiter(2, clock=lambda: now[0], sleeper=sleep)
    limiter.acquire()
    limiter.acquire()
    assert sleeps == [0.5]


def test_structured_output_validates_json_schema():
    parsed = parse_structured_output('{"action_type":"ABSTAIN","concise_decision_summary":"safe"}', SCHEMA)
    assert parsed["action_type"] == "ABSTAIN"


@pytest.mark.parametrize("raw", ["not json", "[]", '{"action_type":"TOOL_CALL","concise_decision_summary":"x"}'])
def test_structured_output_rejects_invalid_values(raw):
    with pytest.raises(ValueError):
        parse_structured_output(raw, SCHEMA)


def test_cache_key_and_artifact_exclude_api_key(tmp_path):
    config = ModelConfig("mock", "m")
    key = LLMCache.key(config, REQUEST, "1" * 64)
    assert "secret" not in key and len(key) == 64
    cache = LLMCache(tmp_path)
    response = MockProvider([{"action_type": "ABSTAIN", "concise_decision_summary": "ok"}]).complete(REQUEST)
    cache.put(key, response, "secret")
    assert cache.get(key).cached
    assert "secret" not in (tmp_path / "raw" / f"{key}.json").read_text()
    assert (tmp_path / "parsed" / f"{key}.json").exists()


def test_provider_and_cache_reject_runtime_profile_objects(tmp_path):
    runtime = RuntimeProfile("p", "get_repository", "response_shape", True, True, 3, None, "x", {})
    with pytest.raises(TypeError):
        MockProvider().complete(runtime)
    with pytest.raises(TypeError):
        LLMCache.key(ModelConfig("mock", "m"), runtime, "0" * 64)


def test_cache_key_rejects_hidden_contract_canary():
    hidden = ProviderRequest(
        ({"role": "user", "content": "runtime_contract"},), SCHEMA, "0" * 64,
        "public-0123456789ab", 1, "standard", 0,
    )
    with pytest.raises(ValueError):
        LLMCache.key(ModelConfig("mock", "m"), hidden, "0" * 64)
