import json
import os

import httpx

from driftguard.contracts.loader import PROJECT_ROOT
from driftguard.llm import LLMCache, ModelConfig, OpenAICompatibleProvider, ProviderRequest
from driftguard.phase10.config import Phase10Config
from driftguard.phase10.credentials import credential_status, load_project_dotenv, safe_status_lines


CONFIG = PROJECT_ROOT / "configs/experiments/phase10_real_pilot.yaml"
REQUEST = ProviderRequest(
    ({"role": "user", "content": "return json"},), {"type": "object"}, "0" * 64,
    "public-0123456789ab", 1, "standard", 0, 20260715, "component", "1" * 64,
)


def test_project_env_loader_only_sets_allowlisted_keys(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    (tmp_path / ".env").write_text("DEEPSEEK_API_KEY=secret-a\nDASHSCOPE_API_KEY=secret-b\nMOONSHOT_API_KEY=secret-c\nOTHER=ignored\n")
    load_project_dotenv(tmp_path)
    assert os.getenv("DEEPSEEK_API_KEY") == "secret-a"
    assert os.getenv("DASHSCOPE_API_KEY") == "secret-b"
    assert os.getenv("MOONSHOT_API_KEY") == "secret-c"
    assert os.getenv("OTHER") is None


def test_credential_preflight_exposes_only_configured_or_missing():
    values = {"DEEPSEEK_API_KEY": "do-not-print", "DASHSCOPE_API_KEY": ""}
    assert credential_status(values) == {"DEEPSEEK_API_KEY": "configured", "DASHSCOPE_API_KEY": "missing", "MOONSHOT_API_KEY": "missing"}
    output = "\n".join(safe_status_lines(values))
    assert "do-not-print" not in output and "configured" in output and "missing" in output


def _provider_and_body(model):
    captured = {}
    def handler(request):
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "r", "model": model.model_id,
            "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        })
    provider = OpenAICompatibleProvider(model, "private", httpx.Client(transport=httpx.MockTransport(handler)))
    return provider, captured


def test_deepseek_and_qwen_use_distinct_thinking_parameters_and_endpoints():
    deepseek, qwen = Phase10Config.load(CONFIG).models
    first, first_body = _provider_and_body(deepseek)
    second, second_body = _provider_and_body(qwen)
    assert first.complete(REQUEST).model == "deepseek-v4-flash"
    assert second.complete(REQUEST).model == "qwen3.7-plus"
    assert first_body["thinking"] == {"type": "disabled"}
    assert second_body["enable_thinking"] is False
    assert "enable_thinking" not in first_body and "thinking" not in second_body
    assert first.config.base_url != second.config.base_url


def test_provider_public_config_and_response_never_contain_key():
    model = Phase10Config.load(CONFIG).models[0]
    provider, _ = _provider_and_body(model)
    response = provider.complete(REQUEST)
    assert "private" not in json.dumps(model.public_dict())
    assert "private" not in json.dumps(response.public_dict())
    assert "api_key" not in json.dumps(model.public_dict()).lower()


def test_cache_isolated_by_model_mode_seed_and_config_hash():
    deepseek, qwen = Phase10Config.load(CONFIG).models
    first = LLMCache.key(deepseek, REQUEST, "a" * 64)
    second = LLMCache.key(qwen, REQUEST, "a" * 64)
    mode_changed = LLMCache.key(deepseek, ProviderRequest(**{**REQUEST.__dict__, "mode": "end_to_end"}), "a" * 64)
    seed_changed = LLMCache.key(deepseek, ProviderRequest(**{**REQUEST.__dict__, "seed": 20260716}), "a" * 64)
    config_changed = LLMCache.key(deepseek, ProviderRequest(**{**REQUEST.__dict__, "config_hash": "2" * 64}), "a" * 64)
    assert len({first, second, mode_changed, seed_changed, config_changed}) == 5


def test_env_and_raw_provider_directories_are_ignored():
    ignore = (PROJECT_ROOT / ".gitignore").read_text().splitlines()
    assert ".env" in ignore
    assert "results/experiments/**/raw_provider/" in ignore
    assert (PROJECT_ROOT / ".env.example").read_text() == "DEEPSEEK_API_KEY=\nDASHSCOPE_API_KEY=\nMOONSHOT_API_KEY=\n"
