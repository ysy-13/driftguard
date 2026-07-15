from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from typing import Any, Mapping


@dataclass(frozen=True)
class ModelCapabilities:
    seed: bool = False
    json_schema: bool = False
    strict_json: bool = True
    tool_calling: bool = False

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)


@dataclass(frozen=True)
class ModelConfig:
    provider: str
    model_id: str
    base_url: str | None = None
    temperature: float = 0.0
    top_p: float = 1.0
    max_output_tokens: int = 512
    seed: int | None = 7
    timeout_seconds: float = 30.0
    max_provider_retries: int = 2
    structured_output_mode: str = "json_schema"
    capabilities: ModelCapabilities = ModelCapabilities()
    api_key_env: str | None = None
    thinking_mode: str = "disabled"
    concurrency: int = 1

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], environ: Mapping[str, str] | None = None) -> "ModelConfig":
        env = os.environ if environ is None else environ
        provider = str(value.get("provider", "mock"))
        model_id = str(value.get("model_id") or env.get("DRIFTGUARD_LLM_MODEL") or "mock-deterministic")
        base_url = value.get("base_url") or env.get("DRIFTGUARD_LLM_BASE_URL")
        capabilities = value.get("capabilities", {})
        return cls(
            provider=provider, model_id=model_id, base_url=str(base_url) if base_url else None,
            temperature=float(value.get("temperature", 0.0)), top_p=float(value.get("top_p", 1.0)),
            max_output_tokens=int(value.get("max_output_tokens", 512)),
            seed=value.get("seed", 7), timeout_seconds=float(value.get("timeout_seconds", 30.0)),
            max_provider_retries=int(value.get("max_provider_retries", 2)),
            structured_output_mode=str(value.get("structured_output_mode", "json_schema")),
            capabilities=ModelCapabilities(**capabilities),
            api_key_env=str(value["api_key_env"]) if value.get("api_key_env") else None,
            thinking_mode=str(value.get("thinking_mode", "disabled")),
            concurrency=int(value.get("concurrency", 1)),
        )

    def public_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("api_key_env", None)
        value["capabilities"] = self.capabilities.to_dict()
        return value


def api_key_from_environment(environ: Mapping[str, str] | None = None) -> str | None:
    env = os.environ if environ is None else environ
    return env.get("DRIFTGUARD_LLM_API_KEY")


def model_api_key(config: ModelConfig, environ: Mapping[str, str] | None = None) -> str | None:
    env = os.environ if environ is None else environ
    return env.get(config.api_key_env) if config.api_key_env else api_key_from_environment(env)
