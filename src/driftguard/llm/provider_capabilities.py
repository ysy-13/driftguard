from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .configuration import ModelConfig


@dataclass(frozen=True)
class ProviderCapabilityAdapter:
    provider: str
    model_id: str
    profile_id: str

    KIMI_K26_PROFILE = "moonshot-kimi-k2.6-thinking-v1"
    KIMI_K26_NONTHINKING_PROFILE = "moonshot-kimi-k2.6-nonthinking-v1"

    @classmethod
    def for_model(cls, config: ModelConfig) -> "ProviderCapabilityAdapter":
        if config.provider not in {"deepseek", "dashscope", "moonshot", "mock", "openai_compatible"}:
            raise ValueError(f"unsupported provider: {config.provider}")
        if config.provider == "moonshot" and config.model_id == "kimi-k2.6":
            profile_id = (
                cls.KIMI_K26_NONTHINKING_PROFILE
                if config.thinking_mode == "disabled"
                else cls.KIMI_K26_PROFILE
            )
        else:
            profile_id = f"{config.provider}-openai-compatible-v1"
        return cls(config.provider, config.model_id, profile_id)

    def sampling_parameters(self, config: ModelConfig) -> dict[str, Any]:
        if self.profile_id not in {self.KIMI_K26_PROFILE, self.KIMI_K26_NONTHINKING_PROFILE}:
            return {"temperature": config.temperature, "top_p": config.top_p}
        expected_temperature = 0.6 if self.profile_id == self.KIMI_K26_NONTHINKING_PROFILE else 1.0
        if config.temperature != expected_temperature:
            raise ValueError(
                f"kimi-k2.6 capability profile requires temperature={expected_temperature}"
            )
        if config.top_p != 0.95:
            raise ValueError("kimi-k2.6 capability profile requires top_p=0.95")
        if self.profile_id == self.KIMI_K26_NONTHINKING_PROFILE and config.max_output_tokens != 1024:
            raise ValueError("kimi-k2.6 nonthinking capability profile requires max_tokens=1024")
        return {
            "temperature": expected_temperature,
            "top_p": 0.95,
            "n": 1,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
        }

    def request_parameters(self, config: ModelConfig) -> dict[str, Any]:
        if self.profile_id == self.KIMI_K26_NONTHINKING_PROFILE:
            if config.thinking_mode != "disabled":
                raise ValueError("kimi-k2.6 nonthinking capability profile requires thinking_mode=disabled")
            return {"thinking": {"type": "disabled"}, "stream": False}
        if self.profile_id == self.KIMI_K26_PROFILE:
            if config.thinking_mode != "enabled":
                raise ValueError("kimi-k2.6 capability profile requires thinking_mode=enabled")
            return {}
        if config.thinking_mode != "disabled":
            raise ValueError("Phase 10 requires thinking_mode=disabled")
        if self.provider == "deepseek":
            return {"thinking": {"type": "disabled"}}
        if self.provider == "dashscope":
            return {"enable_thinking": False}
        return {}

    def structured_output_parameters(self, config: ModelConfig, schema: dict[str, Any]) -> dict[str, Any]:
        if not schema:
            return {}
        if config.structured_output_mode == "json_schema" and config.capabilities.json_schema:
            return {
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "driftguard_output", "schema": schema},
                }
            }
        if config.capabilities.strict_json:
            return {"response_format": {"type": "json_object"}}
        return {}

    def public_profile(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema_version": "provider-capability-profile-v1",
            "profile_id": self.profile_id,
            "provider": self.provider,
            "model_id": self.model_id,
        }
        if self.profile_id in {self.KIMI_K26_PROFILE, self.KIMI_K26_NONTHINKING_PROFILE}:
            nonthinking = self.profile_id == self.KIMI_K26_NONTHINKING_PROFILE
            value.update({
                "thinking_mode": "disabled" if nonthinking else "enabled",
                "sampling_parameters": {
                    "temperature": 0.6 if nonthinking else 1.0,
                    "top_p": 0.95,
                    "n": 1,
                    "presence_penalty": 0.0,
                    "frequency_penalty": 0.0,
                },
                "structured_output": {"type": "json_object"},
            })
            if nonthinking:
                value.update({"max_tokens": 1024, "stream": False})
        return value
