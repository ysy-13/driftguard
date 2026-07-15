from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .configuration import ModelConfig


@dataclass(frozen=True)
class ProviderCapabilityAdapter:
    provider: str

    @classmethod
    def for_model(cls, config: ModelConfig) -> "ProviderCapabilityAdapter":
        if config.provider not in {"deepseek", "dashscope", "mock", "openai_compatible"}:
            raise ValueError(f"unsupported provider: {config.provider}")
        return cls(config.provider)

    def request_parameters(self, config: ModelConfig) -> dict[str, Any]:
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
