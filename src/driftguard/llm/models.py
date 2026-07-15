from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class ProviderRequest:
    messages: tuple[Mapping[str, str], ...]
    response_schema: Mapping[str, Any]
    prompt_hash: str
    public_scenario_id: str
    episode: int
    method: str
    repetition: int
    seed: int | None = None
    mode: str = ""
    config_hash: str = ""
    tools: tuple[Mapping[str, Any], ...] = ()
    tool_choice: str | Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ProviderResponse:
    response_id: str
    model: str
    parsed_output: Mapping[str, Any] | None
    raw_text: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_ms: float
    provider_attempts: int
    finish_reason: str
    cached: bool = False
    error: str | None = None
    tool_calls: tuple[Mapping[str, Any], ...] = ()

    def public_dict(self) -> dict[str, Any]:
        return {
            "response_id": self.response_id, "model": self.model,
            "parsed_output": dict(self.parsed_output) if self.parsed_output is not None else None,
            "raw_text": self.raw_text, "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens, "total_tokens": self.total_tokens,
            "latency_ms": self.latency_ms, "provider_attempts": self.provider_attempts,
            "finish_reason": self.finish_reason, "cached": self.cached, "error": self.error,
            "tool_calls": [dict(item) for item in self.tool_calls],
        }


@dataclass
class UsageCounter:
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    format_repairs: int = 0
    actual_models: list[str] = field(default_factory=list)
    provider_attempts: int = 0

    def add(self, response: ProviderResponse) -> None:
        self.llm_calls += 1
        self.input_tokens += response.input_tokens
        self.output_tokens += response.output_tokens
        self.latency_ms += response.latency_ms
        self.actual_models.append(response.model)
        self.provider_attempts += response.provider_attempts
