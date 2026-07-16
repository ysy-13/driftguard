from __future__ import annotations

from copy import deepcopy
import json
from typing import Any, Iterable

from driftguard.llm import (
    LLMProvider, MockProvider, ModelConfig, OpenAICompatibleProvider,
    ProviderRequest, ProviderResponse, RequestRateLimiter, model_api_key,
)


class ReplayCacheMiss(RuntimeError):
    """Raised when replay tries to fall through to a Provider."""


class CacheOnlyProvider(LLMProvider):
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        self.calls += 1
        raise ReplayCacheMiss(
            f"focused replay cache miss at {request.mode} for {request.public_scenario_id}"
        )


class EvidenceRefMockProvider(MockProvider):
    """Offline scripted provider that resolves only live evidence placeholders."""

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        if not self.outputs:
            return super().complete(request)
        template = deepcopy(self.outputs.pop(0))
        try:
            payload = json.loads(request.messages[-1]["content"])
        except (json.JSONDecodeError, KeyError):
            payload = {}
        trace = payload.get("live_evidence_trace") or payload.get(
            "live_agent_view", {}
        ).get("live_evidence_trace", {})
        events = trace.get("events", [])
        failures = [
            event["event_id"] for event in events
            if event.get("probe_metadata", {}).get("independent_failure")
        ]
        probes = [
            event["event_id"] for event in events
            if event.get("event_type") == "probe_executed"
        ]
        replacements = {
            "@FAILURE_1": failures[0] if failures else "missing-live-failure-1",
            "@FAILURE_2": failures[1] if len(failures) > 1 else "missing-live-failure-2",
            "@PROBE": probes[-1] if probes else "missing-live-probe",
        }

        def replace(value: Any) -> Any:
            if isinstance(value, dict):
                return {key: replace(child) for key, child in value.items()}
            if isinstance(value, list):
                return [replace(child) for child in value]
            return replacements.get(value, value)

        self.outputs.insert(0, replace(template))
        return super().complete(request)


class FocusedProviderAdapter:
    """Creates providers without ever interpreting benchmark labels or patch answers."""

    def __init__(
        self, mode: str, model: ModelConfig, *, allow_real_api: bool = False,
        confirm_focused_canary: bool = False, run_authorized: bool = False,
        cost_controller: Any | None = None,
    ) -> None:
        if mode not in {"mock", "replay", "real"}:
            raise ValueError(f"unsupported Focused provider mode: {mode}")
        self.mode, self.model = mode, model
        self.allow_real_api = allow_real_api
        self.confirm_focused_canary = confirm_focused_canary
        self.run_authorized = run_authorized
        self.cost_controller = cost_controller
        self.created: list[LLMProvider] = []

    def create(
        self, outputs: Iterable[Any] | None = None, *, resolve_evidence_refs: bool = False,
    ) -> LLMProvider:
        if self.mode == "mock":
            provider_class = EvidenceRefMockProvider if resolve_evidence_refs else MockProvider
            provider: LLMProvider = provider_class(outputs, self.model.model_id)
        elif self.mode == "replay":
            provider = CacheOnlyProvider()
        else:
            if not (self.run_authorized and self.allow_real_api and self.confirm_focused_canary):
                raise PermissionError(
                    "Focused real execution requires config authorization, --allow-real-api, "
                    "and --confirm-focused-canary"
                )
            key = model_api_key(self.model)
            if not key:
                raise PermissionError(f"missing configured credential for {self.model.provider}")
            provider = OpenAICompatibleProvider(
                self.model, key, rate_limiter=RequestRateLimiter(1.0),
                cost_controller=self.cost_controller,
            )
        self.created.append(provider)
        return provider

    @property
    def fallback_calls(self) -> int:
        return sum(getattr(provider, "calls", 0) for provider in self.created if isinstance(provider, CacheOnlyProvider))

    @property
    def mock_calls(self) -> int:
        return sum(getattr(provider, "calls", 0) for provider in self.created if isinstance(provider, MockProvider))
