from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
import hashlib
import json
import time
from typing import Any, Iterable, Mapping

import httpx

from .configuration import ModelConfig
from .errors import ProviderError, ProviderTimeout, RateLimited
from .models import ProviderRequest, ProviderResponse
from .redaction import assert_secret_absent
from .leakage import assert_provider_request_visible
from .rate_limit import RequestRateLimiter


class LLMProvider(ABC):
    @abstractmethod
    def complete(self, request: ProviderRequest) -> ProviderResponse:
        raise NotImplementedError


class MockProvider(LLMProvider):
    """Deterministic offline provider. No method in this class performs I/O."""

    def __init__(self, outputs: Iterable[Any] | None = None, model: str = "mock-deterministic"):
        self.outputs = list(outputs or [])
        self.model = model
        self.calls = 0

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        assert_provider_request_visible(request)
        self.calls += 1
        item = self.outputs.pop(0) if self.outputs else {
            "action_type": "FINAL_ANSWER", "answer": "mock completed",
            "concise_decision_summary": "deterministic mock response",
        }
        if isinstance(item, str) and item == "TIMEOUT":
            raise ProviderTimeout("mock timeout")
        if isinstance(item, str) and item == "429":
            raise RateLimited("mock rate limit")
        if isinstance(item, str) and item in {"500", "503"}:
            raise ProviderError(f"mock server error {item}")
        raw = item if isinstance(item, str) else json.dumps(item, sort_keys=True)
        response_id = hashlib.sha256(f"{self.calls}|{raw}".encode()).hexdigest()[:16]
        return ProviderResponse(
            f"mock-{response_id}", self.model, deepcopy(item) if isinstance(item, dict) else None,
            raw, max(1, sum(len(message.get("content", "")) for message in request.messages) // 4),
            max(1, len(raw) // 4),
            max(2, (sum(len(message.get("content", "")) for message in request.messages) + len(raw)) // 4),
            0.0, 1, "stop",
        )


class OpenAICompatibleProvider(LLMProvider):
    def __init__(
        self,
        config: ModelConfig,
        api_key: str,
        client: httpx.Client | None = None,
        rate_limiter: RequestRateLimiter | None = None,
    ):
        if not config.base_url:
            raise ValueError("base_url is required for an OpenAI-compatible provider")
        if not api_key:
            raise ValueError("API key is required")
        self.config, self._api_key = config, api_key
        self._client = client or httpx.Client(timeout=config.timeout_seconds)
        self._rate_limiter = rate_limiter or RequestRateLimiter(None)

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        assert_provider_request_visible(request)
        body: dict[str, Any] = {
            "model": self.config.model_id, "messages": [dict(message) for message in request.messages],
            "temperature": self.config.temperature, "top_p": self.config.top_p,
            "max_tokens": self.config.max_output_tokens,
        }
        if self.config.capabilities.seed and self.config.seed is not None:
            body["seed"] = self.config.seed
        if self.config.structured_output_mode == "json_schema" and self.config.capabilities.json_schema:
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "driftguard_output", "schema": dict(request.response_schema)}}
        started = time.monotonic()
        attempts = 0
        last_error: Exception | None = None
        while attempts <= self.config.max_provider_retries:
            attempts += 1
            try:
                self._rate_limiter.acquire()
                response = self._client.post(
                    self.config.base_url.rstrip("/") + "/chat/completions",
                    headers={"Authorization": f"Bearer {self._api_key}"}, json=body,
                )
                if response.status_code == 429:
                    raise RateLimited("provider returned 429")
                if response.status_code >= 500:
                    raise ProviderError(f"provider returned {response.status_code}")
                response.raise_for_status()
                payload = response.json()
                choice = payload["choices"][0]
                raw = choice["message"]["content"]
                usage = payload.get("usage", {})
                result = ProviderResponse(
                    str(payload.get("id", "provider-response")), self.config.model_id, None, raw,
                    int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0)),
                    int(usage.get("total_tokens", 0)), (time.monotonic() - started) * 1000,
                    attempts, str(choice.get("finish_reason", "stop")),
                )
                assert_secret_absent(result.public_dict(), self._api_key)
                return result
            except httpx.TimeoutException as exc:
                last_error = ProviderTimeout("provider request timed out")
            except RateLimited as exc:
                last_error = exc
            except (httpx.HTTPError, ProviderError) as exc:
                last_error = ProviderError(str(exc)) if not isinstance(exc, ProviderError) else exc
            if attempts <= self.config.max_provider_retries:
                time.sleep(min(0.01 * (2 ** (attempts - 1)), 0.05))
        if isinstance(last_error, ProviderTimeout):
            raise last_error
        if isinstance(last_error, RateLimited):
            raise last_error
        raise last_error or ProviderError("provider failed")
