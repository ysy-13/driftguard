from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
import hashlib
import json
import socket
import ssl
import time
from typing import Any, Iterable, Mapping

import httpx

from .configuration import ModelConfig
from .errors import (
    FailureLayer, ProviderError, ProviderTimeout, RateLimited,
    sanitize_provider_text,
)
from .models import ProviderRequest, ProviderResponse
from .redaction import assert_secret_absent
from .leakage import assert_provider_request_visible
from .rate_limit import RequestRateLimiter
from .provider_capabilities import ProviderCapabilityAdapter


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
    RETRYABLE_HTTP_STATUSES = frozenset({408, 429, 500, 502, 503, 504})
    REQUEST_ID_HEADERS = ("x-request-id", "request-id", "x-ds-request-id", "trace-id")

    def __init__(
        self,
        config: ModelConfig,
        api_key: str,
        client: httpx.Client | None = None,
        rate_limiter: RequestRateLimiter | None = None,
        cost_controller: Any | None = None,
    ):
        if not config.base_url:
            raise ValueError("base_url is required for an OpenAI-compatible provider")
        if not api_key:
            raise ValueError("API key is required")
        self.config, self._api_key = config, api_key
        self._client = client or httpx.Client(timeout=config.timeout_seconds)
        self._rate_limiter = rate_limiter or RequestRateLimiter(None)
        self._adapter = ProviderCapabilityAdapter.for_model(config)
        self._cost_controller = cost_controller

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        assert_provider_request_visible(request)
        body: dict[str, Any] = {
            "model": self.config.model_id, "messages": [dict(message) for message in request.messages],
            "max_tokens": self.config.max_output_tokens,
        }
        body.update(self._adapter.sampling_parameters(self.config))
        body.update(self._adapter.request_parameters(self.config))
        if self.config.capabilities.seed and self.config.seed is not None:
            body["seed"] = self.config.seed
        body.update(self._adapter.structured_output_parameters(self.config, dict(request.response_schema)))
        if request.tools:
            body["tools"] = [dict(item) for item in request.tools]
            if request.tool_choice is not None:
                body["tool_choice"] = request.tool_choice
            body.pop("response_format", None)
        started = time.monotonic()
        attempts = 0
        last_error: ProviderError | None = None
        while attempts <= self.config.max_provider_retries:
            attempts += 1
            reservation = None
            try:
                self._rate_limiter.acquire()
                if self._cost_controller is not None:
                    reservation = self._cost_controller.reserve(self.config, request)
                response = self._client.post(
                    self.config.base_url.rstrip("/") + "/chat/completions",
                    headers={"Authorization": f"Bearer {self._api_key}"}, json=body,
                )
                if response.status_code >= 400:
                    error, error_payload = self._http_error(response, attempts)
                    if reservation is not None and self._settle_error_usage(
                        reservation, response, error_payload, error,
                    ):
                        reservation = None
                    raise error
                response.raise_for_status()
                try:
                    payload = response.json()
                    choice = payload["choices"][0]
                    message = choice["message"]
                except (ValueError, KeyError, IndexError, TypeError) as exc:
                    raise ProviderError(
                        "provider success response could not be parsed",
                        provider_error_code="INVALID_RESPONSE_SCHEMA",
                        request_id=self._request_id(response), retryable=False,
                        attempt_number=attempts,
                        failure_layer=FailureLayer.RESPONSE_PARSE,
                    ) from exc
                tool_calls = tuple(message.get("tool_calls") or ())
                raw = message.get("content")
                if raw is None and tool_calls:
                    raw = json.dumps({"tool_calls": tool_calls}, sort_keys=True)
                if not isinstance(raw, str):
                    raise ProviderError(
                        "provider response content is not text",
                        provider_error_code="NON_TEXT_RESPONSE",
                        request_id=self._request_id(response), retryable=False,
                        attempt_number=attempts,
                        failure_layer=FailureLayer.RESPONSE_PARSE,
                    )
                usage = payload.get("usage", {})
                details = usage.get("completion_tokens_details") or usage.get("output_tokens_details") or {}
                result = ProviderResponse(
                    str(payload.get("id", "provider-response")), str(payload.get("model", self.config.model_id)), None, raw,
                    int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0)),
                    int(usage.get("total_tokens", 0)), (time.monotonic() - started) * 1000,
                    attempts, str(choice.get("finish_reason", "stop")),
                    tool_calls=tool_calls,
                    reasoning_tokens=int(details.get("reasoning_tokens", 0) or 0),
                )
                if self._cost_controller is not None:
                    self._cost_controller.settle(reservation, result)
                    reservation = None
                assert_secret_absent(result.public_dict(), self._api_key)
                return result
            except httpx.TimeoutException as exc:
                last_error = ProviderTimeout(
                    "provider request timed out", attempt_number=attempts,
                    exception_type=type(exc).__name__,
                )
            except httpx.ConnectError as exc:
                last_error = ProviderError(
                    self._transport_summary(exc), retryable=True,
                    attempt_number=attempts, failure_layer=self._connection_layer(exc),
                    exception_type=type(exc).__name__,
                )
            except httpx.NetworkError as exc:
                last_error = ProviderError(
                    self._transport_summary(exc), retryable=True,
                    attempt_number=attempts, failure_layer=FailureLayer.CONNECTION,
                    exception_type=type(exc).__name__,
                )
            except httpx.UnsupportedProtocol as exc:
                last_error = ProviderError(
                    "provider endpoint uses an unsupported protocol", retryable=False,
                    attempt_number=attempts, failure_layer=FailureLayer.CONNECTION,
                    exception_type=type(exc).__name__,
                )
            except (httpx.ProtocolError, httpx.ProxyError) as exc:
                last_error = ProviderError(
                    self._transport_summary(exc), retryable=True,
                    attempt_number=attempts, failure_layer=FailureLayer.CONNECTION,
                    exception_type=type(exc).__name__,
                )
            except ProviderError as exc:
                last_error = exc
            except httpx.HTTPError as exc:
                last_error = ProviderError(
                    "provider HTTP client error", retryable=False,
                    attempt_number=attempts, failure_layer=FailureLayer.UNKNOWN,
                    exception_type=type(exc).__name__,
                )
            finally:
                if reservation is not None and self._cost_controller is not None:
                    self._cost_controller.release(reservation)
            if last_error is not None:
                last_error.latency_ms = (time.monotonic() - started) * 1000
                assert_secret_absent(last_error.public_dict(), self._api_key)
            if last_error is not None and not last_error.retryable:
                raise last_error
            if attempts <= self.config.max_provider_retries:
                time.sleep(min(0.01 * (2 ** (attempts - 1)), 0.05))
        raise last_error or ProviderError("provider failed")

    def _http_error(
        self, response: httpx.Response, attempt_number: int,
    ) -> tuple[ProviderError, Mapping[str, Any] | None]:
        payload: Mapping[str, Any] | None = None
        code: Any = None
        message: Any = None
        try:
            candidate = response.json()
            if isinstance(candidate, Mapping):
                payload = candidate
                nested = candidate.get("error")
                source = nested if isinstance(nested, Mapping) else candidate
                code, message = source.get("code"), source.get("message")
        except (ValueError, TypeError):
            try:
                message = response.text
            except Exception:
                message = "non-JSON provider error response"
        status = int(response.status_code)
        retryable = status in self.RETRYABLE_HTTP_STATUSES
        summary = sanitize_provider_text(
            message or f"provider returned HTTP {status}",
            secrets=(self._api_key,), limit=512,
        )
        kwargs = {
            "status_code": status,
            "provider_error_code": sanitize_provider_text(
                code, secrets=(self._api_key,), limit=128,
            ) if code is not None else None,
            "request_id": self._request_id(response),
            "retryable": retryable,
            "attempt_number": attempt_number,
            "failure_layer": FailureLayer.HTTP,
        }
        error: ProviderError = (
            RateLimited(summary, **kwargs) if status == 429
            else ProviderError(summary, **kwargs)
        )
        return error, payload

    def _request_id(self, response: httpx.Response) -> str | None:
        for name in self.REQUEST_ID_HEADERS:
            value = response.headers.get(name)
            if value:
                return sanitize_provider_text(value, secrets=(self._api_key,), limit=128)
        return None

    def _settle_error_usage(
        self, reservation: Any, response: httpx.Response,
        payload: Mapping[str, Any] | None, error: ProviderError,
    ) -> bool:
        usage = payload.get("usage") if isinstance(payload, Mapping) else None
        if self._cost_controller is None or not isinstance(usage, Mapping):
            return False
        input_tokens = int(usage.get("prompt_tokens", 0) or 0)
        output_tokens = int(usage.get("completion_tokens", 0) or 0)
        total_tokens = int(usage.get("total_tokens", input_tokens + output_tokens) or 0)
        result = ProviderResponse(
            error.request_id or "provider-error", self.config.model_id, None, "",
            input_tokens, output_tokens, total_tokens, 0.0, 1, "error",
            error=error.sanitized_message,
        )
        self._cost_controller.settle(reservation, result)
        return True

    def _transport_summary(self, exc: Exception) -> str:
        return sanitize_provider_text(
            exc, secrets=(self._api_key,), limit=256,
        ) or "provider connection failed"

    @staticmethod
    def _connection_layer(exc: Exception) -> FailureLayer:
        chain: list[BaseException] = []
        current: BaseException | None = exc
        while current is not None and current not in chain:
            chain.append(current)
            current = current.__cause__ or current.__context__
        if any(isinstance(item, socket.gaierror) for item in chain):
            return FailureLayer.DNS
        if any(isinstance(item, ssl.SSLError) for item in chain):
            return FailureLayer.TLS
        text = " ".join(str(item).lower() for item in chain)
        if any(marker in text for marker in ("name resolution", "getaddrinfo", "nodename", "dns")):
            return FailureLayer.DNS
        if any(marker in text for marker in ("ssl", "tls", "certificate")):
            return FailureLayer.TLS
        return FailureLayer.CONNECTION
