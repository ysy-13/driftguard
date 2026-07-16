from __future__ import annotations

import hashlib
import json

import httpx
import pytest

from driftguard.llm import (
    FailureLayer, ModelCapabilities, ModelConfig, OpenAICompatibleProvider,
    ProviderError, ProviderRequest, ProviderTimeout, RateLimited,
)
from driftguard.phase10.costs import CostBudgetManager
from driftguard.phase10.pricing import PricingCatalog
from driftguard.runners.focused_live_healing_runner import LEDGER


SECRET = "provider-test-secret-value"
FORMAL_LEDGER_HASH_AT_IMPORT = hashlib.sha256(LEDGER.read_bytes()).hexdigest()
REQUEST = ProviderRequest(
    ({"role": "system", "content": "safe system"}, {"role": "user", "content": "safe user"}),
    {"type": "object"}, "0" * 64, "public-provider-test", 1, "driftguard_llm", 0,
)
MODEL = ModelConfig(
    provider="deepseek", base_url="https://provider.invalid", model_id="deepseek-v4-flash",
    max_output_tokens=4096, timeout_seconds=1, max_provider_retries=3,
    structured_output_mode="strict_json", thinking_mode="disabled",
    capabilities=ModelCapabilities(strict_json=True, tool_calling=True),
)


@pytest.fixture(autouse=True)
def formal_ledger_is_read_only():
    assert hashlib.sha256(LEDGER.read_bytes()).hexdigest() == FORMAL_LEDGER_HASH_AT_IMPORT
    yield
    assert hashlib.sha256(LEDGER.read_bytes()).hexdigest() == FORMAL_LEDGER_HASH_AT_IMPORT


def _costs() -> CostBudgetManager:
    return CostBudgetManager(
        PricingCatalog.load_default(), 90.0, 100.0, 120_000, 12_000,
    )


def _call(handler, costs=None):
    client = httpx.Client(transport=httpx.MockTransport(handler), timeout=1)
    provider = OpenAICompatibleProvider(MODEL, SECRET, client=client, cost_controller=costs)
    return provider, client


@pytest.mark.parametrize("status", [400, 401, 402, 422])
def test_permanent_http_errors_are_attempted_once(status):
    calls = 0
    costs = _costs()

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(status, json={"error": {"code": f"E{status}", "message": "permanent"}})

    provider, client = _call(handler, costs)
    try:
        with pytest.raises(ProviderError) as caught:
            provider.complete(REQUEST)
    finally:
        client.close()
    error = caught.value
    assert calls == costs.snapshot()["api_attempts"] == 1
    assert error.status_code == status and error.provider_error_code == f"E{status}"
    assert error.retryable is False and error.attempt_number == 1
    assert error.failure_layer == FailureLayer.HTTP
    assert costs.snapshot()["reserved_cny"] == 0


@pytest.mark.parametrize("status,error_type", [
    (408, ProviderError), (429, RateLimited), (500, ProviderError), (502, ProviderError),
    (503, ProviderError), (504, ProviderError),
])
def test_retryable_http_errors_use_initial_plus_three_retries(status, error_type):
    calls = 0
    costs = _costs()

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(status, json={"error": {"code": f"E{status}", "message": "transient"}})

    provider, client = _call(handler, costs)
    try:
        with pytest.raises(error_type) as caught:
            provider.complete(REQUEST)
    finally:
        client.close()
    assert calls == costs.snapshot()["api_attempts"] == 4
    assert caught.value.retryable is True and caught.value.attempt_number == 4
    assert costs.snapshot()["reserved_cny"] == 0


@pytest.mark.parametrize("kind,expected_type,expected_layer", [
    ("timeout", ProviderTimeout, FailureLayer.TIMEOUT),
    ("dns", ProviderError, FailureLayer.DNS),
    ("tls", ProviderError, FailureLayer.TLS),
])
def test_transport_errors_are_retryable_and_classified(kind, expected_type, expected_layer):
    calls = 0
    costs = _costs()

    def handler(request):
        nonlocal calls
        calls += 1
        if kind == "timeout":
            raise httpx.ReadTimeout("read timed out", request=request)
        message = "name resolution failed" if kind == "dns" else "TLS certificate verify failed"
        raise httpx.ConnectError(message, request=request)

    provider, client = _call(handler, costs)
    try:
        with pytest.raises(expected_type) as caught:
            provider.complete(REQUEST)
    finally:
        client.close()
    assert calls == costs.snapshot()["api_attempts"] == 4
    assert caught.value.failure_layer == expected_layer
    assert caught.value.retryable is True and caught.value.actual_network_attempts == 4
    assert costs.snapshot()["reserved_cny"] == 0


def test_http_metadata_is_sanitized_and_request_id_is_preserved():
    long_message = f"Authorization: Bearer {SECRET} api_key={SECRET} " + "x" * 800

    def handler(request):
        return httpx.Response(
            400, headers={"x-request-id": "safe-request-123"},
            json={"error": {"code": "invalid_api_key", "message": long_message}},
        )

    provider, client = _call(handler, _costs())
    try:
        with pytest.raises(ProviderError) as caught:
            provider.complete(REQUEST)
    finally:
        client.close()
    public = caught.value.public_dict()
    encoded = json.dumps(public)
    assert public["status_code"] == 400
    assert public["provider_error_code"] == "invalid_api_key"
    assert public["request_id"] == "safe-request-123"
    assert len(public["sanitized_message"]) <= 512
    assert SECRET not in encoded and "Bearer provider" not in encoded


def test_non_json_error_body_is_safely_truncated_and_redacted():
    body = f"api_key={SECRET} " + "y" * 1000

    def handler(request):
        return httpx.Response(415, text=body)

    provider, client = _call(handler, _costs())
    try:
        with pytest.raises(ProviderError) as caught:
            provider.complete(REQUEST)
    finally:
        client.close()
    assert len(caught.value.sanitized_message) <= 512
    assert SECRET not in caught.value.sanitized_message
    assert caught.value.retryable is False and caught.value.attempt_number == 1


def test_error_response_usage_is_settled_without_reservation_leak():
    costs = _costs()

    def handler(request):
        return httpx.Response(400, json={
            "error": {"code": "bad_request", "message": "charged failure"},
            "usage": {"prompt_tokens": 11, "completion_tokens": 2, "total_tokens": 13},
        })

    provider, client = _call(handler, costs)
    try:
        with pytest.raises(ProviderError):
            provider.complete(REQUEST)
    finally:
        client.close()
    snapshot = costs.snapshot()
    assert snapshot["api_attempts"] == 1
    assert snapshot["provider_reported_input_tokens"] == 11
    assert snapshot["provider_reported_output_tokens"] == 2
    assert snapshot["spent_cny"] > 0 and snapshot["reserved_cny"] == 0


def test_success_response_parse_failure_is_permanent_and_observable():
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(200, text="not-json")

    provider, client = _call(handler, _costs())
    try:
        with pytest.raises(ProviderError) as caught:
            provider.complete(REQUEST)
    finally:
        client.close()
    assert calls == 1
    assert caught.value.failure_layer == FailureLayer.RESPONSE_PARSE
    assert caught.value.provider_error_code == "INVALID_RESPONSE_SCHEMA"
    assert caught.value.retryable is False
