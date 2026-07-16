from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from driftguard.live.focused_config import FOCUSED_CONFIG, FocusedLiveHealingConfig
from driftguard.live.provider_adapter import EvidenceRefMockProvider, ReplayCacheMiss
from driftguard.llm import ModelConfig, ProviderError, ProviderRequest, ProviderResponse
from driftguard.phase10.costs import CostHardLimit
from driftguard.runners.focused_live_healing_runner import (
    AtomicFocusedLedger, CONFIRMATION, LEDGER,
    REAL_CACHE_NAMESPACE, FocusedLiveHealingRunner,
)

FORMAL_LEDGER_HASH_AT_IMPORT = hashlib.sha256(LEDGER.read_bytes()).hexdigest()
TEST_LEDGER_BASELINE = {
    "api_attempts": 863,
    "provider_reported_input_tokens": 3_580_832,
    "provider_reported_output_tokens": 84_182,
    "spent_cny": 5.202644120,
}


@pytest.fixture(scope="module", autouse=True)
def formal_ledger_is_read_only():
    before = hashlib.sha256(LEDGER.read_bytes()).hexdigest()
    assert before == FORMAL_LEDGER_HASH_AT_IMPORT
    yield
    assert hashlib.sha256(LEDGER.read_bytes()).hexdigest() == FORMAL_LEDGER_HASH_AT_IMPORT


def _authorized_config(tmp_path: Path) -> Path:
    raw = yaml.safe_load(FOCUSED_CONFIG.read_text(encoding="utf-8"))
    raw["experiment"]["run_authorized"] = True
    path = tmp_path / "focused-authorized.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def _ledger(tmp_path: Path, **overrides) -> Path:
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps({
        **TEST_LEDGER_BASELINE,
        "reserved_cny": 0.0,
        "soft_warning": False,
        **overrides,
    }), encoding="utf-8")
    return path


def _credentials(configured: bool = True):
    value = "configured" if configured else "missing"
    return {"DEEPSEEK_API_KEY": value, "DASHSCOPE_API_KEY": value}


class _CostedFakeProvider(EvidenceRefMockProvider):
    def __init__(self, outputs, config: ModelConfig, costs, *, fail: bool = False):
        super().__init__(outputs, model=config.model_id)
        self.config = config
        self.costs = costs
        self.fail = fail

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        reservation = self.costs.reserve(self.config, request)
        try:
            if self.fail:
                raise ProviderError(
                    "offline injected Provider failure", status_code=400,
                    provider_error_code="FAKE_BAD_REQUEST",
                    request_id="fake-request-id", retryable=False,
                    attempt_number=1, failure_layer="HTTP",
                )
            response = super().complete(request)
            response = replace(
                response,
                response_id=response.response_id.replace("mock-", "fake-"),
                model=self.config.model_id,
                provider_attempts=1,
            )
            self.costs.settle(reservation, response)
            return response
        except Exception:
            self.costs.release(reservation)
            raise


class _FakeFactory:
    def __init__(self, fail_providers=(), fail_calls=()):
        self.fail_providers = set(fail_providers)
        self.fail_calls = set(fail_calls)
        self.instances = []

    def __call__(self, config, outputs, resolve_evidence_refs, costs):
        if config.provider in self.fail_providers:
            raise ProviderError("offline injected Provider initialization failure")
        provider = _CostedFakeProvider(
            outputs, config, costs, fail=config.provider in self.fail_calls,
        )
        self.instances.append(provider)
        return provider

    @property
    def calls(self):
        return sum(provider.calls for provider in self.instances)


def _real_runner(
    tmp_path: Path, config: Path, ledger: Path, factory: _FakeFactory,
    *, output: Path | None = None, cache: Path | None = None,
    allow: bool = True, confirmation: str | None = CONFIRMATION,
    credentials_configured: bool = True,
):
    return FocusedLiveHealingRunner(
        config,
        mode="real",
        output=output or tmp_path / "real-results",
        cache_root=cache or tmp_path / "real-cache",
        attempt_id="offline-fake-attempt",
        allow_real_api=allow,
        confirm_focused_canary=confirmation,
        provider_factory=factory,
        credential_checker=lambda: _credentials(credentials_configured),
        ledger_path=ledger,
        git_state_provider=lambda: {"commit": "5983a50", "dirty": True},
        secret_values=("FAKE-SECRET-DO-NOT-STORE",),
    )


@pytest.mark.parametrize(
    "authorized,allow,confirmation,credentials_configured,error",
    [
        (False, True, CONFIRMATION, True, "run_authorized:true"),
        (True, False, CONFIRMATION, True, "--allow-real-api"),
        (True, True, None, True, "requires confirmation"),
        (True, True, CONFIRMATION, False, "credentials must be configured"),
    ],
)
def test_real_triple_authorization_and_credentials_fail_before_provider_or_reservation(
    tmp_path, authorized, allow, confirmation, credentials_configured, error,
):
    config = _authorized_config(tmp_path) if authorized else FOCUSED_CONFIG
    ledger = _ledger(tmp_path)
    before = ledger.read_bytes()
    factory = _FakeFactory()
    with pytest.raises(PermissionError, match=error):
        _real_runner(
            tmp_path, config, ledger, factory,
            allow=allow, confirmation=confirmation,
            credentials_configured=credentials_configured,
        )
    assert factory.instances == []
    assert ledger.read_bytes() == before
    assert not (tmp_path / "real-results").exists()


@pytest.mark.parametrize("snapshot", [
    {
        "api_attempts": 863,
        "provider_reported_input_tokens": 3_580_832,
        "provider_reported_output_tokens": 84_182,
        "spent_cny": 5.202644120,
    },
    {
        "api_attempts": 864,
        "provider_reported_input_tokens": 3_580_870,
        "provider_reported_output_tokens": 84_187,
        "spent_cny": 5.202693176,
    },
    {
        "api_attempts": 1_207,
        "provider_reported_input_tokens": 4_200_000,
        "provider_reported_output_tokens": 110_000,
        "spent_cny": 8.75,
    },
])
def test_new_real_attempt_captures_any_valid_dynamic_ledger_baseline(tmp_path, snapshot):
    config = _authorized_config(tmp_path)
    ledger = _ledger(tmp_path, **snapshot)
    ledger_hash = hashlib.sha256(ledger.read_bytes()).hexdigest()
    runner = _real_runner(tmp_path, config, ledger, _FakeFactory())
    assert runner.run_ledger_before == {**snapshot, "reserved_cny": 0.0}
    assert runner.run_ledger_before_hash == ledger_hash
    assert runner._manifest()["ledger_baseline"] == {
        **snapshot, "reserved_cny": 0.0, "sha256": ledger_hash,
    }


@pytest.mark.parametrize("payload,error", [
    ({"api_attempts": -1, "provider_reported_input_tokens": 0,
      "provider_reported_output_tokens": 0, "spent_cny": 0, "reserved_cny": 0},
     "non-negative integer"),
    ({"api_attempts": 1, "provider_reported_input_tokens": 0,
      "provider_reported_output_tokens": 0, "reserved_cny": 0},
     "missing required fields"),
    ({"api_attempts": 1, "provider_reported_input_tokens": 0,
      "provider_reported_output_tokens": 0, "spent_cny": 50, "reserved_cny": 0},
     "cumulative hard limit"),
])
def test_new_real_attempt_rejects_invalid_ledger_structure_and_values(tmp_path, payload, error):
    config = _authorized_config(tmp_path)
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises((ValueError, CostHardLimit), match=error):
        _real_runner(tmp_path, config, ledger, _FakeFactory())


def test_new_real_attempt_rejects_malformed_ledger_json(tmp_path):
    config = _authorized_config(tmp_path)
    ledger = tmp_path / "ledger.json"
    ledger.write_text("{not-json", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        _real_runner(tmp_path, config, ledger, _FakeFactory())


def test_resume_rejects_ledger_rollback_and_hash_anomaly(tmp_path):
    config = _authorized_config(tmp_path)
    ledger = _ledger(tmp_path)
    output = tmp_path / "real-results"
    output.mkdir()
    current = json.loads(ledger.read_text(encoding="utf-8"))
    rolled_forward = {**current, "api_attempts": current["api_attempts"] + 1}
    (output / "manifest.json").write_text(json.dumps({
        "ledger_baseline": {**rolled_forward, "sha256": "0" * 64},
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="rolled back"):
        _real_runner(
            tmp_path, config, ledger, _FakeFactory(), output=output,
        )

    (output / "manifest.json").write_text(json.dumps({
        "ledger_baseline": {
            **current, "sha256": "0" * 64,
        },
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="hash differs"):
        _real_runner(
            tmp_path, config, ledger, _FakeFactory(), output=output,
        )


def test_production_ledger_at_current_preflight_state_is_a_valid_dynamic_start(tmp_path):
    before = hashlib.sha256(LEDGER.read_bytes()).hexdigest()
    current = json.loads(LEDGER.read_text(encoding="utf-8"))
    assert current["api_attempts"] == 864
    runner = _real_runner(
        tmp_path, _authorized_config(tmp_path), LEDGER, _FakeFactory(),
        output=tmp_path / "production-baseline-results",
        cache=tmp_path / "production-baseline-cache",
    )
    assert runner.run_ledger_before == {
        "api_attempts": 864,
        "provider_reported_input_tokens": 3_580_870,
        "provider_reported_output_tokens": 84_187,
        "spent_cny": 5.202693176000003,
        "reserved_cny": 0.0,
    }
    assert hashlib.sha256(LEDGER.read_bytes()).hexdigest() == before


def test_dynamic_attempt_cost_resume_and_other_attempt_progress_are_isolated(tmp_path):
    config = _authorized_config(tmp_path)
    ledger = _ledger(
        tmp_path,
        api_attempts=864,
        provider_reported_input_tokens=3_580_870,
        provider_reported_output_tokens=84_187,
        spent_cny=5.202693176,
    )
    output, cache = tmp_path / "dynamic-results", tmp_path / "dynamic-cache"
    real = _real_runner(
        tmp_path, config, ledger, _FakeFactory(), output=output, cache=cache,
    ).run()
    assert real["manifest"]["ledger_baseline"]["api_attempts"] == 864
    assert real["summary"]["api_attempts_delta"] == 8 * 6
    assert real["summary"]["cost_cny_delta"] == pytest.approx(sum(
        item["cost_cny_delta"] for item in real["records"]
    ))

    other_before = json.loads(ledger.read_text(encoding="utf-8"))
    other_cost = 0.01
    other = {
        **other_before,
        "api_attempts": other_before["api_attempts"] + 2,
        "provider_reported_input_tokens": other_before["provider_reported_input_tokens"] + 100,
        "provider_reported_output_tokens": other_before["provider_reported_output_tokens"] + 10,
        "spent_cny": other_before["spent_cny"] + other_cost,
        "attempt": "other-valid-attempt",
        "attempt_base_spent_cny": other_before["spent_cny"],
        "attempt_incremental_spent_cny": other_cost,
    }
    ledger.write_text(json.dumps(other), encoding="utf-8")
    before_resume = hashlib.sha256(ledger.read_bytes()).hexdigest()
    resumed = _real_runner(
        tmp_path, config, ledger, _FakeFactory(), output=output, cache=cache,
    ).run(resume=True)
    assert resumed["summary"]["records_resumed"] == 8
    assert resumed["summary"]["api_attempts_delta"] == 8 * 6
    assert resumed["summary"]["invocation_api_attempts_delta"] == 0
    assert resumed["summary"]["invocation_cost_cny_delta"] == 0
    assert hashlib.sha256(ledger.read_bytes()).hexdigest() == before_resume


@pytest.fixture(scope="module")
def real_path_artifacts(tmp_path_factory):
    root = tmp_path_factory.mktemp("focused-real-path")
    config = _authorized_config(root)
    ledger = _ledger(root)
    output, cache = root / "real-results", root / "real-cache"
    factory = _FakeFactory()
    real = _real_runner(
        root, config, ledger, factory, output=output, cache=cache,
    ).run()
    replay_before = hashlib.sha256(ledger.read_bytes()).hexdigest()
    replay = FocusedLiveHealingRunner(
        config,
        mode="replay-real",
        output=root / "replay-results",
        cache_root=cache,
        attempt_id="offline-fake-attempt",
        ledger_path=ledger,
        git_state_provider=lambda: {"commit": "5983a50", "dirty": True},
    ).run()
    replay_after = hashlib.sha256(ledger.read_bytes()).hexdigest()
    return root, config, ledger, output, cache, factory, real, replay, replay_before, replay_after


def test_fake_provider_runs_exact_eight_real_path_in_provider_order(real_path_artifacts):
    _, _, _, _, _, factory, real, _, _, _ = real_path_artifacts
    assert real["summary"]["records_completed"] == 8
    assert real["summary"]["execution_mode"] == "REAL_PATH_FAKE_PROVIDER_VALIDATION"
    assert real["summary"]["real_api_status"] == "NOT RUN"
    assert [row["provider"] for row in real["records"]] == ["deepseek"] * 4 + ["dashscope"] * 4
    assert factory.calls == 8 * 6
    assert real["summary"]["network_calls"] == 0


def test_fake_real_path_manifest_results_gate_and_fallback_markers(real_path_artifacts):
    _, _, _, output, _, _, real, _, _, _ = real_path_artifacts
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    gate = json.loads((output / "deepseek_gate_report.json").read_text(encoding="utf-8"))
    assert manifest["provider_kind"] == "FAKE_PROVIDER_TEST"
    assert manifest["attempt_id"] == "offline-fake-attempt"
    assert manifest["git"] == {"commit": "5983a50", "dirty": True}
    assert gate["passed"] and gate["qwen_authorized"]
    assert all(len(row["stage_trace"]) == 16 for row in real["records"])
    assert all(row["ledger_after"]["api_attempts"] >= row["ledger_before"]["api_attempts"] for row in real["records"])
    assert all(row["actual_network_attempts"] == 6 for row in real["records"])
    assert all(row["symbolic_fallback_used"] is False for row in real["records"])
    assert all(row["oracle_fallback_used"] is False for row in real["records"])
    assert all(row["mock_fallback_used"] is False for row in real["records"])


def test_fake_real_path_uses_isolated_real_cache_and_persistent_costs(real_path_artifacts):
    _, _, ledger, _, cache, _, real, _, _, _ = real_path_artifacts
    current = json.loads(ledger.read_text(encoding="utf-8"))
    raw = tuple((cache / REAL_CACHE_NAMESPACE / "raw").glob("*.json"))
    assert len(raw) == 8 * 6
    assert real["summary"]["api_attempts_delta"] == 8 * 6
    assert real["summary"]["input_tokens_delta"] > 0
    assert real["summary"]["output_tokens_delta"] > 0
    assert 0 < real["summary"]["cost_cny_delta"] < 3
    assert current["api_attempts"] == TEST_LEDGER_BASELINE["api_attempts"] + 8 * 6


def test_replay_real_is_cache_only_and_has_zero_usage_or_ledger_change(real_path_artifacts):
    _, _, _, _, _, _, _, replay, before, after = real_path_artifacts
    summary = replay["summary"]
    assert summary["records_completed"] == 8
    assert summary["execution_mode"] == "REAL_CACHE_REPLAY"
    assert summary["real_provider_instances"] == 0
    assert summary["network_calls"] == 0
    assert summary["api_attempts_delta"] == 0
    assert summary["input_tokens_delta"] == summary["output_tokens_delta"] == 0
    assert summary["cost_cny_delta"] == 0
    assert summary["provider_fallback_calls"] == 0
    assert before == after


def test_resume_does_not_repeat_provider_calls_or_billing(real_path_artifacts):
    root, config, ledger, output, cache, _, _, _, _, _ = real_path_artifacts
    before = hashlib.sha256(ledger.read_bytes()).hexdigest()
    factory = _FakeFactory()
    resumed = _real_runner(
        root, config, ledger, factory, output=output, cache=cache,
    ).run(resume=True)
    assert resumed["summary"]["records_resumed"] == 8
    assert resumed["summary"]["api_attempts_delta"] == 8 * 6
    assert resumed["summary"]["invocation_api_attempts_delta"] == 0
    assert resumed["summary"]["invocation_input_tokens_delta"] == 0
    assert resumed["summary"]["invocation_output_tokens_delta"] == 0
    assert resumed["summary"]["invocation_cost_cny_delta"] == 0
    assert factory.instances == []
    assert hashlib.sha256(ledger.read_bytes()).hexdigest() == before


def test_two_deepseek_infrastructure_errors_block_qwen(tmp_path):
    config, ledger = _authorized_config(tmp_path), _ledger(tmp_path)
    factory = _FakeFactory(fail_providers={"deepseek"})
    result = _real_runner(tmp_path, config, ledger, factory).run()
    assert result["summary"]["run_status"] == "BLOCKED_AFTER_DEEPSEEK"
    assert result["summary"]["records_completed"] == 4
    assert result["summary"]["infrastructure_errors"] == 4
    assert all(row["provider"] == "deepseek" for row in result["records"])


def test_structured_provider_error_reaches_focused_record_schema(tmp_path):
    config, ledger = _authorized_config(tmp_path), _ledger(tmp_path)
    factory = _FakeFactory(fail_calls={"deepseek"})
    result = _real_runner(tmp_path, config, ledger, factory).run()
    assert result["summary"]["run_status"] == "BLOCKED_AFTER_DEEPSEEK"
    for record in result["records"]:
        assert len(record["provider_errors"]) == 2
        assert record["actual_network_attempts"] == 2
        assert all(error["category"] == "PROVIDER_ERROR" for error in record["provider_errors"])
        assert all(error["status_code"] == 400 for error in record["provider_errors"])
        assert all(error["provider_error_code"] == "FAKE_BAD_REQUEST" for error in record["provider_errors"])
        assert all(error["request_id"] == "fake-request-id" for error in record["provider_errors"])
        assert all(error["retryable"] is False for error in record["provider_errors"])


def test_replay_real_cache_miss_fails_closed_without_provider_or_ledger_change(tmp_path):
    config, ledger = _authorized_config(tmp_path), _ledger(tmp_path)
    cache = tmp_path / "empty-real-cache"
    _real_runner(tmp_path, config, ledger, _FakeFactory(), cache=cache)
    before = hashlib.sha256(ledger.read_bytes()).hexdigest()
    runner = FocusedLiveHealingRunner(
        config,
        mode="replay-real",
        output=tmp_path / "empty-replay",
        cache_root=cache,
        attempt_id="offline-fake-attempt",
        ledger_path=ledger,
        git_state_provider=lambda: {"commit": "5983a50", "dirty": True},
    )
    with pytest.raises(ReplayCacheMiss):
        runner.run()
    assert runner.real_provider_instances == runner.network_calls == 0
    assert hashlib.sha256(ledger.read_bytes()).hexdigest() == before


def test_replay_real_rejects_mock_marked_cache(tmp_path):
    config, ledger = _authorized_config(tmp_path), _ledger(tmp_path)
    cache = tmp_path / "marked-mock" / REAL_CACHE_NAMESPACE
    (cache / "raw").mkdir(parents=True)
    (cache / "parsed").mkdir()
    (cache / "focused_cache_identity.json").write_text(
        json.dumps({"cache_kind": "MOCK_PROVIDER"}), encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Mock or unmarked Cache"):
        FocusedLiveHealingRunner(
            config,
            mode="replay-real",
            output=tmp_path / "replay",
            cache_root=tmp_path / "marked-mock",
            attempt_id="offline-fake-attempt",
            ledger_path=ledger,
            git_state_provider=lambda: {"commit": "5983a50", "dirty": True},
        )


def test_atomic_focused_ledger_reserve_release_and_commit(tmp_path):
    config_path, ledger_path = _authorized_config(tmp_path), _ledger(tmp_path)
    config = FocusedLiveHealingConfig.load(config_path)
    baseline = json.loads(ledger_path.read_text(encoding="utf-8"))
    costs = AtomicFocusedLedger(ledger_path, config, "ledger-test", baseline)
    model = config.models[0]
    request = ProviderRequest((), {}, "p", "public-M01", 1, "driftguard_llm", 0)
    first = costs.reserve(model, request)
    assert costs.snapshot()["reserved_cny"] > 0
    costs.release(first)
    assert costs.snapshot()["reserved_cny"] == 0
    second = costs.reserve(model, request)
    response = ProviderResponse("fake", model.model_id, {}, "{}", 10, 5, 15, 0, 1, "stop")
    costs.settle(second, response)
    snapshot = costs.snapshot()
    assert snapshot["api_attempts"] == TEST_LEDGER_BASELINE["api_attempts"] + 2
    assert snapshot["provider_reported_input_tokens"] == TEST_LEDGER_BASELINE["provider_reported_input_tokens"] + 10
    assert snapshot["reserved_cny"] == 0 and snapshot["spent_cny"] > TEST_LEDGER_BASELINE["spent_cny"]


def test_api_key_value_and_hidden_fallback_material_absent(real_path_artifacts):
    _, _, _, output, cache, _, real, _, _, _ = real_path_artifacts
    encoded = "".join(
        path.read_text(encoding="utf-8")
        for root in (output, cache)
        for path in root.rglob("*.json")
    )
    assert "FAKE-SECRET-DO-NOT-STORE" not in encoded
    lowered = encoded.lower()
    for forbidden in ("ground_truth", "source_drift_id", "expected_patch", "evaluator_view"):
        assert forbidden not in lowered
    assert real["summary"]["symbolic_fallback_count"] == 0
    assert real["summary"]["oracle_fallback_count"] == 0
    assert real["summary"]["mock_fallback_count"] == 0


def test_formal_ledger_unchanged_by_all_fake_real_path_tests():
    assert hashlib.sha256(LEDGER.read_bytes()).hexdigest() == FORMAL_LEDGER_HASH_AT_IMPORT


def test_production_runner_has_no_fixed_historical_ledger_pass_condition():
    source = (
        Path(__file__).resolve().parents[1]
        / "src/driftguard/runners/focused_live_healing_runner.py"
    ).read_text(encoding="utf-8")
    for historical in ("831", "863", "864"):
        assert historical not in source
    assert "frozen ledger baseline" not in source.lower()
