from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import pytest
import yaml

from driftguard.contracts.loader import PROJECT_ROOT
from driftguard.llm import ProviderError, ProviderRequest, ProviderResponse
from driftguard.phase10.pricing import PricingCatalog
from driftguard.runners.focused_live_healing_runner import AtomicFocusedLedger
from driftguard.specdriftbench import heldout
from driftguard.specdriftbench.heldout import (
    ANALYSIS_PLAN_PATH, CONFIG_PATH, CONFIG_V2_PATH, CONFIRM_432, CONFIRM_432_V2,
    DEVELOPMENT_FAMILIES,
    EVIDENCE_VIEWS, FAKE_CACHE_NAMESPACE, FROZEN_FINGERPRINTS,
    HELDOUT_FAMILIES, PROVIDER_ORDER, REAL_CACHE_NAMESPACE, VARIANTS,
    HeldoutCacheMiss, HeldoutConfig, HeldoutEvidenceViewBuilder, HeldoutFakeProvider,
    HeldoutProtocolError, HeldoutRunner, assert_heldout_provider_visible,
    run_offline_fake_validation, validate_frozen_fingerprints,
    validate_heldout_plan, validate_paired_evidence, _HeldoutLedgerConfigAdapter,
)
from driftguard.specdriftbench.heldout_gate import (
    InfrastructureGateV2, classify_gate_record, descriptive_availability_metrics,
)
from driftguard.specdriftbench.runner import PRICING_PATH
from driftguard.specdriftbench.protocol import TrueEvidenceLeakage


def _ledger(path: Path, **overrides: object) -> Path:
    value = {
        "api_attempts": 10,
        "provider_reported_input_tokens": 100,
        "provider_reported_output_tokens": 20,
        "spent_cny": 1.0,
        "reserved_cny": 0.0,
        "soft_warning": False,
    }
    value.update(overrides)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _config_copy(path: Path, mutate) -> Path:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    mutate(raw)
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def _v2_config_copy(path: Path, mutate) -> Path:
    raw = yaml.safe_load(CONFIG_V2_PATH.read_text(encoding="utf-8"))
    mutate(raw)
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def _gate_record(
    *, status=None, message="transient", error_class="PROVIDER_ERROR",
    predicted=True, correct=True,
):
    if status == "success":
        return {
            "error": None, "error_class": "NONE", "normalized_prediction": {"class": "AE"},
            "raw_response_ref": "cache/raw/success.json", "schema_valid": True,
            "evaluation": {"class_correct": correct}, "leakage": {"passed": True},
            "fallback_used": False, "main_state_pollution": False,
        }
    return {
        "error": {
            "status_code": status, "provider_error_code": None,
            "sanitized_message": message, "failure_layer": "UNKNOWN",
        },
        "error_class": error_class,
        "normalized_prediction": {"class": "AE"} if predicted else None,
        "raw_response_ref": None,
        "schema_valid": False,
        "evaluation": {"class_correct": correct if predicted else None},
        "leakage": {"passed": True}, "fallback_used": False,
        "main_state_pollution": False,
    }


@pytest.fixture(scope="module")
def frozen_config() -> HeldoutConfig:
    return HeldoutConfig.load()


@pytest.fixture(scope="module")
def fake_artifacts(tmp_path_factory):
    root = tmp_path_factory.mktemp("heldout432-fake") / "attempt"
    ledger = _ledger(root.parent / "ledger.json")
    before = hashlib.sha256(ledger.read_bytes()).hexdigest()
    report = run_offline_fake_validation(
        root, attempt_id="heldout432-offline-test", ledger_path=ledger,
    )
    after = hashlib.sha256(ledger.read_bytes()).hexdigest()
    return root, ledger, before, after, report


def test_exact_432_and_all_balance_constraints(frozen_config):
    plan = frozen_config.plan
    assert len(plan) == 432
    assert Counter(row.provider for row in plan) == {name: 144 for name in PROVIDER_ORDER}
    assert Counter(row.evidence_view for row in plan) == {name: 144 for name in EVIDENCE_VIEWS}
    assert Counter(row.variant for row in plan) == {name: 144 for name in VARIANTS}
    assert Counter(row.family for row in plan) == {name: 27 for name in HELDOUT_FAMILIES}
    assert len(set(row.family for row in plan)) == 16
    assert not set(DEVELOPMENT_FAMILIES) & {row.family for row in plan}
    assert frozen_config.counts["heldout_families"] == 16


@pytest.mark.parametrize("delta", [-1, 1])
def test_431_and_433_records_are_rejected(frozen_config, delta):
    plan = list(frozen_config.plan)
    if delta < 0:
        plan.pop()
    else:
        plan.append(replace(plan[-1], ordinal=433))
    with pytest.raises(HeldoutProtocolError, match="exactly 432"):
        validate_heldout_plan(plan)


@pytest.mark.parametrize("family", DEVELOPMENT_FAMILIES)
def test_each_development_family_leak_is_rejected(frozen_config, family):
    plan = list(frozen_config.plan)
    plan[0] = replace(plan[0], family=family)
    with pytest.raises(HeldoutProtocolError):
        validate_heldout_plan(plan)


def test_provider_view_and_variant_cross_balance(frozen_config):
    for provider in PROVIDER_ORDER:
        rows = [row for row in frozen_config.plan if row.provider == provider]
        assert Counter(row.evidence_view for row in rows) == {view: 48 for view in EVIDENCE_VIEWS}
        assert Counter(row.variant for row in rows) == {variant: 48 for variant in VARIANTS}


def test_exactly_144_paired_groups_with_three_views(frozen_config):
    pairs = defaultdict(list)
    for row in frozen_config.plan:
        pairs[row.paired_group_id].append(row)
    assert len(pairs) == 144
    for rows in pairs.values():
        assert len(rows) == 3
        assert Counter(row.evidence_view for row in rows) == {view: 1 for view in EVIDENCE_VIEWS}
        assert len({(row.provider, row.model, row.family, row.variant, row.repetition) for row in rows}) == 1


@pytest.mark.parametrize("mutation", ["missing_view", "duplicate_view", "bad_repetition"])
def test_invalid_pair_or_repetition_is_rejected(frozen_config, mutation):
    plan = list(frozen_config.plan)
    if mutation == "missing_view":
        plan[0] = replace(plan[0], evidence_view="NOT_A_VIEW")
    elif mutation == "duplicate_view":
        plan[0] = replace(plan[0], evidence_view=plan[1].evidence_view)
    else:
        plan[0] = replace(plan[0], repetition=2)
    with pytest.raises(HeldoutProtocolError):
        validate_heldout_plan(plan)


@pytest.mark.parametrize("mutation", ["missing_provider", "duplicate_provider"])
def test_missing_or_duplicate_provider_is_rejected(frozen_config, mutation):
    plan = list(frozen_config.plan)
    if mutation == "missing_provider":
        plan = [row for row in plan if row.provider != "moonshot"]
    else:
        plan = [
            replace(row, provider="deepseek", model=frozen_config.model("deepseek").model_id)
            if row.provider == "moonshot" else row
            for row in plan
        ]
    with pytest.raises(HeldoutProtocolError):
        validate_heldout_plan(plan)


def test_config_freezes_models_controls_order_namespaces_and_budget(frozen_config):
    assert [(model.provider, model.model_id) for model in frozen_config.models] == [
        ("deepseek", "deepseek-v4-flash"),
        ("dashscope", "qwen3.7-plus"),
        ("moonshot", "kimi-k2.6"),
    ]
    assert frozen_config.models[2].thinking_mode == "disabled"
    assert frozen_config.raw["controls"] == {
        "sandbox_enabled": False, "tool_calling_enabled": False,
        "patch_enabled": False, "repair_enabled": False,
        "future_transfer_enabled": False,
    }
    assert frozen_config.raw["execution"]["provider_order"] == list(PROVIDER_ORDER)
    assert frozen_config.raw["execution"]["cache_namespace"] == REAL_CACHE_NAMESPACE
    assert frozen_config.raw["execution"]["fake_cache_namespace"] == FAKE_CACHE_NAMESPACE
    assert frozen_config.raw["budget"]["attempt_soft_limit_cny"] == 15.0
    assert frozen_config.raw["budget"]["attempt_hard_limit_cny"] == 20.0
    assert frozen_config.raw["budget"]["global_hard_limit_cny"] == 50.0
    assert frozen_config.raw["budget"]["max_format_repairs"] == 1


@pytest.mark.parametrize("namespace", [
    "specdriftbench_component_canary_real_v1",
    "phase9_mock_cache",
    "phase10_end_to_end_cache",
])
def test_prohibited_cache_namespace_is_rejected(tmp_path, namespace):
    config = _config_copy(
        tmp_path / f"{namespace}.yaml",
        lambda raw: raw["execution"].__setitem__("cache_namespace", namespace),
    )
    with pytest.raises(HeldoutProtocolError):
        HeldoutConfig.load(config)


@pytest.mark.parametrize("fingerprint", [
    "prompt_sha256", "attribution_schema_sha256",
    "evidence_view_sha256", "tool_registry_sha256",
])
def test_each_frozen_fingerprint_change_is_rejected(monkeypatch, fingerprint):
    actual = dict(FROZEN_FINGERPRINTS)
    actual[fingerprint] = "0" * 64
    monkeypatch.setattr(heldout, "_actual_frozen_fingerprints", lambda: actual)
    with pytest.raises(HeldoutProtocolError, match="canonical assets changed"):
        validate_frozen_fingerprints(FROZEN_FINGERPRINTS)


@pytest.mark.parametrize("hidden", [
    {"ground_truth_label": "PERSISTENT_DRIFT"},
    {"variant": "PD"},
    {"family_id": "M02"},
    {"future_episode_evidence": []},
    {"canary_prediction": "AGENT_ERROR"},
    {"evaluator_metadata": {}},
])
def test_provider_input_rejects_ground_truth_and_hidden_metadata(hidden):
    with pytest.raises(TrueEvidenceLeakage):
        assert_heldout_provider_visible(hidden)


def test_provider_input_rejects_family_identifier_in_text():
    with pytest.raises(TrueEvidenceLeakage):
        assert_heldout_provider_visible("compare against family M02")


def test_paired_evidence_is_monotone_and_context_is_stable():
    builder = HeldoutEvidenceViewBuilder()
    paired = {view: builder.build("M02", "PD", view) for view in EVIDENCE_VIEWS}
    validate_paired_evidence(paired)
    assert len({item["public_scenario_id"] for item in paired.values()}) == 1
    assert len({json.dumps(item["displayed_specification"], sort_keys=True) for item in paired.values()}) == 1


def test_cross_scenario_history_is_rejected():
    builder = HeldoutEvidenceViewBuilder()
    paired = {view: builder.build("M02", "PD", view) for view in EVIDENCE_VIEWS}
    paired["FULL_EVIDENCE"]["evidence_trace"]["trace_id"] = "public-cross-scenario"
    with pytest.raises(HeldoutProtocolError, match="cross-scenario"):
        validate_paired_evidence(paired)


def test_full_evidence_polluting_first_failure_is_rejected():
    builder = HeldoutEvidenceViewBuilder()
    paired = {view: builder.build("M02", "PD", view) for view in EVIDENCE_VIEWS}
    paired["FIRST_FAILURE"]["evidence_trace"]["events"] = list(
        paired["FULL_EVIDENCE"]["evidence_trace"]["events"]
    )
    with pytest.raises(HeldoutProtocolError, match="strict monotone"):
        validate_paired_evidence(paired)


def test_real_mode_rejects_default_unauthorized_config(tmp_path):
    with pytest.raises(PermissionError, match="run_authorized:true"):
        HeldoutRunner(
            attempt_id="real-denied", mode="real", output_root=tmp_path / "real",
            ledger_path=_ledger(tmp_path / "ledger.json"),
            allow_real_api=True, confirmation=CONFIRM_432,
        )


@pytest.mark.parametrize("allow,token", [
    (False, CONFIRM_432),
    (True, "RUN-EXACTLY-36"),
    (True, "WRONG"),
])
def test_real_mode_rejects_missing_flag_or_wrong_confirmation(tmp_path, allow, token):
    config = _config_copy(
        tmp_path / f"authorized-{allow}-{token}.yaml",
        lambda raw: raw.__setitem__("run_authorized", True),
    )
    with pytest.raises(PermissionError, match="RUN-EXACTLY-432"):
        HeldoutRunner(
            config, attempt_id="real-denied", mode="real",
            output_root=tmp_path / "real", ledger_path=_ledger(tmp_path / "ledger.json"),
            allow_real_api=allow, confirmation=token,
        )


def test_outstanding_reservation_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="outstanding reservation"):
        HeldoutRunner(
            attempt_id="fake-reserved", mode="fake", output_root=tmp_path / "fake",
            ledger_path=_ledger(tmp_path / "ledger.json", reserved_cny=0.25),
        )


def test_fake_provider_completes_432_without_network_cost_or_ledger_change(fake_artifacts):
    root, ledger, before, after, report = fake_artifacts
    summary = report["summary"]
    assert report["protocol_status"] == "FAKE_PROVIDER_TEST"
    assert report["validation_status"] == "OFFLINE_PROTOCOL_VALIDATION"
    assert summary["status"] == "COMPLETE" and summary["records_completed"] == 432
    assert summary["provider_instances"] == 3
    assert summary["real_provider_instances"] == summary["network_calls"] == 0
    assert summary["attempts_delta"] == summary["input_tokens_delta"] == 0
    assert summary["output_tokens_delta"] == 0 and summary["cost_cny_delta"] == 0
    assert before == after == hashlib.sha256(ledger.read_bytes()).hexdigest()
    assert len(list((root / "results/records").glob("*.json"))) == 432
    assert not (PROJECT_ROOT / "results/experiments/phase11/specdriftbench_component_heldout432/attempts/heldout432-offline-test").exists()


def test_resume_uses_checkpoints_without_provider_calls_or_ledger_change(fake_artifacts, tmp_path):
    original, _, _, _, _ = fake_artifacts
    root = tmp_path / "resume-attempt"
    shutil.copytree(original, root)
    ledger = _ledger(tmp_path / "ledger.json")
    before = hashlib.sha256(ledger.read_bytes()).hexdigest()

    class MustNotRun:
        def complete(self, request):
            raise AssertionError("resume repeated a completed Provider call")

    runner = HeldoutRunner(
        attempt_id="heldout432-offline-test", mode="fake", output_root=root,
        ledger_path=ledger, provider_factory=lambda model: MustNotRun(),
    )
    report = runner.run(resume=True)
    assert report["summary"]["records_completed"] == 432
    assert hashlib.sha256(ledger.read_bytes()).hexdigest() == before


def test_heldout_ledger_resume_preserves_settled_cost_without_duplicate_charge(frozen_config, tmp_path):
    ledger = _ledger(tmp_path / "ledger.json")
    baseline = json.loads(ledger.read_text(encoding="utf-8"))
    pricing = PricingCatalog(json.loads(PRICING_PATH.read_text(encoding="utf-8")))
    manager = AtomicFocusedLedger(
        ledger, _HeldoutLedgerConfigAdapter(frozen_config), "heldout-ledger-test",
        baseline, pricing_catalog=pricing,
    )
    model = frozen_config.model("deepseek")
    request = ProviderRequest(
        ({"role": "user", "content": "visible held-out evidence"},), {}, "0" * 64,
        "public-a1", 5, "component_attribution", 1,
    )
    reservation = manager.reserve(model, request)
    assert json.loads(ledger.read_text(encoding="utf-8"))["reserved_cny"] > 0
    manager.settle(
        reservation,
        ProviderResponse("fake", model.model_id, None, "{}", 10, 2, 12, 1.0, 1, "stop"),
    )
    settled = manager.snapshot()
    assert settled["reserved_cny"] == 0 and settled["api_attempts"] == 11
    before_resume = hashlib.sha256(ledger.read_bytes()).hexdigest()
    resumed = AtomicFocusedLedger(
        ledger, _HeldoutLedgerConfigAdapter(frozen_config), "heldout-ledger-test",
        baseline, settled["spent_cny"] - baseline["spent_cny"], pricing_catalog=pricing,
    )
    assert resumed.snapshot()["api_attempts"] == 11
    assert hashlib.sha256(ledger.read_bytes()).hexdigest() == before_resume


def test_replay_is_cache_only_and_semantically_exact(fake_artifacts, tmp_path):
    original, _, _, _, _ = fake_artifacts
    root = tmp_path / "replay-attempt"
    shutil.copytree(original, root)
    ledger = _ledger(tmp_path / "ledger.json")
    before = hashlib.sha256(ledger.read_bytes()).hexdigest()
    report = HeldoutRunner(
        attempt_id="heldout432-offline-test", mode="replay",
        output_root=root, ledger_path=ledger,
    ).run()
    summary = report["summary"]
    assert summary["records_completed"] == 432 and summary["cache_misses"] == 0
    assert summary["provider_instances"] == summary["real_provider_instances"] == 0
    assert summary["network_calls"] == summary["attempts_delta"] == 0
    assert summary["input_tokens_delta"] == summary["output_tokens_delta"] == 0
    assert summary["cost_cny_delta"] == 0 and summary["fallback_count"] == 0
    assert hashlib.sha256(ledger.read_bytes()).hexdigest() == before


def test_replay_cache_miss_fails_closed(fake_artifacts, tmp_path):
    original, _, _, _, _ = fake_artifacts
    root = tmp_path / "missing-cache"
    shutil.copytree(original, root)
    entry = next(path for path in (root / "fake_cache").rglob("*.json") if path.name != "identity.json")
    entry.unlink()
    with pytest.raises(HeldoutCacheMiss, match="Cache miss"):
        HeldoutRunner(
            attempt_id="heldout432-offline-test", mode="replay",
            output_root=root, ledger_path=_ledger(tmp_path / "ledger.json"),
        ).run()


def test_replay_rejects_original_provider_fallback(fake_artifacts, tmp_path):
    original, _, _, _, _ = fake_artifacts
    root = tmp_path / "fallback"
    shutil.copytree(original, root)
    record_path = next((root / "results/records").glob("*.json"))
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["fallback_used"] = True
    record_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(HeldoutProtocolError, match="fallback"):
        HeldoutRunner(
            attempt_id="heldout432-offline-test", mode="replay",
            output_root=root, ledger_path=_ledger(tmp_path / "ledger.json"),
        ).run()


def test_replay_rejects_ledger_rollback(fake_artifacts, tmp_path):
    original, _, _, _, _ = fake_artifacts
    root = tmp_path / "rollback"
    shutil.copytree(original, root)
    with pytest.raises(HeldoutProtocolError, match="rolled back"):
        HeldoutRunner(
            attempt_id="heldout432-offline-test", mode="replay",
            output_root=root,
            ledger_path=_ledger(tmp_path / "ledger.json", api_attempts=9),
        )


def test_env_is_untracked_and_api_key_values_do_not_enter_fake_results(fake_artifacts, monkeypatch):
    sentinel = "test-only-secret-value-that-must-not-persist"
    for name in ("DEEPSEEK_API_KEY", "DASHSCOPE_API_KEY", "MOONSHOT_API_KEY"):
        monkeypatch.setenv(name, sentinel)
    model = HeldoutConfig.load().model("deepseek")
    response = HeldoutFakeProvider(model).complete(ProviderRequest(
        ({"role": "user", "content": "visible evidence only"},), {}, "0" * 64,
        "public-a1", 5, "component_attribution", 1,
    ))
    assert sentinel not in json.dumps(response.public_dict())
    root, _, _, _, _ = fake_artifacts
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", ".env"], cwd=PROJECT_ROOT,
        capture_output=True, text=True,
    )
    assert tracked.returncode != 0
    for path in (root / "results").rglob("*.json"):
        assert sentinel not in path.read_text(encoding="utf-8")


def test_analysis_plan_is_frozen_before_results():
    plan = json.loads(ANALYSIS_PLAN_PATH.read_text(encoding="utf-8"))
    assert plan["status"] == "FROZEN_BEFORE_REAL_RESULTS"
    assert plan["bootstrap"] == {
        "unit": "family", "repetitions": 10000, "seed": 20260718,
        "confidence_level": 0.95, "method": "family_clustered_bootstrap",
    }
    assert plan["paired_design"]["comparisons"] == [
        "RETRY_HISTORY - FIRST_FAILURE",
        "FULL_EVIDENCE - FIRST_FAILURE",
        "FULL_EVIDENCE - RETRY_HISTORY",
    ]


def test_v2_config_preserves_432_design_models_and_fingerprints():
    v1, v2 = HeldoutConfig.load(), HeldoutConfig.load(CONFIG_V2_PATH)
    assert v2.infrastructure_gate_version == 2
    assert len(v2.plan) == v2.counts["records"] == 432
    assert v2.counts["paired_groups"] == 144
    assert v2.raw["models"] == v1.raw["models"]
    assert v2.raw["controls"] == v1.raw["controls"]
    assert v2.raw["budget"] == v1.raw["budget"]
    assert v2.raw["development_families"] == v1.raw["development_families"]
    assert v2.raw["heldout_families"] == v1.raw["heldout_families"]
    assert v2.raw["variants"] == v1.raw["variants"]
    assert v2.raw["evidence_views"] == v1.raw["evidence_views"]
    assert v2.raw["frozen_fingerprints"] == v1.raw["frozen_fingerprints"]
    assert v2.raw["analysis_plan"] == v1.raw["analysis_plan"]
    assert not set(DEVELOPMENT_FAMILIES) & {row.family for row in v2.plan}


def test_v2_gate_two_of_63_does_not_stop():
    records = [_gate_record(status=None, predicted=False) if index in {10, 50} else _gate_record(status="success") for index in range(63)]
    gate = InfrastructureGateV2.from_records("deepseek", records)
    assert gate.completed == 63 and gate.final_infrastructure_errors == 2
    assert gate.stop is False


def test_v2_gate_two_of_24_does_not_stop_but_three_of_24_does():
    two = [_gate_record(status=None, predicted=False) if index in {2, 18} else _gate_record(status="success") for index in range(24)]
    gate = InfrastructureGateV2.from_records("deepseek", two)
    assert gate.infrastructure_error_rate == pytest.approx(2 / 24)
    assert gate.stop is False
    three = [_gate_record(status=None, predicted=False) if index in {2, 12, 23} else _gate_record(status="success") for index in range(24)]
    gate = InfrastructureGateV2.from_records("deepseek", three)
    assert gate.stop is True
    assert gate.stop_reason == "FINAL_INFRASTRUCTURE_ERROR_RATE_AT_LEAST_10_PERCENT"


def test_v2_gate_three_consecutive_stops_and_success_resets_counter():
    gate = InfrastructureGateV2("deepseek")
    gate.observe(_gate_record(status=None, predicted=False))
    gate.observe(_gate_record(status=None, predicted=False))
    gate.observe(_gate_record(status="success"))
    assert gate.consecutive_final_infrastructure_errors == 0 and gate.stop is False
    gate.observe(_gate_record(status=None, predicted=False))
    gate.observe(_gate_record(status=None, predicted=False))
    state = gate.observe(_gate_record(status=None, predicted=False))
    assert state["stopped"] is True
    assert state["stop_reason"] == "THREE_CONSECUTIVE_FINAL_INFRASTRUCTURE_ERRORS"


@pytest.mark.parametrize("status,message", [
    (401, "unauthorized"), (402, "payment required"),
    (400, "invalid api key"), (400, "insufficient balance"),
    (403, "permission denied"), (404, "model not found"), (None, "invalid endpoint"),
])
def test_v2_systemic_permanent_provider_errors_stop_immediately(status, message):
    gate = InfrastructureGateV2("deepseek")
    state = gate.observe(_gate_record(status=status, message=message, predicted=False))
    assert state["stopped"] is True
    assert state["stop_reason"] == "IMMEDIATE_PERMANENT_PROVIDER_ERROR"


@pytest.mark.parametrize("status", [400, 413, 415, 422])
def test_v2_single_record_request_errors_are_retained_and_continue(status):
    record = _gate_record(status=status, message="record request rejected", predicted=False)
    assert classify_gate_record(record) == "RECORD_REQUEST_ERROR"
    gate = InfrastructureGateV2("deepseek")
    gate.observe(record)
    assert gate.completed == 1 and gate.final_infrastructure_errors == 0
    assert gate.stop is False


def test_non_evaluable_record_uses_all_record_denominator_and_reduces_coverage():
    records = [
        _gate_record(status="success", correct=True),
        _gate_record(status="success", correct=False),
        _gate_record(status=None, predicted=False),
    ]
    metrics = descriptive_availability_metrics(records)
    assert metrics["all_record_accuracy"] == pytest.approx(1 / 3)
    assert metrics["evaluable_accuracy"] == pytest.approx(1 / 2)
    assert metrics["coverage"] == pytest.approx(2 / 3)
    assert metrics["infrastructure_error_rate"] == pytest.approx(1 / 3)


def test_v1_and_v2_confirmation_tokens_are_isolated(tmp_path):
    v1 = _config_copy(tmp_path / "v1-authorized.yaml", lambda raw: raw.__setitem__("run_authorized", True))
    v2 = _v2_config_copy(tmp_path / "v2-authorized.yaml", lambda raw: raw.__setitem__("run_authorized", True))
    with pytest.raises(PermissionError, match="RUN-EXACTLY-432"):
        HeldoutRunner(
            v1, attempt_id="v1-token-isolation", mode="real", output_root=tmp_path / "v1",
            ledger_path=_ledger(tmp_path / "v1-ledger.json"), allow_real_api=True,
            confirmation=CONFIRM_432_V2,
        )
    with pytest.raises(PermissionError, match="RUN-EXACTLY-432-V2"):
        HeldoutRunner(
            v2, attempt_id="v2-token-isolation", mode="real", output_root=tmp_path / "v2",
            ledger_path=_ledger(tmp_path / "v2-ledger.json"), allow_real_api=True,
            confirmation=CONFIRM_432,
        )


def test_v2_rejects_incomplete_v1_attempt_and_cache_namespace(tmp_path):
    with pytest.raises(HeldoutProtocolError, match="excluded incomplete V1"):
        HeldoutRunner(
            CONFIG_V2_PATH, attempt_id="specdriftbench-heldout432-20260718-af3db03-01",
            mode="fake", output_root=tmp_path / "excluded",
            ledger_path=_ledger(tmp_path / "ledger.json"),
        )
    bad = _v2_config_copy(
        tmp_path / "bad-cache.yaml",
        lambda raw: raw["execution"].__setitem__("cache_namespace", REAL_CACHE_NAMESPACE),
    )
    with pytest.raises(HeldoutProtocolError, match="Cache namespace"):
        HeldoutConfig.load(bad)


def test_v2_failed_record_checkpoint_gate_resume_and_observability(tmp_path):
    root = tmp_path / "v2-attempt"
    ledger = _ledger(tmp_path / "ledger.json")
    before = hashlib.sha256(ledger.read_bytes()).hexdigest()

    class OneFailureProvider(HeldoutFakeProvider):
        def complete(self, request):
            if self.model.provider == "deepseek" and self.calls == 0:
                self.calls += 1
                raise ProviderError(
                    "offline protocol error", retryable=False,
                    attempt_number=1, failure_layer="UNKNOWN",
                    exception_type="OfflineInjectedHTTPError", latency_ms=1.25,
                )
            return super().complete(request)

    report = HeldoutRunner(
        CONFIG_V2_PATH, attempt_id="heldout432-v2-offline-test", mode="fake",
        output_root=root, ledger_path=ledger,
        provider_factory=lambda model: OneFailureProvider(model),
    ).run()
    summary = report["summary"]
    assert summary["status"] == "COMPLETE" and summary["records_completed"] == 432
    assert summary["network_calls"] == summary["attempts_delta"] == 0
    assert hashlib.sha256(ledger.read_bytes()).hexdigest() == before
    failed = next(row for row in report["records"] if row["error"] is not None)
    assert failed["evaluation"]["class_correct"] is None
    assert failed["logical_boundary_calls"] == failed["actual_network_attempts"] == 1
    assert failed["provider_retry_count"] == failed["format_repair_count"] == 0
    assert failed["usage_if_available"] is None and failed["latency_ms"] == pytest.approx(1.25)
    checkpoint = root / "results/checkpoint/deepseek" / f"{failed['record_id']}.json"
    assert checkpoint.exists()
    gate = json.loads((root / "results/provider_gates/deepseek.json").read_text())
    assert gate["infrastructure_gate_version"] == 2
    assert gate["final_infrastructure_errors"] == 1 and gate["stopped"] is False
    manifest = json.loads((root / "results/manifest.json").read_text())
    assert manifest["excluded_attempts"] == ["specdriftbench-heldout432-20260718-af3db03-01"]

    class MustNotRun:
        def complete(self, request):
            raise AssertionError("V2 resume repeated a completed Provider call")

    resumed = HeldoutRunner(
        CONFIG_V2_PATH, attempt_id="heldout432-v2-offline-test", mode="fake",
        output_root=root, ledger_path=ledger,
        provider_factory=lambda model: MustNotRun(),
    ).run(resume=True)
    assert resumed["summary"]["records_completed"] == 432
    assert hashlib.sha256(ledger.read_bytes()).hexdigest() == before
