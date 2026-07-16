from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from driftguard.live.focused_config import FOCUSED_CONFIG, FocusedLiveHealingConfig
from driftguard.live.provider_adapter import FocusedProviderAdapter, ReplayCacheMiss
from driftguard.phase10.config import Phase10Config
from driftguard.runners.focused_live_healing_runner import (
    LEDGER, FocusedLiveHealingRunner,
)

TEST_LEDGER_BASELINE = {
    "api_attempts": 863,
    "provider_reported_input_tokens": 3_580_832,
    "provider_reported_output_tokens": 84_182,
    "spent_cny": 5.202644120,
}


def _temporary_ledger(root: Path) -> Path:
    path = root / "ledger.json"
    path.write_text(json.dumps({
        **TEST_LEDGER_BASELINE, "reserved_cny": 0.0, "soft_warning": False,
    }), encoding="utf-8")
    return path


@pytest.fixture(scope="module", autouse=True)
def formal_ledger_is_read_only():
    before = hashlib.sha256(LEDGER.read_bytes()).hexdigest()
    yield
    assert hashlib.sha256(LEDGER.read_bytes()).hexdigest() == before


@pytest.fixture(scope="module")
def focused_artifacts(tmp_path_factory):
    root = tmp_path_factory.mktemp("focused-runner")
    cache = root / "cache"
    ledger = _temporary_ledger(root)
    mock = FocusedLiveHealingRunner(
        FOCUSED_CONFIG, mode="mock", output=root / "mock", cache_root=cache,
        ledger_path=ledger,
    ).run()
    replay = FocusedLiveHealingRunner(
        FOCUSED_CONFIG, mode="replay", output=root / "replay", cache_root=cache,
        ledger_path=ledger,
    ).run()
    return root, cache, ledger, mock, replay


def test_generic_phase10_path_rejects_focused_config_instead_of_classifying_main():
    with pytest.raises(ValueError, match="generic Phase10Config MAIN path is forbidden"):
        Phase10Config.load(FOCUSED_CONFIG)


def test_validate_config_is_exact_eight_in_deepseek_then_qwen_order():
    report = FocusedLiveHealingRunner.validate_config(FOCUSED_CONFIG)
    assert report["experiment_kind"] == "FOCUSED_LIVE_HEALING_CANARY"
    assert report["records_planned"] == 8 and report["heldout48_overlap"] == 0
    assert [item["provider"] for item in report["plan"]] == ["deepseek"] * 4 + ["dashscope"] * 4
    assert [item["scenario_id"] for item in report["plan"]] == [
        "M01-PD", "M06-PD", "M11-PD", "M16-PD",
    ] * 2


@pytest.mark.parametrize("remove", [True, False])
def test_any_missing_or_extra_plan_record_fails_closed(tmp_path, remove):
    raw = yaml.safe_load(FOCUSED_CONFIG.read_text(encoding="utf-8"))
    if remove:
        raw["experiment"]["selection"].pop()
        raw["experiment"]["planned_records"] = 7
    else:
        raw["experiment"]["selection"].append(dict(raw["experiment"]["selection"][-1]))
        raw["experiment"]["planned_records"] = 9
    path = tmp_path / "mutated.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="Focused config validation failed"):
        FocusedLiveHealingConfig.load(path)


def test_run_authorized_false_refuses_real_provider_before_credentials_or_network():
    config = FocusedLiveHealingConfig.load()
    adapter = FocusedProviderAdapter("real", config.models[0], run_authorized=False)
    with pytest.raises(PermissionError, match="requires config authorization"):
        adapter.create()


def test_dry_run_has_exact_plan_and_zero_provider_or_network_initialization():
    report = FocusedLiveHealingRunner.dry_run(FOCUSED_CONFIG)
    assert report["records_planned"] == 8
    assert report["network_calls"] == report["provider_instances"] == 0


def test_mock_executes_exactly_eight_complete_orchestrator_records(focused_artifacts):
    _, _, _, mock, _ = focused_artifacts
    assert mock["summary"]["records_completed"] == mock["summary"]["accepted"] == 8
    assert mock["summary"]["execution_mode"] == "MOCK_INFRASTRUCTURE_VALIDATION"
    assert mock["summary"]["api_attempts_delta"] == 0
    assert mock["summary"]["input_tokens_delta"] == mock["summary"]["output_tokens_delta"] == 0
    assert all(len(row["stage_trace"]) == 16 for row in mock["records"])
    assert all(any(
        event["provenance"]["controller_event"].startswith("LiveHealingOrchestrator.")
        for event in row["live_evidence_trace"]["events"]
    ) for row in mock["records"])


def test_cache_has_48_unique_stage_keys_and_no_cross_scope_collisions(focused_artifacts):
    _, cache, _, mock, _ = focused_artifacts
    raw_files = tuple((cache / "driftguard_focused_live_healing_v1/raw").glob("*.json"))
    assert len(raw_files) == 8 * 6
    assert len({path.stem for path in raw_files}) == len(raw_files)
    assert {row["cache"]["entries_after"] - row["cache"]["entries_before"] for row in mock["records"]} == {6}


def test_replay_completes_eight_without_provider_fallback_attempt_tokens_or_cost(focused_artifacts):
    _, _, _, _, replay = focused_artifacts
    summary = replay["summary"]
    assert summary["records_completed"] == summary["accepted"] == 8
    assert summary["provider_fallback_calls"] == 0
    assert summary["api_attempts_delta"] == 0 and summary["cost_cny_delta"] == 0
    assert summary["input_tokens_delta"] == summary["output_tokens_delta"] == 0
    assert all(row["cache"]["hits"] == 6 for row in replay["records"])


def test_replay_cache_miss_stops_without_provider_fallback_and_marks_partial(tmp_path):
    ledger = _temporary_ledger(tmp_path)
    runner = FocusedLiveHealingRunner(
        FOCUSED_CONFIG, mode="replay", output=tmp_path / "replay", cache_root=tmp_path / "empty-cache",
        ledger_path=ledger,
    )
    with pytest.raises(ReplayCacheMiss):
        runner.run()
    summary = json.loads((tmp_path / "replay/summary.json").read_text(encoding="utf-8"))
    assert summary["run_status"] == "INCOMPLETE" and summary["records_completed"] == 0


def test_record_checkpoint_resume_skips_all_provider_calls_and_preserves_records(focused_artifacts):
    root, cache, ledger, mock, _ = focused_artifacts
    resumed = FocusedLiveHealingRunner(
        FOCUSED_CONFIG, mode="mock", output=root / "mock", cache_root=cache,
        ledger_path=ledger,
    ).run(resume=True)
    assert resumed["summary"]["records_resumed"] == 8
    assert resumed["summary"]["records_completed"] == 8
    assert resumed["records"] == mock["records"]


def test_resume_after_deepseek_batch_does_not_repeat_completed_records(focused_artifacts, tmp_path):
    _, cache, ledger, mock, _ = focused_artifacts
    output = tmp_path / "deepseek-batch-resume"
    partial = FocusedLiveHealingRunner(
        FOCUSED_CONFIG, mode="mock", output=output, cache_root=cache,
        ledger_path=ledger,
    )
    partial._bind_checkpoint(False)
    partial.writer.write_manifest(partial._manifest())
    for record in mock["records"][:4]:
        partial.writer.write_record(record)
        partial.checkpoints.write(record["record_id"], record)
    resumed = FocusedLiveHealingRunner(
        FOCUSED_CONFIG, mode="mock", output=output, cache_root=cache,
        ledger_path=ledger,
    ).run(resume=True)
    assert resumed["summary"]["records_resumed"] == 4
    assert resumed["summary"]["records_completed"] == 8
    assert all(row["provider_boundary_calls"] == 0 for row in resumed["records"][4:])
    assert all(row["cache"]["hits"] == 6 for row in resumed["records"][4:])


def test_resume_rejects_changed_frozen_identity(focused_artifacts):
    root, cache, ledger, _, _ = focused_artifacts
    identity_path = root / "mock/checkpoint/identity.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    identity["source_snapshot"] = "0" * 64
    identity_path.write_text(json.dumps(identity), encoding="utf-8")
    with pytest.raises(ValueError, match="refusing resume"):
        FocusedLiveHealingRunner(
            FOCUSED_CONFIG, mode="mock", output=root / "mock", cache_root=cache,
            ledger_path=ledger,
        ).run(resume=True)


def test_evidence_history_state_and_patch_registry_are_record_isolated(focused_artifacts):
    _, _, _, mock, _ = focused_artifacts
    trace_ids = [row["live_evidence_trace"]["trace_id"] for row in mock["records"]]
    scope_keys = [row["patch_registry_state"]["scope_key"] for row in mock["records"]]
    assert len(set(trace_ids)) == len(set(scope_keys)) == 8
    assert all(row["patch_registry_state"]["accepted_count"] == 1 for row in mock["records"])
    assert all(row["main_state_pollution"] is False for row in mock["records"])


def test_no_symbolic_or_oracle_fallback_and_no_hidden_cache_material(focused_artifacts):
    _, cache, _, mock, _ = focused_artifacts
    assert mock["summary"]["symbolic_fallback_count"] == mock["summary"]["oracle_fallback_count"] == 0
    encoded = "".join(
        path.read_text(encoding="utf-8")
        for path in (cache / "driftguard_focused_live_healing_v1").rglob("*.json")
    ).lower()
    for forbidden in ("source_drift_id", "expected_patch", "evaluator_view", "runtime_contract", "authorization"):
        assert forbidden not in encoded


def test_formal_ledger_hash_is_unchanged_by_mock_and_replay(focused_artifacts):
    _, _, ledger, mock, replay = focused_artifacts
    current = hashlib.sha256(ledger.read_bytes()).hexdigest()
    assert mock["summary"]["ledger_before_sha256"] == mock["summary"]["ledger_after_sha256"] == current
    assert replay["summary"]["ledger_before_sha256"] == replay["summary"]["ledger_after_sha256"] == current
