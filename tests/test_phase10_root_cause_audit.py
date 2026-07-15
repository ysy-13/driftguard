import json

from driftguard.contracts.loader import PROJECT_ROOT
from driftguard.phase10.root_cause_audit import (
    CacheOnlyProvider, Phase10EndToEndRootCauseAudit, ReadOnlyCache,
)


def test_audit_provider_always_rejects_network_fallback():
    provider = CacheOnlyProvider()
    try:
        provider.complete(type("Request", (), {"public_scenario_id": "p", "method": "m"})())
    except RuntimeError as exc:
        assert "offline audit cache miss" in str(exc)
    else:
        raise AssertionError("audit provider accepted an uncached call")


def test_read_only_cache_rejects_writes():
    cache = ReadOnlyCache(PROJECT_ROOT / "results/experiments/phase10/pilot_v3/.cache/deepseek/end_to_end")
    try:
        cache.put("key", None)
    except PermissionError as exc:
        assert "read-only" in str(exc)
    else:
        raise AssertionError("audit cache accepted a write")


def test_offline_root_cause_audit_covers_all_records_and_required_sections(tmp_path):
    json_path, md_path, report = Phase10EndToEndRootCauseAudit().write(tmp_path)
    assert report["records_audited"] == 120
    assert report["real_api_calls"] == 0 and report["new_cost_cny"] == 0
    assert report["integrity"]["cache_unchanged"] and report["integrity"]["ledger_unchanged"]
    assert report["result_funnel"]["task_success"] == 0
    assert report["oracle_downstream_control"]["provider_calls"] == 0
    assert len(report["root_causes"]) >= 3
    assert report["full_real_model_experiment_status"] == "NOT RUN"
    assert json.loads(json_path.read_text())["records_audited"] == 120
    assert "Root-Cause Audit" in md_path.read_text()
