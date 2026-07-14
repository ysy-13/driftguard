import json

from driftguard.runners.attribution_conformance_runner import AttributionConformanceRunner


def test_sixty_scenario_attribution_conformance():
    report = AttributionConformanceRunner().run()
    summary = report["summary"]
    assert summary["scenarios"] == 60
    assert summary["classification_correct"] == 60
    assert summary["agent_error_correct"] == 20
    assert summary["transient_failure_correct"] == 20
    assert summary["persistent_drift_correct"] == 20
    assert summary["macro_f1"] == 1.0
    assert summary["target_tool_localization_accuracy"] == 1.0
    assert summary["drift_category_localization_accuracy"] == 1.0
    assert summary["exact_location_accuracy"] == 1.0
    assert summary["pd_patch_eligibility_recall"] == 1.0
    assert summary["agent_error_patch_forbidden"] == 20
    assert summary["transient_failure_patch_forbidden"] == 20
    assert summary["persistent_drift_patch_eligible"] == 20
    assert summary["unsafe_probe_count"] == 0
    assert summary["ground_truth_leakage_count"] == 0
    assert summary["deterministic_replay_passed"] == 60
    assert summary["oracle_regression_passed"] == 32
    assert summary["phase6_injection_families_passed"] == 20
    assert summary["canonical_hashes_unchanged"]
    encoded = json.dumps(report)
    assert "runtime_contract" not in encoded
    assert '"fixture_id": "S0"' not in encoded
    assert "verification_token" not in encoded
