from driftguard.runners.injection_conformance_runner import InjectionConformanceRunner


def test_all_sixty_first_failures_and_followups_conform():
    report = InjectionConformanceRunner().run()
    summary = report["summary"]
    assert summary["families"] == 20
    assert summary["first_failure_scenarios"] == 60
    assert summary["agent_error_executed"] == summary["agent_error_corrected"] == 20
    assert summary["transient_failure_executed"] == summary["transient_recovered"] == 20
    assert summary["persistent_drift_executed"] == summary["persistent_reproduced"] == 20
    assert summary["shared_signature_matched"] == 20
    assert summary["canonical_hash_unchanged"] is True
    assert summary["oracle_regression_passed"] == 32
