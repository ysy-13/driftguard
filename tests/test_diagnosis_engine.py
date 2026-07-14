import pytest
import inspect

from driftguard.diagnosis import DiagnosisEngine
from driftguard.diagnosis.classifier import EvidenceClassifier
from driftguard.contracts.loader import load_openapi
from driftguard.evidence import AgentView, EvidenceCollector, EvidenceStore
from driftguard.runtime import ExecutionMode, ExecutionProfile
from phase7_helpers import build_view


@pytest.mark.parametrize("kind,expected,patch", [
    ("AE", "AE", "FORBIDDEN"),
    ("TF", "TF", "FORBIDDEN"),
    ("PD", "PD", "ELIGIBLE"),
    ("SINGLE", "UNRESOLVED", "INSUFFICIENT_EVIDENCE"),
])
def test_evidence_rules_classify_without_ground_truth(kind, expected, patch):
    result = DiagnosisEngine(build_view(kind)).diagnose()
    assert result.predicted_class == expected
    assert result.patch_eligibility.decision == patch
    assert result.transitions


def test_single_failure_or_timeout_cannot_confirm_pd_or_patch():
    result = DiagnosisEngine(build_view("SINGLE")).diagnose()
    assert result.final_state == "INSUFFICIENT_EVIDENCE"
    assert result.patch_eligibility.independent_failure_count == 1


def test_engine_rejects_runtime_profile_or_evaluator_objects():
    with pytest.raises(TypeError):
        DiagnosisEngine(object())


def test_classifier_has_no_family_or_scenario_lookup_table():
    source = inspect.getsource(EvidenceClassifier)
    assert "matched_case_id" not in source
    assert "source_drift_id" not in source
    assert "M01" not in source


def test_pd_localization_uses_visible_schema_and_observation():
    result = DiagnosisEngine(build_view("PD")).diagnose()
    assert result.localization.tool_id == "create_issue"
    assert result.localization.drift_category == "ICD"
    assert result.localization.location_path == "/components/schemas/CreateIssueRequest/required"


def test_patch_gate_requires_probe_regressions_and_two_failures():
    incomplete = DiagnosisEngine(build_view("SINGLE")).diagnose()
    assert incomplete.patch_eligibility.decision == "INSUFFICIENT_EVIDENCE"
    complete = DiagnosisEngine(build_view("PD")).diagnose()
    assert complete.patch_eligibility.independent_failure_count == 2
    assert complete.patch_eligibility.discriminative_probe_passed
    assert complete.patch_eligibility.regression_requirements_identified


def test_response_interpretation_error_is_direct_ae_evidence():
    store = EvidenceStore("T", "public-safe")
    collector = EvidenceCollector(store, load_openapi())
    collector.add("local_validation", 3, "get_repository", local_validation_result={"valid": True})
    collector.add("tool_response", 3, "get_repository", visible_runtime_response={"status_code": 200},
                  probe_metadata={"interpretation_status": "failed"})
    collector.add("probe_result", 4, "get_repository",
                  probe_metadata={"corrected_behavior": True, "success": True})
    result = DiagnosisEngine(AgentView(store.trace, load_openapi(), 4)).diagnose()
    assert result.predicted_class == "AE"
    assert result.patch_eligibility.decision == "FORBIDDEN"


def test_unsafe_write_blocks_otherwise_complete_pd_patch():
    view = build_view("PD")
    # Rebuild the existing immutable trace and append a visible unsafe-probe audit.
    from driftguard.evidence import EvidenceEvent
    event = EvidenceEvent(
        f"{view.trace.trace_id}-unsafe", view.trace.trace_id, view.trace.public_scenario_id,
        5, len(view.trace.events) + 1, "probe_result",
        probe_metadata={"unsafe_write": True},
    )
    unsafe_view = AgentView(view.trace.append(event), load_openapi(), 5)
    result = DiagnosisEngine(unsafe_view).diagnose()
    assert result.predicted_class == "PD"
    assert result.patch_eligibility.decision == "INSUFFICIENT_EVIDENCE"
