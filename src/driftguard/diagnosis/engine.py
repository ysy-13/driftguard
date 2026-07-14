from __future__ import annotations

from driftguard.evidence.models import AgentView
from .classifier import EvidenceClassifier
from .localizer import DriftLocalizer
from .models import DiagnosisResult, DiagnosisState
from .patch_gate import PatchEligibilityGate
from .state_machine import DiagnosisStateMachine


SUSPECTED = {
    "AE": DiagnosisState.AGENT_ERROR_SUSPECTED,
    "TF": DiagnosisState.TRANSIENT_FAILURE_SUSPECTED,
    "PD": DiagnosisState.PERSISTENT_DRIFT_SUSPECTED,
}
CONFIRMED = {
    "AE": DiagnosisState.AGENT_ERROR_CONFIRMED,
    "TF": DiagnosisState.TRANSIENT_FAILURE_CONFIRMED,
    "PD": DiagnosisState.PERSISTENT_DRIFT_CONFIRMED,
}


class DiagnosisEngine:
    def __init__(self, view: AgentView):
        if type(view) is not AgentView:
            raise TypeError("DiagnosisEngine accepts only AgentView")
        self.view = view
        self.classifier = EvidenceClassifier()
        self.localizer = DriftLocalizer()
        self.patch_gate = PatchEligibilityGate()

    def diagnose(self) -> DiagnosisResult:
        classification = self.classifier.classify(self.view)
        machine = DiagnosisStateMachine()
        if classification.label == "UNRESOLVED":
            machine.transition(DiagnosisState.INSUFFICIENT_EVIDENCE, classification.evidence_ids, (), classification.reason_code, 0.0)
        else:
            machine.transition(SUSPECTED[classification.label], classification.evidence_ids[:1], (), "INITIAL_HYPOTHESIS", 0.6)
            rejected = tuple(label for label in ("AE", "TF", "PD") if label != classification.label)
            machine.transition(CONFIRMED[classification.label], classification.evidence_ids, rejected, classification.reason_code, classification.confidence)
        localization = self.localizer.localize(self.view, classification.label)
        patch = self.patch_gate.decide(self.view, classification.label)
        return DiagnosisResult(
            classification.label, machine.state.value, tuple(machine.transitions), localization,
            patch, classification.evidence_ids, classification.confidence,
        )
