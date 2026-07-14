from __future__ import annotations

from dataclasses import dataclass

from driftguard.diagnosis.models import DiagnosisResult
from driftguard.healing import HealingEngine

from .base import RecoveryDecision, RecoveryPolicy


class DriftGuardPolicy(RecoveryPolicy):
    method = "driftguard_symbolic"
    prompt_name = "driftguard_attribution_v1.txt"
    allows_probe = True
    persistent_patch = True

    def __init__(self, backend: str = "symbolic"):
        if backend not in {"symbolic", "llm"}:
            raise ValueError("unknown DriftGuard backend")
        self.backend = backend
        self.method = f"driftguard_{backend}"
        self.raw_llm_prediction = None
        self.guard_decision = None

    def after_failure(self, call, response):
        return RecoveryDecision(request_llm=True, context={"evidence_collection": True, "probe_allowed": True})

    def gate(self, diagnosis: DiagnosisResult) -> bool:
        self.guard_decision = diagnosis.patch_eligibility.decision
        return diagnosis.predicted_class == "PD" and diagnosis.patch_eligibility.decision == "ELIGIBLE"

    def record_llm_prediction(self, value):
        self.raw_llm_prediction = value

    def guarded_llm_patch(self, proposed_patch, validator):
        # No symbolic fallback is permitted. The exact LLM proposal either
        # survives deterministic validation or is rejected.
        return proposed_patch if validator(proposed_patch) else None


class OracleSymbolicPolicy(DriftGuardPolicy):
    method = "oracle_symbolic_upper_bound"

    def __init__(self):
        super().__init__("symbolic")
        self.method = "oracle_symbolic_upper_bound"
        self.is_upper_bound = True

