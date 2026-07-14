from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class DiagnosisState(str, Enum):
    UNRESOLVED = "UNRESOLVED"
    AGENT_ERROR_SUSPECTED = "AGENT_ERROR_SUSPECTED"
    TRANSIENT_FAILURE_SUSPECTED = "TRANSIENT_FAILURE_SUSPECTED"
    PERSISTENT_DRIFT_SUSPECTED = "PERSISTENT_DRIFT_SUSPECTED"
    AGENT_ERROR_CONFIRMED = "AGENT_ERROR_CONFIRMED"
    TRANSIENT_FAILURE_CONFIRMED = "TRANSIENT_FAILURE_CONFIRMED"
    PERSISTENT_DRIFT_CONFIRMED = "PERSISTENT_DRIFT_CONFIRMED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


@dataclass(frozen=True)
class StateTransition:
    previous_state: str
    new_state: str
    supporting_evidence_ids: tuple[str, ...]
    rejected_hypotheses: tuple[str, ...]
    reason_code: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["supporting_evidence_ids"] = list(self.supporting_evidence_ids)
        value["rejected_hypotheses"] = list(self.rejected_hypotheses)
        return value


@dataclass(frozen=True)
class LocalizationResult:
    tool_id: str | None
    drift_category: str
    location_type: str
    location_path: str | None
    expected_behavior: str
    observed_behavior: str
    supporting_evidence_ids: tuple[str, ...]
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["supporting_evidence_ids"] = list(self.supporting_evidence_ids)
        return value


@dataclass(frozen=True)
class PatchEligibilityDecision:
    decision: str
    supporting_evidence_ids: tuple[str, ...]
    independent_failure_count: int
    discriminative_probe_passed: bool
    regression_requirements_identified: bool
    unsafe_write_count: int
    reason_codes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["supporting_evidence_ids"] = list(self.supporting_evidence_ids)
        value["reason_codes"] = list(self.reason_codes)
        return value


@dataclass(frozen=True)
class DiagnosisResult:
    predicted_class: str
    final_state: str
    transitions: tuple[StateTransition, ...]
    localization: LocalizationResult
    patch_eligibility: PatchEligibilityDecision
    evidence_event_refs: tuple[str, ...]
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicted_class": self.predicted_class,
            "final_state": self.final_state,
            "transitions": [transition.to_dict() for transition in self.transitions],
            "localization": self.localization.to_dict(),
            "patch_eligibility": self.patch_eligibility.to_dict(),
            "evidence_event_refs": list(self.evidence_event_refs),
            "confidence": self.confidence,
        }
