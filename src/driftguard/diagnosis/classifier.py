from __future__ import annotations

from dataclasses import dataclass

from driftguard.evidence.models import AgentView, thaw


@dataclass(frozen=True)
class Classification:
    label: str
    evidence_ids: tuple[str, ...]
    reason_code: str
    confidence: float


class EvidenceClassifier:
    def classify(self, view: AgentView) -> Classification:
        events = view.trace.events
        invalid = [
            event for event in events
            if event.event_type == "local_validation"
            and thaw(event.local_validation_result).get("valid") is False
        ]
        interpretations = [
            event for event in events
            if thaw(event.probe_metadata).get("interpretation_status") == "failed"
        ]
        corrections = [
            event for event in events
            if thaw(event.probe_metadata).get("corrected_behavior") is True
            and thaw(event.probe_metadata).get("success") is True
        ]
        if (invalid or interpretations) and corrections:
            refs = tuple(event.event_id for event in (invalid + interpretations + corrections))
            return Classification("AE", refs, "DIRECT_AGENT_BEHAVIOR_ERROR_CORRECTED", 1.0)

        retries = [
            event for event in events
            if event.event_type == "retry_result"
            and thaw(event.probe_metadata).get("exact_retry") is True
        ]
        independent_failures = [
            event for event in events
            if thaw(event.probe_metadata).get("independent_failure") is True
        ]
        discriminative = [
            event for event in events
            if thaw(event.probe_metadata).get("discriminative") is True
            and thaw(event.probe_metadata).get("passed") is True
        ]
        if retries and thaw(retries[-1].probe_metadata).get("success") is True and len(independent_failures) < 2:
            return Classification("TF", tuple(event.event_id for event in retries), "EXACT_LEGAL_RETRY_RECOVERED", 1.0)
        if len(independent_failures) >= 2 and discriminative:
            refs = tuple(event.event_id for event in independent_failures[:2] + discriminative[:1])
            return Classification("PD", refs, "INDEPENDENT_FAILURES_AND_PROBE", 1.0)
        refs = tuple(event.event_id for event in independent_failures + retries)
        return Classification("UNRESOLVED", refs, "INSUFFICIENT_DISCRIMINATIVE_EVIDENCE", 0.0)
