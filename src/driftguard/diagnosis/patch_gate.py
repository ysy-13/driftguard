from __future__ import annotations

from driftguard.evidence.models import AgentView, thaw
from .models import PatchEligibilityDecision


class PatchEligibilityGate:
    def decide(self, view: AgentView, predicted_class: str) -> PatchEligibilityDecision:
        events = view.trace.events
        independent = [event for event in events if thaw(event.probe_metadata).get("independent_failure") is True]
        probes = [event for event in events if thaw(event.probe_metadata).get("discriminative") is True and thaw(event.probe_metadata).get("passed") is True]
        regressions = [event for event in events if thaw(event.probe_metadata).get("regression_check") is True and thaw(event.probe_metadata).get("success") is True]
        unsafe = sum(int(thaw(event.probe_metadata).get("unsafe_write", False)) for event in events)
        refs = tuple(event.event_id for event in independent + probes + regressions)
        if predicted_class in {"AE", "TF"}:
            return PatchEligibilityDecision("FORBIDDEN", refs, len(independent), bool(probes), bool(regressions), unsafe, ("NON_PERSISTENT_ROOT_CAUSE",))
        if predicted_class != "PD":
            return PatchEligibilityDecision("INSUFFICIENT_EVIDENCE", refs, len(independent), bool(probes), bool(regressions), unsafe, ("CLASS_NOT_CONFIRMED",))
        reasons: list[str] = []
        if len(independent) < 2:
            reasons.append("TWO_INDEPENDENT_FAILURES_REQUIRED")
        if not probes:
            reasons.append("DISCRIMINATIVE_PROBE_REQUIRED")
        if len(regressions) < 3:
            reasons.append("THREE_REGRESSION_CHECKS_REQUIRED")
        if unsafe:
            reasons.append("UNSAFE_WRITE_OBSERVED")
        decision = "ELIGIBLE" if not reasons else "INSUFFICIENT_EVIDENCE"
        return PatchEligibilityDecision(decision, refs, len(independent), bool(probes), len(regressions) >= 3, unsafe, tuple(reasons or ["ATTRIBUTION_PROTOCOL_SATISFIED"]))
