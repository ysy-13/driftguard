from __future__ import annotations

from driftguard.contracts.registry import ContractRegistry
from driftguard.diagnosis.models import PatchEligibilityDecision
from driftguard.evidence.models import AgentView, thaw


CLASS_MAP = {
    "AGENT_ERROR": "AE", "TRANSIENT_FAILURE": "TF",
    "PERSISTENT_DRIFT": "PD", "INSUFFICIENT_EVIDENCE": "INSUFFICIENT_EVIDENCE",
}


class LivePatchEligibilityGate:
    """Hard evidence gate; it never changes the LLM attribution."""

    def decide(self, view: AgentView, attribution: dict, *, budget_remaining: bool = True) -> PatchEligibilityDecision:
        predicted = CLASS_MAP.get(attribution.get("predicted_class"), attribution.get("predicted_class"))
        target = attribution.get("target_tool_id")
        events = view.trace.events
        failure_groups: dict[tuple, dict[str, str]] = {}
        successful_probes = []
        regression_refs = []
        unsafe = 0
        agent_explanation = transient_explanation = False
        for event in events:
            metadata = thaw(event.probe_metadata)
            provenance = thaw(event.provenance)
            context_id = provenance.get("execution_context_id")
            if metadata.get("independent_failure") and context_id and event.tool_id == target:
                observation = thaw(event.normalized_observation)
                symptom = (
                    event.tool_id, observation.get("channel"), observation.get("status_code"),
                    observation.get("error_code"), observation.get("field"),
                )
                failure_groups.setdefault(symptom, {}).setdefault(str(context_id), event.event_id)
            if metadata.get("discriminative") and metadata.get("passed"):
                successful_probes.append(event.event_id)
            if metadata.get("regression_requirements_identified"):
                regression_refs.append(event.event_id)
            unsafe += int(bool(metadata.get("unsafe_write")))
            agent_explanation = agent_explanation or bool(metadata.get("agent_correction_succeeded"))
            transient_explanation = transient_explanation or bool(metadata.get("transient_retry_recovered"))
        independent_contexts = max(failure_groups.values(), key=len, default={})
        refs = tuple(dict.fromkeys((*independent_contexts.values(), *successful_probes, *regression_refs)))

        if predicted in {"AE", "TF"}:
            return PatchEligibilityDecision(
                "FORBIDDEN", refs, len(independent_contexts), bool(successful_probes),
                bool(regression_refs), unsafe, ("NON_PERSISTENT_LLM_ATTRIBUTION",),
            )
        if agent_explanation or transient_explanation:
            reasons = []
            if agent_explanation:
                reasons.append("AGENT_ERROR_EVIDENCE_SUFFICIENT")
            if transient_explanation:
                reasons.append("TRANSIENT_RECOVERY_EVIDENCE_SUFFICIENT")
            return PatchEligibilityDecision(
                "FORBIDDEN", refs, len(independent_contexts), bool(successful_probes),
                bool(regression_refs), unsafe, tuple(reasons),
            )
        if predicted != "PD":
            return PatchEligibilityDecision(
                "INSUFFICIENT_EVIDENCE", refs, len(independent_contexts), bool(successful_probes),
                bool(regression_refs), unsafe, ("PERSISTENT_DRIFT_NOT_PREDICTED",),
            )

        reasons: list[str] = []
        location = attribution.get("location_path")
        if not target or target not in ContractRegistry(thaw(view.displayed_spec)).operation_ids():
            reasons.append("VALID_TARGET_TOOL_REQUIRED")
        if not location:
            reasons.append("EXACT_LOCATION_REQUIRED")
        if len(independent_contexts) < 2:
            reasons.append("TWO_INDEPENDENT_FAILURES_REQUIRED")
        if not successful_probes:
            reasons.append("DISCRIMINATIVE_PROBE_REQUIRED")
        if unsafe:
            reasons.append("UNSAFE_WRITE_OBSERVED")
        if not regression_refs:
            reasons.append("REGRESSION_REQUIREMENTS_NOT_IDENTIFIED")
        if not budget_remaining:
            reasons.append("BUDGET_EXHAUSTED")
        decision = "ELIGIBLE" if not reasons else "INSUFFICIENT_EVIDENCE"
        return PatchEligibilityDecision(
            decision, refs, len(independent_contexts), bool(successful_probes),
            bool(regression_refs), unsafe, tuple(reasons or ("LIVE_PROTOCOL_SATISFIED",)),
        )
