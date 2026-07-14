from __future__ import annotations

import hashlib
import json

from driftguard.diagnosis.models import DiagnosisResult
from driftguard.evidence.collector import stable_hash
from driftguard.evidence.models import AgentView, thaw

from .category_generators import generate_icd, generate_rsd, generate_sed, generate_wpd
from .leakage_guard import validate_generator_inputs
from .models import GenerationResult, PatchOperation, ToolSpecPatch


class PatchCandidateGenerator:
    def __init__(self, max_candidates: int = 5):
        if max_candidates < 1 or max_candidates > 5:
            raise ValueError("candidate budget must be between 1 and 5")
        self.max_candidates = max_candidates

    def generate(self, agent_view: AgentView, diagnosis: DiagnosisResult) -> GenerationResult:
        validate_generator_inputs(agent_view, diagnosis)
        if diagnosis.predicted_class != "PD":
            return GenerationResult("FORBIDDEN", reason_codes=("NON_PERSISTENT_CLASS",))
        if diagnosis.patch_eligibility.decision != "ELIGIBLE":
            return GenerationResult("NOT_APPLICABLE", reason_codes=("PATCH_NOT_ELIGIBLE",))
        location = diagnosis.localization
        if not location.tool_id or not location.location_path or location.confidence < 1.0:
            return GenerationResult("INSUFFICIENT_EVIDENCE", reason_codes=("EXACT_LOCATION_REQUIRED",))
        evidence_refs = tuple(dict.fromkeys(
            location.supporting_evidence_ids
            + diagnosis.patch_eligibility.supporting_evidence_ids
        ))
        events = {event.event_id: event for event in agent_view.trace.events}
        failure = next((events[ref] for ref in location.supporting_evidence_ids if ref in events), None)
        if failure is None or not evidence_refs:
            return GenerationResult("INSUFFICIENT_EVIDENCE", reason_codes=("VISIBLE_FAILURE_REQUIRED",))
        spec = thaw(agent_view.displayed_spec)
        generators = {"ICD": generate_icd, "RSD": generate_rsd, "WPD": generate_wpd, "SED": generate_sed}
        try:
            semantics, patch_path, value = generators[location.drift_category](spec, location, failure)
        except (KeyError, TypeError, ValueError) as exc:
            return GenerationResult("INSUFFICIENT_EVIDENCE", reason_codes=(f"GENERATION_FAILED:{type(exc).__name__}",))
        operation = PatchOperation(
            "add" if not _pointer_exists(spec, patch_path) else "replace",
            patch_path, value, evidence_refs, f"{location.drift_category}_VISIBLE_EVIDENCE",
        )
        identity = json.dumps({
            "tool": location.tool_id, "location": location.location_path,
            "operation": operation.to_dict(), "semantics": semantics,
        }, sort_keys=True, separators=(",", ":"))
        patch = ToolSpecPatch(
            patch_id=f"patch-{hashlib.sha256(identity.encode()).hexdigest()[:16]}",
            patch_version="1.0", created_at="2026-01-01T00:00:00Z",
            target_tool_id=location.tool_id,
            source_spec_fingerprint=stable_hash(spec),
            drift_category=location.drift_category,
            location_type=location.location_type,
            location_path=location.location_path,
            evidence_refs=evidence_refs,
            rationale_codes=("PERSISTENT_REPRODUCTION", "DISCRIMINATIVE_PROBE", "MINIMAL_VISIBLE_DELTA"),
            openapi_operations=(operation,),
            semantic_extensions={"x-driftguard-patch-semantics": semantics},
            expected_agent_behavior_change=_behavior_change(location.drift_category),
            confidence=min(diagnosis.confidence, location.confidence),
        )
        return GenerationResult("PROPOSED", (patch,))


def _pointer_exists(document: dict, path: str) -> bool:
    current = document
    try:
        for raw in path[1:].split("/"):
            current = current[raw.replace("~1", "/").replace("~0", "~")]
        return True
    except (KeyError, TypeError, IndexError):
        return False


def _behavior_change(category: str) -> str:
    return {
        "ICD": "transform future requests to the evidence-supported runtime input contract",
        "RSD": "parse runtime responses through the evidence-supported field mapping",
        "WPD": "execute the evidence-supported prerequisite workflow before the target tool",
        "SED": "confirm state effects using the evidence-supported observation policy",
    }[category]

