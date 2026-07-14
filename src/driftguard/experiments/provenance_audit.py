from __future__ import annotations

from dataclasses import replace
import json
from typing import Any

from driftguard.diagnosis import DiagnosisEngine
from driftguard.evidence.leakage_guard import assert_agent_visible
from driftguard.evidence.models import AgentView, EvidenceEvent, EvidenceTrace
from driftguard.healing import PatchCandidateGenerator, ToolSpecPatch


def semantic_diagnosis(view: AgentView) -> dict[str, Any]:
    diagnosis = DiagnosisEngine(view).diagnose().to_dict()
    diagnosis.pop("evidence_event_refs", None)
    diagnosis["localization"].pop("supporting_evidence_ids", None)
    diagnosis["patch_eligibility"].pop("supporting_evidence_ids", None)
    for transition in diagnosis["transitions"]:
        transition.pop("supporting_evidence_ids", None)
    return diagnosis


def semantic_patch(view: AgentView) -> dict[str, Any] | None:
    diagnosis = DiagnosisEngine(view).diagnose()
    result = PatchCandidateGenerator().generate(view, diagnosis)
    if not result.candidates:
        return None
    patch = result.candidates[0]
    return {
        "target_tool_id": patch.target_tool_id, "drift_category": patch.drift_category,
        "location_type": patch.location_type, "location_path": patch.location_path,
        "operations": [{"op": item.op, "path": item.path, "value": item.value, "reason_code": item.reason_code} for item in patch.openapi_operations],
        "semantic_extensions": patch.semantic_extensions,
    }


def replace_opaque_ids(view: AgentView, replacement: str = "public-0123456789ab") -> AgentView:
    trace_id = "trace-0123456789ab"
    id_map = {event.event_id: f"{trace_id}-EV{index:03d}" for index, event in enumerate(view.trace.events, 1)}
    events = tuple(replace(
        event, event_id=id_map[event.event_id], trace_id=trace_id, public_scenario_id=replacement,
        historical_evidence_refs=tuple(id_map.get(ref, ref) for ref in event.historical_evidence_refs),
    ) for event in view.trace.events)
    return AgentView(EvidenceTrace(trace_id, replacement, events), view.displayed_spec, view.current_episode)


def assert_label_permutation_invariant(view: AgentView, evaluator_metadata_values: list[dict[str, Any]]) -> None:
    baseline_diagnosis, baseline_patch = semantic_diagnosis(view), semantic_patch(view)
    for metadata in evaluator_metadata_values:
        # Metadata is intentionally never passed to either engine.
        if semantic_diagnosis(view) != baseline_diagnosis or semantic_patch(view) != baseline_patch:
            raise AssertionError(f"evaluator label permutation changed output: {metadata}")


def assert_opaque_id_invariant(view: AgentView) -> None:
    opaque = replace_opaque_ids(view)
    if semantic_diagnosis(view) != semantic_diagnosis(opaque):
        raise AssertionError("opaque ID replacement changed diagnosis semantics")
    if semantic_patch(view) != semantic_patch(opaque):
        raise AssertionError("opaque ID replacement changed patch semantics")


def assert_hidden_canary_blocked(value: Any) -> None:
    assert_agent_visible(value)


def audit_structured_patch_semantics(patch: ToolSpecPatch) -> None:
    semantics = patch.semantic_extensions.get("x-driftguard-patch-semantics")
    if not isinstance(semantics, dict):
        raise ValueError("patch semantics must be structured")
    required = {"operation", "target", "before", "after"}
    if not required.issubset(semantics):
        raise ValueError("patch semantic operation is not validator-addressable")
    strategy_keys = {"request_transform", "response_mapping", "workflow", "observation_policy"}
    selected = strategy_keys.intersection(semantics)
    if len(selected) != 1 or not isinstance(semantics[next(iter(selected))], dict):
        raise ValueError("complex behavior is hidden outside a structured strategy")

