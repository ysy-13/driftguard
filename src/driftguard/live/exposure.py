from __future__ import annotations

from typing import Any

from driftguard.evidence.models import EvidenceTrace, thaw


def evaluate_drift_exposure(trace: EvidenceTrace, evaluator_target_tool: str) -> dict[str, Any]:
    """Post-execution metric only; its output must never enter AgentView."""

    controller_events = [
        event for event in trace.events
        if event.event_type in {"tool_response", "retry_result"}
    ]
    target_events = [event for event in controller_events if event.tool_id == evaluator_target_tool]
    exposed_event = next((
        event for event in target_events
        if thaw(event.probe_metadata).get("independent_failure")
        and (
            thaw(event.normalized_observation).get("task_complete") is False
            or thaw(event.normalized_observation).get("channel")
            in {"response_error", "missing_output_field", "state_mismatch", "task_incomplete"}
        )
    ), None)
    observation = thaw(exposed_event.normalized_observation) if exposed_event else {}
    signature = None
    if exposed_event is not None:
        signature = {
            "tool_id": exposed_event.tool_id,
            "channel": observation.get("channel"),
            "status_code": observation.get("status_code"),
            "error_code": observation.get("error_code"),
            "field": observation.get("field"),
            "state_diff_present": bool(exposed_event.visible_state_diff),
        }
    drift_exposed = exposed_event is not None
    return {
        "drift_exposed": drift_exposed,
        "observed_target_tool": exposed_event.tool_id if exposed_event else None,
        "observed_failure_signature": signature,
        "attribution_evaluable": drift_exposed,
        "pre_drift_agent_failure": bool(not drift_exposed and any(
            thaw(event.probe_metadata).get("independent_failure") for event in controller_events
        )),
    }
