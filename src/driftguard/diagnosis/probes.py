from __future__ import annotations

from dataclasses import dataclass

from driftguard.evidence.models import AgentView, thaw


SAFE_PROBE_TYPES = {
    "exact_retry", "local_schema_check", "response_shape_inspection",
    "read_after_write", "repeated_read", "independent_instance_reproduction",
    "precondition_inspection", "historical_behavior_comparison", "regression_check",
}


@dataclass(frozen=True)
class ProbePlan:
    probe_id: str
    probe_type: str
    tool_id: str | None
    evidence_refs: tuple[str, ...]
    isolated: bool
    read_only: bool


class ProbeBudgetExceeded(RuntimeError):
    pass


class UnsafeProbeError(ValueError):
    pass


class ProbePlanner:
    def __init__(self, max_probes: int = 8):
        if max_probes < 1:
            raise ValueError("probe budget must be positive")
        self.max_probes = max_probes

    def plan(self, view: AgentView) -> tuple[ProbePlan, ...]:
        failure = next((event for event in view.trace.events if event.normalized_observation), None)
        if failure is None:
            return ()
        channel = thaw(failure.normalized_observation).get("channel")
        discriminative = {
            "response_error": "local_schema_check",
            "missing_output_field": "response_shape_inspection",
            "workflow_rejection": "precondition_inspection",
            "state_mismatch": "read_after_write",
        }.get(channel, "historical_behavior_comparison")
        types = ("exact_retry", discriminative, "independent_instance_reproduction")
        if len(types) > self.max_probes:
            raise ProbeBudgetExceeded("probe budget exhausted")
        return tuple(
            ProbePlan(
                f"probe-{index + 1}", probe_type, failure.tool_id, (failure.event_id,),
                isolated=probe_type in {"exact_retry", "read_after_write", "independent_instance_reproduction"},
                read_only=probe_type not in {"exact_retry", "independent_instance_reproduction"},
            )
            for index, probe_type in enumerate(types)
        )
