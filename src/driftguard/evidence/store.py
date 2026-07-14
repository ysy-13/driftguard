from __future__ import annotations

from .models import EvidenceEvent, EvidenceTrace


class EvidenceStore:
    def __init__(self, trace_id: str, public_scenario_id: str):
        self._trace = EvidenceTrace(trace_id, public_scenario_id)

    @property
    def trace(self) -> EvidenceTrace:
        return self._trace

    def append(self, event: EvidenceEvent) -> None:
        self._trace = self._trace.append(event)

    def reset(self, trace_id: str | None = None, public_scenario_id: str | None = None) -> None:
        self._trace = EvidenceTrace(
            trace_id or self._trace.trace_id,
            public_scenario_id or self._trace.public_scenario_id,
        )
