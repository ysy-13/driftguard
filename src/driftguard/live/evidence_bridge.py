from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from driftguard.evidence.collector import EvidenceCollector, stable_hash
from driftguard.evidence.leakage_guard import assert_agent_visible
from driftguard.evidence.models import AgentView, EvidenceEvent
from driftguard.evidence.store import EvidenceStore


FORBIDDEN_PROVENANCE_KEYS = {
    "runtime_profile", "runtime_contract", "variant", "ground_truth",
    "source_drift_id", "expected_patch", "oracle_action", "evaluator_view",
}


@dataclass(frozen=True)
class LiveScope:
    public_scenario_id: str
    provider: str
    method: str
    repetition: int

    @property
    def key(self) -> str:
        return f"{self.public_scenario_id}:{self.provider}:{self.method}:r{self.repetition}"


class LiveEvidenceBridge:
    """Append-only bridge from occurred Controller events to an AgentView.

    It has no benchmark readers and accepts only caller-supplied, Agent-visible
    fields.  The Controller event name and correlation ID provide provenance;
    no evaluator label or symbolic artifact can be attached.
    """

    def __init__(self, scope: LiveScope, displayed_spec: dict[str, Any]):
        assert_agent_visible(displayed_spec)
        self.scope = scope
        self.displayed_spec = deepcopy(displayed_spec)
        trace_id = "live-" + stable_hash({"scope": scope.key, "spec": displayed_spec})[:20]
        self.store = EvidenceStore(trace_id, scope.public_scenario_id)
        self.collector = EvidenceCollector(self.store, displayed_spec)
        self._last_episode = 0
        self._execution_contexts: set[str] = set()

    @property
    def trace(self):
        return self.store.trace

    def emit(
        self,
        event_type: str,
        episode: int,
        *,
        source_event: str,
        correlation_id: str | None = None,
        execution_context_id: str | None = None,
        tool_id: str | None = None,
        **visible_fields: Any,
    ) -> EvidenceEvent:
        if episode < self._last_episode:
            raise ValueError("live evidence cannot move backward in episode time")
        if any(key in visible_fields for key in FORBIDDEN_PROVENANCE_KEYS):
            raise ValueError("hidden or evaluator-only material cannot enter LiveEvidence")
        provenance = {
            "source": "controller_event",
            "controller_event": source_event,
            "scope_key": self.scope.key,
        }
        if execution_context_id:
            provenance["execution_context_id"] = execution_context_id
            self._execution_contexts.add(execution_context_id)
        assert_agent_visible({"visible_fields": visible_fields, "provenance": provenance})
        sequence = len(self.trace.events) + 1
        event = self.collector.add(
            event_type, episode, tool_id,
            correlation_id=correlation_id or f"{self.scope.key}:EV{sequence:03d}",
            provenance=provenance,
            **deepcopy(visible_fields),
        )
        self._last_episode = episode
        return event

    def agent_view(self, through_episode: int | None = None) -> AgentView:
        episode = self._last_episode if through_episode is None else through_episode
        if episode > self._last_episode:
            raise ValueError("future evidence cannot be requested")
        trace = self.trace.through_episode(episode)
        return AgentView(trace, self.displayed_spec, episode)

    def reset(self) -> None:
        self.store.reset()
        self._last_episode = 0
        self._execution_contexts.clear()

    @property
    def execution_context_count(self) -> int:
        return len(self._execution_contexts)
