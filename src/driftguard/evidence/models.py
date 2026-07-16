from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from .leakage_guard import assert_agent_visible, assert_public_id


EVENT_TYPES = {
    "task_received", "tool_call_proposed", "local_validation", "tool_response",
    "state_observation", "history_retrieval", "retry_result", "probe_started",
    "probe_result", "hypothesis_updated", "diagnosis_emitted",
    "displayed_spec_snapshot", "tool_call_executed", "policy_recovery",
    "probe_requested", "probe_executed", "attribution_emitted", "patch_proposed",
    "patch_validation", "immediate_repair", "future_transfer", "final_evaluation",
}


def freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): freeze(child) for key, child in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze(child) for child in value)
    return deepcopy(value)


def thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: thaw(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [thaw(child) for child in value]
    return deepcopy(value)


@dataclass(frozen=True)
class EvidenceEvent:
    event_id: str
    trace_id: str
    public_scenario_id: str
    episode_number: int
    sequence_number: int
    event_type: str
    tool_id: str | None = None
    displayed_request: Mapping[str, Any] = field(default_factory=dict)
    local_validation_result: Mapping[str, Any] = field(default_factory=dict)
    visible_runtime_response: Mapping[str, Any] = field(default_factory=dict)
    normalized_observation: Mapping[str, Any] = field(default_factory=dict)
    visible_state_diff: tuple[Any, ...] = ()
    before_state_hash: str | None = None
    after_state_hash: str | None = None
    displayed_spec_fingerprint: str | None = None
    historical_evidence_refs: tuple[str, ...] = ()
    probe_metadata: Mapping[str, Any] = field(default_factory=dict)
    correlation_id: str = ""
    provenance: Mapping[str, Any] = field(default_factory=dict)
    timestamp: str = ""

    def __post_init__(self) -> None:
        if self.event_type not in EVENT_TYPES:
            raise ValueError(f"unsupported evidence event type: {self.event_type}")
        if self.episode_number < 1 or self.sequence_number < 1:
            raise ValueError("episode and sequence numbers must be positive")
        assert_public_id(self.public_scenario_id)
        for name in (
            "displayed_request", "local_validation_result", "visible_runtime_response",
            "normalized_observation", "probe_metadata", "provenance",
        ):
            frozen = freeze(getattr(self, name))
            assert_agent_visible(thaw(frozen))
            object.__setattr__(self, name, frozen)
        object.__setattr__(self, "visible_state_diff", tuple(freeze(self.visible_state_diff)))
        object.__setattr__(self, "historical_evidence_refs", tuple(self.historical_evidence_refs))

    def to_dict(self) -> dict[str, Any]:
        value = {
            "event_id": self.event_id, "trace_id": self.trace_id,
            "public_scenario_id": self.public_scenario_id,
            "episode_number": self.episode_number, "sequence_number": self.sequence_number,
            "event_type": self.event_type, "tool_id": self.tool_id,
            "displayed_request": thaw(self.displayed_request),
            "local_validation_result": thaw(self.local_validation_result),
            "visible_runtime_response": thaw(self.visible_runtime_response),
            "normalized_observation": thaw(self.normalized_observation),
            "visible_state_diff": thaw(self.visible_state_diff),
            "before_state_hash": self.before_state_hash, "after_state_hash": self.after_state_hash,
            "displayed_spec_fingerprint": self.displayed_spec_fingerprint,
            "historical_evidence_refs": list(self.historical_evidence_refs),
            "probe_metadata": thaw(self.probe_metadata), "timestamp": self.timestamp,
        }
        # Preserve byte-for-byte compatibility with the v1 Phase 7 schema for
        # legacy events while allowing live events to carry explicit lineage.
        if self.correlation_id:
            value["correlation_id"] = self.correlation_id
        if self.provenance:
            value["provenance"] = thaw(self.provenance)
        return value


@dataclass(frozen=True)
class EvidenceTrace:
    trace_id: str
    public_scenario_id: str
    events: tuple[EvidenceEvent, ...] = ()

    def __post_init__(self) -> None:
        assert_public_id(self.public_scenario_id)
        previous_sequence = 0
        previous_episode = 0
        for event in self.events:
            if event.trace_id != self.trace_id or event.public_scenario_id != self.public_scenario_id:
                raise ValueError("event belongs to another trace or scenario")
            if event.sequence_number != previous_sequence + 1:
                raise ValueError("evidence sequence must be contiguous and strictly increasing")
            if event.episode_number < previous_episode:
                raise ValueError("future evidence cannot precede earlier episodes")
            previous_sequence, previous_episode = event.sequence_number, event.episode_number

    def append(self, event: EvidenceEvent) -> "EvidenceTrace":
        return EvidenceTrace(self.trace_id, self.public_scenario_id, self.events + (event,))

    def through_episode(self, episode_number: int) -> "EvidenceTrace":
        return EvidenceTrace(
            self.trace_id, self.public_scenario_id,
            tuple(event for event in self.events if event.episode_number <= episode_number),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"trace_id": self.trace_id, "public_scenario_id": self.public_scenario_id,
                "events": [event.to_dict() for event in self.events]}


@dataclass(frozen=True)
class AgentView:
    trace: EvidenceTrace
    displayed_spec: Mapping[str, Any]
    current_episode: int

    def __post_init__(self) -> None:
        frozen = freeze(self.displayed_spec)
        assert_agent_visible(thaw(frozen))
        if any(event.episode_number > self.current_episode for event in self.trace.events):
            raise ValueError("AgentView contains future evidence")
        object.__setattr__(self, "displayed_spec", frozen)


@dataclass(frozen=True)
class EvaluatorView:
    agent_view: AgentView
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", freeze(self.metadata))
