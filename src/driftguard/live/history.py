from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from driftguard.evidence.models import EvidenceEvent, thaw

from .evidence_bridge import LiveScope


@dataclass(frozen=True)
class LiveHistoryRecord:
    event_id: str
    scope_key: str
    episode_number: int
    tool_id: str | None
    response_shape: Any
    visible_state_diff: Any
    displayed_spec_fingerprint: str | None
    correlation_id: str


class LiveHistoryStore:
    def __init__(self) -> None:
        self._records: dict[str, list[LiveHistoryRecord]] = {}

    def append_completed(self, scope: LiveScope, event: EvidenceEvent) -> None:
        provenance = thaw(event.provenance)
        if provenance.get("source") != "controller_event":
            raise ValueError("history must originate from a real Controller event")
        if event.event_type not in {"tool_response", "retry_result", "probe_executed"}:
            raise ValueError("only completed visible executions may enter history")
        self._records.setdefault(scope.key, []).append(LiveHistoryRecord(
            event.event_id, scope.key, event.episode_number, event.tool_id,
            thaw(event.visible_runtime_response), thaw(event.visible_state_diff),
            event.displayed_spec_fingerprint, event.correlation_id,
        ))

    def before(self, scope: LiveScope, episode: int) -> tuple[LiveHistoryRecord, ...]:
        return tuple(row for row in self._records.get(scope.key, ()) if row.episode_number < episode)

    def reset(self, scope: LiveScope) -> None:
        self._records.pop(scope.key, None)

    def size(self, scope: LiveScope) -> int:
        return len(self._records.get(scope.key, ()))
