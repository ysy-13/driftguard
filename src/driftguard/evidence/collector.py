from __future__ import annotations

import hashlib
import json
from typing import Any

from .models import EvidenceEvent
from .store import EvidenceStore


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


class EvidenceCollector:
    def __init__(self, store: EvidenceStore, displayed_spec: dict[str, Any]):
        self.store = store
        self.spec_fingerprint = stable_hash(displayed_spec)

    def add(self, event_type: str, episode: int, tool_id: str | None = None, **fields: Any) -> EvidenceEvent:
        sequence = len(self.store.trace.events) + 1
        event = EvidenceEvent(
            event_id=f"{self.store.trace.trace_id}-EV{sequence:03d}",
            trace_id=self.store.trace.trace_id,
            public_scenario_id=self.store.trace.public_scenario_id,
            episode_number=episode,
            sequence_number=sequence,
            event_type=event_type,
            tool_id=tool_id,
            displayed_spec_fingerprint=self.spec_fingerprint,
            timestamp=f"2026-01-01T00:00:{sequence:02d}Z",
            **fields,
        )
        self.store.append(event)
        return event
