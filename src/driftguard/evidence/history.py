from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .models import freeze, thaw


@dataclass(frozen=True)
class HistoryRecord:
    record_id: str
    public_scenario_id: str
    tool_id: str
    displayed_request_shape: Any
    response_shape: Any
    visible_state_diff: Any
    success: bool
    episode_number: int
    spec_fingerprint: str
    timestamp: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "displayed_request_shape", freeze(self.displayed_request_shape))
        object.__setattr__(self, "response_shape", freeze(self.response_shape))
        object.__setattr__(self, "visible_state_diff", freeze(self.visible_state_diff))


class HistoryStore:
    def __init__(self):
        self._records: dict[str, list[HistoryRecord]] = {}

    def append(self, record: HistoryRecord) -> None:
        self._records.setdefault(record.public_scenario_id, []).append(record)

    def before(self, public_scenario_id: str, episode_number: int) -> tuple[HistoryRecord, ...]:
        return tuple(
            record for record in self._records.get(public_scenario_id, [])
            if record.episode_number < episode_number
        )

    def reset(self, public_scenario_id: str) -> None:
        self._records.pop(public_scenario_id, None)
