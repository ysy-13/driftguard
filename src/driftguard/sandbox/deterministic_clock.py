from __future__ import annotations

from datetime import datetime, timedelta, timezone


def _parse(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _format(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


class DeterministicClock:
    def __init__(self, initial_time: str):
        self._initial = _parse(initial_time)
        self._current = self._initial

    def reset(self) -> None:
        self._current = self._initial

    def now(self) -> str:
        return _format(self._current)

    def peek_next(self) -> str:
        return _format(self._current + timedelta(seconds=1))

    def advance_write(self) -> str:
        self._current += timedelta(seconds=1)
        return self.now()
