from __future__ import annotations

from copy import deepcopy


class TaskMemory:
    def __init__(self, persistent: bool = False):
        self.persistent = persistent
        self._events: list[dict] = []

    def append(self, event: dict) -> None:
        self._events.append(deepcopy(event))

    def visible(self) -> list[dict]:
        return deepcopy(self._events)

    def end_task(self) -> None:
        if not self.persistent:
            self._events.clear()

