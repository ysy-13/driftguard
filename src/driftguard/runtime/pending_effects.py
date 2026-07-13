from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PendingEffect:
    trigger_tool: str
    repo_id: str
    path: tuple[str, ...]
    value: Any


@dataclass
class PendingEffects:
    _items: list[PendingEffect] = field(default_factory=list)

    def add(self, effect: PendingEffect) -> None:
        self._items.append(effect)

    def apply(self, tool: str, arguments: dict[str, Any], state: dict[str, Any]) -> bool:
        changed = False
        remaining: list[PendingEffect] = []
        for effect in self._items:
            if tool == effect.trigger_tool and arguments.get("repo_id") == effect.repo_id:
                container: Any = state["repositories"][effect.repo_id]
                for token in effect.path[:-1]:
                    container = container[token]
                container[effect.path[-1]] = deepcopy(effect.value)
                changed = True
            else:
                remaining.append(effect)
        self._items = remaining
        return changed

    def clear(self) -> None:
        self._items.clear()
