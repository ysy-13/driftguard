from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from driftguard.healing.patch_registry import PatchRegistry

from .evidence_bridge import LiveEvidenceBridge, LiveScope
from .history import LiveHistoryStore


@dataclass
class LiveHealingSession:
    """Owns every mutable live-healing store for one exact experiment scope."""

    scope: LiveScope
    displayed_spec: dict[str, Any]
    bridge: LiveEvidenceBridge = field(init=False)
    history: LiveHistoryStore = field(default_factory=LiveHistoryStore)
    patch_registry: PatchRegistry = field(default_factory=PatchRegistry)

    def __post_init__(self) -> None:
        self.bridge = LiveEvidenceBridge(self.scope, self.displayed_spec)

    def reset(self) -> None:
        self.bridge.reset()
        self.history.reset(self.scope)
        self.patch_registry.clear()
