from __future__ import annotations

from typing import Any


class PersistentDriftController:
    @staticmethod
    def metadata(context: Any) -> dict[str, Any]:
        return {"fault_layer": "runtime_contract", "episode": context.episode_index}

    @staticmethod
    def active(context: Any, tool: str) -> bool:
        return context.profile.mode.value == "PD" and context.injection_active(tool)


PersistentDriftHook = PersistentDriftController
