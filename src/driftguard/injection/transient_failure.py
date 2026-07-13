from __future__ import annotations

from typing import Any


class TransientFailureInjector:
    @staticmethod
    def metadata(context: Any) -> dict[str, Any]:
        return {"fault_layer": "runtime_once", "episode": context.episode_index}

    @staticmethod
    def active(context: Any, tool: str) -> bool:
        return context.profile.mode.value == "TF" and context.injection_active(tool)

    @staticmethod
    def consume(context: Any) -> None:
        context.consume()


TransientFailureHook = TransientFailureInjector
