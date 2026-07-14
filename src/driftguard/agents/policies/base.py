from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RecoveryDecision:
    exact_retry: bool = False
    request_llm: bool = True
    stop: bool = False
    context: dict[str, Any] | None = None


class RecoveryPolicy:
    method = "standard"
    prompt_name = "base_tool_agent_v1.txt"
    allows_probe = False
    persistent_patch = False

    def after_failure(self, call: dict[str, Any], response: dict[str, Any]) -> RecoveryDecision:
        return RecoveryDecision(request_llm=True)

    def end_task(self) -> None:
        return None

