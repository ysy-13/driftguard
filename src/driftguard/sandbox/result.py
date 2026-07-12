from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolResult:
    status_code: int
    payload: dict[str, Any]

    @property
    def ok(self) -> bool:
        return self.payload.get("ok") is True

    def to_dict(self) -> dict[str, Any]:
        return {"status_code": self.status_code, "payload": deepcopy(self.payload)}


def success(status_code: int, data: dict[str, Any]) -> ToolResult:
    return ToolResult(status_code=status_code, payload={"ok": True, "data": deepcopy(data)})


def failure(
    status_code: int,
    code: str,
    message: str,
    field: str | None = None,
    retryable: bool = False,
) -> ToolResult:
    return ToolResult(
        status_code=status_code,
        payload={
            "ok": False,
            "error": {
                "code": code,
                "message": message,
                "field": field,
                "retryable": retryable,
            },
        },
    )
